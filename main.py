import io
import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

logger = logging.getLogger("foliai")
logging.basicConfig(level=logging.INFO)

# Canonical 15-class order — do not reorder (must match both checkpoints).
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",  # 0  — v3
    "Pepper__bell___healthy",  # 1  — v3
    "Potato___Early_blight",  # 2  — v3
    "Potato___Late_blight",  # 3  — v5
    "Potato___healthy",  # 4  — average(v3, v5)
    "Tomato_Bacterial_spot",  # 5  — v3
    "Tomato_Early_blight",  # 6  — v5
    "Tomato_Late_blight",  # 7  — v5
    "Tomato_Leaf_Mold",  # 8  — v3
    "Tomato_Septoria_leaf_spot",  # 9  — v3
    "Tomato_Spider_mites_Two_spotted_spider_mite",  # 10 — v3
    "Tomato__Target_Spot",  # 11 — v3
    "Tomato__Tomato_YellowLeaf__Curl_Virus",  # 12 — v3
    "Tomato__Tomato_mosaic_virus",  # 13 — v5
    "Tomato_healthy",  # 14 — v3
]

MODEL_DIR = Path(__file__).parent / "model"
# v3 remains the rollback checkpoint; do not delete or overwrite it.
V3_MODEL_PATH = MODEL_DIR / "best_model.pth"
V5_MODEL_PATH = MODEL_DIR / "best_model_v5.pth"

CONFIDENCE_THRESHOLD = 0.60
# Temperature is applied once per model (to logits) before hybrid routing.
# Do not apply T again to the hybrid vector.
SOFTMAX_TEMPERATURE = 2.5

# Class-specific hybrid routing (validated on external 60-image benchmark):
# v5 is stronger on late/early blight + mosaic; Potato healthy is averaged;
# all other classes keep v3 probabilities.
V5_CLASS_INDICES = frozenset({3, 6, 7, 13})
POTATO_HEALTHY_INDEX = 4

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB
TOP_K = 3
UNKNOWN_MESSAGE = (
    "Image does not clearly match any known plant disease class"
)
GENERIC_INFERENCE_ERROR = "Unable to process the image. Please try again."
GENERIC_INVALID_IMAGE = "Uploaded file is not a valid image."

preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

app = FastAPI(title="Plant Disease API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

device: torch.device = torch.device("cpu")
model_v3: Optional[nn.Module] = None
model_v5: Optional[nn.Module] = None


def _unwrap_state_dict(raw: Any) -> dict:
    """Accept a raw state_dict or common checkpoint wrappers."""
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(raw)!r}")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        nested = raw.get(key)
        if isinstance(nested, dict) and any(
            isinstance(v, torch.Tensor) for v in nested.values()
        ):
            raw = nested
            break

    if any(k.startswith("module.") for k in raw):
        raw = {k.removeprefix("module."): v for k, v in raw.items()}

    return raw


def load_model(checkpoint_path: Path, label: str) -> nn.Module:
    """Build ResNet18 (15-class head) and load a checkpoint onto `device`."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{label} weights not found at {checkpoint_path}")

    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _unwrap_state_dict(raw)

    expected = set(net.state_dict().keys())
    loaded = set(state.keys())
    missing = sorted(expected - loaded)
    unexpected = sorted(loaded - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint does not match ResNet18+15-class head: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    fc_weight = state.get("fc.weight")
    if not isinstance(fc_weight, torch.Tensor) or tuple(fc_weight.shape) != (
        len(CLASS_NAMES),
        512,
    ):
        raise RuntimeError(
            f"{label} fc.weight has unexpected shape "
            f"{getattr(fc_weight, 'shape', None)}; "
            f"expected ({len(CLASS_NAMES)}, 512)"
        )

    net.load_state_dict(state, strict=True)
    net.to(device)
    net.eval()

    if net.training:
        raise RuntimeError(f"{label} remained in training mode after load_model()")

    bn_training = [
        name
        for name, module in net.named_modules()
        if isinstance(module, nn.BatchNorm2d) and module.training
    ]
    if bn_training:
        raise RuntimeError(f"{label} BatchNorm still in training mode: {bn_training}")

    param_count = sum(p.numel() for p in net.parameters())
    logger.info(
        "Loaded %s from %s (%s params, %d classes, device=%s). training=%s",
        label,
        checkpoint_path,
        f"{param_count:,}",
        len(CLASS_NAMES),
        device,
        net.training,
    )
    return net


def build_hybrid_probs(
    probs_v3: torch.Tensor,
    probs_v5: torch.Tensor,
) -> torch.Tensor:
    """
    Class-specific hybrid from temperature-scaled v3/v5 probability vectors.

    probs shape: [batch, 15]

    Routing (validated external benchmark):
      - indices 3, 6, 7, 13  -> v5
      - index 4 (Potato___healthy) -> average(v3, v5)
      - all other indices -> v3

    Renormalize so each row sums to 1 (mixing sources can leave sum != 1).
    """
    if probs_v3.shape != probs_v5.shape:
        raise ValueError(
            f"v3/v5 probability shape mismatch: {tuple(probs_v3.shape)} vs "
            f"{tuple(probs_v5.shape)}"
        )
    if probs_v3.ndim != 2 or probs_v3.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Unexpected probability shape: {tuple(probs_v3.shape)}")

    hybrid = probs_v3.clone()
    for idx in V5_CLASS_INDICES:
        hybrid[:, idx] = probs_v5[:, idx]
    hybrid[:, POTATO_HEALTHY_INDEX] = (
        probs_v3[:, POTATO_HEALTHY_INDEX] + probs_v5[:, POTATO_HEALTHY_INDEX]
    ) / 2.0

    # Renormalize after splicing probabilities from two distributions.
    hybrid = hybrid / hybrid.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return hybrid


def run_hybrid_inference(tensor: torch.Tensor) -> torch.Tensor:
    """
    Preprocessed image tensor [B,3,224,224] -> renormalized hybrid probs [B,15].

    Temperature T=2.5 is applied exactly once to each model's logits
    (not again to the hybrid vector).
    """
    if model_v3 is None or model_v5 is None:
        raise RuntimeError("Hybrid models are not loaded")

    model_v3.eval()
    model_v5.eval()

    with torch.inference_mode():
        logits_v3 = model_v3(tensor)
        logits_v5 = model_v5(tensor)

        if logits_v3.shape != logits_v5.shape or logits_v3.shape[-1] != len(CLASS_NAMES):
            raise RuntimeError(
                f"Unexpected logits shapes: v3={tuple(logits_v3.shape)} "
                f"v5={tuple(logits_v5.shape)}"
            )

        probs_v3 = torch.softmax(logits_v3 / SOFTMAX_TEMPERATURE, dim=1)
        probs_v5 = torch.softmax(logits_v5 / SOFTMAX_TEMPERATURE, dim=1)
        hybrid = build_hybrid_probs(probs_v3, probs_v5)

        if logger.isEnabledFor(logging.DEBUG):
            v3_conf, v3_idx = torch.max(probs_v3[0], dim=0)
            v5_conf, v5_idx = torch.max(probs_v5[0], dim=0)
            hy_conf, hy_idx = torch.max(hybrid[0], dim=0)
            logger.debug(
                "V3 top: %s (%.1f%%) | V5 top: %s (%.1f%%) | "
                "Hybrid top: %s (%.1f%%) | Hybrid sum: %.6f",
                CLASS_NAMES[int(v3_idx)],
                float(v3_conf) * 100.0,
                CLASS_NAMES[int(v5_idx)],
                float(v5_conf) * 100.0,
                CLASS_NAMES[int(hy_idx)],
                float(hy_conf) * 100.0,
                float(hybrid[0].sum()),
            )

        return hybrid


@app.on_event("startup")
def startup() -> None:
    global device, model_v3, model_v5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_v3 = load_model(V3_MODEL_PATH, "v3")
    model_v5 = load_model(V5_MODEL_PATH, "v5")
    logger.info(
        "Hybrid ensemble ready (v3+v5 class-specific routing, T=%.1f, device=%s)",
        SOFTMAX_TEMPERATURE,
        device,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    """Never leak raw Python tracebacks to clients."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": GENERIC_INFERENCE_ERROR},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_v3 is not None and model_v5 is not None,
        "ensemble": "v3+v5_hybrid",
        "device": str(device),
        "model_v3_training": bool(model_v3.training) if model_v3 is not None else None,
        "model_v5_training": bool(model_v5.training) if model_v5 is not None else None,
    }


def _top_predictions(probabilities: torch.Tensor, k: int = TOP_K) -> list[dict]:
    """Top-k from a 1D probability vector (must be hybrid probs in production)."""
    values, indices = torch.topk(probabilities, k=min(k, probabilities.numel()))
    return [
        {
            "class_name": CLASS_NAMES[int(idx)],
            "confidence": float(conf),
        }
        for conf, idx in zip(values.tolist(), indices.tolist())
    ]


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    if model_v3 is None or model_v5 is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="File too large. Maximum upload size is 10MB.",
                )
        except ValueError:
            pass

    try:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum upload size is 10MB.",
            )
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        try:
            image = Image.open(io.BytesIO(contents))
            image.load()
            image = image.convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=GENERIC_INVALID_IMAGE)
        except OSError:
            raise HTTPException(status_code=400, detail=GENERIC_INVALID_IMAGE)

        # Preprocess once; reuse the same tensor for v3 and v5.
        tensor = preprocess(image).unsqueeze(0).to(device)

        logger.debug(
            "tensor shape=%s min=%.6f max=%.6f mean=%.6f",
            tuple(tensor.shape),
            tensor.min().item(),
            tensor.max().item(),
            tensor.mean().item(),
        )

        # Hybrid probabilities: temperature once per model, then class routing.
        hybrid_probs = run_hybrid_inference(tensor)[0]

        if not torch.isfinite(hybrid_probs).all():
            logger.error("Non-finite values in hybrid probability vector")
            raise HTTPException(status_code=500, detail=GENERIC_INFERENCE_ERROR)

        # Prediction / confidence / top-3 / rejection all use HYBRID probs only.
        predictions = _top_predictions(hybrid_probs, TOP_K)
        confidence = predictions[0]["confidence"]
        class_name = predictions[0]["class_name"]

        logger.debug(
            "Hybrid top: %s (%.1f%%) | Hybrid sum: %.6f",
            class_name,
            confidence * 100.0,
            hybrid_probs.sum().item(),
        )

        matched = confidence >= CONFIDENCE_THRESHOLD
        return {
            "class_name": class_name if matched else None,
            "confidence": confidence,
            "matched": matched,
            "message": None if matched else UNKNOWN_MESSAGE,
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=GENERIC_INFERENCE_ERROR)
