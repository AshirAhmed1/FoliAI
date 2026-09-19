"""Lightweight verification of the v3+v5 hybrid inference path."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

# Ensure project root is on path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import main as app_main


def main() -> None:
    app_main.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_main.model_v3 = app_main.load_model(app_main.V3_MODEL_PATH, "v3")
    app_main.model_v5 = app_main.load_model(app_main.V5_MODEL_PATH, "v5")

    assert app_main.model_v3 is not None and app_main.model_v5 is not None
    assert not app_main.model_v3.training and not app_main.model_v5.training

    img = Image.new("RGB", (256, 256), (40, 120, 50))
    tensor = app_main.preprocess(img).unsqueeze(0).to(app_main.device)
    assert tuple(tensor.shape) == (1, 3, 224, 224)

    with torch.inference_mode():
        logits_v3 = app_main.model_v3(tensor)
        logits_v5 = app_main.model_v5(tensor)

    assert tuple(logits_v3.shape) == (1, 15), logits_v3.shape
    assert tuple(logits_v5.shape) == (1, 15), logits_v5.shape

    probs_v3 = torch.softmax(logits_v3 / app_main.SOFTMAX_TEMPERATURE, dim=1)
    probs_v5 = torch.softmax(logits_v5 / app_main.SOFTMAX_TEMPERATURE, dim=1)
    hybrid = app_main.build_hybrid_probs(probs_v3, probs_v5)

    assert tuple(hybrid.shape) == (1, 15), hybrid.shape
    assert torch.isfinite(hybrid).all()
    assert abs(hybrid.sum().item() - 1.0) < 1e-5, hybrid.sum().item()

    # Routing invariants before renormalization.
    pre = probs_v3.clone()
    for idx in app_main.V5_CLASS_INDICES:
        pre[:, idx] = probs_v5[:, idx]
    pre[:, app_main.POTATO_HEALTHY_INDEX] = (
        probs_v3[:, app_main.POTATO_HEALTHY_INDEX]
        + probs_v5[:, app_main.POTATO_HEALTHY_INDEX]
    ) / 2.0
    renorm = pre / pre.sum(dim=1, keepdim=True)
    assert torch.allclose(hybrid, renorm, atol=1e-6)

    for idx in range(15):
        if idx in app_main.V5_CLASS_INDICES:
            assert torch.allclose(pre[:, idx], probs_v5[:, idx]), idx
        elif idx == app_main.POTATO_HEALTHY_INDEX:
            expected = (probs_v3[:, idx] + probs_v5[:, idx]) / 2.0
            assert torch.allclose(pre[:, idx], expected), idx
        else:
            assert torch.allclose(pre[:, idx], probs_v3[:, idx]), idx

    hybrid2 = app_main.run_hybrid_inference(tensor)
    assert tuple(hybrid2.shape) == (1, 15)
    assert abs(hybrid2.sum().item() - 1.0) < 1e-5
    assert torch.isfinite(hybrid2).all()

    top = app_main._top_predictions(hybrid2[0], k=3)
    assert len(top) == 3
    assert top[0]["confidence"] >= top[1]["confidence"] >= top[2]["confidence"]
    assert abs(top[0]["confidence"] - float(hybrid2[0].max())) < 1e-6

    # T applied once: softmax(logits/T), not logits then /T again on hybrid.
    assert app_main.SOFTMAX_TEMPERATURE == 2.5

    print("OK: v3/v5 load, shapes [1,15], hybrid sum~1, routing, top-3 from hybrid")
    print(
        f"device={app_main.device} top={top[0]['class_name']} "
        f"conf={top[0]['confidence']:.4f} sum={hybrid2.sum().item():.6f}"
    )


if __name__ == "__main__":
    main()
