"""
Train a robustness-focused ResNet-18 from the existing FoliAI checkpoint.

Does not modify production inference (main.py / plant_gate.py).
Does not read data/external_test/.
Does not overwrite existing production checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

# Canonical 15-class order — must match plant-disease-api/main.py exactly.
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
EXTERNAL_TEST_DIR = DATA_DIR / "external_test"
MODEL_DIR = PROJECT_ROOT / "model"
SOURCE_CHECKPOINT = MODEL_DIR / "best_model.pth"
OUTPUT_CHECKPOINT = MODEL_DIR / "robust_v1.pth"
METRICS_JSON = MODEL_DIR / "robust_v1_metrics.json"
METRICS_CSV = MODEL_DIR / "robust_v1_history.csv"

PROTECTED_CHECKPOINTS = {
    MODEL_DIR / "best_model.pth",
    MODEL_DIR / "best_model_v5.pth",
    MODEL_DIR / "best_model_v1_backup.pth",
    MODEL_DIR / "best_model_v2_backup.pth",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_state_dict(raw: Any) -> dict:
    """Accept a raw state_dict or common checkpoint wrappers (matches main.py)."""
    if not isinstance(raw, dict):
        raise ValueError("Unexpected checkpoint type: {!r}".format(type(raw)))

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


def build_resnet18_15() -> nn.Module:
    """Same architecture as production: ResNet-18 + Linear(in_features, 15)."""
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
    return net


def load_starting_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Starting checkpoint not found at {}".format(checkpoint_path)
        )

    net = build_resnet18_15()
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _unwrap_state_dict(raw)

    expected = set(net.state_dict().keys())
    loaded = set(state.keys())
    missing = sorted(expected - loaded)
    unexpected = sorted(loaded - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match ResNet18+15-class head: "
            "missing={} unexpected={}".format(missing[:10], unexpected[:10])
        )

    fc_weight = state.get("fc.weight")
    if not isinstance(fc_weight, torch.Tensor) or tuple(fc_weight.shape) != (
        len(CLASS_NAMES),
        512,
    ):
        raise RuntimeError(
            "fc.weight has unexpected shape {}; expected ({}, 512)".format(
                getattr(fc_weight, "shape", None),
                len(CLASS_NAMES),
            )
        )

    net.load_state_dict(state, strict=True)
    net.to(device)
    return net


def train_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224,
                scale=(0.70, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.35,
                hue=0.04,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                p=0.30,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def val_transforms() -> transforms.Compose:
    # Match production inference resize so validation reflects deployed inputs.
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class OrderedImageFolder(Dataset):
    """Image folder that uses CLASS_NAMES order, not ImageFolder's sort."""

    def __init__(self, root: Path, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []

        if not self.root.is_dir():
            raise FileNotFoundError("Dataset directory not found: {}".format(self.root))

        present = {p.name for p in self.root.iterdir() if p.is_dir()}
        missing = [name for name in CLASS_NAMES if name not in present]
        extra = sorted(present - set(CLASS_NAMES))
        if missing:
            raise RuntimeError(
                "{} is missing required class folders: {}".format(self.root, missing)
            )
        if extra:
            raise RuntimeError(
                "{} has unexpected class folders: {}".format(self.root, extra)
            )

        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root / class_name
            files = [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {ext.lower() for ext in IMG_EXTENSIONS}
            ]
            if not files:
                raise RuntimeError("No images found in {}".format(class_dir))
            for path in sorted(files):
                self.samples.append((path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = default_loader(str(path))
        if not isinstance(image, Image.Image):
            image = Image.open(path).convert("RGB")
        else:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def assert_external_test_unused() -> None:
    """Hard guard: this script must never open data/external_test/."""
    if str(EXTERNAL_TEST_DIR) in (str(TRAIN_DIR), str(VAL_DIR)):
        raise RuntimeError("external_test path must not be used as train or val")


def class_counts(dataset: OrderedImageFolder) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in dataset.samples:
        counts[CLASS_NAMES[label]] += 1
    return counts


def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    """Unweighted mean of per-class F1 from a [C, C] confusion matrix."""
    tp = cm.diag().float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / (tp + fn).clamp_min(1e-12)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return float(f1.mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> Tuple[float, float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    cm = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((preds == labels).sum().item())
        total_seen += images.size(0)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[int(t.item()), int(p.item())] += 1

    avg_loss = total_loss / max(total_seen, 1)
    accuracy = total_correct / max(total_seen, 1)
    f1 = macro_f1_from_confusion(cm)
    return avg_loss, accuracy, f1


def resolve_output_path(requested: Path) -> Path:
    if requested.resolve() in {p.resolve() for p in PROTECTED_CHECKPOINTS}:
        raise RuntimeError(
            "Refusing to write to protected production checkpoint: {}".format(requested)
        )
    if requested.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = requested.with_name("{}_{}{}".format(requested.stem, stamp, requested.suffix))
        print(
            "WARNING: {} already exists; writing to {} instead".format(
                requested, alt
            )
        )
        return alt
    return requested


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_history_csv(path: Path, history: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "is_best",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row[k] for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune FoliAI ResNet-18 with robustness augmentations"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_CHECKPOINT,
        help="Existing FoliAI checkpoint to start from (default: model/best_model.pth)",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_CHECKPOINT,
        help="Best-checkpoint destination (default: model/robust_v1.pth)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    assert_external_test_unused()

    if not TRAIN_DIR.is_dir():
        print("ERROR: training directory not found: {}".format(TRAIN_DIR), file=sys.stderr)
        return 1
    if not VAL_DIR.is_dir():
        print(
            "ERROR: validation directory not found: {}".format(VAL_DIR),
            file=sys.stderr,
        )
        return 1
    if TRAIN_DIR.resolve() == EXTERNAL_TEST_DIR.resolve() or VAL_DIR.resolve() == EXTERNAL_TEST_DIR.resolve():
        print("ERROR: refusing to use data/external_test/ for training or validation", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    print("Starting checkpoint: {}".format(args.source))
    print("Train dir: {}".format(TRAIN_DIR))
    print("Val dir:   {}".format(VAL_DIR))
    print("external_test is excluded")

    train_ds = OrderedImageFolder(TRAIN_DIR, train_transforms())
    val_ds = OrderedImageFolder(VAL_DIR, val_transforms())

    print("Train images: {}  {}".format(len(train_ds), class_counts(train_ds)))
    print("Val images:   {}  {}".format(len(val_ds), class_counts(val_ds)))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_starting_checkpoint(args.source, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(args.patience // 2, 2),
    )

    output_path = resolve_output_path(args.output)
    history: List[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    best_record: Dict[str, Any] = {}
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        scheduler.step(val_loss)

        is_best = val_f1 > best_f1
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 6),
            "val_macro_f1": round(val_f1, 6),
            "is_best": is_best,
        }
        history.append(row)

        print(
            "Epoch {epoch:03d}/{total:03d} | "
            "train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            "val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
            "{best_mark}".format(
                epoch=epoch,
                total=args.epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                train_f1=train_f1,
                val_loss=val_loss,
                val_acc=val_acc,
                val_f1=val_f1,
                best_mark="  * best" if is_best else "",
            )
        )

        if is_best:
            best_f1 = val_f1
            best_epoch = epoch
            epochs_without_improve = 0
            best_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "train_loss": train_loss,
                    "class_names": CLASS_NAMES,
                    "source_checkpoint": str(args.source),
                },
                output_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(
                    "Early stopping at epoch {} (no val macro F1 improvement for {} epochs)".format(
                        epoch, args.patience
                    )
                )
                break

    metrics = {
        "best_epoch": best_epoch,
        "best_train_loss": best_record.get("train_loss"),
        "best_val_loss": best_record.get("val_loss"),
        "best_val_accuracy": best_record.get("val_accuracy"),
        "best_val_macro_f1": best_record.get("val_macro_f1"),
        "checkpoint_path": str(output_path),
        "source_checkpoint": str(args.source),
        "class_names": CLASS_NAMES,
        "history": history,
    }
    save_json(METRICS_JSON, metrics)
    save_history_csv(METRICS_CSV, history)

    print("")
    print("==== Best checkpoint ====")
    print("best_epoch:      {}".format(best_epoch))
    print("train_loss:      {:.6f}".format(best_record.get("train_loss", float("nan"))))
    print("val_loss:        {:.6f}".format(best_record.get("val_loss", float("nan"))))
    print("val_accuracy:    {:.6f}".format(best_record.get("val_accuracy", float("nan"))))
    print("val_macro_f1:    {:.6f}".format(best_record.get("val_macro_f1", float("nan"))))
    print("saved_to:        {}".format(output_path))
    print("metrics_json:    {}".format(METRICS_JSON))
    print("metrics_csv:     {}".format(METRICS_CSV))
    return 0


if __name__ == "__main__":
    sys.exit(main())
