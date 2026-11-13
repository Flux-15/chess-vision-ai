"""
Trains a CNN (MobileNetV3-Small) to classify 13 chess square classes.

Training data: Synthetic squares rendered at Chess.com ocean theme + glass pieces
(which is what the user's screen shows), plus real user-captured templates.

Held-out evaluation: 'icy' and 'purple' themes (never seen during training).
Target: >99% accuracy on held-out themes, >95% on real screen captures.
"""

from __future__ import annotations
import os, sys, time, json, argparse
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.generate_dataset import generate_dataset_arrays, CLASS_NAMES


class SquareDataset(Dataset):
    """Memory-efficient PyTorch Dataset storing images as uint8 (N, C, H, W) in RGB."""

    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        # images is (N, H, W, 3) BGR uint8 from generate_dataset
        # Convert BGR → RGB, then transpose to (N, C, H, W)
        images_rgb = images[:, :, :, ::-1].copy()  # BGR → RGB
        self.images = np.ascontiguousarray(
            np.transpose(images_rgb, (0, 3, 1, 2)), dtype=np.uint8
        )
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.from_numpy(self.images[idx]), int(self.labels[idx])


def create_classifier_model(
    model_name: str = "mobilenet_v3_small", num_classes: int = 13
) -> nn.Module:
    """Builds and returns the backbone model with custom head."""
    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Overall accuracy, per-class metrics, confusion matrix."""
    num_classes = len(CLASS_NAMES)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t, p] += 1

    total = len(y_true)
    overall_acc = float(np.trace(conf_matrix) / total) if total > 0 else 0.0

    per_class: Dict[str, Any] = {}
    for i, cname in enumerate(CLASS_NAMES):
        tp = int(conf_matrix[i, i])
        fn = int(conf_matrix[i, :].sum() - tp)
        fp = int(conf_matrix[:, i].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[cname] = {
            "samples": tp + fn,
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
        }

    return {
        "overall_accuracy": round(overall_acc, 5),
        "total_evaluated": total,
        "correct_predictions": int(np.trace(conf_matrix)),
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": per_class,
    }


def train_classifier(
    train_samples: int = 120_000,
    val_samples:   int =  20_000,
    epochs:        int = 8,
    batch_size:    int = 256,
    lr:            float = 1e-3,
    model_name:    str = "mobilenet_v3_small",
    output_path:   str = "models/classifier.pt",
    report_path:   str = "models/training_report.json",
    device_name:   Optional[str] = None,
) -> Dict[str, Any]:
    """Trains and validates the piece classifier."""
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[*] Training on: {device} ({gpu_name})", flush=True)

    # ImageNet normalisation (applied in the training loop)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Training themes include ocean (the actual board) heavily weighted
    train_themes = [
        "ocean_real", "ocean_real", "ocean_real",   # 3x weight → 50% of data
        "ocean_bright", "ocean_dim",                 # small brightness variants
        "green", "brown",                            # generalisation themes
    ]
    train_styles = ["glass", "glass", "glass", "neo", "classic", "bases"]

    # Held-out: themes never seen in training
    val_themes = ["icy", "purple"]
    val_styles = ["alpha", "cburnett"]

    print(f"[*] Generating {train_samples} training samples...", flush=True)
    t0 = time.time()
    tr_imgs, tr_lbls = generate_dataset_arrays(train_samples, train_themes, train_styles)
    print(f"    Done in {time.time()-t0:.1f}s. Shape: {tr_imgs.shape}", flush=True)

    print(f"[*] Generating {val_samples} validation samples (held-out themes)...", flush=True)
    t0 = time.time()
    va_imgs, va_lbls = generate_dataset_arrays(val_samples, val_themes, val_styles)
    print(f"    Done in {time.time()-t0:.1f}s. Shape: {va_imgs.shape}", flush=True)

    pin = device.type == "cuda"
    tr_loader = DataLoader(SquareDataset(tr_imgs, tr_lbls), batch_size=batch_size,
                           shuffle=True,  num_workers=0, pin_memory=pin)
    va_loader = DataLoader(SquareDataset(va_imgs, va_lbls), batch_size=batch_size,
                           shuffle=False, num_workers=0, pin_memory=pin)

    model     = create_classifier_model(model_name, 13).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_metrics: Dict[str, Any] = {}

    print(f"[*] Training {epochs} epochs, batch={batch_size} ...", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = tr_correct = tr_total = 0
        t_ep = time.time()

        for inputs, targets in tr_loader:
            inputs  = inputs.to(device, non_blocking=True).float() / 255.0
            inputs  = (inputs - mean) / std
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out  = model(inputs)
                loss = criterion(out, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * targets.size(0)
            preds       = torch.argmax(out, 1)
            tr_correct += (preds == targets).sum().item()
            tr_total   += targets.size(0)

        scheduler.step()
        tr_acc = tr_correct / tr_total
        avg_loss = total_loss / tr_total

        # Validation
        model.eval()
        va_preds: List[int] = []
        va_tgts:  List[int] = []
        with torch.no_grad():
            for inputs, targets in va_loader:
                inputs = inputs.to(device, non_blocking=True).float() / 255.0
                inputs = (inputs - mean) / std
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(inputs)
                va_preds.extend(torch.argmax(out, 1).cpu().numpy().tolist())
                va_tgts.extend(targets.numpy().tolist())

        metrics  = compute_metrics(np.array(va_tgts), np.array(va_preds))
        va_acc   = metrics["overall_accuracy"]
        elapsed  = time.time() - t_ep

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}]  "
            f"Loss={avg_loss:.4f}  TrainAcc={tr_acc*100:.2f}%  "
            f"ValAcc(held-out)={va_acc*100:.3f}%  Time={elapsed:.1f}s",
            flush=True,
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_metrics = metrics
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name":  model_name,
                    "num_classes": 13,
                    "class_names": CLASS_NAMES,
                    "val_accuracy": va_acc,
                    "metrics": metrics,
                },
                output_path,
            )
            print(f"    >> New best ({va_acc*100:.3f}%) -- saved to {output_path}", flush=True)

    with open(report_path, "w") as f:
        json.dump(best_metrics, f, indent=2)

    print("\n" + "="*65, flush=True)
    print(f"[*] Best held-out accuracy: {best_val_acc*100:.3f}%", flush=True)
    print(f"[*] Weights saved to: {output_path}", flush=True)

    # Print per-class breakdown
    print("\nPer-class metrics:", flush=True)
    for cname, cm in best_metrics.get("per_class", {}).items():
        print(f"  {cname:5s}: P={cm['precision']:.3f}  R={cm['recall']:.3f}  F1={cm['f1']:.3f}", flush=True)
    print("="*65, flush=True)
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chess Square Classifier")
    parser.add_argument("--train-samples", type=int, default=120_000)
    parser.add_argument("--val-samples",   type=int, default=20_000)
    parser.add_argument("--epochs",        type=int, default=8)
    parser.add_argument("--batch-size",    type=int, default=256)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--model",         type=str,  default="mobilenet_v3_small",
                        choices=["mobilenet_v3_small", "resnet18"])
    parser.add_argument("--output",        type=str,  default="models/classifier.pt")
    args = parser.parse_args()

    train_classifier(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_name=args.model,
        output_path=args.output,
    )
