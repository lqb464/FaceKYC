"""Train PAD on train only and lock its threshold on validation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from facekyc.data import read_pad_manifest, validate_pad_manifest
from facekyc.evaluation import select_pad_threshold
from facekyc.pad import (
    PADManifestDataset,
    build_pad_model,
    pad_transform,
    predict_pad_scores,
    subset_rows,
)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> dict:
    import torch
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    rows = read_pad_manifest(args.manifest)
    validate_pad_manifest(rows)
    train_rows = subset_rows(rows, "train", args.max_train_samples, args.seed)
    validation_rows = subset_rows(rows, "validation", args.max_validation_samples, args.seed)
    if not train_rows or not validation_rows:
        raise ValueError("Manifest must contain non-empty train and validation splits")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = PADManifestDataset(
        train_rows, args.dataset_root, pad_transform(args.input_size, training=True)
    )
    validation_data = PADManifestDataset(
        validation_rows, args.dataset_root, pad_transform(args.input_size, training=False)
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device == "cuda",
    )

    model = build_pad_model(architecture=args.architecture, imagenet_pretrained=True).to(device)
    class_counts = Counter(int(row["label"]) for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (2 * class_counts[index]) for index in (0, 1)],
        dtype=torch.float32,
        device=device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_acer = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(labels)
            samples += len(labels)
        scores, labels = predict_pad_scores(model, validation_loader, device)
        threshold, metrics = select_pad_threshold(scores, labels, args.target_apcer)
        epoch_row = {
            "epoch": epoch,
            "train_loss": running_loss / samples,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics.to_dict(),
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False))
        if metrics.acer < best_acer:
            best_acer = metrics.acer
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "class_contract": {"0": "attack", "1": "bona_fide"},
                    "input_size": args.input_size,
                    "validation_threshold": threshold,
                },
                output,
            )
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            break

    checkpoint = torch.load(output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    scores, labels = predict_pad_scores(model, validation_loader, device)
    threshold, metrics = select_pad_threshold(scores, labels, args.target_apcer)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "validation_threshold_lock",
        "dataset": "CelebA-Spoof",
        "dataset_license": "non-commercial research only",
        "subject_disjoint": True,
        "device": device,
        "seed": args.seed,
        "architecture": args.architecture,
        "input_size": args.input_size,
        "weights_path": str(output),
        "best_epoch": best_epoch,
        "target_apcer": args.target_apcer,
        "threshold": threshold,
        "metrics": metrics.to_dict(),
        "train_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "history": history,
        "holdout_accessed": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/celeba_spoof_manifest.csv")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default="artifacts/pad_mobilenet_v3_small.pth")
    parser.add_argument("--report", default="reports/ds/pad_validation_report.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument(
        "--architecture",
        choices=["mobilenet_v3_small", "resnet18"],
        default="mobilenet_v3_small",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-apcer", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), indent=2, ensure_ascii=False))
