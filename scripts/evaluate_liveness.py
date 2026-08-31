"""One-time locked PAD holdout evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from facekyc.data import read_pad_manifest, validate_pad_manifest
from facekyc.evaluation import pad_metrics
from facekyc.pad import (
    PADManifestDataset,
    build_pad_model,
    pad_transform,
    predict_pad_scores,
    subset_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/celeba_spoof_manifest.csv")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--weights", default="artifacts/pad_mobilenet_v3_small.pth")
    parser.add_argument("--validation-report", default="reports/ds/pad_validation_report.json")
    parser.add_argument("--output", default="reports/ds/pad_locked_holdout_report.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unlock-holdout", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if not args.unlock_holdout:
        raise SystemExit("Refusing to access holdout without --unlock-holdout")
    if output.exists():
        raise SystemExit(f"Locked holdout report already exists: {output}")

    import torch
    from torch.utils.data import DataLoader

    validation_report = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    threshold = float(validation_report["threshold"])
    rows = read_pad_manifest(args.manifest)
    validate_pad_manifest(rows)
    test_rows = subset_rows(rows, "test", args.max_test_samples, args.seed)
    if not test_rows:
        raise ValueError("Manifest does not contain a test split")
    input_size = int(validation_report.get("input_size", 224))
    dataset = PADManifestDataset(
        test_rows, args.dataset_root, pad_transform(input_size, training=False)
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    architecture = validation_report["architecture"]
    model = build_pad_model(architecture=architecture, imagenet_pretrained=False).to(device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    scores, labels = predict_pad_scores(model, loader, device)
    metrics = pad_metrics(scores, labels, threshold)
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "locked_holdout",
        "dataset": "CelebA-Spoof",
        "architecture": architecture,
        "threshold_source": args.validation_report,
        "threshold": threshold,
        "metrics": metrics.to_dict(),
        "test_samples": len(test_rows),
        "holdout_evaluation_count": 1,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
