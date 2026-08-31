"""LFW candidate comparison, threshold lock, and one-time holdout evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from facekyc.evaluation import (
    equal_error_rate,
    select_verification_threshold,
    verification_metrics,
)

FOLD_SIZE = 600
DEVELOPMENT_FOLDS = tuple(range(8))
VALIDATION_FOLD = 8
HOLDOUT_FOLD = 9


def load_lfw_pairs(data_home: str | None):
    from sklearn.datasets import fetch_lfw_pairs

    return fetch_lfw_pairs(
        subset="10_folds",
        color=True,
        resize=0.5,
        data_home=data_home,
        download_if_missing=True,
    )


def fold_indices(folds: tuple[int, ...], maximum: int | None = None) -> np.ndarray:
    indices = np.concatenate(
        [np.arange(fold * FOLD_SIZE, (fold + 1) * FOLD_SIZE) for fold in folds]
    )
    return indices[:maximum] if maximum is not None else indices


def embedding_scores(pairs: np.ndarray, pretrained: str, batch_size: int) -> np.ndarray:
    import torch
    from facenet_pytorch import InceptionResnetV1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
    images = np.asarray(pairs, dtype=np.float32).reshape((-1, *pairs.shape[2:]))
    if float(images.max()) <= 1.5:
        images = images * 255.0
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).permute(0, 3, 1, 2)
            batch = torch.nn.functional.interpolate(
                batch, size=(160, 160), mode="bilinear", align_corners=False
            )
            batch = (batch - 127.5) / 128.0
            vector = model(batch.to(device))
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
            embeddings.append(vector.detach().cpu())
    matrix = torch.cat(embeddings).reshape((-1, 2, 512))
    return torch.nn.functional.cosine_similarity(matrix[:, 0], matrix[:, 1]).numpy()


def compare_candidates(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    indices = fold_indices(DEVELOPMENT_FOLDS, args.max_pairs)
    pairs, labels = dataset.pairs[indices], np.asarray(dataset.target)[indices]
    candidates = []
    for pretrained in ("vggface2", "casia-webface"):
        scores = embedding_scores(pairs, pretrained, args.batch_size)
        fold_rows = []
        available_folds = sorted(set(indices // FOLD_SIZE))
        for fold in available_folds:
            evaluation_mask = indices // FOLD_SIZE == fold
            calibration_mask = ~evaluation_mask
            if not calibration_mask.any() or not evaluation_mask.any():
                continue
            threshold, _ = select_verification_threshold(
                scores[calibration_mask], labels[calibration_mask], args.target_fmr
            )
            metric = verification_metrics(
                scores[evaluation_mask], labels[evaluation_mask], threshold
            )
            fold_rows.append(metric.to_dict())
        summary = {
            "pretrained_dataset": pretrained,
            "folds": fold_rows,
            "mean_fmr": mean(row["fmr"] for row in fold_rows),
            "mean_fnmr": mean(row["fnmr"] for row in fold_rows),
            "std_fnmr": pstdev(row["fnmr"] for row in fold_rows),
            "mean_accuracy": mean(row["accuracy"] for row in fold_rows),
        }
        candidates.append(summary)
    selected = min(candidates, key=lambda row: (row["mean_fnmr"], row["mean_fmr"]))
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "development_model_selection",
        "dataset": "LFW 10-fold pairs protocol",
        "folds": list(DEVELOPMENT_FOLDS),
        "target_fmr": args.target_fmr,
        "pairs": len(indices),
        "candidates": candidates,
        "selected_pretrained_dataset": selected["pretrained_dataset"],
        "holdout_accessed": False,
    }


def calibrate(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    candidate_report = json.loads(Path(args.candidate_report).read_text(encoding="utf-8"))
    pretrained = candidate_report["selected_pretrained_dataset"]
    indices = fold_indices((VALIDATION_FOLD,), args.max_pairs)
    labels = np.asarray(dataset.target)[indices]
    scores = embedding_scores(dataset.pairs[indices], pretrained, args.batch_size)
    threshold, metric = select_verification_threshold(scores, labels, args.target_fmr)
    eer_threshold, eer, _ = equal_error_rate(scores, labels)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "validation_threshold_lock",
        "dataset": "LFW 10-fold pairs protocol",
        "fold": VALIDATION_FOLD,
        "pretrained_dataset": pretrained,
        "score": "cosine_similarity",
        "target_fmr": args.target_fmr,
        "threshold": threshold,
        "metrics": metric.to_dict(),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "holdout_accessed": False,
    }


def holdout(dataset: Any, args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if not args.unlock_holdout:
        raise SystemExit("Refusing to access LFW holdout without --unlock-holdout")
    if output.exists():
        raise SystemExit(f"Locked holdout report already exists: {output}")
    validation = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    indices = fold_indices((HOLDOUT_FOLD,), args.max_pairs)
    labels = np.asarray(dataset.target)[indices]
    scores = embedding_scores(
        dataset.pairs[indices], validation["pretrained_dataset"], args.batch_size
    )
    metric = verification_metrics(scores, labels, float(validation["threshold"]))
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "stage": "locked_holdout",
        "dataset": "LFW 10-fold pairs protocol",
        "fold": HOLDOUT_FOLD,
        "pretrained_dataset": validation["pretrained_dataset"],
        "threshold_source": args.validation_report,
        "threshold": validation["threshold"],
        "metrics": metric.to_dict(),
        "holdout_evaluation_count": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["compare", "calibrate", "holdout"])
    parser.add_argument("--data-home")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-fmr", type=float, default=0.01)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--candidate-report", default="reports/ds/verification_candidates.json")
    parser.add_argument(
        "--validation-report", default="reports/ds/verification_validation_report.json"
    )
    parser.add_argument("--output")
    parser.add_argument("--unlock-holdout", action="store_true")
    args = parser.parse_args()
    default_outputs = {
        "compare": args.candidate_report,
        "calibrate": args.validation_report,
        "holdout": "reports/ds/verification_locked_holdout_report.json",
    }
    args.output = args.output or default_outputs[args.stage]
    dataset = load_lfw_pairs(args.data_home)
    if len(dataset.pairs) != 10 * FOLD_SIZE:
        raise ValueError(f"Expected 6,000 LFW pairs, found {len(dataset.pairs)}")
    if args.stage == "compare":
        report = compare_candidates(dataset, args)
    elif args.stage == "calibrate":
        report = calibrate(dataset, args)
    else:
        report = holdout(dataset, args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
