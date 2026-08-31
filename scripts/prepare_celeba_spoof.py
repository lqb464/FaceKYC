"""Build a normalized, subject-disjoint manifest from official CelebA-Spoof JSON."""

from __future__ import annotations

import argparse
import json

from facekyc.data import build_celeba_spoof_manifest, read_pad_manifest, validate_pad_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-annotations", required=True)
    parser.add_argument("--test-annotations", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--output", default="data/processed/celeba_spoof_manifest.csv")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--official-live-value", type=int, choices=[0, 1])
    args = parser.parse_args()
    output = build_celeba_spoof_manifest(
        train_annotations=args.train_annotations,
        test_annotations=args.test_annotations,
        output_path=args.output,
        validation_ratio=args.validation_ratio,
        random_seed=args.seed,
        official_live_value=args.official_live_value,
    )
    report = validate_pad_manifest(read_pad_manifest(output), dataset_root=args.dataset_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
