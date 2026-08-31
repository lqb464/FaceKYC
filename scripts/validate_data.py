"""Validate the PAD manifest without loading biometric pixels into memory."""

from __future__ import annotations

import argparse
import json

from facekyc.data import read_pad_manifest, validate_pad_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/celeba_spoof_manifest.csv")
    parser.add_argument("--dataset-root")
    args = parser.parse_args()
    report = validate_pad_manifest(read_pad_manifest(args.manifest), dataset_root=args.dataset_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
