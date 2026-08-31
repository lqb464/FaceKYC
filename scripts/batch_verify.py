"""Run bounded local batch verification from a CSV manifest."""

from __future__ import annotations

import argparse
import json

from facekyc.batch import run_batch
from facekyc.config import load_settings
from facekyc.pipeline import FaceKYCPipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--bundle")
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="Allow the notebook-derived research candidate; never use for production.",
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    pipeline = FaceKYCPipeline.from_artifacts(
        settings.config_path,
        args.bundle,
        require_approved=not args.allow_candidate,
    )
    summary = run_batch(
        pipeline=pipeline,
        settings=settings,
        manifest_path=args.manifest,
        output_path=args.output,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
