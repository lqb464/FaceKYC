"""Small configuration/artifact validation entry point."""

from __future__ import annotations

import argparse
import json

from facekyc.artifacts import ArtifactError, load_bundle
from facekyc.config import load_settings


def validate_main() -> int:
    parser = argparse.ArgumentParser(description="Validate FaceKYC configuration and artifact")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--bundle")
    parser.add_argument("--allow-missing-artifact", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    bundle_path = settings.resolve(args.bundle or settings.artifact.bundle_path)
    report = {
        "config": str(settings.config_path),
        "project_version": settings.project.version,
        "bundle": str(bundle_path),
        "ready": False,
    }
    try:
        bundle = load_bundle(bundle_path, verify_weights=True)
        report.update(
            ready=bundle["deployment_status"] == "approved",
            model_version=bundle["model_version"],
            deployment_status=bundle["deployment_status"],
        )
    except ArtifactError as exc:
        report["artifact_error"] = str(exc)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if args.allow_missing_artifact else 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 3


if __name__ == "__main__":
    raise SystemExit(validate_main())
