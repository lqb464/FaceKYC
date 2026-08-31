"""Promote the immutable notebook-05 report into a research serving bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from facekyc.artifacts import build_bundle, save_bundle, sha256_file


class PromotionError(ValueError):
    """Raised when locked DS evidence cannot be promoted safely."""


def _load_report(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"Cannot read locked holdout report: {path}") from exc
    if not isinstance(payload, dict):
        raise PromotionError("Locked holdout report must be a JSON object")
    return payload


def _weights_reference(weights: Path, output: Path) -> str:
    try:
        return weights.resolve().relative_to(output.parent.parent.resolve()).as_posix()
    except ValueError:
        return str(weights.resolve())


def promote_notebook_report(
    *,
    report_path: str | Path,
    weights_path: str | Path,
    output_path: str | Path,
    model_version: str,
) -> dict[str, Any]:
    report_file = Path(report_path)
    weights_file = Path(weights_path)
    output_file = Path(output_path)
    report = _load_report(report_file)

    if report.get("deployment_status") != "candidate_research_only":
        raise PromotionError("Notebook report must remain candidate_research_only")
    if report.get("holdout_accessed") is not True:
        raise PromotionError("Locked holdout evidence is missing")
    if report.get("holdout_evaluation_count") != 1:
        raise PromotionError("Holdout must have been evaluated exactly once")
    screening_gates = report.get("screening_gates")
    if not isinstance(screening_gates, dict) or not screening_gates:
        raise PromotionError("Screening gates are missing")
    if report.get("screening_passed") is not True or not all(
        value is True for value in screening_gates.values()
    ):
        raise PromotionError("Locked holdout screening gates did not all pass")
    if not weights_file.exists():
        raise PromotionError(f"PAD checkpoint not found: {weights_file}")

    actual_hash = sha256_file(weights_file)
    expected_hash = report.get("artifact_integrity", {}).get("pad_checkpoint_sha256")
    if actual_hash != expected_hash:
        raise PromotionError("PAD checkpoint SHA-256 does not match notebook 05")

    verification = report.get("verification", {})
    pad_proxy = report.get("pad_proxy", {})
    verification_metrics = verification.get("locked_holdout")
    liveness_metrics = pad_proxy.get("locked_holdout")
    if not isinstance(verification_metrics, dict) or not isinstance(liveness_metrics, dict):
        raise PromotionError("Locked holdout metrics are missing")

    governance = {
        "production_ready": False,
        "research_only": True,
        "source_report": str(report_file),
        "holdout_evaluation_count": 1,
        "screening_gates": screening_gates,
        "screening_passed": True,
        "limitations": list(report.get("limitations", [])),
    }
    bundle = build_bundle(
        model_version=model_version,
        verification_threshold=float(verification_metrics["threshold"]),
        verification_metrics=verification_metrics,
        verification_pretrained_dataset=str(verification["backbone"]),
        liveness_threshold=float(liveness_metrics["threshold"]),
        liveness_metrics=liveness_metrics,
        liveness_weights_path=_weights_reference(weights_file, output_file),
        liveness_weights_sha256=actual_hash,
        liveness_architecture=str(pad_proxy["architecture"]),
        liveness_input_size=224,
        data_protocols={
            "verification": (
                "LFW folds 0-7 development, fold 8 threshold lock, fold 9 one-time locked holdout"
            ),
            "liveness": (
                "Subject-disjoint LFW-derived synthetic print/replay/recapture proxy; "
                "not a real PAD benchmark"
            ),
        },
        governance=governance,
        deployment_status="candidate",
    )
    save_bundle(bundle, output_file)
    return bundle
