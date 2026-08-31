from __future__ import annotations

import json

import pytest

from facekyc.artifacts import sha256_file
from facekyc.promotion import PromotionError, promote_notebook_report


def _report(checkpoint_hash: str) -> dict:
    return {
        "verification": {
            "backbone": "vggface2",
            "locked_holdout": {"threshold": 0.42, "fmr": 0.01, "fnmr": 0.04},
        },
        "pad_proxy": {
            "architecture": "mobilenet_v3_small",
            "locked_holdout": {"threshold": 0.76, "apcer": 0.06, "bpcer": 0.001},
        },
        "deployment_status": "candidate_research_only",
        "holdout_accessed": True,
        "holdout_evaluation_count": 1,
        "screening_gates": {"verification": True, "pad_proxy": True},
        "screening_passed": True,
        "artifact_integrity": {"pad_checkpoint_sha256": checkpoint_hash},
        "limitations": ["Synthetic PAD proxy only."],
    }


def test_promotion_creates_candidate_and_never_self_approves(tmp_path):
    weights = tmp_path / "pad.pt"
    weights.write_bytes(b"weights")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report(sha256_file(weights))), encoding="utf-8")
    bundle = promote_notebook_report(
        report_path=report,
        weights_path=weights,
        output_path=tmp_path / "artifacts" / "bundle.json",
        model_version="research-1",
    )
    assert bundle["deployment_status"] == "candidate"
    assert bundle["governance"]["research_only"] is True
    assert bundle["governance"]["production_ready"] is False


def test_promotion_rejects_checkpoint_mismatch(tmp_path):
    weights = tmp_path / "pad.pt"
    weights.write_bytes(b"weights")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report("0" * 64)), encoding="utf-8")
    with pytest.raises(PromotionError, match="SHA-256"):
        promote_notebook_report(
            report_path=report,
            weights_path=weights,
            output_path=tmp_path / "bundle.json",
            model_version="research-1",
        )
