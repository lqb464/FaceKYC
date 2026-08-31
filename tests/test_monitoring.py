from __future__ import annotations

import pytest

from facekyc.monitoring import (
    assess_monitoring_summary,
    population_stability_index,
    summarize_results,
)


def test_monitoring_summary_contains_no_biometrics():
    report = summarize_results(
        [
            {
                "decision": "verified",
                "similarity_score": 0.8,
                "liveness_score": 0.9,
                "input_warnings": [],
            },
            {
                "decision": "manual_review",
                "similarity_score": 0.7,
                "liveness_score": 0.6,
                "input_warnings": ["blur"],
            },
        ]
    )
    assert report["warning_rate"] == 0.5
    assert report["reason_code_counts"] == {}
    assert "images" not in report
    assert "embeddings" not in report


def test_psi_is_zero_for_identical_histograms():
    assert population_stability_index([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)


def test_monitoring_assessment_is_explicit_about_small_batches():
    assessment = assess_monitoring_summary(
        {"records": 20, "warning_rate": 0.0},
        minimum_batch_size=100,
        warning_rate_limit=0.1,
    )
    assert assessment["status"] == "insufficient_data"
    assert assessment["reason_codes"] == ["minimum_batch_size_not_met"]
