from __future__ import annotations

import pytest

from facekyc.evaluation import (
    equal_error_rate,
    pad_metrics,
    select_pad_threshold,
    select_verification_threshold,
)


def test_verification_threshold_respects_fmr_target():
    scores = [0.95, 0.90, 0.82, 0.74, 0.70, 0.60, 0.42, 0.30]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    threshold, metrics = select_verification_threshold(scores, labels, target_fmr=0.0)
    assert metrics.fmr == 0.0
    assert threshold > 0.70
    assert metrics.fnmr <= 0.25


def test_eer_returns_consistent_operating_point():
    _, eer, metrics = equal_error_rate([0.9, 0.8, 0.6, 0.4], [1, 1, 0, 0])
    assert eer == pytest.approx((metrics.fmr + metrics.fnmr) / 2)


def test_pad_metric_semantics_are_not_reversed():
    metrics = pad_metrics([0.9, 0.3, 0.8, 0.2], [1, 1, 0, 0], threshold=0.5)
    assert metrics.bpcer == 0.5  # one bona-fide sample rejected
    assert metrics.apcer == 0.5  # one attack accepted
    assert metrics.acer == 0.5


def test_pad_threshold_respects_apcer_target():
    threshold, metrics = select_pad_threshold(
        [0.95, 0.85, 0.70, 0.30, 0.20, 0.10], [1, 1, 1, 0, 0, 0], 0.0
    )
    assert metrics.apcer == 0.0
    assert threshold > 0.30
