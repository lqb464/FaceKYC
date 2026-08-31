"""Leakage-safe threshold selection and biometric error metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class VerificationMetrics:
    threshold: float
    genuine_pairs: int
    impostor_pairs: int
    true_accepts: int
    false_rejects: int
    false_accepts: int
    true_rejects: int
    fmr: float
    fnmr: float
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class PADMetrics:
    threshold: float
    bona_fide_samples: int
    attack_samples: int
    bpcer: float
    apcer: float
    acer: float
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _arrays(scores: Iterable[float], labels: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    score_array = np.asarray(list(scores), dtype=float)
    label_array = np.asarray(list(labels), dtype=int)
    if score_array.ndim != 1 or label_array.ndim != 1 or len(score_array) != len(label_array):
        raise ValueError("scores and labels must be one-dimensional and equally sized")
    if len(score_array) == 0 or not np.isfinite(score_array).all():
        raise ValueError("scores must be non-empty and finite")
    if set(np.unique(label_array)) != {0, 1}:
        raise ValueError("labels must contain both 0 and 1")
    return score_array, label_array


def candidate_thresholds(scores: Iterable[float]) -> np.ndarray:
    values = np.unique(np.asarray(list(scores), dtype=float))
    epsilon = max(np.finfo(float).eps, 1e-12)
    return np.concatenate(([values.min() - epsilon], values, [values.max() + epsilon]))


def verification_metrics(
    scores: Iterable[float], labels: Iterable[int], threshold: float
) -> VerificationMetrics:
    values, truth = _arrays(scores, labels)
    accepted = values >= threshold
    genuine = truth == 1
    impostor = ~genuine
    ta = int(np.sum(accepted & genuine))
    fr = int(np.sum(~accepted & genuine))
    fa = int(np.sum(accepted & impostor))
    tr = int(np.sum(~accepted & impostor))
    return VerificationMetrics(
        threshold=float(threshold),
        genuine_pairs=int(genuine.sum()),
        impostor_pairs=int(impostor.sum()),
        true_accepts=ta,
        false_rejects=fr,
        false_accepts=fa,
        true_rejects=tr,
        fmr=fa / int(impostor.sum()),
        fnmr=fr / int(genuine.sum()),
        accuracy=(ta + tr) / len(truth),
    )


def select_verification_threshold(
    scores: Iterable[float], labels: Iterable[int], target_fmr: float
) -> tuple[float, VerificationMetrics]:
    values, truth = _arrays(scores, labels)
    if not 0 <= target_fmr <= 1:
        raise ValueError("target_fmr must be in [0, 1]")
    candidates = [verification_metrics(values, truth, t) for t in candidate_thresholds(values)]
    feasible = [metric for metric in candidates if metric.fmr <= target_fmr]
    selected = min(
        feasible, key=lambda item: (item.fnmr, abs(item.fmr - target_fmr), item.threshold)
    )
    return selected.threshold, selected


def equal_error_rate(
    scores: Iterable[float], labels: Iterable[int]
) -> tuple[float, float, VerificationMetrics]:
    values, truth = _arrays(scores, labels)
    metrics = [verification_metrics(values, truth, t) for t in candidate_thresholds(values)]
    selected = min(metrics, key=lambda item: (abs(item.fmr - item.fnmr), item.fmr + item.fnmr))
    return selected.threshold, (selected.fmr + selected.fnmr) / 2.0, selected


def pad_metrics(scores: Iterable[float], labels: Iterable[int], threshold: float) -> PADMetrics:
    """Compute PAD metrics where label 1 is bona fide and score is P(bona fide)."""
    values, truth = _arrays(scores, labels)
    predicted_live = values >= threshold
    bona_fide = truth == 1
    attacks = ~bona_fide
    bpcer = float(np.mean(~predicted_live[bona_fide]))
    apcer = float(np.mean(predicted_live[attacks]))
    accuracy = float(np.mean(predicted_live == bona_fide))
    return PADMetrics(
        threshold=float(threshold),
        bona_fide_samples=int(bona_fide.sum()),
        attack_samples=int(attacks.sum()),
        bpcer=bpcer,
        apcer=apcer,
        acer=(bpcer + apcer) / 2.0,
        accuracy=accuracy,
    )


def select_pad_threshold(
    scores: Iterable[float], labels: Iterable[int], target_apcer: float
) -> tuple[float, PADMetrics]:
    values, truth = _arrays(scores, labels)
    if not 0 <= target_apcer <= 1:
        raise ValueError("target_apcer must be in [0, 1]")
    candidates = [pad_metrics(values, truth, t) for t in candidate_thresholds(values)]
    feasible = [metric for metric in candidates if metric.apcer <= target_apcer]
    selected = min(
        feasible, key=lambda item: (item.bpcer, abs(item.apcer - target_apcer), item.threshold)
    )
    return selected.threshold, selected
