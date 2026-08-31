"""Privacy-preserving monitoring from score/result records, never raw biometrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np


def population_stability_index(
    reference_counts: Iterable[int], current_counts: Iterable[int], epsilon: float = 1e-6
) -> float:
    reference = np.asarray(list(reference_counts), dtype=float)
    current = np.asarray(list(current_counts), dtype=float)
    if reference.shape != current.shape or reference.ndim != 1:
        raise ValueError("reference_counts and current_counts must have the same 1-D shape")
    if reference.sum() <= 0 or current.sum() <= 0:
        raise ValueError("histograms must contain observations")
    reference = np.clip(reference / reference.sum(), epsilon, None)
    current = np.clip(current / current.sum(), epsilon, None)
    return float(np.sum((current - reference) * np.log(current / reference)))


def summarize_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("At least one result record is required")
    decisions = Counter(str(row.get("decision", "unknown")) for row in rows)
    reason_codes = Counter(
        str(reason)
        for row in rows
        for reason in row.get("reason_codes", [])
        if isinstance(reason, str)
    )
    warning_count = sum(bool(row.get("input_warnings")) for row in rows)
    similarity = [
        row["similarity_score"] for row in rows if row.get("similarity_score") is not None
    ]
    liveness = [row["liveness_score"] for row in rows if row.get("liveness_score") is not None]

    def distribution(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "p05": None, "p50": None, "p95": None}
        array = np.asarray(values, dtype=float)
        return {
            "count": len(values),
            "mean": float(array.mean()),
            "p05": float(np.quantile(array, 0.05)),
            "p50": float(np.quantile(array, 0.50)),
            "p95": float(np.quantile(array, 0.95)),
        }

    return {
        "records": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "reason_code_counts": dict(sorted(reason_codes.items())),
        "warning_rate": warning_count / len(rows),
        "similarity": distribution(similarity),
        "liveness": distribution(liveness),
        "privacy_note": "Summary excludes images and embeddings.",
    }


def assess_monitoring_summary(
    summary: dict[str, Any], *, minimum_batch_size: int, warning_rate_limit: float
) -> dict[str, Any]:
    """Apply transparent operational gates without retaining biometric payloads."""
    records = int(summary.get("records", 0))
    warning_rate = float(summary.get("warning_rate", 0.0))
    if records < minimum_batch_size:
        status = "insufficient_data"
        reasons = ["minimum_batch_size_not_met"]
    elif warning_rate > warning_rate_limit:
        status = "warning"
        reasons = ["input_warning_rate_above_limit"]
    else:
        status = "ok"
        reasons = []
    return {
        "status": status,
        "reason_codes": reasons,
        "observed_records": records,
        "minimum_batch_size": minimum_batch_size,
        "observed_warning_rate": warning_rate,
        "warning_rate_limit": warning_rate_limit,
    }
