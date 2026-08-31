"""Shared data contracts for image stages and API-safe results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image


@dataclass
class FaceObservation:
    tensor: Any
    crop: Image.Image
    bbox: tuple[float, float, float, float]
    probability: float
    face_count: int
    landmarks: list[list[float]] | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageQualityReport:
    accepted: bool
    width: int
    height: int
    brightness: float
    sharpness: float
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        result["rejection_reasons"] = list(self.rejection_reasons)
        return result


@dataclass(frozen=True)
class VerificationResult:
    status: str
    decision: str
    reason_codes: tuple[str, ...]
    similarity_score: float | None
    liveness_score: float | None
    thresholds: dict[str, float]
    image_quality: dict[str, dict[str, Any]]
    input_warnings: tuple[str, ...]
    model_version: str
    deployment_status: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reason_codes"] = list(self.reason_codes)
        result["input_warnings"] = list(self.input_warnings)
        return result
