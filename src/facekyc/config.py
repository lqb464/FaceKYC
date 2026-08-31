"""Typed configuration and project path handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


class ConfigurationError(ValueError):
    """Raised when a configuration cannot support a safe inference pipeline."""


@dataclass(frozen=True)
class InputSettings:
    allowed_formats: tuple[str, ...]
    max_file_bytes: int
    min_width: int
    min_height: int
    max_width: int
    max_height: int
    min_brightness: float
    max_brightness: float
    min_sharpness: float


@dataclass(frozen=True)
class DetectionSettings:
    image_size: int
    margin: int
    min_face_size: int
    min_probability: float
    thresholds: tuple[float, float, float]
    factor: float
    reject_multiple_faces: bool


@dataclass(frozen=True)
class VerificationSettings:
    architecture: str
    pretrained_dataset: str
    score_name: str
    threshold: float | None
    uncertainty_margin: float
    calibration_target_fmr: float


@dataclass(frozen=True)
class LivenessSettings:
    architecture: str
    experiment_candidates: tuple[str, ...]
    live_class_index: int
    input_size: int
    threshold: float | None
    uncertainty_margin: float
    calibration_target_apcer: float
    weights_path: str | None
    required: bool


@dataclass(frozen=True)
class DecisionSettings:
    review_on_input_warning: bool
    review_on_uncertain_score: bool
    low_similarity_action: str
    failed_pad_action: str


@dataclass(frozen=True)
class MonitoringSettings:
    minimum_batch_size: int
    psi_warning: float
    psi_critical: float
    warning_rate_limit: float
    store_biometric_images: bool
    store_embeddings: bool


@dataclass(frozen=True)
class ArtifactSettings:
    bundle_path: str


@dataclass(frozen=True)
class BatchSettings:
    max_records: int


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    version: str
    purpose: str


@dataclass(frozen=True)
class Settings:
    project: ProjectSettings
    input: InputSettings
    detection: DetectionSettings
    verification: VerificationSettings
    liveness: LivenessSettings
    decision: DecisionSettings
    monitoring: MonitoringSettings
    artifact: ArtifactSettings
    batch: BatchSettings
    config_path: Path

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _required(mapping: dict[str, Any], section: str) -> dict[str, Any]:
    value = mapping.get(section)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing configuration section: {section}")
    return value


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be a number or null") from exc


def load_settings(path: str | Path | None = None) -> Settings:
    """Load and validate the single source of truth for training and inference."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    project = ProjectSettings(**_required(raw, "project"))
    input_raw = _required(raw, "input")
    input_settings = InputSettings(
        allowed_formats=tuple(str(item).upper() for item in input_raw["allowed_formats"]),
        **{key: value for key, value in input_raw.items() if key != "allowed_formats"},
    )
    detection_raw = _required(raw, "detection")
    thresholds = tuple(float(item) for item in detection_raw["thresholds"])
    if len(thresholds) != 3:
        raise ConfigurationError("detection.thresholds must contain exactly three values")
    detection = DetectionSettings(
        thresholds=thresholds,
        **{key: value for key, value in detection_raw.items() if key != "thresholds"},
    )
    verification_raw = _required(raw, "verification")
    verification = VerificationSettings(
        threshold=_optional_float(verification_raw.get("threshold"), "verification.threshold"),
        **{key: value for key, value in verification_raw.items() if key != "threshold"},
    )
    liveness_raw = _required(raw, "liveness")
    liveness = LivenessSettings(
        threshold=_optional_float(liveness_raw.get("threshold"), "liveness.threshold"),
        experiment_candidates=tuple(liveness_raw["experiment_candidates"]),
        **{
            key: value
            for key, value in liveness_raw.items()
            if key not in {"threshold", "experiment_candidates"}
        },
    )
    settings = Settings(
        project=project,
        input=input_settings,
        detection=detection,
        verification=verification,
        liveness=liveness,
        decision=DecisionSettings(**_required(raw, "decision")),
        monitoring=MonitoringSettings(**_required(raw, "monitoring")),
        artifact=ArtifactSettings(**_required(raw, "artifact")),
        batch=BatchSettings(**_required(raw, "batch")),
        config_path=config_path.resolve(),
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: Settings) -> None:
    if not settings.input.allowed_formats:
        raise ConfigurationError("input.allowed_formats cannot be empty")
    if settings.input.max_file_bytes <= 0:
        raise ConfigurationError("input.max_file_bytes must be positive")
    if settings.input.min_width > settings.input.max_width:
        raise ConfigurationError("input.min_width cannot exceed input.max_width")
    if settings.input.min_height > settings.input.max_height:
        raise ConfigurationError("input.min_height cannot exceed input.max_height")
    for name, value in (
        ("detection.min_probability", settings.detection.min_probability),
        ("verification.calibration_target_fmr", settings.verification.calibration_target_fmr),
        ("verification.uncertainty_margin", settings.verification.uncertainty_margin),
        ("liveness.calibration_target_apcer", settings.liveness.calibration_target_apcer),
        ("liveness.uncertainty_margin", settings.liveness.uncertainty_margin),
        ("monitoring.psi_warning", settings.monitoring.psi_warning),
        ("monitoring.psi_critical", settings.monitoring.psi_critical),
        ("monitoring.warning_rate_limit", settings.monitoring.warning_rate_limit),
    ):
        if not 0 <= value <= 1:
            raise ConfigurationError(f"{name} must be in [0, 1]")
    if settings.decision.low_similarity_action not in {"not_verified", "manual_review"}:
        raise ConfigurationError("Unsupported decision.low_similarity_action")
    if settings.decision.failed_pad_action not in {"not_verified", "manual_review"}:
        raise ConfigurationError("Unsupported decision.failed_pad_action")
    supported_pad = {"mobilenet_v3_small", "resnet18"}
    if settings.liveness.architecture not in supported_pad:
        raise ConfigurationError("Unsupported liveness.architecture")
    if not set(settings.liveness.experiment_candidates).issubset(supported_pad):
        raise ConfigurationError("Unsupported CNN architecture in liveness.experiment_candidates")
    if settings.liveness.live_class_index != 1:
        raise ConfigurationError("liveness.live_class_index must be 1")
    if settings.liveness.input_size < 32:
        raise ConfigurationError("liveness.input_size must be at least 32")
    if not settings.liveness.required:
        raise ConfigurationError("FaceKYC requires liveness screening")
    if settings.monitoring.minimum_batch_size <= 0:
        raise ConfigurationError("monitoring.minimum_batch_size must be positive")
    if settings.monitoring.psi_warning >= settings.monitoring.psi_critical:
        raise ConfigurationError("monitoring.psi_warning must be below psi_critical")
    if settings.batch.max_records <= 0:
        raise ConfigurationError("batch.max_records must be positive")
