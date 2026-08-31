"""Versioned, checksummed FaceKYC serving bundles."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError):
    """Raised when a model bundle is absent, malformed, or tampered with."""


REQUIRED_KEYS = {
    "schema_version",
    "model_version",
    "created_at",
    "verification",
    "liveness",
    "metrics",
    "data_protocols",
    "governance",
    "deployment_status",
    "bundle_checksum",
}
ALLOWED_DEPLOYMENT_STATUSES = {"candidate", "approved"}
SUPPORTED_VERIFICATION_BACKBONES = {"vggface2", "casia-webface"}
SUPPORTED_PAD_ARCHITECTURES = {"mobilenet_v3_small", "resnet18"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    clean = {key: value for key, value in payload.items() if key != "bundle_checksum"}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def bundle_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ArtifactError(f"Model bundle field {key!r} must be an object")
    return value


def _finite_threshold(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{name} threshold is missing or invalid")
    threshold = float(value)
    if not math.isfinite(threshold) or not lower <= threshold <= upper:
        raise ArtifactError(f"{name} threshold must be finite and in [{lower}, {upper}]")
    return threshold


def validate_bundle(payload: dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(payload)
    if missing:
        raise ArtifactError(f"Model bundle missing keys: {sorted(missing)}")
    if payload["schema_version"] != "1.1":
        raise ArtifactError(f"Unsupported bundle schema: {payload['schema_version']!r}")
    if payload["bundle_checksum"] != bundle_checksum(payload):
        raise ArtifactError("Model bundle checksum mismatch")
    if not isinstance(payload["model_version"], str) or not payload["model_version"].strip():
        raise ArtifactError("model_version must be a non-empty string")
    try:
        datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactError("created_at must be an ISO-8601 timestamp") from exc

    status = payload["deployment_status"]
    if status not in ALLOWED_DEPLOYMENT_STATUSES:
        raise ArtifactError(f"Unsupported deployment_status: {status!r}")

    verification = _mapping(payload, "verification")
    if verification.get("architecture") != "InceptionResnetV1":
        raise ArtifactError("Unsupported verification architecture")
    if verification.get("pretrained_dataset") not in SUPPORTED_VERIFICATION_BACKBONES:
        raise ArtifactError("Unsupported verification pretrained dataset")
    if verification.get("score") != "cosine_similarity":
        raise ArtifactError("Unsupported verification score contract")
    _finite_threshold(verification.get("threshold"), "verification", -1.0, 1.0)

    liveness = _mapping(payload, "liveness")
    if liveness.get("architecture") not in SUPPORTED_PAD_ARCHITECTURES:
        raise ArtifactError("Unsupported liveness architecture")
    if liveness.get("score") != "softmax_probability_class_1":
        raise ArtifactError("Unsupported liveness score contract")
    if liveness.get("class_contract") != {"0": "attack", "1": "bona_fide"}:
        raise ArtifactError("Invalid liveness class contract")
    if liveness.get("live_class_index") != 1:
        raise ArtifactError("liveness.live_class_index must be 1")
    input_size = liveness.get("input_size")
    if isinstance(input_size, bool) or not isinstance(input_size, int) or input_size < 32:
        raise ArtifactError("liveness.input_size must be an integer >= 32")
    _finite_threshold(liveness.get("threshold"), "liveness", 0.0, 1.0)
    if not isinstance(liveness.get("weights_path"), str) or not liveness["weights_path"]:
        raise ArtifactError("Liveness weights path is missing")
    weights_sha256 = liveness.get("weights_sha256")
    if (
        not isinstance(weights_sha256, str)
        or len(weights_sha256) != 64
        or any(character not in "0123456789abcdef" for character in weights_sha256.lower())
    ):
        raise ArtifactError("Liveness weights SHA-256 is invalid")

    _mapping(payload, "metrics")
    _mapping(payload, "data_protocols")
    governance = _mapping(payload, "governance")
    if not isinstance(governance.get("production_ready"), bool):
        raise ArtifactError("governance.production_ready must be boolean")
    if not isinstance(governance.get("research_only"), bool):
        raise ArtifactError("governance.research_only must be boolean")
    if status == "approved" and (not governance["production_ready"] or governance["research_only"]):
        raise ArtifactError("Approved bundles must be production-ready and not research-only")


def build_bundle(
    *,
    model_version: str,
    verification_threshold: float,
    verification_metrics: dict[str, Any],
    liveness_threshold: float,
    liveness_metrics: dict[str, Any],
    liveness_weights_path: str,
    liveness_weights_sha256: str,
    data_protocols: dict[str, Any],
    verification_pretrained_dataset: str = "vggface2",
    liveness_architecture: str = "mobilenet_v3_small",
    liveness_input_size: int = 224,
    governance: dict[str, Any] | None = None,
    deployment_status: str = "candidate",
) -> dict[str, Any]:
    governance_payload = dict(governance or {})
    governance_payload.setdefault("production_ready", deployment_status == "approved")
    governance_payload.setdefault("research_only", deployment_status != "approved")
    governance_payload.setdefault("screening_gates", {})
    governance_payload.setdefault("limitations", [])
    payload: dict[str, Any] = {
        "schema_version": "1.1",
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(),
        "verification": {
            "architecture": "InceptionResnetV1",
            "pretrained_dataset": verification_pretrained_dataset,
            "score": "cosine_similarity",
            "threshold": float(verification_threshold),
        },
        "liveness": {
            "architecture": liveness_architecture,
            "live_class_index": 1,
            "input_size": int(liveness_input_size),
            "class_contract": {"0": "attack", "1": "bona_fide"},
            "score": "softmax_probability_class_1",
            "threshold": float(liveness_threshold),
            "weights_path": liveness_weights_path,
            "weights_sha256": liveness_weights_sha256,
        },
        "metrics": {"verification": verification_metrics, "liveness": liveness_metrics},
        "data_protocols": data_protocols,
        "governance": governance_payload,
        "deployment_status": deployment_status,
    }
    payload["bundle_checksum"] = bundle_checksum(payload)
    validate_bundle(payload)
    return payload


def save_bundle(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["bundle_checksum"] = bundle_checksum(payload)
    validate_bundle(payload)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _resolve_weights_path(bundle_path: Path, weights_value: str) -> Path:
    value = Path(weights_value)
    if value.is_absolute():
        return value
    candidates = [bundle_path.parent / value, bundle_path.parent.parent / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_bundle(path: str | Path, *, verify_weights: bool = True) -> dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.exists():
        raise ArtifactError(f"Model bundle not found: {bundle_path}")
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot read model bundle: {bundle_path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("Model bundle root must be an object")
    validate_bundle(payload)
    if verify_weights:
        weights_path = _resolve_weights_path(bundle_path, payload["liveness"]["weights_path"])
        if not weights_path.exists():
            raise ArtifactError(f"Liveness weights not found: {weights_path}")
        if sha256_file(weights_path) != payload["liveness"]["weights_sha256"]:
            raise ArtifactError("Liveness weights checksum mismatch")
        payload["_resolved_liveness_weights_path"] = str(weights_path.resolve())
    return payload
