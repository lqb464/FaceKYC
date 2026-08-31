from __future__ import annotations

import json

import pytest

from facekyc.artifacts import ArtifactError, build_bundle, load_bundle, save_bundle, sha256_file
from facekyc.pipeline import FaceKYCPipeline


def test_artifact_roundtrip_and_weight_checksum(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    weights = artifacts / "pad.pth"
    weights.write_bytes(b"test-weights")
    payload = build_bundle(
        model_version="test-1",
        verification_threshold=0.7,
        verification_metrics={"fmr": 0.01, "fnmr": 0.1},
        liveness_threshold=0.6,
        liveness_metrics={"apcer": 0.05, "bpcer": 0.1},
        liveness_weights_path="artifacts/pad.pth",
        liveness_weights_sha256=sha256_file(weights),
        data_protocols={"test": "locked"},
        deployment_status="approved",
    )
    path = save_bundle(payload, artifacts / "facekyc_bundle.json")
    loaded = load_bundle(path)
    assert loaded["model_version"] == "test-1"
    assert loaded["_resolved_liveness_weights_path"] == str(weights.resolve())


def test_artifact_tampering_is_detected(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"bundle_checksum": "bad"}), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_bundle(path, verify_weights=False)


def test_candidate_bundle_is_refused_by_default_serving_gate(tmp_path):
    weights = tmp_path / "pad.pth"
    weights.write_bytes(b"test-weights")
    bundle = build_bundle(
        model_version="research-1",
        verification_threshold=0.7,
        verification_metrics={"fmr": 0.01},
        liveness_threshold=0.6,
        liveness_metrics={"apcer": 0.05},
        liveness_weights_path="pad.pth",
        liveness_weights_sha256=sha256_file(weights),
        data_protocols={"test": "locked"},
        deployment_status="candidate",
    )
    path = save_bundle(bundle, tmp_path / "bundle.json")
    with pytest.raises(ArtifactError, match="not approved"):
        FaceKYCPipeline.from_artifacts(bundle_path=path)
