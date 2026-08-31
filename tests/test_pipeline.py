from __future__ import annotations

from dataclasses import replace

from conftest import make_pipeline


def test_pipeline_verified_path(settings, valid_image):
    result = make_pipeline(settings).verify(valid_image, valid_image)
    assert result["decision"] == "verified"
    assert result["similarity_score"] == 0.8
    assert result["liveness_score"] == 0.9
    assert result["model_version"] == "test-1"
    assert result["deployment_status"] == "approved"


def test_pipeline_sends_uncertain_match_to_review(settings, valid_image):
    result = make_pipeline(settings, similarity=0.71).verify(valid_image, valid_image)
    assert result["decision"] == "manual_review"
    assert "uncertain_similarity_score" in result["reason_codes"]


def test_pipeline_can_disable_uncertainty_review(settings, valid_image):
    settings = replace(
        settings,
        decision=replace(settings.decision, review_on_uncertain_score=False),
    )
    result = make_pipeline(settings, similarity=0.71).verify(valid_image, valid_image)
    assert result["decision"] == "verified"


def test_pipeline_does_not_verify_failed_pad(settings, valid_image):
    result = make_pipeline(settings, similarity=0.95, liveness=0.2).verify(valid_image, valid_image)
    assert result["decision"] == "not_verified"
    assert "presentation_attack_suspected" in result["reason_codes"]


def test_pipeline_rejects_multiple_faces(settings, valid_image):
    result = make_pipeline(settings, face_count=2).verify(valid_image, valid_image)
    assert result["decision"] == "recapture"
    assert "multiple_faces_detected" in result["reason_codes"]


def test_pipeline_rejects_low_detection_confidence(settings, valid_image):
    result = make_pipeline(settings, probability=0.4).verify(valid_image, valid_image)
    assert result["decision"] == "recapture"
    assert "id_image_low_detection_confidence" in result["reason_codes"]
