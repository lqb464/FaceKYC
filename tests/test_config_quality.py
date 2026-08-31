from __future__ import annotations

from PIL import Image

from facekyc.quality import assess_image_quality


def test_default_config_has_unlocked_thresholds(settings):
    assert settings.verification.threshold is None
    assert settings.liveness.threshold is None
    assert settings.monitoring.store_biometric_images is False
    assert settings.monitoring.store_embeddings is False


def test_dark_image_requires_recapture(settings):
    report = assess_image_quality(Image.new("RGB", (320, 320), "black"), settings.input)
    assert report.accepted is False
    assert "image_too_dark" in report.rejection_reasons


def test_textured_image_passes_hard_quality_gate(settings, valid_image):
    report = assess_image_quality(valid_image, settings.input)
    assert report.accepted is True
    assert report.width == 320
