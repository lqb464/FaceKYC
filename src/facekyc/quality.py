"""Image-level quality checks that run before biometric models."""

from __future__ import annotations

import numpy as np
from PIL import Image

from facekyc.config import InputSettings
from facekyc.contracts import ImageQualityReport


def _sharpness(gray: np.ndarray) -> float:
    """Variance of a discrete Laplacian without an OpenCV runtime dependency."""
    if min(gray.shape) < 3:
        return 0.0
    center = gray[1:-1, 1:-1] * -4.0
    laplacian = center + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(np.var(laplacian))


def assess_image_quality(image: Image.Image, settings: InputSettings) -> ImageQualityReport:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean()) if gray.size else 0.0
    sharpness = _sharpness(gray)
    rejection_reasons: list[str] = []
    warnings: list[str] = []

    if width < settings.min_width or height < settings.min_height:
        rejection_reasons.append("image_too_small")
    if width > settings.max_width or height > settings.max_height:
        rejection_reasons.append("image_too_large")
    if brightness < settings.min_brightness:
        rejection_reasons.append("image_too_dark")
    if brightness > settings.max_brightness:
        rejection_reasons.append("image_too_bright")
    if sharpness < settings.min_sharpness:
        warnings.append("image_may_be_blurry")

    return ImageQualityReport(
        accepted=not rejection_reasons,
        width=width,
        height=height,
        brightness=round(brightness, 3),
        sharpness=round(sharpness, 3),
        warnings=tuple(warnings),
        rejection_reasons=tuple(rejection_reasons),
    )
