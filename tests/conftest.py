from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
from PIL import Image

from facekyc.config import load_settings
from facekyc.contracts import FaceObservation
from facekyc.pipeline import FaceKYCPipeline


class FakeDetector:
    def __init__(self, *, probability: float = 0.99, face_count: int = 1):
        self.probability = probability
        self.face_count = face_count

    def detect(self, image: Image.Image) -> FaceObservation:
        return FaceObservation(
            tensor=image.size,
            crop=image.crop((0, 0, min(224, image.width), min(224, image.height))),
            bbox=(20.0, 20.0, 220.0, 220.0),
            probability=self.probability,
            face_count=self.face_count,
            warnings=["multiple_faces_detected"] if self.face_count > 1 else [],
        )


class FakeRecognizer:
    def __init__(self, similarity: float):
        self.similarity_score = similarity

    def embedding(self, aligned_face: Any) -> Any:
        return aligned_face

    def similarity(self, first_embedding: Any, second_embedding: Any) -> float:
        return self.similarity_score


class FakeLiveness:
    def __init__(self, score: float):
        self.value = score

    def score(self, face_crop: Image.Image) -> float:
        return self.value


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def valid_image() -> Image.Image:
    rng = np.random.default_rng(42)
    pixels = rng.integers(30, 225, size=(320, 320, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


@pytest.fixture
def image_bytes(valid_image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    valid_image.save(buffer, format="JPEG")
    return buffer.getvalue()


def make_pipeline(settings, *, similarity=0.80, liveness=0.90, face_count=1, probability=0.99):
    bundle = {
        "model_version": "test-1",
        "deployment_status": "approved",
        "verification": {"threshold": 0.70},
        "liveness": {"threshold": 0.60},
    }
    return FaceKYCPipeline(
        settings=settings,
        bundle=bundle,
        detector=FakeDetector(probability=probability, face_count=face_count),
        recognizer=FakeRecognizer(similarity),
        liveness=FakeLiveness(liveness),
    )
