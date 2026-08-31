"""MTCNN face detection, alignment, and crop diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from facekyc.config import DetectionSettings
from facekyc.contracts import FaceObservation


class VisionDependencyError(RuntimeError):
    """Raised when optional computer-vision dependencies are unavailable."""


class MTCNNFaceDetector:
    def __init__(self, settings: DetectionSettings, device: str | None = None):
        try:
            import torch
            from facenet_pytorch import MTCNN
        except ImportError as exc:
            raise VisionDependencyError(
                'Install the vision dependencies with: pip install -e ".[vision]"'
            ) from exc

        self.settings = settings
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MTCNN(
            image_size=settings.image_size,
            margin=settings.margin,
            min_face_size=settings.min_face_size,
            thresholds=list(settings.thresholds),
            factor=settings.factor,
            post_process=True,
            keep_all=True,
            device=self.device,
        )

    def detect(self, image: Image.Image) -> FaceObservation | None:
        rgb = image.convert("RGB")
        boxes, probabilities, landmarks = self.model.detect(rgb, landmarks=True)
        if boxes is None or probabilities is None:
            return None

        boxes_array = np.asarray(boxes, dtype=float)
        probabilities_array = np.asarray(probabilities, dtype=float)
        finite = np.isfinite(boxes_array).all(axis=1) & np.isfinite(probabilities_array)
        boxes_array = boxes_array[finite]
        probabilities_array = probabilities_array[finite]
        if boxes_array.size == 0:
            return None

        landmarks_array = None if landmarks is None else np.asarray(landmarks)[finite]
        areas = np.maximum(0.0, boxes_array[:, 2] - boxes_array[:, 0]) * np.maximum(
            0.0, boxes_array[:, 3] - boxes_array[:, 1]
        )
        selected_index = int(np.argmax(areas))
        selected_box = boxes_array[selected_index]
        aligned = self.model.extract(rgb, boxes_array[[selected_index]], save_path=None)
        if aligned is None:
            return None
        face_tensor: Any = aligned[0] if getattr(aligned, "ndim", 0) == 4 else aligned

        x1, y1, x2, y2 = selected_box.tolist()
        width, height = rgb.size
        expand = self.settings.margin / 2.0
        crop_box = (
            max(0, int(np.floor(x1 - expand))),
            max(0, int(np.floor(y1 - expand))),
            min(width, int(np.ceil(x2 + expand))),
            min(height, int(np.ceil(y2 + expand))),
        )
        crop = rgb.crop(crop_box)
        face_count = int(len(boxes_array))
        warnings = ["multiple_faces_detected"] if face_count > 1 else []
        selected_landmarks = None
        if landmarks_array is not None:
            selected_landmarks = landmarks_array[selected_index].astype(float).tolist()
        return FaceObservation(
            tensor=face_tensor,
            crop=crop,
            bbox=tuple(float(value) for value in selected_box),
            probability=float(probabilities_array[selected_index]),
            face_count=face_count,
            landmarks=selected_landmarks,
            warnings=warnings,
        )
