"""FaceNet embedding adapter with raw cosine similarity."""

from __future__ import annotations

from typing import Any

from facekyc.models.detector import VisionDependencyError


class FaceNetRecognizer:
    def __init__(self, pretrained_dataset: str = "vggface2", device: str | None = None):
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise VisionDependencyError(
                'Install the vision dependencies with: pip install -e ".[vision]"'
            ) from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InceptionResnetV1(pretrained=pretrained_dataset).eval().to(self.device)

    def embedding(self, aligned_face: Any) -> Any:
        torch = self._torch
        tensor = aligned_face.unsqueeze(0) if aligned_face.dim() == 3 else aligned_face
        with torch.inference_mode():
            embedding = self.model(tensor.to(self.device))
            return torch.nn.functional.normalize(embedding, p=2, dim=1)

    def similarity(self, first_embedding: Any, second_embedding: Any) -> float:
        with self._torch.inference_mode():
            score = self._torch.nn.functional.cosine_similarity(first_embedding, second_embedding)
        return float(score.detach().cpu().item())
