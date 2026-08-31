"""MobileNetV3 passive presentation-attack detector."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from facekyc.models.detector import VisionDependencyError


class CNNPAD:
    def __init__(
        self,
        weights_path: str | Path,
        *,
        architecture: str = "mobilenet_v3_small",
        live_class_index: int = 1,
        input_size: int = 224,
        threshold: float | None = None,
        device: str | None = None,
    ):
        try:
            import torch
            from torchvision import transforms

            from facekyc.pad import build_pad_model
        except ImportError as exc:
            raise VisionDependencyError(
                'Install the vision dependencies with: pip install -e ".[vision]"'
            ) from exc

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"PAD weights not found: {weights_path}")
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.live_class_index = live_class_index
        self.model = build_pad_model(architecture=architecture, imagenet_pretrained=False)
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            expected = {
                "architecture": architecture,
                "input_size": input_size,
                "class_contract": {"0": "attack", "1": "bona_fide"},
                "score": "softmax_probability_class_1",
            }
            mismatches = [
                key
                for key, value in expected.items()
                if key in checkpoint and checkpoint[key] != value
            ]
            if threshold is not None and "threshold" in checkpoint:
                if abs(float(checkpoint["threshold"]) - threshold) > 1e-12:
                    mismatches.append("threshold")
            if mismatches:
                raise ValueError(
                    "PAD checkpoint metadata does not match model bundle: "
                    + ", ".join(sorted(mismatches))
                )
        state_dict = (
            checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        )
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def score(self, face_crop: Image.Image) -> float:
        tensor = self.transform(face_crop.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            probabilities = self._torch.softmax(self.model(tensor), dim=1)
        return float(probabilities[0, self.live_class_index].detach().cpu().item())


MobileNetPAD = CNNPAD
