"""Reusable CelebA-Spoof dataset and MobileNetV3 PAD helpers."""

from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image


class PADManifestDataset:
    def __init__(self, rows: Iterable[dict[str, str]], dataset_root: str | Path, transform: Any):
        self.rows = list(rows)
        self.root = Path(dataset_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / row["path"]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"])


def build_pad_model(*, architecture: str, imagenet_pretrained: bool):
    try:
        import torch
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            ResNet18_Weights,
            mobilenet_v3_small,
            resnet18,
        )
    except ImportError as exc:
        raise RuntimeError('Install the vision extra with: pip install -e ".[vision]"') from exc
    if architecture == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if imagenet_pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
        return model
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        model = resnet18(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        return model
    raise ValueError(f"Unsupported PAD CNN architecture: {architecture}")


def pad_transform(input_size: int, *, training: bool):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError('Install the vision extra with: pip install -e ".[vision]"') from exc
    operations: list[Any] = [transforms.Resize((input_size, input_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10),
                transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


def subset_rows(rows: list[dict[str, str]], split: str, maximum: int | None, seed: int):
    selected = [row for row in rows if row["split"] == split]
    if maximum is not None and len(selected) > maximum:
        rng = random.Random(seed)
        selected = rng.sample(selected, maximum)
    return selected


def predict_pad_scores(model: Any, loader: Any, device: str) -> tuple[list[float], list[int]]:
    import torch

    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for images, batch_labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            scores.extend(probabilities.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())
    return scores, labels
