"""PAD manifest creation and subject-leakage validation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REQUIRED_MANIFEST_COLUMNS = {"path", "label", "split", "subject"}
ALLOWED_SPLITS = {"train", "validation", "test"}


def read_pad_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_MANIFEST_COLUMNS - columns
        if missing:
            raise ValueError(f"PAD manifest missing columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def validate_pad_manifest(
    rows: Iterable[dict[str, str]], *, dataset_root: str | Path | None = None
) -> dict[str, Any]:
    records = list(rows)
    if not records:
        raise ValueError("PAD manifest is empty")
    invalid_labels = sorted({row["label"] for row in records if row["label"] not in {"0", "1"}})
    if invalid_labels:
        raise ValueError(f"PAD labels must be 0/1, found: {invalid_labels}")
    invalid_splits = sorted({row["split"] for row in records if row["split"] not in ALLOWED_SPLITS})
    if invalid_splits:
        raise ValueError(f"Unsupported PAD splits: {invalid_splits}")

    subject_splits: dict[str, set[str]] = defaultdict(set)
    class_by_split: dict[str, set[str]] = defaultdict(set)
    duplicates = Counter()
    for row in records:
        if not row["subject"].strip():
            raise ValueError("Every PAD record must have a subject identifier")
        subject_splits[row["subject"]].add(row["split"])
        class_by_split[row["split"]].add(row["label"])
        duplicates[(row["path"], row["split"])] += 1
    leakage = sorted(subject for subject, splits in subject_splits.items() if len(splits) > 1)
    if leakage:
        raise ValueError(f"Subject leakage across splits; first subjects: {leakage[:10]}")
    duplicate_paths = [key for key, count in duplicates.items() if count > 1]
    if duplicate_paths:
        raise ValueError(f"Duplicate manifest rows; first entries: {duplicate_paths[:5]}")
    for split, labels in class_by_split.items():
        if labels != {"0", "1"}:
            raise ValueError(f"Split {split!r} must contain both bona-fide and attack samples")

    missing_files = 0
    if dataset_root is not None:
        root = Path(dataset_root)
        missing_files = sum(not (root / row["path"]).exists() for row in records)
        if missing_files:
            raise FileNotFoundError(f"{missing_files} manifest image(s) do not exist under {root}")
    return {
        "records": len(records),
        "subjects": len(subject_splits),
        "split_counts": dict(sorted(Counter(row["split"] for row in records).items())),
        "class_counts": {
            split: dict(
                sorted(Counter(row["label"] for row in records if row["split"] == split).items())
            )
            for split in sorted(class_by_split)
        },
        "subject_leakage": 0,
        "duplicate_rows": 0,
        "missing_files": missing_files,
    }


def _subject_from_path(image_path: str) -> str:
    parts = Path(image_path.replace("\\", "/")).parts
    lowered = [part.lower() for part in parts]
    for marker in ("train", "test"):
        if marker in lowered:
            index = lowered.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    if len(parts) >= 3:
        return parts[-3]
    raise ValueError(f"Cannot infer subject from CelebA-Spoof path: {image_path}")


def _annotation_items(path: str | Path) -> list[tuple[str, list[Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    items = []
    for image_path, labels in payload.items():
        if not isinstance(labels, list) or len(labels) < 44:
            raise ValueError(f"Invalid annotation vector for {image_path}")
        items.append((str(image_path), labels))
    return items


def infer_official_live_value(items: Iterable[tuple[str, list[Any]]]) -> int:
    """Infer the source live/spoof encoding from paths explicitly containing 'live'."""
    candidates = [
        int(labels[43])
        for path, labels in items
        if "live" in Path(path.lower().replace("\\", "/")).parts
    ]
    if not candidates:
        raise ValueError("Could not infer live label; pass --official-live-value explicitly")
    value, count = Counter(candidates).most_common(1)[0]
    if count / len(candidates) < 0.95:
        raise ValueError("Live-label inference is inconsistent with annotation paths")
    return value


def deterministic_subject_split(subject: str, validation_ratio: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{subject}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return "validation" if value < validation_ratio else "train"


def build_celeba_spoof_manifest(
    *,
    train_annotations: str | Path,
    test_annotations: str | Path,
    output_path: str | Path,
    validation_ratio: float = 0.15,
    random_seed: int = 42,
    official_live_value: int | None = None,
) -> Path:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")
    train_items = _annotation_items(train_annotations)
    test_items = _annotation_items(test_annotations)
    live_value = official_live_value
    if live_value is None:
        live_value = infer_official_live_value(train_items)

    records: list[dict[str, str]] = []
    for source_split, items in (("development", train_items), ("test", test_items)):
        for image_path, labels in items:
            subject = _subject_from_path(image_path)
            split = (
                deterministic_subject_split(subject, validation_ratio, random_seed)
                if source_split == "development"
                else "test"
            )
            source_label = int(labels[43])
            records.append(
                {
                    "path": image_path.replace("\\", "/"),
                    "label": "1" if source_label == live_value else "0",
                    "split": split,
                    "subject": subject,
                    "spoof_type": str(labels[40]),
                    "illumination": str(labels[41]),
                    "environment": str(labels[42]),
                }
            )
    validate_pad_manifest(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output
