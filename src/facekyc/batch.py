"""Bounded manifest-based inference that never writes images or embeddings."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from facekyc.config import Settings


class BatchContractError(ValueError):
    """Raised when a batch manifest violates the public inference contract."""


REQUIRED_COLUMNS = {"record_id", "id_image", "selfie_image"}


def load_manifest(path: str | Path, *, max_records: int) -> list[dict[str, str]]:
    manifest = Path(path)
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise BatchContractError(f"Manifest missing columns: {sorted(missing)}")
            rows = list(reader)
    except OSError as exc:
        raise BatchContractError(f"Cannot read manifest: {manifest}") from exc
    if not rows:
        raise BatchContractError("Manifest must contain at least one record")
    if len(rows) > max_records:
        raise BatchContractError(f"Manifest contains {len(rows)} records; limit is {max_records}")
    identifiers = [str(row.get("record_id", "")).strip() for row in rows]
    if any(not value for value in identifiers):
        raise BatchContractError("record_id cannot be empty")
    if len(set(identifiers)) != len(identifiers):
        raise BatchContractError("record_id values must be unique")
    for row in rows:
        if not str(row.get("id_image", "")).strip() or not str(row.get("selfie_image", "")).strip():
            raise BatchContractError("Image paths cannot be empty")
    return rows


def _read_image(path: Path, settings: Settings) -> Image.Image:
    try:
        if path.stat().st_size > settings.input.max_file_bytes:
            raise BatchContractError("file exceeds size limit")
        image = Image.open(path)
        width, height = image.size
        if width > settings.input.max_width or height > settings.input.max_height:
            raise BatchContractError("image dimensions exceed limit")
        image.load()
    except BatchContractError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise BatchContractError("invalid or unreadable image") from exc
    if (image.format or "").upper() not in settings.input.allowed_formats:
        raise BatchContractError("unsupported image format")
    return image.convert("RGB")


def run_batch(
    *,
    pipeline: Any,
    settings: Settings,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the shared pipeline for each row and atomically write JSONL results."""
    manifest = Path(manifest_path).resolve()
    rows = load_manifest(manifest, max_records=settings.batch.max_records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    succeeded = 0
    failed = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            record_id = str(row["record_id"]).strip()
            try:
                id_image = _read_image((manifest.parent / row["id_image"]).resolve(), settings)
                selfie_image = _read_image(
                    (manifest.parent / row["selfie_image"]).resolve(), settings
                )
                result = {"record_id": record_id, **pipeline.verify(id_image, selfie_image)}
                succeeded += 1
            except BatchContractError as exc:
                result = {
                    "record_id": record_id,
                    "status": "error",
                    "decision": "not_processed",
                    "reason_codes": ["batch_input_error"],
                    "detail": str(exc),
                }
                failed += 1
            except Exception:  # keep one runtime failure from discarding the full batch
                result = {
                    "record_id": record_id,
                    "status": "error",
                    "decision": "not_processed",
                    "reason_codes": ["batch_inference_error"],
                    "detail": "Model inference failed; inspect protected service logs.",
                }
                failed += 1
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return {
        "records": len(rows),
        "succeeded": succeeded,
        "failed": failed,
        "output": str(output),
        "privacy_note": "Output excludes image paths, images and embeddings.",
    }
