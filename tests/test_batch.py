from __future__ import annotations

import json

import pytest
from conftest import make_pipeline

from facekyc.batch import BatchContractError, load_manifest, run_batch


def test_batch_uses_shared_pipeline_and_excludes_image_paths(tmp_path, settings, valid_image):
    valid_image.save(tmp_path / "id.png")
    valid_image.save(tmp_path / "selfie.png")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "record_id,id_image,selfie_image\ncase-1,id.png,selfie.png\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.jsonl"
    summary = run_batch(
        pipeline=make_pipeline(settings),
        settings=settings,
        manifest_path=manifest,
        output_path=output,
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert summary["succeeded"] == 1
    assert record["record_id"] == "case-1"
    assert record["decision"] == "verified"
    assert "id_image" not in record
    assert "selfie_image" not in record


def test_batch_manifest_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "record_id,id_image,selfie_image\ncase-1,a.png,b.png\ncase-1,c.png,d.png\n",
        encoding="utf-8",
    )
    with pytest.raises(BatchContractError, match="unique"):
        load_manifest(manifest, max_records=10)
