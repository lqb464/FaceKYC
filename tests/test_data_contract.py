from __future__ import annotations

import pytest

from facekyc.data import deterministic_subject_split, validate_pad_manifest


def rows():
    return [
        {"path": "train/a/live.jpg", "label": "1", "split": "train", "subject": "a"},
        {"path": "train/b/spoof.jpg", "label": "0", "split": "train", "subject": "b"},
        {"path": "validation/c/live.jpg", "label": "1", "split": "validation", "subject": "c"},
        {"path": "validation/d/spoof.jpg", "label": "0", "split": "validation", "subject": "d"},
        {"path": "test/e/live.jpg", "label": "1", "split": "test", "subject": "e"},
        {"path": "test/f/spoof.jpg", "label": "0", "split": "test", "subject": "f"},
    ]


def test_manifest_contract_accepts_subject_disjoint_splits():
    report = validate_pad_manifest(rows())
    assert report["subject_leakage"] == 0
    assert report["split_counts"] == {"test": 2, "train": 2, "validation": 2}


def test_manifest_contract_rejects_subject_leakage():
    records = rows()
    records[-1]["subject"] = "a"
    with pytest.raises(ValueError, match="Subject leakage"):
        validate_pad_manifest(records)


def test_subject_split_is_reproducible():
    assert deterministic_subject_split("person-17", 0.15, 42) == deterministic_subject_split(
        "person-17", 0.15, 42
    )
