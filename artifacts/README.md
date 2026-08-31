# Local model artifacts

Copy the notebook-04 checkpoint to `artifacts/pad_proxy_selected.pt`. Model binaries are intentionally ignored by Git.

The checkpoint selected by the current DS report must have SHA-256:

```text
490a9c962d6b2262895a739a9507a000850fe6771c46cb01f54d0b99cb03d492
```

Then create the immutable research bundle:

```bash
python scripts/promote_artifact.py --model-version facekyc-research-2026-08-31
python -m facekyc.console
```

The first command verifies the checkpoint hash and the locked-holdout gates. It deliberately creates a `candidate` bundle; it cannot self-approve a synthetic PAD proxy for production.
