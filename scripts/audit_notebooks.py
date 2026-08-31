"""Audit saved notebook execution state without executing locked holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    failures: list[str] = []
    for path in sorted(Path("notebooks").glob("[0-9][0-9]_*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            failures.append(f"{path}: expected nbformat 4")
        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", [])).strip()
            if source and not args.structure_only and cell.get("execution_count") is None:
                failures.append(f"{path}: code cell {index} has no execution_count")
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    failures.append(f"{path}: code cell {index} contains an error output")
        print(f"audited {path}")
    if failures:
        print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
