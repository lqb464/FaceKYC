"""Summarize privacy-safe JSONL inference records and apply operational gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from facekyc.config import load_settings
from facekyc.monitoring import assess_monitoring_summary, summarize_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--config")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    settings = load_settings(args.config)
    summary = summarize_results(records)
    assessment = assess_monitoring_summary(
        summary,
        minimum_batch_size=settings.monitoring.minimum_batch_size,
        warning_rate_limit=settings.monitoring.warning_rate_limit,
    )
    report = {"summary": summary, "assessment": assessment}
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
