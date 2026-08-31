"""Create a checksummed research-candidate bundle from notebook 05 evidence."""

from __future__ import annotations

import argparse
import json

from facekyc.promotion import promote_notebook_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/ds/05_locked_holdout.json")
    parser.add_argument("--weights", default="artifacts/pad_proxy_selected.pt")
    parser.add_argument("--output", default="artifacts/facekyc_bundle.json")
    parser.add_argument("--model-version", required=True)
    args = parser.parse_args()
    bundle = promote_notebook_report(
        report_path=args.report,
        weights_path=args.weights,
        output_path=args.output,
        model_version=args.model_version,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "deployment_status": bundle["deployment_status"],
                "bundle_checksum": bundle["bundle_checksum"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
