#!/usr/bin/env python3
"""Build the aligned ST10010 development cube and run the final fault audit.

This is the single Pipeline-facing entrypoint for the two P30 modules.  Keeping
the asset build and the CIG-Bench comparison in one command prevents a caller
from accidentally evaluating a stale or differently aligned cube.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fault_p30_3d_dev_gate_st10010 as asset
import fault_p30_cigbench_compare_lift_tolerance as comparison


def run(*, device: str | None = None) -> dict[str, Any]:
    built = asset.build_dev_asset(asset.OUTPUT_ROOT)
    if built["gate_result"]["status"] != "READY":
        raise RuntimeError("ST10010 development asset did not pass its data gate")
    evaluated = comparison.run(
        output_root=comparison.OUTPUT_ROOT_V2,
        device=device,
    )
    return {
        "status": "PASS",
        "asset_manifest": built["manifest_path"],
        "comparison": evaluated["outputs"]["comparison_path"],
        "decision": evaluated["report"]["decision"]["default_recommendation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(device=args.device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
