#!/usr/bin/env python3
"""Build real Volve data and emit only project-relative report paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline import build_real_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-step-m", type=float, default=2.0)
    parser.add_argument("--sequence-step-m", type=float, default=0.5)
    args = parser.parse_args()
    if args.depth_step_m <= 0 or args.sequence_step_m <= 0:
        parser.error("depth/sequence step must be positive")
    report = build_real_dataset(args.depth_step_m, args.sequence_step_m)
    nonportable = {
        name: path
        for name, path in report.get("paths", {}).items()
        if Path(path).is_absolute() or ".claude/worktrees" in path
    }
    if nonportable:
        raise RuntimeError(f"构建报告含不可移植路径: {nonportable}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
