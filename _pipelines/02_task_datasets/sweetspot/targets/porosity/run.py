"""CLI for sweet-spot target 6 PHIF and independent PHIE feasibility."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
RESERVOIR_DIR = PROJECT_ROOT / "_pipelines/02_task_datasets/reservoir"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RESERVOIR_DIR))

from p4_pipeline import POROSITY_PHIF, audit_phie_label_version, run_p4_target  # noqa: E402
from task_spec import build_phie_task_spec, build_phif_task_spec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 sweet-spot target 6 porosity runner")
    parser.add_argument("--mode", choices=("audit", "smoke", "baseline"), default="baseline")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(os.environ.get("RESERVOIR_PROCESSED_DIR", PROJECT_ROOT / "_data/processed/reservoir")),
        help="read-only directory containing reservoir train.h5 and test.h5",
    )
    parser.add_argument(
        "--guard-path",
        type=Path,
        default=Path(os.environ.get("RESERVOIR_GUARD_PATH", RESERVOIR_DIR / "_outputs/guard.npz")),
        help="read-only reservoir guard.npz",
    )
    parser.add_argument(
        "--well-log-zip",
        type=Path,
        default=Path(os.environ["VOLVE_WELL_LOG_ZIP"]) if "VOLVE_WELL_LOG_ZIP" in os.environ else None,
        help="read-only Volve_Well_logs.zip used for the independent exact-PHIE audit",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "_outputs")
    parser.add_argument("--model", choices=("reservoir_linear", "reservoir_ridge", "tiny_mlp"), default="reservoir_ridge")
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--l2-strength", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2693)
    args = parser.parse_args()
    phif = run_p4_target(
        task_spec=build_phif_task_spec(),
        definition=POROSITY_PHIF,
        processed_dir=args.processed_dir,
        guard_path=args.guard_path,
        output_dir=args.output_dir / "phif",
        mode=args.mode,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        l2_strength=args.l2_strength,
        seed=args.seed,
    )
    phie = None
    if args.well_log_zip is not None:
        phie = audit_phie_label_version(
            args.well_log_zip,
            args.output_dir / "phie",
            build_phie_task_spec(),
        )
    print(json.dumps({"PHIF": phif, "PHIE": phie}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
