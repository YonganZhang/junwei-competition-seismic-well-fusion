#!/usr/bin/env python3
"""Compare two completed audited runs and emit machine-readable evidence."""
from __future__ import annotations

import argparse
import json

from audit_utils import validated_run_dir, verify_historical_artifacts_if_present


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="audited_v2")
    parser.add_argument("--candidate", required=True)
    return parser.parse_args()


def read_report(run_name: str, filename: str) -> dict:
    path = validated_run_dir(run_name) / filename
    if not path.exists():
        raise FileNotFoundError(f"missing completed-run report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    historical = verify_historical_artifacts_if_present()
    reference_build = read_report(args.reference, "build_summary.json")
    candidate_build = read_report(args.candidate, "build_summary.json")
    reference_metrics = read_report(args.reference, "baseline_metrics.json")
    candidate_metrics = read_report(args.candidate, "baseline_metrics.json")
    reference_visualization = read_report(args.reference, "visualization_report.json")
    candidate_visualization = read_report(args.candidate, "visualization_report.json")

    checks = {
        "input_sha256": reference_build["input_sha256"] == candidate_build["input_sha256"],
        "dataset_sha256": reference_build["dataset_sha256"] == candidate_build["dataset_sha256"],
        "split_plan": reference_build["split_plan"] == candidate_build["split_plan"],
        "sample_manifest": (
            validated_run_dir(args.reference).joinpath("split_manifest.json").read_bytes()
            == validated_run_dir(args.candidate).joinpath("split_manifest.json").read_bytes()
        ),
        "normalization": reference_build["normalization"] == candidate_build["normalization"],
        "rasterization": reference_build["rasterization"] == candidate_build["rasterization"],
        "test_metrics": reference_metrics["test_metrics"] == candidate_metrics["test_metrics"],
        "probability_metrics": (
            reference_metrics["test_probability_metrics"]
            == candidate_metrics["test_probability_metrics"]
        ),
        "selected_threshold": reference_metrics["threshold"] == candidate_metrics["threshold"],
        "best_epoch_and_loss": (
            reference_metrics["best_epoch"],
            reference_metrics["best_val_loss"],
        )
        == (
            candidate_metrics["best_epoch"],
            candidate_metrics["best_val_loss"],
        ),
        "model_and_checkpoint_sha256": all(
            reference_metrics["artifacts_sha256"][key]
            == candidate_metrics["artifacts_sha256"][key]
            for key in ("best_checkpoint", "last_checkpoint", "baseline_model")
        ),
        "loss_curve_sha256": (
            reference_metrics["artifacts_sha256"]["loss_curve"]
            == candidate_metrics["artifacts_sha256"]["loss_curve"]
        ),
        "prediction_visualization_sha256": (
            reference_visualization["output_sha256"]
            == candidate_visualization["output_sha256"]
        ),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    report = {
        "reference": args.reference,
        "candidate": args.candidate,
        "all_checks_passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "historical_artifacts_verified": historical,
    }
    output = validated_run_dir(args.reference) / "reproducibility_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise AssertionError(f"audited runs are not reproducible: {failed}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
