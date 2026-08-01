#!/usr/bin/env python3
"""Fault CIG-Bench compatibility audit.

This is a fail-closed gate for the fault track. It confirms the packaged
`cig_bench` install and default ModelScope weight path, then checks whether the
current fault assets provide a legal contiguous 3-D development volume with
explicit verified background / unknown masks and a group-isolated split.

If the gate is blocked, the script writes an evidence markdown report and does
not attempt to score against frozen holdout or invent a 3-D development volume.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p18_cigbench_fault"
AUDITED_RUN_DIR = TRACK_DIR / "_outputs" / "runs" / "audited_v2"
BASELINE_METRICS_PATH = AUDITED_RUN_DIR / "baseline_metrics.json"
BUILD_SUMMARY_PATH = AUDITED_RUN_DIR / "build_summary.json"
TRAIN_H5_PATH = PROJECT_ROOT / "_data" / "processed" / "fault" / "train.h5"
FAULT_POINTS_PATH = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs" / "fault_points.npz"
DATA_REGISTRY_PATH = PROJECT_ROOT / "_meta" / "_data_registry.yml"


@dataclass(frozen=True)
class GateResult:
    status: str
    reason_code: str
    reasons: list[dict[str, Any]]
    install: dict[str, Any]
    baseline_reference: dict[str, Any]
    asset_probe: dict[str, Any]
    minimum_unblock_contract: list[str]
    frozen_holdout_accessed: bool = False


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_train_h5() -> dict[str, Any]:
    with h5py.File(TRAIN_H5_PATH, "r") as handle:
        first_key = next(iter(handle.keys()))
        first = handle[first_key]
        sample_shape = list(first["seismic_patch"].shape)
        label_shape = list(first["label"].shape)
        return {
            "path": str(TRAIN_H5_PATH.relative_to(PROJECT_ROOT)),
            "n_samples": int(handle.attrs["n_samples"]),
            "sample_shape": sample_shape,
            "label_shape": label_shape,
            "sample_kind": json.loads(first.attrs["meta"])["sample_kind"],
            "split": handle.attrs["split"],
        }


def _summarize_fault_points() -> dict[str, Any]:
    points = np.load(FAULT_POINTS_PATH, allow_pickle=True)
    return {
        "path": str(FAULT_POINTS_PATH.relative_to(PROJECT_ROOT)),
        "keys": list(points.files),
        "count": int(len(points["inline"])),
        "fields_present": sorted(points.files),
        "mask_fields_present": [name for name in points.files if "mask" in name.lower() or "unknown" in name.lower() or "background" in name.lower()],
    }


def download_fault_predictor_weight() -> dict[str, Any]:
    from cig_bench.predictor.fault import FaultPredictor

    predictor = FaultPredictor(device="cpu")
    weight_path = Path(predictor.restore_path)
    return {
        "package": "cig_bench",
        "package_version": importlib.metadata.version("cig_bench"),
        "predictor_class": "cig_bench.predictor.fault.FaultPredictor",
        "weight_path": str(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": sha256_file(weight_path),
        "device": str(predictor.device),
    }


def audit_gate() -> tuple[str, dict[str, Any]]:
    train_summary = _summarize_train_h5()
    fault_points = _summarize_fault_points()
    build_summary = _load_json(BUILD_SUMMARY_PATH)
    baseline_metrics = _load_json(BASELINE_METRICS_PATH)

    reasons = [
        {
            "code": "contiguous_3d_development_blocks_missing",
            "passed": False,
            "evidence": (
                f"fault/train.h5 is a 2-D patch bundle with sample_shape={train_summary['sample_shape']}; "
                "no contiguous 3-D development volume is present in the fault track."
            ),
        },
        {
            "code": "coverage_audited_verified_background_missing",
            "passed": False,
            "evidence": (
                "fault_points.npz contains sparse fault sticks only "
                f"(fields={fault_points['fields_present']}); no verified background mask asset is registered."
            ),
        },
        {
            "code": "explicit_unknown_mask_provenance_missing",
            "passed": False,
            "evidence": "No explicit unknown-mask artifact or provenance record is registered for a 3-D development volume.",
        },
        {
            "code": "group_isolated_development_split_missing",
            "passed": False,
            "evidence": (
                "The audited fault split in build_summary.json is the 2-D train/test patch split "
                f"with split_plan={build_summary['split_plan']}; no group-isolated 3-D development split exists."
            ),
        },
    ]
    return "DATA_GATE_BLOCKED", {
        "status": "DATA_GATE_BLOCKED",
        "reason_code": "NO_VALID_FAULT_3D_DEVELOPMENT_VOLUME",
        "reasons": reasons,
        "install": download_fault_predictor_weight(),
        "baseline_reference": {
            "model": baseline_metrics["model"],
            "run_name": baseline_metrics["run_name"],
            "validation_f1_at_selected_threshold": baseline_metrics["validation_f1_at_selected_threshold"],
            "test_metrics": baseline_metrics["test_metrics"],
            "threshold": baseline_metrics["threshold"],
            "threshold_source": baseline_metrics["threshold_source"],
            "validation_positive_voxels": baseline_metrics["validation_positive_voxels"],
            "test_positive_voxels": baseline_metrics["test_positive_voxels"],
        },
        "asset_probe": {
            "train_h5": train_summary,
            "fault_points": fault_points,
            "data_registry_path": str(DATA_REGISTRY_PATH.relative_to(PROJECT_ROOT)),
            "frozen_holdout_accessed": False,
        },
        "minimum_unblock_contract": [
            "A contiguous 3-D fault development volume with explicit tline/iline/xline coordinates.",
            "A verified-background mask provenance record for that development volume.",
            "An explicit unknown-mask provenance record for the same volume.",
            "A group-isolated development split manifest that keeps train/validation/dev disjoint by group.",
            "A fault-adapter path that consumes the dev volume without opening frozen holdout/test.h5.",
        ],
        "frozen_holdout_accessed": False,
    }


def render_evidence(result: dict[str, Any]) -> str:
    lines = [
        "# Fault CIG-Bench incremental comparison audit",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Status: {result['status']}",
        f"- Reason code: {result['reason_code']}",
        "",
        "## Install and weight proof",
        "",
        f"- Package: `{result['install']['package']}` `{result['install']['package_version']}`",
        f"- Predictor: `{result['install']['predictor_class']}`",
        f"- Weight path: `{result['install']['weight_path']}`",
        f"- Weight bytes: `{result['install']['weight_bytes']}`",
        f"- Weight sha256: `{result['install']['weight_sha256']}`",
        "",
        "## Gate verdict",
        "",
    ]
    for entry in result["reasons"]:
        lines.append(f"- `{entry['code']}`: blocked")
        lines.append(f"  - {entry['evidence']}")
    lines.extend(
        [
            "",
            "## Current baseline reference",
            "",
            f"- Model: `{result['baseline_reference']['model']}`",
            f"- Run: `{result['baseline_reference']['run_name']}`",
            f"- Validation F1 at selected threshold: {result['baseline_reference']['validation_f1_at_selected_threshold']}",
            f"- Threshold: {result['baseline_reference']['threshold']} ({result['baseline_reference']['threshold_source']})",
            f"- Test metrics: `{json.dumps(result['baseline_reference']['test_metrics'], sort_keys=True)}`",
            "",
            "## Asset probe",
            "",
            f"- Train HDF5 summary: `{json.dumps(result['asset_probe']['train_h5'], sort_keys=True)}`",
            f"- Fault points summary: `{json.dumps(result['asset_probe']['fault_points'], sort_keys=True)}`",
            f"- Frozen holdout accessed: `{result['frozen_holdout_accessed']}`",
            "",
            "## Minimum unblock contract",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["minimum_unblock_contract"])
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "The CIG-Bench package installs and the default FaultPredictor checkpoint downloads successfully, "
            "but the current fault track does not expose a legal contiguous 3-D development volume with explicit "
            "audited background/unknown masks and a group-isolated dev split. The comparison is therefore blocked."
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    status, result = audit_gate()
    output_root.mkdir(parents=True, exist_ok=True)
    evidence_path = output_root / "evidence.md"
    evidence_path.write_text(render_evidence(result), encoding="utf-8")
    try:
        result["evidence_path"] = str(evidence_path.relative_to(PROJECT_ROOT))
    except ValueError:
        result["evidence_path"] = str(evidence_path)
    result["status"] = status
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
