#!/usr/bin/env python3
"""Deterministic P5.2 R2 data-acquisition stop gate for fault.

This runner does not import, build, train, forward, or evaluate any model.
It consumes only frozen portable evidence from the fault R0/R1, Stage-2, and
Stage-3 contracts, then writes a zero-training acquisition summary with
observed baseline rows and planned null points for the three lanes:
synthetic_only, masked_weak_label, and formal_audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
BASELINE_COMMIT = "af8c066de0c3fc24fce024abb350b4f2c9e82d9b"
ROOT_SEED = 2693
TASK_ID = "fault_stick_segmentation"
LANE = "fault_3d_segmentation_data_gate"
MODEL_ROSTER_USED: list[str] = []
TRAINING_CELL_COUNT = 0
R3_ALLOWED = False

RESULTS_FILENAME = "p5_r2_results.jsonl"
SUMMARY_FILENAME = "p5_r2_summary.json"
CONTRACT_FILENAME = "p5_r2_data_acquisition_contract.json"
ARTIFACT_MANIFEST_FILENAME = "p5_r2_artifact_manifest.json"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_r2_data_acquisition"

R0_SUMMARY = TRACK_DIR / "_outputs" / "p5_r01_protocol" / "summary.json"
R0_LANES = TRACK_DIR / "_outputs" / "p5_r01_protocol" / "r0_lane_gates.json"
STAGE2_SUMMARY = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_summary.json"
STAGE3_SUMMARY = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_summary.json"
STAGE3_DATA_MANIFEST = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_data_manifest.json"
STAGE3_VISUALIZATION_MANIFEST = (
    TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_visualization_manifest.json"
)
STAGE3_ARTIFACT_MANIFEST = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_artifact_manifest.json"
STAGE3_READINESS_FIGURES = (
    "figures/fault_readiness.svg",
    "figures/fault_negative_coverage.svg",
    "figures/fault_unknown_coverage.svg",
)
SOURCE_LOCKS = PROJECT_ROOT / "_models" / "fault" / "p5_source_locks.json"
P4_BUFFERED_CV_PLAN = TRACK_DIR / "_outputs" / "p4_preflight" / "buffered_cv_plan.json"

LANES = ("synthetic_only", "masked_weak_label", "formal_audited")


class FaultP5R2Error(RuntimeError):
    """Frozen evidence is inconsistent, missing, or unsafe."""


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FaultP5R2Error(f"required frozen evidence is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FaultP5R2Error(f"frozen evidence is not a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FaultP5R2Error(message)


def _planned_null_points(lane_id: str) -> list[dict[str, Any]]:
    if lane_id == "synthetic_only":
        contracts = (
            "registered_dense_synthetic_dataset",
            "independent_generated_volume_split",
            "fixture_only_contract_cannot_unlock",
        )
    elif lane_id == "masked_weak_label":
        contracts = (
            "weak_objective_preregistered",
            "independent_spatial_dev_split",
            "audited_or_approved_weak_evaluation",
        )
    else:
        contracts = (
            "coverage_manifest",
            "verified_negative_mask",
            "buffered_development_folds",
        )
    return [
        {
            "contract_id": contract_id,
            "metric": None,
            "observed": False,
            "value": None,
        }
        for contract_id in contracts
    ]


def _lane_contracts() -> dict[str, list[dict[str, Any]]]:
    return {
        "synthetic_only": [
            {
                "contract_id": "registered_dense_synthetic_dataset",
                "minimum": (
                    "registered dense synthetic dataset with source/label/split hashes; "
                    "contract fixture cannot unlock the lane"
                ),
            },
            {
                "contract_id": "independent_generated_volume_split",
                "minimum": (
                    "independent generated training/validation volume IDs with disjoint split hashes"
                ),
            },
            {
                "contract_id": "dense_ground_truth_and_both_classes",
                "minimum": (
                    "dense ground truth must be registered and both classes must be present "
                    "before any trainable cell can exist"
                ),
            },
        ],
        "masked_weak_label": [
            {
                "contract_id": "weak_objective_preregistered",
                "minimum": (
                    "masked weak / PU / SSL objective must be explicitly preregistered and tested "
                    "before any training is allowed"
                ),
            },
            {
                "contract_id": "independent_spatial_dev_split",
                "minimum": (
                    "independent spatial development split with fold-train-only preprocessing, "
                    "class prior, and consistency fit"
                ),
            },
            {
                "contract_id": "audited_or_approved_weak_evaluation",
                "minimum": (
                    "approved unbiased weak evaluation or audited negatives are required before ranking"
                ),
            },
        ],
        "formal_audited": [
            {
                "contract_id": "coverage_manifest",
                "minimum": (
                    "coordinate-aligned per-volume/per-block/per-slice annotation coverage manifest "
                    "with source SHA-256 and annotation_status"
                ),
            },
            {
                "contract_id": "verified_negative_mask",
                "minimum": (
                    "non-empty coverage-audited verified_negative_mask disjoint from fault positives; "
                    "count and provenance reported for every development block"
                ),
            },
            {
                "contract_id": "unknown_and_proxy_masks",
                "minimum": (
                    "explicit unknown_mask equals complement of valid_label_mask; proxy_mask remains "
                    "separate and never enables formal loss or metrics"
                ),
            },
            {
                "contract_id": "buffered_development_folds",
                "minimum": (
                    "P4-regenerated buffered spatial plan with at least two independent valid folds; "
                    "each train/validation side contains positives and audited negatives"
                ),
            },
            {
                "contract_id": "fold_train_fit_provenance",
                "minimum": (
                    "preprocessing, class weights, target transform, and calibration each carry a "
                    "fold-train-only fit hash before any Stage-3 cell can run"
                ),
            },
            {
                "contract_id": "test_firewall",
                "minimum": (
                    "development coordinates and artifacts exclude frozen/regression-test material; "
                    "test paths, labels, predictions, and metrics remain unavailable to selection"
                ),
            },
        ],
    }


def _planned_points_contract() -> dict[str, list[dict[str, Any]]]:
    return {
        lane_id: _planned_null_points(lane_id)
        for lane_id in LANES
    }


def _validate_frozen_evidence() -> dict[str, Any]:
    r0_summary = _read_json(R0_SUMMARY)
    r0_lanes = _read_json(R0_LANES)
    stage2_summary = _read_json(STAGE2_SUMMARY)
    stage3_summary = _read_json(STAGE3_SUMMARY)
    stage3_data = _read_json(STAGE3_DATA_MANIFEST)
    stage3_visualization = _read_json(STAGE3_VISUALIZATION_MANIFEST)
    stage3_artifacts = _read_json(STAGE3_ARTIFACT_MANIFEST)
    source_locks = _read_json(SOURCE_LOCKS)
    buffered_cv_plan = _read_json(P4_BUFFERED_CV_PLAN)

    _require(r0_summary.get("protocol") == "fault-p5-r01-summary-v1", "R0 summary protocol changed")
    _require(r0_summary.get("root_seed") == ROOT_SEED, "R0 root seed is not frozen")
    _require(r0_summary.get("status") == "completed_with_formal_lane_blocked", "R0 summary status changed")
    _require(r0_summary.get("ranking_status") == "not_rankable", "R0 summary became rankable")
    _require(tuple(lane["lane_id"] for lane in r0_lanes.get("lanes", ())) == LANES, "R0 lane order changed")
    _require(r0_lanes.get("lane_isolation", {}).get("cross_lane_ranking_forbidden") is True, "lane isolation changed")

    _require(stage2_summary.get("protocol") == "fault-p5-stage2-summary-v1", "Stage-2 protocol changed")
    _require(stage2_summary.get("status") == "not_rankable", "Stage-2 summary became rankable")
    _require(stage2_summary.get("counts", {}).get("blocked") == 10, "Stage-2 blocked count changed")
    _require(stage2_summary.get("budget_consumed", {}).get("parameter_updates") == 0, "Stage-2 consumed updates")
    _require(stage2_summary.get("test_firewall", {}).get("frozen_test_accessed") is False, "Stage-2 opened test")

    _require(stage3_summary.get("protocol") == "fault-p5-stage3-summary-v1", "Stage-3 protocol changed")
    _require(stage3_summary.get("protocol_baseline_commit") == "16bebd18a0bc722afcbc4b841610bf76ce9503e4", "Stage-3 baseline changed")
    _require(stage3_summary.get("status") == "not_rankable", "Stage-3 summary became rankable")
    _require(stage3_summary.get("reason_code") == "NO_VALID_FAULT_DEVELOPMENT_FOLDS", "Stage-3 reason changed")
    _require(stage3_summary.get("effective_fold_count") == 0, "Stage-3 fold count changed")
    _require(stage3_summary.get("expected_training_cells") == 0, "Stage-3 invented training cells")
    _require(stage3_summary.get("results_artifact", {}).get("training_cell_count") == 0, "Stage-3 results claim training cells")
    _require(stage3_summary.get("test_firewall", {}).get("frozen_test_accessed") is False, "Stage-3 opened test")

    _require(stage3_data.get("status") == "blocked", "Stage-3 data gate is not blocked")
    _require(stage3_data.get("ranking_status") == "not_rankable", "Stage-3 data gate became rankable")
    _require(stage3_data.get("test_firewall", {}).get("runner_accepts_test_inputs") is False, "Stage-3 firewall opened")
    _require(stage3_visualization.get("status") == "readiness_only", "Stage-3 visualization changed")
    _require(stage3_visualization.get("frozen_test_accessed") is False, "Stage-3 visualization opened test")

    _require(stage3_artifacts.get("checkpoint_count") == 0, "Stage-3 has checkpoints")
    _require(stage3_artifacts.get("prediction_payload_count") == 0, "Stage-3 has predictions")
    _require(stage3_artifacts.get("frozen_test_accessed") is False, "Stage-3 artifacts opened test")

    _require(source_locks.get("models") and len(source_locks["models"]) == 10, "source locks changed")
    _require(buffered_cv_plan.get("status") == "not_feasible", "buffered CV plan is no longer not_feasible")

    source_hashes = {
        "r0_summary": _sha256(R0_SUMMARY),
        "r0_lane_gates": _sha256(R0_LANES),
        "stage2_summary": _sha256(STAGE2_SUMMARY),
        "stage3_summary": _sha256(STAGE3_SUMMARY),
        "stage3_data_manifest": _sha256(STAGE3_DATA_MANIFEST),
        "stage3_visualization_manifest": _sha256(STAGE3_VISUALIZATION_MANIFEST),
        "stage3_artifact_manifest": _sha256(STAGE3_ARTIFACT_MANIFEST),
        "source_locks": _sha256(SOURCE_LOCKS),
        "buffered_cv_plan": _sha256(P4_BUFFERED_CV_PLAN),
    }

    _require(stage2_summary.get("source_hashes", {}).get("p5_stage1/summary.json") is not None, "Stage-2 source hashes missing")
    _require(stage3_summary.get("source_hashes", {}).get("p5_stage2/p5_stage2_summary.json") == source_hashes["stage2_summary"], "Stage-3 source hash link broke")
    _require(stage3_summary.get("data_manifest", {}).get("sha256") == source_hashes["stage3_data_manifest"], "Stage-3 data hash link broke")
    _require(stage3_summary.get("visualization_manifest", {}).get("sha256") == source_hashes["stage3_visualization_manifest"], "Stage-3 visualization hash link broke")

    for record in stage3_artifacts.get("artifacts", ()):
        _require(isinstance(record, dict), "Stage-3 artifact record is malformed")
        relative = str(record.get("path", ""))
        path = TRACK_DIR / "_outputs" / "p5_stage3" / relative
        _require(path.is_file(), f"Stage-3 artifact missing: {relative}")
        _require(record.get("sha256") == _sha256(path), f"Stage-3 artifact hash mismatch: {relative}")
        _require(record.get("bytes") == path.stat().st_size, f"Stage-3 artifact size mismatch: {relative}")

    return {
        "r0_summary": r0_summary,
        "r0_lanes": r0_lanes,
        "stage2_summary": stage2_summary,
        "stage3_summary": stage3_summary,
        "stage3_data": stage3_data,
        "stage3_visualization": stage3_visualization,
        "stage3_artifacts": stage3_artifacts,
        "source_locks": source_locks,
        "buffered_cv_plan": buffered_cv_plan,
        "source_hashes": source_hashes,
    }


def _lane_statuses(r0_lanes: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_lane = {str(item["lane_id"]): dict(item) for item in r0_lanes.get("lanes", ())}
    return {lane_id: by_lane[lane_id] for lane_id in LANES}


def _planned_result_rows(frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    lane_rows = _lane_statuses(frozen["r0_lanes"])
    stage2_summary = frozen["stage2_summary"]
    source_hashes = frozen["source_hashes"]
    lane_contracts = _lane_contracts()
    planned_points = _planned_points_contract()

    rows: list[dict[str, Any]] = []
    for lane_id in LANES:
        lane = lane_rows[lane_id]
        contract = lane_contracts[lane_id]
        point_plan = planned_points[lane_id]
        status = "data_ready_but_not_trainable" if lane_id == "masked_weak_label" else "blocked"
        observed = {
            "observed": True,
            "status": status,
            "data_ready": bool(lane.get("data_ready")),
            "train_allowed": bool(lane.get("train_allowed")),
            "rank_allowed": bool(lane.get("rank_allowed")),
            "reason_codes": list(lane.get("reason_codes", ())),
            "metric": None,
            "value": None,
        }
        split_contract = {
            "lane_id": lane_id,
            "status": status,
            "train_allowed": bool(lane.get("train_allowed")),
            "rank_allowed": bool(lane.get("rank_allowed")),
            "data_ready": bool(lane.get("data_ready")),
            "reason_codes": list(lane.get("reason_codes", ())),
            "current_fold_count": 0,
            "current_training_cells": 0,
        }
        label_contract = {
            "lane_id": lane_id,
            "requirements": lane.get("requirements", {}),
            "reason_codes": list(lane.get("reason_codes", ())),
        }
        config_contract = {
            "lane_id": lane_id,
            "root_seed": ROOT_SEED,
            "model_roster_used": MODEL_ROSTER_USED,
            "planned_null_points": point_plan,
            "no_model_import_or_build": True,
        }
        source_contract = {
            "lane_id": lane_id,
            "frozen_sources": source_hashes,
            "r0_summary_sha256": _sha256(R0_SUMMARY),
            "stage2_summary_sha256": _sha256(STAGE2_SUMMARY),
            "stage3_summary_sha256": _sha256(STAGE3_SUMMARY),
            "source_locks_sha256": _sha256(SOURCE_LOCKS),
        }
        rows.append(
            {
                "protocol": "fault-p5-r2-acquisition-row-v1",
                "track_id": "fault",
                "task_id": TASK_ID,
                "lane_id": lane_id,
                "seed": ROOT_SEED,
                "status": status,
                "ranking_status": "not_rankable",
                "data_ready": bool(lane.get("data_ready")),
                "train_allowed": False,
                "rank_allowed": False,
                "reason_codes": list(lane.get("reason_codes", ())),
                "model_roster_used": MODEL_ROSTER_USED,
                "training_cell_count": TRAINING_CELL_COUNT,
                "observed_acquisition_baseline": observed,
                "planned_null_points": point_plan,
                "minimum_unblock_contract": contract,
                "hashes": {
                    "source": _sha256_payload(source_contract),
                    "label": _sha256_payload(label_contract),
                    "split": _sha256_payload(split_contract),
                    "config": _sha256_payload(config_contract),
                },
                "test_firewall": {
                    "runner_accepts_test_inputs": False,
                    "frozen_test_accessed": False,
                    "historical_holdout_arrays_accessed": False,
                    "test_labels_accessed": False,
                    "test_predictions_accessed": False,
                    "test_metrics_accessed": False,
                    "selection_or_ranking_performed": False,
                },
                "stage2_reuse": {
                    "candidate_count": stage2_summary.get("candidate_count"),
                    "results_sha256": stage2_summary.get("results_artifact", {}).get("sha256"),
                    "status": stage2_summary.get("status"),
                },
            }
        )
    return rows


def run_r2_acquisition(output_dir: Path) -> dict[str, Any]:
    frozen = _validate_frozen_evidence()
    rows = _planned_result_rows(frozen)
    lane_contracts = _lane_contracts()
    planned_points = _planned_points_contract()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / RESULTS_FILENAME
    contract_path = output_dir / CONTRACT_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    manifest_path = output_dir / ARTIFACT_MANIFEST_FILENAME

    _write_jsonl(results_path, rows)

    contract = {
        "protocol": "fault-p5-r2-data-acquisition-contract-v1",
        "baseline_commit": BASELINE_COMMIT,
        "root_seed": ROOT_SEED,
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "model_roster_used": MODEL_ROSTER_USED,
        "training_cell_count": TRAINING_CELL_COUNT,
        "official_learning_curve_generated": False,
        "winner": None,
        "R3_allowed": R3_ALLOWED,
        "current_trainable_lanes": [],
        "observed_acquisition_baselines": [
            {
                "lane_id": row["lane_id"],
                "observed": row["observed_acquisition_baseline"]["observed"],
                "status": row["observed_acquisition_baseline"]["status"],
                "data_ready": row["data_ready"],
                "train_allowed": row["train_allowed"],
                "rank_allowed": row["rank_allowed"],
                "reason_codes": row["reason_codes"],
            }
            for row in rows
        ],
        "planned_null_points": [
            {"lane_id": lane_id, "points": planned_points[lane_id]}
            for lane_id in LANES
        ],
        "minimum_unblock_contract": [
            {"lane_id": lane_id, **item}
            for lane_id, items in lane_contracts.items()
            for item in items
        ],
        "lane_reason_codes": {row["lane_id"]: row["reason_codes"] for row in rows},
        "source_hashes": frozen["source_hashes"],
        "stage3_visualization_reuse": {
            "path": "p5_stage3/p5_stage3_visualization_manifest.json",
            "sha256": frozen["source_hashes"]["stage3_visualization_manifest"],
            "figures": [
                {
                    "path": f"p5_stage3/{figure}",
                    "sha256": _sha256(TRACK_DIR / "_outputs" / "p5_stage3" / figure),
                }
                for figure in STAGE3_READINESS_FIGURES
            ],
        },
        "test_firewall": {
            "runner_accepts_test_inputs": False,
            "frozen_test_accessed": False,
            "historical_holdout_arrays_accessed": False,
            "test_labels_accessed": False,
            "test_predictions_accessed": False,
            "test_metrics_accessed": False,
            "selection_or_ranking_performed": False,
        },
    }
    _write_json(contract_path, contract)

    summary = {
        "protocol": "fault-p5-r2-summary-v1",
        "baseline_commit": BASELINE_COMMIT,
        "root_seed": ROOT_SEED,
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "reason_codes": {
            "synthetic_only": list(rows[0]["reason_codes"]),
            "masked_weak_label": list(rows[1]["reason_codes"]),
            "formal_audited": list(rows[2]["reason_codes"]),
        },
        "model_roster_used": MODEL_ROSTER_USED,
        "training_cell_count": TRAINING_CELL_COUNT,
        "metrics": None,
        "current_trainable_lanes": [],
        "data_ready_lanes": [row["lane_id"] for row in rows if row["data_ready"]],
        "blocked_lanes": [row["lane_id"] for row in rows if row["status"] == "blocked"],
        "official_learning_curve_generated": False,
        "learning_curve_points": [],
        "winner": None,
        "R3_allowed": R3_ALLOWED,
        "observed_acquisition_baseline_count": len(rows),
        "planned_null_point_count": sum(len(row["planned_null_points"]) for row in rows),
        "minimum_unblock_contract_count": len(contract["minimum_unblock_contract"]),
        "results_artifact": {
            "path": RESULTS_FILENAME,
            "sha256": _sha256(results_path),
            "line_count": len(rows),
        },
        "data_acquisition_contract": {
            "path": CONTRACT_FILENAME,
            "sha256": _sha256(contract_path),
        },
        "test_firewall": contract["test_firewall"],
        "source_hashes": frozen["source_hashes"],
        "stage3_visualization_reuse": contract["stage3_visualization_reuse"],
    }
    _write_json(summary_path, summary)

    artifact_manifest = {
        "protocol": "fault-p5-r2-artifact-manifest-v1",
        "baseline_commit": BASELINE_COMMIT,
        "root_seed": ROOT_SEED,
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "blocked",
        "ranking_status": "not_rankable",
        "training_cell_count": TRAINING_CELL_COUNT,
        "current_trainable_lanes": [],
        "artifacts": [
            {
                "path": RESULTS_FILENAME,
                "role": "observed_acquisition_rows",
                "sha256": _sha256(results_path),
                "bytes": results_path.stat().st_size,
            },
            {
                "path": CONTRACT_FILENAME,
                "role": "data_acquisition_contract",
                "sha256": _sha256(contract_path),
                "bytes": contract_path.stat().st_size,
            },
            {
                "path": SUMMARY_FILENAME,
                "role": "summary",
                "sha256": _sha256(summary_path),
                "bytes": summary_path.stat().st_size,
            },
        ],
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "test_firewall": contract["test_firewall"],
        "large_artifacts_committed": False,
        "checkpoint_count": 0,
        "prediction_payload_count": 0,
        "refit_executed": False,
        "holdout_accessed": False,
        "model_roster_used": MODEL_ROSTER_USED,
        "official_learning_curve_generated": False,
        "winner": None,
    }
    _write_json(manifest_path, artifact_manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = run_r2_acquisition(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
