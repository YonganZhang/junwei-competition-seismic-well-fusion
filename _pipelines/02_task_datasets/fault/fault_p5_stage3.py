#!/usr/bin/env python3
"""Fault Stage-3 data-readiness gate for the frozen zero-fold protocol.

There are no scientifically legal training cells for this track.  The runner
reuses the accepted Stage-2 gate, candidate source lock, budget, and P4 split
evidence; writes one non-cell blocked record; quantifies the available label
and spatial coverage; and builds readiness-only figures.  It never accepts a
dataset, model, fold override, prediction, checkpoint, or test argument.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from fault_p5_stage2 import (
    BLIND_AUDIT,
    CV_PLAN,
    FIRST_TEN_MODEL_IDS,
    FIXED_BUDGET,
    SOURCE_LOCK,
    STAGE1_SUMMARY,
    _read_json,
    _sha256,
    _sha256_payload,
    _validate_frozen_inputs,
    _validate_source_hash_links,
    audit_data_gate,
)
from fault_p5_stage3_visualize import build_figures


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
STAGE2_RUNNER = TRACK_DIR / "fault_p5_stage2.py"
STAGE3_VISUALIZER = TRACK_DIR / "fault_p5_stage3_visualize.py"
STAGE2_RESULTS = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_results.jsonl"
STAGE2_SUMMARY = TRACK_DIR / "_outputs" / "p5_stage2" / "p5_stage2_summary.json"
STAGE3_PROTOCOL = PROJECT_ROOT / "_wiki-methodology" / "_top" / "_phases" / "P5_stage3_multiseed_cv.md"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_stage3"

BASELINE_COMMIT = "16bebd18a0bc722afcbc4b841610bf76ce9503e4"
ROOT_SEED = 2693
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
TASK_ID = "fault_stick_segmentation"
LANE = "fault_3d_segmentation_data_gate"
FROZEN_TOP_MODELS: tuple[str, ...] = ()
FROZEN_EFFECTIVE_FOLDS = 0
RESULTS_FILENAME = "p5_stage3_results.jsonl"
SUMMARY_FILENAME = "p5_stage3_summary.json"
DATA_MANIFEST_FILENAME = "p5_stage3_data_manifest.json"
OOF_MANIFEST_FILENAME = "p5_stage3_oof_manifest.json"
VISUALIZATION_MANIFEST_FILENAME = "p5_stage3_visualization_manifest.json"
ARTIFACT_MANIFEST_FILENAME = "p5_stage3_artifact_manifest.json"
LEADERBOARD_PATH = Path("leaderboards") / f"{TASK_ID}__{LANE}.json"


class FaultStage3EvidenceInvalid(RuntimeError):
    """Frozen Stage-1/2/P4 evidence is inconsistent or unsafe."""


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FaultStage3EvidenceInvalid("accepted Stage-2 result rows are missing")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise FaultStage3EvidenceInvalid(f"Stage-2 result line {line_number} is not an object")
        records.append(payload)
    return records


def _source_hashes() -> dict[str, str]:
    paths = {
        "fault_p5_stage3.py": Path(__file__).resolve(),
        "fault_p5_stage3_visualize.py": STAGE3_VISUALIZER,
        "fault_p5_stage2.py": STAGE2_RUNNER,
        "p5_stage2/p5_stage2_results.jsonl": STAGE2_RESULTS,
        "p5_stage2/p5_stage2_summary.json": STAGE2_SUMMARY,
        "p5_stage1/summary.json": STAGE1_SUMMARY,
        "p5_source_locks.json": SOURCE_LOCK,
        "p4_preflight/buffered_cv_plan.json": CV_PLAN,
        "p4_preflight/blind_test_not_feasible.json": BLIND_AUDIT,
        "P5_stage3_multiseed_cv.md": STAGE3_PROTOCOL,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FaultStage3EvidenceInvalid("required frozen source is missing: " + ", ".join(missing))
    return {name: _sha256(path) for name, path in paths.items()}


def _validate_stage2_reuse(
    source_lock: Mapping[str, Any],
    stage1: Mapping[str, Any],
    cv_plan: Mapping[str, Any],
    blind_audit: Mapping[str, Any],
    stage2: Mapping[str, Any],
    stage2_rows: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
) -> None:
    _validate_frozen_inputs(source_lock, stage1, cv_plan, blind_audit)
    stage1_links = {
        "p5_source_locks.json": source_hashes["p5_source_locks.json"],
        "p4_preflight/buffered_cv_plan.json": source_hashes[
            "p4_preflight/buffered_cv_plan.json"
        ],
        "p4_preflight/blind_test_not_feasible.json": source_hashes[
            "p4_preflight/blind_test_not_feasible.json"
        ],
    }
    _validate_source_hash_links(stage1, stage1_links)
    if (
        stage2.get("protocol") != "fault-p5-stage2-summary-v1"
        or stage2.get("track_id") != "fault"
        or stage2.get("task_id") != TASK_ID
        or stage2.get("lane") != LANE
        or tuple(stage2.get("model_ids", ())) != FIRST_TEN_MODEL_IDS
        or stage2.get("fixed_budget") != FIXED_BUDGET
        or stage2.get("status") != "not_rankable"
        or stage2.get("counts", {}).get("blocked") != 10
        or stage2.get("budget_consumed", {}).get("parameter_updates") != 0
        or stage2.get("test_firewall", {}).get("frozen_test_accessed") is not False
    ):
        raise FaultStage3EvidenceInvalid("Stage-2 summary is not the accepted zero-training fault gate")
    result_evidence = stage2.get("results_artifact", {})
    if (
        result_evidence.get("sha256") != source_hashes["p5_stage2/p5_stage2_results.jsonl"]
        or result_evidence.get("line_count") != 10
        or len(stage2_rows) != 10
    ):
        raise FaultStage3EvidenceInvalid("Stage-2 result hash/count no longer matches its summary")
    for expected_model, row in zip(FIRST_TEN_MODEL_IDS, stage2_rows):
        if (
            row.get("model_id") != expected_model
            or row.get("task_id") != TASK_ID
            or row.get("lane") != LANE
            or row.get("status") != "blocked"
            or row.get("ranking_status") != "not_rankable"
            or row.get("updates_completed") != 0
            or row.get("test_firewall", {}).get("frozen_test_accessed") is not False
        ):
            raise FaultStage3EvidenceInvalid(
                f"Stage-2 candidate record is not the accepted blocked cell: {expected_model}"
            )
    if cv_plan.get("effective_n_splits") != FROZEN_EFFECTIVE_FOLDS or cv_plan.get("folds") != []:
        raise FaultStage3EvidenceInvalid("fault Stage-3 protocol requires the frozen zero-fold P4 plan")


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _minimum_data_contract() -> list[dict[str, Any]]:
    return [
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
    ]


def _build_data_manifest(
    *,
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
    cv_plan: Mapping[str, Any],
    blind_audit: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    development = dict(stage1["development_source"])
    shape = [int(value) for value in development["batch_shape"]]
    total_voxels = math.prod(shape)
    positives = int(development["positive_labels"])
    negatives = int(development["verified_negative_labels"])
    unknown = int(development["unknown_labels"])
    valid = int(development["valid_labels"])
    if positives + negatives != valid or valid + unknown != total_voxels:
        raise FaultStage3EvidenceInvalid("Stage-1 positive/negative/unknown counts are inconsistent")
    stage2_gate = audit_data_gate(stage1, cv_plan, blind_audit)
    if stage2_gate.get("status") != "blocked" or stage2_gate.get("formal_training_allowed"):
        raise FaultStage3EvidenceInvalid("fault data gate unexpectedly permits formal training")
    inline_extent = [int(value) for value in blind_audit["searched_seismic_inline_extent"]]
    inline_count = inline_extent[1] - inline_extent[0] + 1
    annotation_coverage = blind_audit.get("annotation_coverage_evidence")
    audited_slice_count = (
        annotation_coverage.get("coverage_audited_slice_count")
        if isinstance(annotation_coverage, dict)
        else None
    )
    complete_blocks = len(blind_audit.get("complete_candidate_blocks", ()))
    return {
        "protocol": "fault-p5-stage3-data-readiness-v1",
        "protocol_baseline_commit": BASELINE_COMMIT,
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "repeat_seeds": list(REPEAT_SEEDS),
        "frozen_top_models": [],
        "stage2_reuse": {
            "candidate_model_ids": list(stage2["model_ids"]),
            "candidate_configuration_sha256": source_hashes["p5_source_locks.json"],
            "fixed_budget": dict(stage2["fixed_budget"]),
            "stage2_status": stage2["status"],
            "stage2_results_sha256": source_hashes["p5_stage2/p5_stage2_results.jsonl"],
            "configuration_changed": False,
            "preprocessing_changed": False,
            "loss_changed": False,
            "update_budget_changed": False,
            "hpo_performed": False,
        },
        "coverage": {
            "positive_annotation": {
                "fault_point_count": int(blind_audit["fault_point_count"]),
                "real_probe_positive_labels": positives,
                "status": "observed_not_sufficient_for_binary_cv",
            },
            "voxel_probe": {
                "scope": "one archived real Stage-1 development smoke crop",
                "batch_shape": shape,
                "sample_count": len(development.get("sample_keys", ())),
                "total_voxels": total_voxels,
                "positive_labels": positives,
                "positive_fraction": _fraction(positives, total_voxels),
                "verified_negative_labels": negatives,
                "verified_negative_fraction": _fraction(negatives, total_voxels),
                "unknown_labels": unknown,
                "unknown_fraction": _fraction(unknown, total_voxels),
                "valid_labels": valid,
                "source_role": development["source_role"],
                "source_sha256": development["source_sha256"],
            },
            "negative_provenance": {
                "status": "blocked",
                "audit_status": development.get("verified_negative_audit_status", "absent"),
                "audit_sha256": development.get("verified_negative_audit_sha256"),
                "verified_negative_labels": negatives,
            },
            "unknown_provenance": {
                "status": "blocked",
                "in_memory_semantics": "unlabelled voxels are unknown and invalid for formal loss",
                "source_mask_audit_status": stage2_gate["unknown_mask"][
                    "source_mask_audit_status"
                ],
                "source_mask_audit_sha256": stage2_gate["unknown_mask"][
                    "source_mask_audit_sha256"
                ],
                "source_mask_kind": stage2_gate["unknown_mask"]["source_mask_kind"],
            },
            "spatial": {
                "searched_inline_extent": inline_extent,
                "searched_inline_count": inline_count,
                "development_unique_inlines": int(cv_plan["metadata"]["observed_unique_inlines"]),
                "buffer_inlines": int(cv_plan["buffer_inlines"]),
                "block_support_count": len(cv_plan.get("block_support", {})),
                "complete_annotation_blocks": complete_blocks,
                "coverage_audited_volume_count": complete_blocks,
                "coverage_audited_slice_count": audited_slice_count,
                "slice_coverage_status": (
                    "not_quantifiable" if audited_slice_count is None else "reported"
                ),
                "scope": "portable aggregate annotation audit; no raw volume or test labels read",
            },
        },
        "split": {
            "source": "p4_preflight/buffered_cv_plan.json",
            "split_hash": source_hashes["p4_preflight/buffered_cv_plan.json"],
            "requested_n_splits": int(cv_plan["requested_n_splits"]),
            "effective_n_splits": int(cv_plan["effective_n_splits"]),
            "fold_count": len(cv_plan["folds"]),
            "status": cv_plan["status"],
            "temporary_split_created": False,
            "temporary_fraction_split_allowed": False,
        },
        "blockers": stage2_gate["blockers"]
        + [
            {
                "code": "NO_FROZEN_STAGE3_TOP3",
                "detail": "the Stage-3 protocol freezes no fault top-3 because effective folds are zero",
            }
        ],
        "minimum_data_contract_to_unblock": _minimum_data_contract(),
        "source_hashes": dict(source_hashes),
        "source_snapshot_sha256": _sha256_payload(source_hashes),
        "test_firewall": {
            "runner_accepts_test_inputs": False,
            "frozen_test_accessed": False,
            "test_labels_accessed": False,
            "test_predictions_accessed": False,
            "test_metrics_accessed": False,
            "blind_audit_used_for": "coverage/firewall status only",
            "historical_test_role": "regression_evidence_only",
        },
    }


def validate_result_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject duplicate records, cross-lane pollution, and invented training cells."""

    seen: set[tuple[Any, ...]] = set()
    for record in records:
        if (
            record.get("track_id") != "fault"
            or record.get("task_id") != TASK_ID
            or record.get("lane") != LANE
        ):
            raise FaultStage3EvidenceInvalid("cross-track/task/lane result pollution detected")
        key = (
            record.get("record_type"),
            record.get("track_id"),
            record.get("task_id"),
            record.get("lane"),
            record.get("model_id"),
            record.get("fold_id"),
            record.get("repeat_seed"),
        )
        if key in seen:
            raise FaultStage3EvidenceInvalid("duplicate Stage-3 result record detected")
        seen.add(key)
        if record.get("record_type") == "training_cell":
            raise FaultStage3EvidenceInvalid("fault Stage-3 has zero legal training folds/cells")
        if record.get("record_type") != "data_readiness_gate":
            raise FaultStage3EvidenceInvalid("unknown fault Stage-3 result record type")
        if any(record.get(field) is not None for field in ("model_id", "fold_id", "repeat_seed")):
            raise FaultStage3EvidenceInvalid("data-readiness gate must not masquerade as a model cell")


def completion_assessment(expected_cells: int, completed_cells: int) -> dict[str, Any]:
    if expected_cells < 0 or completed_cells < 0 or completed_cells > expected_cells:
        raise ValueError("invalid Stage-3 completion counts")
    if expected_cells == 0:
        return {
            "completion_rate": None,
            "minimum_rankable_rate": 0.8,
            "status": "not_rankable",
            "reason": "no scientifically legal training cells exist",
        }
    rate = completed_cells / expected_cells
    return {
        "completion_rate": rate,
        "minimum_rankable_rate": 0.8,
        "status": "rankable" if rate >= 0.8 else "not_rankable",
        "reason": None if rate >= 0.8 else "legal completion rate is below 80%",
    }


def _artifact_record(root: Path, relative_path: str, role: str) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "role": role,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def run_stage3_data_readiness(output_dir: Path) -> dict[str, Any]:
    """Run the zero-fold Stage-3 gate and write only portable artifacts."""

    source_lock = _read_json(SOURCE_LOCK)
    stage1 = _read_json(STAGE1_SUMMARY)
    cv_plan = _read_json(CV_PLAN)
    blind_audit = _read_json(BLIND_AUDIT)
    stage2 = _read_json(STAGE2_SUMMARY)
    stage2_rows = _read_jsonl(STAGE2_RESULTS)
    source_hashes = _source_hashes()
    _validate_stage2_reuse(
        source_lock,
        stage1,
        cv_plan,
        blind_audit,
        stage2,
        stage2_rows,
        source_hashes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_manifest = _build_data_manifest(
        stage1=stage1,
        stage2=stage2,
        cv_plan=cv_plan,
        blind_audit=blind_audit,
        source_hashes=source_hashes,
    )
    data_manifest_path = output_dir / DATA_MANIFEST_FILENAME
    _write_json(data_manifest_path, data_manifest)

    gate_record = {
        "protocol": "fault-p5-stage3-result-v1",
        "record_type": "data_readiness_gate",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "model_id": None,
        "fold_id": None,
        "repeat_seed": None,
        "frozen_repeat_seeds": list(REPEAT_SEEDS),
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "reason": "P4 has zero valid folds and Stage-3 freezes no fault top-3",
        "split_hash": data_manifest["split"]["split_hash"],
        "fixed_budget": dict(FIXED_BUDGET),
        "budget_consumed": {
            "parameter_updates": 0,
            "wall_seconds": 0.0,
            "train_samples": 0,
            "validation_samples": 0,
        },
        "fold_fit_operations": {
            "preprocessing": 0,
            "class_weights": 0,
            "target_transform": 0,
            "calibration": 0,
            "fit_scope": "not_invoked_no_valid_fold",
        },
        "validation_metrics": None,
        "data_manifest": {
            "path": DATA_MANIFEST_FILENAME,
            "sha256": _sha256(data_manifest_path),
        },
        "test_firewall": dict(data_manifest["test_firewall"]),
        "operations": {
            "model_built": False,
            "training_invoked": False,
            "prediction_generated": False,
            "checkpoint_written": False,
            "hpo_invoked": False,
        },
    }
    records = [gate_record]
    validate_result_records(records)
    results_path = output_dir / RESULTS_FILENAME
    _atomic_write_text(
        results_path,
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
    )

    completion = completion_assessment(0, 0)
    leaderboard = {
        "protocol": "fault-p5-stage3-leaderboard-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "not_rankable",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "entries": [],
        "frozen_top_models": [],
        "expected_training_cells": 0,
        "completed_training_cells": 0,
        **completion,
        "ranking_metrics": None,
        "test_firewall": dict(data_manifest["test_firewall"]),
    }
    leaderboard_path = output_dir / LEADERBOARD_PATH
    _write_json(leaderboard_path, leaderboard)

    oof_manifest = {
        "protocol": "fault-p5-stage3-oof-manifest-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "not_generated",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "expected_training_cells": 0,
        "prediction_artifacts": [],
        "prediction_count": 0,
        "large_prediction_policy": "track-private ignored directory only after data gate unblocks",
        "frozen_test_accessed": False,
    }
    oof_path = output_dir / OOF_MANIFEST_FILENAME
    _write_json(oof_path, oof_manifest)

    figures = build_figures(data_manifest_path, output_dir)
    visualization_manifest = {
        "protocol": "fault-p5-stage3-visualization-manifest-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "readiness_only",
        "source_artifact": {
            "path": DATA_MANIFEST_FILENAME,
            "sha256": _sha256(data_manifest_path),
        },
        "oof_prediction_count": 0,
        "figures": figures,
        "rebuild_entrypoint": "fault_p5_stage3_visualize.build_figures",
        "frozen_test_accessed": False,
        "historical_test_metrics_accessed": False,
    }
    visualization_path = output_dir / VISUALIZATION_MANIFEST_FILENAME
    _write_json(visualization_path, visualization_manifest)

    expected_cells = len(FROZEN_TOP_MODELS) * FROZEN_EFFECTIVE_FOLDS * len(REPEAT_SEEDS)
    summary = {
        "protocol": "fault-p5-stage3-summary-v1",
        "protocol_baseline_commit": BASELINE_COMMIT,
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "not_rankable",
        "reason_code": "NO_VALID_FAULT_DEVELOPMENT_FOLDS",
        "root_seed": ROOT_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "frozen_top_models": [],
        "effective_fold_count": FROZEN_EFFECTIVE_FOLDS,
        "expected_training_cells": expected_cells,
        "counts": {
            "data_readiness_gate_records": 1,
            "expected_training_cells": expected_cells,
            "attempted_training_cells": 0,
            "completed_training_cells": 0,
            "blocked_training_cells": 0,
            "failed_training_cells": 0,
            "timeout_training_cells": 0,
            "skip_training_cells": 0,
        },
        "completion": completion,
        "fixed_budget": dict(FIXED_BUDGET),
        "budget_consumed": dict(gate_record["budget_consumed"]),
        "split": dict(data_manifest["split"]),
        "stage2_reuse": dict(data_manifest["stage2_reuse"]),
        "fold_fit_operations": dict(gate_record["fold_fit_operations"]),
        "source_hashes": dict(source_hashes),
        "results_artifact": {
            "path": RESULTS_FILENAME,
            "sha256": _sha256(results_path),
            "bytes": results_path.stat().st_size,
            "record_count": len(records),
            "training_cell_count": 0,
        },
        "data_manifest": {
            "path": DATA_MANIFEST_FILENAME,
            "sha256": _sha256(data_manifest_path),
        },
        "leaderboards": [
            {
                "path": LEADERBOARD_PATH.as_posix(),
                "sha256": _sha256(leaderboard_path),
                "status": "not_rankable",
            }
        ],
        "oof_manifest": {
            "path": OOF_MANIFEST_FILENAME,
            "sha256": _sha256(oof_path),
            "prediction_count": 0,
        },
        "visualization_manifest": {
            "path": VISUALIZATION_MANIFEST_FILENAME,
            "sha256": _sha256(visualization_path),
            "figure_count": len(figures),
        },
        "prohibitions": {
            "hpo_performed": False,
            "candidate_changed": False,
            "preprocessing_changed": False,
            "loss_changed": False,
            "update_budget_changed": False,
            "temporary_split_created": False,
            "training_performed": False,
            "frozen_test_accessed": False,
        },
        "test_firewall": dict(data_manifest["test_firewall"]),
    }
    summary_path = output_dir / SUMMARY_FILENAME
    _write_json(summary_path, summary)

    artifact_specs = [
        (RESULTS_FILENAME, "structured_data_gate_result"),
        (SUMMARY_FILENAME, "stage3_summary"),
        (DATA_MANIFEST_FILENAME, "data_readiness_manifest"),
        (OOF_MANIFEST_FILENAME, "empty_oof_manifest"),
        (VISUALIZATION_MANIFEST_FILENAME, "readiness_visualization_manifest"),
        (LEADERBOARD_PATH.as_posix(), "not_rankable_leaderboard"),
    ] + [(figure["path"], "readiness_figure") for figure in figures]
    artifact_manifest = {
        "protocol": "fault-p5-stage3-artifact-manifest-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "artifacts": [
            _artifact_record(output_dir, relative_path, role)
            for relative_path, role in artifact_specs
        ],
        "large_artifacts_committed": False,
        "checkpoint_count": 0,
        "prediction_payload_count": 0,
        "frozen_test_accessed": False,
    }
    _write_json(output_dir / ARTIFACT_MANIFEST_FILENAME, artifact_manifest)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = run_stage3_data_readiness(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
