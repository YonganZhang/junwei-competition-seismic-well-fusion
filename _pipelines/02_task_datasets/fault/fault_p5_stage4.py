#!/usr/bin/env python3
"""Confirm that fault Stage-4 refit/holdout evaluation remains blocked.

The confirmation gate consumes the frozen, portable Stage-3 manifests only.
It hash-verifies Stage-3 readiness figures without interpreting or copying
them, records the minimum data contract needed to unblock the track, and never
accepts a winner, model, fold, refit, prediction, or holdout input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACK_DIR = Path(__file__).resolve().parent
STAGE3_DIR = TRACK_DIR / "_outputs" / "p5_stage3"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_stage4_confirmation"

TASK_ID = "fault_stick_segmentation"
LANE = "fault_3d_segmentation_data_gate"
REASON = "NO_VALID_FAULT_DEVELOPMENT_FOLDS"
BASELINE_COMMIT = "c9ac3cf8e18191c48cdb1ddfafa34355bf1548c7"

SUMMARY_FILENAME = "p5_stage3_summary.json"
DATA_MANIFEST_FILENAME = "p5_stage3_data_manifest.json"
OOF_MANIFEST_FILENAME = "p5_stage3_oof_manifest.json"
VISUALIZATION_MANIFEST_FILENAME = "p5_stage3_visualization_manifest.json"
ARTIFACT_MANIFEST_FILENAME = "p5_stage3_artifact_manifest.json"
LEADERBOARD_FILENAME = (
    "leaderboards/fault_stick_segmentation__fault_3d_segmentation_data_gate.json"
)
RESULTS_FILENAME = "p5_stage3_results.jsonl"

CONFIRMATION_FILENAME = "p5_stage4_confirmation.json"
VISUALIZATION_REUSE_FILENAME = "p5_stage4_visualization_reuse.json"
STAGE4_ARTIFACT_MANIFEST_FILENAME = "p5_stage4_artifact_manifest.json"

EXPECTED_CONTRACT_IDS = (
    "coverage_manifest",
    "verified_negative_mask",
    "unknown_and_proxy_masks",
    "buffered_development_folds",
    "fold_train_fit_provenance",
    "test_firewall",
)
EXPECTED_FIGURES = (
    "figures/fault_readiness.svg",
    "figures/fault_negative_coverage.svg",
    "figures/fault_unknown_coverage.svg",
)
EXPECTED_STAGE3_ARTIFACTS = {
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    DATA_MANIFEST_FILENAME,
    OOF_MANIFEST_FILENAME,
    VISUALIZATION_MANIFEST_FILENAME,
    LEADERBOARD_FILENAME,
    *EXPECTED_FIGURES,
}
FROZEN_STAGE3_HASHES = {
    SUMMARY_FILENAME: "52351babd768dc89dcf86b753a15e9392193ca140ce4aedecd020506bc8f6985",
    DATA_MANIFEST_FILENAME: "66f7f922fad93dfb8adc218cfcbebd65cbc0b7e4f061dca99a1feb3e884083a1",
    OOF_MANIFEST_FILENAME: "5131ed9fe4a4096d756f121d41e878eb759bc868fd27d614db05056430aa0dfd",
    VISUALIZATION_MANIFEST_FILENAME: "3fe7edbfb236b958d4ffde15521456766dfb49e7ff7817de99e3270f5744e1dd",
    ARTIFACT_MANIFEST_FILENAME: "ae5e709a0ba3838ee0e3ad69c12d1ef32b1c3b6dca8e12e2e2d1e1a5756d5e5e",
    LEADERBOARD_FILENAME: "e81823a5dc8786bc9b99c0de425480f8bb0a1e7d9c2cd29fafc2e6a4e36ea129",
    RESULTS_FILENAME: "03a2afa40d62004d097c57936841418735afd0770073bf603cf7a6c88515919d",
}


class FaultStage4ConfirmationError(RuntimeError):
    """Frozen Stage-3 evidence cannot safely support confirmation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        raise FaultStage4ConfirmationError(f"required frozen Stage-3 manifest missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FaultStage4ConfirmationError(f"Stage-3 manifest is not an object: {path.name}")
    return payload


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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FaultStage4ConfirmationError(message)


def _safe_source_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise FaultStage4ConfirmationError(f"unsafe Stage-3 artifact path: {relative!r}")
    return root / candidate


def _validate_firewall(payload: Mapping[str, Any], *, source: str) -> None:
    _require(
        payload.get("historical_test_role") == "regression_evidence_only",
        f"{source} does not preserve the historical-test regression-only role",
    )
    for field in (
        "frozen_test_accessed",
        "test_labels_accessed",
        "test_predictions_accessed",
        "test_metrics_accessed",
        "runner_accepts_test_inputs",
    ):
        _require(payload.get(field) is False, f"{source} test firewall opened at {field}")


def _validate_artifact_manifest(
    stage3_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _require(payload.get("protocol") == "fault-p5-stage3-artifact-manifest-v1", "bad Stage-3 artifact protocol")
    _require(payload.get("track_id") == "fault", "Stage-3 artifact track mismatch")
    _require(payload.get("task_id") == TASK_ID and payload.get("lane") == LANE, "Stage-3 artifact task/lane mismatch")
    _require(payload.get("checkpoint_count") == 0, "Stage-3 unexpectedly contains checkpoints")
    _require(payload.get("prediction_payload_count") == 0, "Stage-3 unexpectedly contains predictions")
    _require(payload.get("large_artifacts_committed") is False, "Stage-3 large-artifact policy changed")
    _require(payload.get("frozen_test_accessed") is False, "Stage-3 artifact manifest opened frozen test")

    records: dict[str, dict[str, Any]] = {}
    for raw in payload.get("artifacts", ()):
        _require(isinstance(raw, dict), "Stage-3 artifact entry is not an object")
        record = dict(raw)
        relative = str(record.get("path", ""))
        _require(relative not in records, f"duplicate Stage-3 artifact path: {relative}")
        path = _safe_source_path(stage3_dir, relative)
        _require(path.is_file(), f"Stage-3 artifact is missing: {relative}")
        _require(_sha256(path) == record.get("sha256"), f"Stage-3 artifact hash mismatch: {relative}")
        _require(path.stat().st_size == record.get("bytes"), f"Stage-3 artifact size mismatch: {relative}")
        records[relative] = record
    _require(set(records) == EXPECTED_STAGE3_ARTIFACTS, "Stage-3 artifact set changed")
    return records


def validate_frozen_stage3(stage3_dir: Path) -> dict[str, Any]:
    """Validate canonical blocked Stage-3 manifests without reading raw data."""

    artifact_manifest = _read_json(stage3_dir / ARTIFACT_MANIFEST_FILENAME)
    artifacts = _validate_artifact_manifest(stage3_dir, artifact_manifest)
    summary = _read_json(stage3_dir / SUMMARY_FILENAME)
    data = _read_json(stage3_dir / DATA_MANIFEST_FILENAME)
    oof = _read_json(stage3_dir / OOF_MANIFEST_FILENAME)
    visualization = _read_json(stage3_dir / VISUALIZATION_MANIFEST_FILENAME)
    leaderboard = _read_json(stage3_dir / LEADERBOARD_FILENAME)

    _require(summary.get("protocol") == "fault-p5-stage3-summary-v1", "bad Stage-3 summary protocol")
    _require(summary.get("track_id") == "fault", "Stage-3 summary track mismatch")
    _require(summary.get("task_id") == TASK_ID and summary.get("lane") == LANE, "Stage-3 summary task/lane mismatch")
    _require(summary.get("status") == "not_rankable", "Stage-3 summary is unexpectedly rankable")
    _require(summary.get("reason_code") == REASON, "Stage-3 summary reason changed")
    _require(summary.get("frozen_top_models") == [], "Stage-3 unexpectedly froze a winner candidate")
    _require(summary.get("effective_fold_count") == 0, "Stage-3 effective fold count is not zero")
    _require(summary.get("expected_training_cells") == 0, "Stage-3 invented legal training cells")
    counts = summary.get("counts", {})
    for field in (
        "expected_training_cells",
        "attempted_training_cells",
        "completed_training_cells",
        "failed_training_cells",
        "timeout_training_cells",
    ):
        _require(counts.get(field) == 0, f"Stage-3 nonzero training count: {field}")
    _require(summary.get("completion", {}).get("status") == "not_rankable", "Stage-3 completion became rankable")
    _require(summary.get("completion", {}).get("completion_rate") is None, "zero-cell completion must stay null")
    for field in ("parameter_updates", "train_samples", "validation_samples"):
        _require(summary.get("budget_consumed", {}).get(field) == 0, f"Stage-3 consumed training budget: {field}")
    for field in ("preprocessing", "class_weights", "target_transform", "calibration"):
        _require(summary.get("fold_fit_operations", {}).get(field) == 0, f"Stage-3 fold-fit operation ran: {field}")
    prohibitions = summary.get("prohibitions", {})
    for field in ("training_performed", "hpo_performed", "temporary_split_created", "frozen_test_accessed"):
        _require(prohibitions.get(field) is False, f"Stage-3 prohibition violated: {field}")
    _validate_firewall(summary.get("test_firewall", {}), source="Stage-3 summary")

    _require(data.get("protocol") == "fault-p5-stage3-data-readiness-v1", "bad Stage-3 data protocol")
    _require(data.get("track_id") == "fault", "Stage-3 data track mismatch")
    _require(data.get("task_id") == TASK_ID and data.get("lane") == LANE, "Stage-3 data task/lane mismatch")
    _require(data.get("status") == "blocked", "Stage-3 data gate is not blocked")
    _require(data.get("ranking_status") == "not_rankable", "Stage-3 data gate became rankable")
    _require(data.get("reason_code") == REASON, "Stage-3 data reason changed")
    _require(data.get("frozen_top_models") == [], "Stage-3 data manifest froze a model")
    negative = data.get("coverage", {}).get("negative_provenance", {})
    _require(negative.get("verified_negative_labels") == 0, "audited negative count unexpectedly changed")
    _require(negative.get("audit_status") == "absent" and negative.get("audit_sha256") is None, "negative audit provenance is not the frozen absence")
    unknown = data.get("coverage", {}).get("unknown_provenance", {})
    _require(unknown.get("status") == "blocked", "unknown-mask provenance unexpectedly unblocked")
    _require(unknown.get("source_mask_audit_status") == "absent", "unknown-mask audit status changed")
    split = data.get("split", {})
    _require(split.get("effective_n_splits") == 0 and split.get("fold_count") == 0, "Stage-3 split contains legal folds")
    _require(split.get("status") == "not_feasible", "Stage-3 split status changed")
    _require(split.get("temporary_split_created") is False, "Stage-3 used a temporary split")
    contracts = data.get("minimum_data_contract_to_unblock", ())
    _require(isinstance(contracts, list), "minimum data contract is not a list")
    contract_ids = tuple(item.get("contract_id") for item in contracts if isinstance(item, dict))
    _require(contract_ids == EXPECTED_CONTRACT_IDS, "minimum data contract changed or reordered")
    _require(all(str(item.get("minimum", "")).strip() for item in contracts), "minimum data contract contains an empty clause")
    _validate_firewall(data.get("test_firewall", {}), source="Stage-3 data manifest")

    _require(oof.get("protocol") == "fault-p5-stage3-oof-manifest-v1", "bad Stage-3 OOF protocol")
    _require(oof.get("status") == "not_generated", "Stage-3 OOF unexpectedly exists")
    _require(oof.get("prediction_count") == 0 and oof.get("prediction_artifacts") == [], "Stage-3 OOF contains predictions")
    _require(oof.get("frozen_test_accessed") is False, "Stage-3 OOF opened frozen test")

    _require(visualization.get("protocol") == "fault-p5-stage3-visualization-manifest-v1", "bad Stage-3 visualization protocol")
    _require(visualization.get("status") == "readiness_only", "Stage-3 visualization is not readiness-only")
    _require(visualization.get("oof_prediction_count") == 0, "Stage-3 visualization claims OOF predictions")
    _require(visualization.get("frozen_test_accessed") is False, "Stage-3 visualization opened frozen test")
    _require(visualization.get("historical_test_metrics_accessed") is False, "Stage-3 visualization read historical metrics")
    _require(visualization.get("source_artifact", {}).get("sha256") == artifacts[DATA_MANIFEST_FILENAME]["sha256"], "Stage-3 visualization data hash mismatch")
    figures = visualization.get("figures", ())
    _require(tuple(item.get("path") for item in figures if isinstance(item, dict)) == EXPECTED_FIGURES, "Stage-3 readiness figure set changed")
    for figure in figures:
        record = artifacts[figure["path"]]
        _require(figure.get("sha256") == record["sha256"], f"Stage-3 figure hash mismatch: {figure['path']}")
        _require(figure.get("bytes") == record["bytes"], f"Stage-3 figure size mismatch: {figure['path']}")
        _require(figure.get("frozen_test_accessed") is False, f"Stage-3 figure opened frozen test: {figure['path']}")

    _require(leaderboard.get("protocol") == "fault-p5-stage3-leaderboard-v1", "bad Stage-3 leaderboard protocol")
    _require(leaderboard.get("status") == "not_rankable", "Stage-3 leaderboard became rankable")
    _require(leaderboard.get("reason_code") == REASON, "Stage-3 leaderboard reason changed")
    _require(leaderboard.get("entries") == [] and leaderboard.get("frozen_top_models") == [], "Stage-3 leaderboard contains a winner")
    _validate_firewall(leaderboard.get("test_firewall", {}), source="Stage-3 leaderboard")

    summary_links = {
        DATA_MANIFEST_FILENAME: summary.get("data_manifest", {}).get("sha256"),
        OOF_MANIFEST_FILENAME: summary.get("oof_manifest", {}).get("sha256"),
        VISUALIZATION_MANIFEST_FILENAME: summary.get("visualization_manifest", {}).get("sha256"),
        LEADERBOARD_FILENAME: summary.get("leaderboards", [{}])[0].get("sha256"),
        RESULTS_FILENAME: summary.get("results_artifact", {}).get("sha256"),
    }
    for relative, linked_hash in summary_links.items():
        _require(linked_hash == artifacts[relative]["sha256"], f"Stage-3 summary link mismatch: {relative}")
    _require(summary.get("results_artifact", {}).get("training_cell_count") == 0, "Stage-3 results claim training cells")

    source_manifest_hashes = {
        SUMMARY_FILENAME: artifacts[SUMMARY_FILENAME]["sha256"],
        DATA_MANIFEST_FILENAME: artifacts[DATA_MANIFEST_FILENAME]["sha256"],
        OOF_MANIFEST_FILENAME: artifacts[OOF_MANIFEST_FILENAME]["sha256"],
        VISUALIZATION_MANIFEST_FILENAME: artifacts[VISUALIZATION_MANIFEST_FILENAME]["sha256"],
        ARTIFACT_MANIFEST_FILENAME: _sha256(stage3_dir / ARTIFACT_MANIFEST_FILENAME),
        LEADERBOARD_FILENAME: artifacts[LEADERBOARD_FILENAME]["sha256"],
        RESULTS_FILENAME: artifacts[RESULTS_FILENAME]["sha256"],
    }
    _require(
        source_manifest_hashes == FROZEN_STAGE3_HASHES,
        "Stage-3 source lock differs from clean c9ac3cf baseline",
    )
    return {
        "summary": summary,
        "data": data,
        "oof": oof,
        "visualization": visualization,
        "leaderboard": leaderboard,
        "artifact_records": artifacts,
        "source_manifest_hashes": source_manifest_hashes,
    }


def _artifact_record(output_dir: Path, relative: str, role: str) -> dict[str, Any]:
    path = output_dir / relative
    return {
        "path": relative,
        "role": role,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def run_stage4_confirmation(output_dir: Path) -> dict[str, Any]:
    """Write deterministic confirmation evidence from canonical Stage-3 only."""

    frozen = validate_frozen_stage3(STAGE3_DIR)
    summary = frozen["summary"]
    data = frozen["data"]
    visualization = frozen["visualization"]
    source_hashes = frozen["source_manifest_hashes"]

    visualization_reuse = {
        "protocol": "fault-p5-stage4-visualization-reuse-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "reused_readiness_only",
        "reuse_mode": "hash_reference_only",
        "source_visualization_manifest": {
            "path": f"p5_stage3/{VISUALIZATION_MANIFEST_FILENAME}",
            "sha256": source_hashes[VISUALIZATION_MANIFEST_FILENAME],
        },
        "source_data_manifest": {
            "path": f"p5_stage3/{DATA_MANIFEST_FILENAME}",
            "sha256": source_hashes[DATA_MANIFEST_FILENAME],
        },
        "figures": [
            {
                "path": f"p5_stage3/{figure['path']}",
                "sha256": figure["sha256"],
                "bytes": figure["bytes"],
                "role": "stage3_data_readiness_evidence",
            }
            for figure in visualization["figures"]
        ],
        "prediction_source": None,
        "prediction_artifacts": [],
        "prediction_fabricated": False,
        "refit_executed": False,
        "holdout_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_reuse_path = output_dir / VISUALIZATION_REUSE_FILENAME
    _write_json(visualization_reuse_path, visualization_reuse)

    stage3_firewall = data["test_firewall"]
    prior_test_metadata = {
        "metadata_exposure_present": True,
        "exposure_source": "frozen Stage-3 test_firewall declaration",
        "historical_test_role": stage3_firewall["historical_test_role"],
        "blind_audit_used_for": stage3_firewall["blind_audit_used_for"],
        "prior_metric_values_re_emitted": False,
        "historical_test_artifacts_read_by_stage4": False,
        "historical_test_metrics_read_by_stage4": False,
        "independence_status": "blocked",
        "scope_note": (
            "Stage-4 records only the historical regression-evidence role declared by "
            "frozen Stage-3; it does not re-read P4 or audited historical artifacts."
        ),
    }
    confirmation = {
        "protocol": "fault-p5-stage4-confirmation-v1",
        "baseline_commit": BASELINE_COMMIT,
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "blocked",
        "ranking_status": "not_rankable",
        "reason": REASON,
        "reason_detail": "Frozen Stage-3 has zero legal development folds and no frozen winner.",
        "frozen_winner": None,
        "refit_executed": False,
        "holdout_accessed": False,
        "effective_fold_count": summary["effective_fold_count"],
        "verified_negative_labels": data["coverage"]["negative_provenance"][
            "verified_negative_labels"
        ],
        "operations": {
            "model_built": False,
            "training_invoked": False,
            "checkpoint_written": False,
            "prediction_generated": False,
            "metric_computed": False,
            "refit_invoked": False,
            "holdout_loader_created": False,
            "holdout_accessed": False,
        },
        "minimum_unblock_data_contract": [
            dict(item) for item in data["minimum_data_contract_to_unblock"]
        ],
        "prior_test_metadata_exposure": prior_test_metadata,
        "stage3_source": {
            "manifest_hashes": source_hashes,
            "manifest_snapshot_sha256": _sha256_payload(source_hashes),
            "stage3_source_snapshot_sha256": data["source_snapshot_sha256"],
            "status": summary["status"],
            "reason_code": summary["reason_code"],
            "frozen_top_models": summary["frozen_top_models"],
            "expected_training_cells": summary["expected_training_cells"],
            "completed_training_cells": summary["counts"]["completed_training_cells"],
        },
        "visualization_reuse": {
            "path": VISUALIZATION_REUSE_FILENAME,
            "sha256": _sha256(visualization_reuse_path),
            "figure_count": len(visualization_reuse["figures"]),
            "prediction_count": 0,
        },
        "test_firewall": {
            "runner_accepts_test_inputs": False,
            "holdout_loader_created": False,
            "holdout_accessed": False,
            "test_labels_accessed": False,
            "test_predictions_accessed": False,
            "test_metrics_accessed": False,
            "historical_test_artifacts_accessed": False,
        },
    }
    confirmation_path = output_dir / CONFIRMATION_FILENAME
    _write_json(confirmation_path, confirmation)

    artifact_manifest = {
        "protocol": "fault-p5-stage4-artifact-manifest-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "blocked",
        "artifacts": [
            _artifact_record(output_dir, CONFIRMATION_FILENAME, "stage4_confirmation"),
            _artifact_record(
                output_dir,
                VISUALIZATION_REUSE_FILENAME,
                "stage3_visualization_hash_reuse",
            ),
        ],
        "stage3_manifest_snapshot_sha256": confirmation["stage3_source"][
            "manifest_snapshot_sha256"
        ],
        "runner_sha256": confirmation["runner_sha256"],
        "checkpoint_count": 0,
        "prediction_payload_count": 0,
        "refit_executed": False,
        "holdout_accessed": False,
    }
    _write_json(output_dir / STAGE4_ARTIFACT_MANIFEST_FILENAME, artifact_manifest)
    return confirmation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    confirmation = run_stage4_confirmation(args.output_dir)
    print(json.dumps(confirmation, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
