"""Sweetspot-private P5 Stage-4 known-holdout confirmation.

This is deliberately not a fresh-blind final test.  T1--T4 refit one frozen
Stage-3 winner on every P4-authorized development sample, close and hash the
refit state, and only then rebuild the already-consumed P4 holdout.  T5 has no
label; T6/T7 have neither a development-only feature source nor a Stage-3
winner and therefore never enter either data phase.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import resource
import shutil
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _code.ml_framework.model_discovery import discover_model

from .source_lock import DEFAULT_LOCK, inspect_runtime, load_source_lock
from .sweetspot_p5_stage2 import (
    TREE_UPDATE_LIMIT,
    _column_impute,
    _inverse_target,
    _target_transform,
    _tree_config,
)
from .sweetspot_p5_stage2_data import (
    _build_productivity,
    _build_water_breakthrough,
    _flatten_petrophysical,
    _load_development_petrophysical_tables,
    _load_development_production,
    _resolve_source_root,
    _take,
)
from .sweetspot_p5_stage2_labels import (
    DEFAULT_MAPPING_PATH,
    PROJECT_ROOT,
    LabelMappingAudit,
    build_pilot_task_spec,
    canonical_sha256,
    sha256_file,
    validate_label_mapping,
)


ROOT_SEED = 2693
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "stage4_confirmation"
STAGE3_DIR = HERE / "_outputs" / "stage3_cv"
STAGE3_COMMIT = "5a1fefe977efe20d2d7d12f1601143cdcb2d5678"
STAGE3_SUMMARY_SHA256 = "9dc9420f1bafcf91e341c1986e083fa5cb42fc5f3bd4855f774e146290381877"
STAGE3_RESULTS_SHA256 = "34443cf1afa8c1387c80333a3119a88215d79203db005da7af1a60cb674934b8"
STAGE3_OOF_SHA256 = "92907e7ba054ec1557864c9e2e1f78232e029cae8c6aab5aa538ebfe81268abd"
EXPECTED_LABEL_MAPPING_SHA256 = "a66bb5bc0098307ab03c3f3c64ee156a5292725ad1740488eaeb3b8bf572a167"
EVIDENCE_CLASS = "previously_seen_reusable_holdout"

FROZEN_WINNERS: Mapping[str, Mapping[str, Any]] = {
    "T1": {
        "model_id": "lightgbm",
        "updates": 64,
        "task_type": "regression",
        "leaderboard_sha256": "7c992062b83f12da7a6de244ce543c898477a1bcd28765943fb47022f2c12d8b",
    },
    "T2": {
        "model_id": "catboost",
        "updates": 64,
        "task_type": "binary",
        "leaderboard_sha256": "69ed4d406b7d71b2bd65716c4e9afeb76b0b46eeeaa266bf3d9c27b83377ca9b",
    },
    "T3": {
        "model_id": "xgboost",
        "updates": 64,
        "task_type": "regression",
        "leaderboard_sha256": "a68d7cee0bd9ede4907ede0f59dd55530821524999bac429bb7feb1824e772d7",
    },
    "T4": {
        "model_id": "catboost",
        "updates": 64,
        "task_type": "binary",
        "leaderboard_sha256": "0749130e47b6c7645fc2a2b497531d14d2c2ce29827fd9354ab4401a1bb7d048",
    },
}

# Lifecycle files are metadata-only evidence that these holdouts were already
# consumed in P4.  Stage-4 never opens P4 frozen_test metrics or predictions.
P4_EXPOSURE: Mapping[str, Mapping[str, Any]] = {
    "T1": {
        "path": "_pipelines/02_task_datasets/sweetspot/targets/reservoir_quality/_outputs/baseline_v1/lifecycle.json",
        "sha256": "223ca2356e17370452f85278ebbbdd5ad68ed1ca597450f404f4f4d3fe87eac7",
        "experiment_id": "sweetspot.reservoir_quality.baseline_v1",
        "test_rows": 11936,
    },
    "T2": {
        "path": "_pipelines/02_task_datasets/sweetspot/targets/hydrocarbon_pay/_outputs/baseline_v1/lifecycle.json",
        "sha256": "7dc7080d006267cd6a1cbc7da667b3ad05bdc3db185024b56a863aeac868ec24",
        "experiment_id": "sweetspot.hydrocarbon_pay.baseline_v1",
        "test_rows": 12081,
    },
    "T3": {
        "path": "_pipelines/02_task_datasets/sweetspot/targets/productivity/_outputs/baseline_v1/lifecycle.json",
        "sha256": "84ed1f2c28296d0285507d0a1875ac370322bf0a8b363725b04b6b42e496763b",
        "experiment_id": "sweetspot.productivity.baseline_v1",
        "test_rows": 132,
    },
    "T4": {
        "path": "_pipelines/02_task_datasets/sweetspot/targets/water_breakthrough/_outputs/baseline_v1/lifecycle.json",
        "sha256": "4bb592942d70e53e667caac964e2ebc41ce40b838196ad1b66a0238ad58a3bfc",
        "experiment_id": "sweetspot.water_breakthrough.baseline_v1",
        "test_rows": 37,
    },
    "T6": {
        "path": "_pipelines/02_task_datasets/sweetspot/targets/porosity/_outputs/phif/lifecycle.json",
        "sha256": "d1b907cea5cf12d6e8efa2a7f9df1cd7868abdb308efd4f162363355368de064",
        "experiment_id": "target6_porosity_phif-target6-phif-cpi-v1",
        "test_rows": 344,
    },
    "T7": {
        "path": "_pipelines/02_task_datasets/sweetspot/targets/permeability/_outputs/klogh/lifecycle.json",
        "sha256": "f845f0a5f3c8c2de8de98a31ba7f7407576e302094e514359be9c7d75e244d43",
        "experiment_id": "target7_permeability_klogh-target7-klogh-cpi-v1",
        "test_rows": 344,
    },
}


@dataclass(frozen=True)
class PartitionData:
    task_id: str
    partition: str
    sample_ids: tuple[str, ...]
    groups: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    auxiliary_name: str
    auxiliary: tuple[str | float, ...]
    provenance: Mapping[str, Any]


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as exc:
        raise PermissionError("Stage-4 outputs must stay inside sweetspot/p5") from exc
    return resolved


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_stage4_contract(
    audit: LabelMappingAudit,
    *,
    stage3_dir: Path = STAGE3_DIR,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Verify frozen Stage-3 selection, P4 splits and prior exposure metadata."""
    stage3_dir = Path(stage3_dir)
    summary_path = stage3_dir / "p5_stage3_summary.json"
    results_path = stage3_dir / "p5_stage3_results.jsonl"
    oof_path = stage3_dir / "p5_stage3_oof_manifest.json"
    if sha256_file(summary_path) != STAGE3_SUMMARY_SHA256:
        raise ValueError("Stage-3 summary hash changed from commit 5a1fefe")
    summary = _json(summary_path)
    if summary.get("results_sha256") != STAGE3_RESULTS_SHA256 or sha256_file(results_path) != STAGE3_RESULTS_SHA256:
        raise ValueError("Stage-3 result hash mismatch")
    if summary.get("oof_manifest_sha256") != STAGE3_OOF_SHA256 or sha256_file(oof_path) != STAGE3_OOF_SHA256:
        raise ValueError("Stage-3 OOF manifest hash mismatch")
    if summary.get("counts") != {"FAILED": 0, "PASS": 117, "SKIP": 0, "TIMEOUT": 0}:
        raise ValueError("Stage-3 did not retain all 117 legal PASS cells")
    if summary.get("test_accessed") or summary.get("historical_test_metrics_used"):
        raise ValueError("Stage-3 selection was not development-only")
    if audit.mapping_sha256 != EXPECTED_LABEL_MAPPING_SHA256:
        raise ValueError("P5 label mapping changed after Stage-3")

    winners: dict[str, Any] = {}
    split_evidence: dict[str, Any] = {}
    for task_id, frozen in FROZEN_WINNERS.items():
        board_path = stage3_dir / "leaderboards" / f"{task_id}.json"
        actual_board_hash = sha256_file(board_path)
        if actual_board_hash != frozen["leaderboard_sha256"]:
            raise ValueError(f"{task_id}: Stage-3 leaderboard hash changed")
        board = _json(board_path)
        entries = board.get("entries", ())
        if board.get("status") != "rankable" or not entries or entries[0].get("rank") != 1:
            raise ValueError(f"{task_id}: Stage-3 leaderboard has no unique rank-1 winner")
        if entries[0].get("model_id") != frozen["model_id"]:
            raise ValueError(f"{task_id}: winner changed after Stage-3")
        if len(entries) > 1 and entries[0].get("tied"):
            raise ValueError(f"{task_id}: rank-1 winner is tied")
        target = audit.target(task_id)
        split_path = Path(project_root) / target["split_manifest"]["path"]
        if sha256_file(split_path) != target["split_manifest"]["sha256"]:
            raise ValueError(f"{task_id}: P4 split hash changed")
        split = _json(split_path)
        development_ids = set(split["development_sample_ids"])
        test_ids = set(split["test_sample_ids"])
        development_groups = set(split["development_groups"])
        test_groups = set(split["test_groups"])
        if not development_ids or not test_ids or development_ids & test_ids:
            raise ValueError(f"{task_id}: development/holdout IDs are not isolated")
        if not development_groups or not test_groups or development_groups & test_groups:
            raise ValueError(f"{task_id}: development/holdout groups are not isolated")
        if len(test_ids) != int(P4_EXPOSURE[task_id]["test_rows"]):
            raise ValueError(f"{task_id}: P4 holdout count changed")
        winners[task_id] = {
            "model_id": frozen["model_id"],
            "updates": frozen["updates"],
            "leaderboard_path": board_path.relative_to(project_root).as_posix(),
            "leaderboard_sha256": actual_board_hash,
        }
        split_evidence[task_id] = {
            "path": target["split_manifest"]["path"],
            "sha256": target["split_manifest"]["sha256"],
            "development_samples": len(development_ids),
            "known_holdout_samples": len(test_ids),
            "development_groups": sorted(development_groups),
            "known_holdout_groups": sorted(test_groups),
        }

    prior_exposure: dict[str, Any] = {}
    for task_id, evidence in P4_EXPOSURE.items():
        lifecycle_path = Path(project_root) / evidence["path"]
        if sha256_file(lifecycle_path) != evidence["sha256"]:
            raise ValueError(f"{task_id}: P4 lifecycle exposure hash changed")
        lifecycle = _json(lifecycle_path)
        if lifecycle.get("state") != "VERIFIED" or not lifecycle.get("test_consumed_at"):
            raise ValueError(f"{task_id}: prior P4 holdout exposure is not explicit")
        if "TEST_CONSUMED" not in lifecycle.get("evidence", {}):
            raise ValueError(f"{task_id}: P4 TEST_CONSUMED evidence missing")
        if lifecycle.get("experiment_id") != evidence["experiment_id"]:
            raise ValueError(f"{task_id}: P4 lifecycle experiment changed")
        accessed = lifecycle.get("evidence", {}).get("VERIFIED", {}).get("test_rows_accessed")
        if task_id in FROZEN_WINNERS and int(accessed or -1) != int(evidence["test_rows"]):
            raise ValueError(f"{task_id}: P4 lifecycle holdout count changed")
        prior_exposure[task_id] = {
            "lifecycle_path": evidence["path"],
            "lifecycle_sha256": evidence["sha256"],
            "test_consumed_at": lifecycle["test_consumed_at"],
            "test_rows": evidence["test_rows"],
            "prior_test_consumed": True,
        }

    target_status = summary.get("target_status", {})
    if target_status.get("T5", {}).get("status") != "not_feasible":
        raise ValueError("T5 must remain not_feasible")
    for task_id in ("T6", "T7"):
        status = target_status.get(task_id, {})
        if status.get("status") != "blocked" or status.get("development_feature_source_available") is not False:
            raise ValueError(f"{task_id} must remain blocked without a development feature source")
    return {
        "stage3_commit": STAGE3_COMMIT,
        "stage3_summary_sha256": STAGE3_SUMMARY_SHA256,
        "stage3_results_sha256": STAGE3_RESULTS_SHA256,
        "stage3_oof_sha256": STAGE3_OOF_SHA256,
        "winners": winners,
        "splits": split_evidence,
        "prior_exposure": prior_exposure,
        "historical_test_metrics_read": False,
    }


def _stage4_task_spec(audit: LabelMappingAudit, task_id: str) -> Any:
    p5 = build_pilot_task_spec(audit, task_id)
    metadata = dict(p5.metadata)
    metadata.update({
        "p5_stage": "known_holdout_confirmation",
        "fit_scope": "all_legal_development",
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "test_feedback_for_selection": "forbidden",
    })
    return replace(
        p5,
        task_id=f"sweetspot.p5.stage4.{task_id.lower()}.{audit.target(task_id)['slug']}",
        hpo={
            "stage": "P5 Stage-4 known-holdout confirmation",
            "optimization": "forbidden",
            "winner_source": "P5 Stage-3 rank-1 at commit 5a1fefe",
            "root_seed": ROOT_SEED,
            "test_feedback": "forbidden",
        },
        metadata=metadata,
    )


def _partition_auxiliary(task_id: str, sample_ids: Sequence[str]) -> tuple[str, tuple[str | float, ...]]:
    if task_id in {"T1", "T2"}:
        return "depth_m", tuple(float(sample_id.rsplit(":", 1)[1]) for sample_id in sample_ids)
    return "cutoff_date", tuple(sample_id.rsplit(":", 1)[1] for sample_id in sample_ids)


def _rebuild_partition(
    audit: LabelMappingAudit,
    task_id: str,
    partition: str,
    *,
    source_root: Path | None,
) -> PartitionData:
    if task_id not in FROZEN_WINNERS:
        raise PermissionError(f"{task_id}: no Stage-3 winner may enter Stage-4 data loading")
    split = audit.split_manifest(task_id)
    if partition == "development":
        groups = set(split["development_groups"])
        expected_ids = tuple(split["development_sample_ids"])
    elif partition == "known_holdout":
        groups = set(split["test_groups"])
        expected_ids = tuple(split["test_sample_ids"])
    else:
        raise ValueError(f"unknown partition: {partition}")
    resolved_root = _resolve_source_root(source_root)
    if task_id in {"T1", "T2"}:
        tables, provenance = _load_development_petrophysical_tables(resolved_root, groups)
        dataset = _flatten_petrophysical(tables, target_id=task_id)
    else:
        frame, provenance = _load_development_production(resolved_root, groups)
        dataset = _build_productivity(frame) if task_id == "T3" else _build_water_breakthrough(frame)
    rebuilt_ids = set(dataset["sample_ids"])
    if rebuilt_ids != set(expected_ids):
        raise ValueError(
            f"{task_id}/{partition}: rebuilt IDs differ from the frozen P4 manifest "
            f"(missing={len(set(expected_ids) - rebuilt_ids)}, unexpected={len(rebuilt_ids - set(expected_ids))})"
        )
    features, _sequences, targets, rebuilt_groups = _take(dataset, expected_ids)
    if set(rebuilt_groups) != groups:
        raise ValueError(f"{task_id}/{partition}: rebuilt groups changed")
    auxiliary_name, auxiliary = _partition_auxiliary(task_id, expected_ids)
    normalized_provenance = {
        key: value for key, value in provenance.items()
        if key not in {"authorized_development_groups", "source_kind", "test_accessed"}
    }
    authorization = "development" if partition == "development" else "known-holdout"
    source_kind = (
        f"Volve_Well_logs.zip authorized {authorization} members only"
        if task_id in {"T1", "T2"}
        else f"Volve production workbook with row-level {authorization} authorization"
    )
    normalized_provenance.update({
        "source_kind": source_kind,
        "partition": partition,
        "authorized_groups": sorted(groups),
        "values_accessed": len(expected_ids),
        "test_accessed": partition == "known_holdout",
        "p4_split_manifest_path": audit.target(task_id)["split_manifest"]["path"],
        "p4_split_manifest_sha256": audit.target(task_id)["split_manifest"]["sha256"],
    })
    return PartitionData(
        task_id=task_id,
        partition=partition,
        sample_ids=expected_ids,
        groups=rebuilt_groups,
        features=np.asarray(features, dtype=np.float64),
        targets=np.asarray(targets, dtype=np.float64),
        feature_names=tuple(dataset["feature_names"]),
        auxiliary_name=auxiliary_name,
        auxiliary=auxiliary,
        provenance=normalized_provenance,
    )


def _fit_frozen_winner(
    audit: LabelMappingAudit,
    task_id: str,
    development: PartitionData,
    source_lock: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, np.ndarray, bytes, dict[str, Any], dict[str, Any]]:
    winner = FROZEN_WINNERS[task_id]
    model_id = str(winner["model_id"])
    runtime = inspect_runtime(source_lock[model_id])
    if not runtime["available"] or not runtime["version_allowed"]:
        raise RuntimeError(f"{task_id}/{model_id}: frozen runtime unavailable: {runtime}")
    task_spec = _stage4_task_spec(audit, task_id)
    model_config = _tree_config(model_id, ROOT_SEED)
    if int(model_config.get("n_estimators", model_config.get("iterations", -1))) != TREE_UPDATE_LIMIT:
        raise ValueError(f"{task_id}: frozen 64-update config changed")
    train_x, _duplicate, medians = _column_impute(development.features, development.features)
    train_y = _target_transform(development.targets, task_spec)
    started = time.monotonic()
    adapter = discover_model("sweetspot", model_id).build(task_spec, **model_config)
    target_name = task_spec.targets[0]
    adapter.fit(
        train_x,
        {target_name: train_y},
        {target_name: np.ones(len(train_y), dtype=bool)},
    )
    refit_wall = time.monotonic() - started
    payload = pickle.dumps(adapter, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(payload)
    probe = train_x[: min(32, len(train_x))]
    before = np.asarray(adapter.predict(probe).raw[target_name], dtype=float)
    after = np.asarray(restored.predict(probe).raw[target_name], dtype=float)
    delta = float(np.max(np.abs(before - after))) if before.size else 0.0
    if delta > 1e-10:
        raise RuntimeError(f"{task_id}: checkpoint round-trip changed predictions by {delta}")
    config = {
        "schema_version": "sweetspot-p5-stage4-config/v1",
        "task_id": task_id,
        "lane": audit.target(task_id)["slug"],
        "task_spec": task_spec.to_dict(),
        "model_id": model_id,
        "model_config": model_config,
        "update_steps": TREE_UPDATE_LIMIT,
        "root_seed": ROOT_SEED,
        "selection_scope": "frozen Stage-3 development CV rank-1",
        "threshold": 0.5 if task_spec.task_type == "binary" else None,
        "threshold_fit_scope": "fixed_0.5; no holdout fit" if task_spec.task_type == "binary" else "not_applicable",
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "source_lock": {
            "revision": source_lock[model_id]["revision"],
            "license": source_lock[model_id]["license"],
            "runtime": runtime,
        },
    }
    refit = {
        "schema_version": "sweetspot-p5-stage4-refit/v1",
        "task_id": task_id,
        "status": "REFIT_COMPLETE",
        "fit_scope": "all_legal_development",
        "development_samples": len(development.sample_ids),
        "development_groups": sorted(set(development.groups)),
        "feature_count": int(development.features.shape[1]),
        "development_ids_sha256": canonical_sha256(list(development.sample_ids)),
        "development_features_sha256": _array_sha256(development.features),
        "development_targets_sha256": _array_sha256(development.targets),
        "train_only_imputation_sha256": canonical_sha256(medians.tolist()),
        "model_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "model_payload_bytes": len(payload),
        "checkpoint_roundtrip_max_abs_delta": delta,
        "wall_seconds": refit_wall,
        "peak_rss_bytes": _rss_bytes(),
        "known_holdout_accessed_during_refit": False,
        "historical_test_metrics_read": False,
        "development_provenance": development.provenance,
    }
    return restored, medians, payload, config, refit


def _predict_holdout(adapter: Any, medians: np.ndarray, task_spec: Any, holdout: PartitionData) -> np.ndarray:
    x = np.where(np.isfinite(holdout.features), holdout.features, medians[None, :])
    target_name = task_spec.targets[0]
    output = adapter.predict(x)
    if task_spec.task_type == "binary":
        prediction = np.asarray(output.transformed[target_name], dtype=np.float64).reshape(-1)
        if np.any((prediction < 0.0) | (prediction > 1.0)):
            raise ValueError("binary holdout predictions are not probabilities")
    else:
        prediction = _inverse_target(
            np.asarray(output.raw[target_name], dtype=np.float64).reshape(-1), task_spec,
        )
    if prediction.shape != holdout.targets.shape or not np.isfinite(prediction).all():
        raise ValueError("known-holdout prediction has wrong shape or non-finite values")
    return prediction


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score

    residual = prediction - actual
    constant = bool(np.allclose(actual, actual[0]))
    correlation = spearmanr(actual, prediction).statistic if len(actual) >= 2 else float("nan")
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": None if constant else float(r2_score(actual, prediction)),
        "r2_reason": "constant observed target" if constant else None,
        "spearman": _finite(correlation),
        "negative_prediction_fraction": float(np.mean(prediction < 0.0)),
        "sample_count": int(len(actual)),
    }


def _binary_metrics(actual: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score

    return {
        "average_precision": float(average_precision_score(actual.astype(int), probability)),
        "brier": float(brier_score_loss(actual.astype(int), probability)),
        "f1_at_0_5": float(f1_score(actual.astype(int), probability >= 0.5, zero_division=0)),
        "threshold": 0.5,
        "threshold_source": "frozen constant; not fit on known holdout",
        "positive_rate": float(np.mean(actual)),
        "sample_count": int(len(actual)),
    }


def _thickness_diagnostic(data: PartitionData, probability: np.ndarray) -> dict[str, Any]:
    groups = np.asarray(data.groups)
    depths = np.asarray(data.auxiliary, dtype=float)
    rows = []
    for group in sorted(set(data.groups)):
        mask = groups == group
        ordered = np.sort(depths[mask])
        spacing = float(np.median(np.diff(ordered))) if len(ordered) > 1 else 0.0
        actual_m = spacing * float(data.targets[mask].sum())
        predicted_m = spacing * float((probability[mask] >= 0.5).sum())
        rows.append({
            "group": group,
            "median_sample_spacing_m": spacing,
            "actual_net_thickness_m": actual_m,
            "predicted_net_thickness_m": predicted_m,
            "absolute_error_m": abs(predicted_m - actual_m),
        })
    return {
        "definition": "median within-group sample spacing times positive sample count at frozen threshold 0.5",
        "net_thickness_mae_m": float(np.mean([row["absolute_error_m"] for row in rows])),
        "by_group": rows,
    }


def _topk_diagnostic(data: PartitionData, prediction: np.ndarray) -> dict[str, Any]:
    count = max(1, int(math.ceil(len(prediction) * 0.10)))
    actual_order = np.argsort(data.targets)[-count:]
    predicted_order = np.argsort(prediction)[-count:]
    overlap = len(set(actual_order.tolist()) & set(predicted_order.tolist()))
    return {
        "fraction": 0.10,
        "k": count,
        "hit_rate": float(overlap / count),
        "overlap_count": overlap,
        "selection_feedback": "diagnostic_only; never used for model or threshold selection",
    }


def _metrics(task_id: str, data: PartitionData, prediction: np.ndarray) -> dict[str, Any]:
    metrics = (
        _binary_metrics(data.targets, prediction)
        if FROZEN_WINNERS[task_id]["task_type"] == "binary"
        else _regression_metrics(data.targets, prediction)
    )
    if task_id == "T2":
        metrics["thickness_diagnostic"] = _thickness_diagnostic(data, prediction)
    if task_id == "T3":
        metrics["topk_diagnostic"] = _topk_diagnostic(data, prediction)
    return metrics


def _prediction_bytes(data: PartitionData, prediction: np.ndarray) -> bytes:
    text = io.StringIO(newline="")
    fields = ["sample_id", "group", data.auxiliary_name, "actual", "prediction"]
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for sample_id, group, auxiliary, actual, predicted in zip(
        data.sample_ids, data.groups, data.auxiliary, data.targets, prediction,
    ):
        writer.writerow({
            "sample_id": sample_id,
            "group": group,
            data.auxiliary_name: auxiliary,
            "actual": format(float(actual), ".17g"),
            "prediction": format(float(predicted), ".17g"),
        })
    return gzip.compress(text.getvalue().encode("utf-8"), compresslevel=9, mtime=0)


def _render_confirmation_figure(
    task_id: str,
    data: PartitionData,
    prediction: np.ndarray,
    metrics: Mapping[str, Any],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if FROZEN_WINNERS[task_id]["task_type"] == "regression":
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        limit = min(4000, len(prediction))
        indices = np.linspace(0, len(prediction) - 1, limit, dtype=int)
        axes[0].scatter(data.targets[indices], prediction[indices], s=7, alpha=0.35)
        low = float(min(np.min(data.targets[indices]), np.min(prediction[indices])))
        high = float(max(np.max(data.targets[indices]), np.max(prediction[indices])))
        axes[0].plot([low, high], [low, high], "k--", linewidth=1)
        axes[0].set(xlabel="Observed", ylabel="Predicted", title=f"{task_id} known holdout")
        if task_id == "T3":
            order = np.argsort(data.targets)
            axes[1].plot(np.arange(len(order)), data.targets[order], label="observed", linewidth=1)
            axes[1].plot(np.arange(len(order)), prediction[order], label="predicted", linewidth=1)
            axes[1].set(xlabel="Observed rank", ylabel="Oil volume", title="Ranking diagnostic")
            axes[1].legend()
        else:
            axes[1].hist(prediction - data.targets, bins=40, color="#4472C4", alpha=0.8)
            axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")
        figure.suptitle(
            f"Previously seen reusable holdout; MAE={metrics['mae']:.4g}, Spearman={metrics['spearman']}"
        )
    else:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(data.targets.astype(int), prediction)
        fraction, mean = calibration_curve(data.targets.astype(int), prediction, n_bins=8, strategy="quantile")
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(recall, precision, color="#4472C4")
        axes[0].set(xlabel="Recall", ylabel="Precision", title=f"AP={metrics['average_precision']:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1)
        axes[1].plot(mean, fraction, marker="o", color="#ED7D31")
        axes[1].set(xlabel="Mean probability", ylabel="Observed fraction", title=f"Brier={metrics['brier']:.4f}")
        suffix = ""
        if task_id == "T2":
            suffix = f"; thickness MAE={metrics['thickness_diagnostic']['net_thickness_mae_m']:.3g} m"
        figure.suptitle(f"{task_id} previously seen reusable holdout; F1@0.5={metrics['f1_at_0_5']:.4f}{suffix}")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _render_status_figure(task_id: str, status: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 2.8))
    axis.axis("off")
    axis.text(0.5, 0.68, f"{task_id}: {status['status']}", ha="center", va="center", fontsize=18, weight="bold")
    axis.text(0.5, 0.36, status["reason"], ha="center", va="center", fontsize=10, wrap=True)
    axis.text(0.5, 0.10, "Stage-4 data access: none", ha="center", va="center", fontsize=10, color="#9C0006")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _checkpoint_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _run_confirmed_target(
    audit: LabelMappingAudit,
    contract: Mapping[str, Any],
    task_id: str,
    target_dir: Path,
    *,
    source_root: Path | None,
    source_lock: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    # Phase A: development-only reconstruction and refit.  No holdout loader is
    # called until the checkpoint payload and config hash have been closed.
    development = _rebuild_partition(audit, task_id, "development", source_root=source_root)
    adapter, medians, payload, config, refit = _fit_frozen_winner(
        audit, task_id, development, source_lock,
    )
    config_hash = canonical_sha256(config)
    checkpoint = _checkpoint_bytes(payload)
    checkpoint_path = target_dir / "refit" / "model.pkl.gz"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(checkpoint)
    refit.update({
        "config_sha256": config_hash,
        "checkpoint_path": f"targets/{task_id}/refit/model.pkl.gz",
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_size_bytes": len(checkpoint),
    })
    _write_json(target_dir / "config.json", config)
    _write_json(target_dir / "refit.json", refit)

    # Phase B: explicitly authorized confirmation of an already-consumed P4
    # holdout.  Nothing from this phase can alter config, preprocessing or refit.
    holdout_started = time.monotonic()
    holdout = _rebuild_partition(audit, task_id, "known_holdout", source_root=source_root)
    task_spec = _stage4_task_spec(audit, task_id)
    prediction = _predict_holdout(adapter, medians, task_spec, holdout)
    metrics = _metrics(task_id, holdout, prediction)
    holdout_wall = time.monotonic() - holdout_started
    prediction_payload = _prediction_bytes(holdout, prediction)
    prediction_path = target_dir / "predictions.csv.gz"
    prediction_path.write_bytes(prediction_payload)
    metric_payload = {
        "schema_version": "sweetspot-p5-stage4-known-holdout-metrics/v1",
        "task_id": task_id,
        "status": "CONFIRMED_ON_KNOWN_HOLDOUT",
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "selection_feedback_allowed": False,
        "config_sha256": config_hash,
        "checkpoint_sha256": refit["checkpoint_sha256"],
        "holdout_samples": len(holdout.sample_ids),
        "holdout_groups": sorted(set(holdout.groups)),
        "holdout_ids_sha256": canonical_sha256(list(holdout.sample_ids)),
        "holdout_features_sha256": _array_sha256(holdout.features),
        "holdout_targets_sha256": _array_sha256(holdout.targets),
        "prediction_sha256": hashlib.sha256(prediction_payload).hexdigest(),
        "metrics": metrics,
        "holdout_wall_seconds": holdout_wall,
        "holdout_provenance": holdout.provenance,
        "prior_exposure_evidence": contract["prior_exposure"][task_id],
        "historical_p4_metrics_read": False,
    }
    _write_json(target_dir / "metrics.json", metric_payload)
    figure_path = target_dir / "figure.png"
    _render_confirmation_figure(task_id, holdout, prediction, metrics, figure_path)
    return {
        "schema_version": "sweetspot-p5-stage4-target/v1",
        "task_id": task_id,
        "lane": audit.target(task_id)["slug"],
        "status": "confirmed_known_holdout",
        "model_id": FROZEN_WINNERS[task_id]["model_id"],
        "update_steps": 64,
        "development_samples": len(development.sample_ids),
        "known_holdout_samples": len(holdout.sample_ids),
        "metrics": metrics,
        "config_sha256": config_hash,
        "checkpoint_sha256": refit["checkpoint_sha256"],
        "predictions_sha256": metric_payload["prediction_sha256"],
        "holdout_provenance_sha256": canonical_sha256(holdout.provenance),
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "test_accessed": True,
        "historical_p4_metrics_read": False,
        "artifact_dir": f"targets/{task_id}",
    }


def _blocked_status(audit: LabelMappingAudit, contract: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    target = audit.target(task_id)
    if task_id == "T5":
        status = "not_feasible"
        reason = "no approved label; no proxy or synthetic label is permitted"
        prior = False
    else:
        status = "blocked"
        reason = "no development-only feature source and no Stage-3 winner; test.h5 fallback is forbidden"
        prior = bool(contract["prior_exposure"][task_id]["prior_test_consumed"])
    return {
        "schema_version": "sweetspot-p5-stage4-status-gate/v1",
        "task_id": task_id,
        "lane": target["slug"],
        "target_name": target.get("target_name"),
        "label_version": target.get("label_version"),
        "status": status,
        "reason": reason,
        "stage3_winner": None,
        "development_feature_source_available": False,
        "prior_test_consumed": prior,
        "fresh_blind": False,
        "test_accessed": False,
        "historical_p4_metrics_read": False,
        "labels_generated": False,
        "checkpoint_created": False,
        "predictions_created": False,
    }


def _artifact_manifest(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "p5_stage4_manifest.json"):
        artifacts.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return {
        "schema_version": "sweetspot-p5-stage4-artifact-manifest/v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "all_paths_portable": all(not Path(row["path"]).is_absolute() for row in artifacts),
    }


def _verify_artifact_manifest(root: Path) -> str:
    manifest_path = Path(root) / "p5_stage4_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = _json(manifest_path)
    if manifest.get("schema_version") != "sweetspot-p5-stage4-artifact-manifest/v1":
        raise ValueError("existing Stage-4 artifact manifest schema changed")
    if manifest.get("artifact_count") != len(manifest.get("artifacts", ())):
        raise ValueError("existing Stage-4 artifact manifest count is invalid")
    for row in manifest["artifacts"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError("existing Stage-4 artifact path is not portable")
        path = Path(root) / relative
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"existing Stage-4 artifact missing or resized: {relative}")
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"existing Stage-4 artifact hash changed: {relative}")
    return manifest_sha256


def _write_top_level(
    staging: Path,
    results: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    started: float,
    execution_scope: str,
    refresh: Mapping[str, Any] | None = None,
) -> None:
    result_path = staging / "p5_stage4_results.jsonl"
    result_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in results),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "sweetspot-p5-stage4-summary/v1",
        "stage": "known_holdout_confirmation",
        "execution_scope": execution_scope,
        "baseline_commit": STAGE3_COMMIT,
        "root_seed": ROOT_SEED,
        "evidence_class": EVIDENCE_CLASS,
        "prior_test_consumed": True,
        "fresh_blind": False,
        "selection_feedback_allowed": False,
        "winner_target_ids": list(FROZEN_WINNERS),
        "status_only_target_ids": ["T5", "T6", "T7"],
        "target_status": {row["task_id"]: row["status"] for row in results},
        "development_samples": {
            row["task_id"]: row.get("development_samples") for row in results
        },
        "known_holdout_samples": {
            row["task_id"]: row.get("known_holdout_samples") for row in results
        },
        "contract": contract,
        "results_sha256": sha256_file(result_path),
        "wall_seconds": time.monotonic() - started,
        "historical_p4_metrics_read": False,
        "p4_hpo_called": False,
        "labels_generated": False,
        "T6_T7_test_h5_fallback": False,
        "refresh": refresh,
    }
    _write_json(staging / "p5_stage4_summary.json", summary)
    _write_json(staging / "p5_stage4_manifest.json", _artifact_manifest(staging))


def run_stage4(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    source_root: Path | None = None,
) -> dict[str, Any]:
    output = _portable_output_dir(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Stage-4 artifacts: {output}")
    audit = validate_label_mapping(mapping_path)
    contract = validate_stage4_contract(audit)
    source_lock = load_source_lock(DEFAULT_LOCK)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".stage4-confirmation-", dir=output.parent))
    started = time.monotonic()
    try:
        results = []
        for task_id in FROZEN_WINNERS:
            results.append(_run_confirmed_target(
                audit, contract, task_id, staging / "targets" / task_id,
                source_root=source_root, source_lock=source_lock,
            ))
        for task_id in ("T5", "T6", "T7"):
            status = _blocked_status(audit, contract, task_id)
            target_dir = staging / "targets" / task_id
            _write_json(target_dir / "status.json", status)
            _render_status_figure(task_id, status, target_dir / "figure.png")
            results.append(status)
        _write_top_level(
            staging, results, contract, started=started, execution_scope="full_T1_to_T7",
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": output.relative_to(HERE).as_posix(),
        "summary_path": (output / "p5_stage4_summary.json").relative_to(HERE).as_posix(),
        "manifest_path": (output / "p5_stage4_manifest.json").relative_to(HERE).as_posix(),
        "manifest_sha256": sha256_file(output / "p5_stage4_manifest.json"),
        "target_status": {row["task_id"]: row["status"] for row in results},
        "wall_seconds": time.monotonic() - started,
    }


def refresh_t1_t2_provenance(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically rerun only T1/T2 after verifying every existing artifact.

    This maintenance path exists for metadata corrections requested after an
    archived run.  It cannot change the frozen winner/config/threshold and it
    preserves T3--T7 byte-for-byte while rebuilding top-level hashes.
    """
    output = _portable_output_dir(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"Stage-4 output does not exist: {output}")
    previous_manifest_sha256 = _verify_artifact_manifest(output)
    existing_results = [
        json.loads(line)
        for line in (output / "p5_stage4_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    existing_by_task = {row["task_id"]: row for row in existing_results}
    if set(existing_by_task) != {f"T{index}" for index in range(1, 8)}:
        raise ValueError("existing Stage-4 results do not contain exactly T1--T7")
    audit = validate_label_mapping(mapping_path)
    contract = validate_stage4_contract(audit)
    source_lock = load_source_lock(DEFAULT_LOCK)
    staging = Path(tempfile.mkdtemp(prefix=".stage4-refresh-", dir=output.parent))
    started = time.monotonic()
    backup = output.with_name(f".{output.name}.previous-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(staging, ignore_errors=True)
        raise FileExistsError(f"refusing refresh with stale backup: {backup}")
    try:
        shutil.copytree(output, staging, dirs_exist_ok=True)
        refreshed: dict[str, Mapping[str, Any]] = {}
        for task_id in ("T1", "T2"):
            target_dir = staging / "targets" / task_id
            shutil.rmtree(target_dir)
            refreshed[task_id] = _run_confirmed_target(
                audit, contract, task_id, target_dir,
                source_root=source_root, source_lock=source_lock,
            )
        results = [
            refreshed.get(f"T{index}", existing_by_task[f"T{index}"])
            for index in range(1, 8)
        ]
        _write_top_level(
            staging,
            results,
            contract,
            started=started,
            execution_scope="targeted_T1_T2_provenance_refresh",
            refresh={
                "reason": "correct source_kind wording to match authorized known-holdout members",
                "rerun_target_ids": ["T1", "T2"],
                "preserved_target_ids": ["T3", "T4", "T5", "T6", "T7"],
                "previous_manifest_sha256": previous_manifest_sha256,
                "model_config_changed": False,
                "metric_algorithm_changed": False,
            },
        )
        os.replace(output, backup)
        try:
            os.replace(staging, output)
        except BaseException:
            os.replace(backup, output)
            raise
        shutil.rmtree(backup)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_dir": output.relative_to(HERE).as_posix(),
        "manifest_path": (output / "p5_stage4_manifest.json").relative_to(HERE).as_posix(),
        "previous_manifest_sha256": previous_manifest_sha256,
        "manifest_sha256": sha256_file(output / "p5_stage4_manifest.json"),
        "rerun_target_ids": ["T1", "T2"],
        "preserved_target_ids": ["T3", "T4", "T5", "T6", "T7"],
        "wall_seconds": time.monotonic() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-known-holdout",
        action="store_true",
        help="required acknowledgement that P4 already consumed this holdout",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--refresh-t1-t2",
        action="store_true",
        help="verify the existing archive, rerun T1/T2, and atomically rebuild summary/manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.confirm_known_holdout:
        print(
            json.dumps({
                "status": "REFUSED",
                "reason": "--confirm-known-holdout is required; this is not a fresh-blind test",
                "prior_test_consumed": True,
                "fresh_blind": False,
                "output_created": False,
            }, sort_keys=True),
            file=os.sys.stderr,
        )
        return 2
    if args.refresh_t1_t2:
        result = refresh_t1_t2_provenance(args.output_dir, source_root=args.source_root)
    else:
        result = run_stage4(args.output_dir, source_root=args.source_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
