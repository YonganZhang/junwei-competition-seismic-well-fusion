"""Sweetspot-private P5 Stage-3 frozen-fold, three-seed confirmation.

Only the four P4 development-rebuildable targets enter the 117-cell matrix.
The module has no frozen-test loader or test-selection argument.  It reuses the
Stage-2 adapters, update limits, input limits, transforms, and preprocessing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _models.sweetspot.p5_common import AdapterSkip

from .source_lock import DEFAULT_LOCK, inspect_runtime, load_source_lock
from .sweetspot_p5_stage2 import (
    CPU_WALL_LIMIT_SECONDS,
    NEURAL_UPDATE_LIMIT,
    NEURAL_WALL_LIMIT_SECONDS,
    TREE_MODELS,
    TREE_UPDATE_LIMIT,
    _metric_payload,
    _run_inceptiontime_pilot,
    _run_tree_pilot,
    _tree_config,
    _worst_group_metrics,
)
from .sweetspot_p5_stage2_data import (
    TRAIN_SAMPLE_LIMIT,
    VALIDATION_SAMPLE_LIMIT,
    DevelopmentDataUnavailable,
    DevelopmentPilotData,
    load_development_pilot_data,
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


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "stage3_cv"
RESULT_FILENAME = "p5_stage3_results.jsonl"
SUMMARY_FILENAME = "p5_stage3_summary.json"
OOF_MANIFEST_FILENAME = "p5_stage3_oof_manifest.json"
VISUALIZATION_MANIFEST_FILENAME = "p5_stage3_visualization_manifest.json"
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
RANKABLE_COMPLETION_THRESHOLD = 0.80
BOOTSTRAP_ITERATIONS = 2000
BASELINE_COMMIT = "16bebd18a0bc722afcbc4b841610bf76ce9503e4"
PROTOCOL_PATH = PROJECT_ROOT / "_wiki-methodology" / "_top" / "_phases" / "P5_stage3_multiseed_cv.md"

# Frozen by P5_stage3_multiseed_cv.md.  Order is scientific preregistration,
# never a runtime score-derived selection.
FROZEN_TASKS: Mapping[str, Mapping[str, Any]] = {
    "T1": {
        "models": ("lightgbm", "catboost", "xgboost"),
        "folds": (0, 1, 2),
        "primary_metric": "mae",
        "direction": "minimize",
        "task_type": "regression",
    },
    "T2": {
        "models": ("catboost", "xgboost", "lightgbm"),
        "folds": (0, 1, 2),
        "primary_metric": "average_precision",
        "direction": "maximize",
        "task_type": "binary",
    },
    "T3": {
        "models": ("lightgbm", "inceptiontime", "xgboost"),
        "folds": (0, 1, 2, 3),
        "primary_metric": "mae",
        "direction": "minimize",
        "task_type": "regression",
    },
    "T4": {
        "models": ("catboost", "lightgbm", "inceptiontime"),
        "folds": (0, 1, 2),
        "primary_metric": "average_precision",
        "direction": "maximize",
        "task_type": "binary",
    },
}
STATUS_ONLY_TARGETS = ("T5", "T6", "T7")
EXPECTED_CELLS = sum(
    len(contract["models"]) * len(contract["folds"]) * len(REPEAT_SEEDS)
    for contract in FROZEN_TASKS.values()
)


@dataclass(frozen=True)
class PredictionRecord:
    cell_key: str
    task_id: str
    lane: str
    model_id: str
    fold_id: int
    repeat_id: int
    seed: int
    sample_ids: tuple[str, ...]
    groups: tuple[str, ...]
    actual: np.ndarray
    prediction: np.ndarray


def _portable_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as exc:
        raise PermissionError("Stage-3 outputs must stay in the sweetspot-private p5 directory") from exc
    return resolved


def expected_cell_keys() -> tuple[tuple[str, str, int, int, int], ...]:
    return tuple(
        (task_id, model_id, int(fold_id), repeat_id, int(seed))
        for task_id, contract in FROZEN_TASKS.items()
        for model_id in contract["models"]
        for fold_id in contract["folds"]
        for repeat_id, seed in enumerate(REPEAT_SEEDS)
    )


def validate_stage3_contract(audit: LabelMappingAudit) -> dict[str, Any]:
    """Validate every development fold without using test IDs for execution."""
    if EXPECTED_CELLS != 117 or len(set(expected_cell_keys())) != EXPECTED_CELLS:
        raise ValueError("sweetspot Stage-3 matrix must contain 117 unique legal cells")
    split_evidence: dict[str, Any] = {}
    for task_id, contract in FROZEN_TASKS.items():
        target = audit.target(task_id)
        if target["status"] != "approved_for_development_pilot":
            raise PermissionError(f"{task_id}: frozen Stage-3 target is not development-approved")
        if target["task_type"] != contract["task_type"]:
            raise ValueError(f"{task_id}: task type changed after Stage-2")
        if target["primary_metric"] != contract["primary_metric"]:
            raise ValueError(f"{task_id}: primary metric changed after Stage-2")
        if target["primary_metric_direction"] != contract["direction"]:
            raise ValueError(f"{task_id}: metric direction changed after Stage-2")
        split = audit.split_manifest(task_id)
        folds = split.get("folds", ())
        observed_fold_ids = tuple(int(item["fold_id"]) for item in folds)
        if observed_fold_ids != tuple(contract["folds"]):
            raise ValueError(f"{task_id}: P4 fold IDs changed: {observed_fold_ids}")
        development_groups = set(split.get("development_groups", ()))
        development_ids = set(split.get("development_sample_ids", ()))
        fold_rows = []
        for fold in folds:
            train_groups = set(fold.get("train_groups", ()))
            validation_groups = set(fold.get("validation_groups", ()))
            train_ids = set(fold.get("train_sample_ids", ()))
            validation_ids = set(fold.get("validation_sample_ids", ()))
            if not train_groups or not validation_groups or train_groups & validation_groups:
                raise ValueError(f"{task_id}/fold_{fold['fold_id']}: invalid group isolation")
            if not (train_groups | validation_groups) <= development_groups:
                raise ValueError(f"{task_id}/fold_{fold['fold_id']}: group outside development set")
            if not train_ids or not validation_ids or train_ids & validation_ids:
                raise ValueError(f"{task_id}/fold_{fold['fold_id']}: invalid sample isolation")
            if not (train_ids | validation_ids) <= development_ids:
                raise ValueError(f"{task_id}/fold_{fold['fold_id']}: ID outside development set")
            fold_rows.append({
                "fold_id": int(fold["fold_id"]),
                "train_groups": sorted(train_groups),
                "validation_groups": sorted(validation_groups),
                "train_sample_count": len(train_ids),
                "validation_sample_count": len(validation_ids),
            })
        split_evidence[task_id] = {
            "manifest_path": target["split_manifest"]["path"],
            "manifest_sha256": target["split_manifest"]["sha256"],
            "development_group_count": len(development_groups),
            "development_sample_count": len(development_ids),
            "folds": fold_rows,
            "execution_fields": ["development_groups", "development_sample_ids", "folds"],
            "test_partition_used": False,
        }
    if audit.target("T5")["status"] != "not_feasible":
        raise ValueError("T5 must remain not_feasible")
    if audit.target("T6")["target_name"] != "PHIF" or audit.target("T7")["target_name"] != "KLOGH":
        raise ValueError("T6 PHIF and T7 KLOGH heads must remain independent")
    return split_evidence


def _stage3_task_spec(audit: LabelMappingAudit, task_id: str, fold_id: int) -> Any:
    stage2 = build_pilot_task_spec(audit, task_id)
    metadata = dict(stage2.metadata)
    metadata.update({
        "p5_stage": "frozen_multiseed_development_cv",
        "p5_fold_id": int(fold_id),
        "preprocessing_fit_scope": "fold_train_only",
        "class_weight_fit_scope": "fold_train_only_if_defined",
        "target_transform_fit_scope": "fold_train_only",
        "calibration_fit_scope": "fold_train_only_if_defined",
        "test_access": "forbidden",
    })
    return replace(
        stage2,
        task_id=f"sweetspot.p5.stage3.{task_id.lower()}.{audit.target(task_id)['slug']}",
        hpo={
            "stage": "P5 Stage-3 frozen multiseed development CV",
            "optimization": "forbidden",
            "fold_id": int(fold_id),
            "repeat_seeds": list(REPEAT_SEEDS),
            "selection_scope": "P4 manifest development folds only",
            "test_access": "forbidden",
        },
        metadata=metadata,
    )


def _candidate_configuration(model_id: str, data: DevelopmentPilotData, seed: int) -> dict[str, Any]:
    if model_id in TREE_MODELS:
        payload = {
            "adapter": "Stage-2 tree adapter",
            "estimator_kwargs": _tree_config(model_id, seed),
            "update_steps": TREE_UPDATE_LIMIT,
            "input_modality": "tabular",
        }
    elif model_id == "inceptiontime":
        if data.train_sequence is None:
            raise AdapterSkip("input_modality_missing", "development sequence is unavailable")
        payload = {
            "adapter": "Stage-2 InceptionTime adapter",
            "c_in": int(data.train_sequence.shape[1]),
            "seq_len": int(data.train_sequence.shape[2]),
            "nf": 8,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "update_steps": NEURAL_UPDATE_LIMIT,
            "input_modality": "sequence",
        }
    else:
        raise KeyError(model_id)
    return {"payload": payload, "sha256": canonical_sha256(payload)}


def _cell_key(task_id: str, model_id: str, fold_id: int, repeat_id: int, seed: int) -> str:
    return canonical_sha256({
        "task_id": task_id,
        "model_id": model_id,
        "fold_id": int(fold_id),
        "repeat_id": int(repeat_id),
        "seed": int(seed),
    })


def _base_result(
    audit: LabelMappingAudit,
    task_id: str,
    model_id: str,
    fold_id: int,
    repeat_id: int,
    seed: int,
    data: DevelopmentPilotData | None,
) -> dict[str, Any]:
    target = audit.target(task_id)
    key = _cell_key(task_id, model_id, fold_id, repeat_id, seed)
    return {
        "schema_version": "sweetspot-p5-stage3-cell/v1",
        "cell_key": key,
        "task_id": task_id,
        "lane": target["slug"],
        "model_id": model_id,
        "fold_id": int(fold_id),
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "label_status": target["status"],
        "p4_status": target["p4_status"],
        "label_version": target["label_version"],
        "target_name": target["target_name"],
        "is_proxy": target["is_proxy"],
        "proxy_semantics": target["proxy_semantics"],
        "split_sha256": target["split_manifest"]["sha256"],
        "input_budget": None if data is None else data.budget,
        "candidate_configuration": None,
        "fit_scope": {
            "preprocessing": "fold_train_only",
            "class_weights": "not_defined_by_stage2",
            "target_transform": "fold_train_only_or_fixed_formula",
            "calibration": "not_defined_by_stage2",
        },
        "status": None,
        "reason": None,
        "validation_metrics": None,
        "worst_group": None,
        "resource": None,
        "development_provenance": None if data is None else data.provenance,
        "label_generated": False,
        "checkpoint_persisted": False,
        "test_firewall": {
            "test_loader_api_present": False,
            "frozen_test_files_opened": 0,
            "test_accessed": False,
            "historical_test_metrics_used": False,
            "stage2_metrics_used_for_selection": False,
        },
        "label_mapping_sha256": audit.mapping_sha256,
    }


def _failure(
    result: dict[str, Any],
    status: str,
    reason_code: str,
    detail: str,
) -> dict[str, Any]:
    result.update({
        "status": status,
        "reason": {"code": reason_code, "detail": detail},
        "validation_metrics": None,
    })
    return result


def _run_cell(
    audit: LabelMappingAudit,
    source_lock: Mapping[str, Mapping[str, Any]],
    task_id: str,
    model_id: str,
    fold_id: int,
    repeat_id: int,
    seed: int,
    data: DevelopmentPilotData | None,
    data_error: DevelopmentDataUnavailable | None,
    *,
    device: str,
    gpu_lock_path: Path | None,
) -> tuple[dict[str, Any], PredictionRecord | None]:
    result = _base_result(audit, task_id, model_id, fold_id, repeat_id, seed, data)
    if data_error is not None:
        return _failure(result, "SKIP", data_error.reason_code, data_error.detail), None
    if data is None:
        return _failure(result, "SKIP", "development_data_missing", "no development fold object"), None
    runtime = inspect_runtime(source_lock[model_id])
    result["source_lock"] = {
        "revision": source_lock[model_id]["revision"],
        "license": source_lock[model_id]["license"],
        "runtime": runtime,
    }
    if not runtime["available"]:
        return _failure(result, "SKIP", str(runtime["reason_code"]), "source-locked runtime unavailable"), None
    if not runtime["version_allowed"]:
        return _failure(result, "SKIP", "runtime_version_not_locked", json.dumps(runtime, sort_keys=True)), None
    try:
        configuration = _candidate_configuration(model_id, data, seed)
        result["candidate_configuration"] = configuration
        task_spec = _stage3_task_spec(audit, task_id, fold_id)
        if model_id in TREE_MODELS:
            execution = _run_tree_pilot(model_id, task_spec, data, seed)
        elif model_id == "inceptiontime":
            execution = _run_inceptiontime_pilot(
                task_spec, data, seed, device=device, gpu_lock_path=gpu_lock_path,
            )
            execution["pilot"]["device"] = "cuda:0" if device == "cuda" else "cpu"
            if device == "cuda":
                execution["pilot"]["gpu_lock"]["mechanism"] = "flock"
                execution["pilot"]["gpu_lock"]["device"] = "cuda:0"
        else:
            raise KeyError(model_id)
    except AdapterSkip as exc:
        return _failure(result, "SKIP", exc.reason_code, exc.detail), None
    prediction = np.asarray(execution.pop("prediction"), dtype=np.float64).reshape(-1)
    if prediction.shape != np.asarray(data.validation_target).reshape(-1).shape:
        raise ValueError(f"{result['cell_key']}: prediction shape mismatch")
    metrics = _metric_payload(task_spec.task_type, data.validation_target, prediction)
    target = audit.target(task_id)
    worst = _worst_group_metrics(
        task_spec.task_type,
        data.validation_groups,
        data.validation_target,
        prediction,
        str(target["primary_metric"]),
        str(target["primary_metric_direction"]),
    )
    result.update({
        "status": "PASS",
        "reason": None,
        "estimator_head": task_spec.targets[0],
        "validation_metrics": metrics,
        "worst_group": worst,
        "resource": execution["pilot"],
        "tiny_gate": execution["tiny_gate"],
    })
    prediction_record = PredictionRecord(
        cell_key=result["cell_key"],
        task_id=task_id,
        lane=str(target["slug"]),
        model_id=model_id,
        fold_id=int(fold_id),
        repeat_id=int(repeat_id),
        seed=int(seed),
        sample_ids=data.validation_sample_ids,
        groups=data.validation_groups,
        actual=np.asarray(data.validation_target, dtype=np.float64).reshape(-1),
        prediction=prediction,
    )
    return result, prediction_record


def run_stage3(
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    source_root: Path | None = None,
    device: str = "cuda",
    gpu_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[PredictionRecord]]:
    if device == "cuda" and gpu_lock_path is None:
        raise ValueError("device=cuda requires the protocol's explicit shared GPU lock")
    audit = validate_label_mapping(mapping_path)
    split_evidence = validate_stage3_contract(audit)
    source_lock = load_source_lock()
    required_models = {model for contract in FROZEN_TASKS.values() for model in contract["models"]}
    if not required_models <= set(source_lock):
        raise ValueError("frozen Stage-3 model is absent from the Stage-2 source lock")

    results: list[dict[str, Any]] = []
    predictions: list[PredictionRecord] = []
    data_by_fold: dict[tuple[str, int], DevelopmentPilotData | None] = {}
    errors: dict[tuple[str, int], DevelopmentDataUnavailable | None] = {}
    for task_id, contract in FROZEN_TASKS.items():
        for fold_id in contract["folds"]:
            key = (task_id, int(fold_id))
            try:
                data_by_fold[key] = load_development_pilot_data(
                    audit, task_id, source_root=source_root, fold_id=int(fold_id),
                )
                errors[key] = None
            except DevelopmentDataUnavailable as exc:
                data_by_fold[key] = None
                errors[key] = exc

    for task_id, contract in FROZEN_TASKS.items():
        for model_id in contract["models"]:
            for fold_id in contract["folds"]:
                key = (task_id, int(fold_id))
                for repeat_id, seed in enumerate(REPEAT_SEEDS):
                    try:
                        result, prediction = _run_cell(
                            audit,
                            source_lock,
                            task_id,
                            model_id,
                            int(fold_id),
                            repeat_id,
                            int(seed),
                            data_by_fold[key],
                            errors[key],
                            device=device,
                            gpu_lock_path=gpu_lock_path,
                        )
                    except TimeoutError as exc:
                        result = _base_result(
                            audit, task_id, model_id, int(fold_id), repeat_id, int(seed), data_by_fold[key],
                        )
                        result = _failure(result, "TIMEOUT", "cell_wall_limit_exceeded", str(exc))
                        prediction = None
                    except Exception as exc:  # preserve every attempted cell; never invent a number
                        result = _base_result(
                            audit, task_id, model_id, int(fold_id), repeat_id, int(seed), data_by_fold[key],
                        )
                        result = _failure(
                            result,
                            "FAILED",
                            "unexpected_runtime_failure",
                            f"{type(exc).__name__}: {exc}",
                        )
                        prediction = None
                    results.append(result)
                    if prediction is not None:
                        predictions.append(prediction)

    observed = {
        (row["task_id"], row["model_id"], row["fold_id"], row["repeat_id"], row["seed"])
        for row in results
    }
    if len(results) != EXPECTED_CELLS or observed != set(expected_cell_keys()):
        raise RuntimeError("Stage-3 execution did not preserve the frozen 117-cell matrix")
    summary_context = {
        "audit": audit,
        "split_evidence": split_evidence,
        "source_lock": source_lock,
        "data_by_fold": data_by_fold,
        "errors": errors,
        "device": device,
    }
    return results, summary_context, predictions


def _primary_value(row: Mapping[str, Any], metric: str) -> float | None:
    if row.get("status") != "PASS" or not isinstance(row.get("validation_metrics"), Mapping):
        return None
    value = row["validation_metrics"].get(metric)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _bootstrap_ci(values: Sequence[float], *, token: str) -> list[float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [None, None]
    seed = int.from_bytes(hashlib.sha256(f"2693|{token}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(BOOTSTRAP_ITERATIONS, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return [float(low), float(high)]


def _leaderboard(
    task_id: str,
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = FROZEN_TASKS[task_id]
    metric = str(contract["primary_metric"])
    direction = str(contract["direction"])
    rows = [row for row in results if row["task_id"] == task_id]
    expected_task = len(contract["models"]) * len(contract["folds"]) * len(REPEAT_SEEDS)
    valid_task = sum(_primary_value(row, metric) is not None for row in rows)
    task_completion = valid_task / expected_task
    entries: list[dict[str, Any]] = []
    for model_id in contract["models"]:
        model_rows = [row for row in rows if row["model_id"] == model_id]
        values = [
            value for row in model_rows if (value := _primary_value(row, metric)) is not None
        ]
        expected_model = len(contract["folds"]) * len(REPEAT_SEEDS)
        fold_values: dict[str, float] = {}
        for fold_id in contract["folds"]:
            fold_metric = [
                value
                for row in model_rows
                if row["fold_id"] == fold_id
                and (value := _primary_value(row, metric)) is not None
            ]
            if fold_metric:
                fold_values[str(fold_id)] = float(np.mean(fold_metric))
        seed_values: dict[str, float] = {}
        for seed in REPEAT_SEEDS:
            seed_metric = [
                value
                for row in model_rows
                if row["seed"] == seed
                and (value := _primary_value(row, metric)) is not None
            ]
            if seed_metric:
                seed_values[str(seed)] = float(np.mean(seed_metric))
        worst_fold_id = None
        worst_fold_value = None
        if fold_values:
            chooser = max if direction == "minimize" else min
            worst_fold_id, worst_fold_value = chooser(
                fold_values.items(), key=lambda item: float(item[1]),
            )
        resources = [row["resource"] for row in model_rows if isinstance(row.get("resource"), Mapping)]
        entry = {
            "model_id": model_id,
            "expected_cells": expected_model,
            "completed_cells": len(values),
            "completion_rate": len(values) / expected_model,
            "eligible_for_ranking": len(values) / expected_model >= RANKABLE_COMPLETION_THRESHOLD,
            "primary_mean": float(np.mean(values)) if values else None,
            "primary_bootstrap_95ci": _bootstrap_ci(values, token=f"{task_id}|{model_id}"),
            "fold_means": fold_values,
            "worst_fold": {"fold_id": None if worst_fold_id is None else int(worst_fold_id), "value": worst_fold_value},
            "seed_means": seed_values,
            "seed_std": float(np.std(list(seed_values.values()), ddof=0)) if seed_values else None,
            "resource": {
                "mean_wall_seconds": float(np.mean([item["wall_seconds"] for item in resources])) if resources else None,
                "max_peak_rss_bytes": max((int(item["peak_rss_bytes"]) for item in resources), default=None),
                "max_peak_vram_bytes": max((int(item["peak_vram_bytes"]) for item in resources), default=None),
                "download_bytes": sum(int(item.get("download_bytes", 0)) for item in resources),
            },
        }
        entries.append(entry)

    task_rankable = task_completion >= RANKABLE_COMPLETION_THRESHOLD
    sortable = [entry for entry in entries if entry["eligible_for_ranking"] and entry["primary_mean"] is not None]
    sign = 1.0 if direction == "minimize" else -1.0
    sortable.sort(key=lambda entry: (
        sign * float(entry["primary_mean"]),
        sign * float(entry["worst_fold"]["value"]),
        float(entry["seed_std"]),
        float(entry["resource"]["mean_wall_seconds"]),
        entry["model_id"],
    ))
    previous_key: tuple[float, ...] | None = None
    previous_rank = 0
    for position, entry in enumerate(sortable, 1):
        scientific_key = (
            float(entry["primary_mean"]),
            float(entry["worst_fold"]["value"]),
            float(entry["seed_std"]),
            float(entry["resource"]["mean_wall_seconds"]),
        )
        tied = previous_key is not None and all(
            math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(scientific_key, previous_key)
        )
        entry["rank"] = previous_rank if tied else position
        entry["tied"] = tied
        previous_rank = int(entry["rank"])
        previous_key = scientific_key
    by_model = {entry["model_id"]: entry for entry in entries}
    ordered_entries = sortable + [entry for entry in entries if entry not in sortable]
    if set(by_model) != set(contract["models"]):
        raise RuntimeError(f"{task_id}: leaderboard model pollution")
    return {
        "schema_version": "sweetspot-p5-stage3-leaderboard/v1",
        "task_id": task_id,
        "lane": str(rows[0]["lane"]),
        "status": "rankable" if task_rankable else "not_rankable",
        "not_rankable_reason": None if task_rankable else "legal completion rate below 80%",
        "primary_metric": metric,
        "direction": direction,
        "expected_cells": expected_task,
        "completed_cells": valid_task,
        "completion_rate": task_completion,
        "ranking_rule": ["primary_mean", "worst_fold", "seed_std", "mean_wall_seconds"],
        "tie_policy": "equal scientific keys remain tied; runtime order is not a tiebreaker",
        "bootstrap": {"iterations": BOOTSTRAP_ITERATIONS, "unit": "fold-repeat cell", "root_seed": 2693},
        "entries": ordered_entries,
        "test_accessed": False,
    }


def _sample_scatter(records: Sequence[PredictionRecord], limit_per_model: int = 600) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id in sorted({record.model_id for record in records}):
        candidates: list[tuple[str, dict[str, Any]]] = []
        for record in records:
            if record.model_id != model_id:
                continue
            for index, (sample_id, group, actual, prediction) in enumerate(zip(
                record.sample_ids, record.groups, record.actual, record.prediction,
            )):
                point = {
                    "model_id": model_id,
                    "fold_id": record.fold_id,
                    "seed": record.seed,
                    "group": group,
                    "actual": float(actual),
                    "prediction": float(prediction),
                }
                token = hashlib.sha256(
                    f"{record.cell_key}|{sample_id}|{index}".encode("utf-8")
                ).hexdigest()
                candidates.append((token, point))
        output.extend(point for _, point in sorted(candidates)[:limit_per_model])
    return output


def _group_error(records: Sequence[PredictionRecord], task_type: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    models = sorted({record.model_id for record in records})
    for model_id in models:
        model_records = [record for record in records if record.model_id == model_id]
        groups = sorted({group for record in model_records for group in record.groups})
        for group in groups:
            actual_parts = []
            prediction_parts = []
            for record in model_records:
                mask = np.asarray(record.groups) == group
                actual_parts.append(record.actual[mask])
                prediction_parts.append(record.prediction[mask])
            actual = np.concatenate(actual_parts)
            prediction = np.concatenate(prediction_parts)
            metrics = _metric_payload(task_type, actual, prediction)
            output.append({
                "model_id": model_id,
                "group": group,
                "count": int(len(actual)),
                "metrics": metrics,
            })
    return output


def _downsample_curve(values: np.ndarray, limit: int = 200) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= limit:
        return [float(value) for value in array]
    indices = np.unique(np.linspace(0, len(array) - 1, limit).astype(int))
    return [float(array[index]) for index in indices]


def _visualization_payload(
    task_id: str,
    records: Sequence[PredictionRecord],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = FROZEN_TASKS[task_id]
    task_records = [record for record in records if record.task_id == task_id]
    payload: dict[str, Any] = {
        "schema_version": "sweetspot-p5-stage3-visualization-data/v1",
        "task_id": task_id,
        "task_type": contract["task_type"],
        "models": list(contract["models"]),
        "folds": list(contract["folds"]),
        "repeat_seeds": list(REPEAT_SEEDS),
        "primary_metric": contract["primary_metric"],
        "scatter_sample_policy": "sha256-stable maximum 600 points per model",
        "scatter": _sample_scatter(task_records),
        "group_error": _group_error(task_records, str(contract["task_type"])),
        "cell_metrics": [
            {
                "model_id": row["model_id"],
                "fold_id": row["fold_id"],
                "seed": row["seed"],
                "status": row["status"],
                "primary_value": _primary_value(row, str(contract["primary_metric"])),
            }
            for row in results if row["task_id"] == task_id
        ],
        "test_accessed": False,
        "historical_test_metrics_used": False,
    }
    if contract["task_type"] == "binary":
        from sklearn.metrics import precision_recall_curve

        curves = []
        calibration = []
        for model_id in contract["models"]:
            model_records = [record for record in task_records if record.model_id == model_id]
            actual = np.concatenate([record.actual for record in model_records]).astype(int)
            prediction = np.concatenate([record.prediction for record in model_records])
            precision, recall, _ = precision_recall_curve(actual, prediction)
            curves.append({
                "model_id": model_id,
                "precision": _downsample_curve(precision),
                "recall": _downsample_curve(recall),
            })
            bins = np.linspace(0.0, 1.0, 11)
            bin_ids = np.minimum(np.digitize(prediction, bins[1:-1]), 9)
            rows = []
            for bin_id in range(10):
                mask = bin_ids == bin_id
                if mask.any():
                    rows.append({
                        "bin": bin_id,
                        "count": int(mask.sum()),
                        "mean_probability": float(prediction[mask].mean()),
                        "positive_fraction": float(actual[mask].mean()),
                    })
            calibration.append({"model_id": model_id, "bins": rows})
        payload["precision_recall"] = curves
        payload["calibration"] = calibration
    return payload


def _status_payload(audit: LabelMappingAudit, task_id: str) -> dict[str, Any]:
    target = audit.target(task_id)
    if task_id == "T5":
        status = "not_feasible"
        reason = target["proxy_semantics"]
    else:
        status = "blocked"
        reason = (
            "P4 label and split are frozen, but no development-only feature source exists; "
            "materialized test fallback is forbidden"
        )
    return {
        "schema_version": "sweetspot-p5-stage3-status-gate/v1",
        "task_id": task_id,
        "lane": target["slug"],
        "status": status,
        "reason": reason,
        "label_version": target.get("label_version"),
        "target_name": target.get("target_name"),
        "is_proxy": target["is_proxy"],
        "proxy_semantics": target["proxy_semantics"],
        "development_feature_source_available": False,
        "expected_training_cells": 0,
        "test_accessed": False,
        "label_generated": False,
    }


def _save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight", metadata={"Software": "sweetspot-p5-stage3"})
    import matplotlib.pyplot as plt

    plt.close(figure)


def rebuild_figures(visualization_data_dir: Path, figure_dir: Path) -> list[Path]:
    """Rebuild every figure from portable aggregates; raw/test data are not read."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_dir = Path(visualization_data_dir)
    destination = Path(figure_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    colors = {"lightgbm": "#2a9d8f", "catboost": "#e76f51", "xgboost": "#457b9d", "inceptiontime": "#8e5ea2"}

    for task_id, contract in FROZEN_TASKS.items():
        payload = json.loads((data_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        models = list(contract["models"])
        if contract["task_type"] == "regression":
            figure, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4.0), sharex=True, sharey=True)
            for axis, model_id in zip(np.atleast_1d(axes), models):
                points = [row for row in payload["scatter"] if row["model_id"] == model_id]
                actual = np.asarray([row["actual"] for row in points])
                prediction = np.asarray([row["prediction"] for row in points])
                axis.scatter(actual, prediction, s=7, alpha=0.28, color=colors[model_id])
                if len(actual):
                    low = float(min(actual.min(), prediction.min()))
                    high = float(max(actual.max(), prediction.max()))
                    axis.plot([low, high], [low, high], "k--", linewidth=0.8)
                axis.set_title(model_id)
                axis.set_xlabel("P4 development target")
            np.atleast_1d(axes)[0].set_ylabel("Stage-3 OOF prediction")
            figure.suptitle(f"Sweetspot {task_id}: development OOF regression")
            path = destination / f"{task_id}_regression_scatter.png"
            _save_figure(figure, path)
            written.append(path)
        else:
            figure, axes = plt.subplots(1, 2, figsize=(9, 4))
            for row in payload["precision_recall"]:
                axes[0].plot(row["recall"], row["precision"], label=row["model_id"], color=colors[row["model_id"]])
            axes[0].set(xlabel="Recall", ylabel="Precision", title=f"{task_id} development PR")
            axes[0].legend(fontsize=8)
            axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
            for row in payload["calibration"]:
                axes[1].plot(
                    [item["mean_probability"] for item in row["bins"]],
                    [item["positive_fraction"] for item in row["bins"]],
                    marker="o", label=row["model_id"], color=colors[row["model_id"]],
                )
            axes[1].set(xlabel="Mean probability", ylabel="Positive fraction", title=f"{task_id} development calibration")
            axes[1].legend(fontsize=8)
            path = destination / f"{task_id}_pr_calibration.png"
            _save_figure(figure, path)
            written.append(path)

        group_rows = payload["group_error"]
        metric = "mae" if contract["task_type"] == "regression" else "brier"
        labels = sorted({row["group"] for row in group_rows})
        x = np.arange(len(labels), dtype=float)
        width = 0.8 / len(models)
        figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.5), 4))
        for model_index, model_id in enumerate(models):
            by_group = {row["group"]: row["metrics"].get(metric) for row in group_rows if row["model_id"] == model_id}
            axis.bar(
                x + (model_index - (len(models) - 1) / 2) * width,
                [by_group.get(label, np.nan) for label in labels],
                width=width,
                label=model_id,
                color=colors[model_id],
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylabel(metric.upper())
        axis.set_title(f"Sweetspot {task_id}: development well-group error")
        axis.legend(fontsize=8)
        path = destination / f"{task_id}_well_group_error.png"
        _save_figure(figure, path)
        written.append(path)

        figure, axis = plt.subplots(figsize=(8, 4))
        for model_index, model_id in enumerate(models):
            rows = [row for row in payload["cell_metrics"] if row["model_id"] == model_id and row["primary_value"] is not None]
            axis.scatter(
                [row["fold_id"] + (model_index - 1) * 0.08 for row in rows],
                [row["primary_value"] for row in rows],
                label=model_id,
                color=colors[model_id],
                alpha=0.8,
            )
        axis.set_xticks(list(contract["folds"]))
        axis.set_xlabel("Frozen P4 development fold")
        axis.set_ylabel(str(contract["primary_metric"]))
        axis.set_title(f"Sweetspot {task_id}: fold × repeat-seed")
        axis.legend(fontsize=8)
        path = destination / f"{task_id}_fold_seed.png"
        _save_figure(figure, path)
        written.append(path)

    for task_id in STATUS_ONLY_TARGETS:
        payload = json.loads((data_dir / f"{task_id}.json").read_text(encoding="utf-8"))
        figure, axis = plt.subplots(figsize=(7.2, 2.8))
        axis.axis("off")
        color = "#bc4749" if payload["status"] == "not_feasible" else "#f4a261"
        axis.add_patch(plt.Rectangle((0.02, 0.12), 0.96, 0.76, color=color, alpha=0.16, transform=axis.transAxes))
        axis.text(0.05, 0.70, f"{task_id} · {payload['target_name'] or payload['lane']}", fontsize=15, weight="bold", transform=axis.transAxes)
        axis.text(0.05, 0.50, f"STATUS: {payload['status']}", fontsize=12, color=color, weight="bold", transform=axis.transAxes)
        axis.text(0.05, 0.24, payload["reason"], fontsize=8.5, wrap=True, transform=axis.transAxes)
        suffix = "status_gate" if task_id == "T5" else "data_gate"
        path = destination / f"{task_id}_{suffix}.png"
        _save_figure(figure, path)
        written.append(path)
    return written


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _archive_predictions(
    destination: Path,
    predictions: Sequence[PredictionRecord],
    statuses: Mapping[str, str],
) -> dict[str, Any]:
    private_root = destination / "_private_predictions"
    entries = []
    by_key = {record.cell_key: record for record in predictions}
    for cell_key, status in statuses.items():
        record = by_key.get(cell_key)
        entry: dict[str, Any] = {
            "cell_key": cell_key,
            "status": status,
            "prediction_path": None,
            "prediction_sha256": None,
            "sample_count": 0,
            "tracked": False,
            "contains_test": False,
        }
        if record is not None:
            path = (
                private_root / record.task_id / record.model_id / f"fold_{record.fold_id}"
                / f"repeat_{record.repeat_id}_seed_{record.seed}.npz"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                sample_ids=np.asarray(record.sample_ids),
                groups=np.asarray(record.groups),
                actual=np.asarray(record.actual, dtype=np.float32),
                prediction=np.asarray(record.prediction, dtype=np.float32),
            )
            entry.update({
                "task_id": record.task_id,
                "lane": record.lane,
                "model_id": record.model_id,
                "fold_id": record.fold_id,
                "repeat_id": record.repeat_id,
                "seed": record.seed,
                "prediction_path": _relative(path),
                "prediction_sha256": sha256_file(path),
                "sample_count": len(record.sample_ids),
            })
        entries.append(entry)
    return {
        "schema_version": "sweetspot-p5-stage3-oof-manifest/v1",
        "prediction_scope": "P4-manifest development validation folds only",
        "raw_predictions": "private git-ignored npz; not a portable Git artifact",
        "portable_visualization_inputs": "visualization_data/*.json",
        "expected_cells": EXPECTED_CELLS,
        "entries": entries,
        "test_accessed": False,
        "historical_test_metrics_used": False,
    }


def write_outputs(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    predictions: Sequence[PredictionRecord],
) -> dict[str, Any]:
    destination = _portable_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    audit: LabelMappingAudit = context["audit"]

    result_path = destination / RESULT_FILENAME
    result_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in results) + "\n",
        encoding="utf-8",
    )
    leaderboard_dir = destination / "leaderboards"
    leaderboards: dict[str, Any] = {}
    for task_id in FROZEN_TASKS:
        leaderboards[task_id] = _leaderboard(task_id, results)
        _write_json(leaderboard_dir / f"{task_id}.json", leaderboards[task_id])
    for task_id in STATUS_ONLY_TARGETS:
        status_payload = _status_payload(audit, task_id)
        leaderboards[task_id] = {
            "schema_version": "sweetspot-p5-stage3-leaderboard/v1",
            "task_id": task_id,
            "lane": status_payload["lane"],
            "status": "not_rankable",
            "reason": status_payload["reason"],
            "expected_cells": 0,
            "completed_cells": 0,
            "label_version": status_payload["label_version"],
            "test_accessed": False,
        }
        _write_json(leaderboard_dir / f"{task_id}.json", leaderboards[task_id])

    visualization_data_dir = destination / "visualization_data"
    for task_id in FROZEN_TASKS:
        _write_json(
            visualization_data_dir / f"{task_id}.json",
            _visualization_payload(task_id, predictions, results),
        )
    for task_id in STATUS_ONLY_TARGETS:
        _write_json(visualization_data_dir / f"{task_id}.json", _status_payload(audit, task_id))
    figure_paths = rebuild_figures(visualization_data_dir, destination / "figures")

    status_by_key = {str(row["cell_key"]): str(row["status"]) for row in results}
    oof_manifest = _archive_predictions(destination, predictions, status_by_key)
    oof_path = destination / OOF_MANIFEST_FILENAME
    _write_json(oof_path, oof_manifest)

    visualization_entries = []
    for figure_path in sorted(figure_paths):
        task_id = figure_path.name.split("_", 1)[0]
        input_path = visualization_data_dir / f"{task_id}.json"
        visualization_entries.append({
            "task_id": task_id,
            "figure_path": _relative(figure_path),
            "figure_sha256": sha256_file(figure_path),
            "portable_input_path": _relative(input_path),
            "portable_input_sha256": sha256_file(input_path),
            "reads_raw_predictions": False,
            "reads_test": False,
        })
    visualization_manifest = {
        "schema_version": "sweetspot-p5-stage3-visualization-manifest/v1",
        "rebuild_module": "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage3",
        "rebuild_arguments": ["--rebuild-figures-only", "--output-dir", "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv"],
        "entries": visualization_entries,
        "test_accessed": False,
        "historical_test_metrics_used": False,
    }
    visualization_path = destination / VISUALIZATION_MANIFEST_FILENAME
    _write_json(visualization_path, visualization_manifest)

    counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("PASS", "SKIP", "FAILED", "TIMEOUT")
    }
    target_status = {}
    for task_id in FROZEN_TASKS:
        board = leaderboards[task_id]
        target_status[task_id] = {
            "status": board["status"],
            "expected_cells": board["expected_cells"],
            "completed_cells": board["completed_cells"],
            "completion_rate": board["completion_rate"],
            "label_version": audit.target(task_id)["label_version"],
            "is_proxy": audit.target(task_id)["is_proxy"],
            "proxy_semantics": audit.target(task_id)["proxy_semantics"],
        }
    for task_id in STATUS_ONLY_TARGETS:
        target_status[task_id] = _status_payload(audit, task_id)

    stage2_dir = HERE / "_outputs" / "stage2_pilot"
    summary = {
        "schema_version": "sweetspot-p5-stage3-summary/v1",
        "stage": "frozen_multiseed_development_cv",
        "baseline_commit": BASELINE_COMMIT,
        "root_seed": 2693,
        "repeat_seeds": list(REPEAT_SEEDS),
        "expected_cells": EXPECTED_CELLS,
        "attempted_cells": len(results),
        "counts": counts,
        "legal_completion_rate": counts["PASS"] / EXPECTED_CELLS,
        "rankable_completion_threshold": RANKABLE_COMPLETION_THRESHOLD,
        "frozen_tasks": {
            task_id: {"models": list(contract["models"]), "folds": list(contract["folds"])}
            for task_id, contract in FROZEN_TASKS.items()
        },
        "target_status": target_status,
        "split_evidence": context["split_evidence"],
        "budgets": {
            "train_sample_limit_per_fold": TRAIN_SAMPLE_LIMIT,
            "validation_sample_limit_per_fold": VALIDATION_SAMPLE_LIMIT,
            "tree_update_limit": TREE_UPDATE_LIMIT,
            "neural_update_limit": NEURAL_UPDATE_LIMIT,
            "cpu_wall_limit_seconds": CPU_WALL_LIMIT_SECONDS,
            "neural_wall_limit_seconds": NEURAL_WALL_LIMIT_SECONDS,
            "hpo": "forbidden",
            "technical_retries_used": 0,
        },
        "fit_scope": {
            "preprocessing": "each fold train only",
            "class_weights": "not defined by Stage-2",
            "target_transform": "each fold train only or fixed formula",
            "calibration": "not defined by Stage-2",
        },
        "label_mapping_sha256": audit.mapping_sha256,
        "source_lock_sha256": sha256_file(DEFAULT_LOCK),
        "stage3_protocol_path": _relative(PROTOCOL_PATH),
        "stage3_protocol_sha256": sha256_file(PROTOCOL_PATH),
        "stage3_implementation_sha256": sha256_file(HERE / "sweetspot_p5_stage3.py"),
        "stage2_implementation_sha256": sha256_file(HERE / "sweetspot_p5_stage2.py"),
        "stage2_data_implementation_sha256": sha256_file(HERE / "sweetspot_p5_stage2_data.py"),
        "stage2_archived_results_sha256": sha256_file(stage2_dir / "p5_stage2_results.jsonl"),
        "stage2_archived_summary_sha256": sha256_file(stage2_dir / "p5_stage2_summary.json"),
        "results_sha256": sha256_file(result_path),
        "oof_manifest_sha256": sha256_file(oof_path),
        "visualization_manifest_sha256": sha256_file(visualization_path),
        "leaderboard_sha256": {
            task_id: sha256_file(leaderboard_dir / f"{task_id}.json") for task_id in (*FROZEN_TASKS, *STATUS_ONLY_TARGETS)
        },
        "labels_generated": False,
        "test_accessed": False,
        "historical_test_metrics_used": False,
        "frozen_test_metrics_reported": False,
        "checkpoints_persisted": False,
        "full_predictions_tracked": False,
        "device": context["device"],
    }
    portable_files = [result_path, oof_path, visualization_path, *sorted(leaderboard_dir.glob("*.json")), *sorted(visualization_data_dir.glob("*.json")), *sorted(figure_paths)]
    summary["portable_artifacts"] = [
        {"path": _relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in portable_files
    ]
    summary_path = destination / SUMMARY_FILENAME
    _write_json(summary_path, summary)
    return {
        "results": _relative(result_path),
        "summary": _relative(summary_path),
        "oof_manifest": _relative(oof_path),
        "visualization_manifest": _relative(visualization_path),
        "counts": counts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--rebuild-figures-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = _portable_output_dir(args.output_dir)
    if args.rebuild_figures_only:
        paths = rebuild_figures(output_dir / "visualization_data", output_dir / "figures")
        sys.stdout.write(json.dumps({"figures_rebuilt": len(paths)}, sort_keys=True) + "\n")
        return 0
    started = time.monotonic()
    results, context, predictions = run_stage3(
        mapping_path=args.mapping,
        source_root=args.source_root,
        device=args.device,
        gpu_lock_path=args.gpu_lock,
    )
    files = write_outputs(output_dir, results, context, predictions)
    files["wall_seconds"] = time.monotonic() - started
    sys.stdout.write(json.dumps(files, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 1 if files["counts"]["FAILED"] or files["counts"]["TIMEOUT"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
