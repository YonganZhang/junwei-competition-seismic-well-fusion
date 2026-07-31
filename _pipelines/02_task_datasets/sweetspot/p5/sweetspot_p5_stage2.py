"""Sweetspot-private P5 Stage-2 fixed-budget development pilot.

The runner covers the frozen ten-model roster and seven independent targets.
It can read only P4 development manifests and raw-source rows/members authorized
by those manifests.  There is deliberately no test argument or test loader.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import math
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from _code.ml_framework.model_discovery import discover_model
from _models.sweetspot.p5_common import AdapterSkip

from .matrix import MODEL_ORDER, TARGET_ORDER, matrix_gate
from .source_lock import DEFAULT_LOCK, inspect_runtime, load_source_lock
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
    portable_mapping_payload,
    sha256_file,
    validate_label_mapping,
)


ROOT_SEED = 2693
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "stage2_pilot"
CPU_WALL_LIMIT_SECONDS = 300
NEURAL_WALL_LIMIT_SECONDS = 600
NEURAL_UPDATE_LIMIT = 64
TREE_UPDATE_LIMIT = 64
TREE_MODELS = {"xgboost", "catboost", "lightgbm"}
RESULT_FILENAME = "p5_stage2_results.jsonl"
SUMMARY_FILENAME = "p5_stage2_summary.json"
MAPPING_FILENAME = "p5_stage2_label_mapping.json"


def derive_seed(model_id: str, target_id: str) -> int:
    digest = hashlib.sha256(f"{ROOT_SEED}|{target_id}|{model_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _target_transform(values: np.ndarray, task_spec: Any) -> np.ndarray:
    target = task_spec.targets[0]
    transform = str(task_spec.target_transform[target]).lower()
    array = np.asarray(values, dtype=np.float64)
    if "log1p" in transform:
        if np.any(array < 0):
            raise ValueError(f"{task_spec.task_id}: log1p target contains negative values")
        return np.log1p(array)
    return array.copy()


def _inverse_target(values: np.ndarray, task_spec: Any) -> np.ndarray:
    target = task_spec.targets[0]
    transform = str(task_spec.target_transform[target]).lower()
    array = np.asarray(values, dtype=np.float64)
    return np.expm1(array) if "log1p" in transform else array.copy()


def _column_impute(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_array = np.asarray(train, dtype=np.float64)
    validation_array = np.asarray(validation, dtype=np.float64)
    medians = np.zeros(train_array.shape[1], dtype=np.float64)
    for column in range(train_array.shape[1]):
        finite = train_array[np.isfinite(train_array[:, column]), column]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    train_filled = np.where(np.isfinite(train_array), train_array, medians[None, :])
    validation_filled = np.where(np.isfinite(validation_array), validation_array, medians[None, :])
    return train_filled, validation_filled, medians


def _sequence_impute_and_scale(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    train_array = np.asarray(train, dtype=np.float64)
    validation_array = np.asarray(validation, dtype=np.float64)
    if train_array.ndim != 3 or validation_array.ndim != 3:
        raise ValueError("sequence inputs must use [sample, channel, time]")
    medians = np.zeros(train_array.shape[1], dtype=np.float64)
    for channel in range(train_array.shape[1]):
        finite = train_array[:, channel, :][np.isfinite(train_array[:, channel, :])]
        medians[channel] = float(np.median(finite)) if finite.size else 0.0
    fill = medians[None, :, None]
    train_filled = np.where(np.isfinite(train_array), train_array, fill)
    validation_filled = np.where(np.isfinite(validation_array), validation_array, fill)
    means = train_filled.mean(axis=(0, 2))
    stds = train_filled.std(axis=(0, 2))
    stds = np.where(stds < 1e-8, 1.0, stds)
    train_scaled = (train_filled - means[None, :, None]) / stds[None, :, None]
    validation_scaled = (validation_filled - means[None, :, None]) / stds[None, :, None]
    return train_scaled, validation_scaled, {
        "median": medians.tolist(), "mean": means.tolist(), "std": stds.tolist(),
    }


def _metric_payload(task_type: str, actual: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
    from scipy.stats import spearmanr

    truth = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    finite = np.isfinite(truth) & np.isfinite(predicted)
    if not finite.any():
        raise ValueError("validation produced no finite actual/prediction pairs")
    truth = truth[finite]
    predicted = predicted[finite]
    if task_type == "binary":
        if np.any((predicted < 0.0) | (predicted > 1.0)):
            raise ValueError("binary predictions must be probabilities")
        result: dict[str, float | None] = {
            "average_precision": None,
            "brier": _finite_float(brier_score_loss(truth.astype(int), predicted)),
            "f1_at_0_5": _finite_float(f1_score(truth.astype(int), predicted >= 0.5, zero_division=0)),
        }
        if len(np.unique(truth.astype(int))) == 2:
            result["average_precision"] = _finite_float(average_precision_score(truth.astype(int), predicted))
        return result
    residual = predicted - truth
    correlation = spearmanr(truth, predicted).statistic if len(truth) >= 2 else float("nan")
    return {
        "mae": _finite_float(np.mean(np.abs(residual))),
        "rmse": _finite_float(np.sqrt(np.mean(np.square(residual)))),
        "spearman": _finite_float(correlation),
        "negative_prediction_fraction": _finite_float(np.mean(predicted < 0.0)),
    }


def _worst_group_metrics(
    task_type: str,
    groups: Sequence[str],
    actual: np.ndarray,
    prediction: np.ndarray,
    primary_metric: str,
    direction: str,
) -> dict[str, Any]:
    payload: dict[str, Mapping[str, Any]] = {}
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        mask = group_array == group
        payload[group] = _metric_payload(task_type, actual[mask], prediction[mask])
    key = "mae" if primary_metric == "physical_MAE" else primary_metric
    candidates = [
        (group, metrics.get(key)) for group, metrics in payload.items() if metrics.get(key) is not None
    ]
    if not candidates:
        return {"metric": key, "direction": direction, "worst_group": None, "value": None, "by_group": payload}
    chooser = max if direction == "minimize" else min
    group, value = chooser(candidates, key=lambda item: float(item[1]))
    return {"metric": key, "direction": direction, "worst_group": group, "value": value, "by_group": payload}


def _tiny_indices(target: np.ndarray, task_type: str, limit: int = 64) -> np.ndarray:
    y = np.asarray(target)
    if task_type != "binary":
        return np.arange(min(limit, len(y)))
    selected: list[int] = []
    for label in (0, 1):
        positions = np.flatnonzero(y.astype(int) == label)
        if positions.size:
            selected.append(int(positions[0]))
    for index in range(len(y)):
        if len(selected) >= min(limit, len(y)):
            break
        if index not in selected:
            selected.append(index)
    return np.asarray(selected, dtype=int)


@contextmanager
def exclusive_gpu_lock(path: Path | None) -> Iterable[dict[str, Any]]:
    if path is None:
        raise ValueError("GPU execution requires --gpu-lock; no implicit machine path is used")
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    waited = time.monotonic()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield {"acquired": True, "wait_seconds": time.monotonic() - waited, "path_recorded": False}
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _tree_config(model_id: str, seed: int) -> dict[str, Any]:
    if model_id == "xgboost":
        return {"n_estimators": TREE_UPDATE_LIMIT, "random_state": seed, "n_jobs": 1}
    if model_id == "catboost":
        return {"iterations": TREE_UPDATE_LIMIT, "random_seed": seed, "thread_count": 1}
    if model_id == "lightgbm":
        return {"n_estimators": TREE_UPDATE_LIMIT, "random_state": seed, "n_jobs": 1}
    raise KeyError(model_id)


def _run_tree_pilot(
    model_id: str,
    task_spec: Any,
    data: DevelopmentPilotData,
    seed: int,
) -> dict[str, Any]:
    target = task_spec.targets[0]
    train_x, validation_x, medians = _column_impute(data.train_tabular, data.validation_tabular)
    train_y = _target_transform(data.train_target, task_spec)
    config = _tree_config(model_id, seed)
    discovered = discover_model("sweetspot", model_id)
    tiny = _tiny_indices(train_y, task_spec.task_type)
    gate_adapter = discovered.build(task_spec, **config)
    gate = gate_adapter.stage1_smoke(
        {"tabular": train_x[tiny]}, train_y[tiny], np.ones(len(tiny), dtype=bool), seed=seed,
    )
    started = time.monotonic()
    adapter = discovered.build(task_spec, **config)
    adapter.fit(train_x, {target: train_y}, {target: np.ones(len(train_y), dtype=bool)})
    output = adapter.predict(validation_x)
    if task_spec.task_type == "binary":
        prediction = np.asarray(output.transformed[target], dtype=np.float64).reshape(-1)
    else:
        prediction = _inverse_target(np.asarray(output.raw[target], dtype=np.float64).reshape(-1), task_spec)
    wall = time.monotonic() - started
    if wall > CPU_WALL_LIMIT_SECONDS:
        raise TimeoutError(f"CPU pilot exceeded {CPU_WALL_LIMIT_SECONDS}s")
    return {
        "prediction": prediction,
        "tiny_gate": gate,
        "pilot": {
            "estimator_class": type(adapter.estimator).__name__,
            "update_steps": TREE_UPDATE_LIMIT,
            "wall_seconds": wall,
            "peak_rss_bytes": _rss_bytes(),
            "peak_vram_bytes": 0,
            "download_bytes": 0,
            "train_only_imputation_sha256": canonical_sha256(medians.tolist()),
        },
    }


def _next_batch(iterator: Any, loader: Any) -> tuple[Any, Any, Any]:
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch[0], batch[1], iterator


def _run_inceptiontime_pilot(
    task_spec: Any,
    data: DevelopmentPilotData,
    seed: int,
    *,
    device: str,
    gpu_lock_path: Path | None,
) -> dict[str, Any]:
    if data.train_sequence is None or data.validation_sequence is None:
        raise AdapterSkip("input_modality_missing", "approved development sequence is unavailable")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device == "cuda" and not torch.cuda.is_available():
        raise AdapterSkip("cuda_unavailable", "shared interpreter cannot see CUDA")
    train_x, validation_x, preprocessing = _sequence_impute_and_scale(
        data.train_sequence, data.validation_sequence,
    )
    train_y = _target_transform(data.train_target, task_spec).astype(np.float32)
    discovered = discover_model("sweetspot", "inceptiontime")
    config = {
        "c_in": train_x.shape[1], "seq_len": train_x.shape[2], "nf": 8, "device": device,
    }
    lock_context = exclusive_gpu_lock(gpu_lock_path) if device == "cuda" else _null_lock()
    with lock_context as lock_evidence:
        tiny = _tiny_indices(train_y, task_spec.task_type, limit=32)
        gate_adapter = discovered.build(task_spec, **config)
        gate = gate_adapter.stage1_smoke(
            {"sequence": train_x[tiny]}, train_y[tiny], np.ones(len(tiny), dtype=bool), seed=seed,
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        adapter = discovered.build(task_spec, **config)
        module = adapter.module
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(train_x, dtype=torch.float32), torch.as_tensor(train_y, dtype=torch.float32)),
            batch_size=min(64, len(train_x)), shuffle=True, generator=generator, num_workers=0,
        )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        iterator = iter(loader)
        final_loss = None
        module.train()
        for _ in range(NEURAL_UPDATE_LIMIT):
            batch_x, batch_y, iterator = _next_batch(iterator, loader)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = module(batch_x).reshape(-1)
            if task_spec.task_type == "binary":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y)
            else:
                loss = torch.nn.functional.mse_loss(logits, batch_y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("InceptionTime pilot produced non-finite loss")
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        module.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(validation_x), 128):
                batch = torch.as_tensor(validation_x[start:start + 128], dtype=torch.float32, device=device)
                raw = module(batch).reshape(-1)
                if task_spec.task_type == "binary":
                    raw = torch.sigmoid(raw)
                predictions.append(raw.detach().cpu().numpy())
        prediction = np.concatenate(predictions)
        if task_spec.task_type != "binary":
            prediction = _inverse_target(prediction, task_spec)
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        checkpoint_sha = hashlib.sha256(buffer.getvalue()).hexdigest()
        wall = time.monotonic() - started
        if wall > NEURAL_WALL_LIMIT_SECONDS:
            raise TimeoutError(f"neural pilot exceeded {NEURAL_WALL_LIMIT_SECONDS}s")
        peak_vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return {
        "prediction": prediction,
        "tiny_gate": gate,
        "pilot": {
            "estimator_class": type(module).__name__,
            "optimizer": "AdamW",
            "update_steps": NEURAL_UPDATE_LIMIT,
            "final_train_loss": final_loss,
            "wall_seconds": wall,
            "peak_rss_bytes": _rss_bytes(),
            "peak_vram_bytes": peak_vram,
            "download_bytes": 0,
            "checkpoint_bytes": buffer.getbuffer().nbytes,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_persisted": False,
            "train_only_preprocessing_sha256": canonical_sha256(preprocessing),
            "gpu_lock": lock_evidence,
        },
    }


@contextmanager
def _null_lock() -> Iterable[dict[str, Any]]:
    yield {"acquired": False, "reason": "CPU execution", "path_recorded": False}


def _tft_real_smoke(
    task_spec: Any,
    data: DevelopmentPilotData,
    seed: int,
) -> dict[str, Any]:
    """Run an exact-target causal smoke, then stop before incomparable ranking."""
    if data.target_id != "T3" or task_spec.task_type != "regression":
        raise AdapterSkip(
            "adapter_task_type_or_target_unsupported",
            "the locked TFT adapter supports the scalar-regression T3 lane only",
        )
    if data.train_sequence is None:
        raise AdapterSkip("input_modality_missing", "T3 causal histories are unavailable")
    import pandas as pd

    sequence = np.asarray(data.train_sequence, dtype=np.float64).copy()
    for channel in range(sequence.shape[1]):
        finite = sequence[:, channel, :][np.isfinite(sequence[:, channel, :])]
        median = float(np.median(finite)) if finite.size else 0.0
        sequence[:, channel, :] = np.where(
            np.isfinite(sequence[:, channel, :]), sequence[:, channel, :], median,
        )
    transformed = _target_transform(data.train_target, task_spec)
    records: list[dict[str, Any]] = []
    sample_count = min(8, len(sequence))
    history_length = sequence.shape[2]
    for sample_index in range(sample_count):
        history_target = np.log1p(np.maximum(sequence[sample_index, 0], 0.0))
        for time_index, value in enumerate(history_target):
            records.append({
                "time_idx": time_index, "group_id": f"development_{sample_index}",
                "target": float(value), "known": float(time_index),
            })
        records.append({
            "time_idx": history_length, "group_id": f"development_{sample_index}",
            "target": float(transformed[sample_index]), "known": float(history_length),
        })
    frame = pd.DataFrame.from_records(records)
    adapter = discover_model("sweetspot", "temporal_fusion_transformer").build(
        task_spec, encoder_length=history_length, prediction_length=1,
    )
    smoke = adapter.stage1_smoke(
        {"time_series_frame": frame}, transformed[:sample_count],
        np.ones(sample_count, dtype=bool), seed=seed,
    )
    return smoke


def _base_result(
    audit: LabelMappingAudit,
    model_id: str,
    target_id: str,
    data: DevelopmentPilotData | None,
) -> dict[str, Any]:
    target = audit.target(target_id)
    cell = matrix_gate(model_id, target_id)
    return {
        "schema_version": "sweetspot-p5-stage2-cell/v1",
        "model_id": model_id,
        "task_id": target_id,
        "lane": target["slug"],
        "matrix_rating": cell.rating,
        "label_status": target["status"],
        "p4_status": target["p4_status"],
        "label_version": target.get("label_version"),
        "is_proxy": target["is_proxy"],
        "proxy_semantics": target["proxy_semantics"],
        "seed": derive_seed(model_id, target_id),
        "split_hash": None if data is None else data.split_sha256,
        "input_budget": {
            "train_sample_limit": TRAIN_SAMPLE_LIMIT,
            "validation_sample_limit": VALIDATION_SAMPLE_LIMIT,
            "train_samples": None,
            "validation_samples": None,
            "input_budget_sha256": None,
        } if data is None else data.budget,
        "status": None,
        "reason": None,
        "evidence_state": "scouted",
        "validation_metrics": None,
        "worst_group": None,
        "stability": {"seed_count": 1, "status": "not_estimated_in_single-seed_stage2"},
        "resource": None,
        "label_generated": False,
        "test_firewall": {
            "test_loader_api_present": False,
            "frozen_test_files_opened": 0,
            "test_accessed": False,
            "historical_test_metrics_used": False,
        },
        "label_mapping_sha256": audit.mapping_sha256,
    }


def _skip(result: dict[str, Any], reason_code: str, detail: Any, **extra: Any) -> dict[str, Any]:
    result.update({
        "status": "SKIP", "reason": {"code": reason_code, "detail": detail},
        "evidence_state": extra.pop("evidence_state", "scouted"),
    })
    result.update(extra)
    return result


def _run_cell(
    audit: LabelMappingAudit,
    source_lock: Mapping[str, Mapping[str, Any]],
    model_id: str,
    target_id: str,
    data: DevelopmentPilotData | None,
    data_error: DevelopmentDataUnavailable | None,
    *,
    device: str,
    gpu_lock_path: Path | None,
) -> dict[str, Any]:
    result = _base_result(audit, model_id, target_id, data)
    target = audit.target(target_id)
    if target["status"] == "not_feasible":
        return _skip(result, "label_not_feasible", target["proxy_semantics"])
    cell = matrix_gate(model_id, target_id)
    if not cell.eligible:
        return _skip(result, "matrix_not_applicable", "frozen model-target matrix marks this cell inapplicable")
    if data_error is not None:
        return _skip(result, data_error.reason_code, data_error.detail)
    if data is None:
        return _skip(result, "development_data_missing", "no development data object")
    if model_id == "monai_unet3d":
        return _skip(
            result, "target_support_mismatch",
            "P4 labels are well-point/scalar targets; no approved dense 3D label volume exists",
        )
    runtime = inspect_runtime(source_lock[model_id])
    result["source_lock"] = {
        "revision": source_lock[model_id]["revision"],
        "license": source_lock[model_id]["license"],
        "runtime": runtime,
    }
    if not runtime["available"]:
        return _skip(result, str(runtime["reason_code"]), runtime)
    if not runtime["version_allowed"]:
        return _skip(result, "runtime_version_not_locked", runtime)
    seed = int(result["seed"])
    task_spec = build_pilot_task_spec(audit, target_id)
    try:
        if model_id in TREE_MODELS:
            execution = _run_tree_pilot(model_id, task_spec, data, seed)
        elif model_id == "inceptiontime":
            execution = _run_inceptiontime_pilot(
                task_spec, data, seed, device=device, gpu_lock_path=gpu_lock_path,
            )
        elif model_id == "temporal_fusion_transformer":
            smoke = _tft_real_smoke(task_spec, data, seed)
            return _skip(
                result,
                "development_smoke_only_no_comparable_validation_contract",
                "real T3 causal target smoke passed, but the locked thin adapter exposes no held-out prediction API",
                evidence_state="contract_smoked",
                development_smoke=smoke,
            )
        else:
            return _skip(
                result, "stage2_validation_adapter_unavailable",
                "source-locked candidate has no approved Stage-2 validation adapter in the shared environments",
            )
    except AdapterSkip as exc:
        return _skip(result, exc.reason_code, exc.detail)
    prediction = np.asarray(execution.pop("prediction"), dtype=np.float64)
    metrics = _metric_payload(task_spec.task_type, data.validation_target, prediction)
    primary = str(target["primary_metric"])
    direction = str(target["primary_metric_direction"])
    worst = _worst_group_metrics(
        task_spec.task_type, data.validation_groups, data.validation_target, prediction,
        primary, direction,
    )
    result.update({
        "status": "PASS",
        "reason": None,
        "evidence_state": "development_piloted",
        "estimator_head": task_spec.targets[0],
        "validation_metrics": metrics,
        "worst_group": worst,
        "resource": execution["pilot"],
        "tiny_gate": execution["tiny_gate"],
        "development_provenance": data.provenance,
    })
    return result


def _leaderboards(audit: LabelMappingAudit, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    leaderboards: dict[str, Any] = {}
    for target_id in TARGET_ORDER:
        target = audit.target(target_id)
        valid = [item for item in results if item["task_id"] == target_id and item["status"] == "PASS"]
        primary = target.get("primary_metric")
        direction = target.get("primary_metric_direction")
        if len(valid) < 2 or primary is None:
            leaderboards[target_id] = {
                "status": "not_rankable", "reason": "fewer than two comparable development pilots",
                "candidate_count": len(valid), "primary_metric": primary,
            }
            continue
        metric_key = "mae" if primary == "physical_MAE" else primary
        ranked = [item for item in valid if item["validation_metrics"].get(metric_key) is not None]
        ranked.sort(
            key=lambda item: float(item["validation_metrics"][metric_key]),
            reverse=direction == "maximize",
        )
        leaderboards[target_id] = {
            "status": "rankable",
            "scope": "single fixed P4 development fold; not final model ranking",
            "primary_metric": metric_key,
            "direction": direction,
            "entries": [
                {
                    "rank": rank,
                    "model_id": item["model_id"],
                    "value": item["validation_metrics"][metric_key],
                    "worst_group_value": item["worst_group"]["value"],
                    "wall_seconds": item["resource"]["wall_seconds"],
                }
                for rank, item in enumerate(ranked, 1)
            ],
        }
    return leaderboards


def run_stage2(
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    source_root: Path | None = None,
    device: str = "cpu",
    gpu_lock_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    audit = validate_label_mapping(mapping_path)
    source_lock = load_source_lock()
    data_by_target: dict[str, DevelopmentPilotData | None] = {}
    errors: dict[str, DevelopmentDataUnavailable | None] = {}
    for target_id in TARGET_ORDER:
        if audit.target(target_id)["status"] == "not_feasible":
            data_by_target[target_id] = None
            errors[target_id] = None
            continue
        try:
            data_by_target[target_id] = load_development_pilot_data(
                audit, target_id, source_root=source_root,
            )
            errors[target_id] = None
        except DevelopmentDataUnavailable as exc:
            data_by_target[target_id] = None
            errors[target_id] = exc
    results: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for target_id in TARGET_ORDER:
            try:
                result = _run_cell(
                    audit, source_lock, model_id, target_id,
                    data_by_target[target_id], errors[target_id],
                    device=device, gpu_lock_path=gpu_lock_path,
                )
            except Exception as exc:  # unexpected failures remain fail-loud in summary/exit code
                result = _base_result(audit, model_id, target_id, data_by_target[target_id])
                result.update({
                    "status": "FAILED",
                    "reason": {"code": "unexpected_runtime_failure", "detail": f"{type(exc).__name__}: {exc}"},
                    "evidence_state": "scouted",
                })
            results.append(result)
    counts = {
        status: sum(item["status"] == status for item in results)
        for status in ("PASS", "SKIP", "FAILED", "TIMEOUT")
    }
    target_data = {}
    for target_id in TARGET_ORDER:
        data = data_by_target[target_id]
        error = errors[target_id]
        target_data[target_id] = {
            "status": "READY" if data is not None else "NOT_FEASIBLE" if audit.target(target_id)["status"] == "not_feasible" else "BLOCKED",
            "reason": None if error is None else {"code": error.reason_code, "detail": error.detail},
            "input_budget": None if data is None else data.budget,
            "provenance": None if data is None else data.provenance,
        }
    summary = {
        "schema_version": "sweetspot-p5-stage2-summary/v1",
        "stage": "fixed_development_pilot",
        "root_seed": ROOT_SEED,
        "expected_cells": len(MODEL_ORDER) * len(TARGET_ORDER),
        "attempted_cells": len(results),
        "counts": counts,
        "model_order": list(MODEL_ORDER),
        "target_order": list(TARGET_ORDER),
        "target_data": target_data,
        "leaderboards": _leaderboards(audit, results),
        "budgets": {
            "train_sample_limit_per_target": TRAIN_SAMPLE_LIMIT,
            "validation_sample_limit_per_target": VALIDATION_SAMPLE_LIMIT,
            "tree_update_limit": TREE_UPDATE_LIMIT,
            "neural_update_limit": NEURAL_UPDATE_LIMIT,
            "cpu_wall_limit_seconds": CPU_WALL_LIMIT_SECONDS,
            "neural_wall_limit_seconds": NEURAL_WALL_LIMIT_SECONDS,
        },
        "label_mapping_sha256": audit.mapping_sha256,
        "source_lock_sha256": sha256_file(DEFAULT_LOCK),
        "labels_generated": False,
        "test_accessed": False,
        "historical_test_metrics_used": False,
        "frozen_test_metrics_reported": False,
        "checkpoints_persisted": False,
        "device": device,
    }
    return results, summary, portable_mapping_payload(audit)


def _portable_output_dir(path: Path) -> Path:
    resolved = Path(path).resolve()
    private_root = HERE.resolve()
    try:
        resolved.relative_to(private_root)
    except ValueError as exc:
        raise PermissionError("Stage-2 results must stay in the sweetspot-private p5 directory") from exc
    return resolved


def write_outputs(
    output_dir: Path,
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    mapping: Mapping[str, Any],
) -> dict[str, str]:
    destination = _portable_output_dir(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / RESULT_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    mapping_path = destination / MAPPING_FILENAME
    lines = [json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) for item in results]
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_summary = dict(summary)
    final_summary["results_sha256"] = sha256_file(result_path)
    final_summary["portable_output_files"] = [RESULT_FILENAME, SUMMARY_FILENAME, MAPPING_FILENAME]
    summary_path.write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "results": result_path.relative_to(PROJECT_ROOT).as_posix(),
        "summary": summary_path.relative_to(PROJECT_ROOT).as_posix(),
        "label_mapping": mapping_path.relative_to(PROJECT_ROOT).as_posix(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu-lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.device == "cuda" and args.gpu_lock is None:
        raise ValueError("--device=cuda requires the protocol's explicit --gpu-lock path")
    results, summary, mapping = run_stage2(
        mapping_path=args.mapping,
        source_root=args.source_root,
        device=args.device,
        gpu_lock_path=args.gpu_lock,
    )
    files = write_outputs(args.output_dir, results, summary, mapping)
    payload = {"counts": summary["counts"], "files": files}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 1 if summary["counts"]["FAILED"] or summary["counts"]["TIMEOUT"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
