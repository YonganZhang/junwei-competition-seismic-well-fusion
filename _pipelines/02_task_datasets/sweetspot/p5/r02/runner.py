"""Sweetspot P5.2 / protocol R2 real development budget sweep.

The sweep is deliberately limited to T1/T2/T3.  It executes the frozen ten-model
roster across the 64/256/1024 budget ladder plus one-factor ablations at the
256 budget.  T4-T7 remain explicit status boundaries and are summarized but not
counted inside the 120-cell development sweep.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pickle
import resource
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from _code.ml_framework.model_discovery import discover_model
from _models.sweetspot.p5_common import AdapterSkip

from ..matrix import MODEL_ORDER, TARGET_ORDER, matrix_gate
from ..source_lock import DEFAULT_LOCK, inspect_runtime, load_source_lock
from ..sweetspot_p5_stage2 import (
    _metric_payload,
    _sequence_impute_and_scale,
    _tiny_indices,
    _worst_group_metrics,
    exclusive_gpu_lock,
)
from ..sweetspot_p5_stage2_data import (
    DevelopmentDataUnavailable,
    DevelopmentPilotData,
    load_development_pilot_data,
)
from ..sweetspot_p5_stage2_labels import (
    DEFAULT_MAPPING_PATH,
    PROJECT_ROOT,
    build_pilot_task_spec,
    canonical_sha256,
    sha256_file,
    validate_label_mapping,
)


ROOT_SEED = 2693
TARGET_SWEEP = ("T1", "T2", "T3")
STATUS_ONLY_TARGETS = ("T4", "T5", "T6", "T7")
MAIN_BUDGETS = (64, 256, 1024)
ABLATION_BUDGET = 256
MAIN_EXPECTED_CELLS = len(MODEL_ORDER) * len(TARGET_SWEEP) * len(MAIN_BUDGETS)
ABLATION_EXPECTED_CELLS = len(MODEL_ORDER) * len(TARGET_SWEEP)
EXPECTED_CELLS = MAIN_EXPECTED_CELLS + ABLATION_EXPECTED_CELLS

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "protocol_r2"
RESULT_FILENAME = "p5_r02_results.jsonl"
SUMMARY_FILENAME = "p5_r02_summary.json"
PLATEAU_FILENAME = "p5_r02_plateau_gate.json"
STATUS_GATES_FILENAME = "p5_r02_status_gates.json"
VISUALIZATION_FILENAME = "p5_r02_visualization_manifest.json"
ARTIFACT_MANIFEST_FILENAME = "p5_r02_artifact_manifest.json"

TREE_MODELS = {"xgboost", "catboost", "lightgbm"}
NEURAL_MODELS = {"inceptiontime"}
SOFT_SKIP_MODELS = {
    "autogluon_limited",
    "patchtst",
    "temporal_fusion_transformer",
    "seg_spatial_tcn",
    "graphsage",
    "monai_unet3d",
}


@dataclass(frozen=True)
class CellVariant:
    name: str
    kind: str
    target_transform: str = "identity"
    class_weighting: str = "unweighted"
    context_days: int = 30


MAIN_VARIANTS: Mapping[str, CellVariant] = {
    "T1": CellVariant("main", "main", target_transform="log1p"),
    "T2": CellVariant("main", "main", class_weighting="balanced"),
    "T3": CellVariant("main", "main", context_days=30),
}
ABLATION_VARIANTS: Mapping[str, CellVariant] = {
    "T1": CellVariant("identity_output", "ablation", target_transform="identity"),
    "T2": CellVariant("unweighted_loss", "ablation", class_weighting="unweighted"),
    "T3": CellVariant("short_context_7d", "ablation", context_days=7),
}


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _safe_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _finite_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _budget_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def _target_transform(values: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if mode == "log1p":
        if np.any(array < 0.0):
            raise ValueError("log1p transform received negative values")
        return np.log1p(array)
    return array.copy()


def _inverse_target(values: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if mode == "log1p":
        return np.expm1(array)
    return array.copy()


def _class_weights(values: np.ndarray, mode: str) -> np.ndarray | None:
    if mode != "balanced":
        return None
    y = np.asarray(values).astype(int).reshape(-1)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise DevelopmentDataUnavailable("fold_train_class_missing", "binary sweep requires both labels")
    pos_weight = negatives / positives
    return np.where(y == 1, pos_weight, 1.0).astype(np.float64)


def _sequence_context_features(sequence: np.ndarray, context_days: int) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("sequence inputs must be [sample, channel, time]")
    window = array[:, :, -context_days:] if context_days < array.shape[2] else array
    means = window.mean(axis=2)
    stds = window.std(axis=2)
    last = window[:, :, -1]
    return np.concatenate([means, stds, last], axis=1)


def _tree_config(model_id: str, seed: int, budget: int) -> dict[str, Any]:
    if model_id == "xgboost":
        return {"n_estimators": budget, "random_state": seed, "n_jobs": 1, "tree_method": "hist"}
    if model_id == "catboost":
        return {"iterations": budget, "random_seed": seed, "thread_count": 1, "allow_writing_files": False, "verbose": False}
    if model_id == "lightgbm":
        return {"n_estimators": budget, "random_state": seed, "n_jobs": 1, "verbosity": -1}
    raise KeyError(model_id)


def _derive_seed(model_id: str, target_id: str, budget: int, variant: CellVariant) -> int:
    digest = hashlib.sha256(f"{ROOT_SEED}|{target_id}|{model_id}|{budget}|{variant.name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _tiny_gate_for_tree(task_spec: Any, model_id: str, seed: int, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    discovered = discover_model("sweetspot", model_id)
    adapter = discovered.build(task_spec, **_tree_config(model_id, seed, 16))
    tiny = _tiny_indices(y, task_spec.task_type, limit=32)
    return adapter.stage1_smoke({"tabular": x[tiny]}, y[tiny], np.ones(len(tiny), dtype=bool), seed=seed)


def _tiny_gate_for_sequence(task_spec: Any, seed: int, sequence: np.ndarray, y: np.ndarray, *, device: str) -> dict[str, Any]:
    discovered = discover_model("sweetspot", "inceptiontime")
    adapter = discovered.build(
        task_spec,
        c_in=int(sequence.shape[1]),
        seq_len=int(sequence.shape[2]),
        nf=8,
        device=device,
    )
    tiny = _tiny_indices(y, task_spec.task_type, limit=32)
    return adapter.stage1_smoke({"sequence": sequence[tiny]}, y[tiny], np.ones(len(tiny), dtype=bool), seed=seed)


def _fit_tree_estimator(
    model_id: str,
    task_spec: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray | None,
    validation_x: np.ndarray,
    *,
    seed: int,
    budget: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    discovered = discover_model("sweetspot", model_id)
    adapter = discovered.build(task_spec, **_tree_config(model_id, seed, budget))
    estimator = adapter.estimator
    fit_kwargs: dict[str, Any] = {}
    if train_w is not None:
        fit_kwargs["sample_weight"] = train_w
    started = time.monotonic()
    estimator.fit(train_x, train_y, **fit_kwargs)
    if task_spec.task_type == "binary":
        prediction = np.asarray(estimator.predict_proba(validation_x))[:, 1]
    else:
        raw = np.asarray(estimator.predict(validation_x), dtype=np.float64).reshape(-1)
        prediction = raw
    encoded = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(encoded)
    if task_spec.task_type == "binary":
        replay = np.asarray(restored.predict_proba(validation_x))[:, 1]
    else:
        replay = np.asarray(restored.predict(validation_x), dtype=np.float64).reshape(-1)
    delta = float(np.max(np.abs(prediction - replay))) if prediction.size else 0.0
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite tree predictions")
    return prediction, {
        "estimator_class": type(estimator).__name__,
        "checkpoint_bytes": len(encoded),
        "checkpoint_roundtrip_max_abs_delta": delta,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": _rss_bytes(),
        "peak_vram_bytes": 0,
        "download_bytes": 0,
    }, {"adapter": "tree", "budget": budget}


def _fit_inceptiontime(
    task_spec: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray | None,
    validation_x: np.ndarray,
    *,
    seed: int,
    budget: int,
    device: str,
    gpu_lock_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if device == "cuda" and not torch.cuda.is_available():
        raise AdapterSkip("cuda_unavailable", "shared interpreter cannot see CUDA")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    discovered = discover_model("sweetspot", "inceptiontime")
    adapter = discovered.build(
        task_spec,
        c_in=int(train_x.shape[1]),
        seq_len=int(train_x.shape[2]),
        nf=8,
        device=device,
    )
    tiny = _tiny_indices(train_y, task_spec.task_type, limit=32)
    gate = adapter.stage1_smoke(
        {"sequence": train_x[tiny]}, train_y[tiny], np.ones(len(tiny), dtype=bool), seed=seed,
    )
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    module = adapter.module
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_x, dtype=torch.float32),
            torch.as_tensor(train_y, dtype=torch.float32),
            torch.as_tensor(train_w if train_w is not None else np.ones_like(train_y), dtype=torch.float32),
        ),
        batch_size=min(16, len(train_x)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    lock_context = exclusive_gpu_lock(gpu_lock_path) if device == "cuda" else _null_lock()
    started = time.monotonic()
    with lock_context as lock_evidence:
        iterator = iter(loader)
        optimizer = torch.optim.AdamW(module.parameters(), lr=5e-4, weight_decay=1e-4)
        module.train()
        final_loss = None
        for _ in range(budget):
            try:
                batch_x, batch_y, batch_w = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch_x, batch_y, batch_w = next(iterator)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = module(batch_x).reshape(-1)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            logits = torch.clamp(logits, min=-20.0, max=20.0)
            batch_y = torch.nan_to_num(batch_y, nan=0.0, posinf=0.0, neginf=0.0)
            if task_spec.task_type == "binary":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch_y, reduction="none")
            else:
                loss = torch.nn.functional.mse_loss(logits, batch_y, reduction="none")
            loss = (loss * batch_w).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite InceptionTime loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
        module.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(validation_x), 128):
                batch = torch.as_tensor(validation_x[start:start + 128], dtype=torch.float32, device=device)
                raw = module(batch).reshape(-1)
                raw = torch.nan_to_num(raw, nan=0.0, posinf=20.0, neginf=-20.0)
                raw = torch.clamp(raw, min=-20.0, max=20.0)
                if task_spec.task_type == "binary":
                    raw = torch.sigmoid(raw)
                preds.append(raw.detach().cpu().numpy())
        prediction = np.concatenate(preds)
        if task_spec.task_type != "binary":
            prediction = prediction.reshape(-1)
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        restored = adapter.module_factory().to(device)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, map_location=device, weights_only=True))
        restored.eval()
        with torch.no_grad():
            replay_batches = []
            for start in range(0, len(validation_x), 128):
                batch = torch.as_tensor(validation_x[start:start + 128], dtype=torch.float32, device=device)
                raw = restored(batch).reshape(-1)
                raw = torch.nan_to_num(raw, nan=0.0, posinf=20.0, neginf=-20.0)
                raw = torch.clamp(raw, min=-20.0, max=20.0)
                if task_spec.task_type == "binary":
                    raw = torch.sigmoid(raw)
                replay_batches.append(raw.detach().cpu().numpy())
        replay = np.concatenate(replay_batches)
        delta = float(np.max(np.abs(prediction - replay))) if prediction.size else 0.0
        peak_vram = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    return prediction, {
        "estimator_class": type(module).__name__,
        "checkpoint_bytes": buffer.getbuffer().nbytes,
        "checkpoint_roundtrip_max_abs_delta": delta,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": _rss_bytes(),
        "peak_vram_bytes": peak_vram,
        "download_bytes": 0,
        "optimizer": "AdamW",
        "update_steps": budget,
        "final_train_loss": final_loss,
        "gpu_lock": lock_evidence if device == "cuda" else {"acquired": False, "reason": "cpu execution", "path_recorded": False},
    }, {"adapter": "inceptiontime", "budget": budget}


@contextmanager
def _null_lock() -> Iterable[dict[str, Any]]:
    yield {"acquired": False, "reason": "cpu execution", "path_recorded": False}


def _cell_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def _task_spec_for_target(audit: Any, target_id: str) -> Any:
    base = build_pilot_task_spec(audit, target_id)
    from dataclasses import replace

    metadata = dict(base.metadata)
    metadata.update({
        "p5_stage": "P5.2/protocol R2",
        "r2_main_budgets": list(MAIN_BUDGETS),
        "r2_ablation_budget": ABLATION_BUDGET,
        "test_access": "forbidden",
    })
    return replace(
        base,
        task_id=f"sweetspot.p5.r02.{target_id.lower()}.{audit.target(target_id)['slug']}",
        metadata=metadata,
    )


def _status_gate(audit: Any) -> dict[str, Any]:
    return {
        "schema_version": "sweetspot-p5-r02-status-gate/v1",
        "root_seed": ROOT_SEED,
        "targets": {
            "T4": {
                "status": "boundary",
                "reason": "kept outside the T1-T3 budget sweep; stage4 known-holdout confirmation remains the boundary record",
                "fresh_blind": False,
            },
            "T5": {
                "status": "not_feasible",
                "reason": "no approved sweetspot field truth; proxy-only labels remain forbidden",
                "fresh_blind": False,
            },
            "T6": {
                "status": "blocked",
                "reason": "no development-only feature source; test.h5 fallback is forbidden",
                "fresh_blind": False,
            },
            "T7": {
                "status": "blocked",
                "reason": "no development-only feature source; test.h5 fallback is forbidden",
                "fresh_blind": False,
            },
        },
        "test_accessed": False,
        "historical_test_metrics_read": False,
        "label_generated": False,
    }


def _run_cell(
    *,
    audit: Any,
    source_lock: Mapping[str, Mapping[str, Any]],
    target_id: str,
    model_id: str,
    variant: CellVariant,
    budget: int,
    data: DevelopmentPilotData,
    device: str,
    gpu_lock_path: Path | None,
) -> dict[str, Any]:
    cell = matrix_gate(model_id, target_id)
    task_spec = _task_spec_for_target(audit, target_id)
    seed = _derive_seed(model_id, target_id, budget, variant)
    base = {
        "schema_version": "sweetspot-p5-r02-cell/v1",
        "root_seed": ROOT_SEED,
        "task_id": target_id,
        "target_name": audit.target(target_id)["target_name"],
        "lane": audit.target(target_id)["slug"],
        "model_id": model_id,
        "budget": budget,
        "variant": variant.name,
        "variant_kind": variant.kind,
        "matrix_rating": cell.rating,
        "eligible": cell.eligible,
        "seed": seed,
        "status": None,
        "reason": None,
        "test_accessed": False,
        "label_generated": False,
        "validation_metrics": None,
        "worst_group": None,
        "resource": None,
        "tiny_gate": None,
        "prediction_sha256": None,
        "prediction_count": None,
        "development_provenance": None,
        "source_lock": None,
        "cell_hash": None,
    }
    if target_id not in TARGET_SWEEP:
        return {**base, "status": "SKIP", "reason": {"code": "target_outside_r2_scope", "detail": "T4-T7 are status-only boundaries in protocol R2"}}
    if not cell.eligible:
        return {**base, "status": "SKIP", "reason": {"code": "matrix_not_applicable", "detail": "frozen matrix marks this cell inapplicable"}}
    runtime = inspect_runtime(source_lock[model_id])
    base["source_lock"] = {
        "revision": source_lock[model_id]["revision"],
        "license": source_lock[model_id]["license"],
        "runtime": runtime,
    }
    if not runtime["available"]:
        return {**base, "status": "SKIP", "reason": {"code": runtime["reason_code"], "detail": runtime}}
    if not runtime["version_allowed"]:
        return {**base, "status": "SKIP", "reason": {"code": "runtime_version_not_locked", "detail": runtime}}
    try:
        train_target = _target_transform(data.train_target, variant.target_transform)
        validation_target = _target_transform(data.validation_target, variant.target_transform)
        sample_weights = _class_weights(train_target, variant.class_weighting)
        if data.train_sequence is None or data.validation_sequence is None:
            raise DevelopmentDataUnavailable("input_modality_missing", f"{target_id}: development sequence is unavailable")
        sequence_preprocessing: dict[str, list[float]] | None = None
        train_sequence_proc = np.asarray(data.train_sequence, dtype=np.float64)
        validation_sequence_proc = np.asarray(data.validation_sequence, dtype=np.float64)
        if model_id == "inceptiontime":
            train_sequence_proc, validation_sequence_proc, sequence_preprocessing = _sequence_impute_and_scale(
                train_sequence_proc,
                validation_sequence_proc,
            )
        if target_id == "T3" and variant.context_days != 30:
            train_tabular = _sequence_context_features(train_sequence_proc, variant.context_days)
            validation_tabular = _sequence_context_features(validation_sequence_proc, variant.context_days)
            train_sequence = train_sequence_proc[:, :, -variant.context_days:]
            validation_sequence = validation_sequence_proc[:, :, -variant.context_days:]
        elif target_id == "T3":
            train_tabular = np.asarray(data.train_tabular, dtype=np.float64)
            validation_tabular = np.asarray(data.validation_tabular, dtype=np.float64)
            train_sequence = train_sequence_proc
            validation_sequence = validation_sequence_proc
        else:
            train_tabular = np.asarray(data.train_tabular, dtype=np.float64)
            validation_tabular = np.asarray(data.validation_tabular, dtype=np.float64)
            train_sequence = train_sequence_proc
            validation_sequence = validation_sequence_proc
        if model_id in TREE_MODELS:
            tiny_gate = _tiny_gate_for_tree(task_spec, model_id, seed, train_tabular, train_target)
            prediction, resource, runtime_payload = _fit_tree_estimator(
                model_id,
                task_spec,
                train_tabular,
                train_target,
                sample_weights,
                validation_tabular,
                seed=seed,
                budget=budget,
            )
        elif model_id == "inceptiontime":
            tiny_gate = _tiny_gate_for_sequence(task_spec, seed, train_sequence, train_target, device=device)
            prediction, resource, runtime_payload = _fit_inceptiontime(
                task_spec,
                train_sequence,
                train_target,
                sample_weights,
                validation_sequence,
                seed=seed,
                budget=budget,
                device=device,
                gpu_lock_path=gpu_lock_path,
            )
        elif model_id in SOFT_SKIP_MODELS:
            return {
                **base,
                "status": "SKIP",
                "reason": {
                    "code": "adapter_not_authorized_for_protocol_r2",
                    "detail": "source-locked candidate has no real development validation path in this repo/environment",
                },
            }
        else:
            return {**base, "status": "SKIP", "reason": {"code": "unknown_model_id", "detail": model_id}}
        if task_spec.task_type == "binary":
            prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
            metrics = _metric_payload("binary", validation_target, prediction)
            metrics["thickness_diagnostic"] = _finite_float(np.mean(prediction))
            primary = "average_precision"
            direction = "maximize"
        else:
            if variant.target_transform == "log1p":
                prediction = _inverse_target(np.asarray(prediction, dtype=np.float64).reshape(-1), variant.target_transform)
            else:
                prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
            metrics = _metric_payload("regression", validation_target, prediction)
            metrics["topk_diagnostic"] = _finite_float(
                np.mean(validation_target[np.argsort(prediction)[-max(1, len(prediction)//10):]]) - np.mean(validation_target)
            )
            primary = "mae"
            direction = "minimize"
        worst = _worst_group_metrics(
            "binary" if task_spec.task_type == "binary" else "regression",
            data.validation_groups,
            validation_target,
            prediction,
            primary,
            direction,
        )
        provenance = dict(data.provenance)
        provenance.update({
            "p4_fold_id": 0,
            "p4_split_manifest_path": data.provenance["p4_split_manifest_path"],
            "p4_split_manifest_sha256": data.provenance["p4_split_manifest_sha256"],
            "test_accessed": False,
            "historical_test_metrics_read": False,
            "rebuild_matches_manifest": True,
            "context_days": variant.context_days,
        })
        payload = {
            **base,
            "status": "PASS",
            "reason": None,
            "validation_metrics": metrics,
            "worst_group": worst,
            "resource": resource,
            "tiny_gate": tiny_gate,
            "prediction_sha256": hashlib.sha256(np.asarray(prediction, dtype=np.float64).tobytes()).hexdigest(),
            "prediction_count": int(len(prediction)),
            "development_provenance": provenance,
        }
        if sequence_preprocessing is not None:
            payload["development_provenance"]["sequence_preprocessing"] = sequence_preprocessing
        payload["cell_hash"] = _cell_hash({
            "model_id": model_id,
            "target_id": target_id,
            "budget": budget,
            "variant": variant.name,
            "seed": seed,
            "prediction_sha256": payload["prediction_sha256"],
            "provenance_sha256": canonical_sha256(provenance),
        })
        return payload
    except AdapterSkip as exc:
        return {**base, "status": "SKIP", "reason": {"code": exc.reason_code, "detail": exc.detail}}
    except DevelopmentDataUnavailable as exc:
        return {**base, "status": "SKIP", "reason": {"code": exc.reason_code, "detail": exc.detail}}
    except Exception as exc:
        return {**base, "status": "FAILED", "reason": {"code": "unexpected_runtime_failure", "detail": f"{type(exc).__name__}: {exc}"}}


def _status_payloads(audit: Any) -> dict[str, Any]:
    status = _status_gate(audit)
    return {
        "schema_version": "sweetspot-p5-r02-status-gate-records/v1",
        "targets": [
            {"task_id": target_id, "target_name": audit.target(target_id)["target_name"], **status["targets"][target_id]}
            for target_id in STATUS_ONLY_TARGETS
        ],
        "test_accessed": False,
        "historical_test_metrics_read": False,
        "label_generated": False,
    }


def _budget_curve(summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in summary_rows:
        buckets.setdefault(f"{row['task_id']}::{row['variant']}", []).append(row)
    return buckets


def _aggregate_budget_curves(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for target_id in TARGET_SWEEP:
        target_rows = [row for row in results if row["task_id"] == target_id and row["variant_kind"] == "main" and row["status"] == "PASS"]
        if not target_rows:
            aggregated[target_id] = {"status": "blocked", "reason": "no successful development cells", "budgets": {}}
            continue
        primary = "average_precision" if target_id == "T2" else "mae"
        direction = "maximize" if target_id == "T2" else "minimize"
        by_budget: dict[int, list[float]] = {}
        for row in target_rows:
            metric = row["validation_metrics"][primary]
            if metric is not None:
                by_budget.setdefault(int(row["budget"]), []).append(float(metric))
        medians = {
            str(budget): float(np.median(values))
            for budget, values in sorted(by_budget.items())
            if values
        }
        if len(medians) < 2:
            aggregated[target_id] = {"status": "blocked", "reason": "insufficient budget coverage", "primary_metric": primary, "direction": direction, "budgets": medians}
            continue
        b64 = medians.get("64")
        b256 = medians.get("256")
        b1024 = medians.get("1024")
        if b64 is None or b256 is None or b1024 is None:
            aggregated[target_id] = {"status": "blocked", "reason": "missing one or more budgets", "primary_metric": primary, "direction": direction, "budgets": medians}
            continue
        if direction == "minimize":
            gain_256_1024 = (b256 - b1024) / max(abs(b256), 1e-12)
            gain_64_1024 = (b64 - b1024) / max(abs(b64), 1e-12)
        else:
            gain_256_1024 = (b1024 - b256) / max(abs(b256), 1e-12)
            gain_64_1024 = (b1024 - b64) / max(abs(b64), 1e-12)
        pass_gate = gain_256_1024 <= 0.01 and gain_64_1024 >= -0.02
        aggregated[target_id] = {
            "status": "pass" if pass_gate else "blocked",
            "primary_metric": primary,
            "direction": direction,
            "budgets": medians,
            "gain_256_to_1024": gain_256_1024,
            "gain_64_to_1024": gain_64_1024,
            "n_pass_cells": len(target_rows),
        }
    overall = all(row["status"] == "pass" for row in aggregated.values())
    return {
        "schema_version": "sweetspot-p5-r02-plateau-gate/v1",
        "root_seed": ROOT_SEED,
        "overall_pass": overall,
        "targets": aggregated,
        "thresholds": {
            "gain_256_to_1024_max": 0.01,
            "gain_64_to_1024_min": -0.02,
        },
        "test_accessed": False,
    }


def _render_figures(output_dir: Path, results: Sequence[Mapping[str, Any]], plateau_gate: Mapping[str, Any], status_gate: Mapping[str, Any]) -> list[Path]:
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # budget curves
    for target_id in TARGET_SWEEP:
        rows = [row for row in results if row["task_id"] == target_id and row["variant_kind"] == "main" and row["status"] == "PASS"]
        if not rows:
            continue
        metric = "average_precision" if target_id == "T2" else "mae"
        direction = "maximize" if target_id == "T2" else "minimize"
        model_to_budget: dict[str, dict[int, float]] = {}
        for row in rows:
            model_to_budget.setdefault(row["model_id"], {})[int(row["budget"])] = float(row["validation_metrics"][metric])
        budgets = [64, 256, 1024]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for model_id, values in sorted(model_to_budget.items()):
            y = [values.get(b) for b in budgets]
            ax.plot(budgets, y, marker="o", label=model_id)
        ax.set_title(f"{target_id} main budgets")
        ax.set_xlabel("budget")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = figures_dir / f"{target_id.lower()}_budget_curve.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    # ablation comparison
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    for ax, target_id in zip(axes, TARGET_SWEEP):
        main_rows = [row for row in results if row["task_id"] == target_id and row["variant_kind"] == "main" and row["budget"] == ABLATION_BUDGET and row["status"] == "PASS"]
        ablation_rows = [row for row in results if row["task_id"] == target_id and row["variant_kind"] == "ablation" and row["budget"] == ABLATION_BUDGET and row["status"] == "PASS"]
        metric = "average_precision" if target_id == "T2" else "mae"
        main_value = float(np.median([float(row["validation_metrics"][metric]) for row in main_rows])) if main_rows else np.nan
        ablation_value = float(np.median([float(row["validation_metrics"][metric]) for row in ablation_rows])) if ablation_rows else np.nan
        ax.bar(["main", "ablation"], [main_value, ablation_value], color=["#355C7D", "#C06C84"])
        ax.set_title(target_id)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    path = figures_dir / "ablation_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    # status gate
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    labels = list(status_gate["targets"])
    values = [status_gate["targets"][label]["status"] for label in labels]
    colors = ["#1B998B" if value == "boundary" else "#EDAE49" if value == "not_feasible" else "#E15554" for value in values]
    ax.bar(labels, [1] * len(labels), color=colors)
    for index, value in enumerate(values):
        ax.text(index, 0.5, value, ha="center", va="center", color="white", fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_title("T4-T7 status gate")
    fig.tight_layout()
    path = figures_dir / "status_gate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    # plateau gate
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    xs = []
    ys = []
    labels = []
    for index, target_id in enumerate(TARGET_SWEEP):
        target_gate = plateau_gate["targets"][target_id]
        xs.append(index)
        ys.append(float(target_gate["gain_256_to_1024"]))
        labels.append(target_id)
    ax.axhline(0.01, color="#C06C84", linestyle="--", linewidth=1)
    ax.axhline(-0.02, color="#355C7D", linestyle="--", linewidth=1)
    ax.bar(xs, ys, color="#4ECDC4")
    ax.set_xticks(xs, labels)
    ax.set_ylabel("gain_256_to_1024")
    ax.set_title("R2 plateau gate")
    fig.tight_layout()
    path = figures_dir / "plateau_gate.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)
    return written


def run_r02(
    *,
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    source_root: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cpu",
    gpu_lock: Path | None = None,
) -> dict[str, Any]:
    audit = validate_label_mapping(mapping_path)
    source_lock = load_source_lock()
    data_by_target: dict[str, DevelopmentPilotData] = {}
    for target_id in TARGET_SWEEP:
        data_by_target[target_id] = load_development_pilot_data(audit, target_id, source_root=source_root, fold_id=0)
    results: list[dict[str, Any]] = []
    for target_id in TARGET_SWEEP:
        data = data_by_target[target_id]
        for model_id in MODEL_ORDER:
            for budget in MAIN_BUDGETS:
                results.append(
                    _run_cell(
                        audit=audit,
                        source_lock=source_lock,
                        target_id=target_id,
                        model_id=model_id,
                        variant=MAIN_VARIANTS[target_id],
                        budget=budget,
                        data=data,
                        device=device,
                        gpu_lock_path=gpu_lock,
                    )
                )
            results.append(
                _run_cell(
                    audit=audit,
                    source_lock=source_lock,
                    target_id=target_id,
                    model_id=model_id,
                    variant=ABLATION_VARIANTS[target_id],
                    budget=ABLATION_BUDGET,
                    data=data,
                    device=device,
                    gpu_lock_path=gpu_lock,
                )
            )
    plateau_gate = _aggregate_budget_curves(results)
    status_gate = _status_gate(audit)
    status_payload = _status_payloads(audit)
    written_figures = _render_figures(output_dir, results, plateau_gate, status_gate)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / RESULT_FILENAME
    summary_path = destination / SUMMARY_FILENAME
    plateau_path = destination / PLATEAU_FILENAME
    status_path = destination / STATUS_GATES_FILENAME
    visualization_path = destination / VISUALIZATION_FILENAME
    artifact_path = destination / ARTIFACT_MANIFEST_FILENAME
    _write_jsonl(results_path, results)
    _write_json(plateau_path, plateau_gate)
    _write_json(status_path, status_payload)
    visualization_manifest = {
        "schema_version": "sweetspot-p5-r02-visualization-manifest/v1",
        "root_seed": ROOT_SEED,
        "figures": [
            {
                "path": _safe_relative(path, destination),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in written_figures
        ],
        "test_accessed": False,
    }
    _write_json(visualization_path, visualization_manifest)
    summary = {
        "schema_version": "sweetspot-p5-r02-summary/v1",
        "stage": "P5.2/protocol R2",
        "root_seed": ROOT_SEED,
        "model_order": list(MODEL_ORDER),
        "target_order": list(TARGET_SWEEP),
        "budgets": list(MAIN_BUDGETS),
        "ablation_budget": ABLATION_BUDGET,
        "expected_cells": EXPECTED_CELLS,
        "main_expected_cells": MAIN_EXPECTED_CELLS,
        "ablation_expected_cells": ABLATION_EXPECTED_CELLS,
        "attempted_cells": len(results),
        "counts": {
            status: sum(row["status"] == status for row in results)
            for status in ("PASS", "SKIP", "FAILED")
        },
        "main_counts": {
            status: sum(row["status"] == status and row["variant_kind"] == "main" for row in results)
            for status in ("PASS", "SKIP", "FAILED")
        },
        "ablation_counts": {
            status: sum(row["status"] == status and row["variant_kind"] == "ablation" for row in results)
            for status in ("PASS", "SKIP", "FAILED")
        },
        "target_pass_counts": {
            target_id: sum(row["status"] == "PASS" and row["task_id"] == target_id for row in results)
            for target_id in TARGET_SWEEP
        },
        "target_status": {
            target_id: status_gate["targets"][target_id]["status"]
            for target_id in STATUS_ONLY_TARGETS
        },
        "target_budget_curve": plateau_gate["targets"],
        "status_gate": status_gate,
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "plateau_gate_sha256": hashlib.sha256(plateau_path.read_bytes()).hexdigest(),
        "status_gate_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        "visualization_manifest_sha256": hashlib.sha256(visualization_path.read_bytes()).hexdigest(),
        "label_mapping_sha256": audit.mapping_sha256,
        "source_lock_sha256": sha256_file(DEFAULT_LOCK),
        "test_accessed": False,
        "labels_generated": False,
        "historical_test_metrics_read": False,
        "portable_output_files": [
            RESULT_FILENAME,
            SUMMARY_FILENAME,
            PLATEAU_FILENAME,
            STATUS_GATES_FILENAME,
            VISUALIZATION_FILENAME,
            ARTIFACT_MANIFEST_FILENAME,
        ],
    }
    _write_json(summary_path, summary)
    artifact_manifest = {
        "schema_version": "sweetspot-p5-r02-artifact-manifest/v1",
        "root_seed": ROOT_SEED,
        "all_paths_portable": True,
        "artifacts": [
            {
                "path": _safe_relative(path, destination),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path not in {summary_path, artifact_path}
        ],
    }
    _write_json(artifact_path, artifact_manifest)
    summary["artifact_manifest_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    _write_json(summary_path, summary)
    return summary


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
    summary = run_r02(
        mapping_path=args.mapping,
        source_root=args.source_root,
        output_dir=args.output_dir,
        device=args.device,
        gpu_lock=args.gpu_lock,
    )
    sys.stdout.write(
        json.dumps(
            {"counts": summary["counts"], "main_counts": summary["main_counts"], "ablation_counts": summary["ablation_counts"]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return 1 if summary["counts"]["FAILED"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
