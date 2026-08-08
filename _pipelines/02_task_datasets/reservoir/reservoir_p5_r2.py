#!/usr/bin/env python3
"""P5.2 / protocol R2 learning curves and single-factor ablations for property."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import reservoir_p5_stage3 as stage3  # noqa: E402
import reservoir_p5_stage4 as stage4  # noqa: E402
import reservoir_p5_stage2 as stage2  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.property._p5_common import (  # noqa: E402
    PROPERTY_TARGETS,
    Stage1GateError,
    source_lock_sha256,
)
from p5_contract import build_task_spec  # noqa: E402


ROOT_SEED = 2693
DEVELOPMENT_FAMILIES = ("15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12")
REPEAT_SEEDS = stage3.REPEAT_SEEDS
MODELS = ("tabm_regressor", "tabiclv2_regressor", "monai_densenet3d_regressor")
LANES = {
    "tabm_regressor": "scratch_flat_fusion",
    "tabiclv2_regressor": "tabular_pretrained",
    "monai_densenet3d_regressor": "seismic_3d_gpu",
}
DEVICE_BY_MODEL = {
    "tabm_regressor": "cpu",
    "tabiclv2_regressor": "cpu",
    "monai_densenet3d_regressor": "cuda",
}
LOSS_OUTPUT_CURVE = ("mse", "identity")
ABLATION_VARIANTS = (
    ("mse", "identity"),
    ("mae", "identity"),
    ("huber", "identity"),
    ("mse", "bounded"),
    ("huber", "bounded"),
)
CURVE_BUDGETS = {
    "tabm_regressor": (40, 160, 640),
    "tabiclv2_regressor": (1, 2, 4),
    "monai_densenet3d_regressor": (12, 48, 192),
}
ABLATED_MODELS = {"tabm_regressor", "monai_densenet3d_regressor"}
MODEL_DEFAULTS = {
    "tabm_regressor": {
        "arch_type": "tabm-mini",
        "k": 4,
        "n_blocks": 2,
        "d_block": 64,
        "learning_rate": 0.002,
        "weight_decay": 3e-4,
    },
    "tabiclv2_regressor": {
        "batch_size": 1,
        "offload_mode": "auto",
    },
    "monai_densenet3d_regressor": {
        "init_features": 8,
        "growth_rate": 8,
        "block_config": (2, 2),
        "bn_size": 2,
        "dropout_prob": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
    },
}
DEFAULT_ARCHIVE = HERE / "_outputs" / "p5_r2" / "runtime" / "development_logo4.npz"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p5_r2"
DEFAULT_TABICL_CHECKPOINT = None
R2_CONTRACT_PATH = HERE / "reservoir_p5_r2_contract.json"
STAGE3_RESULTS = HERE / "_outputs" / "p5_stage3" / "p5_stage3_results.jsonl"
STAGE3_SUMMARY = HERE / "_outputs" / "p5_stage3" / "p5_stage3_summary.json"
STAGE3_SPLIT = HERE / "_outputs" / "p5_stage3" / "p5_stage3_split_manifest.json"
STAGE3_BUDGET = HERE / "reservoir_p5_stage3_budget.json"
STAGE3_VISUALIZATION = HERE / "_outputs" / "p5_stage3" / "p5_stage3_visualization_manifest.json"
STAGE4_CONTRACT = HERE / "reservoir_p5_stage4_contract.json"


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    return resolved.relative_to(root.resolve()).as_posix()


def load_r2_contract() -> dict[str, Any]:
    stage4_contract = stage4.load_contract()
    current_source_lock_sha256 = source_lock_sha256()
    historical_source_lock_sha256 = stage4_contract["stage3_locks"]["source_lock_sha256"]
    preflight: dict[str, Any] = {
        "stage4_contract_sha256": _hash_file(STAGE4_CONTRACT),
        "historical_source_lock_sha256": historical_source_lock_sha256,
        "current_source_lock_sha256": current_source_lock_sha256,
        "historical_source_lock_mismatch": current_source_lock_sha256 != historical_source_lock_sha256,
        "stage4_validation_status": "unvalidated",
        "stage4_validation_error": None,
    }
    if preflight["historical_source_lock_mismatch"]:
        preflight["stage4_validation_status"] = "historical_source_lock_mismatch"
    try:
        stage4.validate_stage3_and_contract()
        if not preflight["historical_source_lock_mismatch"]:
            preflight["stage4_validation_status"] = "verified"
    except RuntimeError as error:
        message = str(error)
        if "source-lock helper disagrees" in message or "Stage-3 source_lock" in message:
            preflight["stage4_validation_status"] = "historical_source_lock_mismatch"
            preflight["stage4_validation_error"] = {
                "type": type(error).__name__,
                "message": message,
            }
        else:
            preflight["stage4_validation_status"] = "blocked"
            preflight["stage4_validation_error"] = {
                "type": type(error).__name__,
                "message": message,
            }
            raise
    contract = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "protocol": "P5.2_R2",
        "baseline_head": "af8c066de0c3fc24fce024abb350b4f2c9e82d9b",
        "root_seed": ROOT_SEED,
        "development_families": list(DEVELOPMENT_FAMILIES),
        "repeat_seeds": list(REPEAT_SEEDS),
        "lanes": {
            "scratch_flat_fusion": "tabm_regressor",
            "tabular_pretrained": "tabiclv2_regressor",
            "seismic_3d_gpu": "monai_densenet3d_regressor",
        },
        "curve_budgets": {model_id: list(budgets) for model_id, budgets in CURVE_BUDGETS.items()},
        "ablation_variants": {
            model_id: [list(variant) for variant in ABLATION_VARIANTS]
            if model_id in ABLATED_MODELS
            else []
            for model_id in MODELS
        },
        "preflight": preflight,
        "historical_stage4_contract": {
            "baseline_head": stage4_contract["baseline_head"],
            "known_holdout_family": stage4_contract["known_holdout_family"],
            "prior_test_consumed": stage4_contract["prior_test_consumed"],
            "fresh_blind": stage4_contract["fresh_blind"],
        },
        "stage3_contract_hashes": {
            "results": _hash_file(STAGE3_RESULTS),
            "summary": _hash_file(STAGE3_SUMMARY),
            "split_manifest": _hash_file(STAGE3_SPLIT),
            "visualization_manifest": _hash_file(STAGE3_VISUALIZATION),
            "source_lock_sha256": source_lock_sha256(),
        },
        "source_lock_sha256": source_lock_sha256(),
        "runtime_policy": {
            "test_access": False,
            "no_frozen_test": True,
            "no_hpo": True,
            "single_factor_ablation": True,
            "portable_only": True,
        },
    }
    contract["contract_sha256"] = _hash_payload(contract)
    return contract


def _default_paths() -> tuple[Path, Path, Path]:
    train_h5 = PROJECT_ROOT / "_data" / "processed" / "reservoir" / "train.h5"
    guard_npz = HERE / "_outputs" / "guard.npz"
    return train_h5, guard_npz, DEFAULT_ARCHIVE


def prepare_development_archive(
    train_h5: Path,
    guard_npz: Path,
    output_path: Path = DEFAULT_ARCHIVE,
) -> dict[str, Any]:
    manifest = stage3.prepare_logo4(
        Path(train_h5),
        Path(guard_npz),
        PROJECT_ROOT
        / "_pipelines/02_task_datasets/sweetspot/targets/porosity/_outputs/phif/split_manifest.json",
        Path(output_path),
    )
    result = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "root_seed": ROOT_SEED,
        "development_archive": _relative(output_path),
        "development_archive_sha256": _hash_file(output_path),
        "source": {
            "train_h5_sha256": _hash_file(train_h5),
            "guard_npz_sha256": _hash_file(guard_npz),
            "paths_persisted": False,
        },
        "split_hash": manifest["split_hash"],
        "test_firewall": manifest["test_firewall"],
    }
    result["contract_sha256"] = _hash_payload(result)
    _atomic_json(output_path.parent / "prepare_manifest.json", result)
    return result


def _load_batches(development_batch: Path) -> dict[int, tuple[Any, Any, dict[str, Any]]]:
    batches: dict[int, tuple[Any, Any, dict[str, Any]]] = {}
    for fold_id in range(4):
        batches[fold_id] = stage3.load_fold(Path(development_batch), fold_id)
    return batches


def _cell_key(
    *,
    model_id: str,
    phase: str,
    budget: int,
    fold_id: int,
    repeat_id: int,
    loss_name: str,
    output_activation: str,
) -> str:
    return (
        f"{model_id}__{phase}__budget{budget}__fold{fold_id}__repeat{repeat_id}"
        f"__loss-{loss_name}__output-{output_activation}"
    )


def _runtime_cell_paths(
    *,
    output_dir: Path,
    model_id: str,
    phase: str,
    budget: int,
    fold_id: int,
    repeat_id: int,
    loss_name: str,
    output_activation: str,
) -> tuple[Path, Path]:
    variant_parts = (
        (f"loss-{loss_name}__output-{output_activation}",)
        if phase == "ablation"
        else ()
    )
    checkpoint_path = (
        output_dir
        / "runtime"
        / "checkpoints"
        / model_id
        / phase
        / Path(*variant_parts)
        / f"budget{budget}"
        / f"fold{fold_id}__repeat{repeat_id}"
    )
    if model_id == "tabiclv2_regressor":
        checkpoint_path = checkpoint_path / "checkpoint"
    else:
        checkpoint_path = checkpoint_path.with_suffix(".bin")
    oof_path = (
        output_dir
        / "runtime"
        / "oof"
        / model_id
        / phase
        / Path(*variant_parts)
        / f"budget{budget}"
        / f"fold{fold_id}__repeat{repeat_id}.npz"
    )
    return checkpoint_path, oof_path


def _config_for(
    model_id: str,
    *,
    seed: int,
    budget: int,
    device: str,
    loss_name: str,
    output_activation: str,
    tabicl_checkpoint: Path | None,
) -> dict[str, Any]:
    config = {
        "seed": seed,
        "n_features": 153,
        "device": device,
        "loss_name": loss_name,
        "output_activation": output_activation,
        "huber_delta": 1.0,
    }
    config.update(MODEL_DEFAULTS[model_id])
    if model_id == "tabiclv2_regressor":
        config["n_estimators"] = int(budget)
        if tabicl_checkpoint is not None:
            config["checkpoint_path"] = str(tabicl_checkpoint)
    elif model_id == "tabm_regressor":
        config["learning_rate"] = float(MODEL_DEFAULTS[model_id]["learning_rate"])
        config["weight_decay"] = float(MODEL_DEFAULTS[model_id]["weight_decay"])
    elif model_id == "monai_densenet3d_regressor":
        config["learning_rate"] = float(MODEL_DEFAULTS[model_id]["learning_rate"])
        config["weight_decay"] = float(MODEL_DEFAULTS[model_id]["weight_decay"])
    return config


def _run_model_cell(
    *,
    model_id: str,
    phase: str,
    budget: int,
    loss_name: str,
    output_activation: str,
    fold_id: int,
    repeat_id: int,
    repeat_seed: int,
    train_batch: Any,
    validation_batch: Any,
    fold_evidence: Mapping[str, Any],
    output_dir: Path,
    tabicl_checkpoint: Path | None,
) -> dict[str, Any]:
    lane = LANES[model_id]
    task_spec = build_task_spec()
    discovered = discover_model("property", model_id)
    expected_modalities = {
        "tabm_regressor": ["tabular"],
        "tabiclv2_regressor": ["tabular"],
        "monai_densenet3d_regressor": ["seismic_patch"],
    }
    if list(discovered.capabilities["input_modalities"]) != expected_modalities[model_id]:
        raise RuntimeError(
            f"{model_id} capabilities changed: {discovered.capabilities['input_modalities']}"
        )
    cell_id = _cell_key(
        model_id=model_id,
        phase=phase,
        budget=budget,
        fold_id=fold_id,
        repeat_id=repeat_id,
        loss_name=loss_name,
        output_activation=output_activation,
    )
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoint_path, oof_path = _runtime_cell_paths(
        output_dir=output_dir,
        model_id=model_id,
        phase=phase,
        budget=budget,
        fold_id=fold_id,
        repeat_id=repeat_id,
        loss_name=loss_name,
        output_activation=output_activation,
    )
    device = DEVICE_BY_MODEL[model_id]
    config = _config_for(
        model_id,
        seed=repeat_seed,
        budget=budget,
        device=device,
        loss_name=loss_name,
        output_activation=output_activation,
        tabicl_checkpoint=tabicl_checkpoint,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "cell_id": cell_id,
        "model_id": model_id,
        "lane": lane,
        "phase": phase,
        "stage": 2,
        "budget": budget,
        "loss_name": loss_name,
        "output_activation": output_activation,
        "fold_id": fold_id,
        "repeat_id": repeat_id,
        "repeat_seed": repeat_seed,
        "status": "skipped",
        "reason": None,
        "seed": {"root": ROOT_SEED, "model": repeat_seed, "repeat_id": repeat_id},
        "split": {
            "split_hash": fold_evidence["split_hash"],
            "fold_id": fold_id,
            "train_groups": fold_evidence["train_groups"],
            "validation_groups": fold_evidence["validation_groups"],
            "train_sample_ids_sha256": fold_evidence["train_sample_ids_sha256"],
            "validation_sample_ids_sha256": fold_evidence["validation_sample_ids_sha256"],
        },
        "preprocessing": fold_evidence["preprocessing"],
        "training_budget": {
            "kind": "tree" if model_id == "tabiclv2_regressor" else "neural",
            "budget": budget,
            "budget_unit": "n_estimators" if model_id == "tabiclv2_regressor" else "optimizer_steps",
            "loss_name": loss_name,
            "output_activation": output_activation,
            "fit_calls": 0,
            "hpo": False,
        },
        "validation": {},
        "checkpoint": {
            "roundtrip": False,
            "sha256": None,
            "path_persisted": False,
            "git_ignored": True,
        },
        "oof": {
            "rows": 0,
            "sha256": None,
            "relative_path": _relative(oof_path, output_dir),
            "git_ignored": True,
        },
        "resources": {
            "wall_seconds": 0.0,
            "max_rss_kib_end": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "max_rss_kib_delta_lower_bound": 0,
            "peak_cuda_bytes": 0,
            "gpu_lock_wait_seconds_excluded_from_wall": 0.0,
            "download_bytes": 0,
            "device": device,
        },
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family_seen": False,
        },
    }

    try:
        if model_id == "tabiclv2_regressor":
            if tabicl_checkpoint is None:
                raise Stage1GateError(
                    "weight_checkpoint_missing",
                    "TabICLv2 requires a locally provisioned checkpoint for R2",
                    model_id=model_id,
                    auto_download=False,
                )
            config["checkpoint_path"] = str(tabicl_checkpoint)

        if model_id == "monai_densenet3d_regressor":
            with stage2.gpu_flock(device, stage2.GPU_LOCK_PATH):
                result["resources"]["gpu_lock_wait_seconds_excluded_from_wall"] = 0.0
                model = discovered.build(task_spec, **config)
                fit_reports = []
                for _ in range(int(budget)):
                    fit_reports.append(model.fit(train_batch))
                output = model.predict(validation_batch)
        else:
            model = discovered.build(task_spec, **config)
            fit_reports = []
            if model_id == "tabiclv2_regressor":
                fit_reports.append(model.fit(train_batch))
            else:
                for _ in range(int(budget)):
                    fit_reports.append(model.fit(train_batch))
            output = model.predict(validation_batch)

        prediction_matrix = stage2._raw_matrix(output)
        metric = stage2.evaluate_targets(validation_batch, output)
        oof_payload: dict[str, Any] = {}
        total_rows = 0
        for index, target in enumerate(PROPERTY_TARGETS):
            mask = np.asarray(validation_batch.target_masks[target], dtype=bool)
            target_ids = np.asarray(validation_batch.sample_ids)[mask]
            target_families = np.asarray(validation_batch.groups["mother_well_family"])[mask]
            target_wells = np.asarray(validation_batch.groups["well_id"])[mask]
            target_depths = np.asarray(validation_batch.coordinates["depth_m"])[mask]
            truth_domain = np.asarray(validation_batch.targets[target], dtype=np.float64)[mask]
            prediction_domain = prediction_matrix[mask, index]
            truth_physical = stage2.model_to_physical(target, truth_domain, prediction=False)
            prediction_physical = np.asarray(output.transformed[target], dtype=np.float64)[mask]
            oof_payload[f"{target.lower()}_sample_ids"] = target_ids
            oof_payload[f"{target.lower()}_family_ids"] = target_families
            oof_payload[f"{target.lower()}_well_ids"] = target_wells
            oof_payload[f"{target.lower()}_depths_m"] = target_depths
            oof_payload[f"{target.lower()}_truth_model_domain"] = truth_domain
            oof_payload[f"{target.lower()}_prediction_model_domain"] = prediction_domain
            oof_payload[f"{target.lower()}_truth_physical"] = truth_physical
            oof_payload[f"{target.lower()}_prediction_physical"] = prediction_physical
            total_rows += int(mask.sum())
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(oof_path, **oof_payload)
        model.save_checkpoint(checkpoint_path)
        restored = discovered.build(task_spec, **config)
        restored.load_checkpoint(checkpoint_path)
        restored_prediction = stage2._raw_matrix(restored.predict(validation_batch))
        if not np.allclose(prediction_matrix, restored_prediction, rtol=1e-6, atol=1e-7):
            raise RuntimeError("checkpoint round-trip changed predictions")
        wall_seconds = time.perf_counter() - started
        result.update(
            {
                "status": "completed",
                "reason": None,
                "fit_report": fit_reports[-1] if fit_reports else {},
                "validation": {
                    "targets": metric,
                    "independent_target_masks": True,
                    "mother_well_families": sorted(set(validation_batch.groups["mother_well_family"])),
                },
                "checkpoint": {
                    "roundtrip": True,
                    "sha256": stage2._checkpoint_hash(checkpoint_path),
                    "path_persisted": False,
                    "git_ignored": True,
                },
                "oof": {
                    "rows": int(total_rows),
                    "sha256": _hash_file(oof_path),
                    "relative_path": _relative(oof_path, output_dir),
                    "git_ignored": True,
                },
                "resources": {
                    "wall_seconds": wall_seconds,
                    "max_rss_kib_end": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "max_rss_kib_delta_lower_bound": int(
                        max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before)
                    ),
                    "peak_cuda_bytes": 0,
                    "gpu_lock_wait_seconds_excluded_from_wall": 0.0,
                    "download_bytes": 0,
                    "device": device,
                },
                "test_firewall": {
                    "test_access": False,
                    "test_loader_implemented": False,
                    "test_metrics": False,
                    "frozen_test_family_seen": False,
                },
            }
        )
    except Stage1GateError as error:
        result.update(
            {
                "status": "skipped",
                "reason": error.to_dict(),
                "fit_report": {},
                "validation": {},
            }
        )
    except TimeoutError as error:
        result.update(
            {
                "status": "timeout",
                "reason": {"type": type(error).__name__, "message": str(error)},
                "fit_report": {},
                "validation": {},
            }
        )
    except Exception as error:  # pragma: no cover - surfaced through the CLI and tests
        result.update(
            {
                "status": "failed",
                "reason": {
                    "type": type(error).__name__,
                    "message": str(error).replace(str(PROJECT_ROOT), "<project>"),
                },
                "fit_report": {},
                "validation": {},
            }
        )
    return result


def _run_cell_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return _run_model_cell(**spec)


def _run_cell_specs(specs: Sequence[Mapping[str, Any]], *, parallel_workers: int = 1) -> list[dict[str, Any]]:
    if not specs:
        return []
    if parallel_workers <= 1:
        return [_run_cell_spec(spec) for spec in specs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        return list(executor.map(_run_cell_spec, specs))


def _group_curve_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = {
        model_id: {target: defaultdict(list) for target in PROPERTY_TARGETS}
        for model_id in MODELS
    }
    worst = {
        model_id: {target: defaultdict(list) for target in PROPERTY_TARGETS}
        for model_id in MODELS
    }
    for row in rows:
        if row.get("status") != "completed" or row.get("phase") != "curve":
            continue
        model_id = str(row["model_id"])
        if model_id not in grouped:
            raise RuntimeError(f"unexpected property curve model {model_id!r}")
        budget = int(row["budget"])
        for target in PROPERTY_TARGETS:
            metric = row["validation"]["targets"][target]
            grouped[model_id][target][budget].append(float(metric["physical"]["RMSE"]))
            worst[model_id][target][budget].append(
                float(metric["worst_mother_family"]["physical"]["RMSE"])
            )
    result: dict[str, Any] = {}
    for model_id in MODELS:
        result[model_id] = {}
        for target in PROPERTY_TARGETS:
            target_curve = {}
            for budget in sorted(grouped[model_id][target]):
                target_curve[str(budget)] = {
                    "mean_physical_RMSE": float(np.mean(grouped[model_id][target][budget])),
                    "mean_worst_family_RMSE": float(
                        np.mean(worst[model_id][target][budget])
                    ),
                    "completed_cells": len(grouped[model_id][target][budget]),
                }
            result[model_id][target] = target_curve
    return result


def _pooled_curve_metrics(curve_metrics: Mapping[str, Any]) -> dict[str, Any]:
    budgets = sorted(
        {int(budget) for target in PROPERTY_TARGETS for budget in curve_metrics[target]}
    )
    pooled: dict[str, Any] = {}
    for budget in budgets:
        rmse_values: list[float] = []
        worst_values: list[float] = []
        completed: list[int] = []
        for target in PROPERTY_TARGETS:
            target_curve = curve_metrics[target]
            if str(budget) not in target_curve:
                continue
            rmse_values.append(float(target_curve[str(budget)]["mean_physical_RMSE"]))
            worst_values.append(float(target_curve[str(budget)]["mean_worst_family_RMSE"]))
            completed.append(int(target_curve[str(budget)]["completed_cells"]))
        if rmse_values:
            pooled[str(budget)] = {
                "mean_physical_RMSE": float(np.mean(rmse_values)),
                "mean_worst_family_RMSE": float(np.mean(worst_values)),
                "completed_cells": int(np.sum(completed)),
            }
    return pooled


def _plateau_budget(curve: Mapping[str, Any]) -> dict[str, Any]:
    budgets = sorted(int(key) for key in curve)
    if not budgets:
        return {"status": "not_rankable", "budget": None, "reason": "no completed curve rows"}
    if len(budgets) == 1:
        return {"status": "plateaued", "budget": budgets[0], "reason": "single budget"}

    def rel_improvement(prev: float, current: float) -> float:
        if prev == 0.0:
            return 0.0 if current == 0.0 else float("inf")
        return max(0.0, (prev - current) / abs(prev))

    metrics = [
        (
            budget,
            float(curve[str(budget)]["mean_physical_RMSE"]),
            float(curve[str(budget)]["mean_worst_family_RMSE"]),
        )
        for budget in budgets
    ]
    low, mid, high = metrics[0], metrics[1], metrics[-1]
    low_mid_ok = rel_improvement(low[1], mid[1]) < 0.01 and rel_improvement(low[2], mid[2]) < 0.01
    mid_high_ok = rel_improvement(mid[1], high[1]) < 0.01 and rel_improvement(mid[2], high[2]) < 0.01
    if low_mid_ok and mid_high_ok:
        return {"status": "plateaued", "budget": low[0], "reason": "all higher budgets <1% improvement"}
    if mid_high_ok:
        return {"status": "plateaued", "budget": mid[0], "reason": "higher budget <1% improvement"}
    return {
        "status": "not_plateaued",
        "budget": high[0],
        "reason": "highest budget still improves by >=1%",
    }


def _render_budget_curve(
    output_dir: Path,
    model_id: str,
    target: str,
    curve: Mapping[str, Any],
    plateau: Mapping[str, Any],
) -> Path:
    budgets = sorted(int(key) for key in curve)
    values = [curve[str(budget)]["mean_physical_RMSE"] for budget in budgets]
    worst = [curve[str(budget)]["mean_worst_family_RMSE"] for budget in budgets]
    fig, axis = plt.subplots(figsize=(5.5, 4.0))
    axis.plot(budgets, values, marker="o", label="pooled physical RMSE")
    axis.plot(budgets, worst, marker="s", label="worst-family RMSE")
    if plateau.get("budget") is not None:
        axis.axvline(float(plateau["budget"]), linestyle="--", color="#444444", label="plateau / selected")
    axis.set_xscale("log")
    axis.set_xlabel("budget")
    axis.set_ylabel("RMSE")
    axis.set_title(f"{model_id} - {target}")
    axis.grid(alpha=0.25)
    axis.legend()
    path = output_dir / "figures" / model_id / f"{target.lower()}_budget_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _render_ablation_heatmap(
    output_dir: Path,
    model_id: str,
    target: str,
    rows: Sequence[Mapping[str, Any]],
    selected_budget: int,
) -> Path | None:
    if model_id not in ABLATED_MODELS:
        return None
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed" or row.get("phase") != "ablation":
            continue
        if row.get("model_id") != model_id:
            continue
        if int(row["budget"]) != selected_budget:
            continue
        metric = row["validation"]["targets"][target]["physical"]["RMSE"]
        values[(row["loss_name"], row["output_activation"])].append(float(metric))
    if not values:
        return None
    losses = ["mse", "mae", "huber"]
    outputs = ["identity", "bounded"]
    matrix = np.full((len(losses), len(outputs)), np.nan, dtype=np.float64)
    for row_idx, loss in enumerate(losses):
        for col_idx, output in enumerate(outputs):
            cell = values.get((loss, output))
            if cell:
                matrix[row_idx, col_idx] = float(np.mean(cell))
    fig, axis = plt.subplots(figsize=(4.8, 3.2))
    im = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(outputs)), labels=outputs)
    axis.set_yticks(range(len(losses)), labels=losses)
    axis.set_title(f"{model_id} {target} selected-budget ablation")
    fig.colorbar(im, ax=axis, shrink=0.8, label="physical RMSE")
    path = output_dir / "figures" / model_id / f"{target.lower()}_ablation_heatmap.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _selected_budget_for_targets(
    *,
    model_id: str,
    curve_metrics: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pooled = _pooled_curve_metrics(curve_metrics)
    plateau = _plateau_budget(pooled)
    return pooled, plateau


def _write_markdown_table(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_oof_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "root_seed": ROOT_SEED,
        "entries": [
            {
                "cell_id": row["cell_id"],
                "model_id": row["model_id"],
                "phase": row["phase"],
                "budget": row["budget"],
                "fold_id": row["fold_id"],
                "repeat_id": row["repeat_id"],
                "status": row["status"],
                "oof": row["oof"],
                "checkpoint": row["checkpoint"],
                "split": row["split"],
            }
            for row in rows
        ],
    }
    payload["sha256"] = _hash_payload(payload)
    _atomic_json(path, payload)
    return payload


def run_r2(
    *,
    development_batch: Path = DEFAULT_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tabicl_checkpoint: Path | None = DEFAULT_TABICL_CHECKPOINT,
    repair_tabm_ablation: bool = False,
) -> dict[str, Any]:
    contract = load_r2_contract()
    if not Path(development_batch).is_file():
        raise FileNotFoundError(
            f"development archive missing: {development_batch}; run prepare first"
        )
    batches = _load_batches(Path(development_batch))
    rows: list[dict[str, Any]] = []
    if repair_tabm_ablation:
        existing_results = output_dir / "p5_r2_results.jsonl"
        if not existing_results.is_file():
            raise FileNotFoundError(
                f"repair requires completed original results: {existing_results}"
            )
        original_rows = [
            json.loads(line)
            for line in existing_results.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len({row["cell_id"] for row in original_rows}) != len(original_rows):
            raise RuntimeError("repair source contains duplicate R2 cell IDs")
        rows.extend(
            row
            for row in original_rows
            if not (
                row["phase"] == "ablation"
                and row["model_id"] in ABLATED_MODELS
            )
        )
    tabular_curve_specs: list[dict[str, Any]] = []
    monai_curve_specs: list[dict[str, Any]] = []
    if not repair_tabm_ablation:
        for model_id in MODELS:
            for budget in CURVE_BUDGETS[model_id]:
                for fold_id, (train, validation, evidence) in batches.items():
                    for repeat_id, repeat_seed in enumerate(REPEAT_SEEDS):
                        spec = {
                            "model_id": model_id,
                            "phase": "curve",
                            "budget": int(budget),
                            "loss_name": LOSS_OUTPUT_CURVE[0],
                            "output_activation": LOSS_OUTPUT_CURVE[1],
                            "fold_id": fold_id,
                            "repeat_id": repeat_id,
                            "repeat_seed": int(repeat_seed),
                            "train_batch": train,
                            "validation_batch": validation,
                            "fold_evidence": evidence,
                            "output_dir": output_dir,
                            "tabicl_checkpoint": tabicl_checkpoint,
                        }
                        if model_id == "monai_densenet3d_regressor":
                            monai_curve_specs.append(spec)
                        else:
                            tabular_curve_specs.append(spec)

        rows.extend(
            _run_cell_specs(tabular_curve_specs, parallel_workers=min(4, os.cpu_count() or 1))
        )
        rows.extend(_run_cell_specs(monai_curve_specs, parallel_workers=1))

    curve_metrics = _group_curve_metrics(rows)
    pooled_curve = {}
    plateau = {}
    for model_id in MODELS:
        pooled_curve[model_id], plateau[model_id] = _selected_budget_for_targets(
            model_id=model_id,
            curve_metrics=curve_metrics[model_id],
        )
    selected_budget = {model_id: plateau[model_id]["budget"] for model_id in MODELS}

    tabular_ablation_specs: list[dict[str, Any]] = []
    monai_ablation_specs: list[dict[str, Any]] = []
    for model_id in ABLATED_MODELS:
        if repair_tabm_ablation and model_id != "tabm_regressor":
            continue
        budget = selected_budget[model_id]
        if budget is None:
            continue
        for loss_name, output_activation in ABLATION_VARIANTS:
            for fold_id, (train, validation, evidence) in batches.items():
                for repeat_id, repeat_seed in enumerate(REPEAT_SEEDS):
                    spec = {
                        "model_id": model_id,
                        "phase": "ablation",
                        "budget": budget,
                        "loss_name": loss_name,
                        "output_activation": output_activation,
                        "fold_id": fold_id,
                        "repeat_id": repeat_id,
                        "repeat_seed": int(repeat_seed),
                        "train_batch": train,
                        "validation_batch": validation,
                        "fold_evidence": evidence,
                        "output_dir": output_dir,
                        "tabicl_checkpoint": tabicl_checkpoint,
                    }
                    if model_id == "monai_densenet3d_regressor":
                        monai_ablation_specs.append(spec)
                    else:
                        tabular_ablation_specs.append(spec)

    rows.extend(_run_cell_specs(tabular_ablation_specs, parallel_workers=min(4, os.cpu_count() or 1)))
    rows.extend(_run_cell_specs(monai_ablation_specs, parallel_workers=1))

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            row["model_id"],
            row["phase"],
            row["budget"],
            row["loss_name"],
            row["output_activation"],
            row["fold_id"],
            row["repeat_id"],
        ),
    )
    completed_curve_models = {
        row["model_id"]
        for row in ordered_rows
        if row["phase"] == "curve" and row["status"] == "completed"
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "p5_r2_results.jsonl"
    _atomic_text(
        results_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in ordered_rows
        ),
    )

    oof_manifest = _write_oof_manifest(output_dir / "p5_r2_oof_manifest.json", ordered_rows)
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    for model_id in MODELS:
        plateau_info = plateau[model_id]
        for target in PROPERTY_TARGETS:
            curve = curve_metrics[model_id][target]
            budget_curve = _render_budget_curve(output_dir, model_id, target, curve, plateau_info)
            figures.append(
                {
                    "kind": "budget_curve",
                    "model_id": model_id,
                    "target": target,
                    "path": _relative(budget_curve, output_dir),
                    "sha256": _hash_file(budget_curve),
                }
            )
        selected = selected_budget[model_id]
        for target in PROPERTY_TARGETS:
            heatmap = _render_ablation_heatmap(output_dir, model_id, target, ordered_rows, selected)
            if heatmap is not None:
                figures.append(
                    {
                        "kind": "ablation_heatmap",
                        "model_id": model_id,
                        "target": target,
                        "path": _relative(heatmap, output_dir),
                        "sha256": _hash_file(heatmap),
                    }
                )
            tables.append(
                {
                    "model_id": model_id,
                    "target": target,
                    "curve": curve_metrics[model_id][target],
                    "pooled_curve": pooled_curve[model_id],
                    "plateau": plateau_info,
                    "selected_budget": selected,
                }
            )

    summary = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "protocol": "P5.2_R2",
        "baseline_head": contract["baseline_head"],
        "root_seed": ROOT_SEED,
        "development_families": list(DEVELOPMENT_FAMILIES),
        "repeat_seeds": list(REPEAT_SEEDS),
        "counts": {
            "completed": sum(row["status"] == "completed" for row in ordered_rows),
            "skipped": sum(row["status"] == "skipped" for row in ordered_rows),
            "failed": sum(row["status"] == "failed" for row in ordered_rows),
            "timeout": sum(row["status"] == "timeout" for row in ordered_rows),
            "data_blocked": sum(row["status"] == "data_blocked" for row in ordered_rows),
        },
        "expected_cells": len(ordered_rows),
        "legal_completion_rate": float(sum(row["status"] == "completed" for row in ordered_rows) / len(ordered_rows)),
        "r3_ready": all(
            value["status"] == "plateaued"
            for key, value in plateau.items()
            if key != "tabiclv2_regressor"
        ),
        "plateau": plateau,
        "pooled_curve": pooled_curve,
        "selected_budget": selected_budget,
        "ablation_gates": {
            model_id: {
                "status": (
                    "executed" if selected_budget[model_id] is not None else "not_applicable"
                ),
                "selected_budget": selected_budget[model_id],
                "reason": (
                    "model-specific curve selected a legal budget"
                    if selected_budget[model_id] is not None
                    else "model-specific curve has no completed rankable rows"
                ),
            }
            for model_id in ABLATED_MODELS
        },
        "curve_metrics": curve_metrics,
        "visualization_manifest_sha256": None,
        "oof_manifest_sha256": None,
        "portable": {
            "absolute_paths_persisted": False,
            "source_data_copied": False,
            "complete_predictions_persisted": False,
            "runtime_oof_evidence_validated": True,
            "raw_checkpoints_persisted": False,
        },
        "lanes": {
            "tabular_cpu": {
                "models": ["tabm_regressor", "tabiclv2_regressor"],
                "status": "partially_rankable",
                "rankable_models": sorted(
                    completed_curve_models & {"tabm_regressor", "tabiclv2_regressor"}
                ),
                "candidate_count": len(
                    completed_curve_models & {"tabm_regressor", "tabiclv2_regressor"}
                ),
            },
            "seismic_3d_gpu": {
                "models": ["monai_densenet3d_regressor"],
                "status": "not_rankable",
                "rankable_models": sorted(
                    completed_curve_models & {"monai_densenet3d_regressor"}
                ),
                "candidate_count": len(
                    completed_curve_models & {"monai_densenet3d_regressor"}
                ),
            },
        },
        "source_hashes": {
            "contract_sha256": contract["contract_sha256"],
            "development_archive_sha256": _hash_file(Path(development_batch)),
            "source_lock_sha256": source_lock_sha256(),
            "stage3_results_sha256": _hash_file(STAGE3_RESULTS),
            "stage3_summary_sha256": _hash_file(STAGE3_SUMMARY),
            "stage3_split_sha256": _hash_file(STAGE3_SPLIT),
        },
        "preflight": contract["preflight"],
        "tables": tables,
        "generated_at": _utc_now(),
    }
    visualization_manifest = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "root_seed": ROOT_SEED,
        "figures": figures,
        "figure_count": len(figures),
    }
    visualization_manifest["sha256"] = _hash_payload(visualization_manifest)
    summary["visualization_manifest_sha256"] = visualization_manifest["sha256"]
    summary["oof_manifest_sha256"] = oof_manifest["sha256"]
    summary["artifact_hashes"] = {
        "results": _hash_file(results_path),
        "visualization_manifest": visualization_manifest["sha256"],
        "oof_manifest": oof_manifest["sha256"],
    }
    summary["summary_sha256"] = _hash_payload(summary)
    _atomic_json(output_dir / "p5_r2_summary.json", summary)
    _atomic_json(output_dir / "p5_r2_visualization_manifest.json", visualization_manifest)
    _atomic_json(output_dir / "p5_r2_summary_table.json", {"tables": tables})
    return summary


def audit_existing(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    summary_path = output_dir / "p5_r2_summary.json"
    if not summary_path.is_file():
        return {"status": "missing"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results_path = output_dir / "p5_r2_results.jsonl"
    if summary["artifact_hashes"]["results"] != _hash_file(results_path):
        raise RuntimeError("results hash does not match summary")
    summary_without_self_hash = dict(summary)
    claimed_summary_hash = summary_without_self_hash.pop("summary_sha256")
    if _hash_payload(summary_without_self_hash) != claimed_summary_hash:
        raise RuntimeError("summary self-hash mismatch")
    loaded_manifests: dict[str, dict[str, Any]] = {}
    for filename, summary_key, artifact_key in (
        ("p5_r2_oof_manifest.json", "oof_manifest_sha256", "oof_manifest"),
        (
            "p5_r2_visualization_manifest.json",
            "visualization_manifest_sha256",
            "visualization_manifest",
        ),
    ):
        manifest = json.loads((output_dir / filename).read_text(encoding="utf-8"))
        manifest_without_self_hash = dict(manifest)
        claimed_manifest_hash = manifest_without_self_hash.pop("sha256")
        if _hash_payload(manifest_without_self_hash) != claimed_manifest_hash:
            raise RuntimeError(f"manifest self-hash mismatch: {filename}")
        if claimed_manifest_hash != summary[summary_key]:
            raise RuntimeError(f"manifest hash does not match summary: {filename}")
        if claimed_manifest_hash != summary["artifact_hashes"][artifact_key]:
            raise RuntimeError(f"manifest hash does not match artifact table: {filename}")
        loaded_manifests[filename] = manifest
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != summary["expected_cells"]:
        raise RuntimeError("results cell count does not match summary")
    if len({row["cell_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate R2 cell IDs")
    row_by_id = {row["cell_id"]: row for row in rows}
    oof_entries = loaded_manifests["p5_r2_oof_manifest.json"].get("entries")
    if not isinstance(oof_entries, list) or len(oof_entries) != len(rows):
        raise RuntimeError("OOF manifest entry count does not match results")
    if {entry["cell_id"] for entry in oof_entries} != set(row_by_id):
        raise RuntimeError("OOF manifest cell IDs do not match results")
    for entry in oof_entries:
        row = row_by_id[entry["cell_id"]]
        for key in ("model_id", "phase", "budget", "fold_id", "repeat_id", "status", "oof"):
            if entry[key] != row[key]:
                raise RuntimeError(f"OOF manifest row mismatch: {entry['cell_id']} {key}")
    figures = loaded_manifests["p5_r2_visualization_manifest.json"].get("figures")
    if not isinstance(figures, list):
        raise RuntimeError("visualization manifest figures are malformed")
    output_root = output_dir.resolve()
    for figure in figures:
        figure_path = (output_dir / figure["path"]).resolve()
        if output_root not in figure_path.parents or not figure_path.is_file():
            raise RuntimeError(f"visualization artifact missing or escaped output root: {figure['path']}")
        if _hash_file(figure_path) != figure["sha256"]:
            raise RuntimeError(f"visualization artifact hash mismatch: {figure['path']}")
    completed_paths: dict[str, str] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        relative_path = row["oof"]["relative_path"]
        if Path(relative_path).is_absolute() or not relative_path.startswith("runtime/oof/"):
            raise RuntimeError("completed OOF path is not portable runtime evidence")
        if relative_path in completed_paths:
            raise RuntimeError(
                f"completed cells share OOF path: {completed_paths[relative_path]} and {row['cell_id']}"
            )
        completed_paths[relative_path] = row["cell_id"]
        artifact_path = (output_dir / relative_path).resolve()
        if output_root not in artifact_path.parents or not artifact_path.is_file():
            raise RuntimeError(f"completed OOF artifact missing or escaped output root: {relative_path}")
        if _hash_file(artifact_path) != row["oof"]["sha256"]:
            raise RuntimeError(f"completed OOF hash mismatch: {row['cell_id']}")
    return {
        "status": "verified",
        "rows": len(rows),
        "summary_sha256": _hash_file(summary_path),
        "results_sha256": _hash_file(results_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P5.2 / protocol R2 property runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize a frozen development archive")
    prepare.add_argument("--train-h5", type=Path, default=_default_paths()[0])
    prepare.add_argument("--guard-npz", type=Path, default=_default_paths()[1])
    prepare.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)

    run = subparsers.add_parser("run", help="execute the development learning curves and ablations")
    run.add_argument("--development-batch", type=Path, default=DEFAULT_ARCHIVE)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--tabicl-checkpoint", type=Path, default=None)
    run.add_argument(
        "--repair-tabm-ablation",
        action="store_true",
        help="reuse completed original curve rows and replace only TabM ablation cells",
    )

    audit = subparsers.add_parser("audit", help="validate existing portable outputs")
    audit.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_development_archive(args.train_h5, args.guard_npz, args.output)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    if args.command == "run":
        summary = run_r2(
            development_batch=args.development_batch,
            output_dir=args.output_dir,
            tabicl_checkpoint=args.tabicl_checkpoint,
            repair_tabm_ablation=args.repair_tabm_ablation,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    if args.command == "audit":
        audit = audit_existing(args.output_dir)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
