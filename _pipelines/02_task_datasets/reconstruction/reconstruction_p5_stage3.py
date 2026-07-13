#!/usr/bin/env python3
"""Frozen P5 Stage-3 multiseed spatial CV for Volve reconstruction.

This module confirms only the Stage-2 top three for each independent lane on
all five P4 buffered development folds and the three frozen repeat seeds.  It
has no frozen-test command, loader, path argument, label surface, or metric
surface.  Stage-2 model configuration, input budget, loss, and update counts
are reused without HPO.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# The primary interpreter remains the shared torch-common environment.  A
# caller may append (never prepend) an already-provisioned dependency layer for
# optional pure-Python geostat packages.  No package is installed or copied.
_AUX_SITE_PACKAGES = os.environ.get("VOLVE_P5_AUX_SITE_PACKAGES")
if _AUX_SITE_PACKAGES:
    sys.path.append(_AUX_SITE_PACKAGES)

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from ml_framework.contracts import ModelBatch, TaskSpec  # noqa: E402
from ml_framework.model_discovery import DiscoveredModel, discover_model  # noqa: E402
from ml_framework.seeding import SeedTree, seed_everything  # noqa: E402
from ml_framework.splits import Fold, SplitManifest  # noqa: E402

from _models.reconstruction._p5_adapter import AdapterSkip  # noqa: E402


_STAGE2_MODULE_NAME = "reconstruction_p5_stage2_contract_for_stage3"
if _STAGE2_MODULE_NAME in sys.modules:
    stage2 = sys.modules[_STAGE2_MODULE_NAME]
else:
    _stage2_spec = importlib.util.spec_from_file_location(
        _STAGE2_MODULE_NAME, HERE / "reconstruction_p5_stage2.py"
    )
    if _stage2_spec is None or _stage2_spec.loader is None:
        raise RuntimeError("cannot load reconstruction-prefixed Stage-2 contract")
    stage2 = importlib.util.module_from_spec(_stage2_spec)
    sys.modules[_STAGE2_MODULE_NAME] = stage2
    _stage2_spec.loader.exec_module(stage2)

stage1 = stage2.stage1
p4 = stage2.p4


ROOT_SEED = 2693
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
FOLD_IDS = (0, 1, 2, 3, 4)
MODES = ("strict", "conditional")
MODELS: dict[str, tuple[str, ...]] = {
    "strict": ("pykrige_ok3d", "gpytorch_svgp", "gstools_krige_condsrf"),
    "conditional": ("pykrige_ok3d", "gpytorch_svgp", "scipy_rbf_neighbors"),
}
GPU_MODELS = frozenset({"gpytorch_svgp"})
EXPECTED_CELLS = sum(len(MODELS[mode]) for mode in MODES) * len(FOLD_IDS) * len(REPEAT_SEEDS)
SCHEMA_VERSION = "p5-stage3-reconstruction-cell-v1"
CACHE_SCHEMA_VERSION = "p5-stage3-reconstruction-cache-v1"
SUMMARY_SCHEMA_VERSION = "p5-stage3-reconstruction-summary-v1"
LEADERBOARD_SCHEMA_VERSION = "p5-stage3-reconstruction-leaderboard-v1"
OOF_SCHEMA_VERSION = "p5-stage3-reconstruction-oof-manifest-v1"
VISUALIZATION_SCHEMA_VERSION = "p5-stage3-reconstruction-visualization-manifest-v1"


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return path


def expected_cell_keys() -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        (mode, model_id, fold_id, repeat_id)
        for mode in MODES
        for model_id in MODELS[mode]
        for fold_id in FOLD_IDS
        for repeat_id in range(len(REPEAT_SEEDS))
    )


def _cell_id(mode: str, model_id: str, fold_id: int, repeat_id: int) -> str:
    return f"{mode}/{model_id}/fold_{fold_id:02d}/repeat_{repeat_id}"


def _cell_dir(
    cell_root: Path, mode: str, model_id: str, fold_id: int, repeat_id: int
) -> Path:
    return cell_root / mode / model_id / f"fold_{fold_id:02d}" / f"repeat_{repeat_id}"


def _frozen_split_hashes() -> dict[str, str]:
    payload = json.loads((HERE / "p5_stage2_summary.json").read_text(encoding="utf-8"))
    hashes = {mode: str(payload["split_hashes"][mode]) for mode in MODES}
    if any(len(value) != 64 for value in hashes.values()):
        raise ValueError("Stage-2 frozen split hashes are invalid")
    return hashes


def _fold_train_well_coordinates(records: Sequence[Any]) -> np.ndarray:
    """Return unique fold-train well coordinates without reading well values."""
    rows: list[np.ndarray] = []
    for record in records:
        patch = np.asarray(record.seismic_patch)
        observed = (patch[7] > 0.5) & (patch[8] > 0.5)
        if np.any(observed):
            rows.append(patch[3:6].reshape(3, -1).T[observed.reshape(-1)])
    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    coordinates = np.asarray(np.concatenate(rows), dtype=np.float64)
    return np.unique(coordinates, axis=0)


def _nearest_well_distance(xyz: np.ndarray, well_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    well_xyz = np.asarray(well_xyz, dtype=np.float64)
    if well_xyz.size == 0:
        # JSON sidecars reject NaN.  -1 is an explicit unavailable sentinel;
        # the fold audit separately records well_coordinate_count=0 and the
        # visualizer excludes negative distances.
        return np.full(xyz.shape[0], -1.0, dtype=np.float64)
    from scipy.spatial import cKDTree

    anisotropy = np.asarray([1.0, 1.0, 3.0], dtype=np.float64)
    distance, _ = cKDTree(well_xyz * anisotropy).query(xyz * anisotropy, k=1)
    return np.asarray(distance, dtype=np.float64)


def _point_batch(
    task_spec: TaskSpec,
    *,
    mode: str,
    fold_id: int,
    split: str,
    features: np.ndarray,
    target: np.ndarray,
    xyz: np.ndarray,
    metric_indices_kji: np.ndarray | None,
    volume_shape_kji: tuple[int, int, int],
    feature_names: Sequence[str],
    constraint_count: int,
    distance_to_fold_train_well: np.ndarray | None = None,
) -> ModelBatch:
    target_name = task_spec.targets[0]
    coordinates: dict[str, np.ndarray] = {"xyz": np.asarray(xyz, dtype=np.float64)}
    if metric_indices_kji is not None:
        coordinates["metric_indices_kji"] = np.asarray(metric_indices_kji, dtype=np.int64)
        coordinates["volume_shape_kji"] = np.asarray(volume_shape_kji, dtype=np.int64)
    if distance_to_fold_train_well is not None:
        coordinates["distance_to_fold_train_well"] = np.asarray(
            distance_to_fold_train_well, dtype=np.float64
        )
    values = np.asarray(target, dtype=np.float64)
    return ModelBatch(
        inputs={"features": np.asarray(features, dtype=np.float64)},
        targets={target_name: values},
        input_masks={},
        target_masks={target_name: np.ones(values.shape, dtype=bool)},
        sample_ids=[f"{mode}:stage3:fold{fold_id}:{split}:points"],
        groups={"fold": [str(fold_id)], "evaluation_mode": [mode]},
        coordinates=coordinates,
        metadata={
            "evaluation_mode": mode,
            "fold_id": fold_id,
            "split": f"development_{split}",
            "feature_names": list(feature_names),
            "constraint_count_supplied": int(constraint_count),
            "frozen_test_i_blocks_loaded": [],
        },
    )


def prepare_fold_cache(
    mode: str,
    manifest: Any,
    fold: Any,
    development: Sequence[Any],
    cache_dir: Path,
) -> dict[str, Any]:
    """Fit fold-local preprocessing and materialize the Stage-2 point budget."""
    if fold.fold_id not in FOLD_IDS:
        raise ValueError("Stage-3 fold_id is outside the frozen five-fold manifest")
    spec = p4.task_spec(mode)
    prepared = p4.prepare_fold(mode, fold, development)
    if mode == "strict" and prepared.constraint_audit["constraints_supplied_to_model"] != 0:
        raise RuntimeError("strict Stage-3 cache supplied target-derived constraints")
    by_id = {item.sample_id: item for item in development}
    train_records = tuple(by_id[item] for item in fold.purge["effective_train_sample_ids"])
    validation_records = tuple(by_id[item] for item in fold.validation_sample_ids)

    train_indices = stage2._sample_indices(prepared.train_target.size, stage2.POINT_TRAIN_VOXELS)
    constraint_count = int(prepared.constraint_audit["constraints_supplied_to_model"])
    point_train = _point_batch(
        spec,
        mode=mode,
        fold_id=fold.fold_id,
        split="train",
        features=prepared.train_features[train_indices],
        target=prepared.train_target[train_indices],
        xyz=prepared.train_features[train_indices, -3:],
        metric_indices_kji=None,
        volume_shape_kji=prepared.validation_cells.volume_shape_kji,
        feature_names=prepared.feature_names,
        constraint_count=constraint_count,
    )

    validation_record = max(
        validation_records,
        key=lambda item: int(
            np.sum((item.seismic_patch[8] > 0.5) & ~(item.seismic_patch[7] > 0.5))
        ),
    )
    same_patch = prepared.validation_cells.sample_ids == validation_record.sample_id
    eligible = np.flatnonzero(same_patch & prepared.validation_metric_mask)
    selected = eligible[
        stage2._sample_indices(eligible.size, stage2.VALIDATION_VOXELS)
    ]
    metric_indices = prepared.validation_cells.indices_kji[selected]
    validation_xyz = prepared.validation_cells.coordinates[selected]
    well_xyz = _fold_train_well_coordinates(train_records)
    point_validation = _point_batch(
        spec,
        mode=mode,
        fold_id=fold.fold_id,
        split="validation",
        features=prepared.validation_features[selected],
        target=prepared.validation_target[selected],
        xyz=validation_xyz,
        metric_indices_kji=metric_indices,
        volume_shape_kji=prepared.validation_cells.volume_shape_kji,
        feature_names=prepared.feature_names,
        constraint_count=constraint_count,
        distance_to_fold_train_well=_nearest_well_distance(validation_xyz, well_xyz),
    )

    fold_dir = cache_dir / mode / f"fold_{fold.fold_id:02d}"
    batches = {"point_train": point_train, "point_validation": point_validation}
    batch_records: dict[str, Any] = {}
    for name, batch in batches.items():
        path = fold_dir / f"{name}.npz"
        stage1._write_cached_batch(path, batch, spec)  # noqa: SLF001
        batch_records[name] = {"sha256": hash_file(path), "bytes": path.stat().st_size}
    atomic_write_json(fold_dir / "preprocess_stats.json", prepared.preprocess_report)
    atomic_write_json(fold_dir / "constraint_audit.json", prepared.constraint_audit)
    fold_payload = {
        "fold_id": fold.fold_id,
        "fold_hash": hash_payload(
            {
                "fold_id": fold.fold_id,
                "train_groups": list(fold.train_groups),
                "validation_groups": list(fold.validation_groups),
                "train_sample_ids": list(fold.train_sample_ids),
                "validation_sample_ids": list(fold.validation_sample_ids),
                "purge": dict(fold.purge),
                "support": dict(fold.support),
            }
        ),
        "effective_train_sample_ids": list(fold.purge["effective_train_sample_ids"]),
        "validation_sample_ids": list(fold.validation_sample_ids),
        "purged_train_sample_ids": list(fold.purge["purged_train_sample_ids"]),
        "selected_validation_patch": validation_record.sample_id,
        "preprocess": {
            "fit_scope": "fold.purge.effective_train_sample_ids only",
            "preprocess_sha256": hash_file(fold_dir / "preprocess_stats.json"),
            "target_transform": "identity",
            "class_weights": "not_applicable_regression",
            "calibration": "not_applied",
        },
        "constraint_audit": prepared.constraint_audit,
        "posthoc_distance_diagnostic": {
            "uses_fold_train_well_coordinates_only": True,
            "uses_well_values": False,
            "well_coordinate_count": int(well_xyz.shape[0]),
        },
        "input_budget": {
            "point_train_voxels": int(len(train_indices)),
            "shared_validation_voxels": int(len(selected)),
        },
        "batches": batch_records,
        "frozen_test_i_blocks_loaded": [],
    }
    atomic_write_json(fold_dir / "fold_cache_manifest.json", fold_payload)
    return fold_payload


def prepare_mode_cache(
    mode: str,
    manifest: SplitManifest,
    records: Sequence[Any],
    cache_dir: Path,
) -> dict[str, Any]:
    spec = p4.task_spec(mode)
    active = p4.protocol(mode)
    if manifest.effective_n_splits != len(FOLD_IDS):
        raise RuntimeError(
            f"{mode} requires five valid P4 folds, got {manifest.effective_n_splits}"
        )
    if tuple(fold.fold_id for fold in manifest.folds) != FOLD_IDS:
        raise RuntimeError(f"{mode} P4 manifest fold IDs differ from 0..4")
    if manifest.metadata.get("evaluation_mode") != mode:
        raise RuntimeError(f"{mode} Stage-2 manifest has the wrong evaluation mode")
    development = [item for item in records if item.i_block in active.development_i_blocks]
    loaded_blocks = sorted({item.i_block for item in development})
    if loaded_blocks != sorted(active.development_i_blocks):
        raise RuntimeError(f"{mode} cache lacks the complete development I-block set")
    if set(loaded_blocks) & (set(active.test_i_blocks) | set(active.guard_i_blocks)):
        raise RuntimeError(f"{mode} cache crossed the test/guard firewall")
    p4.validate_buffered_manifest(manifest, [item.location for item in development])
    mode_dir = cache_dir / mode
    atomic_write_json(mode_dir / "task_spec.json", spec.to_dict())
    atomic_write_json(mode_dir / "split_manifest.json", manifest.to_dict())
    folds = {
        str(fold.fold_id): prepare_fold_cache(mode, manifest, fold, development, cache_dir)
        for fold in manifest.folds
    }
    payload = {
        "task_id": spec.task_id,
        "evaluation_mode": mode,
        "split_hash": manifest.stable_hash(),
        "effective_n_splits": manifest.effective_n_splits,
        "development_i_blocks": list(active.development_i_blocks),
        "guard_i_blocks": list(active.guard_i_blocks),
        "frozen_test_i_blocks": list(active.test_i_blocks),
        "frozen_test_i_blocks_loaded": [],
        "input_whitelist": list(spec.input_whitelist),
        "folds": folds,
    }
    if payload["split_hash"] != _frozen_split_hashes()[mode]:
        raise RuntimeError(f"{mode} P4 split hash differs from the Stage-2 frozen manifest")
    return payload


def _load_stage2_manifest(stage2_cache: Path, mode: str) -> tuple[SplitManifest, str]:
    path = stage2_cache / mode / "split_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_folds = payload.pop("folds")
    manifest = SplitManifest(
        folds=tuple(
            Fold(
                fold_id=int(item["fold_id"]),
                train_groups=tuple(item["train_groups"]),
                validation_groups=tuple(item["validation_groups"]),
                train_sample_ids=tuple(item["train_sample_ids"]),
                validation_sample_ids=tuple(item["validation_sample_ids"]),
                purge=dict(item["purge"]),
                support=dict(item["support"]),
            )
            for item in raw_folds
        ),
        manifest_version=str(payload["manifest_version"]),
        group_key=str(payload["group_key"]),
        requested_n_splits=int(payload["requested_n_splits"]),
        effective_n_splits=int(payload["effective_n_splits"]),
        downgrade_reason=payload.get("downgrade_reason"),
        test_groups=tuple(payload["test_groups"]),
        test_sample_ids=tuple(payload["test_sample_ids"]),
        development_groups=tuple(payload["development_groups"]),
        development_sample_ids=tuple(payload["development_sample_ids"]),
        metadata=dict(payload["metadata"]),
    )
    if manifest.stable_hash() != _frozen_split_hashes()[mode]:
        raise RuntimeError(f"{mode} Stage-2 manifest hash differs from frozen evidence")
    return manifest, hash_file(path)


def prepare_cache(data_dir: Path, cache_dir: Path, stage2_cache: Path) -> dict[str, Any]:
    """Read development blocks only and materialize all five fold-local caches."""
    stage2_root = json.loads(
        (stage2_cache / "cache_manifest.json").read_text(encoding="utf-8")
    )
    if stage2_root.get("schema_version") != stage2.CACHE_SCHEMA_VERSION:
        raise ValueError("input Stage-2 cache has the wrong schema")
    if stage2_root.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("input Stage-2 cache reports frozen-test array access")
    manifests = {
        mode: _load_stage2_manifest(stage2_cache, mode) for mode in MODES
    }
    records = {
        mode: p4.load_patch_records(p4.protocol(mode).development_i_blocks, data_dir)
        for mode in MODES
    }
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "source_container_names": ["train.h5", "test.h5"],
        "source_note": (
            "legacy physical containers; only mode-specific development I-block arrays were loaded"
        ),
        "stage2_split_manifest_sha256": {
            mode: manifests[mode][1] for mode in MODES
        },
        "frozen_test_i_blocks_loaded": [],
        "modes": {
            mode: prepare_mode_cache(mode, manifests[mode][0], records[mode], cache_dir)
            for mode in MODES
        },
    }
    atomic_write_json(cache_dir / "cache_manifest.json", payload)
    return payload


def load_fold_cache(
    cache_dir: Path, mode: str, fold_id: int
) -> tuple[TaskSpec, ModelBatch, ModelBatch, dict[str, Any], dict[str, Any], str]:
    manifest_path = cache_dir / "cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("invalid reconstruction Stage-3 cache schema")
    if payload.get("repeat_seeds") != list(REPEAT_SEEDS):
        raise ValueError("Stage-3 cache repeat seeds differ from the frozen protocol")
    if payload.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-3 cache reports frozen-test access")
    if mode not in MODES or fold_id not in FOLD_IDS:
        raise ValueError("invalid Stage-3 mode/fold")
    mode_audit = payload["modes"][mode]
    if mode_audit.get("split_hash") != _frozen_split_hashes()[mode]:
        raise RuntimeError("Stage-3 cache split hash differs from the frozen P4 manifest")
    if mode_audit.get("effective_n_splits") != 5:
        raise RuntimeError("Stage-3 cache is not the frozen five-fold P4 split")
    fold_audit = mode_audit["folds"][str(fold_id)]
    if fold_audit.get("fold_id") != fold_id:
        raise RuntimeError("Stage-3 fold cache identity mismatch")
    if fold_audit.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-3 fold cache reports frozen-test access")
    spec = TaskSpec.from_dict(
        json.loads((cache_dir / mode / "task_spec.json").read_text(encoding="utf-8"))
    )
    if spec.task_id != mode_audit["task_id"] or spec.metadata.get("evaluation_mode") != mode:
        raise RuntimeError("Stage-3 TaskSpec/cache lane mismatch")
    expected_features = 6 if mode == "strict" else 7
    if len(spec.input_whitelist) != expected_features:
        raise RuntimeError("Stage-3 TaskSpec feature whitelist differs from the frozen lane")
    if mode == "strict" and fold_audit["constraint_audit"]["constraints_supplied_to_model"] != 0:
        raise RuntimeError("strict Stage-3 fold supplied conditional constraints")
    fold_dir = cache_dir / mode / f"fold_{fold_id:02d}"
    for name, expected in fold_audit["batches"].items():
        path = fold_dir / f"{name}.npz"
        if hash_file(path) != expected["sha256"]:
            raise RuntimeError(f"Stage-3 cached batch hash mismatch: {mode}/fold{fold_id}/{name}")
    train = stage1._read_cached_batch(fold_dir / "point_train.npz")  # noqa: SLF001
    validation = stage1._read_cached_batch(fold_dir / "point_validation.npz")  # noqa: SLF001
    for batch in (train, validation):
        if batch.metadata.get("evaluation_mode") != mode:
            raise RuntimeError("Stage-3 batch crossed lane namespaces")
        if batch.metadata.get("fold_id") != fold_id:
            raise RuntimeError("Stage-3 batch crossed fold namespaces")
        if batch.metadata.get("frozen_test_i_blocks_loaded") != []:
            raise RuntimeError("Stage-3 batch reports frozen-test access")
    return spec, train, validation, mode_audit, fold_audit, hash_file(manifest_path)


def _repeat_seed(repeat_id: int) -> int:
    if repeat_id not in range(len(REPEAT_SEEDS)):
        raise ValueError("Stage-3 repeat_id must be 0, 1, or 2")
    expected = SeedTree(ROOT_SEED).seed("model", "p5-stage3", repeat_id)
    if expected != REPEAT_SEEDS[repeat_id]:
        raise RuntimeError("shared seed derivation differs from the frozen Stage-3 seeds")
    return REPEAT_SEEDS[repeat_id]


def _gpu_lock_audit(device: str) -> Any:
    if not device.startswith("cuda"):
        return stage2.gpu_flock(device)
    if os.environ.get("VOLVE_P5_GPU_LOCK_HELD") != "1":
        raise AdapterSkip(
            "gpu_lock_not_held",
            "Stage-3 GPU cells must be launched under the unified external flock",
        )
    raw_wait = os.environ.get("VOLVE_P5_GPU_LOCK_WAIT_SECONDS")
    if raw_wait is None:
        raise AdapterSkip(
            "gpu_lock_wait_missing",
            "Stage-3 GPU lock evidence must record finite wait seconds",
        )
    try:
        wait_seconds = float(raw_wait)
    except ValueError as exc:
        raise AdapterSkip(
            "gpu_lock_wait_invalid", "GPU lock wait evidence is not numeric"
        ) from exc
    if not math.isfinite(wait_seconds) or wait_seconds < 0 or wait_seconds > 900:
        raise AdapterSkip(
            "gpu_lock_wait_invalid", "GPU lock wait evidence is outside [0, 900] seconds"
        )

    class _ExternalAudit:
        def __enter__(self) -> dict[str, Any]:
            self._context = stage2.gpu_flock(device)
            audit = dict(self._context.__enter__())
            audit["wait_seconds"] = wait_seconds
            audit["lock_environment_variable"] = "VOLVE_P5_GPU_LOCK"
            return audit

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
            return self._context.__exit__(exc_type, exc, traceback)

    return _ExternalAudit()


def _extra_regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else 0.0
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if np.std(target) > 0 and np.std(prediction) > 0
        else 0.0
    )
    return {"bias": float(np.mean(error)), "r2": r2, "pearson_r": correlation}


def _save_oof_prediction(
    path: Path,
    validation: ModelBatch,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    mode: str,
    model_id: str,
    fold_id: int,
    repeat_id: int,
    repeat_seed: int,
) -> dict[str, Any]:
    indices = np.asarray(validation.coordinates["metric_indices_kji"], dtype=np.int64)
    xyz = np.asarray(validation.coordinates["xyz"], dtype=np.float64)
    distance = np.asarray(
        validation.coordinates["distance_to_fold_train_well"], dtype=np.float64
    )
    volume_shape = np.asarray(validation.coordinates["volume_shape_kji"], dtype=np.int64)
    _atomic_write_npz(
        path,
        target=np.asarray(target, dtype=np.float32),
        prediction=np.asarray(prediction, dtype=np.float32),
        indices_kji=indices,
        xyz=xyz.astype(np.float32),
        distance_to_fold_train_well=distance.astype(np.float32),
        volume_shape_kji=volume_shape,
        lane=np.asarray(mode),
        model_id=np.asarray(model_id),
        fold_id=np.asarray(fold_id, dtype=np.int64),
        repeat_id=np.asarray(repeat_id, dtype=np.int64),
        repeat_seed=np.asarray(repeat_seed, dtype=np.int64),
    )
    return {
        "archive_name": path.name,
        "sha256": hash_file(path),
        "bytes": path.stat().st_size,
        "validation_voxels": int(target.size),
        "fields": [
            "target",
            "prediction",
            "indices_kji",
            "xyz",
            "distance_to_fold_train_well",
            "volume_shape_kji",
            "lane",
            "model_id",
            "fold_id",
            "repeat_id",
            "repeat_seed",
        ],
        "scope": "sampled buffered-development OOF validation only",
    }


def run_cell(
    *,
    mode: str,
    model_id: str,
    fold_id: int,
    repeat_id: int,
    cache_dir: Path,
    cell_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run one frozen Stage-3 cell; no raw-data or frozen-test argument exists."""
    if mode not in MODES or model_id not in MODELS.get(mode, ()):
        raise ValueError("model is not in the frozen Stage-3 top-three for this lane")
    if fold_id not in FOLD_IDS:
        raise ValueError("fold_id is outside the frozen P4 manifest")
    repeat_seed = _repeat_seed(repeat_id)
    if model_id in GPU_MODELS and device != "cuda:0":
        raise ValueError("Stage-3 neural/operator cells require device cuda:0")
    if model_id not in GPU_MODELS and device != "cpu":
        raise ValueError("Stage-3 traditional geostat cells require device cpu")

    spec, train, validation, mode_audit, fold_audit, cache_hash = load_fold_cache(
        cache_dir, mode, fold_id
    )
    status_dir = _cell_dir(cell_root, mode, model_id, fold_id, repeat_id)
    status_path = status_dir / "status.json"
    technical_retry: dict[str, Any] | None = None
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        if prior.get("status") == "passed":
            raise FileExistsError(f"refusing to rerun successful Stage-3 cell: {status_path}")
        if prior.get("technical_retry") is not None:
            raise RuntimeError(f"Stage-3 cell already used its one technical retry: {status_path}")
        technical_retry = {
            "attempt": 1,
            "prior_result_hash": prior.get("result_hash"),
            "prior_status": prior.get("status"),
            "prior_reason": prior.get("reason"),
            "same_model_config_budget_split_and_seed": True,
        }
    source_lock = stage1.load_source_lock()
    seed_tree = {
        "root": ROOT_SEED,
        "repeat_id": repeat_id,
        "model": repeat_seed,
        "loader": SeedTree(ROOT_SEED).seed(
            "loader", "p5-stage3", mode, model_id, fold_id, repeat_id
        ),
        "sampler": SeedTree(ROOT_SEED).seed(
            "sampler", "p5-stage3", mode, model_id, fold_id, repeat_id
        ),
    }
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "cell_id": _cell_id(mode, model_id, fold_id, repeat_id),
        "task_id": spec.task_id,
        "lane": mode,
        "evaluation_mode": mode,
        "model_id": model_id,
        "fold_id": fold_id,
        "repeat_id": repeat_id,
        "repeat_seed": repeat_seed,
        "seed": seed_tree,
        "source": stage2._portable_source(model_id, source_lock),  # noqa: SLF001
        "split_hash": mode_audit["split_hash"],
        "fold_hash": fold_audit["fold_hash"],
        "cache_contract_hash": cache_hash,
        "input_budget": fold_audit["input_budget"],
        "fold_fit_audit": {
            **fold_audit["preprocess"],
            "effective_train_sample_ids": fold_audit["effective_train_sample_ids"],
            "validation_sample_ids": fold_audit["validation_sample_ids"],
            "purged_train_sample_ids": fold_audit["purged_train_sample_ids"],
        },
        "mode_isolation": {
            "input_whitelist": mode_audit["input_whitelist"],
            "constraint_audit": fold_audit["constraint_audit"],
            "strict_constraints_supplied": (
                fold_audit["constraint_audit"]["constraints_supplied_to_model"]
                if mode == "strict"
                else None
            ),
        },
        "test_firewall": {
            "development_only": True,
            "frozen_test_i_blocks_loaded": [],
            "test_loader_argument_exists": False,
            "test_path_argument_exists": False,
            "test_metrics_computed": False,
            "historical_test_metrics_read": False,
        },
        "stage2_reuse": {
            "model_config_factory": "p5_stage1.model_config",
            "budget_factory": "reconstruction_p5_stage2.budget_for",
            "metric_factory": "reconstruction_p5_stage2.pilot_metrics",
            "tiny_gate_updates": stage2.TINY_GATE_UPDATES,
            "hpo_performed": False,
            "preprocessing_changed": False,
            "loss_changed": False,
            "updates_changed": False,
        },
        "technical_retry": technical_retry,
    }

    discovered: DiscoveredModel | None = None
    lock_audit: dict[str, Any] = {
        "required": model_id in GPU_MODELS,
        "acquired": False,
        "wait_seconds": 0.0 if model_id not in GPU_MODELS else None,
    }
    try:
        with _gpu_lock_audit(device) as acquired_lock:
            lock_audit = dict(acquired_lock)
            if Path(sys.executable).parent.parent.name != "torch-common":
                raise AdapterSkip(
                    "wrong_primary_environment",
                    "Stage-3 reconstruction cells must use the shared torch-common interpreter",
                    executable_name=Path(sys.executable).name,
                )
            if model_id in GPU_MODELS:
                import torch

                if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                    raise AdapterSkip(
                        "gpu_unavailable",
                        "cuda:0 is unavailable in the frozen torch-common environment",
                    )
            discovered = discover_model("reconstruction", model_id)
            if discovered.capabilities.get("batch_representation") != "point":
                raise RuntimeError("Stage-3 frozen reconstruction candidates must use point batches")
            if bool(discovered.capabilities.get("trainable")) != (model_id in GPU_MODELS):
                raise RuntimeError("Stage-3 frozen resource lane differs from model capabilities")
            budget = stage2.budget_for(discovered.capabilities)
            config = stage1.model_config(
                model_id, spec, train, device=device, seed=repeat_seed
            )
            if int(config["n_training_samples"]) != stage2.POINT_TRAIN_VOXELS:
                raise RuntimeError("Stage-3 model input budget differs from Stage-2")
            environment = stage2._environment(discovered)  # noqa: SLF001
            environment["primary_environment"] = "torch-common"
            environment["aux_dependency_group"] = os.environ.get(
                "VOLVE_P5_AUX_DEPENDENCY_GROUP"
            )
            base.update(
                {
                    "representation": "point",
                    "budget": budget,
                    "model_config": {
                        key: list(value) if isinstance(value, tuple) else value
                        for key, value in config.items()
                    },
                    "environment": environment,
                }
            )
            seed_report = seed_everything(
                repeat_seed, strict=True, include_torch=True
            ).to_dict()
            stage2._reset_cuda_peak(device)  # noqa: SLF001
            started = time.monotonic()
            with stage2._wall_timeout(int(budget["max_wall_seconds"])):  # noqa: SLF001
                gate_checkpoint = status_dir / "tiny_gate.ckpt"
                tiny_gate = stage2._tiny_gate(  # noqa: SLF001
                    discovered, spec, train, validation, config, gate_checkpoint
                )
                seed_everything(repeat_seed, strict=True, include_torch=True)
                model = discovered.build(spec, **config)
                losses: list[float] = []
                for _ in range(int(budget["max_updates"])):
                    step = dict(model.train_batch(train))
                    loss = float(step["loss"])
                    if not math.isfinite(loss):
                        raise FloatingPointError("Stage-3 train loss is non-finite")
                    losses.append(loss)
                target, prediction, _ = stage2._target_prediction(  # noqa: SLF001
                    model, validation, spec
                )
                checkpoint = status_dir / "model_last.ckpt"
                model.save_checkpoint(checkpoint)
                restored = discovered.build(spec, **config)
                restored.load_checkpoint(checkpoint)
                _, restored_prediction, _ = stage2._target_prediction(  # noqa: SLF001
                    restored, validation, spec
                )
                checkpoint_error = float(np.max(np.abs(prediction - restored_prediction)))
                tolerance = 1e-5 if device.startswith("cuda") else 1e-8
                if checkpoint_error > tolerance:
                    raise AssertionError("Stage-3 checkpoint round-trip exceeded tolerance")
                indices = np.asarray(
                    validation.coordinates["metric_indices_kji"], dtype=np.int64
                )
                volume_shape = np.asarray(
                    validation.coordinates["volume_shape_kji"], dtype=np.int64
                )
                metrics = stage2.pilot_metrics(target, prediction, indices, volume_shape)
                metrics.update(_extra_regression_metrics(target, prediction))
                oof = _save_oof_prediction(
                    status_dir / "oof_prediction.npz",
                    validation,
                    target,
                    prediction,
                    mode=mode,
                    model_id=model_id,
                    fold_id=fold_id,
                    repeat_id=repeat_id,
                    repeat_seed=repeat_seed,
                )
            wall_seconds = time.monotonic() - started
            stage2.validate_budget(budget, len(losses), wall_seconds)
            base.update(
                {
                    "status": "passed",
                    "evidence_status": "multiseed_spatial_cv_completed",
                    "reason": None,
                    "updates": len(losses),
                    "wall_seconds": wall_seconds,
                    "metrics": metrics,
                    "train_loss": {
                        "first": losses[0],
                        "last": losses[-1],
                        "minimum": min(losses),
                    },
                    "tiny_gate": tiny_gate,
                    "checkpoint": {
                        "sha256": hash_file(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                        "roundtrip_max_abs_error": checkpoint_error,
                        "storage": "ignored_runtime_directory",
                    },
                    "oof_prediction": oof,
                    "resources": {
                        "peak_rss_kib": int(
                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        ),
                        "peak_cuda_bytes": stage2._cuda_peak(device),  # noqa: SLF001
                        "gpu_lock": lock_audit,
                    },
                    "seed_report": seed_report,
                }
            )
    except stage2.PilotTimeout as exc:
        base.update(
            {
                "status": "timeout",
                "evidence_status": "cell_attempted",
                "reason": {"code": "budget_timeout", "message": str(exc)},
                "updates": 0,
                "wall_seconds": float(base.get("budget", {}).get("max_wall_seconds", 0)),
                "metrics": None,
                "resources": {
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "peak_cuda_bytes": stage2._cuda_peak(device),  # noqa: SLF001
                    "gpu_lock": lock_audit,
                },
            }
        )
    except AdapterSkip as exc:
        reason = exc.to_dict()
        base.update(
            {
                "status": "failed" if reason["code"].startswith("gpu_") else "skipped",
                "evidence_status": "cell_attempted",
                "reason": reason,
                "updates": 0,
                "wall_seconds": 0.0,
                "metrics": None,
                "resources": {
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "peak_cuda_bytes": 0,
                    "gpu_lock": lock_audit,
                },
                "environment": stage2._environment(discovered),  # noqa: SLF001
            }
        )
    except Exception as exc:
        base.update(
            {
                "status": "failed",
                "evidence_status": "cell_attempted",
                "reason": {
                    "code": "pilot_exception",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                "updates": 0,
                "wall_seconds": 0.0,
                "metrics": None,
                "resources": {
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "peak_cuda_bytes": 0,
                    "gpu_lock": lock_audit,
                },
                "environment": stage2._environment(discovered),  # noqa: SLF001
            }
        )
    base["result_hash"] = hash_payload(base)
    atomic_write_json(status_path, base)
    return base


def _stage2_reference_record(mode: str, model_id: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (HERE / "p5_stage2_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [
        row for row in rows if row["lane"] == mode and row["model_id"] == model_id
    ]
    if len(matches) != 1 or matches[0]["status"] != "passed":
        raise RuntimeError(f"Stage-2 frozen reference is unavailable for {mode}/{model_id}")
    return matches[0]


def validate_cell_record(
    record: Mapping[str, Any], mode: str, model_id: str, fold_id: int, repeat_id: int
) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Stage-3 cell schema mismatch")
    expected_hash = hash_payload(
        {key: value for key, value in record.items() if key != "result_hash"}
    )
    if record.get("result_hash") != expected_hash:
        raise ValueError("Stage-3 cell result_hash mismatch")
    expected_identity = _cell_id(mode, model_id, fold_id, repeat_id)
    if record.get("cell_id") != expected_identity:
        raise ValueError("Stage-3 cell identity mismatch")
    if record.get("lane") != mode or record.get("evaluation_mode") != mode:
        raise ValueError("Stage-3 cell crossed task/lane namespaces")
    if record.get("task_id") != p4.protocol(mode).task_id:
        raise ValueError("Stage-3 cell crossed TaskSpec identities")
    if record.get("model_id") != model_id or model_id not in MODELS[mode]:
        raise ValueError("Stage-3 cell model is outside the frozen lane top-three")
    if record.get("fold_id") != fold_id or fold_id not in FOLD_IDS:
        raise ValueError("Stage-3 cell fold differs from the frozen manifest")
    if record.get("repeat_id") != repeat_id:
        raise ValueError("Stage-3 cell repeat identity mismatch")
    if record.get("repeat_seed") != _repeat_seed(repeat_id):
        raise ValueError("Stage-3 cell repeat seed differs from the frozen protocol")
    if record.get("seed", {}).get("model") != _repeat_seed(repeat_id):
        raise ValueError("Stage-3 model seed record is inconsistent")
    if record.get("split_hash") != _frozen_split_hashes()[mode]:
        raise ValueError("Stage-3 cell split hash differs from frozen P4/Stage-2")
    retry = record.get("technical_retry")
    if retry is not None:
        if retry.get("attempt") != 1 or not retry.get("prior_result_hash"):
            raise ValueError("Stage-3 technical retry audit is invalid")
        if retry.get("same_model_config_budget_split_and_seed") is not True:
            raise ValueError("Stage-3 technical retry changed the frozen cell contract")

    firewall = record.get("test_firewall", {})
    if firewall.get("frozen_test_i_blocks_loaded") != []:
        raise ValueError("Stage-3 cell reports frozen-test access")
    forbidden_firewall_flags = (
        "test_loader_argument_exists",
        "test_path_argument_exists",
        "test_metrics_computed",
        "historical_test_metrics_read",
    )
    if any(firewall.get(name) is not False for name in forbidden_firewall_flags):
        raise ValueError("Stage-3 cell exposes or consumed a frozen-test surface")
    if firewall.get("development_only") is not True:
        raise ValueError("Stage-3 cell is not explicitly development-only")

    isolation = record.get("mode_isolation", {})
    expected_features = 6 if mode == "strict" else 7
    if len(isolation.get("input_whitelist", [])) != expected_features:
        raise ValueError("Stage-3 lane input whitelist has the wrong size")
    if mode == "strict" and isolation.get("strict_constraints_supplied") != 0:
        raise ValueError("strict Stage-3 cell received conditional constraints")
    fit = record.get("fold_fit_audit", {})
    train_ids = set(fit.get("effective_train_sample_ids", []))
    validation_ids = set(fit.get("validation_sample_ids", []))
    purged_ids = set(fit.get("purged_train_sample_ids", []))
    if not train_ids or not validation_ids:
        raise ValueError("Stage-3 fold has empty train or validation IDs")
    if train_ids & validation_ids or purged_ids & validation_ids or train_ids & purged_ids:
        raise ValueError("Stage-3 fold train/purge/validation sets overlap")
    if fit.get("fit_scope") != "fold.purge.effective_train_sample_ids only":
        raise ValueError("Stage-3 preprocessing is not fold-train-only")
    if fit.get("target_transform") != "identity":
        raise ValueError("Stage-3 target transform differs from Stage-2 identity")
    if fit.get("class_weights") != "not_applicable_regression":
        raise ValueError("Stage-3 regression class-weight audit is invalid")
    if fit.get("calibration") != "not_applied":
        raise ValueError("Stage-3 calibration differs from the frozen configuration")

    reuse = record.get("stage2_reuse", {})
    if reuse.get("hpo_performed") is not False:
        raise ValueError("Stage-3 cell reports HPO")
    if any(
        reuse.get(name) is not False
        for name in ("preprocessing_changed", "loss_changed", "updates_changed")
    ):
        raise ValueError("Stage-3 cell changed frozen Stage-2 plumbing")

    status = record.get("status")
    if status not in {"passed", "skipped", "failed", "timeout"}:
        raise ValueError("Stage-3 cell status is invalid")
    if status != "passed":
        if record.get("metrics") is not None or not (record.get("reason") or {}).get("code"):
            raise ValueError("non-passed Stage-3 cell lacks a structured reason")
        return

    reference = _stage2_reference_record(mode, model_id)
    if record.get("budget") != reference.get("budget"):
        raise ValueError("Stage-3 cell budget differs from Stage-2")
    config = dict(record.get("model_config", {}))
    reference_config = dict(reference.get("model_config", {}))
    if config.pop("seed", None) != _repeat_seed(repeat_id):
        raise ValueError("Stage-3 model config does not contain the frozen repeat seed")
    reference_config.pop("seed", None)
    if config != reference_config:
        raise ValueError("Stage-3 model configuration differs from Stage-2")
    if int(record.get("updates", -1)) != int(record["budget"]["max_updates"]):
        raise ValueError("Stage-3 successful cell did not use the frozen update count")
    stage2.validate_budget(
        record["budget"], int(record["updates"]), float(record["wall_seconds"])
    )
    metrics = record.get("metrics", {})
    for name in ("rmse", "mae", "spectral_log_rmse", "bias", "r2", "pearson_r"):
        if not math.isfinite(float(metrics[name])):
            raise ValueError(f"Stage-3 cell metric {name} is non-finite")
    if int(metrics.get("valid_voxels", 0)) != int(
        record["input_budget"]["shared_validation_voxels"]
    ):
        raise ValueError("Stage-3 cell validation count differs from its frozen input budget")
    oof = record.get("oof_prediction", {})
    if oof.get("scope") != "sampled buffered-development OOF validation only":
        raise ValueError("Stage-3 OOF prediction scope is invalid")
    if int(oof.get("validation_voxels", 0)) != int(metrics["valid_voxels"]):
        raise ValueError("Stage-3 OOF prediction count differs from metrics")

    lock = record.get("resources", {}).get("gpu_lock", {})
    if model_id in GPU_MODELS:
        if record["model_config"].get("device") != "cuda:0":
            raise ValueError("Stage-3 GPU model did not use cuda:0")
        if lock.get("required") is not True or lock.get("acquired") is not True:
            raise ValueError("Stage-3 GPU result lacks acquired lock evidence")
        if lock.get("mechanism") != "external flock -w 900":
            raise ValueError("Stage-3 GPU result lacks external flock mechanism evidence")
        if lock.get("timeout_seconds") != 900:
            raise ValueError("Stage-3 GPU lock timeout differs from 900 seconds")
        wait_seconds = float(lock.get("wait_seconds"))
        if not math.isfinite(wait_seconds) or wait_seconds < 0 or wait_seconds > 900:
            raise ValueError("Stage-3 GPU result has invalid lock wait evidence")
        if int(record["resources"].get("peak_cuda_bytes", 0)) <= 0:
            raise ValueError("Stage-3 GPU result lacks positive peak VRAM evidence")
    elif record["model_config"].get("device") != "cpu":
        raise ValueError("Stage-3 geostat result did not use the CPU resource lane")


def validate_record_set(records: Sequence[Mapping[str, Any]]) -> None:
    expected = set(expected_cell_keys())
    seen: set[tuple[str, str, int, int]] = set()
    for record in records:
        key = (
            str(record.get("lane")),
            str(record.get("model_id")),
            int(record.get("fold_id", -1)),
            int(record.get("repeat_id", -1)),
        )
        if key in seen:
            raise ValueError(f"duplicate Stage-3 cell: {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"unexpected or cross-lane Stage-3 cell: {key}")
        validate_cell_record(record, *key)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"missing expected Stage-3 cells: {missing[:3]} (total={len(missing)})")


def _bootstrap_ci(values: Sequence[float], *, seed: int) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.isfinite(array).all():
        return None
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(10000, array.size), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(sample, [0.025, 0.975])]


def _aggregate_model(
    mode: str, model_id: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record["lane"] == mode and record["model_id"] == model_id
    ]
    passed = [record for record in selected if record["status"] == "passed"]
    fold_rmse = {
        str(fold_id): float(
            np.mean(
                [
                    record["metrics"]["rmse"]
                    for record in passed
                    if record["fold_id"] == fold_id
                ]
            )
        )
        for fold_id in FOLD_IDS
        if any(record["fold_id"] == fold_id for record in passed)
    }
    seed_rmse = {
        str(repeat_id): float(
            np.mean(
                [
                    record["metrics"]["rmse"]
                    for record in passed
                    if record["repeat_id"] == repeat_id
                ]
            )
        )
        for repeat_id in range(len(REPEAT_SEEDS))
        if any(record["repeat_id"] == repeat_id for record in passed)
    }
    metrics: dict[str, Any] = {}
    for name in ("rmse", "mae", "spectral_log_rmse"):
        values = [float(record["metrics"][name]) for record in passed]
        metrics[name] = {
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    expected = len(FOLD_IDS) * len(REPEAT_SEEDS)
    completion_rate = len(passed) / expected
    return {
        "model_id": model_id,
        "expected_cells": expected,
        "passed_cells": len(passed),
        "completion_rate": completion_rate,
        "eligible": completion_rate >= 0.8,
        "status_counts": {
            status: sum(record["status"] == status for record in selected)
            for status in ("passed", "skipped", "failed", "timeout")
        },
        "metrics": metrics,
        "rmse_95pct_bootstrap_ci": _bootstrap_ci(
            list(fold_rmse.values()),
            seed=SeedTree(ROOT_SEED).seed("diagnostic", "stage3", mode, model_id),
        ),
        "bootstrap_unit": "fold mean RMSE; 10000 deterministic resamples",
        "worst_fold_rmse": max(fold_rmse.values()) if fold_rmse else None,
        "fold_mean_rmse": fold_rmse,
        "seed_mean_rmse": seed_rmse,
        "seed_std_rmse": (
            float(np.std(list(seed_rmse.values()))) if len(seed_rmse) >= 2 else None
        ),
        "resources": {
            "mean_wall_seconds": (
                float(np.mean([record["wall_seconds"] for record in passed]))
                if passed
                else None
            ),
            "max_peak_rss_kib": (
                max(int(record["resources"]["peak_rss_kib"]) for record in passed)
                if passed
                else None
            ),
            "max_peak_cuda_bytes": (
                max(int(record["resources"]["peak_cuda_bytes"]) for record in passed)
                if passed
                else None
            ),
        },
        "result_hashes": [record["result_hash"] for record in passed],
    }


def build_leaderboard(mode: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entries = [_aggregate_model(mode, model_id, records) for model_id in MODELS[mode]]
    expected = len(MODELS[mode]) * len(FOLD_IDS) * len(REPEAT_SEEDS)
    passed = sum(entry["passed_cells"] for entry in entries)
    completion_rate = passed / expected
    rankable = completion_rate >= 0.8 and all(entry["eligible"] for entry in entries)
    if rankable:
        ordered = sorted(
            entries,
            key=lambda entry: (
                float(entry["metrics"]["rmse"]["mean"]),
                float(entry["worst_fold_rmse"]),
                float(entry["seed_std_rmse"]),
                float(entry["resources"]["mean_wall_seconds"]),
                str(entry["model_id"]),
            ),
        )
        rank_by_model = {
            entry["model_id"]: rank for rank, entry in enumerate(ordered, start=1)
        }
    else:
        ordered = entries
        rank_by_model = {entry["model_id"]: None for entry in entries}
    return {
        "schema_version": LEADERBOARD_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "lane": mode,
        "task_id": p4.protocol(mode).task_id,
        "split_hash": _frozen_split_hashes()[mode],
        "fold_ids": list(FOLD_IDS),
        "repeat_seeds": list(REPEAT_SEEDS),
        "ranking_metric": "mean development OOF RMSE",
        "metric_direction": "minimize",
        "tie_breakers": ["worst_fold_rmse", "seed_std_rmse", "mean_wall_seconds", "model_id"],
        "expected_cells": expected,
        "passed_cells": passed,
        "legal_completion_rate": completion_rate,
        "rankable": rankable,
        "not_rankable_reason": (
            None
            if rankable
            else "lane or at least one frozen model has <80% legally completed cells"
        ),
        "development_only": True,
        "frozen_test_i_blocks_loaded": [],
        "entries": [
            {"rank": rank_by_model[entry["model_id"]], **entry} for entry in ordered
        ],
    }


def _portable_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Stage-3 runtime artifact is outside the project worktree") from exc
    return relative.as_posix()


def build_oof_manifest(
    cell_root: Path, records: Sequence[Mapping[str, Any]], output_dir: Path
) -> tuple[dict[str, Any], Path]:
    entries: list[dict[str, Any]] = []
    for record in records:
        if record["status"] != "passed":
            continue
        path = _cell_dir(
            cell_root,
            record["lane"],
            record["model_id"],
            int(record["fold_id"]),
            int(record["repeat_id"]),
        ) / record["oof_prediction"]["archive_name"]
        if not path.is_file():
            raise FileNotFoundError(f"missing Stage-3 OOF archive: {path}")
        if hash_file(path) != record["oof_prediction"]["sha256"]:
            raise ValueError("Stage-3 OOF archive hash differs from its cell record")
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["lane"].item()) != record["lane"]:
                raise ValueError("Stage-3 OOF archive crossed lane namespaces")
            if str(archive["model_id"].item()) != record["model_id"]:
                raise ValueError("Stage-3 OOF archive model identity mismatch")
            if int(archive["fold_id"].item()) != record["fold_id"]:
                raise ValueError("Stage-3 OOF archive fold identity mismatch")
            if int(archive["repeat_id"].item()) != record["repeat_id"]:
                raise ValueError("Stage-3 OOF archive repeat identity mismatch")
            if int(archive["repeat_seed"].item()) != record["repeat_seed"]:
                raise ValueError("Stage-3 OOF archive repeat seed mismatch")
            count = int(np.asarray(archive["target"]).size)
            if count != record["metrics"]["valid_voxels"]:
                raise ValueError("Stage-3 OOF archive voxel count differs from metrics")
        portable_path = _portable_project_path(path)
        if not portable_path.startswith("_tmp/p5_stage3_reconstruction/"):
            raise ValueError("large Stage-3 OOF predictions must stay in the ignored private root")
        entries.append(
            {
                "cell_id": record["cell_id"],
                "task_id": record["task_id"],
                "lane": record["lane"],
                "model_id": record["model_id"],
                "fold_id": record["fold_id"],
                "repeat_id": record["repeat_id"],
                "repeat_seed": record["repeat_seed"],
                "path": portable_path,
                "sha256": hash_file(path),
                "bytes": path.stat().st_size,
                "validation_voxels": count,
                "scope": "sampled buffered-development OOF validation only",
                "frozen_test_i_blocks_loaded": [],
            }
        )
    manifest = {
        "schema_version": OOF_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "development_only": True,
        "prediction_storage": "ignored track-private runtime directory",
        "prediction_budget_note": (
            "Each cell archives the same 2048-voxel maximum used in Stage 2; this is sampled OOF, "
            "not a full-volume export."
        ),
        "expected_cells": EXPECTED_CELLS,
        "archived_passed_cells": len(entries),
        "split_hashes": _frozen_split_hashes(),
        "frozen_test_i_blocks_loaded": [],
        "entries": entries,
        "missing_prediction_cell_ids": [
            record["cell_id"] for record in records if record["status"] != "passed"
        ],
    }
    path = output_dir / "p5_stage3_oof_manifest.json"
    atomic_write_json(path, manifest)
    return manifest, path


def _load_oof_entry(entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = PROJECT_ROOT / str(entry["path"])
    if hash_file(path) != entry["sha256"]:
        raise ValueError("visualizer source OOF hash mismatch")
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]) for name in archive.files}
    if "test" in payload:
        raise ValueError("visualizer archive unexpectedly contains a test field")
    return payload


def _dense_crop(
    values: np.ndarray, indices_kji: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices_kji, dtype=np.int64)
    minimum = indices.min(axis=0)
    maximum = indices.max(axis=0)
    shape = tuple(int(value) for value in maximum - minimum + 1)
    dense = np.full(shape, np.nan, dtype=np.float64)
    local = indices - minimum
    dense[tuple(local.T)] = np.asarray(values, dtype=np.float64)
    return dense, minimum


def _slice_with_most_support(volume: np.ndarray, axis: int) -> np.ndarray:
    support = np.sum(np.isfinite(volume), axis=tuple(item for item in range(3) if item != axis))
    index = int(np.argmax(support))
    return np.take(volume, index, axis=axis)


def _radial_spectrum(volume: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(volume)
    filled = np.where(finite, volume, float(np.nanmean(volume)))
    magnitude = np.abs(np.fft.fftshift(np.fft.fftn(filled)))
    grids = np.meshgrid(
        *[np.fft.fftshift(np.fft.fftfreq(size)) for size in filled.shape],
        indexing="ij",
    )
    radius = np.sqrt(sum(grid**2 for grid in grids))
    edges = np.linspace(0.0, float(radius.max()) + 1e-12, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.asarray(
        [
            float(np.mean(magnitude[(radius >= left) & (radius < right)]))
            if np.any((radius >= left) & (radius < right))
            else np.nan
            for left, right in zip(edges[:-1], edges[1:])
        ]
    )
    return centers, profile


def _empirical_variogram(
    xyz: np.ndarray, values: np.ndarray, *, seed: int, bins: int = 15
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.distance import pdist

    xyz = np.asarray(xyz, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(values) > 600:
        selected = np.random.default_rng(seed).choice(len(values), size=600, replace=False)
        xyz = xyz[selected]
        values = values[selected]
    distance = pdist(xyz * np.asarray([1.0, 1.0, 3.0]))
    semivariance = 0.5 * pdist(values[:, None], metric="sqeuclidean")
    positive = distance > 0
    distance = distance[positive]
    semivariance = semivariance[positive]
    if distance.size == 0:
        return np.asarray([]), np.asarray([])
    edges = np.linspace(0.0, float(np.quantile(distance, 0.95)), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    gamma = np.asarray(
        [
            float(np.mean(semivariance[(distance >= left) & (distance < right)]))
            if np.any((distance >= left) & (distance < right))
            else np.nan
            for left, right in zip(edges[:-1], edges[1:])
        ]
    )
    return centers, gamma


def _choose_display_model(leaderboard: Mapping[str, Any]) -> tuple[str, str]:
    entries = [
        entry
        for entry in leaderboard["entries"]
        if entry["passed_cells"] > 0 and entry["metrics"]["rmse"]["mean"] is not None
    ]
    if not entries:
        raise RuntimeError(f"{leaderboard['lane']} has no passed OOF prediction to visualize")
    if leaderboard["rankable"]:
        winner = min(entries, key=lambda entry: int(entry["rank"]))
        return str(winner["model_id"]), "ranked Stage-3 winner"
    winner = min(entries, key=lambda entry: float(entry["metrics"]["rmse"]["mean"]))
    return str(winner["model_id"]), "display-only best available; lane not rankable"


def _render_lane_figure(
    mode: str,
    leaderboard: Mapping[str, Any],
    oof_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_id, selection_note = _choose_display_model(leaderboard)
    source_entries = [
        entry
        for entry in oof_manifest["entries"]
        if entry["lane"] == mode and entry["model_id"] == model_id
    ]
    if not source_entries:
        raise RuntimeError(f"no OOF archive exists for {mode}/{model_id}")
    payloads = [(entry, _load_oof_entry(entry)) for entry in source_entries]
    cell_records = [
        record
        for record in records
        if record["lane"] == mode
        and record["model_id"] == model_id
        and record["status"] == "passed"
    ]
    median_record = sorted(cell_records, key=lambda item: item["metrics"]["rmse"])[
        len(cell_records) // 2
    ]
    representative_entry, representative = next(
        (entry, payload)
        for entry, payload in payloads
        if entry["cell_id"] == median_record["cell_id"]
    )
    truth, _ = _dense_crop(representative["target"], representative["indices_kji"])
    prediction, _ = _dense_crop(
        representative["prediction"], representative["indices_kji"]
    )
    residual = prediction - truth

    fig, axes = plt.subplots(5, 3, figsize=(17, 23), constrained_layout=True)
    axis_names = ("K / time-depth", "J / crossline", "I / inline")
    for row, (axis, axis_name) in enumerate(zip(range(3), axis_names)):
        slices = (
            _slice_with_most_support(truth, axis),
            _slice_with_most_support(prediction, axis),
            _slice_with_most_support(residual, axis),
        )
        for column, (values, title) in enumerate(
            zip(slices, ("reference porosity", "reconstructed porosity", "residual"))
        ):
            kwargs: dict[str, Any] = {"origin": "lower", "aspect": "auto"}
            if column == 2:
                limit = max(float(np.nanmax(np.abs(values))), 1e-6)
                kwargs.update({"cmap": "coolwarm", "vmin": -limit, "vmax": limit})
            else:
                kwargs.update({"cmap": "viridis"})
            image = axes[row, column].imshow(values, **kwargs)
            axes[row, column].set_title(f"{axis_name}: {title}")
            fig.colorbar(image, ax=axes[row, column], shrink=0.75)

    target_all = np.concatenate([payload["target"].reshape(-1) for _, payload in payloads])
    prediction_all = np.concatenate(
        [payload["prediction"].reshape(-1) for _, payload in payloads]
    )
    target_sorted = np.sort(target_all)
    prediction_sorted = np.sort(prediction_all)
    axes[3, 0].plot(target_sorted, np.linspace(0, 1, len(target_sorted)), label="reference")
    axes[3, 0].plot(
        prediction_sorted, np.linspace(0, 1, len(prediction_sorted)), label="prediction"
    )
    axes[3, 0].set(title="OOF porosity CDF", xlabel="porosity", ylabel="cumulative probability")
    axes[3, 0].legend()

    frequency, truth_spectrum = _radial_spectrum(truth)
    _, prediction_spectrum = _radial_spectrum(prediction)
    axes[3, 1].plot(frequency, truth_spectrum, label="reference")
    axes[3, 1].plot(frequency, prediction_spectrum, label="prediction")
    axes[3, 1].set_yscale("log")
    axes[3, 1].set(title="Representative 3-D radial spectrum", xlabel="spatial frequency")
    axes[3, 1].legend()

    repeat_zero = [payload for entry, payload in payloads if int(entry["repeat_id"]) == 0]
    xyz = np.concatenate([payload["xyz"] for payload in repeat_zero])
    variogram_target = np.concatenate([payload["target"] for payload in repeat_zero])
    variogram_prediction = np.concatenate([payload["prediction"] for payload in repeat_zero])
    lag, truth_gamma = _empirical_variogram(
        xyz, variogram_target, seed=SeedTree(ROOT_SEED).seed("diagnostic", mode, "truth")
    )
    _, prediction_gamma = _empirical_variogram(
        xyz,
        variogram_prediction,
        seed=SeedTree(ROOT_SEED).seed("diagnostic", mode, "truth"),
    )
    axes[3, 2].plot(lag, truth_gamma, marker="o", label="reference")
    axes[3, 2].plot(lag, prediction_gamma, marker="o", label="prediction")
    axes[3, 2].set(title="Empirical variogram", xlabel="anisotropic lag", ylabel="semivariance")
    axes[3, 2].legend()

    distance = np.concatenate(
        [payload["distance_to_fold_train_well"].reshape(-1) for _, payload in payloads]
    )
    absolute_error = np.abs(prediction_all - target_all)
    finite_distance = np.isfinite(distance) & (distance >= 0)
    if np.sum(finite_distance) >= 10 and np.ptp(distance[finite_distance]) > 0:
        edges = np.quantile(distance[finite_distance], np.linspace(0, 1, 7))
        edges = np.unique(edges)
        centers: list[float] = []
        errors: list[float] = []
        for left, right in zip(edges[:-1], edges[1:]):
            selected = finite_distance & (distance >= left) & (distance <= right)
            if np.any(selected):
                centers.append(float((left + right) / 2))
                errors.append(float(np.mean(absolute_error[selected])))
        axes[4, 0].plot(centers, errors, marker="o")
    else:
        axes[4, 0].text(0.5, 0.5, "No fold-train well coordinate in these folds", ha="center")
    axes[4, 0].set(
        title="Error vs distance to fold-train well",
        xlabel="anisotropic normalized distance",
        ylabel="MAE",
    )

    heatmap = np.full((len(FOLD_IDS), len(REPEAT_SEEDS)), np.nan)
    for record in cell_records:
        heatmap[int(record["fold_id"]), int(record["repeat_id"])] = record["metrics"]["rmse"]
    image = axes[4, 1].imshow(heatmap, cmap="magma", aspect="auto")
    axes[4, 1].set_xticks(range(3), [f"seed {index}" for index in range(3)])
    axes[4, 1].set_yticks(range(5), [f"fold {index}" for index in range(5)])
    axes[4, 1].set_title("Fold × repeat-seed RMSE")
    fig.colorbar(image, ax=axes[4, 1], shrink=0.75)

    winning_entry = next(
        entry for entry in leaderboard["entries"] if entry["model_id"] == model_id
    )
    axes[4, 2].axis("off")
    axes[4, 2].text(
        0.0,
        1.0,
        "\n".join(
            [
                f"lane: {mode}",
                f"display model: {model_id}",
                selection_note,
                f"mean OOF RMSE: {winning_entry['metrics']['rmse']['mean']:.6f}",
                f"worst fold RMSE: {winning_entry['worst_fold_rmse']:.6f}",
                f"seed std RMSE: {winning_entry['seed_std_rmse']:.6f}",
                f"representative: {representative_entry['cell_id']}",
                "development OOF only; frozen test was not read or scored",
                "strict and conditional lanes are never pooled",
            ]
        ),
        va="top",
        family="monospace",
    )
    fig.suptitle(
        f"Volve reconstruction Stage-3 — {mode} — {model_id}", fontsize=16
    )
    figure_path = output_dir / "_outputs" / f"p5_stage3_reconstruction_{mode}.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = figure_path.with_name(figure_path.name + ".tmp")
    fig.savefig(temporary, format="png", dpi=150)
    plt.close(fig)
    os.replace(temporary, figure_path)
    return figure_path, {
        "lane": mode,
        "task_id": leaderboard["task_id"],
        "display_model": model_id,
        "selection_note": selection_note,
        "rankable": leaderboard["rankable"],
        "path": _portable_project_path(figure_path),
        "sha256": hash_file(figure_path),
        "bytes": figure_path.stat().st_size,
        "source_oof_cell_ids": [entry["cell_id"] for entry in source_entries],
        "source_oof_hashes": [entry["sha256"] for entry in source_entries],
        "diagnostics": [
            "K/time-depth truth-prediction-residual",
            "J/crossline truth-prediction-residual",
            "I/inline truth-prediction-residual",
            "porosity CDF",
            "3-D radial spectrum",
            "empirical variogram",
            "distance-to-fold-train-well error",
            "fold-by-repeat-seed RMSE",
        ],
        "frozen_test_i_blocks_loaded": [],
    }


def render_visualizations(
    leaderboards: Mapping[str, Mapping[str, Any]],
    oof_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    figures: dict[str, Any] = {}
    for mode in MODES:
        _, figures[mode] = _render_lane_figure(
            mode, leaderboards[mode], oof_manifest, records, output_dir
        )
    payload = {
        "schema_version": VISUALIZATION_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "generator": "_pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage3.py",
        "input_manifest": "p5_stage3_oof_manifest.json",
        "development_only": True,
        "modes_are_independent": True,
        "frozen_test_i_blocks_loaded": [],
        "figures": figures,
    }
    path = output_dir / "p5_stage3_visualization_manifest.json"
    atomic_write_json(path, payload)
    return payload, path


def collate(cell_root: Path, output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for mode, model_id, fold_id, repeat_id in expected_cell_keys():
        path = _cell_dir(cell_root, mode, model_id, fold_id, repeat_id) / "status.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing expected Stage-3 cell: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    validate_record_set(records)

    results_path = _atomic_write_jsonl(output_dir / "p5_stage3_results.jsonl", records)
    leaderboards: dict[str, dict[str, Any]] = {}
    leaderboard_paths: dict[str, Path] = {}
    for mode in MODES:
        leaderboard = build_leaderboard(mode, records)
        path = output_dir / f"p5_stage3_leaderboard_{mode}.json"
        atomic_write_json(path, leaderboard)
        leaderboards[mode] = leaderboard
        leaderboard_paths[mode] = path

    oof_manifest, oof_path = build_oof_manifest(cell_root, records, output_dir)
    visualization_manifest, visualization_path = render_visualizations(
        leaderboards, oof_manifest, records, output_dir
    )
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("passed", "skipped", "failed", "timeout")
    }
    source_lock_path = PROJECT_ROOT / "_models" / "reconstruction" / "source_lock.json"
    not_feasible = [
        {
            "cell_id": record["cell_id"],
            "status": record["status"],
            "reason": record["reason"],
        }
        for record in records
        if record["status"] != "passed"
    ]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "baseline_commit": "16bebd18a0bc722afcbc4b841610bf76ce9503e4",
        "root_seed": ROOT_SEED,
        "repeat_seeds": list(REPEAT_SEEDS),
        "fold_ids": list(FOLD_IDS),
        "expected_cells": EXPECTED_CELLS,
        "attempted_cells": len(records),
        "counts": counts,
        "legal_completion_rate": counts["passed"] / EXPECTED_CELLS,
        "rankable_by_lane": {
            mode: bool(leaderboards[mode]["rankable"]) for mode in MODES
        },
        "development_only": True,
        "frozen_test_i_blocks_loaded": [],
        "historical_test_metrics_read": False,
        "modes_are_independent": True,
        "hpo_performed": False,
        "stage2_configuration_reused": True,
        "split_hashes": _frozen_split_hashes(),
        "cache_contract_hashes": sorted(
            {str(record["cache_contract_hash"]) for record in records}
        ),
        "source_sha256": {
            "_pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage3.py": hash_file(
                HERE / "reconstruction_p5_stage3.py"
            ),
            "_pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage2.py": hash_file(
                HERE / "reconstruction_p5_stage2.py"
            ),
            "_pipelines/02_task_datasets/reconstruction/p5_stage1.py": hash_file(
                HERE / "p5_stage1.py"
            ),
        },
        "source_lock": "_models/reconstruction/source_lock.json",
        "source_lock_sha256": hash_file(source_lock_path),
        "results": results_path.name,
        "results_sha256": hash_file(results_path),
        "leaderboards": {
            mode: {
                "path": path.name,
                "sha256": hash_file(path),
                "rankable": leaderboards[mode]["rankable"],
            }
            for mode, path in leaderboard_paths.items()
        },
        "oof_manifest": {
            "path": oof_path.name,
            "sha256": hash_file(oof_path),
            "archived_passed_cells": oof_manifest["archived_passed_cells"],
        },
        "visualization_manifest": {
            "path": visualization_path.name,
            "sha256": hash_file(visualization_path),
            "figures": {
                mode: {
                    "path": visualization_manifest["figures"][mode]["path"],
                    "sha256": visualization_manifest["figures"][mode]["sha256"],
                }
                for mode in MODES
            },
        },
        "not_feasible_cells": not_feasible,
        "technical_retries": [
            {"cell_id": record["cell_id"], **record["technical_retry"]}
            for record in records
            if record.get("technical_retry") is not None
        ],
        "scientific_note": (
            "Five-fold x three-seed buffered-development confirmation only; no frozen-test "
            "array, label, prediction, or historical test metric was read."
        ),
    }
    atomic_write_json(output_dir / "p5_stage3_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-cache", help="freeze all five fold-local development caches"
    )
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument(
        "--stage2-cache",
        type=Path,
        required=True,
        help="read-only Stage-2 cache containing the frozen split manifests",
    )

    cell = subparsers.add_parser(
        "run-cell", help="run one frozen model/fold/repeat development cell"
    )
    cell.add_argument("--mode", choices=MODES, required=True)
    cell.add_argument(
        "--model", choices=tuple(dict.fromkeys((*MODELS["strict"], *MODELS["conditional"]))), required=True
    )
    cell.add_argument("--fold-id", type=int, choices=FOLD_IDS, required=True)
    cell.add_argument("--repeat-id", type=int, choices=range(len(REPEAT_SEEDS)), required=True)
    cell.add_argument("--cache-dir", type=Path, required=True)
    cell.add_argument("--cell-root", type=Path, required=True)
    cell.add_argument("--device", choices=("cpu", "cuda:0"), required=True)

    aggregate = subparsers.add_parser(
        "collate", help="validate 90 cells and write portable Stage-3 evidence"
    )
    aggregate.add_argument("--cell-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, default=HERE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-cache":
        payload = prepare_cache(args.data_dir, args.cache_dir, args.stage2_cache)
        print(
            json.dumps(
                {
                    "cache_manifest": str(args.cache_dir / "cache_manifest.json"),
                    "modes": {
                        mode: payload["modes"][mode]["effective_n_splits"]
                        for mode in MODES
                    },
                    "repeat_seeds": payload["repeat_seeds"],
                    "frozen_test_i_blocks_loaded": payload["frozen_test_i_blocks_loaded"],
                },
                indent=2,
            )
        )
    elif args.command == "run-cell":
        record = run_cell(
            mode=args.mode,
            model_id=args.model,
            fold_id=args.fold_id,
            repeat_id=args.repeat_id,
            cache_dir=args.cache_dir,
            cell_root=args.cell_root,
            device=args.device,
        )
        print(
            json.dumps(
                {
                    "cell_id": record["cell_id"],
                    "status": record["status"],
                    "metrics": record.get("metrics"),
                    "reason": record.get("reason"),
                },
                indent=2,
            )
        )
        if record["status"] in {"failed", "timeout"}:
            raise SystemExit(1)
    elif args.command == "collate":
        print(json.dumps(collate(args.cell_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
