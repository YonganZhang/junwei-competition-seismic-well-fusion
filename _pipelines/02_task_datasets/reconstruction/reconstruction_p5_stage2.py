#!/usr/bin/env python3
"""Fixed-budget P5 Stage-2 pilot for Volve 3-D reconstruction.

Only P4's first buffered development fold is accepted.  The pilot has no
frozen-test command, loader, path, label, or metric surface.  Strict and
conditional cells use separate TaskSpecs, caches, status roots, and
leaderboards.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from ml_framework.contracts import ModelBatch, TaskSpec  # noqa: E402
from ml_framework.model_discovery import DiscoveredModel, discover_model  # noqa: E402
from ml_framework.seeding import DEFAULT_ROOT_SEED, SeedTree, seed_everything  # noqa: E402

from _models.reconstruction._p5_adapter import AdapterSkip  # noqa: E402


_STAGE1_MODULE_NAME = "reconstruction_p5_stage1_contract"
if _STAGE1_MODULE_NAME in sys.modules:
    stage1 = sys.modules[_STAGE1_MODULE_NAME]
else:
    _stage1_spec = importlib.util.spec_from_file_location(
        _STAGE1_MODULE_NAME, HERE / "p5_stage1.py"
    )
    if _stage1_spec is None or _stage1_spec.loader is None:
        raise RuntimeError("cannot load reconstruction-prefixed Stage-1 contract module")
    stage1 = importlib.util.module_from_spec(_stage1_spec)
    sys.modules[_STAGE1_MODULE_NAME] = stage1
    _stage1_spec.loader.exec_module(stage1)


MODES = ("strict", "conditional")
CANDIDATES = stage1.STAGE1_MODELS
ROOT_SEED = 2693
GPU_LOCK_PATH = Path.home() / ".cache" / "volve-p5" / "locks" / "gpu0.lock"
SCHEMA_VERSION = "p5-stage2-reconstruction-cell-v1"
CACHE_SCHEMA_VERSION = "p5-stage2-reconstruction-cache-v1"

# One fixed input budget is shared by every candidate within each
# representation.  Point and volume models see exactly the same sampled
# validation voxels; volume training masks expose the same total number of
# target voxels as the point lane.
POINT_TRAIN_VOXELS = 512
VALIDATION_VOXELS = 2048
VOLUME_TRAIN_PATCHES = 4
VOLUME_TARGETS_PER_PATCH = POINT_TRAIN_VOXELS // VOLUME_TRAIN_PATCHES
TINY_GATE_UPDATES = 3
POINT_NEURAL_UPDATES = 100
VOLUME_NEURAL_UPDATES = 20

PRESERVED_STAGE1_SKIPS = {
    "mpslib_snesim3d": {
        "code": "missing_legal_training_image",
        "message": "No independently licensed and approved MPS training image is available.",
    },
    "tcnn_hashgrid_inr": {
        "code": "dependency_missing",
        "message": "tinycudann remains unavailable; Stage 2 does not compile or install extensions.",
    },
}
GPU_CANDIDATES = {
    "gpytorch_svgp",
    "monai_basicunet3d",
    "monai_segresnet3d",
    "neuralop_fno3d",
    "tcnn_hashgrid_inr",
    "siren_inr",
}


class _LazyP4:
    """Keep HDF5-bound P4 imports out of dependency-isolated cell workers."""

    module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self.module is None:
            self.module = importlib.import_module("p4_reconstruction")
        return getattr(self.module, name)


p4 = _LazyP4()


class PilotTimeout(TimeoutError):
    """A Stage-2 cell exceeded its frozen model-class wall-time budget."""


def budget_for(capabilities: Mapping[str, Any]) -> dict[str, Any]:
    representation = str(capabilities.get("batch_representation", ""))
    trainable = bool(capabilities.get("trainable"))
    if not trainable:
        return {
            "model_class": "traditional_cpu",
            "max_updates": 1,
            "max_wall_seconds": 300,
        }
    if representation == "volume":
        return {
            "model_class": "3d_neural_or_operator",
            "max_updates": VOLUME_NEURAL_UPDATES,
            "protocol_update_cap": 80,
            "max_wall_seconds": 900,
        }
    return {
        "model_class": "point_neural",
        "max_updates": POINT_NEURAL_UPDATES,
        "protocol_update_cap": 200,
        "max_wall_seconds": 600,
    }


def validate_budget(budget: Mapping[str, Any], updates: int, wall_seconds: float) -> None:
    if updates < 0 or updates > int(budget["max_updates"]):
        raise ValueError("Stage-2 update count exceeds the frozen budget")
    protocol_cap = budget.get("protocol_update_cap")
    if protocol_cap is not None and updates > int(protocol_cap):
        raise ValueError("Stage-2 update count exceeds the protocol class cap")
    if wall_seconds < 0 or wall_seconds > float(budget["max_wall_seconds"]):
        raise ValueError("Stage-2 model wall time exceeds the frozen budget")


def _sample_indices(size: int, maximum: int) -> np.ndarray:
    count = min(int(size), int(maximum))
    if count <= 0:
        raise ValueError("cannot sample an empty Stage-2 population")
    return np.linspace(0, size - 1, num=count, dtype=np.int64)


def _replace_batch_mask(
    batch: ModelBatch,
    task_spec: TaskSpec,
    selected_flat_indices: np.ndarray,
    *,
    coordinates: Mapping[str, np.ndarray] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ModelBatch:
    target_name = task_spec.targets[0]
    if batch.targets is None:
        raise ValueError("Stage-2 volume batch has no target")
    target = np.asarray(batch.targets[target_name])
    mask = np.zeros(target.size, dtype=bool)
    mask[np.asarray(selected_flat_indices, dtype=np.int64)] = True
    mask = mask.reshape(target.shape)
    if not np.any(mask):
        raise ValueError("Stage-2 volume mask is empty")
    merged_coordinates = dict(batch.coordinates)
    if coordinates:
        merged_coordinates.update(coordinates)
    merged_metadata = dict(batch.metadata)
    if metadata:
        merged_metadata.update(metadata)
    return ModelBatch(
        inputs=dict(batch.inputs),
        targets=dict(batch.targets),
        input_masks=dict(batch.input_masks),
        target_masks={target_name: mask},
        sample_ids=list(batch.sample_ids),
        groups={key: list(value) for key, value in batch.groups.items()},
        coordinates=merged_coordinates,
        metadata=merged_metadata,
    )


def _point_batch(
    task_spec: TaskSpec,
    *,
    mode: str,
    split: str,
    features: np.ndarray,
    target: np.ndarray,
    xyz: np.ndarray,
    metric_indices_kji: np.ndarray | None,
    volume_shape_kji: tuple[int, int, int],
    feature_names: Sequence[str],
    constraint_count: int,
) -> ModelBatch:
    target_name = task_spec.targets[0]
    coordinates: dict[str, np.ndarray] = {"xyz": np.asarray(xyz, dtype=np.float64)}
    if metric_indices_kji is not None:
        coordinates["metric_indices_kji"] = np.asarray(metric_indices_kji, dtype=np.int64)
        coordinates["volume_shape_kji"] = np.asarray(volume_shape_kji, dtype=np.int64)
    values = np.asarray(target, dtype=np.float64)
    return ModelBatch(
        inputs={"features": np.asarray(features, dtype=np.float64)},
        targets={target_name: values},
        input_masks={},
        target_masks={target_name: np.ones(values.shape, dtype=bool)},
        sample_ids=[f"{mode}:stage2:fold0:{split}:points"],
        groups={"fold": ["0"], "evaluation_mode": [mode]},
        coordinates=coordinates,
        metadata={
            "evaluation_mode": mode,
            "split": f"development_{split}",
            "feature_names": list(feature_names),
            "constraint_count_supplied": int(constraint_count),
            "frozen_test_i_blocks_loaded": [],
        },
    )


def _record_metric_flat_indices(record: Any, global_indices: np.ndarray) -> np.ndarray:
    start = np.asarray(record.location.patch_start_kji, dtype=np.int64)
    local = np.asarray(global_indices, dtype=np.int64) - start[None, :]
    shape = tuple(int(value) for value in record.label.shape)
    if np.any(local < 0) or np.any(local >= np.asarray(shape)[None, :]):
        raise ValueError("validation metric index falls outside the selected patch")
    return np.ravel_multi_index(tuple(local.T), shape)


def prepare_mode_cache(
    mode: str,
    catalog: Sequence[Any],
    records: Sequence[Any],
    cache_dir: Path,
) -> dict[str, Any]:
    """Freeze P4 fold 0 and materialize representation-neutral pilot batches."""
    spec = p4.task_spec(mode)
    active = p4.protocol(mode)
    manifest = p4.build_spatial_manifest(mode, catalog)
    fold = manifest.folds[0]
    if fold.fold_id != 0:
        raise RuntimeError("Stage-2 must use the first P4 development fold")
    development = [item for item in records if item.i_block in active.development_i_blocks]
    loaded_blocks = sorted({item.i_block for item in development})
    if loaded_blocks != sorted(active.development_i_blocks):
        raise RuntimeError(f"{mode} cache lacks the complete development I-block set")
    if set(loaded_blocks) & (set(active.test_i_blocks) | set(active.guard_i_blocks)):
        raise RuntimeError(f"{mode} cache crossed the test/guard firewall")
    prepared = p4.prepare_fold(mode, fold, development)
    if mode == "strict" and prepared.constraint_audit["constraints_supplied_to_model"] != 0:
        raise RuntimeError("strict Stage-2 cache supplied target-derived constraints")
    by_id = {item.sample_id: item for item in development}
    train_records = tuple(by_id[item] for item in fold.purge["effective_train_sample_ids"])
    validation_records = tuple(by_id[item] for item in fold.validation_sample_ids)
    bundle = stage1.DevelopmentBundle(mode, fold, prepared, train_records, validation_records)

    train_indices = _sample_indices(prepared.train_target.size, POINT_TRAIN_VOXELS)
    constraint_count = int(prepared.constraint_audit["constraints_supplied_to_model"])
    point_train = _point_batch(
        spec,
        mode=mode,
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
    selected_validation = eligible[_sample_indices(eligible.size, VALIDATION_VOXELS)]
    metric_indices = prepared.validation_cells.indices_kji[selected_validation]
    point_validation = _point_batch(
        spec,
        mode=mode,
        split="validation",
        features=prepared.validation_features[selected_validation],
        target=prepared.validation_target[selected_validation],
        xyz=prepared.validation_cells.coordinates[selected_validation],
        metric_indices_kji=metric_indices,
        volume_shape_kji=prepared.validation_cells.volume_shape_kji,
        feature_names=prepared.feature_names,
        constraint_count=constraint_count,
    )

    volume_validation_base = stage1._record_feature_volume(  # noqa: SLF001
        spec, bundle, validation_record, validation=True
    )
    validation_flat = _record_metric_flat_indices(validation_record, metric_indices)
    volume_validation = _replace_batch_mask(
        volume_validation_base,
        spec,
        validation_flat,
        coordinates={
            "metric_indices_kji": metric_indices,
            "volume_shape_kji": np.asarray(
                prepared.validation_cells.volume_shape_kji, dtype=np.int64
            ),
        },
        metadata={"shared_validation_voxel_count": int(len(metric_indices))},
    )
    point_target = np.asarray(point_validation.targets[spec.targets[0]])
    volume_target = np.asarray(volume_validation.targets[spec.targets[0]])[
        np.asarray(volume_validation.target_masks[spec.targets[0]], dtype=bool)
    ]
    if point_target.shape != volume_target.shape or not np.allclose(point_target, volume_target):
        raise RuntimeError("point/volume lanes do not share identical validation targets")

    ranked_train = sorted(
        train_records,
        key=lambda item: (-int(np.sum(item.seismic_patch[8] > 0.5)), item.sample_id),
    )[:VOLUME_TRAIN_PATCHES]
    volume_train: list[ModelBatch] = []
    for record in ranked_train:
        batch = stage1._record_feature_volume(spec, bundle, record, validation=False)  # noqa: SLF001
        active_flat = np.flatnonzero(
            np.asarray(batch.target_masks[spec.targets[0]], dtype=bool).reshape(-1)
        )
        selected = active_flat[_sample_indices(active_flat.size, VOLUME_TARGETS_PER_PATCH)]
        volume_train.append(
            _replace_batch_mask(
                batch,
                spec,
                selected,
                metadata={"fixed_train_target_voxels": int(len(selected))},
            )
        )
    if len(volume_train) != VOLUME_TRAIN_PATCHES:
        raise RuntimeError("Stage-2 volume lane lacks the fixed number of train patches")

    mode_dir = cache_dir / mode
    atomic_write_json(mode_dir / "task_spec.json", spec.to_dict())
    atomic_write_json(mode_dir / "split_manifest.json", manifest.to_dict())
    batches = {
        "point_train": point_train,
        "point_validation": point_validation,
        "volume_validation": volume_validation,
        **{f"volume_train_{index:02d}": batch for index, batch in enumerate(volume_train)},
    }
    batch_records: dict[str, Any] = {}
    for name, batch in batches.items():
        path = mode_dir / f"{name}.npz"
        stage1._write_cached_batch(path, batch, spec)  # noqa: SLF001
        batch_records[name] = {"sha256": hash_file(path), "bytes": path.stat().st_size}
    return {
        "task_id": spec.task_id,
        "evaluation_mode": mode,
        "split_hash": manifest.stable_hash(),
        "fold_id": fold.fold_id,
        "development_i_blocks": list(active.development_i_blocks),
        "guard_i_blocks": list(active.guard_i_blocks),
        "frozen_test_i_blocks": list(active.test_i_blocks),
        "frozen_test_i_blocks_loaded": [],
        "effective_train_sample_ids": list(fold.purge["effective_train_sample_ids"]),
        "validation_sample_ids": list(fold.validation_sample_ids),
        "purged_train_sample_ids": list(fold.purge["purged_train_sample_ids"]),
        "selected_validation_patch": validation_record.sample_id,
        "input_whitelist": list(spec.input_whitelist),
        "feature_names": list(prepared.feature_names),
        "constraint_audit": prepared.constraint_audit,
        "input_budget": {
            "point_train_voxels": int(len(train_indices)),
            "volume_train_patches": len(volume_train),
            "volume_targets_per_patch": VOLUME_TARGETS_PER_PATCH,
            "total_volume_train_target_voxels": int(
                sum(
                    np.asarray(batch.target_masks[spec.targets[0]], dtype=bool).sum()
                    for batch in volume_train
                )
            ),
            "shared_validation_voxels": int(len(selected_validation)),
        },
        "batches": batch_records,
    }


def prepare_cache(data_dir: Path, cache_dir: Path) -> dict[str, Any]:
    """Read metadata plus development arrays; never load frozen-test arrays."""
    catalog = p4.scan_patch_catalog(data_dir)
    records_by_mode = {
        mode: p4.load_patch_records(p4.protocol(mode).development_i_blocks, data_dir)
        for mode in MODES
    }
    report = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "source_container_names": ["train.h5", "test.h5"],
        "source_note": (
            "legacy physical containers; only mode-specific development I-block arrays were loaded"
        ),
        "frozen_test_i_blocks_loaded": [],
        "modes": {
            mode: prepare_mode_cache(mode, catalog, records_by_mode[mode], cache_dir)
            for mode in MODES
        },
    }
    atomic_write_json(cache_dir / "cache_manifest.json", report)
    return report


def load_mode_cache(
    cache_dir: Path,
    mode: str,
    representation: str,
) -> tuple[TaskSpec, list[ModelBatch], ModelBatch, dict[str, Any], str]:
    manifest_path = cache_dir / "cache_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("invalid reconstruction Stage-2 cache schema")
    if payload.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-2 cache reports frozen-test access")
    if mode not in MODES:
        raise ValueError(f"invalid Stage-2 mode {mode!r}")
    audit = payload["modes"][mode]
    if audit.get("fold_id") != 0 or audit.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-2 cache is not P4 fold 0 or crossed the test firewall")
    spec = TaskSpec.from_dict(
        json.loads((cache_dir / mode / "task_spec.json").read_text(encoding="utf-8"))
    )
    if spec.metadata.get("evaluation_mode") != mode or spec.task_id != audit["task_id"]:
        raise RuntimeError("Stage-2 TaskSpec/cache mode mismatch")
    if mode == "strict":
        if len(spec.input_whitelist) != 6:
            raise RuntimeError("strict Stage-2 must expose exactly six features")
        if audit["constraint_audit"]["constraints_supplied_to_model"] != 0:
            raise RuntimeError("strict Stage-2 cache contains supplied constraints")
    elif len(spec.input_whitelist) != 7:
        raise RuntimeError("conditional Stage-2 must expose exactly seven features")
    if representation == "point":
        train = [stage1._read_cached_batch(cache_dir / mode / "point_train.npz")]  # noqa: SLF001
        validation = stage1._read_cached_batch(  # noqa: SLF001
            cache_dir / mode / "point_validation.npz"
        )
    elif representation == "volume":
        train = [
            stage1._read_cached_batch(cache_dir / mode / f"volume_train_{index:02d}.npz")  # noqa: SLF001
            for index in range(VOLUME_TRAIN_PATCHES)
        ]
        validation = stage1._read_cached_batch(  # noqa: SLF001
            cache_dir / mode / "volume_validation.npz"
        )
    else:
        raise AdapterSkip(
            "unsupported_stage2_representation",
            f"Stage 2 has no pilot lane for representation {representation!r}",
        )
    for batch in (*train, validation):
        if batch.metadata.get("evaluation_mode") != mode:
            raise RuntimeError("Stage-2 batch crossed mode namespaces")
        if batch.metadata.get("frozen_test_i_blocks_loaded") != []:
            raise RuntimeError("Stage-2 batch reports frozen-test access")
    return spec, train, validation, audit, hash_file(manifest_path)


def _target_prediction(
    model: Any,
    batch: ModelBatch,
    task_spec: TaskSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_name = task_spec.targets[0]
    if batch.targets is None:
        raise ValueError("Stage-2 evaluation batch has no target")
    target = np.asarray(batch.targets[target_name], dtype=np.float64)
    mask = np.asarray(batch.target_masks[target_name], dtype=bool)
    prediction = np.asarray(model.predict(batch).raw[target_name], dtype=np.float64)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Stage-2 prediction/target/mask shapes differ")
    if not np.any(mask) or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Stage-2 prediction contract is empty or non-finite")
    return target[mask], prediction[mask], mask


def _dense_volume(
    values: np.ndarray,
    indices_kji: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    dense = np.full(shape, np.nan, dtype=np.float64)
    dense[tuple(np.asarray(indices_kji, dtype=np.int64).T)] = values
    return dense


def pilot_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    indices_kji: np.ndarray,
    volume_shape_kji: Sequence[int],
) -> dict[str, Any]:
    """P4-equivalent RMSE/MAE/spectral metric without importing h5py."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    indices = np.asarray(indices_kji, dtype=np.int64)
    shape = tuple(int(value) for value in volume_shape_kji)
    if target.shape != prediction.shape or target.ndim != 1 or len(indices) != len(target):
        raise ValueError("Stage-2 metric arrays must describe matching sampled voxels")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Stage-2 metric arrays are non-finite")
    error = prediction - target
    truth = _dense_volume(target, indices, shape)
    pred = _dense_volume(prediction, indices, shape)
    valid = np.isfinite(truth) & np.isfinite(pred)
    truth_fill = np.where(valid, truth, float(np.mean(truth[valid])))
    pred_fill = np.where(valid, pred, float(np.mean(pred[valid])))
    truth_spectrum = np.log1p(np.abs(np.fft.rfftn(truth_fill)))
    pred_spectrum = np.log1p(np.abs(np.fft.rfftn(pred_fill)))
    values = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "spectral_log_rmse": float(
            np.sqrt(np.mean((truth_spectrum - pred_spectrum) ** 2))
        ),
        "valid_voxels": int(target.size),
    }
    if not all(math.isfinite(values[name]) for name in ("rmse", "mae", "spectral_log_rmse")):
        raise FloatingPointError("Stage-2 metric is non-finite")
    return values


@contextlib.contextmanager
def _wall_timeout(seconds: int) -> Iterable[None]:
    def handler(_signum: int, _frame: Any) -> None:
        raise PilotTimeout(f"Stage-2 cell exceeded {seconds} model-wall seconds")

    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def gpu_flock(device: str) -> Iterable[dict[str, Any]]:
    """Hold the protocol's single-GPU flock; wait time is outside model time."""
    if not str(device).startswith("cuda"):
        yield {"required": False, "acquired": False, "wait_seconds": 0.0}
        return
    path = Path(os.environ.get("VOLVE_P5_GPU_LOCK", str(GPU_LOCK_PATH)))
    if os.environ.get("VOLVE_P5_GPU_LOCK_HELD") == "1":
        yield {
            "required": True,
            "acquired": True,
            "mechanism": "external flock -w 900",
            "lock_id": "volve-p5/locks/gpu0.lock",
            "timeout_seconds": 900,
            "wait_seconds": None,
            "wait_excluded_from_model_wall_time": True,
        }
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    wait_started = time.monotonic()
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        wait_seconds = time.monotonic() - wait_started
        try:
            yield {
                "required": True,
                "acquired": True,
                "mechanism": "fcntl.flock(LOCK_EX)",
                "lock_id": "volve-p5/locks/gpu0.lock",
                "timeout_seconds": 900,
                "wait_seconds": wait_seconds,
                "wait_excluded_from_model_wall_time": True,
            }
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _environment(discovered: DiscoveredModel | None) -> dict[str, Any]:
    distributions = ["numpy", "torch", "scipy", "pykrige", "gstools", "gpytorch", "monai", "neuraloperator"]
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "distributions": versions,
        "dependency_group": (
            None if discovered is None else discovered.capabilities.get("dependency_group")
        ),
        "downloads_performed_bytes": 0,
    }


def _reset_cuda_peak(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    selected = torch.device(device)
    # The shared CUDA 13 builds require an initialized context before the
    # peak-memory API accepts a device argument.
    torch.cuda.set_device(selected)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(selected)


def _cuda_peak(device: str) -> int:
    if not device.startswith("cuda"):
        return 0
    import torch

    return int(torch.cuda.max_memory_allocated(torch.device(device)))


def _tiny_gate(
    discovered: DiscoveredModel,
    task_spec: TaskSpec,
    train_batch: ModelBatch,
    validation_batch: ModelBatch,
    config: Mapping[str, Any],
    checkpoint: Path,
) -> dict[str, Any]:
    model = discovered.build(task_spec, **dict(config))
    losses: list[float] = []
    updates = TINY_GATE_UPDATES if discovered.capabilities.get("trainable") else 1
    for _ in range(updates):
        step = dict(model.train_batch(train_batch))
        loss = float(step["loss"])
        if not math.isfinite(loss):
            raise FloatingPointError("tiny gate produced non-finite train loss")
        losses.append(loss)
    _, before_restore, _ = _target_prediction(model, validation_batch, task_spec)
    model.save_checkpoint(checkpoint)
    restored = discovered.build(task_spec, **dict(config))
    restored.load_checkpoint(checkpoint)
    _, after_restore, _ = _target_prediction(restored, validation_batch, task_spec)
    checkpoint_error = float(np.max(np.abs(before_restore - after_restore)))
    tolerance = 1e-5 if str(config.get("device", "cpu")).startswith("cuda") else 1e-8
    if checkpoint_error > tolerance:
        raise AssertionError("tiny gate checkpoint round-trip exceeded tolerance")
    return {
        "status": "passed",
        "updates": updates,
        "finite_shape_passed": True,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "checkpoint_roundtrip_max_abs_error": checkpoint_error,
        "checkpoint_bytes": checkpoint.stat().st_size,
    }


def _portable_source(model_id: str, source_lock: Mapping[str, Any]) -> dict[str, Any]:
    source = source_lock["models"][model_id]
    return {
        key: source[key]
        for key in ("upstream_url", "revision", "distribution", "version_constraint", "license")
    }


def run_cell(
    *,
    mode: str,
    model_id: str,
    cache_dir: Path,
    cell_root: Path,
    device: str,
    root_seed: int = ROOT_SEED,
) -> dict[str, Any]:
    """Run one preregistered development cell; no raw-data argument exists."""
    if root_seed != ROOT_SEED:
        raise ValueError("Stage-2 reconstruction root_seed is frozen at 2693")
    if mode not in MODES or model_id not in CANDIDATES:
        raise ValueError("unknown Stage-2 mode/model cell")
    source_lock = stage1.load_source_lock()
    task_payload = json.loads((cache_dir / mode / "task_spec.json").read_text(encoding="utf-8"))
    task_spec = TaskSpec.from_dict(task_payload)
    cache_manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    mode_audit = cache_manifest["modes"][mode]
    model_seed = SeedTree(root_seed).seed("model", "stage2", mode, model_id)
    seed_tree = {
        "root": root_seed,
        "model": model_seed,
        "loader": SeedTree(root_seed).seed("loader", "stage2", mode, model_id),
        "sampler": SeedTree(root_seed).seed("sampler", "stage2", mode, model_id),
    }
    status_path = cell_root / mode / model_id / "status.json"
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "model_id": model_id,
        "task_id": task_spec.task_id,
        "lane": mode,
        "evaluation_mode": mode,
        "source": _portable_source(model_id, source_lock),
        "seed": seed_tree,
        "split_hash": mode_audit["split_hash"],
        "fold_id": mode_audit["fold_id"],
        "input_budget": mode_audit["input_budget"],
        "test_firewall": {
            "development_only": True,
            "frozen_test_i_blocks_loaded": [],
            "test_loader_argument_exists": False,
            "test_path_argument_exists": False,
            "test_metrics_computed": False,
        },
        "mode_isolation": {
            "input_whitelist": list(task_spec.input_whitelist),
            "constraint_audit": mode_audit["constraint_audit"],
            "strict_constraints_supplied": (
                mode_audit["constraint_audit"]["constraints_supplied_to_model"]
                if mode == "strict"
                else None
            ),
        },
    }
    if model_id in PRESERVED_STAGE1_SKIPS:
        base.update(
            {
                "status": "skipped",
                "evidence_status": "scouted",
                "reason": {
                    **PRESERVED_STAGE1_SKIPS[model_id],
                    "gate": "preserved_stage1_gate",
                },
                "budget": None,
                "updates": 0,
                "wall_seconds": 0.0,
                "metrics": None,
                "resources": {"peak_rss_kib": 0, "peak_cuda_bytes": 0},
                "environment": _environment(None),
            }
        )
        base["result_hash"] = hash_payload(base)
        atomic_write_json(status_path, base)
        return base

    discovered: DiscoveredModel | None = None
    try:
        discovered = discover_model("reconstruction", model_id)
        representation = str(discovered.capabilities["batch_representation"])
        if bool(discovered.capabilities.get("trainable")) and not device.startswith("cuda"):
            raise AdapterSkip(
                "gpu_required_by_stage2_protocol",
                "P5 Stage 2 requires every neural/operator candidate to run on the locked single GPU lane",
                model_id=model_id,
                requested_device=device,
            )
        task_spec, train_batches, validation_batch, mode_audit, cache_hash = load_mode_cache(
            cache_dir, mode, representation
        )
        budget = budget_for(discovered.capabilities)
        config = stage1.model_config(
            model_id,
            task_spec,
            train_batches[0],
            device=device,
            seed=model_seed,
        )
        base.update(
            {
                "representation": representation,
                "budget": budget,
                "cache_contract_hash": cache_hash,
                "model_config": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in config.items()
                },
                "environment": _environment(discovered),
            }
        )
        with gpu_flock(device) as lock_audit:
            seed_report = seed_everything(model_seed, strict=True, include_torch=True).to_dict()
            determinism_audit = {
                "strict_requested": True,
                "warn_only": False,
                "warning": None,
            }
            if model_id == "monai_basicunet3d":
                # CUDA max_pool3d backward has no deterministic implementation
                # in the frozen Torch build.  P4's seed SOP permits an explicit
                # warning when strict execution is unavailable; CPU scores are
                # forbidden from the neural leaderboard.
                import torch

                torch.use_deterministic_algorithms(True, warn_only=True)
                determinism_audit = {
                    "strict_requested": True,
                    "warn_only": True,
                    "warning": (
                        "Torch CUDA max_pool3d backward lacks a deterministic implementation; "
                        "the fixed-seed GPU pilot ran with deterministic warn-only semantics"
                    ),
                }
            _reset_cuda_peak(device)
            started = time.monotonic()
            with _wall_timeout(int(budget["max_wall_seconds"])):
                gate_checkpoint = status_path.parent / "tiny_gate.ckpt"
                tiny_gate = _tiny_gate(
                    discovered,
                    task_spec,
                    train_batches[0],
                    validation_batch,
                    config,
                    gate_checkpoint,
                )
                seed_everything(model_seed, strict=True, include_torch=True)
                if model_id == "monai_basicunet3d":
                    torch.use_deterministic_algorithms(True, warn_only=True)
                model = discovered.build(task_spec, **config)
                losses: list[float] = []
                for update in range(int(budget["max_updates"])):
                    step = dict(model.train_batch(train_batches[update % len(train_batches)]))
                    loss = float(step["loss"])
                    if not math.isfinite(loss):
                        raise FloatingPointError("pilot train loss is non-finite")
                    losses.append(loss)
                target, prediction, _ = _target_prediction(model, validation_batch, task_spec)
                checkpoint = status_path.parent / "pilot_last.ckpt"
                model.save_checkpoint(checkpoint)
                restored = discovered.build(task_spec, **config)
                restored.load_checkpoint(checkpoint)
                _, restored_prediction, _ = _target_prediction(
                    restored, validation_batch, task_spec
                )
                checkpoint_error = float(np.max(np.abs(restored_prediction - prediction)))
                tolerance = 1e-5 if device.startswith("cuda") else 1e-8
                if checkpoint_error > tolerance:
                    raise AssertionError("pilot checkpoint round-trip exceeded tolerance")
                indices = np.asarray(
                    validation_batch.coordinates["metric_indices_kji"], dtype=np.int64
                )
                volume_shape = np.asarray(
                    validation_batch.coordinates["volume_shape_kji"], dtype=np.int64
                )
                metrics = pilot_metrics(target, prediction, indices, volume_shape)
            wall_seconds = time.monotonic() - started
            validate_budget(budget, len(losses), wall_seconds)
            base.update(
                {
                    "status": "passed",
                    "evidence_status": "development_piloted",
                    "reason": None,
                    "updates": len(losses),
                    "wall_seconds": wall_seconds,
                    "train_loss": {
                        "first": losses[0],
                        "last": losses[-1],
                        "minimum": min(losses),
                    },
                    "metrics": metrics,
                    "tiny_gate": tiny_gate,
                    "checkpoint": {
                        "sha256": hash_file(checkpoint),
                        "bytes": checkpoint.stat().st_size,
                        "roundtrip_max_abs_error": checkpoint_error,
                    },
                    "resources": {
                        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        "peak_cuda_bytes": _cuda_peak(device),
                        "gpu_lock": lock_audit,
                    },
                    "seed_report": seed_report,
                    "determinism_audit": determinism_audit,
                }
            )
    except PilotTimeout as exc:
        base.update(
            {
                "status": "timeout",
                "evidence_status": "contract_smoked",
                "reason": {"code": "budget_timeout", "message": str(exc)},
                "updates": 0,
                "wall_seconds": float(base.get("budget", {}).get("max_wall_seconds", 0)),
                "metrics": None,
                "resources": {
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "peak_cuda_bytes": _cuda_peak(device),
                },
            }
        )
    except AdapterSkip as exc:
        base.update(
            {
                "status": "skipped",
                "evidence_status": "contract_smoked",
                "reason": exc.to_dict(),
                "updates": 0,
                "wall_seconds": 0.0,
                "metrics": None,
                "resources": {
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "peak_cuda_bytes": 0,
                },
                "environment": _environment(discovered),
            }
        )
    except Exception as exc:
        message = str(exc)
        gpu_skip_code = None
        if device.startswith("cuda") and "out of memory" in message.lower():
            gpu_skip_code = "gpu_out_of_memory"
        elif device.startswith("cuda") and "deterministic implementation" in message.lower():
            gpu_skip_code = "cuda_determinism_unavailable"
        base.update(
            {
                "status": "skipped" if gpu_skip_code else "failed",
                "evidence_status": "contract_smoked",
                "reason": {
                    "code": gpu_skip_code or "pilot_exception",
                    "exception_type": type(exc).__name__,
                    "message": message,
                },
                "updates": 0,
                "wall_seconds": 0.0,
                "metrics": None,
                "resources": {
                    "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "peak_cuda_bytes": 0,
                },
                "environment": _environment(discovered),
            }
        )
    base["result_hash"] = hash_payload(base)
    atomic_write_json(status_path, base)
    return base


def validate_cell_record(record: Mapping[str, Any], mode: str, model_id: str) -> None:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Stage-2 cell schema mismatch")
    expected_hash = hash_payload(
        {key: value for key, value in record.items() if key != "result_hash"}
    )
    if record.get("result_hash") != expected_hash:
        raise ValueError("Stage-2 cell result_hash mismatch")
    if record.get("lane") != mode or record.get("evaluation_mode") != mode:
        raise ValueError("Stage-2 cell is in the wrong mode lane")
    if record.get("model_id") != model_id or record.get("fold_id") != 0:
        raise ValueError("Stage-2 cell model/fold mismatch")
    unresolved_code = (record.get("reason") or {}).get("code")
    if unresolved_code in {"pilot_exception", "gpu_required_by_stage2_protocol"}:
        raise ValueError(f"unresolved {unresolved_code} cannot enter Stage-2 collation")
    firewall = record.get("test_firewall", {})
    if firewall.get("frozen_test_i_blocks_loaded") != []:
        raise ValueError("Stage-2 cell reports frozen-test access")
    if any(
        firewall.get(key) is not False
        for key in ("test_loader_argument_exists", "test_path_argument_exists", "test_metrics_computed")
    ):
        raise ValueError("Stage-2 cell exposes or consumed a frozen-test surface")
    if mode == "strict":
        isolation = record.get("mode_isolation", {})
        if len(isolation.get("input_whitelist", [])) != 6:
            raise ValueError("strict cell does not have six whitelisted features")
        if isolation.get("strict_constraints_supplied") != 0:
            raise ValueError("strict cell received conditional constraints")
    if record.get("status") == "passed":
        metrics = record.get("metrics", {})
        for name in ("rmse", "mae", "spectral_log_rmse"):
            if not math.isfinite(float(metrics[name])):
                raise ValueError(f"Stage-2 cell has invalid {name}")
        validate_budget(record["budget"], int(record["updates"]), float(record["wall_seconds"]))
        if model_id in GPU_CANDIDATES:
            if not str(record.get("model_config", {}).get("device", "")).startswith("cuda"):
                raise ValueError("neural/operator Stage-2 result did not run on CUDA")
            lock = record.get("resources", {}).get("gpu_lock", {})
            if lock.get("required") is not True or lock.get("acquired") is not True:
                raise ValueError("neural/operator Stage-2 result lacks acquired GPU flock evidence")
            if lock.get("timeout_seconds") != 900:
                raise ValueError("neural/operator Stage-2 flock timeout differs from 900 seconds")
            if lock.get("mechanism") != "external flock -w 900":
                raise ValueError("neural/operator Stage-2 result lacks external flock -w 900 evidence")


def _atomic_write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _leaderboard(mode: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["lane"] == mode and record["status"] == "passed"]
    ordered = sorted(
        valid,
        key=lambda item: (
            float(item["metrics"]["rmse"]),
            float(item["metrics"]["mae"]),
            float(item["metrics"]["spectral_log_rmse"]),
            str(item["model_id"]),
        ),
    )
    return {
        "schema_version": "p5-stage2-reconstruction-leaderboard-v1",
        "track_id": "reconstruction",
        "lane": mode,
        "task_id": p4.protocol(mode).task_id,
        "ranking_metric": "rmse",
        "metric_direction": "minimize",
        "not_frozen_test": True,
        "rankable": len(ordered) >= 2,
        "not_rankable_reason": None if len(ordered) >= 2 else "fewer than two valid pilot results",
        "entries": [
            {
                "rank": rank,
                "model_id": record["model_id"],
                "rmse": record["metrics"]["rmse"],
                "mae": record["metrics"]["mae"],
                "spectral_log_rmse": record["metrics"]["spectral_log_rmse"],
                "updates": record["updates"],
                "wall_seconds": record["wall_seconds"],
                "peak_rss_kib": record["resources"]["peak_rss_kib"],
                "peak_cuda_bytes": record["resources"]["peak_cuda_bytes"],
                "result_hash": record["result_hash"],
            }
            for rank, record in enumerate(ordered, start=1)
        ],
    }


def collate(cell_root: Path, output_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for mode in MODES:
        for model_id in CANDIDATES:
            path = cell_root / mode / model_id / "status.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing preregistered Stage-2 cell: {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_cell_record(record, mode, model_id)
            records.append(record)
    results_path = _atomic_write_jsonl(output_dir / "p5_stage2_results.jsonl", records)
    leaderboard_paths: dict[str, Path] = {}
    for mode in MODES:
        path = output_dir / f"p5_stage2_leaderboard_{mode}.json"
        atomic_write_json(path, _leaderboard(mode, records))
        leaderboard_paths[mode] = path
    counts = {name: 0 for name in ("passed", "skipped", "failed", "timeout")}
    for record in records:
        counts[record["status"]] += 1
    source_lock_path = PROJECT_ROOT / "_models" / "reconstruction" / "source_lock.json"
    summary = {
        "schema_version": "p5-stage2-reconstruction-summary-v1",
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "expected_cells": len(MODES) * len(CANDIDATES),
        "attempted_cells": len(records),
        "counts": counts,
        "development_only": True,
        "frozen_test_i_blocks_loaded": [],
        "modes_are_independent": True,
        "source_lock": "_models/reconstruction/source_lock.json",
        "source_lock_sha256": hash_file(source_lock_path),
        "results": "p5_stage2_results.jsonl",
        "results_sha256": hash_file(results_path),
        "leaderboards": {
            mode: {
                "path": path.name,
                "sha256": hash_file(path),
            }
            for mode, path in leaderboard_paths.items()
        },
        "split_hashes": {
            mode: next(record["split_hash"] for record in records if record["lane"] == mode)
            for mode in MODES
        },
        "stage3_selection_performed": False,
        "scientific_note": (
            "Fixed-fold development pilot only; scores are not CV confirmation or frozen-test evidence."
        ),
    }
    atomic_write_json(output_dir / "p5_stage2_summary.json", summary)
    return summary


def _parse_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return CANDIDATES
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected or any(item not in CANDIDATES for item in selected):
        raise argparse.ArgumentTypeError("--models must be all or registered candidate IDs")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-cache", help="freeze fold-0 development batches")
    prepare.add_argument("--data-dir", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)

    cell = subparsers.add_parser("run-cell", help="run one development-only model/mode cell")
    cell.add_argument("--mode", choices=MODES, required=True)
    cell.add_argument("--model", choices=CANDIDATES, required=True)
    cell.add_argument("--cache-dir", type=Path, required=True)
    cell.add_argument("--cell-root", type=Path, required=True)
    cell.add_argument("--device", default="cpu")
    cell.add_argument("--root-seed", type=int, default=ROOT_SEED)

    aggregate = subparsers.add_parser("collate", help="write portable results and mode leaderboards")
    aggregate.add_argument("--cell-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, default=HERE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-cache":
        report = prepare_cache(args.data_dir, args.cache_dir)
        print(
            json.dumps(
                {
                    "cache_manifest": str(args.cache_dir / "cache_manifest.json"),
                    "modes": list(report["modes"]),
                    "frozen_test_i_blocks_loaded": report["frozen_test_i_blocks_loaded"],
                },
                indent=2,
            )
        )
    elif args.command == "run-cell":
        record = run_cell(
            mode=args.mode,
            model_id=args.model,
            cache_dir=args.cache_dir,
            cell_root=args.cell_root,
            device=args.device,
            root_seed=args.root_seed,
        )
        print(
            json.dumps(
                {
                    "model_id": record["model_id"],
                    "lane": record["lane"],
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
