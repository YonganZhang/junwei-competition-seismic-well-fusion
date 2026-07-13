#!/usr/bin/env python3
"""P5 Stage-1 contract smoke for the first ten reconstruction candidates.

The runner has no test command and accepts no test loader.  It loads only the
mode-specific P4 development I-blocks, constructs one buffered development
fold, and archives strict/conditional evidence under disjoint directories.
The historical files named ``train.h5``/``test.h5`` are storage containers;
the scientific firewall is the frozen I-block contract, which is asserted
before any selected group array is returned by ``load_patch_records``.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import atomic_write_json, hash_file  # noqa: E402
from ml_framework.contracts import ModelBatch, TaskSpec  # noqa: E402
from ml_framework.model_discovery import DiscoveredModel, discover_model  # noqa: E402
from ml_framework.preprocess import NormStats, normalize  # noqa: E402
from ml_framework.seeding import DEFAULT_ROOT_SEED, SeedTree, seed_everything  # noqa: E402
from ml_framework.splits import Fold  # noqa: E402

from _models.reconstruction._p5_adapter import AdapterSkip  # noqa: E402


MODES = ("conditional", "strict")
STAGE1_MODELS = (
    "scipy_rbf_neighbors",
    "pykrige_ok3d",
    "gstools_krige_condsrf",
    "mpslib_snesim3d",
    "gpytorch_svgp",
    "monai_basicunet3d",
    "monai_segresnet3d",
    "neuralop_fno3d",
    "tcnn_hashgrid_inr",
    "siren_inr",
)
SOURCE_LOCK = PROJECT_ROOT / "_models" / "reconstruction" / "source_lock.json"


class _LazyP4:
    """Avoid importing h5py-bound P4 data code in dependency-only workers."""

    module: Any | None = None

    def __getattr__(self, name: str) -> Any:
        if self.module is None:
            self.module = importlib.import_module("p4_reconstruction")
        return getattr(self.module, name)


p4 = _LazyP4()


@dataclass(frozen=True)
class DevelopmentBundle:
    mode: str
    fold: Fold
    prepared: p4.PreparedFold
    train_records: tuple[p4.PatchRecord, ...]
    validation_records: tuple[p4.PatchRecord, ...]


def load_source_lock(path: Path = SOURCE_LOCK) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("lock_version") != "p5-source-lock-v1" or payload.get("track_id") != "reconstruction":
        raise ValueError("invalid reconstruction P5 source lock")
    models = payload.get("models")
    if not isinstance(models, dict) or tuple(models) != STAGE1_MODELS:
        raise ValueError("source lock model order/content differs from frozen Stage-1 list")
    for model_name, record in models.items():
        revision = str(record.get("revision", ""))
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError(f"{model_name} source lock lacks an exact 40-character commit")
        if not record.get("upstream_url") or not record.get("license"):
            raise ValueError(f"{model_name} source lock lacks URL/license")
        weights = record.get("weights", {})
        if weights.get("required") is not False or weights.get("sha256") is not None:
            raise ValueError(f"{model_name} violates scratch/no-weight Stage-1 policy")
    return payload


def _runtime_environment(lock: Mapping[str, Any]) -> dict[str, Any]:
    distributions = sorted(
        {
            str(record["distribution"])
            for record in lock["models"].values()
            if str(record["distribution"]) not in {"local-thin-adapter-on-torch"}
        }
        | {"numpy", "torch"}
    )
    versions: dict[str, str | None] = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    torch_environment: dict[str, Any] = {"available": False}
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch_environment = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": (
                list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
            ),
        }
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "distributions": versions,
        "torch": torch_environment,
        "downloads_performed_bytes": 0,
    }


def load_development_records(mode: str, data_dir: Path) -> list[p4.PatchRecord]:
    """Load only the selected mode's development blocks and fail closed."""
    active = p4.protocol(mode)
    records = p4.load_patch_records(active.development_i_blocks, data_dir)
    loaded_blocks = sorted({record.i_block for record in records})
    forbidden_blocks = sorted(set(loaded_blocks) & set(active.test_i_blocks))
    guard_blocks = sorted(set(loaded_blocks) & set(active.guard_i_blocks))
    if forbidden_blocks or guard_blocks or loaded_blocks != sorted(active.development_i_blocks):
        raise RuntimeError(
            f"{mode} development loader crossed firewall: loaded={loaded_blocks}, "
            f"frozen_test={forbidden_blocks}, guard={guard_blocks}"
        )
    return records


def _development_fold(mode: str, records: Sequence[p4.PatchRecord]) -> DevelopmentBundle:
    if not records:
        raise ValueError("Stage-1 requires development records")
    groups = sorted({record.k_block for record in records})
    if len(groups) < 4:
        raise ValueError("Stage-1 buffered fold requires at least four development K-blocks")
    validation_k = groups[len(groups) // 2]
    purged_k = {value for value in groups if value != validation_k and abs(value - validation_k) <= 1}
    validation_ids = tuple(record.sample_id for record in records if record.k_block == validation_k)
    candidate_ids = tuple(record.sample_id for record in records if record.k_block != validation_k)
    purged_ids = tuple(record.sample_id for record in records if record.k_block in purged_k)
    effective_ids = tuple(sample_id for sample_id in candidate_ids if sample_id not in set(purged_ids))
    if not effective_ids or not validation_ids:
        raise ValueError("Stage-1 fold has empty effective train or validation set")
    fold = Fold(
        fold_id=0,
        train_groups=tuple(f"dev_k{value:02d}" for value in groups if value != validation_k),
        validation_groups=(f"dev_k{validation_k:02d}",),
        train_sample_ids=candidate_ids,
        validation_sample_ids=validation_ids,
        purge={
            "axis": "k_block", "buffer_blocks": 1, "purged_k_blocks": sorted(purged_k),
            "purged_train_sample_ids": list(purged_ids),
            "effective_train_sample_ids": list(effective_ids),
        },
        support={
            "candidate_train_patches": len(candidate_ids),
            "purged_train_patches": len(purged_ids),
            "effective_train_patches": len(effective_ids),
            "validation_patches": len(validation_ids),
        },
    )
    prepared = p4.prepare_fold(mode, fold, records)
    by_id = {record.sample_id: record for record in records}
    train_records = tuple(by_id[sample_id] for sample_id in effective_ids)
    validation_records = tuple(by_id[sample_id] for sample_id in validation_ids)
    if mode == "strict" and prepared.constraint_audit["constraints_supplied_to_model"] != 0:
        raise RuntimeError("strict Stage-1 fold supplied target-derived well constraints")
    return DevelopmentBundle(mode, fold, prepared, train_records, validation_records)


def synthetic_development_bundle(mode: str) -> DevelopmentBundle:
    _, records = p4.synthetic_catalog_and_records()
    allowed = set(p4.protocol(mode).development_i_blocks)
    return _development_fold(mode, [record for record in records if record.i_block in allowed])


def real_development_bundle(mode: str, data_dir: Path) -> DevelopmentBundle:
    return _development_fold(mode, load_development_records(mode, data_dir))


def _sample_indices(size: int, maximum: int) -> np.ndarray:
    count = min(size, maximum)
    if count <= 0:
        raise ValueError("cannot sample an empty Stage-1 array")
    return np.linspace(0, size - 1, num=count, dtype=np.int64)


def _point_batches(
    task_spec: TaskSpec,
    bundle: DevelopmentBundle,
    *,
    max_train_points: int,
    max_validation_points: int,
) -> tuple[ModelBatch, ModelBatch]:
    prepared = bundle.prepared
    train_index = _sample_indices(prepared.train_target.size, max_train_points)
    validation_valid = np.flatnonzero(prepared.validation_metric_mask)
    validation_index = validation_valid[
        _sample_indices(validation_valid.size, max_validation_points)
    ]

    def make(split: str, features: np.ndarray, target: np.ndarray, index: np.ndarray) -> ModelBatch:
        target_name = task_spec.targets[0]
        selected = np.asarray(features[index], dtype=np.float64)
        values = np.asarray(target[index], dtype=np.float64)
        return ModelBatch(
            inputs={"features": selected},
            targets={target_name: values},
            input_masks={},
            target_masks={target_name: np.ones(values.shape, dtype=bool)},
            sample_ids=[f"{bundle.mode}:{split}:points"],
            groups={"i_block": [f"development_{bundle.mode}"], "k_block": [f"fold_{bundle.fold.fold_id}"]},
            coordinates={"xyz": selected[:, -3:]},
            metadata={
                "evaluation_mode": bundle.mode,
                "split": f"development_{split}",
                "feature_names": list(prepared.feature_names),
                "frozen_test_i_blocks_loaded": [],
            },
        )

    return (
        make("train", prepared.train_features, prepared.train_target, train_index),
        make("validation", prepared.validation_features, prepared.validation_target, validation_index),
    )


def _record_feature_volume(
    task_spec: TaskSpec,
    bundle: DevelopmentBundle,
    record: p4.PatchRecord,
    *,
    validation: bool,
) -> ModelBatch:
    prepared = bundle.prepared
    patch = np.asarray(record.seismic_patch, dtype=np.float64)
    constraints = p4.constraints_from_records(bundle.train_records)
    fallback = float(np.mean(prepared.train_target))
    if bundle.mode == "conditional":
        coordinate_rows = patch[3:6].reshape(3, -1).T
        idw = p4.idw_predict(coordinate_rows, constraints, fallback=fallback).reshape(patch.shape[1:])
        raw = np.concatenate([idw[None, ...], patch[0:6]], axis=0)
    else:
        if constraints.size and p4.protocol("strict").idw_feature_name is not None:
            raise RuntimeError("strict volume adapter unexpectedly requested constraints")
        raw = patch[0:6].copy()
    if raw.shape[0] != len(task_spec.input_whitelist):
        raise ValueError("volume feature count differs from TaskSpec whitelist")
    stats = [NormStats.from_dict(dict(item)) for item in prepared.preprocess_report["stats"]]
    normalized = np.stack(
        [normalize(raw[index], stats[index]) for index in range(raw.shape[0])]
    ).astype(np.float32)
    active = patch[8] > 0.5
    normalized[:, ~active] = 0.0
    metric_mask = active & (~(patch[7] > 0.5) if validation else True)
    target_name = task_spec.targets[0]
    return ModelBatch(
        inputs={"volume": normalized[None, ...]},
        targets={target_name: np.asarray(record.label, dtype=np.float32)[None, None, ...]},
        input_masks={"active": active[None, None, ...]},
        target_masks={target_name: metric_mask[None, None, ...]},
        sample_ids=[record.sample_id],
        groups={"i_block": [str(record.i_block)], "k_block": [str(record.k_block)]},
        coordinates={"patch_start_kji": np.asarray(record.location.patch_start_kji)[None, :]},
        metadata={
            "evaluation_mode": bundle.mode,
            "split": "development_validation" if validation else "development_train",
            "feature_names": list(prepared.feature_names),
            "constraint_count_supplied": int(constraints.shape[0]) if bundle.mode == "conditional" else 0,
            "frozen_test_i_blocks_loaded": [],
        },
    )


def _volume_batches(task_spec: TaskSpec, bundle: DevelopmentBundle) -> tuple[ModelBatch, ModelBatch]:
    train = max(bundle.train_records, key=lambda record: int(np.sum(record.seismic_patch[8] > 0.5)))
    validation = max(
        bundle.validation_records,
        key=lambda record: int(np.sum((record.seismic_patch[8] > 0.5) & ~(record.seismic_patch[7] > 0.5))),
    )
    return (
        _record_feature_volume(task_spec, bundle, train, validation=False),
        _record_feature_volume(task_spec, bundle, validation, validation=True),
    )


def _write_cached_batch(path: Path, batch: ModelBatch, task_spec: TaskSpec) -> None:
    if len(batch.inputs) != 1:
        raise ValueError("Stage-1 cache supports one canonical input tensor per representation")
    input_name, input_values = next(iter(batch.inputs.items()))
    target_name = task_spec.targets[0]
    if batch.targets is None:
        raise ValueError("cannot cache target-free Stage-1 batch")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            input_name=np.asarray(input_name),
            input_values=np.asarray(input_values),
            target_name=np.asarray(target_name),
            target_values=np.asarray(batch.targets[target_name]),
            target_mask=np.asarray(batch.target_masks[target_name], dtype=bool),
        )
    atomic_write_json(
        path.with_suffix(".json"),
        {
            "sample_ids": list(batch.sample_ids),
            "groups": {key: list(values) for key, values in batch.groups.items()},
            "metadata": dict(batch.metadata),
            "coordinates": {
                key: np.asarray(values).tolist() for key, values in batch.coordinates.items()
            },
            "input_masks": {
                key: np.asarray(values).tolist() for key, values in batch.input_masks.items()
            },
            "npz_sha256": hash_file(path),
        },
    )


def _read_cached_batch(path: Path) -> ModelBatch:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if hash_file(path) != metadata["npz_sha256"]:
        raise RuntimeError(f"Stage-1 cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        input_name = str(payload["input_name"])
        target_name = str(payload["target_name"])
        input_values = np.asarray(payload["input_values"])
        target_values = np.asarray(payload["target_values"])
        target_mask = np.asarray(payload["target_mask"], dtype=bool)
    return ModelBatch(
        inputs={input_name: input_values},
        targets={target_name: target_values},
        input_masks={key: np.asarray(value) for key, value in metadata["input_masks"].items()},
        target_masks={target_name: target_mask},
        sample_ids=metadata["sample_ids"],
        groups=metadata["groups"],
        coordinates={key: np.asarray(value) for key, value in metadata["coordinates"].items()},
        metadata=metadata["metadata"],
    )


def prepare_batch_cache(
    *,
    data_dir: Path,
    cache_dir: Path,
    max_train_points: int = 96,
    max_validation_points: int = 48,
) -> dict[str, Any]:
    """Materialize small development-only batches for dependency-isolated workers."""
    report: dict[str, Any] = {
        "schema_version": "p5-stage1-batch-cache-v1",
        "track_id": "reconstruction",
        "source_data_dir": str(data_dir.resolve()),
        "frozen_test_i_blocks_loaded": [],
        "modes": {},
    }
    for mode in MODES:
        spec = p4.task_spec(mode)
        atomic_write_json(cache_dir / mode / "task_spec.json", spec.to_dict())
        real = real_development_bundle(mode, data_dir)
        synthetic = synthetic_development_bundle(mode)
        mode_report = {
            "task_id": spec.task_id,
            "development_i_blocks": list(p4.protocol(mode).development_i_blocks),
            "guard_i_blocks": list(p4.protocol(mode).guard_i_blocks),
            "frozen_test_i_blocks": list(p4.protocol(mode).test_i_blocks),
            "frozen_test_i_blocks_loaded": [],
            "feature_names": list(real.prepared.feature_names),
            "constraint_audit": real.prepared.constraint_audit,
            "fit_scope": real.prepared.preprocess_report["fit_scope"],
            "batches": {},
        }
        for kind, bundle in (("synthetic", synthetic), ("real_development", real)):
            for representation in ("point", "volume"):
                train, validation = make_batches(
                    spec, bundle, representation,
                    max_train_points=max_train_points,
                    max_validation_points=max_validation_points,
                )
                for split, batch in (("train", train), ("validation", validation)):
                    relative = Path(mode) / f"{kind}_{representation}_{split}.npz"
                    _write_cached_batch(cache_dir / relative, batch, spec)
                    mode_report["batches"][relative.name] = {
                        "sha256": hash_file(cache_dir / relative),
                        "bytes": (cache_dir / relative).stat().st_size,
                    }
        if mode == "strict" and mode_report["constraint_audit"]["constraints_supplied_to_model"] != 0:
            raise RuntimeError("strict cache preparation supplied forbidden constraints")
        report["modes"][mode] = mode_report
    atomic_write_json(cache_dir / "cache_manifest.json", report)
    return report


def load_cached_contract(
    cache_dir: Path,
    mode: str,
    representation: str,
) -> tuple[TaskSpec, tuple[ModelBatch, ModelBatch], tuple[ModelBatch, ModelBatch], dict[str, Any]]:
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "p5-stage1-batch-cache-v1":
        raise ValueError("invalid Stage-1 batch cache manifest")
    if manifest.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError("Stage-1 cache claims frozen-test access")
    mode_audit = manifest["modes"][mode]
    if mode_audit.get("frozen_test_i_blocks_loaded") != []:
        raise RuntimeError(f"{mode} Stage-1 cache crossed frozen-test firewall")
    spec = TaskSpec.from_dict(
        json.loads((cache_dir / mode / "task_spec.json").read_text(encoding="utf-8"))
    )

    def pair(kind: str) -> tuple[ModelBatch, ModelBatch]:
        return (
            _read_cached_batch(cache_dir / mode / f"{kind}_{representation}_train.npz"),
            _read_cached_batch(cache_dir / mode / f"{kind}_{representation}_validation.npz"),
        )

    return spec, pair("synthetic"), pair("real_development"), mode_audit


def make_batches(
    task_spec: TaskSpec,
    bundle: DevelopmentBundle,
    representation: str,
    *,
    max_train_points: int,
    max_validation_points: int,
) -> tuple[ModelBatch, ModelBatch]:
    if representation == "point":
        return _point_batches(
            task_spec, bundle,
            max_train_points=max_train_points,
            max_validation_points=max_validation_points,
        )
    if representation == "volume":
        return _volume_batches(task_spec, bundle)
    raise AdapterSkip(
        "unsupported_stage1_representation",
        f"Stage-1 has no approved batch adapter for representation {representation!r}",
        representation=representation,
    )


def model_config(
    model_name: str,
    task_spec: TaskSpec,
    train_batch: ModelBatch,
    *,
    device: str,
    seed: int,
) -> dict[str, Any]:
    input_name = "features" if "features" in train_batch.inputs else "volume"
    inputs = np.asarray(train_batch.inputs[input_name])
    n_features = int(inputs.shape[1])
    target_name = task_spec.targets[0]
    n_training_samples = int(np.asarray(train_batch.target_masks[target_name], dtype=bool).sum())
    common = {
        "n_features": n_features,
        "n_training_samples": n_training_samples,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "seed": seed,
        "device": device,
    }
    specific: dict[str, dict[str, Any]] = {
        # Local development samples can be coplanar (one J slice).  The
        # linear RBF has degree-0 polynomial requirements and therefore does
        # not invent a full-rank 3-D affine system for that legitimate case.
        "scipy_rbf_neighbors": {"neighbors": 16, "kernel": "linear", "smoothing": 0.0},
        "pykrige_ok3d": {"variogram_model": "linear", "nlags": 4},
        "gstools_krige_condsrf": {"len_scale": 0.25, "covariance": "Exponential"},
        "mpslib_snesim3d": {"training_image_provenance_approved": False},
        "gpytorch_svgp": {"num_inducing": 16, "learning_rate": 0.01},
        "monai_basicunet3d": {"base_channels": 4},
        "monai_segresnet3d": {"init_filters": 4},
        "neuralop_fno3d": {"n_modes": (2, 4, 4), "hidden_channels": 8, "n_layers": 2},
        "tcnn_hashgrid_inr": {"n_levels": 4, "hidden_features": 32},
        "siren_inr": {"hidden_features": 32, "hidden_layers": 2, "omega_0": 30.0, "learning_rate": 1e-4},
    }
    return {**common, **specific[model_name]}


def _target_array(batch: ModelBatch, task_spec: TaskSpec) -> np.ndarray:
    if batch.targets is None:
        raise ValueError("Stage-1 validation batch has no target")
    return np.asarray(batch.targets[task_spec.targets[0]])


def _prediction_array(model: Any, batch: ModelBatch, task_spec: TaskSpec) -> np.ndarray:
    prediction = np.asarray(model.predict(batch).raw[task_spec.targets[0]])
    target = _target_array(batch, task_spec)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} != target shape {target.shape}")
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Stage-1 prediction is non-finite")
    return prediction


def _reset_peak_memory(device: str) -> None:
    if device.startswith("cuda") and importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(device))


def _peak_memory_bytes(device: str) -> int:
    if device.startswith("cuda") and importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(torch.device(device)))
    return 0


def exercise_contract(
    discovered: DiscoveredModel,
    task_spec: TaskSpec,
    train_batch: ModelBatch,
    validation_batch: ModelBatch,
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    root_seed: int,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reset_peak_memory(device)
    started = time.perf_counter()
    seed_report = seed_everything(root_seed, strict=True, include_torch=True).to_dict()
    model = discovered.build(task_spec, **dict(config))
    initial_loss = None
    if discovered.capabilities.get("trainable"):
        initial_loss = float(model.validation_loss(train_batch))
    step = dict(model.train_batch(train_batch))
    prediction = _prediction_array(model, validation_batch, task_spec)
    validation_loss = float(model.validation_loss(validation_batch))
    if not math.isfinite(float(step["loss"])) or not math.isfinite(validation_loss):
        raise FloatingPointError("Stage-1 loss is non-finite")
    if bool(step.get("backward")) != bool(discovered.capabilities.get("trainable")):
        raise ValueError("adapter backward flag disagrees with declared trainable capability")

    checkpoint = output_dir / "checkpoint_stage1.bin"
    model.save_checkpoint(checkpoint)
    restored = discovered.build(task_spec, **dict(config))
    restored.load_checkpoint(checkpoint)
    restored_prediction = _prediction_array(restored, validation_batch, task_spec)
    checkpoint_error = float(np.max(np.abs(restored_prediction - prediction)))

    seed_everything(root_seed, strict=True, include_torch=True)
    replica = discovered.build(task_spec, **dict(config))
    replica.train_batch(train_batch)
    replica_prediction = _prediction_array(replica, validation_batch, task_spec)
    deterministic_error = float(np.max(np.abs(replica_prediction - prediction)))
    tolerance = 1e-5 if device.startswith("cuda") else 1e-8
    if checkpoint_error > tolerance or deterministic_error > tolerance:
        raise AssertionError(
            f"checkpoint/determinism tolerance failed: {checkpoint_error}, {deterministic_error} > {tolerance}"
        )
    elapsed = time.perf_counter() - started
    return {
        "status": "passed",
        "initial_train_loss": initial_loss,
        "step_loss": float(step["loss"]),
        "validation_loss": validation_loss,
        "valid_count": int(step["valid_count"]),
        "backward_executed": bool(step.get("backward")),
        "prediction_shape": list(prediction.shape),
        "finite_prediction": True,
        "checkpoint_roundtrip_max_abs_error": checkpoint_error,
        "same_seed_max_abs_error": deterministic_error,
        "determinism_tolerance": tolerance,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hash_file(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "wall_seconds": elapsed,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "peak_cuda_bytes": _peak_memory_bytes(device),
        "seed_report": seed_report,
    }


def run_one_model_mode(
    model_name: str,
    mode: str,
    *,
    data_dir: Path | None,
    batch_cache: Path | None,
    output_root: Path,
    root_seed: int,
    device: str,
    max_train_points: int,
    max_validation_points: int,
    source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    if batch_cache is not None:
        task_spec = TaskSpec.from_dict(
            json.loads((batch_cache / mode / "task_spec.json").read_text(encoding="utf-8"))
        )
        mode_audit = json.loads((batch_cache / "cache_manifest.json").read_text(encoding="utf-8"))["modes"][mode]
        development_i_blocks = mode_audit["development_i_blocks"]
        guard_i_blocks = mode_audit["guard_i_blocks"]
        frozen_test_i_blocks = mode_audit["frozen_test_i_blocks"]
    else:
        if data_dir is None:
            raise ValueError("data_dir is required when no Stage-1 batch cache is provided")
        task_spec = p4.task_spec(mode)
        active = p4.protocol(mode)
        development_i_blocks = list(active.development_i_blocks)
        guard_i_blocks = list(active.guard_i_blocks)
        frozen_test_i_blocks = list(active.test_i_blocks)
    status_path = output_root / mode / model_name / "status.json"
    record: dict[str, Any] = {
        "schema_version": "p5-stage1-status-v1",
        "track_id": "reconstruction",
        "model_id": model_name,
        "evaluation_mode": mode,
        "task_id": task_spec.task_id,
        "source": dict(source_lock["models"][model_name]),
        "runtime_environment": _runtime_environment(source_lock),
        "evidence_status": "scouted",
        "firewall": {
            "development_i_blocks": development_i_blocks,
            "guard_i_blocks": guard_i_blocks,
            "frozen_test_i_blocks": frozen_test_i_blocks,
            "frozen_test_i_blocks_loaded": [],
            "test_loader_argument_exists": False,
            "strict_constraints_supplied": 0 if mode == "strict" else None,
        },
    }
    try:
        discovered = discover_model("reconstruction", model_name)
        record["capabilities"] = dict(discovered.capabilities)
        representation = str(discovered.capabilities.get("batch_representation", ""))
        # MPS must trip its legal-training-image gate before any data adapter is selected.
        if model_name == "mpslib_snesim3d":
            discovered.build(
                task_spec,
                n_features=len(task_spec.input_whitelist),
                training_image_provenance_approved=False,
            )
            raise AssertionError("MPS legal training-image gate unexpectedly passed")

        if batch_cache is not None:
            task_spec, synthetic_batches, real_batches, mode_audit = load_cached_contract(
                batch_cache, mode, representation
            )
            if mode == "strict" and mode_audit["constraint_audit"]["constraints_supplied_to_model"] != 0:
                raise RuntimeError("strict cached development bundle contains supplied constraints")
        else:
            assert data_dir is not None
            real_bundle = real_development_bundle(mode, data_dir)
            synthetic_bundle = synthetic_development_bundle(mode)
            if mode == "strict" and real_bundle.prepared.constraint_audit["constraints_supplied_to_model"] != 0:
                raise RuntimeError("strict real development bundle contains supplied constraints")
            synthetic_batches = make_batches(
                task_spec, synthetic_bundle, representation,
                max_train_points=max_train_points, max_validation_points=max_validation_points,
            )
            real_batches = make_batches(
                task_spec, real_bundle, representation,
                max_train_points=max_train_points, max_validation_points=max_validation_points,
            )
        seed = SeedTree(root_seed).seed("model", mode, model_name)
        config = model_config(
            model_name, task_spec, real_batches[0], device=device, seed=seed
        )
        record["model_config"] = {
            key: list(value) if isinstance(value, tuple) else value for key, value in config.items()
        }
        record["synthetic"] = exercise_contract(
            discovered, task_spec, *synthetic_batches,
            config=model_config(model_name, task_spec, synthetic_batches[0], device=device, seed=seed),
            output_dir=status_path.parent / "synthetic", root_seed=seed, device=device,
        )
        record["real_development"] = exercise_contract(
            discovered, task_spec, *real_batches, config=config,
            output_dir=status_path.parent / "real_development", root_seed=seed, device=device,
        )
        record["evidence_status"] = "contract_smoked"
        record["status"] = "passed"
        if batch_cache is not None:
            record["development_audit"] = {
                "batch_cache": str(batch_cache),
                "feature_names": mode_audit["feature_names"],
                "fit_scope": mode_audit["fit_scope"],
                "constraint_audit": mode_audit["constraint_audit"],
                "frozen_test_i_blocks_loaded": [],
            }
        else:
            record["development_audit"] = {
                "fold_id": real_bundle.fold.fold_id,
                "feature_names": list(real_bundle.prepared.feature_names),
                "fit_scope": real_bundle.prepared.preprocess_report["fit_scope"],
                "constraint_audit": real_bundle.prepared.constraint_audit,
                "frozen_test_i_blocks_loaded": [],
            }
    except AdapterSkip as exc:
        record["evidence_status"] = "scouted"
        record["status"] = "skipped"
        record["skip"] = exc.to_dict()
    except Exception as exc:  # fail-loud evidence; caller decides nonzero exit
        record["status"] = "failed"
        record["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
    atomic_write_json(status_path, record)
    return record


def _read_existing_summary(output_root: Path) -> dict[str, Any]:
    path = output_root / "summary.json"
    if not path.is_file():
        return {"schema_version": "p5-stage1-summary-v1", "track_id": "reconstruction", "results": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "p5-stage1-summary-v1" or payload.get("track_id") != "reconstruction":
        raise ValueError("existing Stage-1 summary has a different schema/track")
    return payload


def run_stage1(
    *,
    modes: Iterable[str],
    models: Sequence[str],
    data_dir: Path | None,
    batch_cache: Path | None = None,
    output_root: Path,
    root_seed: int = DEFAULT_ROOT_SEED,
    device: str = "cpu",
    max_train_points: int = 96,
    max_validation_points: int = 48,
) -> dict[str, Any]:
    unknown = sorted(set(models) - set(STAGE1_MODELS))
    if unknown:
        raise ValueError(f"unknown Stage-1 models: {unknown}")
    source_lock = load_source_lock()
    if data_dir is None and batch_cache is None:
        raise ValueError("run_stage1 requires data_dir or a prepared batch_cache")
    summary = _read_existing_summary(output_root)
    summary.update({
        "source_lock": SOURCE_LOCK.relative_to(PROJECT_ROOT).as_posix(),
        "source_lock_sha256": hash_file(SOURCE_LOCK),
        "root_seed": root_seed,
        "runtime_environment": _runtime_environment(source_lock),
        "data_policy": {
            "data_dir": None if data_dir is None else str(data_dir.resolve()),
            "batch_cache": None if batch_cache is None else str(batch_cache.resolve()),
            "development_only": True,
            "frozen_test_i_blocks_loaded": [],
            "weights_downloaded_bytes": 0,
        },
    })
    results = summary.setdefault("results", {})
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unsupported mode {mode!r}")
        mode_results = results.setdefault(mode, {})
        for model_name in models:
            mode_results[model_name] = run_one_model_mode(
                model_name, mode, data_dir=data_dir, output_root=output_root,
                batch_cache=batch_cache,
                root_seed=root_seed, device=device,
                max_train_points=max_train_points,
                max_validation_points=max_validation_points,
                source_lock=source_lock,
            )
            atomic_write_json(output_root / "summary.json", summary)
    counts = {"passed": 0, "skipped": 0, "failed": 0}
    for mode_results in results.values():
        for result in mode_results.values():
            counts[result["status"]] += 1
    summary["counts"] = counts
    atomic_write_json(output_root / "summary.json", summary)
    return summary


def _parse_models(value: str) -> tuple[str, ...]:
    if value == "all":
        return STAGE1_MODELS
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise argparse.ArgumentTypeError("--models must be all or a comma-separated list")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "both"), default="both")
    parser.add_argument("--models", type=_parse_models, default=STAGE1_MODELS)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--batch-cache", type=Path)
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-points", type=int, default=96)
    parser.add_argument("--max-validation-points", type=int, default=48)
    parser.add_argument("--fail-on-failed", action="store_true")
    args = parser.parse_args()
    if args.prepare_cache_only:
        if args.data_dir is None or args.batch_cache is None:
            parser.error("--prepare-cache-only requires --data-dir and --batch-cache")
        report = prepare_batch_cache(
            data_dir=args.data_dir,
            cache_dir=args.batch_cache,
            max_train_points=args.max_train_points,
            max_validation_points=args.max_validation_points,
        )
        print(json.dumps({
            "cache_manifest": str(args.batch_cache / "cache_manifest.json"),
            "modes": list(report["modes"]),
            "frozen_test_i_blocks_loaded": report["frozen_test_i_blocks_loaded"],
        }, indent=2))
        return
    if args.output_root is None:
        parser.error("Stage-1 execution requires --output-root")
    if args.data_dir is None and args.batch_cache is None:
        parser.error("Stage-1 execution requires --data-dir or --batch-cache")
    modes = MODES if args.mode == "both" else (args.mode,)
    summary = run_stage1(
        modes=modes, models=args.models, data_dir=args.data_dir, batch_cache=args.batch_cache,
        output_root=args.output_root, root_seed=args.root_seed,
        device=args.device, max_train_points=args.max_train_points,
        max_validation_points=args.max_validation_points,
    )
    compact = {
        "counts": summary["counts"],
        "statuses": {
            mode: {model: result["status"] for model, result in values.items()}
            for mode, values in summary["results"].items()
        },
        "summary": str(args.output_root / "summary.json"),
    }
    print(json.dumps(compact, indent=2))
    if args.fail_on_failed and summary["counts"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
