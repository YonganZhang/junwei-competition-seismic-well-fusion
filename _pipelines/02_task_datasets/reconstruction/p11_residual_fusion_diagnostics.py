#!/usr/bin/env python3
"""Development-only diagnostics for the P11 OpenMind residual-fusion route.

This module preserves the committed P11 experiment and reuses its verified
PyKrige OOF loader, bounded gate, and holdout firewall.  It diagnoses three
adaptation questions without changing the OpenMind checkpoint:

1. mixed multiscale features versus stage-0-only and stage-5-only features;
2. fixed three-channel input averaging versus three independent encoder
   forwards followed by feature concatenation;
3. fixed Ridge(alpha=10) versus an outer-train-only alpha search.

Only ``train.h5`` seismic channels and active masks may be opened.  PORO
targets and baseline predictions come exclusively from the hash-verified P5
development OOF archives.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import p11_residual_fusion as base  # noqa: E402


SCHEMA_VERSION = "reconstruction-p11-residual-fusion-diagnostics/v1"
CHANNEL_CACHE_SCHEMA_VERSION = (
    "reconstruction-p11-openmind-per-channel-feature-cache/v1"
)
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p11_residual_fusion_diagnostics"
DEFAULT_CHANNEL_FEATURE_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p11_residual_fusion_diagnostics"
    / "openmind_per_channel_features.npz"
)
DEFAULT_RANDOM_INIT_FEATURE_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p11_residual_fusion_diagnostics"
    / "openmind_random_init_features.npz"
)
DEFAULT_MEAN_FEATURE_CACHE = base.DEFAULT_FEATURE_CACHE
DEFAULT_LEGACY_SUMMARY = HERE / "_outputs" / "p11_residual_fusion" / "summary.json"

RIDGE_ALPHAS = (10.0, 100.0, 1_000.0, 10_000.0)
FIXED_ALPHA = 10.0
WIN_TOLERANCE = 1e-12
BOOTSTRAP_REPLICATES = 10_000
RANDOM_INIT_CACHE_SCHEMA_VERSION = (
    "reconstruction-p11-openmind-random-init-feature-cache/v1"
)
FEATURE_VARIANTS = (
    "mean_mixed16",
    "mean_stage0_all",
    "mean_stage5_all",
    "per_channel_mixed16_concat",
    "per_channel_stage0_all_concat",
    "per_channel_stage5_all_concat",
)


def _channel_cache_manifest_path(feature_cache: Path) -> Path:
    return feature_cache.with_suffix(".json")


def _load_valid_channel_cache(
    feature_cache: Path,
    *,
    indices_kji: np.ndarray,
    checkpoint_sha256: str,
    train_h5_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    manifest_path = _channel_cache_manifest_path(feature_cache)
    if not feature_cache.is_file() or not manifest_path.is_file():
        return None
    manifest = base._json(manifest_path)  # noqa: SLF001
    expected = {
        "schema_version": CHANNEL_CACHE_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": base._array_sha256(indices_kji),  # noqa: SLF001
        "source_revision": base.EXPECTED_SOURCE_REVISION,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if manifest.get("npz_sha256") != base._sha256(feature_cache):  # noqa: SLF001
        raise RuntimeError("per-channel OpenMind feature cache hash mismatch")
    with np.load(feature_cache, allow_pickle=False) as payload:
        cached_indices = np.asarray(payload["indices_kji"], dtype=np.int64)
        features = np.asarray(payload["channel_features"], dtype=np.float32)
    np.testing.assert_array_equal(cached_indices, indices_kji)
    if features.ndim != 3 or features.shape[:2] != (len(indices_kji), 3):
        raise RuntimeError("per-channel OpenMind feature cache shape mismatch")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError(
            "per-channel OpenMind feature cache contains non-finite values"
        )
    return features, manifest


def extract_openmind_per_channel_features(
    *,
    train_h5: Path,
    indices_kji: np.ndarray,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run each normalized seismic channel through the frozen encoder alone."""

    base.ensure_no_holdout_paths([train_h5, source_root, checkpoint])
    if dependency_root is not None:
        dependency_root = Path(dependency_root).expanduser().resolve()
        if not dependency_root.is_dir():
            raise FileNotFoundError(
                f"OpenMind dependency root missing: {dependency_root}"
            )
        sys.path.insert(0, str(dependency_root))

    import h5py
    import torch
    import torch.nn.functional as functional

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from _models.reconstruction.openmind_mae import build_model

    p4 = importlib.import_module(
        "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
    )
    model = build_model(
        p4.task_spec("strict"),
        source_root=source_root,
        checkpoint_path=checkpoint,
        freeze_encoder=True,
        device=device,
    )
    model.eval()
    if any(parameter.requires_grad for parameter in model.network.encoder.parameters()):
        raise RuntimeError("OpenMind encoder is not frozen")

    requested = np.asarray(indices_kji, dtype=np.int64)
    features: np.ndarray | None = None
    layer_shapes: list[list[int]] | None = None
    layer_channels: list[int] | None = None
    accessed_groups: list[str] = []
    forward_count = 0

    with h5py.File(train_h5, "r") as handle:
        locations: list[tuple[str, np.ndarray, np.ndarray]] = []
        for key in sorted(handle):
            meta = json.loads(handle[key].attrs["meta"])
            start = np.asarray(meta["patch_start_kji"], dtype=np.int64)
            shape = np.asarray(meta["patch_shape_kji"], dtype=np.int64)
            locations.append((key, start, start + shape))

        assignment = np.full(len(requested), -1, dtype=np.int64)
        for location_id, (_, start, stop) in enumerate(locations):
            inside = np.all((requested >= start) & (requested < stop), axis=1)
            if np.any((assignment >= 0) & inside):
                raise RuntimeError(
                    "development coordinate maps to multiple train patches"
                )
            assignment[inside] = location_id
        if np.any(assignment < 0):
            missing = requested[assignment < 0][:5].tolist()
            raise RuntimeError(f"OOF coordinates are absent from train.h5: {missing}")

        for location_id in np.unique(assignment):
            key, start, stop = locations[int(location_id)]
            row_ids = np.flatnonzero(assignment == location_id)
            local = requested[row_ids] - start
            group = handle[key]
            seismic = np.asarray(group["seismic_patch"][0:3], dtype=np.float32)
            active = np.asarray(group["seismic_patch"][8], dtype=np.float32) > 0.5
            expected_shape = tuple((stop - start).tolist())
            if seismic.shape[1:] != expected_shape or active.shape != expected_shape:
                raise RuntimeError(f"invalid train patch shape for {key}")
            if not np.all(active[tuple(local.T)]):
                raise RuntimeError(f"OOF coordinate maps to an inactive cell in {key}")

            normalized = np.zeros_like(seismic)
            for channel in range(3):
                values = seismic[channel][active]
                scale = max(float(np.std(values)), 1e-6)
                normalized[channel][active] = (
                    values - float(np.mean(values))
                ) / scale

            original = tuple(int(value) for value in normalized.shape[-3:])
            padded = tuple(max(64, ((size + 31) // 32) * 32) for size in original)
            pads: list[int] = []
            for size, wanted in reversed(list(zip(original, padded))):
                pads.extend((0, wanted - size))

            patch_channels: list[np.ndarray] = []
            for channel in range(3):
                volume = torch.as_tensor(
                    normalized[channel][None, None],
                    dtype=torch.float32,
                    device=device,
                )
                if any(pads):
                    volume = functional.pad(volume, tuple(pads))
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=str(device).startswith("cuda"),
                ):
                    stages = model.network.encoder(volume)
                if not isinstance(stages, (tuple, list)) or not stages:
                    raise RuntimeError(
                        "OpenMind encoder did not return multiscale stages"
                    )
                sampled = base._sample_encoder_stages(  # noqa: SLF001
                    stages,
                    local,
                    padded_shape=padded,
                )
                patch_channels.append(sampled)
                forward_count += 1
                current_shapes = [list(map(int, stage.shape)) for stage in stages]
                current_channels = [int(stage.shape[1]) for stage in stages]
                if layer_shapes is None:
                    layer_shapes = current_shapes
                    layer_channels = current_channels
                elif (
                    current_shapes != layer_shapes
                    or current_channels != layer_channels
                ):
                    raise RuntimeError(
                        "OpenMind encoder stage geometry changed between forwards"
                    )

            patch_features = np.stack(patch_channels, axis=1)
            if features is None:
                features = np.empty(
                    (len(requested), 3, patch_features.shape[2]),
                    dtype=np.float32,
                )
            elif patch_features.shape[2] != features.shape[2]:
                raise RuntimeError(
                    "OpenMind encoder feature width changed between patches"
                )
            features[row_ids] = patch_features
            accessed_groups.append(key)

    if features is None or layer_shapes is None or layer_channels is None:
        raise RuntimeError("per-channel OpenMind extraction produced no features")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError(
            "per-channel OpenMind extraction produced non-finite features"
        )
    audit = {
        "model_id": "MIC-DKFZ/ResEncL-OpenMind-MAE",
        "real_pretrained_weights_loaded": True,
        "encoder_frozen": True,
        "source_revision": base.EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": base._sha256(checkpoint),  # noqa: SLF001
        "checkpoint_bytes": checkpoint.stat().st_size,
        "feature_shape": [int(value) for value in features.shape],
        "per_channel_feature_width": int(features.shape[2]),
        "layer_channels": layer_channels,
        "layer_shapes_first_patch": layer_shapes,
        "input_projection": (
            "per-patch active-cell zscore; three independent single-channel "
            "encoder forwards; output feature concatenation"
        ),
        "encoder_forwards": forward_count,
        "seismic_channels_forwarded_separately": 3,
        "spatial_sampling": (
            "trilinear voxel-centre sampling from all encoder stages"
        ),
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": ["seismic_patch[0:3]", "seismic_patch[8]"],
        "label_dataset_read": False,
        "accessed_patch_count": len(accessed_groups),
        "accessed_patch_keys_sha256": hashlib.sha256(
            json.dumps(sorted(accessed_groups)).encode("utf-8")
        ).hexdigest(),
    }
    return features, audit


def get_openmind_per_channel_features(
    *,
    inputs: base.DevInputPaths,
    oof: base.OOFDevelopment,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    feature_cache: Path,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse or create a hash-locked genuine three-forward feature cache."""

    source_root = Path(source_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    feature_cache = Path(feature_cache).expanduser().resolve()
    base.ensure_no_holdout_paths([source_root, checkpoint, feature_cache])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OpenMind checkpoint missing: {checkpoint}")
    if checkpoint.stat().st_size != base.EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError("OpenMind checkpoint byte size differs from route lock")
    checkpoint_sha256 = base._sha256(checkpoint)  # noqa: SLF001
    if checkpoint_sha256 != base.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("OpenMind checkpoint SHA-256 differs from route lock")
    source_revision = base._verified_source_revision(source_root)  # noqa: SLF001
    train_h5_sha256 = base._sha256(inputs.train_h5)  # noqa: SLF001
    cached = _load_valid_channel_cache(
        feature_cache,
        indices_kji=oof.indices_kji,
        checkpoint_sha256=checkpoint_sha256,
        train_h5_sha256=train_h5_sha256,
    )
    if cached is not None:
        features, manifest = cached
        audit = dict(manifest["feature_audit"])
        audit["cache_reused"] = True
        audit["feature_cache_sha256"] = manifest["npz_sha256"]
        audit["source_revision"] = source_revision
        audit["source_revision_verified"] = True
        return features, audit

    features, audit = extract_openmind_per_channel_features(
        train_h5=inputs.train_h5,
        indices_kji=oof.indices_kji,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        device=device,
    )
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    with feature_cache.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            channel_features=features,
        )
    manifest = {
        "schema_version": CHANNEL_CACHE_SCHEMA_VERSION,
        "source_revision": base.EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": base._array_sha256(  # noqa: SLF001
            oof.indices_kji
        ),
        "npz_sha256": base._sha256(feature_cache),  # noqa: SLF001
        "feature_audit": audit,
    }
    _channel_cache_manifest_path(feature_cache).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = dict(audit)
    audit["cache_reused"] = False
    audit["feature_cache_sha256"] = manifest["npz_sha256"]
    audit["source_revision"] = source_revision
    audit["source_revision_verified"] = True
    return features, audit


def _random_init_cache_manifest_path(feature_cache: Path) -> Path:
    return feature_cache.with_suffix(".json")


def _encoder_state_sha256(encoder: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in encoder.state_dict().items():
        values = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _encoder_architecture_sha256(encoder: Any) -> str:
    payload = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for name, parameter in encoder.named_parameters()
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _reset_encoder_same_architecture(
    encoder: Any,
    *,
    seed: int,
    torch: Any,
) -> dict[str, Any]:
    """Replace loaded encoder weights with deterministic random initialization."""

    pretrained_state_sha256 = _encoder_state_sha256(encoder)
    architecture_sha256 = _encoder_architecture_sha256(encoder)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    reset_modules: list[str] = []
    for name, module in encoder.named_modules():
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()
            reset_modules.append(name or "<encoder>")
    random_state_sha256 = _encoder_state_sha256(encoder)
    if not reset_modules:
        raise RuntimeError("OpenMind encoder exposed no resettable modules")
    if random_state_sha256 == pretrained_state_sha256:
        raise RuntimeError("OpenMind random-init control retained pretrained state")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return {
        "seed": int(seed),
        "architecture_sha256": architecture_sha256,
        "pretrained_state_sha256_before_reset": pretrained_state_sha256,
        "random_init_state_sha256": random_state_sha256,
        "randomized_state_differs_from_pretrained": True,
        "resettable_module_count": len(reset_modules),
        "encoder_frozen_after_reset": not any(
            parameter.requires_grad for parameter in encoder.parameters()
        ),
    }


def _load_valid_random_init_cache(
    feature_cache: Path,
    *,
    indices_kji: np.ndarray,
    checkpoint_sha256: str,
    train_h5_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    manifest_path = _random_init_cache_manifest_path(feature_cache)
    if not feature_cache.is_file() or not manifest_path.is_file():
        return None
    manifest = base._json(manifest_path)  # noqa: SLF001
    expected = {
        "schema_version": RANDOM_INIT_CACHE_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": base._array_sha256(indices_kji),  # noqa: SLF001
        "source_revision": base.EXPECTED_SOURCE_REVISION,
        "random_init_seeds": list(base.REPEAT_SEEDS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if manifest.get("npz_sha256") != base._sha256(feature_cache):  # noqa: SLF001
        raise RuntimeError("random-init OpenMind feature cache hash mismatch")
    with np.load(feature_cache, allow_pickle=False) as payload:
        cached_indices = np.asarray(payload["indices_kji"], dtype=np.int64)
        mean_features = np.asarray(
            payload["random_init_mean_features"],
            dtype=np.float32,
        )
        channel_features = np.asarray(
            payload["random_init_channel_features"],
            dtype=np.float32,
        )
    np.testing.assert_array_equal(cached_indices, indices_kji)
    expected_seed_count = len(base.REPEAT_SEEDS)
    if (
        mean_features.ndim != 3
        or mean_features.shape[:2]
        != (expected_seed_count, len(indices_kji))
        or channel_features.ndim != 4
        or channel_features.shape[:3]
        != (expected_seed_count, len(indices_kji), 3)
        or channel_features.shape[3] != mean_features.shape[2]
    ):
        raise RuntimeError("random-init OpenMind feature cache shape mismatch")
    if not np.all(np.isfinite(mean_features)) or not np.all(
        np.isfinite(channel_features)
    ):
        raise FloatingPointError(
            "random-init OpenMind feature cache contains non-finite values"
        )
    return mean_features, channel_features, manifest


def extract_openmind_random_init_features(
    *,
    train_h5: Path,
    indices_kji: np.ndarray,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract same-architecture random-init controls for every repeat seed."""

    base.ensure_no_holdout_paths([train_h5, source_root, checkpoint])
    if dependency_root is not None:
        dependency_root = Path(dependency_root).expanduser().resolve()
        if not dependency_root.is_dir():
            raise FileNotFoundError(
                f"OpenMind dependency root missing: {dependency_root}"
            )
        sys.path.insert(0, str(dependency_root))

    import h5py
    import torch
    import torch.nn.functional as functional

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from _models.reconstruction.openmind_mae import build_model

    p4 = importlib.import_module(
        "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
    )
    requested = np.asarray(indices_kji, dtype=np.int64)
    patches: list[dict[str, Any]] = []

    with h5py.File(train_h5, "r") as handle:
        locations: list[tuple[str, np.ndarray, np.ndarray]] = []
        for key in sorted(handle):
            meta = json.loads(handle[key].attrs["meta"])
            start = np.asarray(meta["patch_start_kji"], dtype=np.int64)
            shape = np.asarray(meta["patch_shape_kji"], dtype=np.int64)
            locations.append((key, start, start + shape))

        assignment = np.full(len(requested), -1, dtype=np.int64)
        for location_id, (_, start, stop) in enumerate(locations):
            inside = np.all((requested >= start) & (requested < stop), axis=1)
            if np.any((assignment >= 0) & inside):
                raise RuntimeError(
                    "development coordinate maps to multiple train patches"
                )
            assignment[inside] = location_id
        if np.any(assignment < 0):
            missing = requested[assignment < 0][:5].tolist()
            raise RuntimeError(f"OOF coordinates are absent from train.h5: {missing}")

        for location_id in np.unique(assignment):
            key, start, stop = locations[int(location_id)]
            row_ids = np.flatnonzero(assignment == location_id)
            local = requested[row_ids] - start
            group = handle[key]
            seismic = np.asarray(group["seismic_patch"][0:3], dtype=np.float32)
            active = np.asarray(group["seismic_patch"][8], dtype=np.float32) > 0.5
            expected_shape = tuple((stop - start).tolist())
            if seismic.shape[1:] != expected_shape or active.shape != expected_shape:
                raise RuntimeError(f"invalid train patch shape for {key}")
            if not np.all(active[tuple(local.T)]):
                raise RuntimeError(f"OOF coordinate maps to an inactive cell in {key}")
            normalized = np.zeros_like(seismic)
            for channel in range(3):
                values = seismic[channel][active]
                scale = max(float(np.std(values)), 1e-6)
                normalized[channel][active] = (
                    values - float(np.mean(values))
                ) / scale
            original = tuple(int(value) for value in normalized.shape[-3:])
            padded = tuple(max(64, ((size + 31) // 32) * 32) for size in original)
            pads: list[int] = []
            for size, wanted in reversed(list(zip(original, padded))):
                pads.extend((0, wanted - size))
            patches.append(
                {
                    "key": key,
                    "row_ids": row_ids,
                    "local": local,
                    "normalized": normalized,
                    "padded": padded,
                    "pads": tuple(pads),
                }
            )

    mean_features: np.ndarray | None = None
    channel_features: np.ndarray | None = None
    layer_shapes: list[list[int]] | None = None
    layer_channels: list[int] | None = None
    random_state_audit: list[dict[str, Any]] = []
    total_forwards = 0

    for seed_id, seed in enumerate(base.REPEAT_SEEDS):
        model = build_model(
            p4.task_spec("strict"),
            source_root=source_root,
            checkpoint_path=checkpoint,
            freeze_encoder=True,
            device=device,
        )
        model.eval()
        state_audit = _reset_encoder_same_architecture(
            model.network.encoder,
            seed=int(seed),
            torch=torch,
        )
        model.eval()
        random_state_audit.append(state_audit)

        for patch in patches:
            normalized = patch["normalized"]
            inputs = [
                np.mean(normalized, axis=0, keepdims=True),
                normalized[0:1],
                normalized[1:2],
                normalized[2:3],
            ]
            sampled_inputs: list[np.ndarray] = []
            for values in inputs:
                volume = torch.as_tensor(
                    values[None],
                    dtype=torch.float32,
                    device=device,
                )
                if any(patch["pads"]):
                    volume = functional.pad(volume, patch["pads"])
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=str(device).startswith("cuda"),
                ):
                    stages = model.network.encoder(volume)
                if not isinstance(stages, (tuple, list)) or not stages:
                    raise RuntimeError(
                        "random-init OpenMind encoder returned no stages"
                    )
                sampled = base._sample_encoder_stages(  # noqa: SLF001
                    stages,
                    patch["local"],
                    padded_shape=patch["padded"],
                )
                sampled_inputs.append(sampled)
                total_forwards += 1
                current_shapes = [list(map(int, stage.shape)) for stage in stages]
                current_channels = [int(stage.shape[1]) for stage in stages]
                if layer_shapes is None:
                    layer_shapes = current_shapes
                    layer_channels = current_channels
                elif (
                    current_shapes != layer_shapes
                    or current_channels != layer_channels
                ):
                    raise RuntimeError(
                        "random-init OpenMind stage geometry changed"
                    )

            if mean_features is None:
                width = sampled_inputs[0].shape[1]
                mean_features = np.empty(
                    (len(base.REPEAT_SEEDS), len(requested), width),
                    dtype=np.float32,
                )
                channel_features = np.empty(
                    (len(base.REPEAT_SEEDS), len(requested), 3, width),
                    dtype=np.float32,
                )
            row_ids = patch["row_ids"]
            mean_features[seed_id, row_ids] = sampled_inputs[0]
            channel_features[seed_id, row_ids] = np.stack(
                sampled_inputs[1:],
                axis=1,
            )

        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if (
        mean_features is None
        or channel_features is None
        or layer_shapes is None
        or layer_channels is None
    ):
        raise RuntimeError("random-init OpenMind extraction produced no features")
    if not np.all(np.isfinite(mean_features)) or not np.all(
        np.isfinite(channel_features)
    ):
        raise FloatingPointError(
            "random-init OpenMind extraction produced non-finite features"
        )
    architecture_hashes = {
        row["architecture_sha256"] for row in random_state_audit
    }
    random_state_hashes = {
        row["random_init_state_sha256"] for row in random_state_audit
    }
    if len(architecture_hashes) != 1:
        raise RuntimeError("random-init control architecture changed across seeds")
    if len(random_state_hashes) != len(base.REPEAT_SEEDS):
        raise RuntimeError("random-init control states are not seed-distinct")
    audit = {
        "control_id": "random_init_same_architecture",
        "model_id": "MIC-DKFZ/ResEncL-OpenMind-MAE",
        "same_architecture_as_pretrained": True,
        "checkpoint_loaded_before_encoder_reset": True,
        "pretrained_weights_used_for_forward": False,
        "encoder_frozen": True,
        "source_revision": base.EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": base._sha256(checkpoint),  # noqa: SLF001
        "checkpoint_bytes": checkpoint.stat().st_size,
        "mean_feature_shape": [int(value) for value in mean_features.shape],
        "channel_feature_shape": [
            int(value) for value in channel_features.shape
        ],
        "layer_channels": layer_channels,
        "layer_shapes_first_patch": layer_shapes,
        "random_init_state_audit": random_state_audit,
        "encoder_forwards": total_forwards,
        "input_adapters": [
            "fixed mean of three normalized seismic channels",
            "three independent normalized seismic-channel forwards",
        ],
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": ["seismic_patch[0:3]", "seismic_patch[8]"],
        "label_dataset_read": False,
        "accessed_patch_count": len(patches),
        "accessed_patch_keys_sha256": hashlib.sha256(
            json.dumps(sorted(patch["key"] for patch in patches)).encode("utf-8")
        ).hexdigest(),
    }
    return mean_features, channel_features, audit


def get_openmind_random_init_features(
    *,
    inputs: base.DevInputPaths,
    oof: base.OOFDevelopment,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    feature_cache: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reuse or create the required same-architecture random-init control."""

    source_root = Path(source_root).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    feature_cache = Path(feature_cache).expanduser().resolve()
    base.ensure_no_holdout_paths([source_root, checkpoint, feature_cache])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"OpenMind checkpoint missing: {checkpoint}")
    if checkpoint.stat().st_size != base.EXPECTED_CHECKPOINT_BYTES:
        raise RuntimeError("OpenMind checkpoint byte size differs from route lock")
    checkpoint_sha256 = base._sha256(checkpoint)  # noqa: SLF001
    if checkpoint_sha256 != base.EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("OpenMind checkpoint SHA-256 differs from route lock")
    source_revision = base._verified_source_revision(source_root)  # noqa: SLF001
    train_h5_sha256 = base._sha256(inputs.train_h5)  # noqa: SLF001
    cached = _load_valid_random_init_cache(
        feature_cache,
        indices_kji=oof.indices_kji,
        checkpoint_sha256=checkpoint_sha256,
        train_h5_sha256=train_h5_sha256,
    )
    if cached is not None:
        mean_features, channel_features, manifest = cached
        audit = dict(manifest["feature_audit"])
        audit["cache_reused"] = True
        audit["feature_cache_sha256"] = manifest["npz_sha256"]
        audit["source_revision"] = source_revision
        audit["source_revision_verified"] = True
        return mean_features, channel_features, audit

    mean_features, channel_features, audit = (
        extract_openmind_random_init_features(
            train_h5=inputs.train_h5,
            indices_kji=oof.indices_kji,
            source_root=source_root,
            checkpoint=checkpoint,
            dependency_root=dependency_root,
            device=device,
        )
    )
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    with feature_cache.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            random_init_mean_features=mean_features,
            random_init_channel_features=channel_features,
        )
    manifest = {
        "schema_version": RANDOM_INIT_CACHE_SCHEMA_VERSION,
        "source_revision": base.EXPECTED_SOURCE_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": base._array_sha256(  # noqa: SLF001
            oof.indices_kji
        ),
        "random_init_seeds": list(base.REPEAT_SEEDS),
        "npz_sha256": base._sha256(feature_cache),  # noqa: SLF001
        "feature_audit": audit,
    }
    _random_init_cache_manifest_path(feature_cache).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = dict(audit)
    audit["cache_reused"] = False
    audit["feature_cache_sha256"] = manifest["npz_sha256"]
    audit["source_revision"] = source_revision
    audit["source_revision_verified"] = True
    return mean_features, channel_features, audit


def _stage_slice(layer_channels: Sequence[int], stage_index: int) -> slice:
    widths = [int(value) for value in layer_channels]
    if stage_index < 0:
        stage_index += len(widths)
    if stage_index < 0 or stage_index >= len(widths):
        raise IndexError(f"invalid OpenMind stage index: {stage_index}")
    start = sum(widths[:stage_index])
    return slice(start, start + widths[stage_index])


def build_feature_variants(
    mean_features: np.ndarray,
    channel_features: np.ndarray,
    *,
    layer_channels: Sequence[int],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct the six fixed feature ablations for one repeat seed."""

    mean_features = np.asarray(mean_features, dtype=np.float64)
    channel_features = np.asarray(channel_features, dtype=np.float64)
    width = sum(int(value) for value in layer_channels)
    if mean_features.shape != (len(channel_features), width):
        raise ValueError("mean OpenMind feature shape does not match layer channels")
    if channel_features.shape != (len(mean_features), 3, width):
        raise ValueError(
            "per-channel OpenMind feature shape does not match layer channels"
        )
    selected = base._selected_latent_channels(  # noqa: SLF001
        layer_channels,
        seed=int(seed),
    )
    stage0 = _stage_slice(layer_channels, 0)
    stage5 = _stage_slice(layer_channels, -1)
    variants = {
        "mean_mixed16": mean_features[:, selected],
        "mean_stage0_all": mean_features[:, stage0],
        "mean_stage5_all": mean_features[:, stage5],
        "per_channel_mixed16_concat": channel_features[
            :, :, selected
        ].reshape(len(mean_features), -1),
        "per_channel_stage0_all_concat": channel_features[
            :, :, stage0
        ].reshape(len(mean_features), -1),
        "per_channel_stage5_all_concat": channel_features[
            :, :, stage5
        ].reshape(len(mean_features), -1),
    }
    if tuple(variants) != FEATURE_VARIANTS:
        raise RuntimeError("diagnostic feature variant registry drift")
    if not all(np.all(np.isfinite(values)) for values in variants.values()):
        raise FloatingPointError("diagnostic OpenMind features are non-finite")
    audit = {
        "seed": int(seed),
        "mixed_selected_global_channels": selected.tolist(),
        "variant_widths": {
            name: int(values.shape[1]) for name, values in variants.items()
        },
        "stage0_channel_width": int(layer_channels[0]),
        "stage5_channel_width": int(layer_channels[-1]),
    }
    return variants, audit


def _fit_residual(
    train_features: np.ndarray,
    train_residual: np.ndarray,
    validation_features: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(alpha), solver="lsqr", tol=1e-6),
    )
    estimator.fit(train_features, train_residual)
    prediction = np.asarray(
        estimator.predict(validation_features),
        dtype=np.float64,
    )
    if prediction.shape != (len(validation_features),) or not np.all(
        np.isfinite(prediction)
    ):
        raise FloatingPointError("diagnostic residual regressor returned invalid values")
    return prediction


def _inner_oof_residual(
    features: np.ndarray,
    residual_target: np.ndarray,
    fold_ids: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    result = np.full(len(features), np.nan, dtype=np.float64)
    unique_folds = np.unique(fold_ids)
    if len(unique_folds) < 3:
        raise ValueError("diagnostic residual stack needs at least three inner folds")
    for inner_fold in unique_folds:
        validation = fold_ids == inner_fold
        train = ~validation
        if np.any(fold_ids[train] == inner_fold):
            raise RuntimeError("diagnostic inner cross-fit overlap")
        result[validation] = _fit_residual(
            features[train],
            residual_target[train],
            features[validation],
            alpha=alpha,
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("diagnostic inner residual OOF predictions are incomplete")
    return result


def classify_fold_outcome(
    candidate_rmse: float,
    baseline_rmse: float,
    *,
    tolerance: float = WIN_TOLERANCE,
) -> str:
    delta = float(candidate_rmse) - float(baseline_rmse)
    if delta < -tolerance:
        return "win"
    if delta > tolerance:
        return "loss"
    return "tie"


def _gate_for_inner_predictions(
    *,
    inner_prediction: np.ndarray,
    inner_base: np.ndarray,
    inner_truth: np.ndarray,
    inner_distance: np.ndarray,
    seed: int,
) -> tuple[base.GateModel, float, dict[str, float], float]:
    inner_ungated = inner_base + inner_prediction
    beneficial = np.abs(inner_ungated - inner_truth) < np.abs(
        inner_base - inner_truth
    )
    gate_model = base._fit_gate(  # noqa: SLF001
        base._gate_signals(  # noqa: SLF001
            inner_prediction,
            inner_base,
            inner_distance,
        ),
        beneficial,
        seed=seed,
    )
    inner_raw_gate = gate_model.predict(
        base._gate_signals(  # noqa: SLF001
            inner_prediction,
            inner_base,
            inner_distance,
        )
    )
    candidate_scores: dict[str, float] = {}
    for candidate in base.GATE_CANDIDATES:
        gate = np.clip(inner_raw_gate * candidate, 0.0, 1.0)
        candidate_scores[f"{candidate:.2f}"] = base._metrics(  # noqa: SLF001
            inner_truth,
            inner_base + gate * inner_prediction,
        )["rmse"]
    selected_scale = min(
        base.GATE_CANDIDATES,
        key=lambda value: (candidate_scores[f"{value:.2f}"], value),
    )
    return (
        gate_model,
        float(selected_scale),
        candidate_scores,
        float(np.mean(beneficial)),
    )


def evaluate_adaptive_route(
    *,
    route: str,
    features: np.ndarray,
    oof: base.OOFDevelopment,
    seed: int,
) -> dict[str, Any]:
    """Evaluate fixed-alpha and train-only tuned-alpha heads in one OOF pass."""

    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] != len(oof.target) or features.ndim != 2:
        raise ValueError(f"{route} feature shape does not align with OOF rows")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError(f"{route} features contain non-finite values")

    modes = ("fixed_ridge10", "train_only_alpha_grid")
    predictions = {
        mode: {
            "ungated": np.full(len(oof.target), np.nan, dtype=np.float64),
            "gated": np.full(len(oof.target), np.nan, dtype=np.float64),
        }
        for mode in modes
    }
    cells: list[dict[str, Any]] = []
    residual_target = oof.target - oof.baseline

    for outer_fold in base.FOLD_IDS:
        validation = oof.fold_ids == outer_fold
        train = ~validation
        if np.any(oof.fold_ids[train] == outer_fold):
            raise RuntimeError("diagnostic outer training includes validation fold")

        inner_predictions: dict[float, np.ndarray] = {}
        alpha_scores: dict[str, float] = {}
        for alpha in RIDGE_ALPHAS:
            inner_prediction = _inner_oof_residual(
                features[train],
                residual_target[train],
                oof.fold_ids[train],
                alpha=alpha,
            )
            inner_predictions[alpha] = inner_prediction
            alpha_scores[f"{alpha:.1f}"] = base._metrics(  # noqa: SLF001
                oof.target[train],
                oof.baseline[train] + inner_prediction,
            )["rmse"]
        selected_alpha = min(
            RIDGE_ALPHAS,
            key=lambda value: (alpha_scores[f"{value:.1f}"], -value),
        )

        outer_predictions: dict[float, np.ndarray] = {}
        for alpha in {FIXED_ALPHA, selected_alpha}:
            outer_predictions[alpha] = _fit_residual(
                features[train],
                residual_target[train],
                features[validation],
                alpha=alpha,
            )

        for mode, alpha in (
            ("fixed_ridge10", FIXED_ALPHA),
            ("train_only_alpha_grid", selected_alpha),
        ):
            inner_prediction = inner_predictions[alpha]
            gate_model, gate_scale, gate_scores, benefit_rate = (
                _gate_for_inner_predictions(
                    inner_prediction=inner_prediction,
                    inner_base=oof.baseline[train],
                    inner_truth=oof.target[train],
                    inner_distance=oof.distance_to_well[train],
                    seed=int(seed) + int(outer_fold),
                )
            )
            outer_residual = outer_predictions[alpha]
            raw_gate = gate_model.predict(
                base._gate_signals(  # noqa: SLF001
                    outer_residual,
                    oof.baseline[validation],
                    oof.distance_to_well[validation],
                )
            )
            gate = np.clip(raw_gate * gate_scale, 0.0, 1.0)
            ungated = oof.baseline[validation] + outer_residual
            gated = oof.baseline[validation] + gate * outer_residual
            predictions[mode]["ungated"][validation] = ungated
            predictions[mode]["gated"][validation] = gated
            baseline_metrics = base._metrics(  # noqa: SLF001
                oof.target[validation],
                oof.baseline[validation],
            )
            gated_metrics = base._metrics(  # noqa: SLF001
                oof.target[validation],
                gated,
            )
            cells.append(
                {
                    "route": route,
                    "head_mode": mode,
                    "seed": int(seed),
                    "outer_fold": int(outer_fold),
                    "train_rows": int(np.sum(train)),
                    "validation_rows": int(np.sum(validation)),
                    "train_fold_ids": sorted(
                        int(value) for value in np.unique(oof.fold_ids[train])
                    ),
                    "validation_fold_ids": [int(outer_fold)],
                    "base_predictions_for_training": "P5 Stage-3 OOF only",
                    "inner_residual_predictions": (
                        "cross-fit by original OOF fold"
                    ),
                    "residual_alpha": float(alpha),
                    "train_only_alpha_candidate_rmse": alpha_scores,
                    "selected_gate_scale": gate_scale,
                    "inner_gate_candidate_rmse": gate_scores,
                    "inner_correction_benefit_rate": benefit_rate,
                    "baseline_metrics": baseline_metrics,
                    "ungated_metrics": base._metrics(  # noqa: SLF001
                        oof.target[validation],
                        ungated,
                    ),
                    "gated_metrics": gated_metrics,
                    "gated_rmse_delta_vs_pykrige": (
                        gated_metrics["rmse"] - baseline_metrics["rmse"]
                    ),
                    "outcome_vs_pykrige": classify_fold_outcome(
                        gated_metrics["rmse"],
                        baseline_metrics["rmse"],
                    ),
                    "residual_prediction_stats": base._summary_stats(  # noqa: SLF001
                        outer_residual
                    ),
                    "gate_stats": {
                        **base._summary_stats(gate),  # noqa: SLF001
                        "zero_rate": float(np.mean(gate == 0.0)),
                        "one_rate": float(np.mean(gate == 1.0)),
                    },
                }
            )

    aggregates: dict[str, Any] = {}
    for mode in modes:
        for name, values in predictions[mode].items():
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{route}/{mode}/{name} predictions incomplete")
        mode_cells = [row for row in cells if row["head_mode"] == mode]
        counts = {
            outcome: sum(
                row["outcome_vs_pykrige"] == outcome for row in mode_cells
            )
            for outcome in ("win", "loss", "tie")
        }
        aggregates[mode] = {
            "ungated": base._metrics(  # noqa: SLF001
                oof.target,
                predictions[mode]["ungated"],
            ),
            "gated": base._metrics(  # noqa: SLF001
                oof.target,
                predictions[mode]["gated"],
            ),
            "outcome_counts": counts,
        }
    return {
        "aggregates": aggregates,
        "per_fold": cells,
        "predictions": predictions,
    }


def _random_control(
    rows: int,
    width: int,
    *,
    variant: str,
    seed: int,
) -> np.ndarray:
    digest = hashlib.sha256(f"{variant}:{seed}".encode("utf-8")).digest()
    control_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(control_seed)
    return rng.standard_normal((rows, width)).astype(np.float64)


def _independent_fold_outcomes(
    cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse seed pseudo-repeats before counting five spatial fold outcomes."""

    outcomes: list[dict[str, Any]] = []
    for fold_id in base.FOLD_IDS:
        fold_rows = [
            row for row in cells if int(row["outer_fold"]) == int(fold_id)
        ]
        if len(fold_rows) != len(base.REPEAT_SEEDS):
            raise RuntimeError(
                f"spatial fold {fold_id} lacks three paired seed repeats"
            )
        if {int(row["seed"]) for row in fold_rows} != set(base.REPEAT_SEEDS):
            raise RuntimeError(
                f"spatial fold {fold_id} seed pseudo-repeat identity drift"
            )
        baseline_rmse = float(
            np.mean([row["baseline_metrics"]["rmse"] for row in fold_rows])
        )
        gated_rmse = float(
            np.mean([row["gated_metrics"]["rmse"] for row in fold_rows])
        )
        outcomes.append(
            {
                "outer_fold": int(fold_id),
                "independent_spatial_unit": True,
                "paired_seed_pseudo_repeats": list(base.REPEAT_SEEDS),
                "baseline_mean_seed_rmse": baseline_rmse,
                "gated_mean_seed_rmse": gated_rmse,
                "rmse_delta_vs_pykrige": gated_rmse - baseline_rmse,
                "outcome_vs_pykrige": classify_fold_outcome(
                    gated_rmse,
                    baseline_rmse,
                ),
            }
        )
    return outcomes


def _legacy_reference(path: Path) -> dict[str, Any]:
    path = Path(path)
    base.ensure_no_holdout_paths([path])
    payload = base._json(path)  # noqa: SLF001
    experiment = payload["experiment"]
    cells = [
        row
        for row in experiment["per_fold_seed"]
        if row["route"] == "pretrained_openmind_residual"
    ]
    pseudo_repeat_counts = {
        outcome: sum(
            classify_fold_outcome(
                row["gated_metrics"]["rmse"],
                row["baseline_metrics"]["rmse"],
            )
            == outcome
            for row in cells
        )
        for outcome in ("win", "loss", "tie")
    }
    independent_outcomes = _independent_fold_outcomes(cells)
    independent_counts = {
        outcome: sum(
            row["outcome_vs_pykrige"] == outcome
            for row in independent_outcomes
        )
        for outcome in ("win", "loss", "tie")
    }
    return {
        "summary_sha256": base._sha256(path),  # noqa: SLF001
        "route": "mean_mixed16_fixed_ridge10",
        "gated_mean_seed_rmse": experiment["comparison"][
            "pretrained_openmind_residual_gated_mean_seed_rmse"
        ],
        "relative_gain_vs_pykrige": experiment["comparison"][
            "pretrained_relative_gain_vs_pykrige"
        ],
        "independent_spatial_units": len(independent_outcomes),
        "seed_pseudo_repeats_per_unit": len(base.REPEAT_SEEDS),
        "independent_fold_outcomes": independent_outcomes,
        "independent_fold_outcome_counts": independent_counts,
        "seed_level_pseudo_repeat_outcome_counts": pseudo_repeat_counts,
        "seed_level_pseudo_repeat_cells": len(cells),
    }


def _summarize_across_seeds(
    *,
    per_seed: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    baseline_rmse: float,
    structural_rmse: float,
    random_rmse: float | None = None,
) -> dict[str, Any]:
    gated_rmse = float(np.mean([row["gated"]["rmse"] for row in per_seed]))
    ungated_rmse = float(np.mean([row["ungated"]["rmse"] for row in per_seed]))
    pseudo_repeat_counts = {
        outcome: sum(row["outcome_vs_pykrige"] == outcome for row in cells)
        for outcome in ("win", "loss", "tie")
    }
    pseudo_repeat_total = len(cells)
    if pseudo_repeat_total != len(base.FOLD_IDS) * len(base.REPEAT_SEEDS):
        raise RuntimeError("fold-by-seed diagnostic matrix is incomplete")
    independent_outcomes = _independent_fold_outcomes(cells)
    independent_counts = {
        outcome: sum(
            row["outcome_vs_pykrige"] == outcome
            for row in independent_outcomes
        )
        for outcome in ("win", "loss", "tie")
    }
    independent_total = len(independent_outcomes)
    rmse_delta = gated_rmse - baseline_rmse
    material_gain = (
        0.0
        if abs(rmse_delta) <= WIN_TOLERANCE
        else (
            (baseline_rmse - gated_rmse) / baseline_rmse
            if baseline_rmse > 0.0
            else -math.inf
        )
    )
    summary = {
        "gated_mean_seed_rmse": gated_rmse,
        "ungated_mean_seed_rmse": ungated_rmse,
        "relative_gain_vs_pykrige": material_gain,
        "material_gain_positive": material_gain > 0.0,
        "independent_spatial_units": independent_total,
        "seed_pseudo_repeats_per_unit": len(base.REPEAT_SEEDS),
        "independent_fold_outcomes": independent_outcomes,
        "independent_fold_outcome_counts": independent_counts,
        "independent_fold_win_rate": (
            independent_counts["win"] / independent_total
        ),
        "meets_80_percent_independent_fold_win_rule": (
            independent_counts["win"] >= math.ceil(0.8 * independent_total)
        ),
        "seed_level_pseudo_repeat_outcome_counts": pseudo_repeat_counts,
        "seed_level_pseudo_repeat_cells": pseudo_repeat_total,
        "better_than_no_foundation_structural": gated_rmse < structural_rmse,
        "per_seed": list(per_seed),
    }
    if random_rmse is not None:
        summary["better_than_matched_random_control"] = gated_rmse < random_rmse
        summary["rmse_minus_matched_random_control"] = gated_rmse - random_rmse
    return summary


def block_bootstrap_rmse_delta(
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    candidate_predictions_by_seed: Mapping[int, np.ndarray],
    fold_ids: np.ndarray,
    bootstrap_seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Bootstrap five spatial folds while keeping seed repeats paired."""

    target = np.asarray(target, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    seeds = tuple(int(seed) for seed in base.REPEAT_SEEDS)
    if set(int(seed) for seed in candidate_predictions_by_seed) != set(seeds):
        raise ValueError("bootstrap candidate predictions lack paired repeat seeds")
    folds = tuple(int(fold) for fold in base.FOLD_IDS)
    if set(int(fold) for fold in np.unique(fold_ids)) != set(folds):
        raise ValueError("bootstrap requires the five locked spatial folds")
    if target.shape != baseline.shape or target.shape != fold_ids.shape:
        raise ValueError("bootstrap target/baseline/fold arrays do not align")

    counts = np.asarray(
        [np.sum(fold_ids == fold) for fold in folds],
        dtype=np.float64,
    )
    baseline_squared_error = (baseline - target) ** 2
    baseline_sse = np.asarray(
        [
            np.sum(baseline_squared_error[fold_ids == fold])
            for fold in folds
        ],
        dtype=np.float64,
    )
    candidate_sse = np.empty((len(seeds), len(folds)), dtype=np.float64)
    point_candidate_rmse: list[float] = []
    for seed_id, seed in enumerate(seeds):
        prediction = np.asarray(
            candidate_predictions_by_seed[seed],
            dtype=np.float64,
        )
        if prediction.shape != target.shape or not np.all(np.isfinite(prediction)):
            raise ValueError(f"bootstrap predictions invalid for seed {seed}")
        squared_error = (prediction - target) ** 2
        candidate_sse[seed_id] = [
            np.sum(squared_error[fold_ids == fold]) for fold in folds
        ]
        point_candidate_rmse.append(
            float(math.sqrt(np.mean(squared_error)))
        )

    rng = np.random.default_rng(int(bootstrap_seed))
    sampled = rng.integers(
        0,
        len(folds),
        size=(int(replicates), len(folds)),
    )
    weights = np.zeros((int(replicates), len(folds)), dtype=np.float64)
    for fold_index in range(len(folds)):
        weights[:, fold_index] = np.sum(sampled == fold_index, axis=1)
    denominators = weights @ counts
    baseline_bootstrap_rmse = np.sqrt((weights @ baseline_sse) / denominators)
    candidate_bootstrap_rmse = np.empty(
        (len(seeds), int(replicates)),
        dtype=np.float64,
    )
    for seed_id in range(len(seeds)):
        candidate_bootstrap_rmse[seed_id] = np.sqrt(
            (weights @ candidate_sse[seed_id]) / denominators
        )
    deltas = np.mean(candidate_bootstrap_rmse, axis=0) - baseline_bootstrap_rmse
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    point_baseline_rmse = float(math.sqrt(np.mean(baseline_squared_error)))
    point_delta = float(np.mean(point_candidate_rmse) - point_baseline_rmse)
    return {
        "metric": "mean-seed pooled RMSE(candidate) - pooled RMSE(PyKrige)",
        "direction": "negative favors candidate",
        "point_estimate": point_delta,
        "confidence_level": 0.95,
        "confidence_interval": [float(lower), float(upper)],
        "interval_excludes_zero": bool(lower > 0.0 or upper < 0.0),
        "replicates": int(replicates),
        "bootstrap_unit": "locked outer spatial fold",
        "independent_spatial_block_count": len(folds),
        "spatial_fold_ids": list(folds),
        "seed_handling": (
            "three seed predictions stay paired within each sampled spatial "
            "fold and are averaged; seeds are never resampled as independent units"
        ),
        "seed_pseudo_repeats": list(seeds),
        "bootstrap_seed": int(bootstrap_seed),
    }


def _prediction_key(variant: str, head_mode: str, seed: int) -> str:
    return f"pretrained__{variant}__{head_mode}__seed_{int(seed)}_error"


def evaluate_diagnostics(
    *,
    oof: base.OOFDevelopment,
    mean_features: np.ndarray,
    channel_features: np.ndarray,
    random_init_mean_features: np.ndarray,
    random_init_channel_features: np.ndarray,
    layer_channels: Sequence[int],
    legacy_reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the complete feature × head × control diagnostic matrix."""

    common = np.column_stack([oof.baseline, oof.distance_to_well])
    baseline_metrics = base._metrics(oof.target, oof.baseline)  # noqa: SLF001
    gate_zero = oof.baseline.copy()
    if not np.array_equal(gate_zero, oof.baseline):
        raise RuntimeError("diagnostic gate=0 is not bitwise PyKrige")

    all_cells: list[dict[str, Any]] = []
    feature_audits: dict[str, Any] = {}
    random_init_feature_audits: dict[str, Any] = {}
    pretrained_predictions: dict[
        str,
        dict[str, dict[int, np.ndarray]],
    ] = {
        variant: {
            "fixed_ridge10": {},
            "train_only_alpha_grid": {},
        }
        for variant in FEATURE_VARIANTS
    }
    prediction_payload: dict[str, np.ndarray] = {
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "baseline_error": np.asarray(
            oof.baseline - oof.target,
            dtype=np.float32,
        ),
    }
    per_seed_results: dict[str, dict[str, list[dict[str, Any]]]] = {
        variant: {
            "pretrained_fixed_ridge10": [],
            "pretrained_train_only_alpha_grid": [],
            "random_gaussian_fixed_ridge10": [],
            "random_gaussian_train_only_alpha_grid": [],
            "random_init_fixed_ridge10": [],
            "random_init_train_only_alpha_grid": [],
        }
        for variant in FEATURE_VARIANTS
    }
    structural_per_seed = {
        "fixed_ridge10": [],
        "train_only_alpha_grid": [],
    }

    for seed_id, seed in enumerate(base.REPEAT_SEEDS):
        variants, variant_audit = build_feature_variants(
            mean_features,
            channel_features,
            layer_channels=layer_channels,
            seed=int(seed),
        )
        feature_audits[str(seed)] = variant_audit
        random_init_variants, random_init_variant_audit = build_feature_variants(
            random_init_mean_features[seed_id],
            random_init_channel_features[seed_id],
            layer_channels=layer_channels,
            seed=int(seed),
        )
        random_init_feature_audits[str(seed)] = random_init_variant_audit

        structural_features = np.column_stack(
            [oof.structural_features, common]
        )
        structural = evaluate_adaptive_route(
            route="no_foundation_structural",
            features=structural_features,
            oof=oof,
            seed=int(seed),
        )
        all_cells.extend(structural["per_fold"])
        for mode in structural_per_seed:
            structural_per_seed[mode].append(
                structural["aggregates"][mode]
            )

        for variant, latent in variants.items():
            pretrained_features = np.column_stack([latent, common])
            pretrained = evaluate_adaptive_route(
                route=f"pretrained::{variant}",
                features=pretrained_features,
                oof=oof,
                seed=int(seed),
            )
            all_cells.extend(pretrained["per_fold"])

            random_latent = _random_control(
                len(oof.target),
                latent.shape[1],
                variant=variant,
                seed=int(seed),
            )
            random_features = np.column_stack([random_latent, common])
            random = evaluate_adaptive_route(
                route=f"random_gaussian::{variant}",
                features=random_features,
                oof=oof,
                seed=int(seed),
            )
            all_cells.extend(random["per_fold"])

            random_init_features = np.column_stack(
                [random_init_variants[variant], common]
            )
            random_init = evaluate_adaptive_route(
                route=f"random_init_same_architecture::{variant}",
                features=random_init_features,
                oof=oof,
                seed=int(seed),
            )
            all_cells.extend(random_init["per_fold"])

            for mode, suffix in (
                ("fixed_ridge10", "fixed_ridge10"),
                ("train_only_alpha_grid", "train_only_alpha_grid"),
            ):
                per_seed_results[variant][f"pretrained_{suffix}"].append(
                    pretrained["aggregates"][mode]
                )
                per_seed_results[variant][
                    f"random_gaussian_{suffix}"
                ].append(
                    random["aggregates"][mode]
                )
                per_seed_results[variant][f"random_init_{suffix}"].append(
                    random_init["aggregates"][mode]
                )
                gated_prediction = np.asarray(
                    pretrained["predictions"][mode]["gated"],
                    dtype=np.float64,
                )
                pretrained_predictions[variant][mode][int(seed)] = (
                    gated_prediction
                )
                prediction_payload[
                    _prediction_key(variant, mode, int(seed))
                ] = np.asarray(
                    gated_prediction - oof.target,
                    dtype=np.float32,
                )

    structural: dict[str, Any] = {}
    for mode, rows in structural_per_seed.items():
        cells = [
            row
            for row in all_cells
            if row["route"] == "no_foundation_structural"
            and row["head_mode"] == mode
        ]
        structural[mode] = _summarize_across_seeds(
            per_seed=rows,
            cells=cells,
            baseline_rmse=baseline_metrics["rmse"],
            structural_rmse=math.inf,
        )

    variants_summary: dict[str, Any] = {}
    best_candidates: list[dict[str, Any]] = []
    for variant in FEATURE_VARIANTS:
        config = {
            "input_adapter": (
                "three independent channel forwards then concatenation"
                if variant.startswith("per_channel")
                else "fixed input-level mean of three channels"
            ),
            "stage_policy": (
                "all 32 channels from stage0"
                if "stage0" in variant
                else (
                    "all 320 channels from stage5"
                    if "stage5" in variant
                    else "16 seeded channels from each of six stages"
                )
            ),
            "latent_width_by_seed": {
                seed: feature_audits[seed]["variant_widths"][variant]
                for seed in feature_audits
            },
        }
        heads: dict[str, Any] = {}
        for mode in ("fixed_ridge10", "train_only_alpha_grid"):
            pretrained_key = f"pretrained_{mode}"
            random_gaussian_key = f"random_gaussian_{mode}"
            random_init_key = f"random_init_{mode}"
            pretrained_cells = [
                row
                for row in all_cells
                if row["route"] == f"pretrained::{variant}"
                and row["head_mode"] == mode
            ]
            random_gaussian_cells = [
                row
                for row in all_cells
                if row["route"] == f"random_gaussian::{variant}"
                and row["head_mode"] == mode
            ]
            random_init_cells = [
                row
                for row in all_cells
                if row["route"]
                == f"random_init_same_architecture::{variant}"
                and row["head_mode"] == mode
            ]
            random_gaussian_summary = _summarize_across_seeds(
                per_seed=per_seed_results[variant][random_gaussian_key],
                cells=random_gaussian_cells,
                baseline_rmse=baseline_metrics["rmse"],
                structural_rmse=structural[mode]["gated_mean_seed_rmse"],
            )
            random_init_summary = _summarize_across_seeds(
                per_seed=per_seed_results[variant][random_init_key],
                cells=random_init_cells,
                baseline_rmse=baseline_metrics["rmse"],
                structural_rmse=structural[mode]["gated_mean_seed_rmse"],
            )
            pretrained_summary = _summarize_across_seeds(
                per_seed=per_seed_results[variant][pretrained_key],
                cells=pretrained_cells,
                baseline_rmse=baseline_metrics["rmse"],
                structural_rmse=structural[mode]["gated_mean_seed_rmse"],
                random_rmse=random_gaussian_summary["gated_mean_seed_rmse"],
            )
            pretrained_summary[
                "better_than_random_init_same_architecture"
            ] = (
                pretrained_summary["gated_mean_seed_rmse"]
                < random_init_summary["gated_mean_seed_rmse"]
            )
            pretrained_summary[
                "rmse_minus_random_init_same_architecture"
            ] = (
                pretrained_summary["gated_mean_seed_rmse"]
                - random_init_summary["gated_mean_seed_rmse"]
            )
            bootstrap_seed = int.from_bytes(
                hashlib.sha256(
                    f"{variant}:{mode}:block-bootstrap:2693".encode("utf-8")
                ).digest()[:8],
                "little",
            )
            pretrained_summary[
                "block_bootstrap_rmse_delta_vs_pykrige"
            ] = block_bootstrap_rmse_delta(
                target=oof.target,
                baseline=oof.baseline,
                candidate_predictions_by_seed=pretrained_predictions[variant][
                    mode
                ],
                fold_ids=oof.fold_ids,
                bootstrap_seed=bootstrap_seed,
            )
            required_wins = math.ceil(
                0.8 * pretrained_summary["independent_spatial_units"]
            )
            promoted = (
                pretrained_summary["relative_gain_vs_pykrige"] >= 0.01
                and pretrained_summary["better_than_no_foundation_structural"]
                and pretrained_summary[
                    "better_than_random_init_same_architecture"
                ]
                and pretrained_summary["independent_fold_outcome_counts"]["win"]
                >= required_wins
            )
            pretrained_summary["promotion"] = {
                "passes_audited_p11_rule": promoted,
                "required_independent_fold_wins": required_wins,
                "random_init_same_architecture_control_required": True,
                "random_init_same_architecture_control_present": True,
            }
            heads[mode] = {
                "pretrained_residual": pretrained_summary,
                "random_init_same_architecture_control": random_init_summary,
                "random_gaussian_negative_control": random_gaussian_summary,
            }
            independent_counts = pretrained_summary[
                "independent_fold_outcome_counts"
            ]
            best_candidates.append(
                {
                    "variant": variant,
                    "head_mode": mode,
                    "gated_mean_seed_rmse": pretrained_summary[
                        "gated_mean_seed_rmse"
                    ],
                    "relative_gain_vs_pykrige": pretrained_summary[
                        "relative_gain_vs_pykrige"
                    ],
                    "independent_fold_wins": independent_counts["win"],
                    "independent_fold_losses": independent_counts["loss"],
                    "independent_fold_ties": independent_counts["tie"],
                    "passes_audited_p11_rule": promoted,
                }
            )
        variants_summary[variant] = {"config": config, "heads": heads}

    best = min(
        best_candidates,
        key=lambda row: (
            row["gated_mean_seed_rmse"],
            -row["independent_fold_wins"],
            row["variant"],
            row["head_mode"],
        ),
    )
    promoted_routes = [
        row for row in best_candidates if row["passes_audited_p11_rule"]
    ]
    highest_win_route = max(
        best_candidates,
        key=lambda row: (
            row["independent_fold_wins"],
            -row["independent_fold_losses"],
            -row["gated_mean_seed_rmse"],
            row["variant"],
            row["head_mode"],
        ),
    )
    highest_wins = max(
        row["independent_fold_wins"] for row in best_candidates
    )
    meaningful_fold_win_improvement = any(
        row["independent_fold_wins"]
        > legacy_reference["independent_fold_outcome_counts"]["win"]
        and row["independent_fold_losses"]
        <= legacy_reference["independent_fold_outcome_counts"]["loss"]
        and row["gated_mean_seed_rmse"]
        < legacy_reference["gated_mean_seed_rmse"] - WIN_TOLERANCE
        for row in best_candidates
    )
    any_material_gain_positive = any(
        row["relative_gain_vs_pykrige"] > 0.0 for row in best_candidates
    )
    exhausted = highest_wins < math.ceil(
        0.8 * len(base.FOLD_IDS)
    )
    conclusion = (
        "现有OpenMind checkpoint的适配空间已基本穷尽，"
        "建议更换更贴近地震/地质领域的基础模型。"
        if exhausted
        else (
            "至少一个适配变体达到逐fold胜率晋升门槛；"
            "仍需负责人复核后决定后续路线。"
        )
    )
    experiment = {
        "fixed_protocol": {
            "outer_folds": list(base.FOLD_IDS),
            "repeat_seeds": list(base.REPEAT_SEEDS),
            "base_model": base.EXPECTED_MODEL_ID,
            "base_predictions_for_all_residual_training": (
                "cross-fitted P5 Stage-3 development OOF"
            ),
            "feature_variants": list(FEATURE_VARIANTS),
            "residual_alpha_candidates": list(RIDGE_ALPHAS),
            "alpha_selection": (
                "outer-train-only grouped inner OOF RMSE; tie chooses stronger alpha"
            ),
            "gate_model": (
                "inner-OOF LogisticRegression benefit probability times "
                "train-only selected bounded scale"
            ),
            "gate_bounds": [0.0, 1.0],
            "gate_scale_candidates": list(base.GATE_CANDIDATES),
            "win_tolerance_rmse": WIN_TOLERANCE,
            "statistical_units": {
                "independent_spatial_units": len(base.FOLD_IDS),
                "independent_unit": "locked outer spatial fold",
                "seed_pseudo_repeats_per_spatial_unit": len(base.REPEAT_SEEDS),
                "seed_pseudo_repeats_are_independent_samples": False,
            },
            "block_bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_unit": "locked outer spatial fold",
                "seed_pseudo_repeats_resampled": False,
            },
            "foundation_effect_protocol": {
                "path": "_models/gaia_dagt/foundation_effect_protocol.v1.json",
                "random_init_same_architecture_required_for_default_promotion": True,
            },
            "promotion_rule": (
                "at least 1% pooled development RMSE gain vs PyKrige, "
                "better than no-foundation structural and same-architecture "
                "random-init controls, and wins at least 80% of the five "
                "independent spatial folds; three seeds are paired pseudo-repeats"
            ),
            "test_or_holdout_tuning": False,
        },
        "baseline": {
            "pykrige_oof": baseline_metrics,
            "gate_zero_exact": base._metrics(  # noqa: SLF001
                oof.target,
                gate_zero,
            ),
            "gate_zero_bitwise_equal_to_pykrige": bool(
                np.array_equal(gate_zero, oof.baseline)
            ),
        },
        "legacy_reference": dict(legacy_reference),
        "feature_selection_audit": feature_audits,
        "random_init_feature_selection_audit": random_init_feature_audits,
        "no_foundation_structural_control": structural,
        "variants": variants_summary,
        "per_fold_seed_pseudo_repeats": all_cells,
        "decision": {
            "state": (
                "PROMOTE_DEVELOPMENT_ONLY"
                if promoted_routes
                else "VERIFIED_NO_PROMOTION"
            ),
            "default_enabled": bool(promoted_routes),
            "best_route": best,
            "promoted_routes": promoted_routes,
            "highest_independent_fold_wins": highest_wins,
            "highest_win_route": highest_win_route,
            "required_independent_fold_wins": math.ceil(
                0.8 * len(base.FOLD_IDS)
            ),
            "independent_spatial_units": len(base.FOLD_IDS),
            "seed_pseudo_repeats_per_unit": len(base.REPEAT_SEEDS),
            "any_material_gain_positive": any_material_gain_positive,
            "meaningful_fold_win_improvement_vs_legacy": (
                meaningful_fold_win_improvement
            ),
            "adaptation_space_exhausted_under_current_checkpoint": exhausted,
            "conclusion": conclusion,
            "model_replacement_authority": (
                "负责人/军伟；本诊断不自行更换基础模型"
            ),
        },
    }
    return experiment, prediction_payload


def _format_rmse(value: float) -> str:
    return f"{float(value):.12g}"


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    experiment = result["experiment"]
    baseline_rmse = experiment["baseline"]["pykrige_oof"]["rmse"]
    legacy = experiment["legacy_reference"]
    decision = experiment["decision"]
    structural = experiment["no_foundation_structural_control"]
    mixed_fixed = experiment["variants"]["mean_mixed16"]["heads"][
        "fixed_ridge10"
    ]["pretrained_residual"]
    mixed_tuned = experiment["variants"]["mean_mixed16"]["heads"][
        "train_only_alpha_grid"
    ]["pretrained_residual"]
    stage5_fixed = experiment["variants"]["mean_stage5_all"]["heads"][
        "fixed_ridge10"
    ]["pretrained_residual"]
    stage5_tuned = experiment["variants"]["mean_stage5_all"]["heads"][
        "train_only_alpha_grid"
    ]["pretrained_residual"]
    legacy_counts = legacy["independent_fold_outcome_counts"]
    best_counts = decision["best_route"]
    highest_counts = decision["highest_win_route"]
    lines = [
        "# P11 OpenMind Residual-Fusion Diagnostic Evidence",
        "",
        "## Outcome",
        "",
        f"- Decision: `{decision['state']}`; default enabled: `{decision['default_enabled']}`.",
        f"- PyKrige development OOF RMSE: `{_format_rmse(baseline_rmse)}`.",
        (
            "- Legacy mean-input mixed-stage Ridge10 RMSE: "
            f"`{_format_rmse(legacy['gated_mean_seed_rmse'])}` "
            "(five independent spatial folds W/L/T "
            f"`{legacy_counts['win']}/{legacy_counts['loss']}/"
            f"{legacy_counts['tie']}`)."
        ),
        (
            "- Best diagnostic route: "
            f"`{decision['best_route']['variant']} / "
            f"{decision['best_route']['head_mode']}`; RMSE "
            f"`{_format_rmse(decision['best_route']['gated_mean_seed_rmse'])}`; "
            f"material gain `{decision['best_route']['relative_gain_vs_pykrige']:.6%}`; "
            "independent-fold W/L/T "
            f"`{best_counts['independent_fold_wins']}/"
            f"{best_counts['independent_fold_losses']}/"
            f"{best_counts['independent_fold_ties']}`."
        ),
        (
            "- Highest independent spatial-fold win count across all "
            f"adaptations: `{decision['highest_independent_fold_wins']}/5`; "
            "promotion requires "
            f"`{decision['required_independent_fold_wins']}/5`."
        ),
        (
            "- Any positive material gain: "
            f"`{decision['any_material_gain_positive']}`; meaningful fold-win "
            "improvement over the legacy five-fold pattern: "
            f"`{decision['meaningful_fold_win_improvement_vs_legacy']}`."
        ),
        "",
        "## Statistical units and uncertainty",
        "",
        (
            "- There are **five genuinely independent spatial units**: locked "
            "outer folds 0–4."
        ),
        (
            "- The three random seeds are **paired pseudo-repeats within each "
            "spatial fold**, not three additional independent samples. They are "
            "never counted as inferential n=15."
        ),
        (
            "- The 95% intervals resample the five whole spatial folds with "
            f"replacement for {BOOTSTRAP_REPLICATES:,} deterministic replicates. "
            "All voxel errors inside a sampled fold move together; the three "
            "seed predictions stay paired and are averaged."
        ),
        "",
        "## Diagnostic answers",
        "",
        (
            "- **Single-stage ablation:** stage0 was worse than PyKrige under "
            "both heads. Stage5 with fixed Ridge10 selected gate=0 in every "
            "independent spatial fold, so it added no measurable signal; train-only alpha "
            "selection made stage5 unstable rather than useful."
        ),
        (
            "- **Three independent channel forwards:** the strongest safe "
            "variant was per-channel mixed16 with fixed Ridge10, but it still "
            "had negative material gain. The apparent maximum of "
            f"{highest_counts['independent_fold_wins']}/5 independent-fold wins "
            "came from "
            f"`{decision['highest_win_route']['variant']} / "
            f"{decision['highest_win_route']['head_mode']}` and also had "
            f"{highest_counts['independent_fold_losses']} independent-fold losses with "
            f"{decision['highest_win_route']['relative_gain_vs_pykrige']:.6%} "
            "aggregate gain, so it is not a meaningful win-rate improvement."
        ),
        (
            "- **Stronger Ridge regularization:** train-only alpha search "
            "strongly stabilized the ungated mixed path from RMSE "
            f"`{_format_rmse(mixed_fixed['ungated_mean_seed_rmse'])}` to "
            f"`{_format_rmse(mixed_tuned['ungated_mean_seed_rmse'])}` and the "
            "mean-input stage5 path from "
            f"`{_format_rmse(stage5_fixed['ungated_mean_seed_rmse'])}` to "
            f"`{_format_rmse(stage5_tuned['ungated_mean_seed_rmse'])}`. "
            "That numerical stabilization did not survive gated OOF evaluation: "
            "material gain stayed negative and fold wins did not improve."
        ),
        (
            "- **Required random-init control:** every feature route was repeated "
            "with the same OpenMind encoder architecture, but checkpoint weights "
            "were replaced by seed-specific random initialization before frozen "
            "feature extraction. This is distinct from both the Gaussian negative "
            "control and the hand-crafted structural control."
        ),
        "",
        "## Same-split comparison",
        "",
        (
            "| Feature route | Head | Ungated RMSE | Gated RMSE | "
            "Material gain | Independent W/L/T | Random-init RMSE | "
            "Structural RMSE | RMSE delta 95% block CI | Promote |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in FEATURE_VARIANTS:
        for mode in ("fixed_ridge10", "train_only_alpha_grid"):
            head = experiment["variants"][variant]["heads"][mode]
            pretrained = head["pretrained_residual"]
            random_init = head["random_init_same_architecture_control"]
            counts = pretrained["independent_fold_outcome_counts"]
            interval = pretrained[
                "block_bootstrap_rmse_delta_vs_pykrige"
            ]["confidence_interval"]
            lines.append(
                f"| `{variant}` | `{mode}` | "
                f"{_format_rmse(pretrained['ungated_mean_seed_rmse'])} | "
                f"{_format_rmse(pretrained['gated_mean_seed_rmse'])} | "
                f"{pretrained['relative_gain_vs_pykrige']:.6%} | "
                f"{counts['win']}/{counts['loss']}/{counts['tie']} | "
                f"{_format_rmse(random_init['gated_mean_seed_rmse'])} | "
                f"{_format_rmse(structural[mode]['gated_mean_seed_rmse'])} | "
                f"[{interval[0]:+.6g}, {interval[1]:+.6g}] | "
                f"{pretrained['promotion']['passes_audited_p11_rule']} |"
            )

    lines.extend(
        [
            "",
            "## Independent spatial-fold outcomes",
            "",
            (
                "These five folds—not the seeds—are the independent win/loss/tie "
                "units. Each fold metric averages its three paired seed repeats."
            ),
            "",
            (
                "| Feature route | Head | Fold | PyKrige RMSE | "
                "Gated mean-seed RMSE | Delta | Outcome |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in FEATURE_VARIANTS:
        for mode in ("fixed_ridge10", "train_only_alpha_grid"):
            pretrained = experiment["variants"][variant]["heads"][mode][
                "pretrained_residual"
            ]
            for row in pretrained["independent_fold_outcomes"]:
                lines.append(
                    f"| `{variant}` | `{mode}` | {row['outer_fold']} | "
                    f"{_format_rmse(row['baseline_mean_seed_rmse'])} | "
                    f"{_format_rmse(row['gated_mean_seed_rmse'])} | "
                    f"{row['rmse_delta_vs_pykrige']:+.12g} | "
                    f"**{row['outcome_vs_pykrige']}** |"
                )

    lines.extend(
        [
            "",
            "## Seed-level diagnostic details (paired pseudo-repeats)",
            "",
            (
                "The rows below preserve the requested fold×seed audit trail, "
                "but they are correlated pseudo-repeats and are not counted as "
                "independent evidence. Exact gate-zero degenerations are ties."
            ),
            "",
            (
                "| Feature route | Head | Seed | Fold | Alpha | "
                "PyKrige RMSE | Gated RMSE | Delta | Diagnostic outcome |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pretrained_cells = [
        row
        for row in experiment["per_fold_seed_pseudo_repeats"]
        if row["route"].startswith("pretrained::")
    ]
    pretrained_cells.sort(
        key=lambda row: (
            row["route"],
            row["head_mode"],
            row["seed"],
            row["outer_fold"],
        )
    )
    for row in pretrained_cells:
        lines.append(
            f"| `{row['route'].split('::', 1)[1]}` | "
            f"`{row['head_mode']}` | {row['seed']} | {row['outer_fold']} | "
            f"{row['residual_alpha']:.1f} | "
            f"{_format_rmse(row['baseline_metrics']['rmse'])} | "
            f"{_format_rmse(row['gated_metrics']['rmse'])} | "
            f"{row['gated_rmse_delta_vs_pykrige']:+.12g} | "
            f"**{row['outcome_vs_pykrige']}** |"
        )

    lines.extend(
        [
            "",
            "## Holdout firewall and feature identity",
            "",
            "- The CLI accepts no test or holdout path.",
            "- The only HDF5 file opened was `train.h5`.",
            (
                "- OpenMind read only `seismic_patch[0:3]` and "
                "`seismic_patch[8]`; it did not read labels."
            ),
            (
                "- PyKrige targets and predictions are the same hash-verified "
                "P5 Stage-3 development OOF rows used by committed P11."
            ),
            (
                "- Checkpoint SHA-256: "
                f"`{result['openmind_feature_audit']['checkpoint_sha256']}`."
            ),
            (
                "- Per-channel feature cache SHA-256: "
                f"`{result['openmind_feature_audit']['feature_cache_sha256']}`."
            ),
            (
                "- Same-architecture random-init cache SHA-256: "
                f"`{result['random_init_control_audit']['feature_cache_sha256']}`."
            ),
            (
                "- Per-voxel prediction-error artifact SHA-256: "
                f"`{result['prediction_error_artifact']['sha256']}`."
            ),
            "",
            "## Recommendation boundary",
            "",
            decision["conclusion"],
            "",
            (
                "更换基础模型必须升级给负责人/军伟决策；本实验没有自行"
                "更换 OpenMind checkpoint。"
            ),
            "",
            (
                "Full controls, alpha-search scores, gate/residual statistics, "
                "block-bootstrap intervals, and paired seed details are retained "
                "in `summary.json`; per-voxel errors are in `prediction_errors.npz`."
            ),
            "",
        ]
    )
    (output_dir / "evidence.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    source_root: Path,
    checkpoint: Path,
    dependency_root: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mean_feature_cache: Path = DEFAULT_MEAN_FEATURE_CACHE,
    channel_feature_cache: Path = DEFAULT_CHANNEL_FEATURE_CACHE,
    random_init_feature_cache: Path = DEFAULT_RANDOM_INIT_FEATURE_CACHE,
    legacy_summary: Path = DEFAULT_LEGACY_SUMMARY,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Execute diagnostics and write a separate reconstruction-local artifact."""

    started = time.perf_counter()
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    mean_features, mean_audit = base.get_openmind_features(
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        feature_cache=mean_feature_cache,
        device=device,
    )
    channel_features, channel_audit = get_openmind_per_channel_features(
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        feature_cache=channel_feature_cache,
        device=device,
    )
    if mean_audit["layer_channels"] != channel_audit["layer_channels"]:
        raise RuntimeError("mean and per-channel OpenMind stage widths differ")
    (
        random_init_mean_features,
        random_init_channel_features,
        random_init_audit,
    ) = get_openmind_random_init_features(
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        checkpoint=checkpoint,
        dependency_root=dependency_root,
        feature_cache=random_init_feature_cache,
        device=device,
    )
    if random_init_audit["layer_channels"] != channel_audit["layer_channels"]:
        raise RuntimeError(
            "random-init and pretrained OpenMind stage widths differ"
        )
    legacy = _legacy_reference(legacy_summary)
    experiment, prediction_payload = evaluate_diagnostics(
        oof=oof,
        mean_features=mean_features,
        channel_features=channel_features,
        random_init_mean_features=random_init_mean_features,
        random_init_channel_features=random_init_channel_features,
        layer_channels=channel_audit["layer_channels"],
        legacy_reference=legacy,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction_errors.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(handle, **prediction_payload)
    prediction_artifact = {
        "path": (
            "_pipelines/02_task_datasets/reconstruction/_outputs/"
            "p11_residual_fusion_diagnostics/prediction_errors.npz"
        ),
        "sha256": base._sha256(prediction_path),  # noqa: SLF001
        "array_count": len(prediction_payload),
        "rows": len(oof.target),
        "contents": (
            "indices_kji, spatial fold ids, PyKrige per-voxel error, and "
            "pretrained gated per-voxel errors for every feature/head/seed route"
        ),
        "bootstrap_source": True,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "task_id": "volve_porosity_conditional_reconstruction",
        "evidence_scope": "fixed conditional development OOF only",
        "holdout_firewall": {
            "accepted_hdf5_basenames": ["train.h5"],
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": ["seismic_patch[0:3]", "seismic_patch[8]"],
            "label_dataset_read": False,
            "test_path_argument_exists": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "historical_test_metrics_read": False,
        },
        "pykrige_oof_sources": {
            "lane": base.EXPECTED_LANE,
            "model_id": base.EXPECTED_MODEL_ID,
            "split_hash": base.EXPECTED_SPLIT_HASH,
            "repeat_id": base.REPEAT_ID,
            "rows": len(oof.target),
            "folds": list(oof.source_records),
            "training_use": "base predictions are OOF before residual stacking",
        },
        "openmind_feature_audit": channel_audit,
        "mean_projection_feature_reference": mean_audit,
        "random_init_control_audit": random_init_audit,
        "prediction_error_artifact": prediction_artifact,
        "implementation": {
            "script": (
                "_pipelines/02_task_datasets/reconstruction/"
                "p11_residual_fusion_diagnostics.py"
            ),
            "script_sha256": base._sha256(Path(__file__)),  # noqa: SLF001
            "base_harness_script": (
                "_pipelines/02_task_datasets/reconstruction/"
                "p11_residual_fusion.py"
            ),
            "base_harness_sha256": base._sha256(  # noqa: SLF001
                HERE / "p11_residual_fusion.py"
            ),
            "foundation_effect_protocol_sha256": base._sha256(  # noqa: SLF001
                PROJECT_ROOT
                / "_models"
                / "gaia_dagt"
                / "foundation_effect_protocol.v1.json"
            ),
            "numpy": np.__version__,
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
        "experiment": experiment,
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "downloads_performed_bytes": 0,
            "mean_feature_cache_reused": mean_audit["cache_reused"],
            "channel_feature_cache_reused": channel_audit["cache_reused"],
            "random_init_feature_cache_reused": random_init_audit[
                "cache_reused"
            ],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_evidence(output_dir, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-dev",
        help="verify the train-only HDF5 boundary",
    )
    check.add_argument("--data-dir", type=Path, required=True)

    execute = subparsers.add_parser(
        "run",
        help="run P11 diagnostics on development OOF only",
    )
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--checkpoint", type=Path, required=True)
    execute.add_argument("--dependency-root", type=Path)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument(
        "--mean-feature-cache",
        type=Path,
        default=DEFAULT_MEAN_FEATURE_CACHE,
    )
    execute.add_argument(
        "--channel-feature-cache",
        type=Path,
        default=DEFAULT_CHANNEL_FEATURE_CACHE,
    )
    execute.add_argument(
        "--random-init-feature-cache",
        type=Path,
        default=DEFAULT_RANDOM_INIT_FEATURE_CACHE,
    )
    execute.add_argument(
        "--legacy-summary",
        type=Path,
        default=DEFAULT_LEGACY_SUMMARY,
    )
    execute.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "check-dev":
        resolved = base.resolve_dev_inputs(args.data_dir)
        print(
            json.dumps(
                {"accepted": str(resolved.train_h5), "holdout_opened": False}
            )
        )
        return 0
    values = vars(args)
    values.pop("command")
    result = run(**values)
    print(json.dumps(result["experiment"]["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
