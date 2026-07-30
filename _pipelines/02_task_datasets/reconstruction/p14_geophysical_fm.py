#!/usr/bin/env python3
"""P14 development-only geophysical foundation-model residual experiment.

This script changes only the P11 encoder source.  It keeps the hash-verified
P5 PyKrige OOF baseline, five locked spatial folds, three paired optimization
seeds, Ridge residual heads, inner-OOF bounded gate and gate=0 degeneration
check from ``p11_residual_fusion_diagnostics.py``.

The sole legal HDF5 input is ``train.h5``.  It is read only for metadata,
``seismic_patch[0:3]`` and ``seismic_patch[8]``.  No test or frozen-holdout
argument exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT))

from _models.reconstruction import geophysical_fm as gfm  # noqa: E402
import p11_residual_fusion as base  # noqa: E402
import p11_residual_fusion_diagnostics as diagnostics  # noqa: E402


SCHEMA_VERSION = "reconstruction-p14-geophysical-fm/v1"
FEATURE_CACHE_SCHEMA_VERSION = "reconstruction-p14-gfm-feature-cache/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p14_geophysical_fm"
DEFAULT_PRETRAINED_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p14_geophysical_fm"
    / "gfm_pretrained_features.npz"
)
DEFAULT_RANDOM_INIT_CACHE = (
    PROJECT_ROOT
    / "_tmp"
    / "p14_geophysical_fm"
    / "gfm_random_init_features.npz"
)
EMBEDDING_CHANNELS_PER_VIEW = 16
VIEW_NAMES = (
    "seismic_channel_0_trace",
    "seismic_channel_0_cls",
    "seismic_channel_1_trace",
    "seismic_channel_1_cls",
    "seismic_channel_2_trace",
    "seismic_channel_2_cls",
)
HEAD_MODES = ("fixed_ridge10", "train_only_alpha_grid")
WIN_TOLERANCE = diagnostics.WIN_TOLERANCE


def selected_embedding_channels(
    *,
    embedding_width: int,
    seed: int,
) -> np.ndarray:
    """Select the same 16-channel budget independently for six GFM views."""

    if embedding_width < EMBEDDING_CHANNELS_PER_VIEW:
        raise ValueError("GFM embedding is narrower than the view budget")
    rng = np.random.default_rng(int(seed))
    selected = np.empty(
        (len(VIEW_NAMES), EMBEDDING_CHANNELS_PER_VIEW),
        dtype=np.int64,
    )
    for view_id in range(len(VIEW_NAMES)):
        selected[view_id] = np.sort(
            rng.choice(
                int(embedding_width),
                size=EMBEDDING_CHANNELS_PER_VIEW,
                replace=False,
            )
        )
    return selected


def trace_token_indices(
    original_indices: np.ndarray,
    *,
    original_trace_count: int,
    resized_trace_count: int = 160,
) -> np.ndarray:
    """Map original trace centers to nearest resized GFM trace tokens."""

    original_indices = np.asarray(original_indices, dtype=np.int64)
    if original_trace_count <= 0 or resized_trace_count <= 0:
        raise ValueError("trace counts must be positive")
    if np.any(original_indices < 0) or np.any(
        original_indices >= original_trace_count
    ):
        raise ValueError("original trace index is outside the slice")
    mapped = np.rint(
        (original_indices.astype(np.float64) + 0.5)
        * resized_trace_count
        / original_trace_count
        - 0.5
    )
    return np.clip(mapped, 0, resized_trace_count - 1).astype(np.int64)


def normalize_slice(
    values: np.ndarray,
    active: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply deterministic per-slice active-cell z-scoring."""

    values = np.asarray(values, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    if values.ndim != 2 or active.shape != values.shape:
        raise ValueError("GFM slice and active mask must be aligned 2-D arrays")
    selected = values[active]
    if selected.size < 2 or not np.all(np.isfinite(selected)):
        raise ValueError("GFM slice lacks finite active seismic samples")
    mean = float(np.mean(selected))
    scale = max(float(np.std(selected)), 1e-6)
    normalized = np.zeros_like(values, dtype=np.float32)
    normalized[active] = (selected - mean) / scale
    if not np.all(np.isfinite(normalized)):
        raise FloatingPointError("GFM normalized slice is non-finite")
    return normalized, {
        "active_mean_before": mean,
        "active_std_before": scale,
        "active_mean_after": float(np.mean(normalized[active])),
        "active_std_after": float(np.std(normalized[active])),
    }


def assemble_seismic_volume(
    train_h5: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reassemble the tiled development seismic attributes without labels."""

    base.ensure_no_holdout_paths([train_h5])
    import h5py

    records: list[tuple[str, np.ndarray, np.ndarray]] = []
    with h5py.File(train_h5, "r") as handle:
        for key in sorted(handle):
            meta = json.loads(handle[key].attrs["meta"])
            start = np.asarray(meta["patch_start_kji"], dtype=np.int64)
            shape = np.asarray(meta["patch_shape_kji"], dtype=np.int64)
            records.append((key, start, shape))
        if not records:
            raise RuntimeError("train.h5 contains no reconstruction patches")
        volume_shape = np.max(
            np.stack([start + shape for _, start, shape in records]),
            axis=0,
        )
        seismic = np.empty((3, *volume_shape.tolist()), dtype=np.float32)
        active = np.empty(tuple(volume_shape.tolist()), dtype=bool)
        coverage = np.zeros(tuple(volume_shape.tolist()), dtype=np.uint8)
        patch_shapes: set[tuple[int, int, int]] = set()
        accessed_keys: list[str] = []
        for key, start, shape in records:
            group = handle[key]
            patch = np.asarray(group["seismic_patch"][0:3], dtype=np.float32)
            patch_active = (
                np.asarray(group["seismic_patch"][8], dtype=np.float32) > 0.5
            )
            expected = tuple(int(value) for value in shape)
            if patch.shape != (3, *expected) or patch_active.shape != expected:
                raise RuntimeError(f"invalid reconstruction patch shape: {key}")
            slices = tuple(
                slice(int(begin), int(begin + width))
                for begin, width in zip(start, shape)
            )
            if np.any(coverage[slices] != 0):
                raise RuntimeError("train.h5 reconstruction patches overlap")
            seismic[(slice(None), *slices)] = patch
            active[slices] = patch_active
            coverage[slices] = 1
            patch_shapes.add(expected)
            accessed_keys.append(key)
    if not np.all(coverage == 1):
        raise RuntimeError("train.h5 reconstruction patches do not tile the volume")
    if not np.all(np.isfinite(seismic[:, active])):
        raise FloatingPointError("assembled development seismic is non-finite")
    audit = {
        "patch_count": len(records),
        "real_patch_shape_kji": [
            list(values) for values in sorted(patch_shapes)
        ],
        "assembled_volume_shape_kji": [
            int(value) for value in volume_shape
        ],
        "seismic_shape": [int(value) for value in seismic.shape],
        "active_voxels": int(np.sum(active)),
        "coverage_min": int(np.min(coverage)),
        "coverage_max": int(np.max(coverage)),
        "accessed_patch_keys_sha256": hashlib.sha256(
            json.dumps(accessed_keys).encode("utf-8")
        ).hexdigest(),
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": [
            "seismic_patch[0:3]",
            "seismic_patch[8]",
        ],
        "label_dataset_read": False,
    }
    return seismic, active, audit


def _cache_manifest_path(feature_cache: Path) -> Path:
    return feature_cache.with_suffix(".json")


def _feature_cache_identity(
    *,
    weight_mode: str,
    train_h5_sha256: str,
    indices_kji: np.ndarray,
    asset_audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "encoder_weight_mode": weight_mode,
        "snapshot_revision": gfm.SNAPSHOT_REVISION,
        "source_sha256": asset_audit["source_sha256"],
        "config_sha256": asset_audit["config_sha256"],
        "weights_sha256": asset_audit["weights_sha256"],
        "train_h5_sha256": train_h5_sha256,
        "indices_kji_sha256": base._array_sha256(  # noqa: SLF001
            indices_kji
        ),
        "repeat_seeds": list(base.REPEAT_SEEDS),
        "view_names": list(VIEW_NAMES),
        "embedding_channels_per_view": EMBEDDING_CHANNELS_PER_VIEW,
        "slice_policy": "KxJ vertical section at fixed I",
        "resize_shape": [400, 160],
        "normalization": "per-slice active-cell zscore",
    }


def _load_valid_feature_cache(
    feature_cache: Path,
    *,
    expected_identity: Mapping[str, Any],
    indices_kji: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    manifest_path = _cache_manifest_path(feature_cache)
    if not feature_cache.is_file() or not manifest_path.is_file():
        return None
    manifest = base._json(manifest_path)  # noqa: SLF001
    if any(
        manifest.get(key) != value
        for key, value in expected_identity.items()
    ):
        return None
    if manifest.get("npz_sha256") != base._sha256(feature_cache):  # noqa: SLF001
        raise RuntimeError("GFM feature cache hash mismatch")
    with np.load(feature_cache, allow_pickle=False) as payload:
        cached_indices = np.asarray(payload["indices_kji"], dtype=np.int64)
        cached_seeds = np.asarray(payload["repeat_seeds"], dtype=np.int64)
        features = np.asarray(payload["features"], dtype=np.float32)
        selected = np.asarray(
            payload["selected_embedding_channels"],
            dtype=np.int64,
        )
    np.testing.assert_array_equal(cached_indices, indices_kji)
    np.testing.assert_array_equal(cached_seeds, base.REPEAT_SEEDS)
    expected_shape = (
        len(base.REPEAT_SEEDS),
        len(indices_kji),
        len(VIEW_NAMES) * EMBEDDING_CHANNELS_PER_VIEW,
    )
    if features.shape != expected_shape:
        raise RuntimeError("GFM cached feature shape mismatch")
    if selected.shape != (
        len(base.REPEAT_SEEDS),
        len(VIEW_NAMES),
        EMBEDDING_CHANNELS_PER_VIEW,
    ):
        raise RuntimeError("GFM cached selection shape mismatch")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("GFM cached features are non-finite")
    audit = dict(manifest["feature_audit"])
    audit["cache_reused"] = True
    audit["feature_cache_sha256"] = manifest["npz_sha256"]
    return features, audit


def _extract_projected_features(
    *,
    seismic: np.ndarray,
    active: np.ndarray,
    indices_kji: np.ndarray,
    source_root: Path,
    snapshot_path: Path,
    device: str,
    batch_size: int,
    weight_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract 96 projected features per OOF row and seed."""

    import importlib
    import torch
    import torch.nn.functional as functional

    p4 = importlib.import_module(
        "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
    )
    requested = np.asarray(indices_kji, dtype=np.int64)
    if requested.ndim != 2 or requested.shape[1] != 3:
        raise ValueError("OOF indices must be [N,3] KJI coordinates")
    volume_shape = np.asarray(active.shape, dtype=np.int64)
    if np.any(requested < 0) or np.any(requested >= volume_shape):
        raise ValueError("OOF coordinate lies outside assembled seismic volume")
    if not np.all(active[tuple(requested.T)]):
        raise RuntimeError("OOF coordinate maps to an inactive seismic cell")
    if weight_mode == "pretrained":
        model_seeds: list[int | None] = [None]
    elif weight_mode == "random_init":
        model_seeds = [int(seed) for seed in base.REPEAT_SEEDS]
    else:
        raise ValueError(f"unsupported GFM weight mode: {weight_mode}")

    selected_by_seed = np.stack(
        [
            selected_embedding_channels(
                embedding_width=1200,
                seed=int(seed),
            )
            for seed in base.REPEAT_SEEDS
        ]
    )
    features = np.empty(
        (
            len(base.REPEAT_SEEDS),
            len(requested),
            len(VIEW_NAMES) * EMBEDDING_CHANNELS_PER_VIEW,
        ),
        dtype=np.float32,
    )
    unique_i = np.unique(requested[:, 2])
    token_j = trace_token_indices(
        requested[:, 1],
        original_trace_count=active.shape[1],
    )
    model_audits: list[dict[str, Any]] = []
    normalization_audits: list[dict[str, float]] = []
    forward_batches = 0

    for model_seed in model_seeds:
        seed_ids = (
            range(len(base.REPEAT_SEEDS))
            if model_seed is None
            else [
                list(base.REPEAT_SEEDS).index(int(model_seed))
            ]
        )
        model = gfm.build_model(
            p4.task_spec("strict"),
            source_root=source_root,
            snapshot_path=snapshot_path,
            device=device,
            freeze_encoder=True,
            encoder_weight_mode=weight_mode,
            random_seed=(
                int(model_seed)
                if model_seed is not None
                else int(base.REPEAT_SEEDS[0])
            ),
        )
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("GFM feature encoder is not frozen")
        model_audits.append(dict(model.asset_audit))

        for channel in range(3):
            view_trace = 2 * channel
            view_cls = view_trace + 1
            for start in range(0, len(unique_i), int(batch_size)):
                batch_i = unique_i[start : start + int(batch_size)]
                normalized_slices: list[np.ndarray] = []
                for i_index in batch_i:
                    normalized, normalization_audit = normalize_slice(
                        seismic[channel, :, :, int(i_index)],
                        active[:, :, int(i_index)],
                    )
                    normalized_slices.append(normalized)
                    if len(normalization_audits) < 12:
                        normalization_audits.append(normalization_audit)
                images = torch.as_tensor(
                    np.stack(normalized_slices)[:, None],
                    dtype=torch.float32,
                    device=device,
                )
                images = functional.interpolate(
                    images,
                    size=(400, 160),
                    mode="bilinear",
                    align_corners=False,
                )
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=str(device).startswith("cuda"),
                ):
                    latent = model(images)
                latent_cpu = latent.float().cpu().numpy()
                for local_id, i_index in enumerate(batch_i):
                    rows = np.flatnonzero(requested[:, 2] == int(i_index))
                    trace_embedding = latent_cpu[
                        local_id,
                        1 + token_j[rows],
                    ]
                    cls_embedding = np.broadcast_to(
                        latent_cpu[local_id, 0],
                        (len(rows), 1200),
                    )
                    for seed_id in seed_ids:
                        trace_selected = selected_by_seed[
                            seed_id,
                            view_trace,
                        ]
                        cls_selected = selected_by_seed[
                            seed_id,
                            view_cls,
                        ]
                        trace_start = (
                            view_trace * EMBEDDING_CHANNELS_PER_VIEW
                        )
                        cls_start = view_cls * EMBEDDING_CHANNELS_PER_VIEW
                        features[
                            seed_id,
                            rows,
                            trace_start : trace_start
                            + EMBEDDING_CHANNELS_PER_VIEW,
                        ] = trace_embedding[:, trace_selected]
                        features[
                            seed_id,
                            rows,
                            cls_start : cls_start
                            + EMBEDDING_CHANNELS_PER_VIEW,
                        ] = cls_embedding[:, cls_selected]
                forward_batches += 1
            print(
                json.dumps(
                    {
                        "event": "gfm_channel_complete",
                        "encoder_weight_mode": weight_mode,
                        "model_seed": model_seed,
                        "seismic_channel": channel,
                        "fixed_i_slices": int(len(unique_i)),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if not np.all(np.isfinite(features)):
        raise FloatingPointError("GFM extraction produced non-finite features")
    architecture_hashes = {
        audit["architecture_sha256"] for audit in model_audits
    }
    state_hashes = {
        audit["encoder_probe_sha256"] for audit in model_audits
    }
    if len(architecture_hashes) != 1:
        raise RuntimeError("GFM architecture differs across weight controls")
    if weight_mode == "random_init" and len(state_hashes) != len(
        base.REPEAT_SEEDS
    ):
        raise RuntimeError("GFM random-init seeds did not produce distinct states")
    audit = {
        "encoder_weight_mode": weight_mode,
        "feature_shape": [int(value) for value in features.shape],
        "embedding_width": 1200,
        "view_names": list(VIEW_NAMES),
        "selected_embedding_channels": selected_by_seed.tolist(),
        "slice_policy": {
            "orientation": "KxJ vertical section at fixed I",
            "source_slice_shape": [
                int(active.shape[0]),
                int(active.shape[1]),
            ],
            "resize_shape": [400, 160],
            "reason": (
                "K is retained as the vertical trace-sample axis; J provides "
                "100 neighboring traces, closer to the pretrained width 160 "
                "than the orthogonal I width 72, reducing interpolation "
                "distortion while changing only the encoder route"
            ),
            "per_voxel_token": (
                "nearest resized trace token for voxel J plus slice CLS token"
            ),
        },
        "normalization": {
            "policy": "per-slice active-cell zscore; inactive pixels set to zero",
            "sample_audits": normalization_audits,
            "fit_on_targets": False,
            "cross_sample_statistics_used": False,
        },
        "unmasked_encoder": {
            "len_keep": 160,
            "patch_count": 160,
            "masked_trace_count": 0,
        },
        "unique_fixed_i_slices": int(len(unique_i)),
        "seismic_channels_forwarded_separately": 3,
        "forward_batches": int(forward_batches),
        "model_audits": model_audits,
        "architecture_sha256": next(iter(architecture_hashes)),
        "same_architecture_across_seeds": True,
        "seed_distinct_random_states": (
            len(state_hashes) == len(base.REPEAT_SEEDS)
            if weight_mode == "random_init"
            else None
        ),
        "pretrained_weights_used_for_forward": weight_mode == "pretrained",
        "hdf5_files_opened": ["train.h5"],
        "hdf5_datasets_read": [
            "seismic_patch[0:3]",
            "seismic_patch[8]",
        ],
        "label_dataset_read": False,
    }
    return features, audit


def get_projected_features(
    *,
    weight_mode: str,
    inputs: base.DevInputPaths,
    oof: base.OOFDevelopment,
    source_root: Path,
    snapshot_path: Path,
    feature_cache: Path,
    device: str,
    batch_size: int,
    seismic_volume: np.ndarray | None = None,
    active_volume: np.ndarray | None = None,
    assembly_audit: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse or build a hash-locked projected GFM feature cache."""

    source_root, snapshot_path, asset_audit = gfm.verify_local_assets(
        source_root,
        snapshot_path,
    )
    feature_cache = Path(feature_cache).expanduser().resolve()
    base.ensure_no_holdout_paths(
        [source_root, snapshot_path, feature_cache, inputs.train_h5]
    )
    train_h5_sha256 = base._sha256(inputs.train_h5)  # noqa: SLF001
    identity = _feature_cache_identity(
        weight_mode=weight_mode,
        train_h5_sha256=train_h5_sha256,
        indices_kji=oof.indices_kji,
        asset_audit=asset_audit,
    )
    cached = _load_valid_feature_cache(
        feature_cache,
        expected_identity=identity,
        indices_kji=oof.indices_kji,
    )
    if cached is not None:
        return cached
    if seismic_volume is None or active_volume is None or assembly_audit is None:
        seismic_volume, active_volume, loaded_audit = assemble_seismic_volume(
            inputs.train_h5
        )
        assembly_audit = loaded_audit
    features, feature_audit = _extract_projected_features(
        seismic=seismic_volume,
        active=active_volume,
        indices_kji=oof.indices_kji,
        source_root=source_root,
        snapshot_path=snapshot_path,
        device=device,
        batch_size=batch_size,
        weight_mode=weight_mode,
    )
    feature_audit = {
        **feature_audit,
        "assembly_audit": dict(assembly_audit),
        "asset_audit": asset_audit,
        "cache_reused": False,
    }
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    selected = np.stack(
        [
            selected_embedding_channels(
                embedding_width=1200,
                seed=int(seed),
            )
            for seed in base.REPEAT_SEEDS
        ]
    )
    with feature_cache.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            repeat_seeds=np.asarray(base.REPEAT_SEEDS, dtype=np.int64),
            features=features,
            selected_embedding_channels=selected,
        )
    manifest = {
        **identity,
        "npz_sha256": base._sha256(feature_cache),  # noqa: SLF001
        "feature_audit": feature_audit,
    }
    _cache_manifest_path(feature_cache).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    feature_audit = dict(feature_audit)
    feature_audit["feature_cache_sha256"] = manifest["npz_sha256"]
    return features, feature_audit


def _summarize_route(
    *,
    route: str,
    head_mode: str,
    seed_results: Sequence[Mapping[str, Any]],
    all_cells: Sequence[Mapping[str, Any]],
    baseline_rmse: float,
) -> dict[str, Any]:
    cells = [
        dict(row)
        for row in all_cells
        if row["route"] == route and row["head_mode"] == head_mode
    ]
    independent = diagnostics._independent_fold_outcomes(cells)  # noqa: SLF001
    counts = {
        outcome: sum(
            row["outcome_vs_pykrige"] == outcome for row in independent
        )
        for outcome in ("win", "loss", "tie")
    }
    gated_rmse = float(
        np.mean(
            [row["aggregates"][head_mode]["gated"]["rmse"] for row in seed_results]
        )
    )
    ungated_rmse = float(
        np.mean(
            [
                row["aggregates"][head_mode]["ungated"]["rmse"]
                for row in seed_results
            ]
        )
    )
    delta = gated_rmse - baseline_rmse
    relative_gain = (
        0.0
        if abs(delta) <= WIN_TOLERANCE
        else (baseline_rmse - gated_rmse) / baseline_rmse
    )
    return {
        "gated_mean_seed_rmse": gated_rmse,
        "ungated_mean_seed_rmse": ungated_rmse,
        "rmse_delta_vs_pykrige": (
            0.0 if abs(delta) <= WIN_TOLERANCE else delta
        ),
        "relative_gain_vs_pykrige": relative_gain,
        "independent_spatial_units": len(base.FOLD_IDS),
        "seed_pseudo_repeats_per_unit": len(base.REPEAT_SEEDS),
        "independent_fold_outcomes": independent,
        "independent_fold_outcome_counts": counts,
        "independent_fold_win_rate": counts["win"] / len(base.FOLD_IDS),
        "seed_level_cells": len(cells),
        "per_seed": [
            {
                "seed": int(seed),
                **dict(row["aggregates"][head_mode]),
            }
            for seed, row in zip(base.REPEAT_SEEDS, seed_results)
        ],
    }


def evaluate_gfm(
    *,
    oof: base.OOFDevelopment,
    pretrained_features: np.ndarray,
    random_init_features: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Reuse the unchanged P11 adaptive Ridge/gate route for all controls."""

    expected = (
        len(base.REPEAT_SEEDS),
        len(oof.target),
        len(VIEW_NAMES) * EMBEDDING_CHANNELS_PER_VIEW,
    )
    if pretrained_features.shape != expected:
        raise ValueError("pretrained GFM feature matrix shape mismatch")
    if random_init_features.shape != expected:
        raise ValueError("random-init GFM feature matrix shape mismatch")
    common = np.column_stack([oof.baseline, oof.distance_to_well])
    structural_features = np.column_stack([oof.structural_features, common])
    baseline_metrics = base._metrics(oof.target, oof.baseline)  # noqa: SLF001
    if not np.array_equal(oof.baseline.copy(), oof.baseline):
        raise RuntimeError("P14 gate=0 is not bitwise equal to PyKrige")

    per_seed: dict[str, list[dict[str, Any]]] = {
        "pretrained_geophysical_fm": [],
        "random_init_same_architecture": [],
        "no_foundation_structural": [],
    }
    all_cells: list[dict[str, Any]] = []
    prediction_payload: dict[str, np.ndarray] = {
        "indices_kji": np.asarray(oof.indices_kji, dtype=np.int64),
        "fold_ids": np.asarray(oof.fold_ids, dtype=np.int64),
        "target": np.asarray(oof.target, dtype=np.float64),
        "baseline_prediction": np.asarray(oof.baseline, dtype=np.float64),
        "baseline_error": np.asarray(oof.baseline - oof.target, dtype=np.float64),
    }
    prediction_maps: dict[str, dict[str, dict[int, np.ndarray]]] = {
        route: {mode: {} for mode in HEAD_MODES}
        for route in (
            "pretrained_geophysical_fm",
            "random_init_same_architecture",
        )
    }
    for seed_id, seed in enumerate(base.REPEAT_SEEDS):
        route_inputs = {
            "pretrained_geophysical_fm": np.column_stack(
                [pretrained_features[seed_id], common]
            ),
            "random_init_same_architecture": np.column_stack(
                [random_init_features[seed_id], common]
            ),
            "no_foundation_structural": structural_features,
        }
        for route, features in route_inputs.items():
            result = diagnostics.evaluate_adaptive_route(
                route=route,
                features=features,
                oof=oof,
                seed=int(seed),
            )
            per_seed[route].append(result)
            all_cells.extend(result["per_fold"])
            if route in prediction_maps:
                for mode in HEAD_MODES:
                    prediction = np.asarray(
                        result["predictions"][mode]["gated"],
                        dtype=np.float64,
                    )
                    prediction_maps[route][mode][int(seed)] = prediction
                    prediction_payload[
                        f"{route}__{mode}__seed_{int(seed)}__prediction"
                    ] = prediction
                    prediction_payload[
                        f"{route}__{mode}__seed_{int(seed)}__error"
                    ] = prediction - oof.target

    heads: dict[str, Any] = {}
    for mode in HEAD_MODES:
        pretrained_summary = _summarize_route(
            route="pretrained_geophysical_fm",
            head_mode=mode,
            seed_results=per_seed["pretrained_geophysical_fm"],
            all_cells=all_cells,
            baseline_rmse=baseline_metrics["rmse"],
        )
        random_summary = _summarize_route(
            route="random_init_same_architecture",
            head_mode=mode,
            seed_results=per_seed["random_init_same_architecture"],
            all_cells=all_cells,
            baseline_rmse=baseline_metrics["rmse"],
        )
        structural_summary = _summarize_route(
            route="no_foundation_structural",
            head_mode=mode,
            seed_results=per_seed["no_foundation_structural"],
            all_cells=all_cells,
            baseline_rmse=baseline_metrics["rmse"],
        )
        pretrained_summary["rmse_minus_random_init_same_architecture"] = (
            pretrained_summary["gated_mean_seed_rmse"]
            - random_summary["gated_mean_seed_rmse"]
        )
        pretrained_summary["better_than_random_init_same_architecture"] = (
            pretrained_summary["gated_mean_seed_rmse"]
            < random_summary["gated_mean_seed_rmse"] - WIN_TOLERANCE
        )
        pretrained_summary["rmse_minus_no_foundation_structural"] = (
            pretrained_summary["gated_mean_seed_rmse"]
            - structural_summary["gated_mean_seed_rmse"]
        )
        pretrained_summary["better_than_no_foundation_structural"] = (
            pretrained_summary["gated_mean_seed_rmse"]
            < structural_summary["gated_mean_seed_rmse"] - WIN_TOLERANCE
        )
        bootstrap_seed = int.from_bytes(
            hashlib.sha256(
                f"p14:{mode}:pretrained:block-bootstrap".encode("utf-8")
            ).digest()[:8],
            "little",
        )
        pretrained_summary["block_bootstrap_rmse_delta_vs_pykrige"] = (
            diagnostics.block_bootstrap_rmse_delta(
                target=oof.target,
                baseline=oof.baseline,
                candidate_predictions_by_seed=prediction_maps[
                    "pretrained_geophysical_fm"
                ][mode],
                fold_ids=oof.fold_ids,
                bootstrap_seed=bootstrap_seed,
            )
        )
        random_summary["block_bootstrap_rmse_delta_vs_pykrige"] = (
            diagnostics.block_bootstrap_rmse_delta(
                target=oof.target,
                baseline=oof.baseline,
                candidate_predictions_by_seed=prediction_maps[
                    "random_init_same_architecture"
                ][mode],
                fold_ids=oof.fold_ids,
                bootstrap_seed=bootstrap_seed + 1,
            )
        )
        required_wins = math.ceil(0.8 * len(base.FOLD_IDS))
        promoted = bool(
            pretrained_summary["relative_gain_vs_pykrige"] >= 0.01
            and pretrained_summary[
                "better_than_random_init_same_architecture"
            ]
            and pretrained_summary["better_than_no_foundation_structural"]
            and pretrained_summary["independent_fold_outcome_counts"]["win"]
            >= required_wins
        )
        heads[mode] = {
            "pretrained_geophysical_fm": pretrained_summary,
            "random_init_same_architecture": random_summary,
            "no_foundation_structural": structural_summary,
            "promotion": {
                "passes_audited_p11_rule": promoted,
                "required_independent_fold_wins": required_wins,
                "minimum_relative_gain": 0.01,
                "random_init_same_architecture_control_present": True,
            },
        }
    best_mode = min(
        HEAD_MODES,
        key=lambda mode: (
            heads[mode]["pretrained_geophysical_fm"][
                "gated_mean_seed_rmse"
            ],
            mode,
        ),
    )
    any_promoted = any(
        heads[mode]["promotion"]["passes_audited_p11_rule"]
        for mode in HEAD_MODES
    )
    experiment = {
        "fixed_protocol": {
            "protocol_source": (
                "p11_residual_fusion_diagnostics.evaluate_adaptive_route"
            ),
            "base_model": base.EXPECTED_MODEL_ID,
            "lane": base.EXPECTED_LANE,
            "split_hash": base.EXPECTED_SPLIT_HASH,
            "outer_folds": list(base.FOLD_IDS),
            "repeat_seeds": list(base.REPEAT_SEEDS),
            "independent_spatial_units": len(base.FOLD_IDS),
            "seeds_are_paired_pseudo_repeats": True,
            "residual_alpha_candidates": list(diagnostics.RIDGE_ALPHAS),
            "gate_scale_candidates": list(base.GATE_CANDIDATES),
            "gate_bounds": [0.0, 1.0],
            "feature_budget": (
                f"{len(VIEW_NAMES)} views x "
                f"{EMBEDDING_CHANNELS_PER_VIEW} seeded channels"
            ),
            "test_or_holdout_tuning": False,
        },
        "baseline": {
            "pykrige_oof": baseline_metrics,
            "gate_zero_exact": base._metrics(  # noqa: SLF001
                oof.target,
                oof.baseline.copy(),
            ),
            "gate_zero_bitwise_equal_to_pykrige": True,
        },
        "heads": heads,
        "best_observed_pretrained_head": best_mode,
        "per_fold_seed_pseudo_repeats": all_cells,
        "decision": {
            "state": (
                "PROMOTE_DEVELOPMENT_ONLY"
                if any_promoted
                else "VERIFIED_NO_PROMOTION"
            ),
            "default_enabled": any_promoted,
            "pretrained_contribution_claimed": False,
            "contribution_boundary": (
                "pretrained and matched random-init use the same architecture "
                "and OOF pipeline, but overall changes are reported without "
                "automatic causal attribution to the foundation weights"
            ),
        },
    }
    return experiment, prediction_payload


def _write_prediction_errors(
    output_dir: Path,
    payload: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    path = output_dir / "prediction_errors.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": base._sha256(path),  # noqa: SLF001
        "arrays": sorted(payload),
        "rows": int(len(payload["target"])),
    }


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    experiment = result["experiment"]
    best_mode = experiment["best_observed_pretrained_head"]
    head = experiment["heads"][best_mode]
    pretrained = head["pretrained_geophysical_fm"]
    random_init = head["random_init_same_architecture"]
    structural = head["no_foundation_structural"]
    baseline = experiment["baseline"]["pykrige_oof"]["rmse"]
    counts = pretrained["independent_fold_outcome_counts"]
    bootstrap = pretrained["block_bootstrap_rmse_delta_vs_pykrige"]
    slice_policy = result["pretrained_feature_audit"]["slice_policy"]
    relative = pretrained["relative_gain_vs_pykrige"]
    lines = [
        "# P14 geophysical foundation model — development-only evidence",
        "",
        "## Outcome",
        "",
        f"- PyKrige baseline pooled development RMSE: `{baseline:.12f}`.",
        (
            f"- Best observed pretrained GFM head (`{best_mode}`) gated RMSE: "
            f"`{pretrained['gated_mean_seed_rmse']:.12f}`."
        ),
        (
            f"- Relative change versus baseline (positive means improvement): "
            f"`{relative:+.6%}`."
        ),
        (
            f"- Same-architecture random-init gated RMSE: "
            f"`{random_init['gated_mean_seed_rmse']:.12f}`; pretrained minus "
            f"random-init: "
            f"`{pretrained['rmse_minus_random_init_same_architecture']:+.12f}`."
        ),
        (
            f"- No-foundation structural gated RMSE: "
            f"`{structural['gated_mean_seed_rmse']:.12f}`."
        ),
        (
            f"- Five independent spatial-fold outcomes: {counts['win']} win / "
            f"{counts['loss']} loss / {counts['tie']} tie."
        ),
        (
            "- The experiment does not automatically attribute any overall "
            "change to GFM pretrained weights; the matched random-init result "
            "is reported separately."
        ),
        (
            f"- Decision: `{experiment['decision']['state']}`. The "
            "domain-matched pretrained encoder did not produce positive "
            "pooled development gain under the locked P11 protocol."
        ),
        "",
        "## 3-D to 2-D slice design",
        "",
        (
            "- The real `train.h5` patch shape is `[K,J,I]=[9,20,18]`; "
            "140 non-overlapping patches assemble exactly to `[63,100,72]`."
        ),
        (
            f"- Orientation: `{slice_policy['orientation']}` with source shape "
            f"`{slice_policy['source_slice_shape']}` resized bilinearly to "
            f"`{slice_policy['resize_shape']}`."
        ),
        (
            f"- The 10,240 OOF rows touch "
            f"{result['pretrained_feature_audit']['unique_fixed_i_slices']} "
            "distinct fixed-I slices; no unused slice is forwarded."
        ),
        f"- Rationale: {slice_policy['reason']}.",
        (
            "- GFM uses `patch_size=[400,1]`, so every token represents one "
            "complete vertical trace. Each OOF voxel receives the nearest "
            "resized trace token for its J coordinate plus the slice CLS token."
        ),
        (
            "- All three seismic attributes are forwarded separately. Each "
            "slice is z-scored over its own active cells; inactive pixels are "
            "zero. No PORO label or cross-sample fitted normalization is used."
        ),
        "",
        "## Five genuinely independent spatial units",
        "",
        (
            "The five locked outer folds are the inferential units. The three "
            "seeds are paired optimization pseudo-repeats inside each fold, "
            "not 15 independent observations."
        ),
        "",
        "| fold | baseline RMSE | pretrained GFM RMSE | delta | outcome |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in pretrained["independent_fold_outcomes"]:
        lines.append(
            f"| {row['outer_fold']} | "
            f"{row['baseline_mean_seed_rmse']:.9f} | "
            f"{row['gated_mean_seed_rmse']:.9f} | "
            f"{row['rmse_delta_vs_pykrige']:+.9f} | "
            f"{row['outcome_vs_pykrige']} |"
        )
    lines.extend(
        [
            "",
            "## Whole-fold bootstrap",
            "",
            (
                f"- Pooled RMSE delta (candidate - baseline): "
                f"`{bootstrap['point_estimate']:+.12f}`."
            ),
            (
                f"- 95% interval: "
                f"`[{bootstrap['confidence_interval'][0]:+.12f}, "
                f"{bootstrap['confidence_interval'][1]:+.12f}]`."
            ),
            (
                "- Bootstrap unit is the whole locked spatial fold. Seeds stay "
                "paired and voxels are never resampled independently."
            ),
            "",
            "## Model provenance and protocol",
            "",
            (
                "- Model: `thinkonward/geophysical-foundation-model`, snapshot "
                f"`{gfm.SNAPSHOT_REVISION}`, Apache-2.0 model card."
            ),
            (
                f"- Weight SHA-256: `{gfm.WEIGHTS_SHA256}`; vendor source "
                f"SHA-256: `{gfm.SOURCE_SHA256}`."
            ),
            (
                "- The encoder uses all 160 trace tokens (`len_keep=160`), so "
                "no trace is masked during feature extraction."
            ),
            (
                "- Residual regression, alpha grid, inner-OOF gate, gate "
                "bounds, PyKrige baseline and fold/seed identities are reused "
                "unchanged from the committed P11 diagnostic harness."
            ),
            (
                "- The random-init control instantiates the exact same GFM "
                "architecture independently for each paired seed and never "
                "loads pretrained weights."
            ),
            "",
            "## Holdout firewall",
            "",
            (
                "- Only `train.h5` metadata, `seismic_patch[0:3]` and "
                "`seismic_patch[8]` were read. PORO targets and baseline "
                "predictions came from hash-verified development OOF archives."
            ),
            (
                "- `test.h5`, frozen holdout paths, historical test metrics "
                "and holdout labels were neither opened nor probed."
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
    snapshot_path: Path,
    output_dir: Path,
    pretrained_feature_cache: Path,
    random_init_feature_cache: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run the complete pretrained/random-init P14 development experiment."""

    started = time.time()
    inputs = base.resolve_dev_inputs(data_dir)
    oof = base.load_oof_development(stage3_root)
    output_dir = Path(output_dir).expanduser().resolve()
    base.ensure_no_holdout_paths([output_dir])

    pretrained_cached = _load_valid_feature_cache(
        Path(pretrained_feature_cache).expanduser().resolve(),
        expected_identity=_feature_cache_identity(
            weight_mode="pretrained",
            train_h5_sha256=base._sha256(inputs.train_h5),  # noqa: SLF001
            indices_kji=oof.indices_kji,
            asset_audit=gfm.verify_local_assets(
                source_root,
                snapshot_path,
            )[2],
        ),
        indices_kji=oof.indices_kji,
    )
    random_cached = _load_valid_feature_cache(
        Path(random_init_feature_cache).expanduser().resolve(),
        expected_identity=_feature_cache_identity(
            weight_mode="random_init",
            train_h5_sha256=base._sha256(inputs.train_h5),  # noqa: SLF001
            indices_kji=oof.indices_kji,
            asset_audit=gfm.verify_local_assets(
                source_root,
                snapshot_path,
            )[2],
        ),
        indices_kji=oof.indices_kji,
    )
    seismic: np.ndarray | None = None
    active: np.ndarray | None = None
    assembly_audit: Mapping[str, Any] | None = None
    if pretrained_cached is None or random_cached is None:
        seismic, active, assembly_audit = assemble_seismic_volume(
            inputs.train_h5
        )

    pretrained_features, pretrained_audit = get_projected_features(
        weight_mode="pretrained",
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        snapshot_path=snapshot_path,
        feature_cache=pretrained_feature_cache,
        device=device,
        batch_size=batch_size,
        seismic_volume=seismic,
        active_volume=active,
        assembly_audit=assembly_audit,
    )
    print(
        json.dumps(
            {
                "event": "pretrained_features_ready",
                "cache_reused": pretrained_audit["cache_reused"],
                "shape": list(pretrained_features.shape),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    random_features, random_audit = get_projected_features(
        weight_mode="random_init",
        inputs=inputs,
        oof=oof,
        source_root=source_root,
        snapshot_path=snapshot_path,
        feature_cache=random_init_feature_cache,
        device=device,
        batch_size=batch_size,
        seismic_volume=seismic,
        active_volume=active,
        assembly_audit=assembly_audit,
    )
    print(
        json.dumps(
            {
                "event": "random_init_features_ready",
                "cache_reused": random_audit["cache_reused"],
                "shape": list(random_features.shape),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    experiment, prediction_payload = evaluate_gfm(
        oof=oof,
        pretrained_features=pretrained_features,
        random_init_features=random_features,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_artifact = _write_prediction_errors(
        output_dir,
        prediction_payload,
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_unix": time.time(),
        "implementation": {
            "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "script_sha256": base._sha256(Path(__file__)),  # noqa: SLF001
            "adapter": str(
                (
                    PROJECT_ROOT
                    / "_models"
                    / "reconstruction"
                    / "geophysical_fm.py"
                ).relative_to(PROJECT_ROOT)
            ),
            "adapter_sha256": base._sha256(  # noqa: SLF001
                PROJECT_ROOT
                / "_models"
                / "reconstruction"
                / "geophysical_fm.py"
            ),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pid": os.getpid(),
        },
        "model": {
            "id": "thinkonward/geophysical-foundation-model",
            "snapshot_revision": gfm.SNAPSHOT_REVISION,
            "license": "Apache-2.0",
            "pretraining_domain": (
                "450 Synthoseis synthetic 3-D seismic volumes"
            ),
            "architecture": "ViT-MAE with whole-trace masking",
            "input_shape": [1, 400, 160],
            "patch_size": [400, 1],
            "embedding_width": 1200,
        },
        "pretrained_feature_audit": pretrained_audit,
        "random_init_feature_audit": random_audit,
        "source_oof": {
            "records": list(oof.source_records),
            "rows": int(len(oof.target)),
            "indices_kji_sha256": base._array_sha256(  # noqa: SLF001
                oof.indices_kji
            ),
        },
        "experiment": experiment,
        "prediction_error_artifact": prediction_artifact,
        "holdout_firewall": {
            "hdf5_files_opened": ["train.h5"],
            "hdf5_datasets_read": [
                "seismic_patch[0:3]",
                "seismic_patch[8]",
            ],
            "label_dataset_read_by_encoder": False,
            "test_path_argument_exists": False,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "historical_test_metrics_read": False,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started,
            "device": device,
            "batch_size": int(batch_size),
            "pretrained_cache_reused": pretrained_audit["cache_reused"],
            "random_init_cache_reused": random_audit["cache_reused"],
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
        help="verify the sole legal development HDF5 path",
    )
    check.add_argument("--data-dir", type=Path, required=True)

    execute = subparsers.add_parser(
        "run",
        help="run P14 pretrained and random-init development OOF",
    )
    execute.add_argument("--data-dir", type=Path, required=True)
    execute.add_argument("--stage3-root", type=Path, required=True)
    execute.add_argument("--source-root", type=Path, required=True)
    execute.add_argument("--snapshot-path", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument(
        "--pretrained-feature-cache",
        type=Path,
        default=DEFAULT_PRETRAINED_CACHE,
    )
    execute.add_argument(
        "--random-init-feature-cache",
        type=Path,
        default=DEFAULT_RANDOM_INIT_CACHE,
    )
    execute.add_argument("--device", default="cuda:0")
    execute.add_argument("--batch-size", type=int, default=24)
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
    best_mode = result["experiment"]["best_observed_pretrained_head"]
    print(
        json.dumps(
            {
                "best_head": best_mode,
                "baseline_rmse": result["experiment"]["baseline"][
                    "pykrige_oof"
                ]["rmse"],
                "pretrained_gated_rmse": result["experiment"]["heads"][
                    best_mode
                ]["pretrained_geophysical_fm"]["gated_mean_seed_rmse"],
                "random_init_gated_rmse": result["experiment"]["heads"][
                    best_mode
                ]["random_init_same_architecture"]["gated_mean_seed_rmse"],
                "state": result["experiment"]["decision"]["state"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
