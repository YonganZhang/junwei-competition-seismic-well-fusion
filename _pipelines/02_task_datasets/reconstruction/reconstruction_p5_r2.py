#!/usr/bin/env python3
"""P5.2 / R2 development-only reconstruction study.

This runner is intentionally narrow:

- strict and conditional are separate TaskSpecs;
- only the physical development ``train.h5`` container is read;
- one frozen buffered development fold is used for the shared K4 pseudo-test
  block and metric mask;
- train/val preprocessing fits only on fold-train;
- budgets are frozen at 25/100/400 epochs with no HPO;
- conditional B0/B1/shuffled reuse one checkpoint; strict refuses B1/shuffled.

The output is portable evidence only: compact JSON/JSONL, figures and README
notes.  Checkpoints and any large temporary arrays stay under ``_tmp/``.
"""
from __future__ import annotations

import argparse
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

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from ml_framework.model_discovery import discover_model  # noqa: E402
from ml_framework.preprocess import denormalize, fit_minmax, fit_zscore, normalize  # noqa: E402
from ml_framework.seeding import SeedTree, seed_everything  # noqa: E402
from ml_framework.train import TrainHistory, train_loop  # noqa: E402
from ml_framework.visualize import plot_loss_curve  # noqa: E402

import p4_reconstruction as p4  # noqa: E402
import reconstruction_p5_r01 as r01  # noqa: E402


SCHEMA_VERSION = "p5.2-r2-reconstruction-v1"
ROOT_SEED = 2693
MODES = ("strict", "conditional")
MODEL_NAMES = ("reconstruction_linear_sgd", "reconstruction_tiny_mlp")
LOSS_NAMES = ("mse", "huber")
BUDGETS = (25, 100, 400)
FEATURE_VARIANTS = {
    "strict": ("full", "drop_coordinates", "drop_seismic"),
    "conditional": ("full", "drop_coordinates", "drop_seismic", "drop_well"),
}
TASK_TASK_IDS = {mode: p4.protocol(mode).task_id for mode in MODES}
TRAIN_POINT_CAP = 512
VALIDATION_POINT_CAP = 2048
PSEUDO_WELL_COUNT = 32
REQUESTED_FOLDS = 5
SHARED_PSEUDO_FOLD_ID = 4
BUFFER_BLOCKS = 1
MIN_BAND_SUPPORT = 32
COORD_NAMES = ("x_normalized", "y_normalized", "depth_normalized")
SEISMIC_NAMES = (
    "seismic_amplitude",
    "seismic_local_rms",
    "seismic_vertical_gradient",
)
COND_FULL_NAMES = ("conditional_idw_porosity", *SEISMIC_NAMES, *COORD_NAMES)
STRICT_FULL_NAMES = (*SEISMIC_NAMES, *COORD_NAMES)
MODEL_CONFIGS = {
    "reconstruction_linear_sgd": {
        "learning_rate": 0.01,
        "ridge_alpha": 0.1,
    },
    "reconstruction_tiny_mlp": {
        "learning_rate": 0.005,
        "ridge_alpha": 0.1,
        "hidden_features": 8,
    },
}


@dataclass(frozen=True)
class ModeBundle:
    mode: str
    geometry: r01.GeometryBundle
    split_manifest: Mapping[str, Any]
    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_local_indices: np.ndarray
    validation_local_indices: np.ndarray
    train_global_indices: np.ndarray
    validation_global_indices: np.ndarray
    pseudo_well_local_indices: np.ndarray
    pseudo_well_global_indices: np.ndarray
    train_target: np.ndarray
    validation_target: np.ndarray
    pseudo_well_values: np.ndarray
    common_metric_mask: np.ndarray
    pseudo_test_distances: np.ndarray
    distance_edges: tuple[float, ...]
    train_constraints: np.ndarray
    pseudo_test_constraints: np.ndarray
    access_audit: Mapping[str, Any]
    manifest: Mapping[str, Any]


def _relative(path: Path) -> str:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return resolved.relative_to(root).as_posix()
    return path.name


def _train_path(data_dir: Path | None) -> Path:
    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "_data" / "processed" / "reconstruction"
    path = root / "train.h5"
    if not path.is_file():
        raise FileNotFoundError(f"development train.h5 is missing: {path}")
    if path.name != "train.h5":
        raise RuntimeError("R2 only accepts the development train.h5 container")
    return path


def _contiguous_buckets(groups: Sequence[int], n_splits: int) -> list[tuple[int, ...]]:
    q, remainder = divmod(len(groups), n_splits)
    buckets: list[tuple[int, ...]] = []
    start = 0
    for index in range(n_splits):
        size = q + (1 if index < remainder else 0)
        buckets.append(tuple(int(value) for value in groups[start : start + size]))
        start += size
    return buckets


def load_mode_geometry(mode: str, data_dir: Path | None = None) -> r01.GeometryBundle:
    """Read only train.h5 geometry/input slices; never read PORO or well_log_seq."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    allowed = set(int(value) for value in p4.protocol(mode).development_i_blocks)
    path = _train_path(data_dir)
    records: list[r01.RecordGeometry] = []
    indices: list[np.ndarray] = []
    seismic: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    i_blocks: list[np.ndarray] = []
    k_blocks: list[np.ndarray] = []
    max_shape = np.zeros(3, dtype=np.int64)
    cursor = 0
    with h5py.File(path, "r") as handle:
        for source_key in sorted(handle):
            group = handle[source_key]
            meta = json.loads(group.attrs["meta"])
            k_block, j_block, i_block = (int(value) for value in meta["patch_index_kji"])
            if i_block not in allowed:
                continue
            start = tuple(int(value) for value in meta["patch_start_kji"])
            shape = tuple(int(value) for value in meta["patch_shape_kji"])
            patch = np.asarray(group["seismic_patch"][()], dtype=np.float32)
            if patch.shape != (9, *shape):
                raise ValueError(f"invalid seismic patch shape for {source_key}: {patch.shape}")
            signal_and_coordinates = np.asarray(patch[0:6], dtype=np.float32)
            masks = np.asarray(patch[7:9], dtype=np.float32)
            active_flat = np.flatnonzero(masks[1].reshape(-1) > 0.5)
            if not active_flat.size:
                continue
            local = np.indices(shape, dtype=np.int64).reshape(3, -1).T[active_flat]
            global_kji = local + np.asarray(start, dtype=np.int64)
            count = int(active_flat.size)
            indices.append(global_kji)
            seismic.append(r01.denoise_identity(signal_and_coordinates[0:3].reshape(3, -1).T[active_flat]))
            coordinates.append(signal_and_coordinates[3:6].reshape(3, -1).T[active_flat])
            observed.append(masks[0].reshape(-1)[active_flat] > 0.5)
            i_blocks.append(np.full(count, i_block, dtype=np.int16))
            k_blocks.append(np.full(count, k_block, dtype=np.int16))
            records.append(
                r01.RecordGeometry(
                    source_key=source_key,
                    sample_id=r01._sample_id(meta),  # noqa: SLF001
                    i_block=i_block,
                    j_block=j_block,
                    k_block=k_block,
                    patch_start_kji=start,
                    patch_shape_kji=shape,
                    active_flat_indices=active_flat.astype(np.int64),
                    cell_start=cursor,
                    cell_stop=cursor + count,
                )
            )
            cursor += count
            max_shape = np.maximum(max_shape, np.asarray(start) + np.asarray(shape))
    if not records:
        raise ValueError("development train.h5 contains no usable cells for this mode")
    bundle = r01.GeometryBundle(
        records=tuple(records),
        indices_kji=np.concatenate(indices).astype(np.int64),
        seismic=np.concatenate(seismic).astype(np.float64),
        coordinates=np.concatenate(coordinates).astype(np.float64),
        original_observed_mask=np.concatenate(observed).astype(bool),
        cell_i_blocks=np.concatenate(i_blocks),
        cell_k_blocks=np.concatenate(k_blocks),
        volume_shape_kji=tuple(int(value) for value in max_shape),
        access_audit={
            "physical_containers_opened": ["train.h5"],
            "physical_test_h5_opened": False,
            "datasets_read": ["group.attrs.meta", "seismic_patch[0:6]", "seismic_patch[7:9]"],
            "reference_sparse_poro_channel_6_read": False,
            "well_log_seq_read": False,
            "known_or_frozen_metrics_read": False,
            "known_or_frozen_predictions_read": False,
        },
    )
    if set(np.unique(bundle.cell_i_blocks).tolist()) != allowed:
        raise ValueError(f"{mode} development I-block coverage differs from the frozen protocol")
    return bundle


def load_mode_target(geometry: r01.GeometryBundle, data_dir: Path | None = None) -> tuple[np.ndarray, Mapping[str, Any]]:
    target, audit = r01.load_development_target(geometry, data_dir)
    if target.shape != (geometry.indices_kji.shape[0],) or not np.all(np.isfinite(target)):
        raise ValueError("development target is non-finite or misaligned")
    return target, audit


def _mode_split(mode: str, geometry: r01.GeometryBundle) -> Mapping[str, Any]:
    groups = sorted(int(value) for value in np.unique(geometry.cell_k_blocks))
    if len(groups) < REQUESTED_FOLDS:
        raise ValueError("frozen K4 development fold requires at least five K groups")
    buckets = _contiguous_buckets(groups, REQUESTED_FOLDS)
    validation = buckets[SHARED_PSEUDO_FOLD_ID]
    purged = tuple(
        value
        for value in groups
        if value not in validation
        and min(abs(value - held) for held in validation) <= BUFFER_BLOCKS
    )
    train = tuple(value for value in groups if value not in validation and value not in purged)
    if not train or not validation:
        raise ValueError("shared K4 fold has empty train or validation support")
    record = {
        "contract": "P5.2 R2 shared K4 buffered development fold",
        "mode": mode,
        "requested_n_splits": REQUESTED_FOLDS,
        "effective_n_splits": REQUESTED_FOLDS,
        "fold_id": SHARED_PSEUDO_FOLD_ID,
        "axis": "k_block",
        "buffer_blocks": BUFFER_BLOCKS,
        "development_i_blocks": list(p4.protocol(mode).development_i_blocks),
        "effective_train_k_blocks": list(train),
        "purged_k_blocks": list(purged),
        "validation_k_blocks": list(validation),
        "known_or_frozen_test_sample_ids_read": False,
    }
    return {**record, "split_hash": hash_payload(record)}


def _select_points(
    geometry: r01.GeometryBundle,
    mask: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    masked_positions = np.flatnonzero(mask)
    local = r01.select_spatial_points(geometry.coordinates[mask], geometry.indices_kji[mask], count)
    if local.size != count:
        raise ValueError("deterministic point selection returned the wrong count")
    return local.astype(np.int64), masked_positions[local].astype(np.int64)


def _build_point_selection(mode: str, geometry: r01.GeometryBundle) -> Mapping[str, Any]:
    split_manifest = _mode_split(mode, geometry)
    train_mask = np.isin(geometry.cell_k_blocks, np.asarray(split_manifest["effective_train_k_blocks"]))
    validation_mask = np.isin(geometry.cell_k_blocks, np.asarray(split_manifest["validation_k_blocks"]))
    train_count = min(TRAIN_POINT_CAP, int(train_mask.sum()))
    validation_count = min(VALIDATION_POINT_CAP, int(validation_mask.sum()))
    if train_count < 2 or validation_count < 2:
        raise ValueError("insufficient sampled development support")
    train_local, train_global = _select_points(geometry, train_mask, train_count)
    validation_local, validation_global = _select_points(geometry, validation_mask, validation_count)
    pseudo_local = r01.select_spatial_points(
        geometry.coordinates[validation_mask][validation_local],
        geometry.indices_kji[validation_mask][validation_local],
        min(PSEUDO_WELL_COUNT, validation_count),
    )
    pseudo_global = validation_global[pseudo_local].astype(np.int64)
    if pseudo_global.size < 2:
        raise ValueError("pseudo-test selection is too small")
    return {
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "train_local_indices": train_local,
        "validation_local_indices": validation_local,
        "train_global_indices": train_global,
        "validation_global_indices": validation_global,
        "pseudo_well_local_indices": pseudo_local.astype(np.int64),
        "pseudo_well_global_indices": pseudo_global,
        "train_selection_hash": r01._hash_arrays(indices_kji=train_global),  # noqa: SLF001
        "validation_selection_hash": r01._hash_arrays(indices_kji=validation_global),  # noqa: SLF001
        "pseudo_well_selection_hash": r01._hash_arrays(indices_kji=pseudo_global),  # noqa: SLF001
    }


def _feature_names(mode: str) -> tuple[str, ...]:
    return COND_FULL_NAMES if mode == "conditional" else STRICT_FULL_NAMES


def _variant_indices(mode: str, variant: str) -> tuple[int, ...]:
    variants = {
        "strict": {
            "full": (0, 1, 2, 3, 4, 5),
            "drop_coordinates": (0, 1, 2),
            "drop_seismic": (3, 4, 5),
        },
        "conditional": {
            "full": (0, 1, 2, 3, 4, 5, 6),
            "drop_coordinates": (0, 1, 2, 3),
            "drop_seismic": (0, 4, 5, 6),
            "drop_well": (1, 2, 3, 4, 5, 6),
        },
    }
    try:
        return variants[mode][variant]
    except KeyError as exc:
        raise ValueError(f"unsupported variant {variant!r} for mode {mode!r}") from exc


def _raw_features(
    mode: str,
    geometry: r01.GeometryBundle,
    global_indices: np.ndarray,
    constraints: np.ndarray | None,
    *,
    fallback: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if mode == "conditional":
        if constraints is None:
            raise ValueError("conditional mode requires a constraint array")
        idw = r01._idw(geometry.coordinates[global_indices], constraints)  # noqa: SLF001
        values = np.column_stack([idw, geometry.seismic[global_indices], geometry.coordinates[global_indices]])
        names = COND_FULL_NAMES
    else:
        values = np.column_stack([geometry.seismic[global_indices], geometry.coordinates[global_indices]])
        names = STRICT_FULL_NAMES
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("raw feature matrix is non-finite")
    if mode == "conditional" and not np.isfinite(fallback):
        raise ValueError("conditional fallback must be finite")
    return values, names


def _fit_preprocess(train_raw: np.ndarray, names: Sequence[str]) -> tuple[list[Any], np.ndarray, float]:
    stats: list[Any] = []
    columns: list[np.ndarray] = []
    for column, name in enumerate(names):
        stat = fit_minmax(train_raw[:, column]) if name in COORD_NAMES else fit_zscore(train_raw[:, column])
        stats.append(stat)
        columns.append(normalize(train_raw[:, column], stat))
    transformed = np.column_stack(columns).astype(np.float64)
    restored = np.column_stack([denormalize(transformed[:, column], stats[column]) for column in range(transformed.shape[1])])
    error = float(np.max(np.abs(restored - train_raw)))
    if error > 1e-10:
        raise ValueError(f"fold preprocessing round-trip failed: {error}")
    return stats, transformed, error


def _apply_preprocess(raw: np.ndarray, stats: Sequence[Any]) -> np.ndarray:
    return np.column_stack([normalize(raw[:, column], stats[column]) for column in range(raw.shape[1])]).astype(np.float64)


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> Mapping[str, Any]:
    return r01._regression_metrics(target, prediction)  # noqa: SLF001


def _distance_band_metrics(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    distances: np.ndarray,
    edges: Sequence[float],
) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for index in range(len(edges) - 1):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (distances > lower) & (distances <= upper)
        if index == 0:
            selected = (distances >= lower) & (distances <= upper)
        support = int(selected.sum())
        if support < MIN_BAND_SUPPORT:
            raise ValueError(f"distance band {index} support {support} < required {MIN_BAND_SUPPORT}")
        band_metrics = {
            name: _regression_metrics(target[selected], prediction[selected])
            for name, prediction in predictions.items()
        }
        record: dict[str, Any] = {
            "band_id": index,
            "lower_exclusive": lower if index else None,
            "upper_inclusive": None if math.isinf(upper) else upper,
            "voxel_count": support,
            "conditions": band_metrics,
        }
        if "B0" in band_metrics and "B1" in band_metrics:
            record["delta_rmse"] = {
                "B1_minus_B0": float(band_metrics["B1"]["rmse"] - band_metrics["B0"]["rmse"]),
                "shuffled_minus_B0": float(band_metrics.get("shuffled", band_metrics["B0"])["rmse"] - band_metrics["B0"]["rmse"]),
            }
        records.append(record)
    return records


def _curve_visible(history: TrainHistory) -> Mapping[str, Any]:
    values = np.asarray(history.val_loss, dtype=np.float64)
    if values.size == 0 or history.best_epoch < 0:
        return {"visible": False, "reason": "empty_history"}
    best = float(history.best_val_loss)
    best_epoch = int(history.best_epoch)
    window = 50
    tail = values[min(len(values), best_epoch + 1) :]
    if tail.size == 0:
        tail = values[-1:]
    threshold = best * 1.01
    sustained = tail >= threshold
    sustained_fraction = float(np.mean(sustained))
    ratio = float(np.min(tail) / best) if best > 0 else float("inf")
    return {
        "visible": bool(
            history.train_loss[0] > history.train_loss[min(best_epoch, len(history.train_loss) - 1)]
            and values[-1] > best
            and tail.size >= 5
            and sustained_fraction >= 0.6
        ),
        "best_epoch": best_epoch + 1,
        "window_epochs": window,
        "tail_epochs": int(tail.size),
        "fraction_tail_above_1pct": sustained_fraction,
        "minimum_tail_to_best_ratio": ratio,
        "best_val_loss": best,
        "last_val_loss": float(values[-1]),
    }


def _build_train_batches(features: np.ndarray, target: np.ndarray) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return ((np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)),)


def _model_config(mode: str, model_id: str, n_features: int, n_training_samples: int, loss_name: str, seed: int) -> dict[str, Any]:
    config = {
        **MODEL_CONFIGS[model_id],
        "n_features": int(n_features),
        "n_training_samples": int(n_training_samples),
        "loss_name": loss_name,
        "huber_delta": 0.01,
    }
    if model_id == "reconstruction_tiny_mlp":
        config["seed"] = int(seed)
    return config


def _load_best_model(mode: str, model_id: str, task_spec: Any, config: Mapping[str, Any], checkpoint_path: Path) -> Any:
    discovered = discover_model("reconstruction", model_id)
    model = discovered.build(task_spec, **dict(config))
    model.load_checkpoint(checkpoint_path)
    return model


def _point_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    label: str,
) -> Mapping[str, Any]:
    metrics = dict(_regression_metrics(target, prediction))
    metrics["label"] = label
    metrics["prediction_hash"] = r01._hash_arrays(prediction=np.asarray(prediction, dtype=np.float64))  # noqa: SLF001
    return metrics


def _render_visualization(
    *,
    mode: str,
    output_path: Path,
    history: TrainHistory,
    indices_kji: np.ndarray,
    volume_shape_kji: tuple[int, int, int],
    truth: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    caveat: str,
) -> Path:
    def dense(values: np.ndarray) -> np.ndarray:
        volume = np.full(volume_shape_kji, np.nan, dtype=np.float64)
        volume[tuple(indices_kji.T)] = values
        return volume

    def best_plane(valid: np.ndarray, axis: int) -> int:
        reduce_axes = tuple(index for index in range(3) if index != axis)
        counts = np.sum(valid, axis=reduce_axes)
        return int(np.argmax(counts))

    def plane(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
        return np.take(volume, index, axis=axis)

    def finite_fill(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        return np.where(finite, values, float(np.mean(values[finite])))

    def radial_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filled = finite_fill(values)
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(filled))))
        yy, xx = np.indices(spectrum.shape)
        radius = np.sqrt((yy - (spectrum.shape[0] - 1) / 2.0) ** 2 + (xx - (spectrum.shape[1] - 1) / 2.0) ** 2)
        bins = np.arange(0, int(radius.max()) + 2)
        index = np.digitize(radius.ravel(), bins) - 1
        profile = np.asarray(
            [spectrum.ravel()[index == value].mean() for value in range(len(bins) - 1) if np.any(index == value)]
        )
        return np.arange(profile.size), profile

    truth_volume = dense(truth)
    prediction_volume = dense(predictions["B0"])
    residual_volume = dense(predictions["B0"] - truth)
    valid = np.isfinite(truth_volume) & np.isfinite(prediction_volume)
    planes = (
        (2, best_plane(valid, 2), "inline / I"),
        (1, best_plane(valid, 1), "crossline / J"),
        (0, best_plane(valid, 0), "time-depth / K"),
    )
    finite_truth = np.asarray(truth, dtype=np.float64)
    finite_prediction = np.asarray(predictions["B0"], dtype=np.float64)
    finite_residual = np.asarray(predictions["B0"] - truth, dtype=np.float64)
    property_limits = (float(min(finite_truth.min(), finite_prediction.min())), float(max(finite_truth.max(), finite_prediction.max())))
    residual_limit = float(max(1e-12, np.max(np.abs(finite_residual))))

    fig = plt.figure(figsize=(18, 18))
    grid = fig.add_gridspec(4, 3, height_ratios=(1.0, 1.0, 1.0, 0.95), hspace=0.30, wspace=0.20)
    for row, (axis, index, label) in enumerate(planes):
        for column, (volume, title, cmap, limits) in enumerate(
            (
                (truth_volume, "reference Eclipse porosity", "viridis", property_limits),
                (prediction_volume, "reconstructed porosity", "viridis", property_limits),
                (residual_volume, "residual (prediction-reference)", "coolwarm", (-residual_limit, residual_limit)),
            )
        ):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(plane(volume, axis, index), origin="lower", aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_title(f"{label}={index}: {title}")
            ax.set_xlabel("grid axis")
            ax.set_ylabel("grid axis")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax_residual = fig.add_subplot(grid[3, 0])
    ax_residual.hist(finite_residual, bins=50, color="tab:red", alpha=0.8)
    ax_residual.axvline(0.0, color="black", linewidth=1)
    ax_residual.set_title("voxel error distribution")
    ax_residual.set_xlabel("prediction - reference porosity")
    ax_residual.set_ylabel("count")

    ax_distribution = fig.add_subplot(grid[3, 1])
    seismic = np.asarray(predictions["seismic"], dtype=np.float64)
    for values, label, color in (
        (finite_truth, "reference porosity", "black"),
        (finite_prediction, "prediction", "tab:blue"),
        (seismic, "seismic amplitude proxy", "tab:orange"),
    ):
        scale = float(np.std(values))
        standardized = (values - float(np.mean(values))) / (scale if scale > 0 else 1.0)
        ax_distribution.hist(standardized, bins=50, density=True, histtype="step", linewidth=1.5, label=label, color=color)
    ax_distribution.set_title("standardized attribute/property distributions")
    ax_distribution.set_xlabel("z-score (display only)")
    ax_distribution.legend(fontsize=8)

    ax_loss = fig.add_subplot(grid[3, 2])
    epochs = np.arange(1, len(history.train_loss) + 1, dtype=np.int64)
    ax_loss.plot(epochs, history.train_loss, label="train loss")
    ax_loss.plot(epochs, history.val_loss, label="val loss")
    if history.best_epoch >= 0:
        best_epoch = history.best_epoch + 1
        ax_loss.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch={best_epoch}")
        left = max(1, best_epoch - 20)
        right = min(int(epochs[-1]), best_epoch + 20)
        inset = ax_loss.inset_axes([0.54, 0.53, 0.42, 0.40])
        inset.plot(epochs, history.train_loss, color="tab:blue")
        inset.plot(epochs, history.val_loss, color="tab:orange")
        inset.axvline(best_epoch, color="gray", linestyle="--")
        inset.set_xlim(left, right)
        valley = min(history.val_loss[max(0, best_epoch - 2) : min(len(history.val_loss), best_epoch + 3)])
        inset.set_ylim(max(0.0, valley * 0.98), max(history.val_loss[left - 1 : right]) * 1.02)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("best-epoch zoom", fontsize=8)
    ax_loss.set_title("learning curve")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=8)
    ax_loss.text(
        0.02,
        0.02,
        caveat,
        transform=ax_loss.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        wrap=True,
    )

    primary_metrics = metrics
    if isinstance(metrics, Mapping) and "B1" in metrics and isinstance(metrics["B1"], Mapping):
        primary_metrics = metrics["B1"] if mode == "conditional" else metrics["B0"]
    fig.suptitle(
        "\n".join(
            [
                f"P5.2 R2 {mode.upper()} development-only reconstruction",
                f"RMSE={float(primary_metrics['rmse']):.6f} | MAE={float(primary_metrics['mae']):.6f} | R²={float(primary_metrics['r2']):.6f}",
                caveat,
            ]
        ),
        fontsize=14,
        y=0.995,
    )
    fig.text(
        0.01,
        0.005,
        "Read-only development evidence; no physical test.h5, frozen metrics, or test-region holdout is consumed.",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _run_one_cell(
    *,
    mode: str,
    model_id: str,
    loss_name: str,
    budget: int,
    feature_variant: str,
    bundle: ModeBundle,
    output_root: Path,
) -> dict[str, Any]:
    task_spec = p4.task_spec(mode)
    seed = SeedTree(ROOT_SEED).seed("model", mode, model_id, loss_name, feature_variant, budget)
    config = _model_config(mode, model_id, len(_variant_indices(mode, feature_variant)), len(bundle.train_target), loss_name, seed)
    train_raw_full, full_names = _raw_features(mode, bundle.geometry, bundle.train_global_indices, bundle.train_constraints, fallback=float(np.mean(bundle.train_target)))
    val_raw_b0_full, _ = _raw_features(mode, bundle.geometry, bundle.validation_global_indices, bundle.train_constraints, fallback=float(np.mean(bundle.train_target)))
    if mode == "conditional":
        b1_constraints = np.concatenate([bundle.train_constraints, bundle.pseudo_test_constraints])
        shuffled_values = np.roll(bundle.pseudo_well_values, 1 + ROOT_SEED % max(1, bundle.pseudo_well_values.size - 1))
        if np.array_equal(shuffled_values, bundle.pseudo_well_values):
            raise RuntimeError("conditional shuffled pseudo-well values are identical to the reference values")
        shuffled_constraints = np.column_stack(
            [bundle.geometry.coordinates[bundle.pseudo_well_global_indices], shuffled_values]
        ).astype(np.float64)
        val_raw_b1_full, _ = _raw_features(mode, bundle.geometry, bundle.validation_global_indices, b1_constraints, fallback=float(np.mean(bundle.train_target)))
        val_raw_shuffled_full, _ = _raw_features(mode, bundle.geometry, bundle.validation_global_indices, np.concatenate([bundle.train_constraints, shuffled_constraints]), fallback=float(np.mean(bundle.train_target)))
    else:
        b1_constraints = None
        shuffled_constraints = None
        shuffled_values = None
        val_raw_b1_full = None
        val_raw_shuffled_full = None

    indices = _variant_indices(mode, feature_variant)
    train_raw = train_raw_full[:, indices]
    val_raw_b0 = val_raw_b0_full[:, indices]
    if mode == "conditional":
        assert val_raw_b1_full is not None and val_raw_shuffled_full is not None
        val_raw_b1 = val_raw_b1_full[:, indices]
        val_raw_shuffled = val_raw_shuffled_full[:, indices]
    else:
        val_raw_b1 = None
        val_raw_shuffled = None
    feature_names = tuple(full_names[index] for index in indices)

    stats, train_features, roundtrip_error = _fit_preprocess(train_raw, feature_names)
    val_features_b0 = _apply_preprocess(val_raw_b0, stats)
    if mode == "conditional":
        val_features_b1 = _apply_preprocess(val_raw_b1, stats)
        val_features_shuffled = _apply_preprocess(val_raw_shuffled, stats)
    else:
        val_features_b1 = None
        val_features_shuffled = None
    if not np.all(np.isfinite(train_features)) or not np.all(np.isfinite(val_features_b0)):
        raise FloatingPointError("normalized features are non-finite")

    discovered = discover_model("reconstruction", model_id)
    model = discovered.build(
        task_spec,
        **config,
    )
    train_batch = (np.asarray(train_features, dtype=np.float64), np.asarray(bundle.train_target, dtype=np.float64))
    val_batch = (np.asarray(val_features_b0, dtype=np.float64), np.asarray(bundle.validation_target, dtype=np.float64))
    checkpoint_dir = output_root / "cells" / mode / model_id / loss_name / feature_variant / f"updates_{budget:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed_report = seed_everything(ROOT_SEED, strict=True, include_torch=False).to_dict()
    history = train_loop(
        model=model,
        train_step_fn=lambda batch: float(model.train_batch(batch)),
        val_step_fn=lambda batch: float(model.validation_loss(batch)),
        train_batches_fn=lambda: _build_train_batches(*train_batch),
        val_batches_fn=lambda: _build_train_batches(*val_batch),
        epochs=budget,
        save_checkpoint_fn=lambda model_obj, path: model_obj.save_checkpoint(path),
        checkpoint_dir=checkpoint_dir,
    )
    best_ckpt = checkpoint_dir / "best.ckpt"
    last_ckpt = checkpoint_dir / "last.ckpt"
    if not best_ckpt.is_file() or not last_ckpt.is_file():
        raise RuntimeError("train_loop did not write best/last checkpoints")
    best_model = _load_best_model(mode, model_id, task_spec, config, best_ckpt)
    probe = train_features[: min(64, len(train_features))]
    roundtrip_model = _load_best_model(mode, model_id, task_spec, config, best_ckpt)
    if probe.size == 0:
        raise RuntimeError("training probe is empty")
    roundtrip_error_model = float(np.max(np.abs(best_model.predict_array(probe) - roundtrip_model.predict_array(probe))))
    if roundtrip_error_model > 1e-9:
        raise AssertionError(f"checkpoint round-trip exceeded tolerance: {roundtrip_error_model}")
    b0_prediction = np.asarray(best_model.predict_array(val_features_b0), dtype=np.float64)
    if mode == "conditional":
        assert val_features_b1 is not None and val_features_shuffled is not None
        b1_prediction = np.asarray(best_model.predict_array(val_features_b1), dtype=np.float64)
        shuffled_prediction = np.asarray(best_model.predict_array(val_features_shuffled), dtype=np.float64)
        predictions = {"B0": b0_prediction, "B1": b1_prediction, "shuffled": shuffled_prediction}
    else:
        predictions = {"B0": b0_prediction}
    if not all(np.all(np.isfinite(values)) for values in predictions.values()):
        raise FloatingPointError("prediction array contains non-finite values")

    validation_mask = bundle.validation_mask
    validation_global = np.asarray(bundle.validation_global_indices, dtype=np.int64)
    common_local = np.asarray(bundle.common_metric_mask[validation_global], dtype=bool)
    if common_local.shape != bundle.validation_target.shape:
        raise ValueError("common metric mask must align with validation target")
    metric_target = bundle.validation_target[common_local]
    metric_predictions = {name: values[common_local] for name, values in predictions.items()}
    metrics = {name: _regression_metrics(metric_target, values) for name, values in metric_predictions.items()}
    metric_distances = bundle.pseudo_test_distances[common_local]
    band_metrics = _distance_band_metrics(metric_target, metric_predictions, metric_distances, bundle.distance_edges)
    feature_hashes = {
        "B0": r01._hash_arrays(features=val_raw_b0),  # noqa: SLF001
    }
    if mode == "conditional":
        feature_hashes["B1"] = r01._hash_arrays(features=val_raw_b1)  # noqa: SLF001
        feature_hashes["shuffled"] = r01._hash_arrays(features=val_raw_shuffled)  # noqa: SLF001
    prediction_hashes = {name: r01._hash_arrays(prediction=value) for name, value in metric_predictions.items()}  # noqa: SLF001
    curve_visible = _curve_visible(history)
    condition_audit = {
        "B0": {
            "formal_name": "no_pseudo_test_PORO_condition",
            "pseudo_test_constraints": 0,
            "fixed_fold_train_constraints": int(bundle.train_constraints.shape[0]),
            "fixed_weak_tie_seismic_sampling_retained": True,
        },
        "exact_pseudo_test_cells_excluded_from_metrics": int(bundle.pseudo_well_global_indices.shape[0]),
        "shared_checkpoint_all_conditions": True,
        "conditional_reconstruction_not_strict_holdout": mode == "conditional",
    }
    if mode == "conditional":
        condition_audit["B1"] = {
            "formal_name": "correct_synthetic_reference_revealed_pseudo_wells",
            "pseudo_test_constraints": int(bundle.pseudo_test_constraints.shape[0]),
            "values_hash": r01._hash_arrays(values=bundle.pseudo_well_values),  # noqa: SLF001
        }
        condition_audit["shuffled"] = {
            "formal_name": "seed2693_shuffled_pseudo_well_values_fixed_locations",
            "pseudo_test_constraints": int(bundle.pseudo_test_constraints.shape[0]),
            "values_hash": r01._hash_arrays(values=shuffled_values),  # noqa: SLF001
            "non_identity": True,
        }
    summary_predictions = {
        "seismic": bundle.geometry.seismic[bundle.validation_global_indices][:, 0],
        "B0": metric_predictions["B0"],
    }
    if mode == "conditional":
        summary_predictions["B1"] = metric_predictions["B1"]
        summary_predictions["shuffled"] = metric_predictions["shuffled"]
    mode_dir = output_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    loss_curve_path = mode_dir / "loss_curve.png"
    plot_loss_curve(history, loss_curve_path)
    visualization_path = mode_dir / "prediction_visualization.png"
    caveat = (
        "CONDITIONAL reconstruction given pseudo-test well constraints; NOT strict holdout generalization."
        if mode == "conditional"
        else "STRICT reconstruction with no pseudo-test well input; not a blind field-test claim."
    )
    _render_visualization(
        mode=mode,
        output_path=visualization_path,
        history=history,
        indices_kji=bundle.geometry.indices_kji[bundle.validation_global_indices][common_local],
        volume_shape_kji=bundle.geometry.volume_shape_kji,
        truth=metric_target,
        predictions=summary_predictions,
        metrics=metrics,
        caveat=caveat,
    )
    wall_seconds = time.perf_counter() - started
    checkpoint_roundtrip = {
        "best": {
            "path": _relative(best_ckpt),
            "sha256": hash_file(best_ckpt),
            "bytes": best_ckpt.stat().st_size,
        },
        "last": {
            "path": _relative(last_ckpt),
            "sha256": hash_file(last_ckpt),
            "bytes": last_ckpt.stat().st_size,
        },
        "roundtrip_max_abs_error": roundtrip_error_model,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "model_id": model_id,
        "loss_name": loss_name,
        "feature_variant": feature_variant,
        "budget_updates": budget,
        "root_seed": ROOT_SEED,
        "split_hash": bundle.split_manifest["split_hash"],
        "selection_hashes": {
            "train": (bundle.manifest.get("selection_hashes") or bundle.manifest["selection"])["train_selection_hash"],
            "validation": (bundle.manifest.get("selection_hashes") or bundle.manifest["selection"])["validation_selection_hash"],
            "pseudo_well": (bundle.manifest.get("selection_hashes") or bundle.manifest["selection"])["pseudo_well_selection_hash"],
        },
        "status": "passed",
        "reason": None,
        "training": {
            "fit_scope": "fold-train only",
            "train_points": int(train_batch[0].shape[0]),
            "validation_points": int(val_batch[0].shape[0]),
            "pseudo_well_points": int(bundle.pseudo_well_global_indices.shape[0]),
            "history": history.to_dict(),
            "train_loss_first": float(history.train_loss[0]),
            "train_loss_last": float(history.train_loss[-1]),
            "best_epoch": int(history.best_epoch + 1),
            "best_val_loss": float(history.best_val_loss),
            "preprocess_roundtrip_max_abs_error": roundtrip_error,
            "preprocess_stats": [getattr(stat, "to_dict", lambda: stat)() for stat in stats],
            "feature_names": list(feature_names),
            "curve_visible": curve_visible,
            "hpo_performed": False,
        },
        "checkpoint": checkpoint_roundtrip,
        "metrics": metrics,
        "band_metrics": band_metrics,
        "prediction_hashes": prediction_hashes,
        "feature_hashes": feature_hashes,
        "condition_audit": condition_audit,
        "condition_status": {
            "B1": "passed" if mode == "conditional" else "not_applicable",
            "shuffled": "passed" if mode == "conditional" else "not_applicable",
        },
        "r3_gate": {
            "core_condition": "B1" if mode == "conditional" else "B0",
            "budget_100_rmse": None,
            "budget_400_rmse": None,
            "budget_100_to_400_delta_ratio": None,
            "r3_allowed": None,
        },
        "access_audit": dict(bundle.access_audit),
        "test_firewall": {
            "physical_test_h5_opened": False,
            "known_or_frozen_arrays_read": False,
            "known_or_frozen_metrics_read": False,
            "known_or_frozen_predictions_read": False,
            "global_well_log_seq_read": False,
            "only_physical_train_h5_development_channels": True,
        },
        "resources": {
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "wall_seconds": wall_seconds,
            "python": platform.python_version(),
            "python_executable": Path(sys.executable).name,
            "downloads_performed_bytes": 0,
        },
        "artifacts": {
            "checkpoint_dir": _relative(checkpoint_dir),
            "loss_curve": _relative(loss_curve_path),
            "visualization": _relative(visualization_path),
        },
        "fresh_blind": False,
        "field_generalization": False,
        "development_protocol_mechanism_only": True,
    }
    result["result_hash"] = hash_payload(result)
    atomic_write_json(checkpoint_dir / "status.json", result)
    return result


def _build_leaderboard(mode: str, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    subset = [record for record in records if record["mode"] == mode and record["status"] == "passed"]
    if mode == "conditional":
        sort_key = lambda item: (float(item["metrics"]["B1"]["rmse"]), float(item["metrics"]["B0"]["rmse"]), float(item["training"]["best_val_loss"]), item["model_id"], item["loss_name"], item["feature_variant"], int(item["budget_updates"]))
        primary = "B1_rmse"
    else:
        sort_key = lambda item: (float(item["metrics"]["B0"]["rmse"]), float(item["training"]["best_val_loss"]), item["model_id"], item["loss_name"], item["feature_variant"], int(item["budget_updates"]))
        primary = "B0_rmse"
    ordered = sorted(subset, key=sort_key)
    entries = []
    for rank, record in enumerate(ordered, start=1):
        primary_metric = float(record["metrics"]["B1"]["rmse"] if mode == "conditional" else record["metrics"]["B0"]["rmse"])
        entries.append(
            {
                "rank": rank,
                "cell_id": f"{record['mode']}/{record['model_id']}/{record['loss_name']}/{record['feature_variant']}/updates_{record['budget_updates']:03d}",
                "model_id": record["model_id"],
                "loss_name": record["loss_name"],
                "feature_variant": record["feature_variant"],
                "budget_updates": record["budget_updates"],
                "primary_metric": primary_metric,
                "metrics": record["metrics"],
                "best_epoch": record["training"]["best_epoch"],
                "wall_seconds": record["resources"]["wall_seconds"],
                "condition_audit": record["condition_audit"],
                "checkpoint_sha256": record["checkpoint"]["best"]["sha256"],
                "result_hash": record["result_hash"],
            }
        )
    by_key = {(record["model_id"], record["loss_name"], record["feature_variant"], record["budget_updates"]): record for record in subset}
    r3_gate = []
    for model_id in MODEL_NAMES:
        for loss_name in LOSS_NAMES:
            key_100 = (model_id, loss_name, "full", 100)
            key_400 = (model_id, loss_name, "full", 400)
            if key_100 in by_key and key_400 in by_key:
                rmse_100 = float(by_key[key_100]["metrics"]["B1"]["rmse"] if mode == "conditional" else by_key[key_100]["metrics"]["B0"]["rmse"])
                rmse_400 = float(by_key[key_400]["metrics"]["B1"]["rmse"] if mode == "conditional" else by_key[key_400]["metrics"]["B0"]["rmse"])
                delta_ratio = (rmse_100 - rmse_400) / max(rmse_100, 1e-12)
                r3_gate.append(
                    {
                        "model_id": model_id,
                        "loss_name": loss_name,
                        "rmse_100": rmse_100,
                        "rmse_400": rmse_400,
                        "delta_ratio": delta_ratio,
                        "plateau_reached": bool(delta_ratio < 0.01),
                    }
                )
    leaderboard = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "development_only": True,
        "rankable": False,
        "primary_metric": primary,
        "entries": entries,
        "split_hash": next((record.get("split_hash") for record in records if record.get("split_hash") is not None), None),
        "selection_hashes": next((record.get("selection_hashes") for record in records if record.get("selection_hashes") is not None), None),
        "count": len(entries),
        "r3_allowed": bool(r3_gate and all(item["plateau_reached"] for item in r3_gate)),
        "r3_gate": r3_gate,
    }
    leaderboard["leaderboard_hash"] = hash_payload(leaderboard)
    return leaderboard


def _summarize_mode(mode: str, records: Sequence[Mapping[str, Any]], leaderboard: Mapping[str, Any]) -> Mapping[str, Any]:
    mode_records = [record for record in records if record["mode"] == mode]
    counts = {"passed": 0, "blocked": 0, "not_rankable": 0}
    for record in mode_records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    first_condition_record = next((record for record in mode_records if "condition_audit" in record), None)
    full_full = [
        record for record in mode_records if record["status"] == "passed" and record["feature_variant"] == "full"
    ]
    core_pairs = []
    for record in full_full:
        if record["budget_updates"] == 100:
            core_pairs.append(("100", record))
        if record["budget_updates"] == 400:
            core_pairs.append(("400", record))
    by_key = {(record["model_id"], record["loss_name"], record["feature_variant"], record["budget_updates"]): record for record in full_full}
    r3_gate = []
    for model_id in MODEL_NAMES:
        for loss_name in LOSS_NAMES:
            key_100 = (model_id, loss_name, "full", 100)
            key_400 = (model_id, loss_name, "full", 400)
            if key_100 in by_key and key_400 in by_key:
                rmse_100 = float(by_key[key_100]["metrics"]["B1"]["rmse"] if mode == "conditional" else by_key[key_100]["metrics"]["B0"]["rmse"])
                rmse_400 = float(by_key[key_400]["metrics"]["B1"]["rmse"] if mode == "conditional" else by_key[key_400]["metrics"]["B0"]["rmse"])
                delta_ratio = (rmse_100 - rmse_400) / max(rmse_100, 1e-12)
                r3_gate.append(
                    {
                        "model_id": model_id,
                        "loss_name": loss_name,
                        "rmse_100": rmse_100,
                        "rmse_400": rmse_400,
                        "delta_ratio": delta_ratio,
                        "plateau_reached": bool(delta_ratio < 0.01),
                    }
                )
    r3_allowed = bool(r3_gate and all(item["plateau_reached"] for item in r3_gate))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "root_seed": ROOT_SEED,
        "development_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "shared_k4_block": SHARED_PSEUDO_FOLD_ID,
        "shared_k4_metric_mask_count": int(first_condition_record["condition_audit"]["exact_pseudo_test_cells_excluded_from_metrics"]) if first_condition_record else 0,
        "counts": counts,
        "r3_allowed": r3_allowed,
        "r3_gate": r3_gate,
        "leaderboard": {
            "path": f"p5_r2_evidence/{mode}/leaderboard.json",
            "sha256": leaderboard["leaderboard_hash"],
        },
        "results": f"p5_r2_evidence/{mode}/results.jsonl",
        "visualizations": f"p5_r2_evidence/{mode}/visualization_manifest.json",
    }
    summary["summary_hash"] = hash_payload(summary)
    return summary


def _write_mode_outputs(mode: str, records: Sequence[Mapping[str, Any]], output_dir: Path) -> Mapping[str, Any]:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    results_path = mode_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for record in records if record["mode"] == mode),
        encoding="utf-8",
    )
    leaderboard = _build_leaderboard(mode, records)
    atomic_write_json(mode_dir / "leaderboard.json", leaderboard)
    visualization_manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "figures": [
            {
                "path": f"p5_r2_evidence/{mode}/prediction_visualization.png",
                "sha256": hash_file(mode_dir / "prediction_visualization.png") if (mode_dir / "prediction_visualization.png").is_file() else None,
            },
            {
                "path": f"p5_r2_evidence/{mode}/loss_curve.png",
                "sha256": hash_file(mode_dir / "loss_curve.png") if (mode_dir / "loss_curve.png").is_file() else None,
            },
        ],
        "development_only": True,
        "fresh_blind": False,
    }
    atomic_write_json(mode_dir / "visualization_manifest.json", visualization_manifest)
    oof_manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "development_only": True,
        "fresh_blind": False,
        "evaluation_cells": [
            {
                "cell_id": f"{record['mode']}/{record['model_id']}/{record['loss_name']}/{record['feature_variant']}/updates_{record['budget_updates']:03d}",
                "split_hash": record["split_hash"],
                "result_hash": record["result_hash"],
                "checkpoint_sha256": record["checkpoint"]["best"]["sha256"],
                "condition_audit": record["condition_audit"],
            }
            for record in records
            if record["mode"] == mode and record["status"] == "passed"
        ],
    }
    atomic_write_json(mode_dir / "oof_manifest.json", oof_manifest)
    summary = _summarize_mode(mode, records, leaderboard)
    atomic_write_json(mode_dir / "summary.json", summary)
    return summary


def _write_mode_blocked(mode: str, output_dir: Path, exc: Exception) -> Mapping[str, Any]:
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "status": "blocked",
        "reason": {"code": type(exc).__name__, "message": str(exc)},
        "fresh_blind": False,
        "field_generalization": False,
        "development_protocol_mechanism_only": True,
    }
    record["result_hash"] = hash_payload(record)
    (mode_dir / "results.jsonl").write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    leaderboard = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "development_only": True,
        "rankable": False,
        "primary_metric": "B1_rmse" if mode == "conditional" else "B0_rmse",
        "entries": [],
        "split_hash": None,
        "selection_hashes": None,
        "count": 0,
        "r3_allowed": False,
        "blocked_reason": record["reason"],
    }
    leaderboard["leaderboard_hash"] = hash_payload(leaderboard)
    atomic_write_json(mode_dir / "leaderboard.json", leaderboard)
    atomic_write_json(
        mode_dir / "visualization_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "mode": mode,
            "task_id": TASK_TASK_IDS[mode],
            "development_only": True,
            "fresh_blind": False,
            "figures": [],
        },
    )
    atomic_write_json(
        mode_dir / "oof_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "mode": mode,
            "task_id": TASK_TASK_IDS[mode],
            "development_only": True,
            "fresh_blind": False,
            "evaluation_cells": [],
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "root_seed": ROOT_SEED,
        "development_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "shared_k4_block": SHARED_PSEUDO_FOLD_ID,
        "shared_k4_metric_mask_count": 0,
        "counts": {"passed": 0, "blocked": 1, "not_rankable": 0},
        "r3_allowed": False,
        "r3_gate": [],
        "leaderboard": {"path": f"p5_r2_evidence/{mode}/leaderboard.json", "sha256": leaderboard["leaderboard_hash"]},
        "results": f"p5_r2_evidence/{mode}/results.jsonl",
        "visualizations": f"p5_r2_evidence/{mode}/visualization_manifest.json",
        "blocked_reason": record["reason"],
    }
    summary["summary_hash"] = hash_payload(summary)
    atomic_write_json(mode_dir / "summary.json", summary)
    return summary


def run_mode(mode: str, data_dir: Path, output_dir: Path) -> Mapping[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    try:
        geometry = load_mode_geometry(mode, data_dir)
    except Exception as exc:
        return _write_mode_blocked(mode, output_dir, exc)
    split_manifest = _mode_split(mode, geometry)
    selection = _build_point_selection(mode, geometry)
    train_mask = np.asarray(selection["train_mask"], dtype=bool)
    validation_mask = np.asarray(selection["validation_mask"], dtype=bool)
    train_local_indices = np.asarray(selection["train_local_indices"], dtype=np.int64)
    validation_local_indices = np.asarray(selection["validation_local_indices"], dtype=np.int64)
    train_global_indices = np.asarray(selection["train_global_indices"], dtype=np.int64)
    validation_global_indices = np.asarray(selection["validation_global_indices"], dtype=np.int64)
    pseudo_well_local_indices = np.asarray(selection["pseudo_well_local_indices"], dtype=np.int64)
    pseudo_well_global_indices = np.asarray(selection["pseudo_well_global_indices"], dtype=np.int64)
    target, target_audit = load_mode_target(geometry, data_dir)
    train_target = target[train_global_indices]
    validation_target = target[validation_global_indices]
    pseudo_well_values = target[pseudo_well_global_indices]
    train_constraints = np.column_stack([geometry.coordinates[train_global_indices], train_target]).astype(np.float64)
    pseudo_test_constraints = np.column_stack([geometry.coordinates[pseudo_well_global_indices], pseudo_well_values]).astype(np.float64)
    evaluation_mask = np.zeros_like(validation_mask, dtype=bool)
    evaluation_mask[validation_global_indices] = True
    pseudo_well_eval_local_indices = np.flatnonzero(np.isin(validation_global_indices, pseudo_well_global_indices))
    common_metric_mask, pseudo_test_distances, distance_edges = r01._metric_mask_and_bands(  # noqa: SLF001
        geometry, evaluation_mask, pseudo_well_eval_local_indices
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "task_id": TASK_TASK_IDS[mode],
        "root_seed": ROOT_SEED,
        "development_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "sources": [
            {
                "path": "_pipelines/02_task_datasets/reconstruction/build_summary.json",
                "sha256": hash_file(HERE / "build_summary.json"),
                "bytes": (HERE / "build_summary.json").stat().st_size,
            },
            {
                "path": "_pipelines/02_task_datasets/reconstruction/model_inspection.json",
                "sha256": hash_file(HERE / "model_inspection.json"),
                "bytes": (HERE / "model_inspection.json").stat().st_size,
            },
            {
                "path": "_pipelines/02_task_datasets/reconstruction/reconstruction_p5_r01.py",
                "sha256": hash_file(HERE / "reconstruction_p5_r01.py"),
                "bytes": (HERE / "reconstruction_p5_r01.py").stat().st_size,
            },
        ],
        "geometry": {
            "development_i_blocks": list(p4.protocol(mode).development_i_blocks),
            "train_h5_opened": True,
            "test_h5_opened": False,
            "physical_test_h5_opened": False,
            "reference_sparse_poro_channel_6_read": False,
            "well_log_seq_read": False,
        },
        "split_manifest": split_manifest,
        "selection": {
            "train_points": int(train_global_indices.size),
            "validation_points": int(validation_global_indices.size),
            "pseudo_well_points": int(pseudo_well_global_indices.size),
            "train_selection_hash": selection["train_selection_hash"],
            "validation_selection_hash": selection["validation_selection_hash"],
            "pseudo_well_selection_hash": selection["pseudo_well_selection_hash"],
        },
        "label_audit": target_audit,
        "condition_audit": {
            "well_conditions_are_synthetic_reference_revealed_eclipse_samples": True,
            "well_conditions_are_not_measured_phie": True,
            "pseudo_test_constraints_used_only_for_b1_and_shuffled": mode == "conditional",
            "strict_refuses_b1_and_shuffled": mode == "strict",
        },
        "hashes": {
            "split_hash": split_manifest["split_hash"],
            "feature_hash": hash_payload(_feature_names(mode)),
            "train_constraint_hash": r01._hash_arrays(values=train_constraints),  # noqa: SLF001
            "pseudo_test_constraint_hash": r01._hash_arrays(values=pseudo_test_constraints),  # noqa: SLF001
            "common_metric_mask_hash": r01._hash_arrays(indices_kji=geometry.indices_kji[common_metric_mask]),  # noqa: SLF001
        },
        "support": {
            "development_cells": int(geometry.indices_kji.shape[0]),
            "train_cells": int(train_mask.sum()),
            "validation_cells": int(validation_mask.sum()),
            "common_metric_cells": int(np.asarray(common_metric_mask[validation_global_indices], dtype=bool).sum()),
            "exact_pseudo_test_cells_excluded_from_metrics": int(pseudo_well_global_indices.size),
        },
        "test_firewall": {
            "physical_test_h5_opened": False,
            "known_or_frozen_arrays_read": False,
            "known_or_frozen_metrics_read": False,
            "known_or_frozen_predictions_read": False,
            "global_well_log_seq_read": False,
            "only_physical_train_h5_development_channels": True,
        },
    }
    bundle = ModeBundle(
        mode=mode,
        geometry=geometry,
        split_manifest=split_manifest,
        train_mask=train_mask,
        validation_mask=validation_mask,
        train_local_indices=train_local_indices,
        validation_local_indices=validation_local_indices,
        train_global_indices=train_global_indices,
        validation_global_indices=validation_global_indices,
        pseudo_well_local_indices=pseudo_well_local_indices,
        pseudo_well_global_indices=pseudo_well_global_indices,
        train_target=train_target,
        validation_target=validation_target,
        pseudo_well_values=pseudo_well_values,
        common_metric_mask=common_metric_mask,
        pseudo_test_distances=pseudo_test_distances,
        distance_edges=distance_edges,
        train_constraints=train_constraints,
        pseudo_test_constraints=pseudo_test_constraints,
        access_audit=geometry.access_audit,
        manifest=manifest,
    )
    records: list[Mapping[str, Any]] = []
    for model_id in MODEL_NAMES:
        for loss_name in LOSS_NAMES:
            for budget in BUDGETS:
                for feature_variant in FEATURE_VARIANTS[mode]:
                    try:
                        record = _run_one_cell(
                            mode=mode,
                            model_id=model_id,
                            loss_name=loss_name,
                            budget=budget,
                            feature_variant=feature_variant,
                            bundle=bundle,
                            output_root=output_dir,
                        )
                    except Exception as exc:  # fail-closed structured evidence
                        record = {
                            "schema_version": SCHEMA_VERSION,
                            "track_id": "reconstruction",
                            "mode": mode,
                            "task_id": TASK_TASK_IDS[mode],
                            "model_id": model_id,
                            "loss_name": loss_name,
                            "feature_variant": feature_variant,
                            "budget_updates": budget,
                            "root_seed": ROOT_SEED,
                            "split_hash": split_manifest["split_hash"],
                            "status": "blocked",
                            "reason": {
                                "code": type(exc).__name__,
                                "message": str(exc),
                            },
                            "fresh_blind": False,
                            "field_generalization": False,
                            "development_protocol_mechanism_only": True,
                        }
                        record["result_hash"] = hash_payload(record)
                        atomic_write_json(
                            output_dir / "cells" / mode / model_id / loss_name / feature_variant / f"updates_{budget:03d}" / "status.json",
                            record,
                        )
                    records.append(record)
    mode_summary = _write_mode_outputs(mode, records, output_dir)
    atomic_write_json(output_dir / f"run_manifest_{mode}.json", {**manifest, "mode_summary_hash": mode_summary["summary_hash"]})
    return mode_summary


def collate(output_dir: Path) -> Mapping[str, Any]:
    strict_summary = json.loads((output_dir / "strict" / "summary.json").read_text(encoding="utf-8"))
    conditional_summary = json.loads((output_dir / "conditional" / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "development_only": True,
        "fresh_blind": False,
        "field_generalization": False,
        "modes_are_independent": True,
        "shared_k4_block": SHARED_PSEUDO_FOLD_ID,
        "shared_k4_metric_mask_count": strict_summary["shared_k4_metric_mask_count"],
        "modes": {
            "strict": {
                "summary": "p5_r2_evidence/strict/summary.json",
                "results": "p5_r2_evidence/strict/results.jsonl",
                "leaderboard": "p5_r2_evidence/strict/leaderboard.json",
                "visualization_manifest": "p5_r2_evidence/strict/visualization_manifest.json",
                "oof_manifest": "p5_r2_evidence/strict/oof_manifest.json",
            },
            "conditional": {
                "summary": "p5_r2_evidence/conditional/summary.json",
                "results": "p5_r2_evidence/conditional/results.jsonl",
                "leaderboard": "p5_r2_evidence/conditional/leaderboard.json",
                "visualization_manifest": "p5_r2_evidence/conditional/visualization_manifest.json",
                "oof_manifest": "p5_r2_evidence/conditional/oof_manifest.json",
            },
        },
        "r3_allowed": bool(strict_summary["r3_allowed"] and conditional_summary["r3_allowed"]),
        "strict": strict_summary,
        "conditional": conditional_summary,
    }
    summary["summary_hash"] = hash_payload(summary)
    atomic_write_json(output_dir / "p5_r2_summary.json", summary)
    atomic_write_json(
        output_dir / "p5_r2_oof_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "root_seed": ROOT_SEED,
            "development_only": True,
            "fresh_blind": False,
            "strict": json.loads((output_dir / "strict" / "oof_manifest.json").read_text(encoding="utf-8")),
            "conditional": json.loads((output_dir / "conditional" / "oof_manifest.json").read_text(encoding="utf-8")),
        },
    )
    atomic_write_json(
        output_dir / "p5_r2_visualization_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "root_seed": ROOT_SEED,
            "development_only": True,
            "fresh_blind": False,
            "strict": json.loads((output_dir / "strict" / "visualization_manifest.json").read_text(encoding="utf-8")),
            "conditional": json.loads((output_dir / "conditional" / "visualization_manifest.json").read_text(encoding="utf-8")),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-mode", help="run one development-only mode")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--data-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=HERE / "p5_r2_evidence")
    aggregate = subparsers.add_parser("collate", help="validate both mode outputs and summarize")
    aggregate.add_argument("--output-dir", type=Path, default=HERE / "p5_r2_evidence")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run-mode":
        print(json.dumps(run_mode(args.mode, args.data_dir, args.output_dir), indent=2))
    elif args.command == "collate":
        print(json.dumps(collate(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
