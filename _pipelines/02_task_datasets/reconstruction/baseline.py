#!/usr/bin/env python3
"""Registered linear Ridge baseline using the shared ``ml_framework``.

The geological method is unchanged: interpolate sparse well porosity with 3-D
IDW, then fit a linear L2-regularised model from IDW, coordinates and seismic
attributes.  Pipeline plumbing (normalisation, epoch loop, best checkpoint and
loss plot) is delegated to the shared framework.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from dataset_io import load_dataset  # noqa: E402
from ml_framework.model_registry import get_model  # noqa: E402
from ml_framework.preprocess import (  # noqa: E402
    NormStats,
    denoise_identity,
    denormalize,
    fit_minmax,
    fit_zscore,
    normalize,
)
from ml_framework.train import TrainHistory, train_loop  # noqa: E402
from ml_framework.visualize import plot_loss_curve  # noqa: E402


STRUCTURAL_FEATURE_NAMES = [
    "idw_porosity",
    "x_normalized",
    "y_normalized",
    "depth_normalized",
]
SEISMIC_FEATURE_NAMES = [
    "idw_porosity",
    "seismic_amplitude",
    "seismic_local_rms",
    "seismic_vertical_gradient",
    "x_normalized",
    "y_normalized",
    "depth_normalized",
]

EVALUATION_MODES = ("conditional", "strict")
MODE_I_BLOCKS = {
    "conditional": {
        "train": (0, 1, 2, 3),
        "guard": (),
        "test": (4, 5),
    },
    # Deterministic reverse spatial split.  The eastern train region contains
    # the 90 usable SR constraints, I-block 3 is a guard, and the western test
    # region contains no well constraints.  This is spatial extrapolation from
    # one physical well, not cross-well or cross-field generalisation.
    "strict": {
        "train": (4, 5),
        "guard": (3,),
        "test": (0, 1, 2),
    },
}


@dataclass
class FlatSplit:
    seismic: np.ndarray
    coordinates: np.ndarray
    observed_mask: np.ndarray
    target: np.ndarray
    well_sequence: np.ndarray
    cell_patch_i_block: np.ndarray
    cell_patch_k_block: np.ndarray
    patch_i_blocks: set[int]
    n_samples: int


def _iter_region_samples(i_blocks: tuple[int, ...]):
    """Yield unified-interface samples in deterministic order for selected I-blocks."""
    allowed = set(i_blocks)
    for source_split in ("train", "test"):
        for sample in load_dataset("reconstruction", source_split):
            patch_i = int(sample["meta"]["patch_index_kji"][2])
            if patch_i in allowed:
                yield sample


def _flatten_region(i_blocks: tuple[int, ...]) -> FlatSplit:
    if not i_blocks:
        raise ValueError("cannot flatten an empty region")
    seismic: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    target: list[np.ndarray] = []
    cell_patch_i_block: list[np.ndarray] = []
    cell_patch_k_block: list[np.ndarray] = []
    patch_i_blocks: set[int] = set()
    canonical_wells: np.ndarray | None = None
    n_samples = 0
    for sample in _iter_region_samples(i_blocks):
        n_samples += 1
        patch = np.asarray(sample["seismic_patch"], dtype=np.float32)
        label = np.asarray(sample["label"], dtype=np.float32)
        if patch.ndim != 4 or patch.shape[0] != 9 or label.shape != patch.shape[1:]:
            raise ValueError(f"unexpected sample shapes: patch={patch.shape}, label={label.shape}")
        active = patch[8].ravel() > 0.5
        raw_seismic = patch[0:3].reshape(3, -1).T[active]
        # Explicit shared no-op: sharp reflectors/discontinuities may be real
        # geology, so no generic smoothing is applied by default.
        seismic.append(denoise_identity(raw_seismic))
        coordinates.append(patch[3:6].reshape(3, -1).T[active])
        observed.append((patch[7].ravel() > 0.5)[active])
        target.append(label.ravel()[active])
        patch_k = int(sample["meta"]["patch_index_kji"][0])
        patch_i = int(sample["meta"]["patch_index_kji"][2])
        cell_patch_i_block.append(np.full(int(active.sum()), patch_i, dtype=np.int8))
        cell_patch_k_block.append(np.full(int(active.sum()), patch_k, dtype=np.int8))
        wells = np.asarray(sample["well_log_seq"], dtype=np.float32)
        if canonical_wells is None:
            canonical_wells = wells
        elif not np.array_equal(canonical_wells, wells):
            raise ValueError("well_log_seq differs between patches")
        patch_i_blocks.add(patch_i)
    if not target or canonical_wells is None:
        raise ValueError(f"no active reconstruction samples in I-blocks {i_blocks}")
    return FlatSplit(
        seismic=np.concatenate(seismic),
        coordinates=np.concatenate(coordinates),
        observed_mask=np.concatenate(observed),
        target=np.concatenate(target),
        well_sequence=canonical_wells,
        cell_patch_i_block=np.concatenate(cell_patch_i_block),
        cell_patch_k_block=np.concatenate(cell_patch_k_block),
        patch_i_blocks=patch_i_blocks,
        n_samples=n_samples,
    )


def _flatten_split(split: str) -> FlatSplit:
    """Backward-compatible loader for the historical conditional source splits."""
    if split == "train":
        return _flatten_region(MODE_I_BLOCKS["conditional"]["train"])
    if split == "test":
        return _flatten_region(MODE_I_BLOCKS["conditional"]["test"])
    raise ValueError(f"unknown split {split!r}")


def _well_rows_for_region(region: FlatSplit, global_wells: np.ndarray) -> np.ndarray:
    """Select global well rows whose exact sparse cells lie inside ``region``."""
    observed_xyz = np.asarray(region.coordinates[region.observed_mask], dtype=np.float32)
    observed_target = np.asarray(region.target[region.observed_mask], dtype=np.float32)
    if observed_xyz.size == 0:
        return np.empty((0, global_wells.shape[1]), dtype=np.float32)
    observed = {
        tuple(xyz.tolist()): float(value) for xyz, value in zip(observed_xyz, observed_target)
    }
    selected = []
    for row in np.asarray(global_wells, dtype=np.float32):
        key = tuple(row[0:3].tolist())
        if key in observed:
            if not np.isclose(float(row[3]), observed[key], rtol=0.0, atol=1e-7):
                raise ValueError("well_sequence porosity disagrees with sparse channel target")
            selected.append(row)
    result = np.asarray(selected, dtype=np.float32)
    if result.shape != (observed_xyz.shape[0], global_wells.shape[1]):
        raise ValueError(
            "well rows do not match exact sparse cells in region: "
            f"rows={result.shape[0]}, cells={observed_xyz.shape[0]}"
        )
    return result


def load_evaluation_regions(
    evaluation_mode: str,
) -> tuple[FlatSplit, FlatSplit | None, FlatSplit, np.ndarray, dict]:
    """Load mode-specific train/guard/test regions and the only allowed IDW rows."""
    if evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"evaluation_mode must be one of {EVALUATION_MODES}")
    blocks = MODE_I_BLOCKS[evaluation_mode]
    train = _flatten_region(blocks["train"])
    guard = _flatten_region(blocks["guard"]) if blocks["guard"] else None
    test = _flatten_region(blocks["test"])
    global_wells = train.well_sequence
    if not np.array_equal(global_wells, test.well_sequence):
        raise ValueError("train/test regions do not carry the same source well table")
    if guard is not None and not np.array_equal(global_wells, guard.well_sequence):
        raise ValueError("guard region does not carry the same source well table")

    train_rows = _well_rows_for_region(train, global_wells)
    guard_rows = (
        _well_rows_for_region(guard, global_wells)
        if guard is not None
        else np.empty((0, global_wells.shape[1]), dtype=np.float32)
    )
    test_rows = _well_rows_for_region(test, global_wells)
    if train_rows.shape[0] + guard_rows.shape[0] + test_rows.shape[0] != global_wells.shape[0]:
        raise ValueError("train/guard/test regions do not account for all global well constraints")

    train_blocks = set(train.patch_i_blocks)
    guard_blocks = set() if guard is None else set(guard.patch_i_blocks)
    test_blocks = set(test.patch_i_blocks)
    if train_blocks & guard_blocks or train_blocks & test_blocks or guard_blocks & test_blocks:
        raise ValueError("train/guard/test spatial I-blocks overlap")

    if evaluation_mode == "conditional":
        allowed_wells = global_wells
        strict_holdout = False
    else:
        allowed_wells = train_rows
        strict_holdout = True
        if guard_rows.shape[0] == 0:
            raise ValueError("strict guard unexpectedly contains no excluded constraint")
        if test_rows.shape[0] != 0:
            raise ValueError("strict test region contains well values and cannot be evaluated")
        if allowed_wells.shape[0] == 0:
            raise ValueError("strict train region has no usable well constraints")

    audit = {
        "evaluation_mode": evaluation_mode,
        "train_i_blocks": sorted(train_blocks),
        "guard_i_blocks": sorted(guard_blocks),
        "test_i_blocks": sorted(test_blocks),
        "spatial_regions_zero_overlap": True,
        "n_global_well_constraints": int(global_wells.shape[0]),
        "n_train_region_well_constraints": int(train_rows.shape[0]),
        "n_guard_region_well_constraints": int(guard_rows.shape[0]),
        "n_test_region_well_constraints": int(test_rows.shape[0]),
        "n_well_constraints_supplied_to_idw": int(allowed_wells.shape[0]),
        "guard_well_constraints_supplied_to_idw": False,
        "test_well_constraints_supplied_to_idw": bool(
            evaluation_mode == "conditional" and test_rows.shape[0] > 0
        ),
        "strict_spatial_holdout_generalization": strict_holdout,
        "cross_well_generalization": False,
        "cross_field_generalization": False,
    }
    return train, guard, test, allowed_wells, audit


def _idw_predict(coordinates: np.ndarray, well_sequence: np.ndarray, k: int = 8) -> np.ndarray:
    well_xyz = np.asarray(well_sequence[:, 0:3], dtype=np.float64)
    well_poro = np.asarray(well_sequence[:, 3], dtype=np.float64)
    if well_xyz.shape[0] == 0:
        raise ValueError("no sparse well constraints")
    anisotropy = np.array([1.0, 1.0, 3.0], dtype=np.float64)
    tree = cKDTree(well_xyz * anisotropy)
    distance, indices = tree.query(
        np.asarray(coordinates, dtype=np.float64) * anisotropy,
        k=min(k, well_xyz.shape[0]),
    )
    if distance.ndim == 1:
        distance = distance[:, None]
        indices = indices[:, None]
    exact = distance < 1e-10
    weights = 1.0 / np.maximum(distance, 1e-6) ** 2
    prediction = np.sum(weights * well_poro[indices], axis=1) / np.sum(weights, axis=1)
    exact_rows = np.flatnonzero(exact.any(axis=1))
    if exact_rows.size:
        first_exact = np.argmax(exact[exact_rows], axis=1)
        prediction[exact_rows] = well_poro[indices[exact_rows, first_exact]]
    return prediction.astype(np.float64)


def _roundtrip_audit(
    name: str,
    original: np.ndarray,
    transformed: np.ndarray,
    stats: NormStats,
) -> dict[str, float | bool]:
    restored = np.asarray(denormalize(transformed, stats), dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    max_abs_error = float(np.max(np.abs(restored - original)))
    scale = max(1.0, float(np.max(np.abs(original))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    passed = bool(max_abs_error <= tolerance)
    if not passed:
        raise ValueError(
            f"shared normalisation round-trip failed for {name}: "
            f"{max_abs_error} > {tolerance}"
        )
    return {
        "passed": passed,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
    }


def _fit_feature_normalization(
    train_features: np.ndarray,
    named_feature_sets: dict[str, np.ndarray],
    feature_names: list[str],
) -> tuple[dict[str, np.ndarray], list[NormStats], dict]:
    train_features = np.asarray(train_features, dtype=np.float64)
    stats = [fit_zscore(train_features[:, column]) for column in range(train_features.shape[1])]
    transformed_sets: dict[str, np.ndarray] = {}
    audits: dict[str, dict] = {}
    for set_name, values in named_feature_sets.items():
        values = np.asarray(values, dtype=np.float64)
        transformed = np.column_stack(
            [normalize(values[:, column], stats[column]) for column in range(values.shape[1])]
        )
        transformed_sets[set_name] = transformed
        restored = np.column_stack(
            [denormalize(transformed[:, column], stats[column]) for column in range(values.shape[1])]
        )
        max_abs_error = float(np.max(np.abs(restored - values)))
        scale = max(1.0, float(np.max(np.abs(values))))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        if max_abs_error > tolerance:
            raise ValueError(
                f"shared zscore round-trip failed for {set_name}: {max_abs_error} > {tolerance}"
            )
        audits[set_name] = {
            "passed": True,
            "max_abs_error": max_abs_error,
            "tolerance": tolerance,
        }
    return transformed_sets, stats, {
        "method": "ml_framework.preprocess.fit_zscore + normalize/denormalize",
        "fit_scope": "optimization train cells only",
        "feature_names": feature_names,
        "stats": [item.to_dict() for item in stats],
        "roundtrip": audits,
    }


def _fit_target_normalization(
    optimization_target: np.ndarray,
    named_targets: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], NormStats, dict]:
    stats = fit_minmax(np.asarray(optimization_target, dtype=np.float64))
    transformed = {}
    audits = {}
    for name, values in named_targets.items():
        values = np.asarray(values, dtype=np.float64)
        normalized = np.asarray(normalize(values, stats), dtype=np.float64)
        transformed[name] = normalized
        audits[name] = _roundtrip_audit(name, values, normalized, stats)
    return transformed, stats, {
        "method": "ml_framework.preprocess.fit_minmax + normalize/denormalize",
        "fit_scope": "optimization train targets only",
        "stats": stats.to_dict(),
        "roundtrip": audits,
    }


def _train_registered_model(
    model_name: str,
    train_features: np.ndarray,
    train_target: np.ndarray,
    val_features: np.ndarray,
    val_target: np.ndarray,
    epochs: int,
    learning_rate: float,
    ridge_alpha: float,
    output_dir: Path,
) -> tuple[object, TrainHistory, dict]:
    model = get_model(
        model_name,
        models_package="models",
        n_features=train_features.shape[1],
        learning_rate=learning_rate,
        ridge_alpha=ridge_alpha,
        n_training_samples=train_target.size,
    )
    checkpoint_dir = output_dir / "checkpoints"
    history = train_loop(
        model=model,
        train_step_fn=model.train_batch,
        val_step_fn=model.validation_loss,
        # Each call constructs a fresh list.  The arrays are intentionally
        # shared/read-only inputs, while the iterable itself is never reused or
        # exhausted across epochs.
        train_batches_fn=lambda: [(train_features, train_target)],
        val_batches_fn=lambda: [(val_features, val_target)],
        epochs=epochs,
        save_checkpoint_fn=lambda current_model, path: current_model.save_checkpoint(path),
        checkpoint_dir=checkpoint_dir,
        min_epochs_before_early_check=50,
    )
    if history.best_val_loss != min(history.val_loss):
        raise ValueError("shared train_loop best validation loss is inconsistent with history")
    best_path = checkpoint_dir / "best.ckpt"
    last_path = checkpoint_dir / "last.ckpt"
    model.load_checkpoint(best_path)
    curve_path = plot_loss_curve(history, output_dir / "loss_curve.png")
    zoom_curve_path = _plot_best_epoch_zoom(history, output_dir / "loss_curve_best_epoch_zoom.png")
    persistent_window = min(50, max(10, epochs // 10))
    enough_post_best_epochs = history.best_epoch <= epochs - persistent_window - 1
    tail_val = np.asarray(history.val_loss[-persistent_window:], dtype=np.float64)
    tail_train = np.asarray(history.train_loss[-persistent_window:], dtype=np.float64)
    elevation_threshold = history.best_val_loss * 1.01
    elevated_mask = tail_val > elevation_threshold
    persistent_val_elevation = bool(enough_post_best_epochs and np.all(elevated_mask))
    persistent_train_decrease = bool(
        enough_post_best_epochs
        and float(tail_train.mean()) < history.train_loss[history.best_epoch]
    )
    curve_visible = bool(persistent_val_elevation and persistent_train_decrease)
    return model, history, {
        "best_epoch_1_based": history.best_epoch + 1,
        "best_val_loss_normalized_mse": history.best_val_loss,
        "last_val_loss_normalized_mse": history.val_loss[-1],
        "train_loss_first": history.train_loss[0],
        "train_loss_at_best": history.train_loss[history.best_epoch],
        "train_loss_last": history.train_loss[-1],
        "underfit_to_overfit_curve_visible": curve_visible,
        "persistent_trend_check": {
            "window_epochs": persistent_window,
            "window_start_epoch_1_based": epochs - persistent_window + 1,
            "window_end_epoch_1_based": epochs,
            "required_val_elevation_vs_best_percent": 1.0,
            "fraction_of_window_above_threshold": float(np.mean(elevated_mask)),
            "minimum_val_loss_ratio_to_best_in_window": float(
                tail_val.min() / history.best_val_loss
            ),
            "all_window_epochs_elevated": persistent_val_elevation,
            "mean_train_loss_below_train_loss_at_best": persistent_train_decrease,
        },
        "best_checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
        "last_checkpoint": str(last_path.relative_to(PROJECT_ROOT)),
        "history": str((checkpoint_dir / "history.json").relative_to(PROJECT_ROOT)),
        "loss_curve": str(curve_path.relative_to(PROJECT_ROOT)),
        "loss_curve_best_epoch_zoom": str(zoom_curve_path.relative_to(PROJECT_ROOT)),
    }


def _plot_best_epoch_zoom(
    history: TrainHistory,
    out_path: Path,
    radius: int = 75,
) -> Path:
    """Supplement the shared full curve with a validation-loss view near its minimum."""
    if history.best_epoch < 0:
        raise ValueError("cannot zoom a history without a best epoch")
    start = max(0, history.best_epoch - radius)
    stop = min(len(history.val_loss), history.best_epoch + radius + 1)
    epoch_numbers = np.arange(start + 1, stop + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epoch_numbers, history.val_loss[start:stop], label="val loss", color="tab:orange")
    ax.axvline(
        history.best_epoch + 1,
        color="gray",
        linestyle="--",
        label=f"best epoch={history.best_epoch + 1}",
    )
    ax.scatter(
        [history.best_epoch + 1],
        [history.best_val_loss],
        color="black",
        s=24,
        zorder=3,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss near best epoch")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - target
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.sum((target - target.mean()) ** 2))
    r2_defined = bool(denom > 0)
    r2 = float(1.0 - np.sum(error**2) / denom) if r2_defined else None
    if target.std() > 0 and prediction.std() > 0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
        correlation_defined = True
        correlation_reason = None
    else:
        correlation = None
        correlation_defined = False
        correlation_reason = (
            "Pearson correlation is undefined because target or prediction has zero variance"
        )
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": correlation,
        "pearson_r_defined": correlation_defined,
        "pearson_r_reason": correlation_reason,
        "r2": r2,
        "r2_defined": r2_defined,
        "r2_reason": None if r2_defined else "R2 is undefined because target has zero variance",
    }
    numeric = [value for value in (rmse, mae, correlation, r2) if value is not None]
    if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
        raise ValueError(f"non-finite final metric detected: {metrics}")
    return metrics


def _clip_from_train(prediction: np.ndarray, train_target: np.ndarray) -> np.ndarray:
    return np.clip(prediction, float(train_target.min()), float(train_target.max()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_run_manifest(
    evaluation_mode: str,
    model_name: str,
    epochs: int,
    results_path: Path,
    results: dict,
) -> Path:
    """Persist enough hashes and environment facts to audit a mode-specific rerun."""
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_paths = [
        HERE / "baseline.py",
        HERE / "models" / f"{model_name}.py",
        PROJECT_ROOT / "_code/ml_framework/preprocess.py",
        PROJECT_ROOT / "_code/ml_framework/model_registry.py",
        PROJECT_ROOT / "_code/ml_framework/train.py",
        PROJECT_ROOT / "_code/ml_framework/visualize.py",
        PROJECT_ROOT / "_code/dataset_io.py",
    ]
    data_paths = [
        PROJECT_ROOT / "_data/processed/reconstruction/train.h5",
        PROJECT_ROOT / "_data/processed/reconstruction/test.h5",
    ]
    artifact_paths = [results_path, PROJECT_ROOT / results["preprocessing"]["stats_file"]]
    for variant in ("structural", "seismic"):
        training = results["training"][variant]
        artifact_paths.extend(
            PROJECT_ROOT / training[key]
            for key in (
                "best_checkpoint",
                "last_checkpoint",
                "history",
                "loss_curve",
                "loss_curve_best_epoch_zoom",
            )
        )
    manifest = {
        "task": "reconstruction",
        "evaluation_mode": evaluation_mode,
        "command": (
            "python3 _pipelines/02_task_datasets/reconstruction/baseline.py "
            f"--model {model_name} --evaluation-mode {evaluation_mode} --epochs {epochs}"
        ),
        "git_head": git_head,
        "note": (
            "reconstruction sources are currently untracked; source SHA-256 values below, not "
            "git_head alone, identify the executed code"
        ),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "matplotlib", "h5py")
        },
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths
        },
        "input_data_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in data_paths
        },
        "artifact_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in artifact_paths
        },
    }
    manifest_path = HERE / f"run_manifest_{evaluation_mode}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run_baseline(
    model_name: str = "ridge_linear",
    evaluation_mode: str = "conditional",
    epochs: int = 600,
    learning_rate: float = 0.01,
    ridge_alpha: float = 10.0,
) -> dict:
    if epochs < 50:
        raise ValueError("epochs must be >= 50; short runs cannot expose the validation-loss turn")
    train, guard, test, allowed_wells, constraint_audit = load_evaluation_regions(
        evaluation_mode
    )

    train_idw = _idw_predict(train.coordinates, allowed_wells)
    test_idw = _idw_predict(test.coordinates, allowed_wells)
    full_train_structural = np.column_stack([train_idw, train.coordinates])
    full_test_structural = np.column_stack([test_idw, test.coordinates])
    full_train_seismic = np.column_stack([train_idw, train.seismic, train.coordinates])
    full_test_seismic = np.column_stack([test_idw, test.seismic, test.coordinates])

    # Hold out one central depth block for validation while retaining all four
    # train I-blocks in optimization.  This preserves the original horizontal
    # reconstruction domain and gives a genuinely disjoint vertical validation
    # slab on which the underfit-to-overfit turn is observable.
    validation_mask = train.cell_patch_k_block == 3
    optimization_mask = ~validation_mask
    if not optimization_mask.any() or not validation_mask.any():
        raise ValueError("blocked internal train/validation split is empty")

    structural_sets, _, structural_preprocess = _fit_feature_normalization(
        full_train_structural[optimization_mask],
        {
            "optimization_train": full_train_structural[optimization_mask],
            "validation": full_train_structural[validation_mask],
            "test": full_test_structural,
        },
        STRUCTURAL_FEATURE_NAMES,
    )
    seismic_sets, _, seismic_preprocess = _fit_feature_normalization(
        full_train_seismic[optimization_mask],
        {
            "optimization_train": full_train_seismic[optimization_mask],
            "validation": full_train_seismic[validation_mask],
            "test": full_test_seismic,
        },
        SEISMIC_FEATURE_NAMES,
    )
    target_sets, target_stats, target_preprocess = _fit_target_normalization(
        train.target[optimization_mask],
        {
            "optimization_train": train.target[optimization_mask],
            "validation": train.target[validation_mask],
            "test": test.target,
        },
    )

    model_output = HERE / "_outputs" / model_name / evaluation_mode
    structural_model, structural_history, structural_training = _train_registered_model(
        model_name,
        structural_sets["optimization_train"],
        target_sets["optimization_train"],
        structural_sets["validation"],
        target_sets["validation"],
        epochs,
        learning_rate,
        ridge_alpha,
        model_output / "structural",
    )
    seismic_model, seismic_history, seismic_training = _train_registered_model(
        model_name,
        seismic_sets["optimization_train"],
        target_sets["optimization_train"],
        seismic_sets["validation"],
        target_sets["validation"],
        epochs,
        learning_rate,
        ridge_alpha,
        model_output / "seismic",
    )
    if not seismic_training["underfit_to_overfit_curve_visible"]:
        raise RuntimeError(
            "validation loss did not show a >1% post-best rise; increase --epochs rather than "
            "reporting a truncated training curve"
        )

    structural_prediction = _clip_from_train(
        denormalize(structural_model.predict(structural_sets["test"]), target_stats),
        train.target,
    )
    seismic_prediction = _clip_from_train(
        denormalize(seismic_model.predict(seismic_sets["test"]), target_stats),
        train.target,
    )
    mean_prediction = np.full(test.target.shape, float(train.target.mean()), dtype=np.float64)
    eval_mask = ~test.observed_mask
    if not eval_mask.any():
        raise ValueError("all test cells are direct well observations")

    rng = np.random.default_rng(20260710)
    shuffled_features = seismic_sets["test"].copy()
    permutation = rng.permutation(shuffled_features.shape[0])
    shuffled_features[:, 1:4] = shuffled_features[permutation, 1:4]
    shuffled_prediction = _clip_from_train(
        denormalize(seismic_model.predict(shuffled_features), target_stats), train.target
    )

    structural_metric_key = (
        "ridge_idw_coordinates" if model_name == "ridge_linear" else f"{model_name}_idw_coordinates"
    )
    seismic_metric_key = (
        "ridge_idw_seismic_coordinates"
        if model_name == "ridge_linear"
        else f"{model_name}_idw_seismic_coordinates"
    )
    shuffled_metric_key = (
        "ridge_with_test_seismic_shuffled"
        if model_name == "ridge_linear"
        else f"{model_name}_with_test_seismic_shuffled"
    )
    models_metrics = {
        "train_mean": _metrics(test.target[eval_mask], mean_prediction[eval_mask]),
        "sparse_well_idw": _metrics(test.target[eval_mask], test_idw[eval_mask]),
        structural_metric_key: _metrics(
            test.target[eval_mask], structural_prediction[eval_mask]
        ),
        seismic_metric_key: _metrics(
            test.target[eval_mask], seismic_prediction[eval_mask]
        ),
        shuffled_metric_key: _metrics(
            test.target[eval_mask], shuffled_prediction[eval_mask]
        ),
    }
    defined_numeric_metrics = [
        value
        for metrics in models_metrics.values()
        for key, value in metrics.items()
        if key in ("rmse", "mae", "pearson_r", "r2") and value is not None
    ]
    all_final_metrics_finite = bool(
        np.isfinite(np.asarray(defined_numeric_metrics, dtype=np.float64)).all()
    )
    if not all_final_metrics_finite:
        raise ValueError("one or more reported final metrics are non-finite")

    preprocess_report = {
        "denoise": {
            "function": "ml_framework.preprocess.denoise_identity",
            "reason": "sharp seismic/well-log features may be real geology; no smoothing by default",
        },
        "structural_features": structural_preprocess,
        "seismic_features": seismic_preprocess,
        "target": target_preprocess,
    }
    preprocess_path = model_output / "preprocess_stats.json"
    preprocess_path.parent.mkdir(parents=True, exist_ok=True)
    preprocess_path.write_text(
        json.dumps(preprocess_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = {
        "task": "reconstruction",
        "target": "Volve Eclipse final PORO",
        "evaluation_mode": evaluation_mode,
        "evaluation_protocol": (
            "conditional reconstruction given global well constraints; NOT strict spatial "
            "holdout generalization"
            if evaluation_mode == "conditional"
            else "strict spatial holdout: test prediction uses train-region well constraints "
            "only; one-well spatial extrapolation, NOT cross-well/cross-field generalization"
        ),
        "primary_baseline": seismic_metric_key,
        "registered_model": model_name,
        "model_plugin": type(seismic_model).__module__,
        "train": {
            "n_samples": train.n_samples,
            "n_active_cells": int(train.target.size),
            "patch_i_blocks": sorted(train.patch_i_blocks),
            "optimization_patch_i_blocks": sorted(train.patch_i_blocks),
            "validation_patch_k_blocks": [3],
            "optimization_patch_k_blocks": [0, 1, 2, 4, 5, 6],
            "n_optimization_cells": int(optimization_mask.sum()),
            "n_validation_cells": int(validation_mask.sum()),
            "target_mean": float(train.target.mean()),
            "target_min": float(train.target.min()),
            "target_max": float(train.target.max()),
        },
        "test": {
            "n_samples": test.n_samples,
            "n_active_cells": int(test.target.size),
            "patch_i_blocks": sorted(test.patch_i_blocks),
            "n_direct_well_cells_excluded_from_metrics": int(test.observed_mask.sum()),
            "n_evaluation_cells": int(eval_mask.sum()),
        },
        "guard": {
            "n_samples": 0 if guard is None else guard.n_samples,
            "n_active_cells": 0 if guard is None else int(guard.target.size),
            "patch_i_blocks": [] if guard is None else sorted(guard.patch_i_blocks),
            "used_for_training_or_evaluation": False,
        },
        "training": {
            "framework": "ml_framework.train.train_loop",
            "epochs_completed": epochs,
            "learning_rate": learning_rate,
            "ridge_alpha": ridge_alpha,
            "early_stopping": False,
            "checkpoint_selection": "minimum validation loss",
            "batch_factory": "zero-arg lambda returning a fresh one-batch list per epoch",
            "pre_fix_batch_iterable_audit": (
                "previous code passed reusable Python lists, not one-shot generators; "
                "the exhausted-generator bug did not affect the prior histories"
            ),
            "structural": structural_training,
            "seismic": seismic_training,
        },
        "preprocessing": {
            "framework": "ml_framework.preprocess",
            "stats_file": str(preprocess_path.relative_to(PROJECT_ROOT)),
            "all_roundtrip_checks_passed": True,
        },
        "n_sparse_well_observations": int(allowed_wells.shape[0]),
        "n_intersecting_wells": int(np.unique(allowed_wells[:, 7]).size),
        "models": models_metrics,
        "all_final_metrics_finite": all_final_metrics_finite,
        "undefined_metric_policy": (
            "mathematically undefined Pearson/R2 values are null with an explicit defined flag "
            "and reason; they are never replaced by 0"
        ),
        "seismic_rmse_delta_vs_structural_model": float(
            models_metrics[structural_metric_key]["rmse"]
            - models_metrics[seismic_metric_key]["rmse"]
        ),
        "primary_rmse_improvement_vs_train_mean_percent": float(
            100.0
            * (
                models_metrics["train_mean"]["rmse"]
                - models_metrics[seismic_metric_key]["rmse"]
            )
            / models_metrics["train_mean"]["rmse"]
        ),
        "seismic_rmse_improvement_vs_structural_model_percent": float(
            100.0
            * (
                models_metrics[structural_metric_key]["rmse"]
                - models_metrics[seismic_metric_key]["rmse"]
            )
            / models_metrics[structural_metric_key]["rmse"]
        ),
        "leakage_checks": {
            "protocol": constraint_audit,
            "test_patch_blocks_disjoint_from_train_and_validation": bool(
                not (train.patch_i_blocks & test.patch_i_blocks)
            ),
            "validation_depth_block_disjoint_from_optimization_train": bool(
                not np.any(validation_mask & optimization_mask)
            ),
            "normalization_fit_cell_count": int(optimization_mask.sum()),
            "normalization_fit_region": "mode train region excluding validation K-block 3",
            "best_checkpoint_matches_minimum_validation_loss": bool(
                structural_history.best_epoch == int(np.argmin(structural_history.val_loss))
                and seismic_history.best_epoch == int(np.argmin(seismic_history.val_loss))
            ),
            "dense_test_labels_role": "final metrics only",
            "direct_well_observation_cells_excluded_from_test_metrics": bool(
                not np.any(test.observed_mask & eval_mask)
            ),
            "n_direct_test_well_cells_excluded": int(np.count_nonzero(test.observed_mask)),
            "rms_reference_role": "ground-truth value multiset cross-check only",
            "shuffled_test_seismic_sanity_check_metric": shuffled_metric_key,
        },
        "limitations": [
            "Only 15/9-19 SR intersects the final active Eclipse grid; this is a one-well reconstruction.",
            (
                f"Mode={evaluation_mode}: IDW receives "
                f"{constraint_audit['n_well_constraints_supplied_to_idw']} constraints; test "
                f"region contains {constraint_audit['n_test_region_well_constraints']}."
            ),
            (
                "Sparse porosity constraints are Eclipse reference-cell values sampled along the "
                "weakly mapped well trajectory, not independent measured PHIE log values."
            ),
            "Layer-1 weak ties use MD against TVD-like model depth and are not checkshot/VSP ties.",
            (
                "Strict mode excludes guard/test porosity constraints, but seismic sampling still "
                "uses the project-wide Layer-1 weak time mapping as a fixed input geometry."
            ),
            "No mode establishes cross-well or cross-field generalisation.",
            (
                "Source HDF5 metadata still says 'blocked east holdout'; mode-specific results and "
                "runtime audits are authoritative because write scope forbids rebuilding shared HDF5."
            ),
        ],
    }
    if model_name == "ridge_linear":
        # Backward-compatible result aliases from the pre-framework baseline.
        results["seismic_rmse_delta_vs_structural_ridge"] = results[
            "seismic_rmse_delta_vs_structural_model"
        ]
        results["seismic_rmse_improvement_vs_structural_ridge_percent"] = results[
            "seismic_rmse_improvement_vs_structural_model_percent"
        ]
    if hasattr(seismic_model, "weights"):
        results["ridge_standardized_coefficients"] = {
            name: float(value)
            for name, value in zip(SEISMIC_FEATURE_NAMES, seismic_model.weights)
        }
    results_path = HERE / f"results_{evaluation_mode}.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_run_manifest(
        evaluation_mode=evaluation_mode,
        model_name=model_name,
        epochs=epochs,
        results_path=results_path,
        results=results,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ridge_linear")
    parser.add_argument("--evaluation-mode", choices=EVALUATION_MODES, default="conditional")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_baseline(
                model_name=args.model,
                evaluation_mode=args.evaluation_mode,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                ridge_alpha=args.ridge_alpha,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
