#!/usr/bin/env python3
"""Plot real checkpoint predictions for conditional and strict evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
sys.path.insert(0, str(HERE))

from ml_framework.model_registry import get_model  # noqa: E402
from ml_framework.preprocess import NormStats, denormalize, normalize  # noqa: E402

import baseline  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_real_checkpoint_prediction(evaluation_mode: str):
    """Reproduce a mode's saved best-checkpoint prediction without fitting."""
    results_path = HERE / f"results_{evaluation_mode}.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    preprocess_path = PROJECT_ROOT / results["preprocessing"]["stats_file"]
    preprocess = json.loads(preprocess_path.read_text(encoding="utf-8"))
    train, _, test, allowed_wells, audit = baseline.load_evaluation_regions(evaluation_mode)

    test_idw = baseline._idw_predict(test.coordinates, allowed_wells)
    raw_features = np.column_stack([test_idw, test.seismic, test.coordinates])
    feature_stats = [
        NormStats.from_dict(item) for item in preprocess["seismic_features"]["stats"]
    ]
    features = np.column_stack(
        [normalize(raw_features[:, column], stats) for column, stats in enumerate(feature_stats)]
    )
    target_stats = NormStats.from_dict(preprocess["target"]["stats"])

    model_name = results["registered_model"]
    model = get_model(
        model_name,
        models_package="models",
        n_features=features.shape[1],
        n_training_samples=1,
    )
    checkpoint = PROJECT_ROOT / results["training"]["seismic"]["best_checkpoint"]
    model.load_checkpoint(checkpoint)
    prediction = baseline._clip_from_train(
        denormalize(model.predict(features), target_stats),
        train.target,
    )

    eval_mask = ~test.observed_mask
    metrics = baseline._metrics(test.target[eval_mask], prediction[eval_mask])
    stored_metrics = results["models"][results["primary_baseline"]]
    for name in ("rmse", "mae", "pearson_r", "r2"):
        actual = metrics[name]
        stored = stored_metrics[name]
        if actual is None or stored is None:
            if actual is not stored:
                raise ValueError(f"checkpoint metric definition mismatch for {name}")
        elif not np.isclose(actual, stored, rtol=1e-12, atol=1e-15):
            raise ValueError(
                f"checkpoint visualization metric {name}={actual} "
                f"does not reproduce results value {stored}"
            )
    if audit != results["leakage_checks"]["protocol"]:
        raise ValueError("runtime constraint audit differs from saved mode result")
    return train, test, np.asarray(prediction, dtype=np.float64), metrics, results


def _stitch_test_region(evaluation_mode: str, flat_prediction: np.ndarray):
    """Restore active-cell predictions to the selected test I-block volume."""
    test_blocks = baseline.MODE_I_BLOCKS[evaluation_mode]["test"]
    samples = list(baseline._iter_region_samples(test_blocks))
    if not samples:
        raise ValueError(f"{evaluation_mode} test region is empty")
    patch_shape = np.asarray(samples[0]["meta"]["patch_shape_kji"], dtype=int)
    block_indices = np.asarray(
        [sample["meta"]["patch_index_kji"] for sample in samples], dtype=int
    )
    min_i_block = int(block_indices[:, 2].min())
    volume_shape = (
        int((block_indices[:, 0].max() + 1) * patch_shape[0]),
        int((block_indices[:, 1].max() + 1) * patch_shape[1]),
        int((block_indices[:, 2].max() - min_i_block + 1) * patch_shape[2]),
    )
    reference = np.full(volume_shape, np.nan, dtype=np.float64)
    prediction = np.full(volume_shape, np.nan, dtype=np.float64)
    well_mask = np.zeros(volume_shape, dtype=bool)

    cursor = 0
    for sample in samples:
        patch = np.asarray(sample["seismic_patch"], dtype=np.float32)
        label = np.asarray(sample["label"], dtype=np.float32)
        kb, jb, ib = map(int, sample["meta"]["patch_index_kji"])
        active = patch[8] > 0.5
        n_active = int(active.sum())
        chunk = flat_prediction[cursor : cursor + n_active]
        if chunk.size != n_active:
            raise ValueError("prediction ended before all active test cells were stitched")
        sl = np.s_[
            kb * patch_shape[0] : (kb + 1) * patch_shape[0],
            jb * patch_shape[1] : (jb + 1) * patch_shape[1],
            (ib - min_i_block) * patch_shape[2] : (ib - min_i_block + 1) * patch_shape[2],
        ]
        reference[sl][active] = label[active]
        prediction[sl][active] = chunk
        well_mask[sl] = patch[7] > 0.5
        cursor += n_active
    if cursor != flat_prediction.size:
        raise ValueError(f"stitched {cursor} predictions, expected {flat_prediction.size}")
    return reference, prediction, well_mask, min_i_block


def _select_informative_depth_slice(reference: np.ndarray) -> int:
    scores = np.full(reference.shape[0], -np.inf, dtype=np.float64)
    for k_index in range(reference.shape[0]):
        values = reference[k_index][np.isfinite(reference[k_index])]
        if values.size:
            scores[k_index] = values.size * float(values.std())
    if not np.isfinite(scores).any():
        raise ValueError("test volume has no active reference cells")
    return int(np.argmax(scores))


def _mode_caveat(results: dict) -> str:
    audit = results["leakage_checks"]["protocol"]
    if results["evaluation_mode"] == "conditional":
        return (
            "CONDITIONAL, NOT STRICT HOLDOUT: all "
            f"{audit['n_well_constraints_supplied_to_idw']} global constraints are supplied; "
            f"{audit['n_test_region_well_constraints']} lie in the test region. Exact well cells "
            "are excluded from metrics, but IDW propagates their values."
        )
    return (
        "STRICT SPATIAL HOLDOUT: IDW uses only "
        f"{audit['n_train_region_well_constraints']} train-region constraints; "
        f"{audit['n_guard_region_well_constraints']} guard and "
        f"{audit['n_test_region_well_constraints']} test constraints are excluded. "
        "One-well spatial extrapolation only—not cross-well/cross-field generalization."
    )


def create_figure(evaluation_mode: str) -> dict:
    train, test, flat_prediction, metrics, results = _load_real_checkpoint_prediction(
        evaluation_mode
    )
    reference, prediction, well_mask, min_i_block = _stitch_test_region(
        evaluation_mode, flat_prediction
    )
    k_index = _select_informative_depth_slice(reference)
    reference_slice = reference[k_index]
    prediction_slice = prediction[k_index]
    finite = np.isfinite(reference_slice)

    history_path = PROJECT_ROOT / results["training"]["seismic"]["history"]
    history = json.loads(history_path.read_text(encoding="utf-8"))
    train_loss = np.asarray(history["train_loss"], dtype=np.float64)
    val_loss = np.asarray(history["val_loss"], dtype=np.float64)
    best_epoch = int(history["best_epoch"])
    if best_epoch != int(np.argmin(val_loss)):
        raise ValueError("saved best epoch does not match validation-loss minimum")

    shared_values = np.concatenate(
        [reference_slice[finite], prediction_slice[np.isfinite(prediction_slice)]]
    )
    vmin, vmax = map(float, (shared_values.min(), shared_values.max()))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(
        1, 3, figsize=(17.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.0, 1.25]}
    )
    image_kwargs = dict(
        origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )
    image = axes[0].imshow(np.ma.masked_invalid(reference_slice), **image_kwargs)
    axes[1].imshow(np.ma.masked_invalid(prediction_slice), **image_kwargs)
    i_min = min_i_block * 18
    i_max = i_min + reference.shape[2] - 1
    for ax, title in zip(
        axes[:2], ("Reference: Eclipse PORO", "Reconstruction: Ridge + IDW + seismic")
    ):
        ax.set_title(f"{title}\n{evaluation_mode.upper()} test slice K={k_index}")
        ax.set_xlabel(f"test-region I cell (global I={i_min}–{i_max})")
        ax.set_ylabel("J cell")
    well_y, well_x = np.nonzero(well_mask[k_index])
    if well_x.size:
        axes[1].scatter(
            well_x, well_y, marker="x", s=36, linewidths=1.4, color="red",
            label=f"test-region supplied well cells (n={well_x.size})",
        )
        axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    colorbar = fig.colorbar(image, ax=axes[:2], shrink=0.82, pad=0.025)
    colorbar.set_label("Porosity (fraction)")

    epochs = np.arange(1, train_loss.size + 1)
    axes[2].semilogy(epochs, train_loss, label="train loss", linewidth=1.7)
    axes[2].semilogy(epochs, val_loss, label="validation loss", linewidth=1.7)
    axes[2].axvline(
        best_epoch + 1, color="black", linestyle="--", linewidth=1.2,
        label=f"best epoch = {best_epoch + 1}",
    )
    axes[2].scatter([best_epoch + 1], [val_loss[best_epoch]], color="black", s=28, zorder=3)
    axes[2].set_title(f"{evaluation_mode.upper()} training history")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Normalized MSE (log scale)")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=9)
    r2_text = "undefined" if metrics["r2"] is None else f"{metrics['r2']:.4f}"
    axes[2].text(
        0.04, 0.05,
        f"Evaluated cells: {int((~test.observed_mask).sum()):,}\n"
        f"RMSE = {metrics['rmse']:.6f}\nR² = {r2_text}",
        transform=axes[2].transAxes, fontsize=10, va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.suptitle(
        f"Volve 3-D porosity reconstruction — {evaluation_mode.upper()} real-data evaluation",
        fontsize=15, y=0.98,
    )
    fig.text(
        0.5, 0.025, _mode_caveat(results), ha="center", va="bottom", fontsize=10.0,
        color="#7a2d00",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fff3df", "edgecolor": "#d37b32"},
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.86, bottom=0.18, wspace=0.28)
    output_path = HERE / f"_outputs/prediction_visualization_{evaluation_mode}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)

    metadata = {
        "output": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": _sha256(output_path),
        "visualization_source_sha256": _sha256(HERE / "visualize_prediction.py"),
        "evaluation_mode": evaluation_mode,
        "source": "reconstruction HDF5 + mode-specific ridge_linear seismic best checkpoint",
        "slice_k_index": k_index,
        "n_active_cells_on_slice": int(finite.sum()),
        "n_test_well_cells_on_slice": int(well_mask[k_index].sum()),
        "metrics_on_evaluated_test_cells": metrics,
        "best_epoch_1_based": best_epoch + 1,
        "constraint_audit": results["leakage_checks"]["protocol"],
        "caveat": _mode_caveat(results),
    }
    metadata_path = HERE / f"visualization_metadata_{evaluation_mode}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-mode", choices=(*baseline.EVALUATION_MODES, "both"), default="both"
    )
    args = parser.parse_args()
    modes = baseline.EVALUATION_MODES if args.evaluation_mode == "both" else (args.evaluation_mode,)
    print(json.dumps([create_figure(mode) for mode in modes], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
