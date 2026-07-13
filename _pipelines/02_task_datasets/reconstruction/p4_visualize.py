#!/usr/bin/env python3
"""Render P4 reconstruction diagnostics from archived predictions only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "_code"))


PROJECT_TASK_IDS = {
    "conditional": "volve_porosity_conditional_reconstruction",
    "strict": "volve_porosity_strict_spatial_reconstruction",
}


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_archived_prediction(prediction_path: Path, metrics_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load immutable prediction/metric artifacts; never import a model or dataset."""
    from ml_framework.run_layout import assert_visualization_is_read_only

    assert_visualization_is_read_only(prediction_path=prediction_path, metrics_path=metrics_path)
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {
            "mode",
            "task_id",
            "indices_kji",
            "volume_shape_kji",
            "truth",
            "prediction",
            "residual",
            "amplitude",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"prediction archive missing fields: {missing}")
        payload = {name: archive[name].copy() for name in archive.files}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    mode = _scalar_text(payload["mode"])
    task_id = _scalar_text(payload["task_id"])
    if mode not in PROJECT_TASK_IDS or task_id != PROJECT_TASK_IDS[mode]:
        raise ValueError("prediction archive has an unknown mode/task pairing")
    if metrics.get("evaluation_mode") != mode or metrics.get("task_id") != task_id:
        raise ValueError("prediction and metric artifacts belong to different mode/task runs")
    for name in ("truth", "prediction", "residual", "amplitude"):
        values = np.asarray(payload[name], dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"archive field {name} must be a finite vector")
    if not np.allclose(payload["prediction"] - payload["truth"], payload["residual"]):
        raise ValueError("archived residual does not equal prediction-truth")
    return payload, metrics


def _dense(values: np.ndarray, indices: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    volume = np.full(shape, np.nan, dtype=np.float64)
    volume[tuple(indices.T)] = values
    return volume


def _best_plane(valid: np.ndarray, axis: int) -> int:
    reduce_axes = tuple(index for index in range(3) if index != axis)
    counts = np.sum(valid, axis=reduce_axes)
    return int(np.argmax(counts))


def _plane(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume, index, axis=axis)


def _finite_fill(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("selected visualization slice has no valid cells")
    return np.where(finite, values, float(np.mean(values[finite])))


def _radial_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled = _finite_fill(values)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(filled))))
    yy, xx = np.indices(spectrum.shape)
    radius = np.sqrt((yy - (spectrum.shape[0] - 1) / 2.0) ** 2 + (xx - (spectrum.shape[1] - 1) / 2.0) ** 2)
    bins = np.arange(0, int(radius.max()) + 2)
    index = np.digitize(radius.ravel(), bins) - 1
    profile = np.asarray(
        [spectrum.ravel()[index == value].mean() for value in range(len(bins) - 1) if np.any(index == value)]
    )
    return np.arange(profile.size), profile


def _metric(metrics: dict[str, Any], suffix: str) -> float | None:
    matches = [value for key, value in metrics.items() if key.endswith(suffix)]
    if len(matches) != 1:
        return None
    value = matches[0]
    return float(value) if value is not None else None


def render_archived_visualization(
    *,
    prediction_path: Path,
    metrics_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    archive, metrics = load_archived_prediction(prediction_path, metrics_path)
    mode = _scalar_text(archive["mode"])
    task_id = _scalar_text(archive["task_id"])
    indices = np.asarray(archive["indices_kji"], dtype=np.int64)
    shape = tuple(int(value) for value in archive["volume_shape_kji"])
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices_kji must have shape [N,3]")
    truth = _dense(archive["truth"], indices, shape)
    prediction = _dense(archive["prediction"], indices, shape)
    residual = _dense(archive["residual"], indices, shape)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    planes = (
        (2, _best_plane(valid, 2), "inline / I"),
        (1, _best_plane(valid, 1), "crossline / J"),
        (0, _best_plane(valid, 0), "time-depth / K"),
    )
    finite_truth = archive["truth"].astype(np.float64)
    finite_prediction = archive["prediction"].astype(np.float64)
    finite_residual = archive["residual"].astype(np.float64)
    property_limits = (float(min(finite_truth.min(), finite_prediction.min())), float(max(finite_truth.max(), finite_prediction.max())))
    residual_limit = float(max(1e-12, np.max(np.abs(finite_residual))))

    fig = plt.figure(figsize=(18, 18))
    grid = fig.add_gridspec(4, 3, height_ratios=(1.0, 1.0, 1.0, 0.9), hspace=0.30, wspace=0.20)
    selected: dict[str, int] = {}
    for row, (axis, index, label) in enumerate(planes):
        selected[label] = index
        for column, (volume, title, cmap, limits) in enumerate(
            (
                (truth, "reference Eclipse porosity", "viridis", property_limits),
                (prediction, "reconstructed porosity", "viridis", property_limits),
                (residual, "residual (prediction-reference)", "coolwarm", (-residual_limit, residual_limit)),
            )
        ):
            ax = fig.add_subplot(grid[row, column])
            image = ax.imshow(_plane(volume, axis, index), origin="lower", aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
            ax.set_title(f"{label}={index}: {title}")
            ax.set_xlabel("grid axis")
            ax.set_ylabel("grid axis")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax_error = fig.add_subplot(grid[3, 0])
    ax_error.hist(finite_residual, bins=50, color="tab:red", alpha=0.8)
    ax_error.axvline(0.0, color="black", linewidth=1)
    ax_error.set_title("voxel error distribution")
    ax_error.set_xlabel("prediction - reference porosity")
    ax_error.set_ylabel("count")

    ax_distribution = fig.add_subplot(grid[3, 1])
    for values, label, color in (
        (finite_truth, "reference porosity", "black"),
        (finite_prediction, "prediction", "tab:blue"),
        (archive["amplitude"].astype(np.float64), "seismic amplitude", "tab:orange"),
    ):
        scale = float(np.std(values))
        standardized = (values - float(np.mean(values))) / (scale if scale > 0 else 1.0)
        ax_distribution.hist(standardized, bins=50, density=True, histtype="step", linewidth=1.5, label=label, color=color)
    ax_distribution.set_title("standardized attribute/property distributions")
    ax_distribution.set_xlabel("z-score (display only)")
    ax_distribution.legend(fontsize=8)

    ax_spectrum = fig.add_subplot(grid[3, 2])
    horizontal_index = selected["time-depth / K"]
    for volume, label, color in (
        (truth, "reference", "black"),
        (prediction, "prediction", "tab:blue"),
    ):
        frequency, power = _radial_spectrum(_plane(volume, 0, horizontal_index))
        ax_spectrum.plot(frequency, power, label=label, color=color)
    ax_spectrum.set_title("radial log-amplitude spectrum diagnostic")
    ax_spectrum.set_xlabel("radial spatial-frequency bin")
    ax_spectrum.set_ylabel("mean log(1+|FFT|)")
    ax_spectrum.legend()

    rmse = _metric(metrics, "_rmse")
    r2 = _metric(metrics, "_r2")
    caveat = (
        "CONDITIONAL reconstruction given test-region well constraints; NOT strict holdout generalization."
        if mode == "conditional"
        else "STRICT spatial-block reconstruction: no guard/test-region target or well constraint is an input; legacy block is not a new blind field test."
    )
    metric_text = f"RMSE={rmse:.6f}" if rmse is not None and math.isfinite(rmse) else "RMSE=n/a"
    metric_text += f" | R2={r2:.6f}" if r2 is not None and math.isfinite(r2) else " | R2=n/a"
    fig.suptitle(f"P4 {mode.upper()} 3-D reconstruction | {task_id}\n{metric_text}\n{caveat}", fontsize=14, y=0.995)
    fig.text(0.01, 0.005, f"Read-only sources: {prediction_path.name} + {metrics_path.name}; no model inference or threshold selection performed.", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    def relative_to_output(path: Path) -> str:
        return os.path.relpath(path.resolve(), output_path.parent.resolve())

    sidecar = {
        "evaluation_mode": mode,
        "task_id": task_id,
        "prediction_archive": relative_to_output(prediction_path),
        "metrics_archive": relative_to_output(metrics_path),
        "output": output_path.name,
        "selected_slices": selected,
        "panels": [
            "inline truth/prediction/residual",
            "crossline truth/prediction/residual",
            "time-depth truth/prediction/residual",
            "error distribution",
            "attribute/property distributions",
            "radial spectrum",
        ],
        "caveat": caveat,
    }
    output_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render_archived_visualization(
                prediction_path=args.predictions,
                metrics_path=args.metrics,
                output_path=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
