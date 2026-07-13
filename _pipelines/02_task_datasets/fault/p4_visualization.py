"""Fault-specific figures generated only from archived predictions and metrics."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import binary_erosion, label as connected_components  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _code.ml_framework.artifacts import atomic_write_json, hash_file  # noqa: E402
from _code.ml_framework.run_layout import assert_visualization_is_read_only  # noqa: E402


REQUIRED_ARRAYS = (
    "amplitude",
    "target",
    "valid_label_mask",
    "proxy_mask",
    "probability",
)
REQUIRED_PROVENANCE = (
    "threshold",
    "threshold_source",
    "config_hash",
    "split_hash",
    "checkpoint_hash",
    "prediction_role",
)


def _load_archive(prediction_path: Path, metrics_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    assert_visualization_is_read_only(
        prediction_path=prediction_path,
        metrics_path=metrics_path,
    )
    with np.load(prediction_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"fault prediction archive is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in REQUIRED_ARRAYS}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    missing_provenance = sorted(set(REQUIRED_PROVENANCE) - set(metrics))
    if missing_provenance:
        raise ValueError(f"fault metric archive is missing provenance: {missing_provenance}")
    if metrics["threshold_source"] != "pooled_oof":
        raise ValueError("visualization threshold must come from pooled OOF, never test/picture selection")
    threshold = float(metrics["threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError("archived threshold must lie strictly inside (0,1)")
    return arrays, metrics


def _validate_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    amplitude = np.asarray(arrays["amplitude"], dtype=np.float32)
    target = np.asarray(arrays["target"])
    valid = np.asarray(arrays["valid_label_mask"], dtype=bool)
    proxy = np.asarray(arrays["proxy_mask"], dtype=bool)
    probability = np.asarray(arrays["probability"], dtype=np.float32)
    if amplitude.ndim != 3:
        raise ValueError(f"archived fault volume must be [D,H,W], received {amplitude.shape}")
    if not (amplitude.shape == target.shape == valid.shape == proxy.shape == probability.shape):
        raise ValueError("all archived fault arrays must have one [D,H,W] shape")
    if not np.isfinite(amplitude).all() or not np.isfinite(probability).all():
        raise ValueError("archived amplitude/probability arrays must be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("archived fault probabilities must lie in [0,1]")
    if not np.isin(target, (0, 1)).all():
        raise ValueError("archived fault target must be binary")
    target = target.astype(bool)
    if np.any(target & ~valid):
        raise ValueError("fault-stick positives must be included in valid_label_mask")
    if np.any(proxy & valid):
        raise ValueError("proxy_mask must not overlap valid_label_mask")
    valid_targets = target[valid]
    if not len(valid_targets) or len(np.unique(valid_targets)) < 2:
        raise ValueError("formal PR/confusion visualization requires audited positive and negative labels")
    return {
        "amplitude": amplitude,
        "target": target,
        "valid": valid,
        "proxy": proxy,
        "probability": probability,
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _overlay_figure(data: dict[str, np.ndarray], threshold: float, output: Path) -> None:
    depth = data["amplitude"].shape[0] // 2
    amplitude = data["amplitude"][depth]
    probability = data["probability"][depth]
    target = data["target"][depth]
    valid = data["valid"][depth]
    prediction = probability >= threshold
    ground_truth = np.full(target.shape, np.nan, dtype=np.float32)
    ground_truth[valid] = target[valid].astype(np.float32)
    confusion = np.full(target.shape, np.nan, dtype=np.float32)
    confusion[valid & ~target & ~prediction] = 0.0
    confusion[valid & target & prediction] = 1.0
    confusion[valid & ~target & prediction] = 2.0
    confusion[valid & target & ~prediction] = 3.0

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    axes[0].imshow(amplitude, cmap="gray", aspect="auto")
    axes[0].set_title("Input amplitude")
    axes[1].imshow(ground_truth, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("GT (unknown masked)")
    image = axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1, aspect="auto")
    axes[2].set_title("Archived probability")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    axes[3].imshow(confusion, cmap="tab10", vmin=0, vmax=3, aspect="auto")
    axes[3].set_title("TN / TP / FP / FN")
    for axis in axes:
        axis.set_xlabel("time")
        axis.set_ylabel("crossline")
    figure.suptitle(f"Fault archived prediction — fixed threshold={threshold:.4f}")
    _save_figure(figure, output)


def _orthogonal_figure(data: dict[str, np.ndarray], output: Path) -> None:
    volume = data["probability"]
    depth, height, width = (dimension // 2 for dimension in volume.shape)
    slices = (
        (volume[depth, :, :], "Inline/depth plane"),
        (volume[:, height, :], "Crossline plane"),
        (volume[:, :, width], "Time plane"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for axis, (values, title) in zip(axes, slices):
        image = axis.imshow(values, cmap="magma", vmin=0, vmax=1, aspect="auto")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Three orthogonal views from archived probability")
    _save_figure(figure, output)


def _pr_threshold_figure(
    data: dict[str, np.ndarray],
    threshold: float,
    output: Path,
) -> dict[str, float]:
    truth = data["target"][data["valid"]].astype(np.uint8)
    probability = data["probability"][data["valid"]].astype(np.float64)
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    average_precision = float(average_precision_score(truth, probability))
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(recall, precision)
    axes[0].set(xlabel="Recall", ylabel="Precision", title=f"PR (AP={average_precision:.4f})")
    if len(thresholds):
        axes[1].plot(thresholds, precision[:-1], label="precision")
        axes[1].plot(thresholds, recall[:-1], label="recall")
    axes[1].axvline(threshold, color="black", linestyle="--", label="fixed pooled-OOF threshold")
    axes[1].set(xlabel="Threshold", ylabel="Score", title="Threshold diagnostics")
    axes[1].legend(fontsize=8)
    _save_figure(figure, output)
    return {"average_precision_from_archive": average_precision}


def _boundary_components_figure(
    data: dict[str, np.ndarray],
    threshold: float,
    output: Path,
) -> dict[str, int | float]:
    target = data["target"] & data["valid"]
    prediction = (data["probability"] >= threshold) & data["valid"]
    structure = np.ones((3, 3, 3), dtype=bool)
    target_boundary = target & ~binary_erosion(target, structure=structure, border_value=0)
    prediction_boundary = prediction & ~binary_erosion(prediction, structure=structure, border_value=0)
    target_components, target_count = connected_components(target, structure=structure)
    prediction_components, prediction_count = connected_components(prediction, structure=structure)
    del target_components, prediction_components
    overlap = int(np.sum(target_boundary & prediction_boundary))
    denominator = int(target_boundary.sum() + prediction_boundary.sum())
    boundary_f1 = 2.0 * overlap / denominator if denominator else 1.0
    depth = target.shape[0] // 2

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    axes[0].imshow(target_boundary[depth], cmap="Blues", aspect="auto")
    axes[0].set_title("GT boundary")
    axes[1].imshow(prediction_boundary[depth], cmap="Reds", aspect="auto")
    axes[1].set_title("Prediction boundary")
    axes[2].bar(["GT", "Prediction"], [target_count, prediction_count], color=["tab:blue", "tab:red"])
    axes[2].set_title("3-D connected components")
    figure.suptitle(f"Boundary/components (diagnostic boundary F1={boundary_f1:.4f})")
    _save_figure(figure, output)
    return {
        "target_components": int(target_count),
        "prediction_components": int(prediction_count),
        "boundary_f1_diagnostic": float(boundary_f1),
    }


def render_fault_visualizations(
    *,
    prediction_path: Path,
    metrics_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render four required figures without model, loader, threshold selection, or training."""

    arrays, metrics = _load_archive(prediction_path, metrics_path)
    data = _validate_arrays(arrays)
    threshold = float(metrics["threshold"])
    outputs = {
        "input_gt_probability_confusion": output_dir / "input_gt_probability_confusion.png",
        "orthogonal_views": output_dir / "orthogonal_views.png",
        "pr_threshold": output_dir / "pr_threshold.png",
        "boundary_components": output_dir / "boundary_components.png",
    }
    _overlay_figure(data, threshold, outputs["input_gt_probability_confusion"])
    _orthogonal_figure(data, outputs["orthogonal_views"])
    diagnostics = _pr_threshold_figure(data, threshold, outputs["pr_threshold"])
    diagnostics.update(_boundary_components_figure(data, threshold, outputs["boundary_components"]))
    report = {
        "visualizer_id": "fault_archived_volume",
        "prediction_archive": str(prediction_path),
        "prediction_sha256": hash_file(prediction_path),
        "metrics_archive": str(metrics_path),
        "metrics_sha256": hash_file(metrics_path),
        "threshold": threshold,
        "threshold_source": metrics["threshold_source"],
        "prediction_role": metrics["prediction_role"],
        "config_hash": metrics["config_hash"],
        "split_hash": metrics["split_hash"],
        "checkpoint_hash": metrics["checkpoint_hash"],
        "valid_label_count": int(data["valid"].sum()),
        "unknown_label_count": int((~data["valid"]).sum()),
        "proxy_label_count": int(data["proxy"].sum()),
        "figures": {name: str(path) for name, path in outputs.items()},
        "figure_sha256": {name: hash_file(path) for name, path in outputs.items()},
        "diagnostics": diagnostics,
        "selection_performed": False,
    }
    if not all(math.isfinite(float(value)) for value in diagnostics.values()):
        raise RuntimeError("fault visualization diagnostics contain non-finite values")
    atomic_write_json(output_dir / "visualization_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_fault_visualizations(
        prediction_path=args.prediction,
        metrics_path=args.metrics,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
