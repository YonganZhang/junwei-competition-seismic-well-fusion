#!/usr/bin/env python3
"""Render facies diagnostics from archived predictions and metrics only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import atomic_write_json, hash_file  # noqa: E402
from _code.ml_framework.run_layout import assert_visualization_is_read_only  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def render_archived_diagnostics(
    *,
    prediction_path: Path,
    metrics_path: Path,
    output_path: Path,
    sidecar_path: Path,
    diagnostic_seed: int = 2693,
) -> Path:
    """Read two immutable artifacts; never load a model, data source, or threshold."""
    assert_visualization_is_read_only(
        prediction_path=prediction_path, metrics_path=metrics_path
    )
    metrics = _load_json(metrics_path)
    with np.load(prediction_path, allow_pickle=False) as archive:
        required = {
            "sample_ids",
            "inline",
            "seismic",
            "labels",
            "prediction",
            "confidence",
            "entropy",
            "error",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"prediction archive missing {missing}")
        arrays = {name: archive[name] for name in required}
    count = len(arrays["sample_ids"])
    if count <= 0:
        raise ValueError("prediction archive is empty")
    sample_index = int(diagnostic_seed % count)
    seismic = np.asarray(arrays["seismic"][sample_index], dtype=np.float32)
    label = np.asarray(arrays["labels"][sample_index], dtype=np.int64)
    prediction = np.asarray(arrays["prediction"][sample_index], dtype=np.int64)
    confidence = np.asarray(arrays["confidence"][sample_index], dtype=np.float32)
    entropy = np.asarray(arrays["entropy"][sample_index], dtype=np.float32)
    error = np.asarray(arrays["error"][sample_index], dtype=np.uint8)
    if not (
        seismic.shape
        == label.shape
        == prediction.shape
        == confidence.shape
        == entropy.shape
        == error.shape
    ):
        raise ValueError("archived diagnostic arrays are not spatially aligned")
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    num_classes = matrix.shape[0]
    if matrix.shape != (num_classes, num_classes):
        raise ValueError("confusion matrix must be square")

    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 12, height_ratios=(1.0, 1.15))
    top_axes = [fig.add_subplot(grid[0, 2 * index : 2 * index + 2]) for index in range(6)]
    amplitude = float(np.percentile(np.abs(seismic), 99.0))
    if not np.isfinite(amplitude) or amplitude <= 0:
        amplitude = 1.0
    top_axes[0].imshow(seismic, cmap="gray", vmin=-amplitude, vmax=amplitude)
    class_cmap = plt.get_cmap("tab10", num_classes)
    top_axes[1].imshow(label, cmap=class_cmap, vmin=-0.5, vmax=num_classes - 0.5)
    top_axes[2].imshow(prediction, cmap=class_cmap, vmin=-0.5, vmax=num_classes - 0.5)
    top_axes[3].imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    top_axes[4].imshow(entropy, cmap="magma", vmin=0.0, vmax=1.0)
    top_axes[5].imshow(error, cmap="Reds", vmin=0, vmax=1)
    for axis, title in zip(
        top_axes,
        (
            "Fold-normalized seismic",
            "Ground truth",
            "Prediction",
            "Confidence",
            "Normalized entropy",
            "Error",
        ),
    ):
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    confusion_axis = fig.add_subplot(grid[1, 0:4])
    row_sum = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sum > 0,
    )
    image = confusion_axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    confusion_axis.set_title("Row-normalized confusion")
    confusion_axis.set_xlabel("Predicted class")
    confusion_axis.set_ylabel("Ground-truth class")
    confusion_axis.set_xticks(range(num_classes))
    confusion_axis.set_yticks(range(num_classes))
    fig.colorbar(image, ax=confusion_axis, fraction=0.046)

    class_axis = fig.add_subplot(grid[1, 4:8])
    support = np.asarray(metrics["per_class_support"], dtype=np.float64)
    f1 = np.asarray(metrics["per_class_f1"], dtype=np.float64)
    positions = np.arange(num_classes)
    class_axis.bar(positions - 0.2, support, width=0.4, color="#5975A4", label="support")
    class_axis.set_yscale("log")
    class_axis.set_xlabel("Facies class ID")
    class_axis.set_ylabel("Support pixels (log)")
    class_axis.set_xticks(positions)
    f1_axis = class_axis.twinx()
    f1_axis.bar(positions + 0.2, f1, width=0.4, color="#CC8963", label="F1")
    f1_axis.set_ylim(0.0, 1.0)
    f1_axis.set_ylabel("F1")
    class_axis.set_title("Per-class support and F1")

    reliability_axis = fig.add_subplot(grid[1, 8:12])
    bins = list(metrics["reliability_bins"])
    x = [entry["mean_confidence"] for entry in bins if entry["count"] > 0]
    y = [entry["accuracy"] for entry in bins if entry["count"] > 0]
    reliability_axis.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    reliability_axis.plot(x, y, marker="o", color="#55A868")
    reliability_axis.set_xlim(0.0, 1.0)
    reliability_axis.set_ylim(0.0, 1.0)
    reliability_axis.set_xlabel("Mean confidence")
    reliability_axis.set_ylabel("Accuracy")
    reliability_axis.set_title(
        f"Reliability\nECE={metrics['ece']:.4f}, NLL={metrics['nll']:.4f}"
    )

    context = dict(metrics.get("artifact_context", {}))
    fig.suptitle(
        f"{context.get('task_id', 'facies')} | sample={arrays['sample_ids'][sample_index]} | "
        f"inline={int(arrays['inline'][sample_index])}\n"
        f"mIoU={metrics['miou']:.4f}, macro-F1={metrics['macro_f1']:.4f}, "
        f"Accuracy={metrics['accuracy']:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    atomic_write_json(
        sidecar_path,
        {
            "visualizer_id": "facies_archived_dense_diagnostics_v1",
            "selection": "diagnostic_seed modulo archived sample count; prediction-independent",
            "diagnostic_seed": diagnostic_seed,
            "sample_index": sample_index,
            "sample_id": str(arrays["sample_ids"][sample_index]),
            "inline": int(arrays["inline"][sample_index]),
            "prediction_sha256": hash_file(prediction_path),
            "metrics_sha256": hash_file(metrics_path),
            "artifact_context": context,
            "no_model_or_dataset_loaded": True,
            "no_threshold_selected": True,
        },
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--diagnostic-seed", type=int, default=2693)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_archived_diagnostics(
        prediction_path=args.predictions,
        metrics_path=args.metrics,
        output_path=args.output,
        sidecar_path=args.sidecar,
        diagnostic_seed=args.diagnostic_seed,
    )
    print(output)


if __name__ == "__main__":
    main()
