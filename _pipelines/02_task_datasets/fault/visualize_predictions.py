#!/usr/bin/env python3
"""Visualize real best-checkpoint predictions on real fault test samples."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TRACK_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
from audit_utils import (  # noqa: E402
    sha256_file,
    validated_run_dir,
    verify_historical_artifacts_if_present,
)
from baseline import aggregate_physical_voxels, binary_metrics, load_samples  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="audited_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_historical_artifacts_if_present()
    run_dir = validated_run_dir(args.run_name)
    checkpoint_path = run_dir / "checkpoints" / "best.ckpt"
    metrics_path = run_dir / "baseline_metrics.json"
    output_path = run_dir / "prediction_visualization.png"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint not found: {checkpoint_path}")

    test = load_samples("test")
    patches = test.patches
    labels = test.labels

    # joblib imports models.fault_local_logistic from this track while loading;
    # these probabilities therefore come from the persisted real best model.
    model = joblib.load(checkpoint_path)
    probabilities = model.predict_batch(patches)
    if probabilities.shape != labels.shape:
        raise ValueError(f"prediction/label shape mismatch: {probabilities.shape} vs {labels.shape}")

    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = float(saved["threshold"])
    physical_labels, physical_probabilities, coverage = aggregate_physical_voxels(test, probabilities)
    metrics = binary_metrics(physical_labels, physical_probabilities >= threshold)
    for key in ("precision", "recall", "f1"):
        if not np.isclose(metrics[key], saved["test_metrics"][key], rtol=0.0, atol=1e-15):
            raise AssertionError(
                f"recomputed {key}={metrics[key]} differs from saved metric "
                f"{saved['test_metrics'][key]}"
            )

    # Choose fault-containing samples at evenly spaced spatial quantiles. This
    # is deterministic and does not cherry-pick examples by prediction quality.
    positive_indices = [i for i, label in enumerate(labels) if label.any()]
    if len(positive_indices) < 3:
        raise ValueError(f"need at least 3 fault-containing test samples, found {len(positive_indices)}")
    positive_indices.sort(
        key=lambda i: (
            int(test.positions[i]["inline"]),
            int(test.positions[i]["crossline"]),
            float(test.positions[i]["time_ms"]),
        )
    )
    selected = [positive_indices[i] for i in np.linspace(0, len(positive_indices) - 1, 3).round().astype(int)]

    selected_seismic = patches[selected, 0]
    seismic_vmin, seismic_vmax = np.percentile(selected_seismic, [1.0, 99.0])
    fig, axes = plt.subplots(len(selected), 3, figsize=(12.5, 9.2), squeeze=False)
    probability_images = []
    for row, sample_index in enumerate(selected):
        position = test.positions[sample_index]
        axes[row, 0].imshow(
            patches[sample_index, 0],
            cmap="seismic",
            aspect="auto",
            vmin=seismic_vmin,
            vmax=seismic_vmax,
            interpolation="nearest",
        )
        axes[row, 1].imshow(
            labels[sample_index],
            cmap="gray_r",
            aspect="auto",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        probability_images.append(
            axes[row, 2].imshow(
                probabilities[sample_index],
                cmap="magma",
                aspect="auto",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
        )
        axes[row, 0].set_ylabel(
            f"IL {position['inline']}\nXL {position['crossline']}\nTWT {position['time_ms']:.0f} ms"
        )
        for ax in axes[row]:
            ax.set_xlabel("time-sample offset")
            ax.set_yticks([])
        axes[row, 1].text(
            0.02,
            0.96,
            f"fault voxels={int(labels[sample_index].sum())}",
            transform=axes[row, 1].transAxes,
            va="top",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )

    axes[0, 0].set_title("Real seismic input patch (z-score)")
    axes[0, 1].set_title("Ground-truth fault label")
    axes[0, 2].set_title("Predicted fault probability")
    fig.suptitle(
        "Fault detection on real Volve test data\n"
        f"Physical-voxel deduplicated test (patches={len(test.patches)}) — Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  threshold={threshold:.2f}",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Three fault-containing samples selected by spatial quantiles, not prediction score",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.07, top=0.89, hspace=0.34, wspace=0.16)
    colorbar = fig.colorbar(probability_images[-1], ax=axes[:, 2], fraction=0.035, pad=0.04)
    colorbar.set_label("fault probability")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    report = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path.relative_to(TRACK_DIR)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_sha256": sha256_file(Path(__file__)),
        "test_samples": len(test.patches),
        "selected_sample_indices": selected,
        "selected_positions": [test.positions[i] for i in selected],
        "metrics": {key: metrics[key] for key in ("precision", "recall", "f1")},
        "physical_voxel_coverage": coverage,
        "output": str(output_path.relative_to(TRACK_DIR)),
        "output_sha256": sha256_file(output_path),
    }
    (run_dir / "visualization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verify_historical_artifacts_if_present()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
