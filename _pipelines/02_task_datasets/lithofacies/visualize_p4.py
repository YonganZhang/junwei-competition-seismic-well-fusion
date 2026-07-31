#!/usr/bin/env python3
"""Render lithofacies P4 figures from archived predictions and metrics only."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.artifacts import atomic_write_json, hash_file  # noqa: E402
from _code.ml_framework.run_layout import assert_visualization_is_read_only  # noqa: E402
from p4_contract import CLASS_NAMES, finite_center_md  # noqa: E402


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return resolved.name


def _records(payload: Any) -> list[dict[str, Any]]:
    records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("prediction archive must contain a non-empty records list")
    required = {
        "sample_id",
        "well_id",
        "true_class_id",
        "predicted_class_id",
        "confidence",
        "error",
    }
    for record in records:
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"prediction record missing fields: {missing}")
    return [dict(record) for record in records]


def _plot_depth_tracks(records: Sequence[Mapping[str, Any]], path: Path) -> Path:
    if not finite_center_md(records):
        raise ValueError(
            "archived predictions lack real center_md_m; rebuild with persisted sampling centers, "
            "never substitute interval midpoints"
        )
    by_well: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_well[str(record["well_id"])].append(record)
    wells = sorted(by_well)
    fig, axes = plt.subplots(
        len(wells), 3, figsize=(11, max(4.0, 3.0 * len(wells))), squeeze=False
    )
    for row, well in enumerate(wells):
        ordered = sorted(by_well[well], key=lambda item: float(item["center_md_m"]))
        depth = np.asarray([float(item["center_md_m"]) for item in ordered])
        truth = np.asarray([int(item["true_class_id"]) for item in ordered])
        prediction = np.asarray([int(item["predicted_class_id"]) for item in ordered])
        confidence = np.asarray([float(item["confidence"]) for item in ordered])
        error = np.asarray([bool(item["error"]) for item in ordered])
        axes[row, 0].step(truth, depth, where="mid", label="GT", linewidth=2)
        axes[row, 0].step(prediction, depth, where="mid", label="prediction", alpha=0.8)
        axes[row, 0].set_xticks(range(len(CLASS_NAMES)))
        axes[row, 0].set_xticklabels(range(len(CLASS_NAMES)), fontsize=7)
        axes[row, 0].set_ylabel("MD (m)")
        axes[row, 0].set_title(f"{well}: facies track")
        axes[row, 0].legend(fontsize=7)
        axes[row, 1].plot(confidence, depth, color="tab:blue")
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_xlabel("max softmax confidence")
        axes[row, 1].set_title("Confidence")
        axes[row, 2].scatter(error.astype(int), depth, c=np.where(error, "crimson", "green"), s=12)
        axes[row, 2].set_xlim(-0.2, 1.2)
        axes[row, 2].set_xticks((0, 1), ("correct", "error"))
        axes[row, 2].set_title("Error by depth")
        for axis in axes[row]:
            axis.invert_yaxis()
            axis.grid(alpha=0.2)
    fig.suptitle("Archived GM09 truth, prediction, confidence and error", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusions(metrics: Mapping[str, Any], path: Path) -> Path:
    counts = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    normalized = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=np.float64)
    expected = (len(CLASS_NAMES), len(CLASS_NAMES))
    if counts.shape != expected or normalized.shape != expected:
        raise ValueError(f"confusion matrices must be {expected}")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for axis, matrix, title, fmt in (
        (axes[0], counts, "Count confusion", "d"),
        (axes[1], normalized, "Row-normalized confusion", ".2f"),
    ):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=None if fmt == "d" else 1)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=60, ha="right", fontsize=7)
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=7)
        axis.set_xlabel("Prediction")
        axis.set_ylabel("Ground truth")
        axis.set_title(title)
        for row in range(len(CLASS_NAMES)):
            for column in range(len(CLASS_NAMES)):
                value = matrix[row, column]
                text = format(int(value), fmt) if fmt == "d" else format(float(value), fmt)
                axis.text(column, row, text, ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_per_class(metrics: Mapping[str, Any], path: Path) -> Path:
    rows = list(metrics["per_class"])
    if len(rows) != len(CLASS_NAMES):
        raise ValueError("per_class metrics must retain all nine classes")
    x = np.arange(len(CLASS_NAMES))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": (2, 1)})
    for offset, key, label in (
        (-width, "precision", "precision"),
        (0.0, "recall", "recall"),
        (width, "f1", "F1"),
    ):
        axes[0].bar(x + offset, [float(row[key]) for row in rows], width=width, label=label)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)
    support = [int(row["support"]) for row in rows]
    axes[1].bar(x, support, color="tab:gray")
    for index, count in enumerate(support):
        axes[1].text(index, count, str(count), ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("Support")
    axes[1].set_xticks(x, CLASS_NAMES, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Per-class precision, recall, F1 and support (fixed GM09 schema)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_calibration(metrics: Mapping[str, Any], path: Path) -> Path:
    calibration = metrics["calibration"]
    bins = calibration["bins"]
    counts = np.asarray([int(row["count"]) for row in bins])
    confidence = np.asarray([float(row["mean_confidence"]) for row in bins])
    accuracy = np.asarray([float(row["accuracy"]) for row in bins])
    keep = counts > 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot((0, 1), (0, 1), linestyle="--", color="gray", label="perfect")
    axes[0].plot(confidence[keep], accuracy[keep], marker="o", label="observed")
    for x, y, count in zip(confidence[keep], accuracy[keep], counts[keep]):
        axes[0].annotate(f"n={count}", (x, y), fontsize=7)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Mean confidence")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Reliability (ECE={float(calibration['expected_calibration_error']):.3f})")
    axes[0].legend()
    axes[1].bar(np.arange(len(counts)), counts)
    axes[1].set_xlabel("Confidence bin")
    axes[1].set_ylabel("Samples")
    axes[1].set_title("Calibration-bin support")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def render_archived_visualizations(
    *, prediction_path: Path, metrics_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Never train, infer, select thresholds, or rewrite the input archives."""
    assert_visualization_is_read_only(prediction_path=prediction_path, metrics_path=metrics_path)
    before = {"predictions": hash_file(prediction_path), "metrics": hash_file(metrics_path)}
    records = _records(_read_json(prediction_path))
    metrics = _read_json(metrics_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    not_feasible: dict[str, Any] = {}
    try:
        path = _plot_depth_tracks(records, output_dir / "depth_facies_track.png")
        figures["depth_facies_track"] = path.name
    except ValueError as exc:
        not_feasible["depth_facies_track"] = str(exc)
        atomic_write_json(
            output_dir / "not_feasible_depth_facies_track.json",
            {
                "status": "not_feasible",
                "reason": str(exc),
                "forbidden_fallback": "do not substitute label interval midpoint or inferred coordinates",
            },
        )
    for name, function, filename in (
        ("confusion_count_and_row_normalized", _plot_confusions, "confusion_count_and_normalized.png"),
        ("per_class_precision_recall_f1_support", _plot_per_class, "per_class_metrics.png"),
        ("calibration_reliability", _plot_calibration, "calibration.png"),
    ):
        path = function(metrics, output_dir / filename)
        figures[name] = path.name
    after = {"predictions": hash_file(prediction_path), "metrics": hash_file(metrics_path)}
    if before != after:
        raise RuntimeError("archived prediction/metric inputs changed during visualization")
    report = {
        "status": "PASS" if not not_feasible else "PARTIAL_NOT_FEASIBLE",
        "input_hashes": before,
        "prediction_path": _portable_path(prediction_path),
        "metrics_path": _portable_path(metrics_path),
        "figures": figures,
        "not_feasible": not_feasible,
        "read_only_inputs_verified": True,
    }
    atomic_write_json(output_dir / "visualization_manifest.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.resolve()
    track = TRACK_DIR.resolve()
    if track not in output.parents:
        raise ValueError(f"output directory must stay under {TRACK_DIR}")
    report = render_archived_visualizations(
        prediction_path=args.predictions.resolve(),
        metrics_path=args.metrics.resolve(),
        output_dir=output,
    )
    if output.name == "visualizations" and (output.parent / "lifecycle.json").is_file():
        from p4_runner import refresh_artifact_manifest

        refresh_artifact_manifest(output.parent)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
