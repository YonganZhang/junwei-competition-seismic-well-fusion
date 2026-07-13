#!/usr/bin/env python3
"""Visualize real facies predictions from best checkpoints on real test patches."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.dataset_io import load_dataset  # noqa: E402
from _code.ml_framework.model_registry import get_model  # noqa: E402
from _code.ml_framework.preprocess import NormStats, denormalize  # noqa: E402
from pipeline_contract import PIPELINE_VERSION, TASK_SCHEMAS  # noqa: E402

TASKS = tuple(TASK_SCHEMAS)
TASK_LABELS = {"facies_f3": "F3", "facies_penobscot": "Penobscot"}
DEFAULT_OUTPUT = TRACK_DIR / "_outputs" / "prediction_visualization.png"
DEFAULT_ARTIFACT_ROOT = TRACK_DIR / "_outputs" / PIPELINE_VERSION
DEFAULT_EVIDENCE = DEFAULT_ARTIFACT_ROOT / "prediction_visualization_evidence.json"
UKIYOE_10 = [
    "#E76254",
    "#EF8A47",
    "#F7AA58",
    "#FFD06F",
    "#FFE6B7",
    "#AADCE0",
    "#72BCD5",
    "#528FAD",
    "#376795",
    "#1E466E",
]

# Times New Roman is not installed on this node; Liberation Serif is its
# metric-compatible serif fallback and avoids Matplotlib's repeated warnings.
FONT_FAMILY = "Liberation Serif"


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def normalize_fonts(fig: plt.Figure) -> None:
    """Apply one readable serif family and minimum size before saving."""
    for text in fig.findobj(match=Text):
        text.set_fontfamily(FONT_FAMILY)
        if text.get_fontsize() < 7:
            text.set_fontsize(7)


def select_evenly_spaced_samples(
    samples: list[dict[str, Any]], count: int, min_ground_truth_classes: int
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select deterministic GT-diverse interior quantiles, never using predictions."""
    if count <= 0 or count > len(samples):
        raise ValueError(f"sample count must be in 1..{len(samples)}, got {count}")
    candidates = [
        (index, sample)
        for index, sample in enumerate(samples)
        if len(np.unique(sample["label"])) >= min_ground_truth_classes
    ]
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} test samples contain at least "
            f"{min_ground_truth_classes} ground-truth classes"
        )
    candidate_positions = np.linspace(
        0, len(candidates) - 1, count + 2, dtype=int
    )[1:-1].tolist()
    selected = [candidates[position] for position in candidate_positions]
    indices = [index for index, _ in selected]
    if len(set(indices)) != count:
        raise ValueError("test split is too small for unique evenly spaced samples")
    return indices, [sample for _, sample in selected]


def load_predictions(
    task: str,
    *,
    samples_per_task: int,
    device: torch.device,
    artifact_root: Path,
    min_ground_truth_classes: int,
) -> dict[str, Any]:
    """Load one real best checkpoint and infer on deterministic real test samples."""
    model_dir = artifact_root / task / "small_unet"
    checkpoint_path = model_dir / "best.ckpt"
    metrics_path = model_dir / "metrics.json"
    if not checkpoint_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing trained artifacts under {model_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("task") != task:
        raise ValueError(
            f"checkpoint task mismatch: requested {task}, found {checkpoint.get('task')}"
        )
    if checkpoint.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("checkpoint does not belong to the leakage-fixed pipeline")
    model = get_model(
        checkpoint["model_name"],
        models_package="models",
        num_classes=checkpoint["num_classes"],
        **checkpoint["model_kwargs"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test_samples = list(load_dataset(task, "test"))
    indices, selected = select_evenly_spaced_samples(
        test_samples, samples_per_task, min_ground_truth_classes
    )
    model_inputs = torch.from_numpy(
        np.stack(
            [np.asarray(sample["seismic_patch"], dtype=np.float32) for sample in selected]
        )[:, None, ...]
    ).to(device)
    with torch.no_grad():
        predictions = model(model_inputs).argmax(dim=1).cpu().numpy()

    raw_seismic: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for sample in selected:
        stats = NormStats.from_dict(sample["meta"]["normalization_stats"])
        raw_seismic.append(
            np.asarray(denormalize(sample["seismic_patch"], stats), dtype=np.float32)
        )
        labels.append(np.asarray(sample["label"], dtype=np.uint8))

    metrics_record = json.loads(metrics_path.read_text())
    metrics = metrics_record["test_metrics_from_best_checkpoint"]
    return {
        "task": task,
        "indices": indices,
        "samples": selected,
        "seismic": raw_seismic,
        "labels": labels,
        "predictions": predictions,
        "num_classes": int(checkpoint["num_classes"]),
        "metrics": metrics,
        "checkpoint": checkpoint_path,
    }


def render(results: list[dict[str, Any]], output_path: Path) -> Path:
    """Render a Mode-D 7.2-inch multi-panel diagnostic figure."""
    samples_per_task = len(results[0]["indices"])
    if any(len(result["indices"]) != samples_per_task for result in results):
        raise ValueError("all tasks must contribute the same number of samples")

    nrows = len(results) * samples_per_task
    fig, axes = plt.subplots(nrows, 3, figsize=(7.2, 7.2), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.14, right=0.90, top=0.93, bottom=0.05, wspace=0.04, hspace=0.12)

    column_centers = [0.265, 0.525, 0.785]
    for center, label in zip(column_centers, ("Seismic input", "Ground truth", "Prediction")):
        fig.text(center, 0.955, label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    row_offset = 0
    for task_index, result in enumerate(results):
        num_classes = result["num_classes"]
        class_cmap = mcolors.ListedColormap(UKIYOE_10[:num_classes])
        class_norm = mcolors.BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1), num_classes)
        task_rows = list(range(row_offset, row_offset + samples_per_task))

        for local_row, row in enumerate(task_rows):
            seismic = result["seismic"][local_row]
            label = result["labels"][local_row]
            prediction = result["predictions"][local_row]
            if seismic.shape != label.shape or label.shape != prediction.shape:
                raise ValueError(
                    f"unaligned visualization arrays: {seismic.shape}, {label.shape}, {prediction.shape}"
                )
            amplitude_limit = float(np.percentile(np.abs(seismic), 99.0))
            if not np.isfinite(amplitude_limit) or amplitude_limit <= 0:
                raise ValueError("invalid seismic amplitude range")

            axes[row, 0].imshow(
                seismic,
                cmap="gray",
                vmin=-amplitude_limit,
                vmax=amplitude_limit,
                interpolation="nearest",
            )
            axes[row, 1].imshow(
                label,
                cmap=class_cmap,
                norm=class_norm,
                interpolation="nearest",
            )
            axes[row, 2].imshow(
                prediction,
                cmap=class_cmap,
                norm=class_norm,
                interpolation="nearest",
            )
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_linewidth(0.4)
                    spine.set_color("#333333")

            sample = result["samples"][local_row]
            axes[row, 0].set_ylabel(
                f"test #{result['indices'][local_row]}\ninline {sample['position']['inline']}",
                fontsize=7,
                rotation=0,
                ha="right",
                va="center",
                labelpad=8,
            )

        metrics = result["metrics"]
        group_y = 0.755 if task_index == 0 else 0.315
        panel_letter = chr(ord("a") + task_index)
        fig.text(
            0.012,
            group_y,
            panel_letter,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        fig.text(
            0.035,
            group_y,
            f"{TASK_LABELS[result['task']]}\nAccuracy {metrics['accuracy']:.4f}"
            f"\nmIoU {metrics['miou']:.4f}\nMacro-F1 {metrics['macro_f1']:.4f}",
            ha="left",
            va="center",
            fontsize=7,
        )
        scalar_mappable = plt.cm.ScalarMappable(norm=class_norm, cmap=class_cmap)
        colorbar = fig.colorbar(
            scalar_mappable,
            ax=axes[task_rows, 1:].ravel().tolist(),
            ticks=np.arange(num_classes),
            fraction=0.025,
            pad=0.018,
            aspect=20,
        )
        colorbar.set_label("Facies class ID", fontsize=7)
        colorbar.ax.tick_params(labelsize=7, width=0.4, length=2)
        row_offset += samples_per_task

    separator_y = 0.49
    fig.add_artist(
        Line2D([0.02, 0.98], [separator_y, separator_y], transform=fig.transFigure, color="#777777", linewidth=0.5)
    )
    normalize_fonts(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--samples-per-task", type=int, default=2)
    parser.add_argument("--min-ground-truth-classes", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    results = [
        load_predictions(
            task,
            samples_per_task=args.samples_per_task,
            device=device,
            artifact_root=args.artifact_root,
            min_ground_truth_classes=args.min_ground_truth_classes,
        )
        for task in TASKS
    ]
    output_path = render(results, args.output)
    evidence = {
        result["task"]: {
            "checkpoint": project_relative(result["checkpoint"]),
            "test_indices": result["indices"],
            "inline_numbers": [sample["position"]["inline"] for sample in result["samples"]],
            "ground_truth_class_ids": [
                np.unique(sample["label"]).tolist() for sample in result["samples"]
            ],
            "accuracy": result["metrics"]["accuracy"],
            "miou": result["metrics"]["miou"],
            "macro_f1": result["metrics"]["macro_f1"],
        }
        for result in results
    }
    evidence_record = {
        "pipeline_version": PIPELINE_VERSION,
        "output": project_relative(output_path),
        "selection_policy": (
            "interior quantiles among real test patches with at least "
            f"{args.min_ground_truth_classes} ground-truth classes; predictions not used"
        ),
        "evidence": evidence,
    }
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence_record, indent=2) + "\n")
    print(json.dumps(evidence_record, indent=2))


if __name__ == "__main__":
    main()
