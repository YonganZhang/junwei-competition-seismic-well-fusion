#!/usr/bin/env python3
"""Verify trained checkpoints, curves, metrics, and visualization provenance."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
if str(TRACK_DIR) not in sys.path:
    sys.path.insert(0, str(TRACK_DIR))

from pipeline_contract import (  # noqa: E402
    PIPELINE_VERSION,
    TASK_SCHEMAS,
    segmentation_metrics_from_confusion,
)

DEFAULT_ROOT = TRACK_DIR / "_outputs" / PIPELINE_VERSION
DEFAULT_VISUALIZATION = TRACK_DIR / "_outputs" / "prediction_visualization.png"
DEFAULT_OUTPUT = DEFAULT_ROOT / "artifact_verification.json"


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def verify_task(task: str, artifact_root: Path) -> dict[str, Any]:
    model_dir = artifact_root / task / "small_unet"
    metrics_path = model_dir / "metrics.json"
    record = json.loads(metrics_path.read_text())
    history = record["history"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = int(record["hyperparameters"]["epochs"])
    if len(train_loss) != epochs or len(val_loss) != epochs:
        raise ValueError(f"{task} history length does not equal configured epochs")
    if not np.isfinite(train_loss).all() or not np.isfinite(val_loss).all():
        raise ValueError(f"{task} history contains NaN/Inf")
    best_index = int(np.argmin(val_loss))
    if record["best_epoch_number"] != best_index + 1:
        raise ValueError(f"{task} best epoch is not argmin validation loss")
    if best_index + 1 >= epochs:
        raise ValueError(f"{task} best epoch is still the final epoch")
    if not record["overfit_turn_observed"]:
        raise ValueError(f"{task} does not record a post-best validation rise")
    if max(val_loss[best_index + 1 :]) <= val_loss[best_index] * 1.01:
        raise ValueError(f"{task} post-best validation rise is less than 1%")

    confusion = np.asarray(record["confusion_matrix"], dtype=np.int64)
    recomputed = segmentation_metrics_from_confusion(confusion)
    saved = record["test_metrics_from_best_checkpoint"]
    for key in ("accuracy", "miou", "macro_f1"):
        if not close(float(saved[key]), float(recomputed[key])):
            raise ValueError(f"{task} saved {key} differs from confusion-matrix recomputation")
    for key in ("per_class_support", "per_class_iou", "per_class_f1"):
        if not np.allclose(saved[key], recomputed[key], rtol=0.0, atol=1e-10):
            raise ValueError(f"{task} saved {key} differs from recomputation")
    if len(saved["per_class_support"]) != TASK_SCHEMAS[task].num_classes:
        raise ValueError(f"{task} per-class metric width violates fixed schema")

    best_path = project_path(record["best_checkpoint"])
    last_path = project_path(record["last_checkpoint"])
    loss_curve_path = project_path(record["loss_curve"])
    for path in (best_path, last_path, loss_curve_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != task or checkpoint.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError(f"{task} best checkpoint metadata mismatch")
    if checkpoint.get("num_classes") != TASK_SCHEMAS[task].num_classes:
        raise ValueError(f"{task} checkpoint classifier width violates schema")

    return {
        "epochs": epochs,
        "best_epoch": best_index + 1,
        "best_val_loss": float(val_loss[best_index]),
        "last_val_loss": float(val_loss[-1]),
        "overfit_turn_observed": True,
        "metrics_recomputed_from_confusion": {
            key: recomputed[key] for key in ("accuracy", "miou", "macro_f1")
        },
        "best_checkpoint": project_relative(best_path),
        "loss_curve": project_relative(loss_curve_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--visualization", type=Path, default=DEFAULT_VISUALIZATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    tasks = {task: verify_task(task, args.artifact_root) for task in TASK_SCHEMAS}
    evidence_path = args.artifact_root / "prediction_visualization_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("prediction evidence pipeline version mismatch")
    with Image.open(args.visualization) as image:
        visualization = {
            "path": project_relative(args.visualization),
            "format": image.format,
            "mode": image.mode,
            "size": list(image.size),
            "bytes": args.visualization.stat().st_size,
        }
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "tasks": tasks,
        "prediction_evidence": evidence,
        "visualization": visualization,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
