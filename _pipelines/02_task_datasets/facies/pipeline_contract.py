"""Leakage and evaluation contract for both independent facies tasks."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TaskSchema:
    task: str
    num_classes: int
    valid_label_ids: tuple[int, ...]
    ignore_index: int | None
    source: str


TASK_SCHEMAS = {
    "facies_f3": TaskSchema(
        task="facies_f3",
        num_classes=10,
        valid_label_ids=tuple(range(10)),
        ignore_index=None,
        source="Zenodo 1471548 declares 10 classes; observed masks use IDs 0..9",
    ),
    "facies_penobscot": TaskSchema(
        task="facies_penobscot",
        num_classes=8,
        valid_label_ids=tuple(range(8)),
        ignore_index=None,
        source="published dataset-log.txt declares num_classes=8; HDF5 uses IDs 0..7",
    ),
}

DEFAULT_TEST_FRACTION = 0.20
DEFAULT_EXTERNAL_GUARD_FRACTION = 0.05
DEFAULT_VALIDATION_FRACTION = 0.20
DEFAULT_VALIDATION_GUARD_FRACTION = 0.05
NEAR_CONSTANT_PEAK_TO_PEAK_EPSILON = 1e-6
DOMINANT_AMPLITUDE_FRACTION_THRESHOLD = 0.50
PIPELINE_VERSION = "leakage_fixed_v2"


def get_task_schema(task: str) -> TaskSchema:
    try:
        return TASK_SCHEMAS[task]
    except KeyError as exc:
        raise ValueError(f"unknown facies task {task!r}; expected {tuple(TASK_SCHEMAS)}") from exc


def ordered_spatial_split(
    line_numbers: Iterable[int], holdout_fraction: float, guard_fraction: float
) -> tuple[set[int], set[int], set[int]]:
    """Return lower train, middle guard, and upper holdout inline sets."""
    lines = sorted(set(int(value) for value in line_numbers))
    if len(lines) < 3:
        raise ValueError("at least three inline sections are required")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    if not 0.0 <= guard_fraction < 1.0:
        raise ValueError("guard_fraction must be in [0, 1)")

    n_holdout = max(1, math.ceil(len(lines) * holdout_fraction))
    n_guard = math.ceil(len(lines) * guard_fraction)
    n_train = len(lines) - n_guard - n_holdout
    if n_train < 1:
        raise ValueError("holdout_fraction + guard_fraction leaves no training lines")
    train = set(lines[:n_train])
    guard = set(lines[n_train : n_train + n_guard])
    holdout = set(lines[n_train + n_guard :])
    assert_spatial_isolation(train, guard, holdout)
    return train, guard, holdout


def assert_spatial_isolation(
    train_lines: set[int], guard_lines: set[int], holdout_lines: set[int]
) -> None:
    """Fail loudly unless three ordered inline regions are nonempty and disjoint."""
    if not train_lines or not holdout_lines:
        raise ValueError("train and holdout inline sets must both be nonempty")
    if train_lines & guard_lines or train_lines & holdout_lines or guard_lines & holdout_lines:
        raise ValueError("train/guard/holdout inline sets overlap")
    if max(train_lines) >= min(holdout_lines):
        raise ValueError("holdout must be spatially above the training inline range")
    if guard_lines:
        if max(train_lines) >= min(guard_lines) or max(guard_lines) >= min(holdout_lines):
            raise ValueError("guard inline range must lie strictly between train and holdout")


def validate_label_array(label: np.ndarray, schema: TaskSchema) -> None:
    array = np.asarray(label)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{schema.task} labels must be integer encoded, got {array.dtype}")
    if array.size == 0:
        raise ValueError(f"{schema.task} label array is empty")
    observed = set(int(value) for value in np.unique(array))
    invalid = observed.difference(schema.valid_label_ids)
    if invalid:
        raise ValueError(
            f"{schema.task} labels {sorted(invalid)} violate fixed schema "
            f"{list(schema.valid_label_ids)}"
        )


def is_near_constant_patch(
    patch: np.ndarray,
    epsilon: float = NEAR_CONSTANT_PEAK_TO_PEAK_EPSILON,
    dominant_fraction_threshold: float = DOMINANT_AMPLITUDE_FRACTION_THRESHOLD,
) -> bool:
    """Detect constant or mostly fill-valued patches without consulting labels."""
    array = np.asarray(patch, dtype=np.float32)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("seismic patch is empty or contains NaN/Inf")
    if epsilon < 0:
        raise ValueError("near-constant epsilon must be nonnegative")
    if not 0.0 < dominant_fraction_threshold <= 1.0:
        raise ValueError("dominant amplitude fraction threshold must be in (0, 1]")
    if float(np.ptp(array)) <= epsilon:
        return True
    _, counts = np.unique(array, return_counts=True)
    dominant_fraction = float(counts.max() / array.size)
    return dominant_fraction >= dominant_fraction_threshold


def segmentation_metrics_from_confusion(confusion: np.ndarray) -> dict[str, object]:
    """Compute finite all-class segmentation metrics and per-class support."""
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"confusion matrix must be nonempty and square, got {matrix.shape}")
    if np.any(matrix < 0) or int(matrix.sum()) == 0:
        raise ValueError("confusion matrix must contain nonnegative counts and nonzero support")

    true_positive = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    false_positive = matrix.sum(axis=0) - true_positive
    false_negative = support - true_positive
    union = true_positive + false_positive + false_negative
    f1_denominator = 2.0 * true_positive + false_positive + false_negative
    if np.any(support <= 0) or np.any(union <= 0) or np.any(f1_denominator <= 0):
        raise ValueError(
            "every configured class must have positive test support and finite IoU/F1 denominators"
        )
    iou = true_positive / union
    f1 = 2.0 * true_positive / f1_denominator
    accuracy = float(true_positive.sum() / matrix.sum())
    metrics: dict[str, object] = {
        "accuracy": accuracy,
        "miou": float(iou.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_support": support.tolist(),
        "per_class_iou": iou.tolist(),
        "per_class_f1": f1.tolist(),
        "evaluated_pixels": int(matrix.sum()),
        "ignored_pixels": 0,
    }
    scalars = [metrics["accuracy"], metrics["miou"], metrics["macro_f1"]]
    vectors = [metrics["per_class_iou"], metrics["per_class_f1"]]
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("aggregate segmentation metric is NaN/Inf")
    if not all(math.isfinite(float(value)) for vector in vectors for value in vector):
        raise ValueError("per-class segmentation metric is NaN/Inf")
    return metrics
