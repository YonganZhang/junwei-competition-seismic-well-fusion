"""Track-private helpers for P5 lithofacies model adapters.

The helpers deliberately know nothing about dataset paths.  Stage-1 supplies
already fold-local, finite arrays and every adapter consumes both the 26-row
well tensor (13 values plus 13 observed masks) and the seismic patch.
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec


NUM_CLASSES = 9
WELL_CHANNELS = 26
SEISMIC_SPATIAL_SHAPE = (3, 3)


class OptionalDependencyUnavailable(RuntimeError):
    """A source-locked optional backend is absent from the active environment."""

    def __init__(self, model_id: str, dependency: str) -> None:
        self.model_id = model_id
        self.dependency = dependency
        super().__init__(f"{model_id} requires optional dependency {dependency!r}")


def require_dependency(model_id: str, dependency: str) -> Any:
    try:
        return importlib.import_module(dependency)
    except ImportError as exc:
        raise OptionalDependencyUnavailable(model_id, dependency) from exc


def validate_task(task_spec: TaskSpec, *, num_classes: int) -> None:
    if task_spec.track_id != "lithofacies" or task_spec.task_type != "multiclass":
        raise ValueError("P5 lithofacies adapters require the multiclass lithofacies TaskSpec")
    frozen = int(task_spec.metadata.get("class_count", 0))
    if num_classes != NUM_CLASSES or frozen != NUM_CLASSES:
        raise ValueError(f"GM09 output schema is frozen at {NUM_CLASSES} classes")


def validate_shapes(well_log_shape: tuple[int, ...], seismic_shape: tuple[int, ...]) -> None:
    if len(well_log_shape) != 2 or well_log_shape[0] != WELL_CHANNELS:
        raise ValueError(f"well_log_shape must be [26,L], got {well_log_shape}")
    if len(seismic_shape) != 3 or tuple(seismic_shape[:2]) != SEISMIC_SPATIAL_SHAPE:
        raise ValueError(f"seismic_shape must be [3,3,L], got {seismic_shape}")
    if well_log_shape[-1] != seismic_shape[-1]:
        raise ValueError("well-log and seismic sequence axes must align")


def standard_capabilities(
    *, lane: str, backend: str, dependency_group: str, supports_modality_availability: bool = False
) -> dict[str, Any]:
    if lane not in {"P", "S"}:
        raise ValueError("leaderboard lane must be P or S")
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["well_log_sequence", "st0202_seismic_patch"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "supports_modality_availability": supports_modality_availability,
        "leaderboard_lane": lane,
        "backend": backend,
        "dependency_group": dependency_group,
        "fixed_class_count": NUM_CLASSES,
    }


def multimodal_numpy_features(well_log_seq: Any, seismic_patch: Any) -> np.ndarray:
    """Flatten both real modalities without removing the log observation masks."""
    logs = np.asarray(well_log_seq, dtype=np.float32)
    seismic = np.asarray(seismic_patch, dtype=np.float32)
    if logs.ndim != 3 or logs.shape[1] != WELL_CHANNELS:
        raise ValueError(f"well_log_seq must be [B,26,L], got {logs.shape}")
    if seismic.ndim != 4 or tuple(seismic.shape[1:3]) != SEISMIC_SPATIAL_SHAPE:
        raise ValueError(f"seismic_patch must be [B,3,3,L], got {seismic.shape}")
    if logs.shape[0] != seismic.shape[0] or logs.shape[-1] != seismic.shape[-1]:
        raise ValueError("well-log and seismic batches must align")
    features = np.concatenate((logs.reshape(len(logs), -1), seismic.reshape(len(seismic), -1)), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("P5 adapters require finite fold-local inputs")
    return features


def fixed_schema_logits(scores: Any, classes: Any, *, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Map estimator decision columns into the immutable nine-class schema."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim == 1:
        values = np.column_stack((-values, values))
    class_ids = np.asarray(classes, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(class_ids):
        raise ValueError("estimator scores/classes disagree")
    floor = float(values.min() - 20.0) if values.size else -20.0
    logits = np.full((values.shape[0], num_classes), floor, dtype=np.float64)
    if np.any((class_ids < 0) | (class_ids >= num_classes)):
        raise ValueError("estimator emitted a class outside the GM09 schema")
    logits[:, class_ids] = values
    if not np.isfinite(logits).all():
        raise ValueError("estimator logits contain NaN/Inf")
    return logits.astype(np.float32)


def probability_loss(probabilities: Any, labels: Any) -> float:
    probability = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64).reshape(-1)
    selected = np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0)
    return float(-np.log(selected).mean())
