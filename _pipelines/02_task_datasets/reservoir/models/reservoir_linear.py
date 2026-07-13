"""Compatibility shim for the canonical ``_models.property.reservoir_linear``."""
from __future__ import annotations

from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from ml_framework.model_registry import register_model
from _models.property.reservoir_linear import ReservoirLinearSGD as _CanonicalLinear


def compatibility_task_spec(n_outputs: int) -> TaskSpec:
    targets = tuple(f"target_{index}" for index in range(int(n_outputs)))
    return TaskSpec(
        track_id="property",
        task_id="legacy_reservoir_compatibility",
        task_type="regression",
        input_modalities=("tabular",),
        targets=targets,
        units={target: "unknown" for target in targets},
        label_version="legacy-compat-v1",
        target_masks={target: "finite" for target in targets},
        group_keys=("legacy_group",),
        target_transform={target: "identity" for target in targets},
        inverse_transform={target: "identity" for target in targets},
        train_loss={target: "MSE" for target in targets},
        inference_transform={target: "identity" for target in targets},
        threshold_policy={},
        calibration_policy={},
        primary_metrics=("MAE",),
        metric_directions={"MAE": "minimize"},
        visualizer_id="legacy_compatibility",
        required_figures=("prediction_vs_truth",),
    )


class ReservoirLinearSGD(_CanonicalLinear):
    """Old array-returning API backed entirely by the canonical model."""

    def __init__(self, n_features: int, n_outputs: int = 3, **kwargs: Any) -> None:
        super().__init__(compatibility_task_spec(n_outputs), n_features=n_features, **kwargs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_array(features)


@register_model("reservoir_linear")
def build_model(**kwargs: Any) -> ReservoirLinearSGD:
    return ReservoirLinearSGD(**kwargs)
