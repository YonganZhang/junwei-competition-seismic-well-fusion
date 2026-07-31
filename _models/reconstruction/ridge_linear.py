"""Canonical ridge-linear baseline for 3-D reconstruction samples."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction.reconstruction_linear_sgd import ReconstructionLinearSGD


model_id = "ridge_linear"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction", "regression"],
        "input_modalities": ["coordinates", "seismic", "well_constraints"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class NumpyRidgeRegressor(ReconstructionLinearSGD):
    def __init__(
        self, task_spec: TaskSpec, *, n_features: int, learning_rate: float = 0.01,
        ridge_alpha: float = 10.0, n_training_samples: int = 1, **_: Any,
    ) -> None:
        super().__init__(
            task_spec,
            n_features=n_features,
            learning_rate=learning_rate,
            ridge_alpha=ridge_alpha,
            n_training_samples=n_training_samples,
        )


def build_model(task_spec: TaskSpec, **config: Any) -> NumpyRidgeRegressor:
    return NumpyRidgeRegressor(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "ridge_alpha": trial.suggest_float("ridge_alpha", 0.0, 20.0),
    }
