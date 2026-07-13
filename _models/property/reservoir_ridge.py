"""Canonical NumPy ridge-SGD baseline for one or more reservoir targets."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.property.reservoir_linear import ReservoirLinearSGD


model_id = "reservoir_ridge"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["seismic_patch", "well_log_sequence", "tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class ReservoirRidgeSGD(ReservoirLinearSGD):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        learning_rate: float = 0.002,
        l2_strength: float = 1e-3,
        seed: int = 2693,
        **_: Any,
    ) -> None:
        super().__init__(
            task_spec,
            n_features=n_features,
            learning_rate=learning_rate,
            l2_strength=l2_strength,
            seed=seed,
        )


def build_model(task_spec: TaskSpec, **config: Any) -> ReservoirRidgeSGD:
    return ReservoirRidgeSGD(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "l2_strength": trial.suggest_float("l2_strength", 1e-6, 1e-1, log=True),
    }
