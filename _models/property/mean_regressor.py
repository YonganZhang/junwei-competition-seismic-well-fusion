"""Dependency-light mean baseline used to verify canonical model discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "mean_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["any"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


@dataclass
class MeanRegressor:
    task_spec: TaskSpec
    means: dict[str, float] = field(default_factory=dict)

    def fit(self, targets: Mapping[str, Sequence[float]], masks: Mapping[str, Sequence[bool]]) -> "MeanRegressor":
        for target in self.task_spec.targets:
            values = [float(value) for value, valid in zip(targets[target], masks[target]) if valid]
            if not values:
                raise ValueError(f"target {target!r} has no valid labels")
            self.means[target] = sum(values) / len(values)
        return self

    def predict(self, sample_count: int) -> ModelOutput:
        if sample_count <= 0:
            raise ValueError("sample_count must be >0")
        if set(self.means) != set(self.task_spec.targets):
            raise RuntimeError("model must be fitted before prediction")
        return ModelOutput(raw={target: [mean] * sample_count for target, mean in self.means.items()})


def build_model(task_spec: TaskSpec, **model_config: Any) -> MeanRegressor:
    if task_spec.task_type != "regression":
        raise ValueError("mean_regressor supports regression TaskSpec only")
    if model_config:
        unknown = sorted(model_config)
        raise ValueError(f"mean_regressor has no hyperparameters; unknown config={unknown}")
    return MeanRegressor(task_spec)
