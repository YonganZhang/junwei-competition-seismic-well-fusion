"""Canonical leakage-agnostic robust linear baseline for tabular regressions.

Data splitting, target transforms and metric selection remain outside the model
module.  The fitted preprocessing pipeline therefore receives fold-train data
only when used by a compliant adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "robust_linear"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression", "ranking"],
        "input_modalities": ["tabular", "well_logs", "causal_production_history"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


@dataclass
class RobustLinear:
    task_spec: TaskSpec
    estimator: str = "huber"
    alpha: float = 1e-4
    epsilon: float = 1.35

    def __post_init__(self) -> None:
        if len(self.task_spec.targets) != 1:
            raise ValueError("robust_linear currently requires exactly one target")
        if self.estimator not in {"huber", "ridge"}:
            raise ValueError("estimator must be 'huber' or 'ridge'")
        if self.alpha < 0:
            raise ValueError("alpha must be >=0")
        if self.epsilon <= 1.0:
            raise ValueError("Huber epsilon must be >1")
        regressor = (
            HuberRegressor(alpha=self.alpha, epsilon=self.epsilon, max_iter=500)
            if self.estimator == "huber"
            else Ridge(alpha=self.alpha)
        )
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", regressor),
        ])
        self._fitted = False

    @property
    def target(self) -> str:
        return self.task_spec.targets[0]

    def fit(
        self,
        features: Sequence[Sequence[float]],
        targets: Mapping[str, Sequence[float]],
        masks: Mapping[str, Sequence[bool]],
    ) -> "RobustLinear":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets[self.target], dtype=float)
        mask = np.asarray(masks[self.target], dtype=bool) & np.isfinite(y)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be [sample, feature] and align with targets")
        if mask.sum() < 2:
            raise ValueError("at least two valid labels are required")
        self.pipeline.fit(x[mask], y[mask])
        self._fitted = True
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> ModelOutput:
        if not self._fitted:
            raise RuntimeError("model must be fitted before prediction")
        prediction = self.pipeline.predict(np.asarray(features, dtype=float))
        return ModelOutput(raw={self.target: prediction})


def build_model(task_spec: TaskSpec, **model_config: Any) -> RobustLinear:
    if task_spec.task_type not in {"regression", "ranking"}:
        raise ValueError("robust_linear supports regression/ranking TaskSpec only")
    return RobustLinear(task_spec, **model_config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    estimator = trial.suggest_categorical("estimator", ["huber", "ridge"])
    config: dict[str, Any] = {
        "estimator": estimator,
        "alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True),
    }
    if estimator == "huber":
        config["epsilon"] = trial.suggest_float("epsilon", 1.05, 2.0)
    return config
