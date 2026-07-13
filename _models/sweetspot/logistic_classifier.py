"""Canonical logistic baseline returning both raw logits and probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "logistic_classifier"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["binary"],
        "input_modalities": ["tabular", "well_logs", "causal_production_history"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


@dataclass
class LogisticClassifier:
    task_spec: TaskSpec
    c: float = 1.0
    class_weight: str | None = "balanced"
    max_iter: int = 500

    def __post_init__(self) -> None:
        if len(self.task_spec.targets) != 1:
            raise ValueError("logistic_classifier currently requires exactly one target")
        if self.c <= 0 or self.max_iter <= 0:
            raise ValueError("c and max_iter must be positive")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'")
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=self.c, class_weight=self.class_weight, max_iter=self.max_iter,
                solver="liblinear", random_state=2693,
            )),
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
    ) -> "LogisticClassifier":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets[self.target], dtype=float)
        mask = np.asarray(masks[self.target], dtype=bool) & np.isfinite(y)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features must be [sample, feature] and align with targets")
        labels = y[mask].astype(int)
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("fold-train data must contain both binary classes")
        self.pipeline.fit(x[mask], labels)
        self._fitted = True
        return self

    def predict(self, features: Sequence[Sequence[float]]) -> ModelOutput:
        if not self._fitted:
            raise RuntimeError("model must be fitted before prediction")
        x = np.asarray(features, dtype=float)
        logits = self.pipeline.decision_function(x)
        probabilities = self.pipeline.predict_proba(x)[:, 1]
        return ModelOutput(raw={self.target: logits}, transformed={self.target: probabilities})


def build_model(task_spec: TaskSpec, **model_config: Any) -> LogisticClassifier:
    if task_spec.task_type != "binary":
        raise ValueError("logistic_classifier supports binary TaskSpec only")
    return LogisticClassifier(task_spec, **model_config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "c": trial.suggest_float("c", 1e-4, 1e3, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
