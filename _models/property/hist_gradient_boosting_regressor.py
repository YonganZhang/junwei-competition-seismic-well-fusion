"""Thin scikit-learn HistGradientBoosting adapter for P5 properties."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.property._p5_common import IndependentTargetEstimatorAdapter, require_model_dependencies


model_id = "hist_gradient_boosting_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "target_strategy": "three independent estimators",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> IndependentTargetEstimatorAdapter:
    ensemble = require_model_dependencies(model_id)["sklearn.ensemble"]
    seed = int(config.get("seed", 2693))

    def factory(target: str) -> Any:
        del target
        return ensemble.HistGradientBoostingRegressor(
            max_iter=int(config.get("max_iter", 20)),
            max_leaf_nodes=int(config.get("max_leaf_nodes", 15)),
            max_depth=int(config.get("max_depth", 4)),
            min_samples_leaf=int(config.get("min_samples_leaf", 8)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            l2_regularization=float(config.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=seed,
        )

    return IndependentTargetEstimatorAdapter(
        model_id=model_id, task_spec=task_spec, estimator_factory=factory
    )
