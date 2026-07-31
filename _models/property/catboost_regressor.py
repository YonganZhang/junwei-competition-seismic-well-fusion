"""Thin CatBoost adapter for the frozen P5 property contract."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.property._p5_common import IndependentTargetEstimatorAdapter, require_model_dependencies


model_id = "catboost_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "target_strategy": "three independent estimators",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> IndependentTargetEstimatorAdapter:
    module = require_model_dependencies(model_id)["catboost"]
    seed = int(config.get("seed", 2693))

    def factory(target: str) -> Any:
        del target
        return module.CatBoostRegressor(
            iterations=int(config.get("iterations", 20)),
            depth=int(config.get("depth", 4)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            l2_leaf_reg=float(config.get("l2_leaf_reg", 3.0)),
            loss_function="RMSE",
            random_seed=seed,
            thread_count=int(config.get("thread_count", 1)),
            allow_writing_files=False,
            verbose=False,
        )

    return IndependentTargetEstimatorAdapter(
        model_id=model_id, task_spec=task_spec, estimator_factory=factory
    )
