"""Thin LightGBM adapter for independent P5 property targets."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.property._p5_common import IndependentTargetEstimatorAdapter, require_model_dependencies


model_id = "lightgbm_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "target_strategy": "three independent estimators",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> IndependentTargetEstimatorAdapter:
    module = require_model_dependencies(model_id)["lightgbm"]
    seed = int(config.get("seed", 2693))

    def factory(target: str) -> Any:
        del target
        return module.LGBMRegressor(
            n_estimators=int(config.get("n_estimators", 30)),
            num_leaves=int(config.get("num_leaves", 8)),
            max_depth=int(config.get("max_depth", 4)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            min_child_samples=int(config.get("min_child_samples", 8)),
            reg_lambda=float(config.get("reg_lambda", 1e-3)),
            random_state=seed,
            n_jobs=int(config.get("n_jobs", 1)),
            verbosity=-1,
        )

    return IndependentTargetEstimatorAdapter(
        model_id=model_id, task_spec=task_spec, estimator_factory=factory
    )
