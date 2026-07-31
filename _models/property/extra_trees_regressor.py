"""Thin scikit-learn ExtraTrees adapter for P5 properties."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.property._p5_common import IndependentTargetEstimatorAdapter, require_model_dependencies


model_id = "extra_trees_regressor"


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
        return ensemble.ExtraTreesRegressor(
            n_estimators=int(config.get("n_estimators", 32)),
            max_depth=int(config.get("max_depth", 8)),
            min_samples_leaf=int(config.get("min_samples_leaf", 2)),
            max_features=float(config.get("max_features", 0.7)),
            random_state=seed,
            n_jobs=int(config.get("n_jobs", 1)),
        )

    return IndependentTargetEstimatorAdapter(
        model_id=model_id, task_spec=task_spec, estimator_factory=factory
    )
