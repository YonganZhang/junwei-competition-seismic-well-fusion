"""Thin PyTabKit RealMLP adapter using three independent regressors."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.contracts import TaskSpec
from _models.property._p5_common import IndependentTargetEstimatorAdapter, require_model_dependencies


model_id = "realmlp_regressor"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "target_strategy": "three independent estimators",
    }


def build_model(task_spec: TaskSpec, **config: Any) -> IndependentTargetEstimatorAdapter:
    pytabkit = require_model_dependencies(model_id)["pytabkit"]
    seed = int(config.get("seed", 2693))

    def factory(target: str) -> Any:
        target_offset = task_spec.targets.index(target)
        return pytabkit.RealMLP_TD_Regressor(
            device=str(config.get("device", "cpu")),
            random_state=seed + target_offset,
            n_cv=1,
            n_refit=0,
            n_threads=int(config.get("n_threads", 1)),
            n_epochs=int(config.get("n_epochs", 2)),
            batch_size=int(config.get("batch_size", 64)),
            hidden_sizes=list(config.get("hidden_sizes", [64, 64])),
            use_early_stopping=False,
            verbosity=0,
        )

    return IndependentTargetEstimatorAdapter(
        model_id=model_id, task_spec=task_spec, estimator_factory=factory
    )
