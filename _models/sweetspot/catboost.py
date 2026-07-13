"""Thin CatBoost adapter for one approved sweetspot target at a time."""
from __future__ import annotations

from typing import Any

from .p5_common import TabularEstimatorAdapter, require_single_target

model_id = "catboost"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression", "ranking"], "input_modalities": ["tabular"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "tabular"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("catboost") from exc
    common = {"iterations": 16, "depth": 3, "learning_rate": 0.1, "verbose": False, "thread_count": 1, "random_seed": 2693, "allow_writing_files": False}
    common.update(config)
    estimator = CatBoostClassifier(**common) if task_spec.task_type in {"binary", "multiclass"} else CatBoostRegressor(**common)
    return TabularEstimatorAdapter(task_spec, estimator)
