"""Thin XGBoost adapter for one approved sweetspot target at a time."""
from __future__ import annotations

from typing import Any

from .p5_common import TabularEstimatorAdapter, require_single_target

model_id = "xgboost"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression", "ranking"], "input_modalities": ["tabular"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "tabular"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("xgboost") from exc
    common = {"n_estimators": 16, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": 1, "random_state": 2693}
    common.update(config)
    estimator = XGBClassifier(**common) if task_spec.task_type in {"binary", "multiclass"} else XGBRegressor(**common)
    return TabularEstimatorAdapter(task_spec, estimator)
