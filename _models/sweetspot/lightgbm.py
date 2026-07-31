"""Thin LightGBM adapter for one approved sweetspot target at a time."""
from __future__ import annotations

from typing import Any

from .p5_common import TabularEstimatorAdapter, require_single_target

model_id = "lightgbm"


def capabilities() -> dict[str, Any]:
    return {"task_types": ["binary", "multiclass", "regression", "ranking"], "input_modalities": ["tabular"], "supports_missing_mask": True, "supports_uncertainty": False, "stage1_input_key": "tabular"}


def build_model(task_spec, **config):
    require_single_target(task_spec, capabilities()["task_types"])
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        from .p5_common import dependency_skip
        raise dependency_skip("lightgbm") from exc
    common = {"n_estimators": 16, "num_leaves": 7, "max_depth": 3, "learning_rate": 0.1, "min_child_samples": 4, "n_jobs": 1, "random_state": 2693, "verbosity": -1}
    common.update(config)
    estimator = LGBMClassifier(**common) if task_spec.task_type in {"binary", "multiclass"} else LGBMRegressor(**common)
    return TabularEstimatorAdapter(task_spec, estimator)
