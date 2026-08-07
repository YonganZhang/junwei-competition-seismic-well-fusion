"""Compatibility shim for ``_models.lithofacies.xgboost_multisoftprob_window``."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.model_registry import register_model
from _models.lithofacies.xgboost_multisoftprob_window import XGBoostWindowAdapter
from models import compatibility_task_spec


@register_model("xgboost_multisoftprob_window")
def build_model(**config: Any) -> XGBoostWindowAdapter:
    from _models.lithofacies.xgboost_multisoftprob_window import build_model as canonical_build

    return canonical_build(compatibility_task_spec(), **config)
