"""Compatibility shim for ``_models.lithofacies.lithofacies_late_fusion``."""
from __future__ import annotations
from typing import Any
from _code.ml_framework.model_registry import register_model
from _models.lithofacies.lithofacies_late_fusion import LithofaciesLateFusion
from models import compatibility_task_spec


@register_model("lithofacies_late_fusion")
def build_model(**config: Any) -> LithofaciesLateFusion:
    from _models.lithofacies.lithofacies_late_fusion import build_model as canonical_build
    return canonical_build(compatibility_task_spec(), **config)
