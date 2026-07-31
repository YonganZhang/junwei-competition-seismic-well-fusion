"""Compatibility shim for ``_models.lithofacies.lithofacies_concat_linear``."""
from __future__ import annotations
from typing import Any
from _code.ml_framework.model_registry import register_model
from _models.lithofacies.lithofacies_concat_linear import LithofaciesConcatLinear
from models import compatibility_task_spec


@register_model("lithofacies_concat_linear")
def build_model(**config: Any) -> LithofaciesConcatLinear:
    from _models.lithofacies.lithofacies_concat_linear import build_model as canonical_build
    return canonical_build(compatibility_task_spec(), **config)
