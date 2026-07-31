"""Compatibility shim for ``_models.facies.facies_linear_pixel``."""
from __future__ import annotations
from typing import Any
from _code.ml_framework.model_registry import register_model
from _models.facies.facies_linear_pixel import FaciesLinearPixel
from models import compatibility_task_spec


@register_model("facies_linear_pixel")
def build_model(num_classes: int, **config: Any) -> FaciesLinearPixel:
    from _models.facies.facies_linear_pixel import build_model as canonical_build
    return canonical_build(compatibility_task_spec(num_classes), num_classes=num_classes, **config)
