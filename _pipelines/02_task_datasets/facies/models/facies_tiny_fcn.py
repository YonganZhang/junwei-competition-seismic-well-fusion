"""Compatibility shim for ``_models.facies.facies_tiny_fcn``."""
from __future__ import annotations
from typing import Any
from _code.ml_framework.model_registry import register_model
from _models.facies.facies_tiny_fcn import FaciesTinyFCN
from models import compatibility_task_spec


@register_model("facies_tiny_fcn")
def build_model(num_classes: int, **config: Any) -> FaciesTinyFCN:
    from _models.facies.facies_tiny_fcn import build_model as canonical_build
    return canonical_build(compatibility_task_spec(num_classes), num_classes=num_classes, **config)
