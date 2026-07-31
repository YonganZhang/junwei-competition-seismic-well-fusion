"""Compatibility shim for ``_models.facies.small_unet``."""
from __future__ import annotations
from typing import Any
from _code.ml_framework.model_registry import register_model
from _models.facies.small_unet import DoubleConv, SmallUNet
from models import compatibility_task_spec


@register_model("small_unet")
def build_model(num_classes: int, **config: Any) -> SmallUNet:
    from _models.facies.small_unet import build_model as canonical_build
    return canonical_build(compatibility_task_spec(num_classes), num_classes=num_classes, **config)
