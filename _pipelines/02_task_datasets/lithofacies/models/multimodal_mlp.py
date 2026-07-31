"""Compatibility shim for ``_models.lithofacies.multimodal_mlp``."""
from __future__ import annotations
from typing import Any
from _code.ml_framework.model_registry import register_model
from _models.lithofacies.multimodal_mlp import MultimodalMLP
from models import compatibility_task_spec


@register_model("multimodal_mlp")
def build_model(**config: Any) -> MultimodalMLP:
    from _models.lithofacies.multimodal_mlp import build_model as canonical_build
    return canonical_build(compatibility_task_spec(), **config)
