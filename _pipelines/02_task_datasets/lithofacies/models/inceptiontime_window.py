"""Compatibility shim for ``_models.lithofacies.inceptiontime_window``."""
from __future__ import annotations

from typing import Any

from _code.ml_framework.model_registry import register_model
from _models.lithofacies.inceptiontime_window import TsaiInceptionTimeAdapter
from models import compatibility_task_spec


@register_model("inceptiontime_window")
def build_model(**config: Any) -> TsaiInceptionTimeAdapter:
    from _models.lithofacies.inceptiontime_window import build_model as canonical_build

    return canonical_build(compatibility_task_spec(), **config)
