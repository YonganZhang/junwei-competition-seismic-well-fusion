"""Compatibility shim for the canonical ``_models.property.tiny_mlp``."""
from __future__ import annotations

from typing import Any

import numpy as np

from ml_framework.model_registry import register_model
from _models.property.tiny_mlp import TinyMultiOutputMLP as _CanonicalMLP
from models.reservoir_linear import compatibility_task_spec


class TinyMultiOutputMLP(_CanonicalMLP):
    def __init__(self, n_features: int, n_outputs: int = 3, **kwargs: Any) -> None:
        super().__init__(compatibility_task_spec(n_outputs), n_features=n_features, **kwargs)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_array(features)


@register_model("tiny_mlp")
def build_model(**kwargs: Any) -> TinyMultiOutputMLP:
    return TinyMultiOutputMLP(**kwargs)
