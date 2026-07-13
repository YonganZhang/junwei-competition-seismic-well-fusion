"""Compatibility shim for the canonical ``_models.property.reservoir_ridge``."""
from __future__ import annotations

from typing import Any

import numpy as np

from ml_framework.model_registry import register_model
from _models.property.reservoir_ridge import ReservoirRidgeSGD as _CanonicalRidge
from models.reservoir_linear import compatibility_task_spec


class ReservoirRidgeSGD(_CanonicalRidge):
    def __init__(
        self,
        n_features: int,
        n_outputs: int = 3,
        l2_strength: float = 1e-3,
        weight_decay: float | None = None,
        **kwargs: Any,
    ) -> None:
        if weight_decay is not None:
            l2_strength = float(weight_decay)
        super().__init__(
            compatibility_task_spec(n_outputs),
            n_features=n_features,
            l2_strength=l2_strength,
            **kwargs,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_array(features)


@register_model("reservoir_ridge")
def build_model(**kwargs: Any) -> ReservoirRidgeSGD:
    return ReservoirRidgeSGD(**kwargs)
