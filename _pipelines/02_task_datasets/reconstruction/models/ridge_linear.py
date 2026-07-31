"""Compatibility shim for canonical reconstruction ridge linear."""
from __future__ import annotations
from typing import Any
import numpy as np
from ml_framework.model_registry import register_model
from _models.reconstruction.ridge_linear import NumpyRidgeRegressor as _Canonical
from models import compatibility_task_spec


class NumpyRidgeRegressor(_Canonical):
    def __init__(self, **config: Any) -> None:
        super().__init__(compatibility_task_spec(), **config)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_array(features)


@register_model("ridge_linear")
def build_model(**config: Any) -> NumpyRidgeRegressor:
    return NumpyRidgeRegressor(**config)
