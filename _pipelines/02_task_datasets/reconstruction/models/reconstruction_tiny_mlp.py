"""Compatibility shim for canonical reconstruction tiny MLP."""
from __future__ import annotations
from typing import Any
import numpy as np
from ml_framework.model_registry import register_model
from _models.reconstruction.reconstruction_tiny_mlp import ReconstructionTinyMLP as _Canonical
from models import compatibility_task_spec


class ReconstructionTinyMLP(_Canonical):
    def __init__(self, **config: Any) -> None:
        super().__init__(compatibility_task_spec(), **config)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_array(features)


@register_model("reconstruction_tiny_mlp")
def build_model(**config: Any) -> ReconstructionTinyMLP:
    return ReconstructionTinyMLP(**config)
