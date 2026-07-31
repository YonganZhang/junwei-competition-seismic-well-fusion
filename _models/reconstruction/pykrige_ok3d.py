"""Thin adapter for PyKrige OrdinaryKriging3D."""
from __future__ import annotations

from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import FittedPointAdapter, require_dependency


model_id = "pykrige_ok3d"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["coordinates", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": True,
        "batch_representation": "point",
        "trainable": False,
        "dependency_group": "geostat-cpu",
    }


class PyKrigeOK3DAdapter(FittedPointAdapter):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        variogram_model: str = "linear",
        nlags: int = 6,
    ) -> None:
        super().__init__(task_spec, n_features=n_features, model_id=model_id)
        if nlags <= 0:
            raise ValueError("nlags must be positive")
        self.variogram_model = str(variogram_model)
        self.nlags = int(nlags)
        module = require_dependency("pykrige.ok3d", model_id=model_id, distribution="pykrige")
        self._ok3d_class = module.OrdinaryKriging3D
        self._backend: Any | None = None

    def _fit_backend(self, features: np.ndarray, target: np.ndarray) -> None:
        xyz = features[:, -3:]
        self._backend = self._ok3d_class(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], target,
            variogram_model=self.variogram_model,
            nlags=min(self.nlags, max(2, len(target) // 4)),
            verbose=False,
            enable_plotting=False,
        )

    def _predict_backend(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._backend is None:
            raise RuntimeError("PyKrige adapter is not fitted")
        xyz = features[:, -3:]
        prediction, variance = self._backend.execute(
            "points", xyz[:, 0], xyz[:, 1], xyz[:, 2]
        )
        return np.asarray(prediction, dtype=np.float64), np.asarray(variance, dtype=np.float64)


def build_model(task_spec: TaskSpec, **config: Any) -> PyKrigeOK3DAdapter:
    allowed = {"n_features", "variogram_model", "nlags"}
    return PyKrigeOK3DAdapter(task_spec, **{key: value for key, value in config.items() if key in allowed})
