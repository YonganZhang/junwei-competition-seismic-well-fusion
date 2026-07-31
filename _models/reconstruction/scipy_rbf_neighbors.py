"""Thin adapter for SciPy's local-neighbour RBFInterpolator."""
from __future__ import annotations

from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import FittedPointAdapter, require_dependency


model_id = "scipy_rbf_neighbors"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction"],
        "input_modalities": ["coordinates", "seismic", "well_constraints_conditional_only"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "batch_representation": "point",
        "trainable": False,
        "dependency_group": "geostat-cpu",
    }


class SciPyRBFAdapter(FittedPointAdapter):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        neighbors: int = 16,
        smoothing: float = 0.0,
        kernel: str = "thin_plate_spline",
    ) -> None:
        super().__init__(task_spec, n_features=n_features, model_id=model_id)
        if neighbors <= 0 or smoothing < 0:
            raise ValueError("RBF neighbors must be positive and smoothing non-negative")
        self.neighbors = int(neighbors)
        self.smoothing = float(smoothing)
        self.kernel = str(kernel)
        scipy_interpolate = require_dependency(
            "scipy.interpolate", model_id=model_id, distribution="scipy"
        )
        self._rbf_class = scipy_interpolate.RBFInterpolator
        self._backend: Any | None = None

    def _fit_backend(self, features: np.ndarray, target: np.ndarray) -> None:
        coordinates = features[:, -3:]
        self._backend = self._rbf_class(
            coordinates,
            target,
            neighbors=min(self.neighbors, len(coordinates)),
            smoothing=self.smoothing,
            kernel=self.kernel,
        )

    def _predict_backend(self, features: np.ndarray) -> tuple[np.ndarray, None]:
        if self._backend is None:
            raise RuntimeError("RBF adapter is not fitted")
        return np.asarray(self._backend(features[:, -3:]), dtype=np.float64), None


def build_model(task_spec: TaskSpec, **config: Any) -> SciPyRBFAdapter:
    allowed = {"n_features", "neighbors", "smoothing", "kernel"}
    return SciPyRBFAdapter(task_spec, **{key: value for key, value in config.items() if key in allowed})
