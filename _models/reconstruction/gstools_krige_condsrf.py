"""Thin adapter for GSTools ordinary kriging with predictive variance."""
from __future__ import annotations

from typing import Any

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _models.reconstruction._p5_adapter import FittedPointAdapter, require_dependency


model_id = "gstools_krige_condsrf"


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


class GSToolsKrigingAdapter(FittedPointAdapter):
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        len_scale: float = 0.25,
        covariance: str = "Exponential",
    ) -> None:
        super().__init__(task_spec, n_features=n_features, model_id=model_id)
        if len_scale <= 0:
            raise ValueError("GSTools len_scale must be positive")
        self.len_scale = float(len_scale)
        self.covariance = str(covariance)
        self._gs = require_dependency("gstools", model_id=model_id, distribution="gstools")
        self._backend: Any | None = None

    def _fit_backend(self, features: np.ndarray, target: np.ndarray) -> None:
        covariance_class = getattr(self._gs, self.covariance, None)
        if covariance_class is None:
            raise ValueError(f"unknown GSTools covariance {self.covariance!r}")
        variance = max(float(np.var(target)), 1e-8)
        covariance_model = covariance_class(dim=3, var=variance, len_scale=self.len_scale)
        self._backend = self._gs.krige.Ordinary(
            covariance_model,
            cond_pos=features[:, -3:].T,
            cond_val=target,
            exact=True,
        )

    def _predict_backend(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._backend is None:
            raise RuntimeError("GSTools adapter is not fitted")
        prediction, variance = self._backend(features[:, -3:].T, return_var=True)
        return np.asarray(prediction, dtype=np.float64), np.asarray(variance, dtype=np.float64)


def build_model(task_spec: TaskSpec, **config: Any) -> GSToolsKrigingAdapter:
    allowed = {"n_features", "len_scale", "covariance"}
    return GSToolsKrigingAdapter(task_spec, **{key: value for key, value in config.items() if key in allowed})
