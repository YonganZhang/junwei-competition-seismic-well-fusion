"""Canonical linear SGD baseline for scalar 3-D reconstruction samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "reconstruction_linear_sgd"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction", "regression"],
        "input_modalities": ["coordinates", "seismic", "well_constraints"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class ReconstructionLinearSGD:
    def __init__(
        self, task_spec: TaskSpec, *, n_features: int, learning_rate: float = 0.01,
        ridge_alpha: float = 0.0, n_training_samples: int = 1,
    ) -> None:
        if task_spec.task_type not in {"reconstruction", "regression"} or len(task_spec.targets) != 1:
            raise ValueError("reconstruction_linear_sgd requires one reconstruction/regression target")
        if n_features <= 0 or n_training_samples <= 0 or learning_rate <= 0 or ridge_alpha < 0:
            raise ValueError("invalid model dimensions or optimization values")
        self.task_spec = task_spec; self.n_features = int(n_features); self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha); self.l2 = self.ridge_alpha / float(n_training_samples)
        self.weights = np.zeros(self.n_features, dtype=float); self.bias = 0.0; self.update_count = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features or not np.isfinite(x).all():
            raise ValueError(f"expected finite [N,{self.n_features}] features")
        return x

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        return self._features(features) @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> ModelOutput:
        return ModelOutput(raw={self.task_spec.targets[0]: self.predict_array(features)})

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch; x = self._features(x); y = np.asarray(target, dtype=float)
        if y.shape != (len(x),) or not np.isfinite(y).all():
            raise ValueError(f"expected finite target shape ({len(x)},)")
        error = self.predict_array(x) - y; loss = float(np.mean(error ** 2))
        self.weights -= self.learning_rate * (2.0 * (x.T @ error) / len(y) + 2.0 * self.l2 * self.weights)
        self.bias -= self.learning_rate * 2.0 * float(error.mean()); self.update_count += 1
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise FloatingPointError("linear update produced non-finite parameters")
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y = batch
        return float(np.mean((self.predict_array(x) - np.asarray(y, dtype=float)) ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, n_features=self.n_features, learning_rate=self.learning_rate,
                     ridge_alpha=self.ridge_alpha, weights=self.weights, bias=self.bias,
                     update_count=self.update_count)

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["n_features"]) != self.n_features:
                raise ValueError("checkpoint feature count mismatch")
            self.weights = checkpoint["weights"].astype(float); self.bias = float(checkpoint["bias"])
            self.update_count = int(checkpoint["update_count"])


def build_model(task_spec: TaskSpec, **config: Any) -> ReconstructionLinearSGD:
    return ReconstructionLinearSGD(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "ridge_alpha": trial.suggest_float("ridge_alpha", 0.0, 10.0),
    }
