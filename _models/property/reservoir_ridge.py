"""Canonical NumPy ridge-SGD baseline for one or more reservoir targets."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "reservoir_ridge"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["seismic_patch", "well_log_sequence", "tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class ReservoirRidgeSGD:
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        learning_rate: float = 0.002,
        l2_strength: float = 1e-3,
        seed: int = 2693,
    ) -> None:
        if task_spec.task_type != "regression":
            raise ValueError("reservoir_ridge requires regression TaskSpec")
        if n_features <= 0 or learning_rate <= 0 or l2_strength < 0:
            raise ValueError("invalid n_features/learning_rate/l2_strength")
        self.task_spec = task_spec
        self.n_features = int(n_features)
        self.n_outputs = len(task_spec.targets)
        self.learning_rate = float(learning_rate)
        self.l2_strength = float(l2_strength)
        self.seed = int(seed)
        self.weights = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        self.bias = np.zeros(self.n_outputs, dtype=np.float64)
        self.update_count = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features or not np.isfinite(x).all():
            raise ValueError(f"expected finite [N,{self.n_features}] features, got {x.shape}")
        return x

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        return self._features(features) @ self.weights + self.bias

    def predict(self, features: np.ndarray) -> ModelOutput:
        values = self.predict_array(features)
        return ModelOutput(raw={target: values[:, i] for i, target in enumerate(self.task_spec.targets)})

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x); y = np.asarray(target, dtype=np.float64)
        if y.shape != (len(x), self.n_outputs) or not np.isfinite(y).all():
            raise ValueError(f"expected finite target {(len(x), self.n_outputs)}, got {y.shape}")
        error = self.predict_array(x) - y
        loss = float(np.mean(error ** 2))
        gradient = 2.0 * error / error.size
        self.weights -= self.learning_rate * (x.T @ gradient + self.l2_strength * self.weights)
        self.bias -= self.learning_rate * gradient.sum(axis=0)
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise FloatingPointError("ridge update produced non-finite parameters")
        self.update_count += 1
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y = batch
        return float(np.mean((self.predict_array(x) - np.asarray(y, dtype=float)) ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, n_features=self.n_features, targets=np.asarray(self.task_spec.targets),
                     learning_rate=self.learning_rate, l2_strength=self.l2_strength, seed=self.seed,
                     weights=self.weights, bias=self.bias, update_count=self.update_count)

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["n_features"]) != self.n_features:
                raise ValueError("checkpoint feature count mismatch")
            if tuple(checkpoint["targets"].tolist()) != self.task_spec.targets:
                raise ValueError("checkpoint target order mismatch")
            self.weights = checkpoint["weights"].astype(float); self.bias = checkpoint["bias"].astype(float)
            self.update_count = int(checkpoint["update_count"])


def build_model(task_spec: TaskSpec, **config: Any) -> ReservoirRidgeSGD:
    return ReservoirRidgeSGD(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "l2_strength": trial.suggest_float("l2_strength", 1e-6, 1e-1, log=True),
    }
