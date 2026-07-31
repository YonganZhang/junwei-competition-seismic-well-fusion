"""Canonical deterministic single-hidden-layer reconstruction MLP."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "reconstruction_tiny_mlp"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["reconstruction", "regression"],
        "input_modalities": ["coordinates", "seismic", "well_constraints"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class ReconstructionTinyMLP:
    def __init__(
        self, task_spec: TaskSpec, *, n_features: int, learning_rate: float = 0.01,
        ridge_alpha: float = 0.0, n_training_samples: int = 1,
        hidden_features: int = 8, seed: int = 20260713,
    ) -> None:
        if task_spec.task_type not in {"reconstruction", "regression"} or len(task_spec.targets) != 1:
            raise ValueError("reconstruction_tiny_mlp requires one reconstruction target")
        if n_features <= 0 or n_training_samples <= 0 or hidden_features <= 0:
            raise ValueError("model dimensions and training sample count must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0 or ridge_alpha < 0:
            raise ValueError("invalid learning rate or ridge strength")
        self.task_spec = task_spec
        self.n_features = int(n_features)
        self.hidden_features = int(hidden_features)
        self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha)
        self.l2 = self.ridge_alpha / float(n_training_samples)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.hidden_weights = rng.normal(
            0.0, np.sqrt(2.0 / (self.n_features + self.hidden_features)),
            size=(self.n_features, self.hidden_features),
        )
        self.hidden_bias = np.zeros(self.hidden_features, dtype=np.float64)
        self.output_weights = rng.normal(
            0.0, np.sqrt(2.0 / (self.hidden_features + 1)), size=self.hidden_features,
        )
        self.output_bias = 0.0
        self.update_count = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features or not np.isfinite(x).all():
            raise ValueError(f"expected finite [N,{self.n_features}] features")
        return x

    @staticmethod
    def _target(target: np.ndarray, n_rows: int) -> np.ndarray:
        y = np.asarray(target, dtype=np.float64)
        if y.shape != (n_rows,) or not np.isfinite(y).all():
            raise ValueError(f"expected finite target shape ({n_rows},)")
        return y

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.hidden_weights + self.hidden_bias)
        prediction = hidden @ self.output_weights + self.output_bias
        if not np.isfinite(prediction).all():
            raise FloatingPointError("tiny MLP prediction is non-finite")
        return hidden, prediction

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        return self._forward(self._features(features))[1]

    def predict(self, features: np.ndarray) -> ModelOutput:
        return ModelOutput(raw={self.task_spec.targets[0]: self.predict_array(features)})

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x)
        y = self._target(target, len(x))
        hidden, prediction = self._forward(x)
        error = prediction - y
        loss = float(np.mean(error**2))
        grad_prediction = 2.0 * error / len(y)
        grad_output_weights = hidden.T @ grad_prediction + 2.0 * self.l2 * self.output_weights
        grad_hidden = (grad_prediction[:, None] * self.output_weights[None, :]) * (1.0 - hidden**2)
        self.output_weights -= self.learning_rate * grad_output_weights
        self.output_bias -= self.learning_rate * float(grad_prediction.sum())
        self.hidden_weights -= self.learning_rate * (
            x.T @ grad_hidden + 2.0 * self.l2 * self.hidden_weights
        )
        self.hidden_bias -= self.learning_rate * grad_hidden.sum(axis=0)
        self.update_count += 1
        parameters = (self.hidden_weights, self.hidden_bias, self.output_weights, np.asarray(self.output_bias))
        if not all(np.isfinite(value).all() for value in parameters):
            raise FloatingPointError("tiny MLP update produced non-finite parameters")
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x)
        y = self._target(target, len(x))
        return float(np.mean((self.predict_array(x) - y) ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle, n_features=self.n_features, hidden_features=self.hidden_features,
                learning_rate=self.learning_rate, ridge_alpha=self.ridge_alpha, seed=self.seed,
                hidden_weights=self.hidden_weights, hidden_bias=self.hidden_bias,
                output_weights=self.output_weights, output_bias=self.output_bias,
                update_count=self.update_count,
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["n_features"]) != self.n_features:
                raise ValueError("checkpoint feature count mismatch")
            if int(checkpoint["hidden_features"]) != self.hidden_features:
                raise ValueError("checkpoint hidden width mismatch")
            self.hidden_weights = checkpoint["hidden_weights"].astype(float)
            self.hidden_bias = checkpoint["hidden_bias"].astype(float)
            self.output_weights = checkpoint["output_weights"].astype(float)
            self.output_bias = float(checkpoint["output_bias"])
            self.update_count = int(checkpoint["update_count"])


def build_model(task_spec: TaskSpec, **config: Any) -> ReconstructionTinyMLP:
    return ReconstructionTinyMLP(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "hidden_features": trial.suggest_categorical("hidden_features", [4, 8, 16, 32]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-2, log=True),
        "ridge_alpha": trial.suggest_float("ridge_alpha", 0.0, 10.0),
    }
