"""Lightly regularised NumPy linear model for reconstruction experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_framework.model_registry import register_model


class ReconstructionLinearSGD:
    """Linear regressor updated one supplied mini-batch at a time."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        ridge_alpha: float = 0.0,
        n_training_samples: int = 1,
    ) -> None:
        if n_features <= 0 or n_training_samples <= 0:
            raise ValueError("n_features and n_training_samples must be positive")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative and finite")
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha)
        # Match the existing Ridge scaling while keeping regularisation light
        # for the large reconstruction training arrays.
        self.l2 = self.ridge_alpha / float(n_training_samples)
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0
        self.update_count = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (*, {self.n_features}) features, got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("features must be finite")
        return x

    @staticmethod
    def _target(target: np.ndarray, n_rows: int) -> np.ndarray:
        y = np.asarray(target, dtype=np.float64)
        if y.ndim != 1 or y.shape[0] != n_rows:
            raise ValueError(f"expected target shape ({n_rows},), got {y.shape}")
        if not np.all(np.isfinite(y)):
            raise ValueError("target must be finite")
        return y

    def predict(self, features: np.ndarray) -> np.ndarray:
        prediction = self._features(features) @ self.weights + self.bias
        if prediction.ndim != 1 or not np.all(np.isfinite(prediction)):
            raise FloatingPointError("linear model produced non-finite predictions")
        return prediction

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x)
        target = self._target(target, x.shape[0])
        error = self.predict(x) - target
        mse = float(np.mean(error**2))
        grad_weights = 2.0 * (x.T @ error) / target.size + 2.0 * self.l2 * self.weights
        grad_bias = 2.0 * float(np.mean(error))
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        if not np.all(np.isfinite(self.weights)) or not np.isfinite(self.bias):
            raise FloatingPointError("linear model update produced non-finite parameters")
        self.update_count += 1
        return mse

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x)
        target = self._target(target, x.shape[0])
        loss = float(np.mean((self.predict(x) - target) ** 2))
        if not np.isfinite(loss):
            raise FloatingPointError("linear validation loss is non-finite")
        return loss

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                n_features=np.int64(self.n_features),
                learning_rate=np.float64(self.learning_rate),
                ridge_alpha=np.float64(self.ridge_alpha),
                l2=np.float64(self.l2),
                weights=self.weights,
                bias=np.float64(self.bias),
                update_count=np.int64(self.update_count),
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["n_features"]) != self.n_features:
                raise ValueError("checkpoint feature count does not match model")
            weights = checkpoint["weights"].astype(np.float64)
            bias = float(checkpoint["bias"])
            if weights.shape != (self.n_features,) or not np.all(np.isfinite(weights)):
                raise ValueError("checkpoint contains invalid linear weights")
            if not np.isfinite(bias):
                raise ValueError("checkpoint contains an invalid linear bias")
            self.weights = weights
            self.bias = bias
            self.update_count = int(checkpoint["update_count"])


@register_model("reconstruction_linear_sgd")
def build_model(**kwargs) -> ReconstructionLinearSGD:
    """Build the dynamically discoverable linear SGD alternative."""
    return ReconstructionLinearSGD(**kwargs)
