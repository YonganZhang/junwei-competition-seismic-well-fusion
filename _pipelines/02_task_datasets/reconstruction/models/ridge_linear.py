"""Small NumPy linear Ridge model used by the reconstruction baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_framework.model_registry import register_model


class NumpyRidgeRegressor:
    """Linear regression with L2 weight decay and full/mini-batch GD steps.

    Epoch orchestration, train/validation history and checkpoint selection are
    intentionally not implemented here; those belong to
    ``ml_framework.train.train_loop``.
    """

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        ridge_alpha: float = 10.0,
        n_training_samples: int = 1,
    ) -> None:
        if n_features <= 0 or n_training_samples <= 0:
            raise ValueError("n_features and n_training_samples must be positive")
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha)
        # sklearn Ridge uses sum-of-squares + alpha*||w||^2.  The step below
        # uses mean squared error, so divide alpha by the training sample count.
        self.l2 = self.ridge_alpha / float(n_training_samples)
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0
        self.update_count = 0

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (*, {self.n_features}) features, got {x.shape}")
        return x @ self.weights + self.bias

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = np.asarray(x, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        prediction = self.predict(x)
        error = prediction - target
        mse = float(np.mean(error**2))
        grad_weights = 2.0 * (x.T @ error) / target.size + 2.0 * self.l2 * self.weights
        grad_bias = 2.0 * float(np.mean(error))
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        self.update_count += 1
        return mse

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        error = self.predict(x) - np.asarray(target, dtype=np.float64)
        return float(np.mean(error**2))

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
            self.weights = checkpoint["weights"].astype(np.float64)
            self.bias = float(checkpoint["bias"])
            self.update_count = int(checkpoint["update_count"])


@register_model("ridge_linear")
def build_model(**kwargs) -> NumpyRidgeRegressor:
    """Registry entrypoint required by the shared model skeleton."""
    return NumpyRidgeRegressor(**kwargs)
