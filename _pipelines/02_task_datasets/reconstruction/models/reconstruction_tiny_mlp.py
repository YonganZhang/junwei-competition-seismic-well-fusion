"""Single-hidden-layer NumPy MLP for reconstruction experiments."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_framework.model_registry import register_model


class ReconstructionTinyMLP:
    """A deterministic tanh MLP trained by mini-batch gradient descent."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.01,
        ridge_alpha: float = 0.0,
        n_training_samples: int = 1,
        hidden_features: int = 8,
        seed: int = 20260713,
    ) -> None:
        if n_features <= 0 or n_training_samples <= 0 or hidden_features <= 0:
            raise ValueError(
                "n_features, n_training_samples and hidden_features must be positive"
            )
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be non-negative and finite")
        self.n_features = int(n_features)
        self.hidden_features = int(hidden_features)
        self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha)
        self.l2 = self.ridge_alpha / float(n_training_samples)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        hidden_scale = np.sqrt(2.0 / (self.n_features + self.hidden_features))
        output_scale = np.sqrt(2.0 / (self.hidden_features + 1))
        self.hidden_weights = rng.normal(
            0.0, hidden_scale, size=(self.n_features, self.hidden_features)
        )
        self.hidden_bias = np.zeros(self.hidden_features, dtype=np.float64)
        self.output_weights = rng.normal(0.0, output_scale, size=self.hidden_features)
        self.output_bias = 0.0
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

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.tanh(x @ self.hidden_weights + self.hidden_bias)
        prediction = hidden @ self.output_weights + self.output_bias
        if prediction.ndim != 1 or not np.all(np.isfinite(prediction)):
            raise FloatingPointError("tiny MLP produced non-finite predictions")
        return hidden, prediction

    def predict(self, features: np.ndarray) -> np.ndarray:
        _, prediction = self._forward(self._features(features))
        return prediction

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x)
        target = self._target(target, x.shape[0])
        hidden, prediction = self._forward(x)
        error = prediction - target
        mse = float(np.mean(error**2))

        grad_prediction = 2.0 * error / target.size
        grad_output_weights = hidden.T @ grad_prediction + 2.0 * self.l2 * self.output_weights
        grad_output_bias = float(np.sum(grad_prediction))
        grad_hidden = grad_prediction[:, None] * self.output_weights[None, :]
        grad_hidden_pre_activation = grad_hidden * (1.0 - hidden**2)
        grad_hidden_weights = (
            x.T @ grad_hidden_pre_activation + 2.0 * self.l2 * self.hidden_weights
        )
        grad_hidden_bias = np.sum(grad_hidden_pre_activation, axis=0)

        self.hidden_weights -= self.learning_rate * grad_hidden_weights
        self.hidden_bias -= self.learning_rate * grad_hidden_bias
        self.output_weights -= self.learning_rate * grad_output_weights
        self.output_bias -= self.learning_rate * grad_output_bias
        parameters = (
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            np.asarray(self.output_bias),
        )
        if not all(np.all(np.isfinite(parameter)) for parameter in parameters):
            raise FloatingPointError("tiny MLP update produced non-finite parameters")
        self.update_count += 1
        return mse

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, target = batch
        x = self._features(x)
        target = self._target(target, x.shape[0])
        loss = float(np.mean((self.predict(x) - target) ** 2))
        if not np.isfinite(loss):
            raise FloatingPointError("tiny MLP validation loss is non-finite")
        return loss

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                n_features=np.int64(self.n_features),
                hidden_features=np.int64(self.hidden_features),
                learning_rate=np.float64(self.learning_rate),
                ridge_alpha=np.float64(self.ridge_alpha),
                l2=np.float64(self.l2),
                seed=np.int64(self.seed),
                hidden_weights=self.hidden_weights,
                hidden_bias=self.hidden_bias,
                output_weights=self.output_weights,
                output_bias=np.float64(self.output_bias),
                update_count=np.int64(self.update_count),
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["n_features"]) != self.n_features:
                raise ValueError("checkpoint feature count does not match model")
            if int(checkpoint["hidden_features"]) != self.hidden_features:
                raise ValueError("checkpoint hidden width does not match model")
            hidden_weights = checkpoint["hidden_weights"].astype(np.float64)
            hidden_bias = checkpoint["hidden_bias"].astype(np.float64)
            output_weights = checkpoint["output_weights"].astype(np.float64)
            output_bias = float(checkpoint["output_bias"])
            expected_shapes = (
                (hidden_weights, (self.n_features, self.hidden_features)),
                (hidden_bias, (self.hidden_features,)),
                (output_weights, (self.hidden_features,)),
            )
            for values, shape in expected_shapes:
                if values.shape != shape or not np.all(np.isfinite(values)):
                    raise ValueError("checkpoint contains invalid tiny MLP parameters")
            if not np.isfinite(output_bias):
                raise ValueError("checkpoint contains an invalid tiny MLP output bias")
            self.hidden_weights = hidden_weights
            self.hidden_bias = hidden_bias
            self.output_weights = output_weights
            self.output_bias = output_bias
            self.update_count = int(checkpoint["update_count"])


@register_model("reconstruction_tiny_mlp")
def build_model(**kwargs) -> ReconstructionTinyMLP:
    """Build the dynamically discoverable one-hidden-layer MLP."""
    return ReconstructionTinyMLP(**kwargs)
