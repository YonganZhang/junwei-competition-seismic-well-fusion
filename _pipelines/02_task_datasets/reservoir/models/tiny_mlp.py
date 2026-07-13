"""A small NumPy one-hidden-layer MLP for the three fixed reservoir targets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_framework.model_registry import register_model


class TinyMultiOutputMLP:
    def __init__(
        self,
        n_features: int,
        n_outputs: int = 3,
        hidden_dim: int = 24,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-5,
        seed: int = 2693,
    ) -> None:
        if n_features <= 0 or n_outputs != 3 or hidden_dim <= 0:
            raise ValueError("n_features/hidden_dim须为正且固定为3输出")
        rng = np.random.default_rng(seed)
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.params = {
            "w1": rng.normal(0.0, np.sqrt(2.0 / n_features), (n_features, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, n_outputs)),
            "b2": np.zeros(n_outputs),
        }
        self.mom1 = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mom2 = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.update_count = 0

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (*,{self.n_features}), got {x.shape}")
        pre = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(pre, 0.0)
        prediction = hidden @ self.params["w2"] + self.params["b2"]
        return pre, hidden, prediction

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[2]

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        features, target = batch
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        pre, hidden, prediction = self._forward(x)
        error = prediction - y
        loss = float(np.mean(error ** 2))
        scale = 2.0 / error.size
        grad_prediction = scale * error
        grads = {
            "w2": hidden.T @ grad_prediction + self.weight_decay * self.params["w2"],
            "b2": grad_prediction.sum(axis=0),
        }
        grad_hidden = grad_prediction @ self.params["w2"].T
        grad_pre = grad_hidden * (pre > 0.0)
        grads["w1"] = x.T @ grad_pre + self.weight_decay * self.params["w1"]
        grads["b1"] = grad_pre.sum(axis=0)
        self.update_count += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for key, grad in grads.items():
            self.mom1[key] = beta1 * self.mom1[key] + (1.0 - beta1) * grad
            self.mom2[key] = beta2 * self.mom2[key] + (1.0 - beta2) * (grad ** 2)
            first = self.mom1[key] / (1.0 - beta1 ** self.update_count)
            second = self.mom2[key] / (1.0 - beta2 ** self.update_count)
            self.params[key] -= self.learning_rate * first / (np.sqrt(second) + epsilon)
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        features, target = batch
        error = self.predict(features) - np.asarray(target, dtype=np.float64)
        return float(np.mean(error ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                n_features=np.int64(self.n_features),
                n_outputs=np.int64(self.n_outputs),
                hidden_dim=np.int64(self.hidden_dim),
                learning_rate=np.float64(self.learning_rate),
                weight_decay=np.float64(self.weight_decay),
                seed=np.int64(self.seed),
                update_count=np.int64(self.update_count),
                **self.params,
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            expected = (self.n_features, self.n_outputs, self.hidden_dim)
            found = (
                int(checkpoint["n_features"]),
                int(checkpoint["n_outputs"]),
                int(checkpoint["hidden_dim"]),
            )
            if expected != found:
                raise ValueError(f"checkpoint shape mismatch: expected={expected}, found={found}")
            for key in self.params:
                self.params[key] = checkpoint[key].astype(np.float64)
            self.update_count = int(checkpoint["update_count"])


@register_model("tiny_mlp")
def build_model(**kwargs) -> TinyMultiOutputMLP:
    return TinyMultiOutputMLP(**kwargs)

