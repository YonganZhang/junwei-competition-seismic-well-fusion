"""Simple multi-output linear SGD baseline for the reservoir targets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ml_framework.model_registry import register_model


class ReservoirLinearSGD:
    """Linear regression trained one mini-batch at a time with SGD."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 3,
        learning_rate: float = 0.002,
        l2_strength: float = 0.0,
        seed: int = 2693,
        hidden_dim: int | None = None,
        weight_decay: float | None = None,
        **_: object,
    ) -> None:
        del hidden_dim, weight_decay
        if n_features <= 0 or n_outputs != 3:
            raise ValueError("n_features须为正且固定为3输出")
        if learning_rate <= 0.0 or l2_strength < 0.0:
            raise ValueError("learning_rate须为正且l2_strength不得为负")
        self.n_features = int(n_features)
        self.n_outputs = int(n_outputs)
        self.learning_rate = float(learning_rate)
        self.l2_strength = float(l2_strength)
        self.seed = int(seed)
        self.weights = np.zeros((self.n_features, self.n_outputs), dtype=np.float64)
        self.bias = np.zeros(self.n_outputs, dtype=np.float64)
        self.update_count = 0

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (*,{self.n_features}), got {x.shape}")
        if x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError("features必须是非空有限值批次")
        return x

    def _batch(self, batch: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        features, target = batch
        x = self._features(features)
        y = np.asarray(target, dtype=np.float64)
        if y.shape != (x.shape[0], self.n_outputs):
            raise ValueError(f"expected target {(x.shape[0], self.n_outputs)}, got {y.shape}")
        if not np.isfinite(y).all():
            raise ValueError("target必须全部为有限值")
        return x, y

    def predict(self, features: np.ndarray) -> np.ndarray:
        prediction = self._features(features) @ self.weights + self.bias
        if not np.isfinite(prediction).all():
            raise FloatingPointError("线性模型预测出现非有限值")
        return prediction

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y = self._batch(batch)
        error = x @ self.weights + self.bias - y
        loss = float(np.mean(error ** 2))
        grad_prediction = (2.0 / error.size) * error
        grad_weights = x.T @ grad_prediction + self.l2_strength * self.weights
        grad_bias = grad_prediction.sum(axis=0)
        next_weights = self.weights - self.learning_rate * grad_weights
        next_bias = self.bias - self.learning_rate * grad_bias
        if not np.isfinite(next_weights).all() or not np.isfinite(next_bias).all():
            raise FloatingPointError("SGD更新出现非有限参数")
        self.weights = next_weights
        self.bias = next_bias
        self.update_count += 1
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y = self._batch(batch)
        error = self.predict(x) - y
        return float(np.mean(error ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                n_features=np.int64(self.n_features),
                n_outputs=np.int64(self.n_outputs),
                learning_rate=np.float64(self.learning_rate),
                l2_strength=np.float64(self.l2_strength),
                seed=np.int64(self.seed),
                update_count=np.int64(self.update_count),
                weights=self.weights,
                bias=self.bias,
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            found_shape = (int(checkpoint["n_features"]), int(checkpoint["n_outputs"]))
            expected_shape = (self.n_features, self.n_outputs)
            found_l2 = float(checkpoint["l2_strength"])
            if found_shape != expected_shape or not np.isclose(found_l2, self.l2_strength):
                raise ValueError(
                    "checkpoint mismatch: "
                    f"expected shape/l2={expected_shape}/{self.l2_strength}, "
                    f"found={found_shape}/{found_l2}"
                )
            weights = checkpoint["weights"].astype(np.float64)
            bias = checkpoint["bias"].astype(np.float64)
            if weights.shape != self.weights.shape or bias.shape != self.bias.shape:
                raise ValueError("checkpoint参数形状不匹配")
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError("checkpoint包含非有限参数")
            self.weights = weights
            self.bias = bias
            self.update_count = int(checkpoint["update_count"])


@register_model("reservoir_linear")
def build_model(**kwargs) -> ReservoirLinearSGD:
    return ReservoirLinearSGD(**kwargs)
