"""Canonical NumPy linear-SGD baseline for reservoir regression tasks."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "reservoir_linear"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["seismic_patch", "well_log_sequence", "tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class ReservoirLinearSGD:
    """One-or-more-output linear regression trained one mini-batch at a time."""

    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        learning_rate: float = 0.002,
        l2_strength: float = 0.0,
        seed: int = 2693,
        **_: Any,
    ) -> None:
        if task_spec.task_type != "regression":
            raise ValueError("reservoir_linear requires regression TaskSpec")
        if n_features <= 0 or learning_rate <= 0.0 or l2_strength < 0.0:
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
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected (*,{self.n_features}), got {x.shape}")
        if x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError("features must be a nonempty finite batch")
        return x

    def _batch(self, batch: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        features, target = batch
        x = self._features(features)
        y = np.asarray(target, dtype=np.float64)
        if y.shape != (x.shape[0], self.n_outputs) or not np.isfinite(y).all():
            raise ValueError(f"expected finite target {(x.shape[0], self.n_outputs)}, got {y.shape}")
        return x, y

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        prediction = self._features(features) @ self.weights + self.bias
        if not np.isfinite(prediction).all():
            raise FloatingPointError("linear prediction produced non-finite values")
        return prediction

    def predict(self, features: np.ndarray) -> ModelOutput:
        prediction = self.predict_array(features)
        return ModelOutput(
            raw={target: prediction[:, index] for index, target in enumerate(self.task_spec.targets)}
        )

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y = self._batch(batch)
        error = x @ self.weights + self.bias - y
        loss = float(np.mean(error**2))
        grad_prediction = (2.0 / error.size) * error
        next_weights = self.weights - self.learning_rate * (
            x.T @ grad_prediction + self.l2_strength * self.weights
        )
        next_bias = self.bias - self.learning_rate * grad_prediction.sum(axis=0)
        if not np.isfinite(next_weights).all() or not np.isfinite(next_bias).all():
            raise FloatingPointError("SGD update produced non-finite parameters")
        self.weights = next_weights
        self.bias = next_bias
        self.update_count += 1
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        x, y = self._batch(batch)
        return float(np.mean((self.predict_array(x) - y) ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                n_features=np.int64(self.n_features),
                targets=np.asarray(self.task_spec.targets),
                learning_rate=np.float64(self.learning_rate),
                l2_strength=np.float64(self.l2_strength),
                seed=np.int64(self.seed),
                update_count=np.int64(self.update_count),
                weights=self.weights,
                bias=self.bias,
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["n_features"]) != self.n_features:
                raise ValueError("checkpoint feature count mismatch")
            if tuple(checkpoint["targets"].tolist()) != self.task_spec.targets:
                raise ValueError("checkpoint target order mismatch")
            if not np.isclose(float(checkpoint["l2_strength"]), self.l2_strength):
                raise ValueError("checkpoint L2 strength mismatch")
            weights = checkpoint["weights"].astype(np.float64)
            bias = checkpoint["bias"].astype(np.float64)
            if weights.shape != self.weights.shape or bias.shape != self.bias.shape:
                raise ValueError("checkpoint parameter shape mismatch")
            if not np.isfinite(weights).all() or not np.isfinite(bias).all():
                raise ValueError("checkpoint contains non-finite parameters")
            self.weights = weights
            self.bias = bias
            self.update_count = int(checkpoint["update_count"])


def build_model(task_spec: TaskSpec, **config: Any) -> ReservoirLinearSGD:
    return ReservoirLinearSGD(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-2, log=True),
        "l2_strength": trial.suggest_float("l2_strength", 0.0, 1e-2),
    }
