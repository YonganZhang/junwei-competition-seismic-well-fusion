"""Canonical dependency-light one-hidden-layer MLP for reservoir regression."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "tiny_mlp"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["seismic_patch", "well_log_sequence", "tabular"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class TinyMultiOutputMLP:
    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        n_features: int,
        hidden_dim: int = 24,
        learning_rate: float = 0.002,
        weight_decay: float = 1e-5,
        seed: int = 2693,
        **_: Any,
    ) -> None:
        if task_spec.task_type != "regression":
            raise ValueError("tiny_mlp requires regression TaskSpec")
        if n_features <= 0 or hidden_dim <= 0 or learning_rate <= 0 or weight_decay < 0:
            raise ValueError("invalid n_features/hidden_dim/learning_rate/weight_decay")
        rng = np.random.default_rng(seed)
        self.task_spec = task_spec
        self.n_features = int(n_features)
        self.n_outputs = len(task_spec.targets)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.params = {
            "w1": rng.normal(0.0, np.sqrt(2.0 / n_features), (n_features, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "w2": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, self.n_outputs)),
            "b2": np.zeros(self.n_outputs),
        }
        self.mom1 = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.mom2 = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.update_count = 0

    def _forward(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features or not np.isfinite(x).all():
            raise ValueError(f"expected finite (*,{self.n_features}) features, got {x.shape}")
        pre = x @ self.params["w1"] + self.params["b1"]
        hidden = np.maximum(pre, 0.0)
        return pre, hidden, hidden @ self.params["w2"] + self.params["b2"]

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        return self._forward(features)[2]

    def predict(self, features: np.ndarray) -> ModelOutput:
        prediction = self.predict_array(features)
        return ModelOutput(
            raw={target: prediction[:, index] for index, target in enumerate(self.task_spec.targets)}
        )

    def train_batch(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        features, target = batch
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if y.shape != (x.shape[0], self.n_outputs) or not np.isfinite(y).all():
            raise ValueError(f"expected finite target {(x.shape[0], self.n_outputs)}, got {y.shape}")
        pre, hidden, prediction = self._forward(x)
        error = prediction - y
        loss = float(np.mean(error**2))
        grad_prediction = (2.0 / error.size) * error
        grads = {
            "w2": hidden.T @ grad_prediction + self.weight_decay * self.params["w2"],
            "b2": grad_prediction.sum(axis=0),
        }
        grad_pre = (grad_prediction @ self.params["w2"].T) * (pre > 0.0)
        grads["w1"] = x.T @ grad_pre + self.weight_decay * self.params["w1"]
        grads["b1"] = grad_pre.sum(axis=0)
        self.update_count += 1
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for key, grad in grads.items():
            self.mom1[key] = beta1 * self.mom1[key] + (1.0 - beta1) * grad
            self.mom2[key] = beta2 * self.mom2[key] + (1.0 - beta2) * (grad**2)
            first = self.mom1[key] / (1.0 - beta1**self.update_count)
            second = self.mom2[key] / (1.0 - beta2**self.update_count)
            self.params[key] -= self.learning_rate * first / (np.sqrt(second) + epsilon)
        return loss

    def validation_loss(self, batch: tuple[np.ndarray, np.ndarray]) -> float:
        features, target = batch
        y = np.asarray(target, dtype=np.float64)
        if y.shape != (len(features), self.n_outputs) or not np.isfinite(y).all():
            raise ValueError(f"expected finite target {(len(features), self.n_outputs)}, got {y.shape}")
        return float(np.mean((self.predict_array(features) - y) ** 2))

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                n_features=np.int64(self.n_features),
                targets=np.asarray(self.task_spec.targets),
                hidden_dim=np.int64(self.hidden_dim),
                learning_rate=np.float64(self.learning_rate),
                weight_decay=np.float64(self.weight_decay),
                seed=np.int64(self.seed),
                update_count=np.int64(self.update_count),
                **self.params,
            )

    def load_checkpoint(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as checkpoint:
            expected = (self.n_features, self.task_spec.targets, self.hidden_dim)
            found = (
                int(checkpoint["n_features"]),
                tuple(checkpoint["targets"].tolist()),
                int(checkpoint["hidden_dim"]),
            )
            if expected != found:
                raise ValueError(f"checkpoint shape/target mismatch: expected={expected}, found={found}")
            for key in self.params:
                self.params[key] = checkpoint[key].astype(np.float64)
            self.update_count = int(checkpoint["update_count"])


def build_model(task_spec: TaskSpec, **config: Any) -> TinyMultiOutputMLP:
    return TinyMultiOutputMLP(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [16, 24, 48]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
    }
