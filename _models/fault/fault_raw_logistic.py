"""Canonical weighted fault logistic baseline using raw amplitude only."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "fault_raw_logistic"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["binary"],
        "input_modalities": ["seismic_patch"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
    }


class FaultRawLogistic:
    description = "weighted_logistic_pixel_classifier_with_raw_amplitude_only"

    def __init__(self, task_spec: TaskSpec, *, seed: int = 2693, alpha: float = 1e-4) -> None:
        if task_spec.task_type != "binary" or len(task_spec.targets) != 1:
            raise ValueError("fault_raw_logistic requires one binary target")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.task_spec = task_spec
        self.estimator = SGDClassifier(
            loss="log_loss", penalty="l2", alpha=alpha, max_iter=1, tol=None,
            random_state=seed, average=True,
        )
        self._initialized = False

    @staticmethod
    def _features(patches: np.ndarray) -> np.ndarray:
        array = np.asarray(patches, dtype=np.float32)
        if array.ndim != 4 or array.shape[1] != 1:
            raise ValueError(f"expected [B,1,H,W] seismic patches, received {array.shape}")
        features = array[:, 0].reshape(-1, 1)
        if not np.isfinite(features).all():
            raise ValueError("raw-amplitude features must be finite")
        return features

    @staticmethod
    def _target_weight(labels: np.ndarray, weights: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(labels, dtype=np.uint8).ravel()
        weight = np.asarray(weights, dtype=np.float32).ravel()
        if target.size != size or weight.size != size:
            raise ValueError("labels and weights must match the patch voxel count")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("fault labels must be binary")
        if not np.isfinite(weight).all() or np.any(weight < 0) or weight.sum() <= 0:
            raise ValueError("weights must be finite, non-negative and have positive sum")
        return target, weight

    def train_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        features = self._features(patches)
        target, weight = self._target_weight(labels, weights, len(features))
        kwargs = {"classes": np.asarray([0, 1], dtype=np.uint8)} if not self._initialized else {}
        self.estimator.partial_fit(features, target, sample_weight=weight, **kwargs)
        self._initialized = True
        probability = self.estimator.predict_proba(features)[:, 1]
        return float(log_loss(target, probability, sample_weight=weight, labels=[0, 1]))

    def loss_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        if not self._initialized:
            raise RuntimeError("model must receive a training batch before validation")
        features = self._features(patches)
        target, weight = self._target_weight(labels, weights, len(features))
        probability = self.estimator.predict_proba(features)[:, 1]
        return float(log_loss(target, probability, sample_weight=weight, labels=[0, 1]))

    def decision_batch(self, patches: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        array = np.asarray(patches)
        logits = np.asarray(self.estimator.decision_function(self._features(array)), dtype=float)
        return logits.reshape(array.shape[0], *array.shape[-2:])

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        logits = self.decision_batch(patches)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))

    def predict_output(self, patches: np.ndarray) -> ModelOutput:
        target = self.task_spec.targets[0]
        logits = self.decision_batch(patches)
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        return ModelOutput(raw={target: logits}, transformed={target: probability})


def build_model(task_spec: TaskSpec, **config: Any) -> FaultRawLogistic:
    return FaultRawLogistic(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True)}
