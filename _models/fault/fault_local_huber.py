"""Canonical modified-Huber fault baseline over deterministic local features."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.ndimage import sobel, uniform_filter
from sklearn.linear_model import SGDClassifier

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "fault_local_huber"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["binary"],
        "input_modalities": ["seismic_patch"],
        "supports_missing_mask": False,
        "supports_uncertainty": False,
    }


class FaultLocalHuber:
    description = "modified_huber_pixel_classifier_with_local_convolutional_features"

    def __init__(self, task_spec: TaskSpec, *, seed: int = 2693, alpha: float = 1e-4) -> None:
        if task_spec.task_type != "binary" or len(task_spec.targets) != 1:
            raise ValueError("fault_local_huber requires one binary target")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.task_spec = task_spec
        self.estimator = SGDClassifier(
            loss="modified_huber", penalty="l2", alpha=alpha, max_iter=1, tol=None,
            random_state=seed, average=True,
        )
        self._initialized = False

    @staticmethod
    def _features(patches: np.ndarray) -> np.ndarray:
        array = np.asarray(patches, dtype=np.float32)
        if array.ndim != 4 or array.shape[1] != 1:
            raise ValueError(f"expected [B,1,H,W] seismic patches, received {array.shape}")
        features: list[np.ndarray] = []
        for patch in array[:, 0]:
            if not np.isfinite(patch).all():
                raise ValueError("seismic features must be finite")
            local_mean = uniform_filter(patch, size=3, mode="nearest")
            local_sq_mean = uniform_filter(patch * patch, size=3, mode="nearest")
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
            features.append(np.column_stack((
                patch.ravel(), local_mean.ravel(), local_std.ravel(),
                (sobel(patch, axis=0, mode="nearest") / 8.0).ravel(),
                (sobel(patch, axis=1, mode="nearest") / 8.0).ravel(),
            )))
        return np.concatenate(features).astype(np.float32)

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

    def _loss(self, features: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
        score = np.asarray(self.estimator.decision_function(features), dtype=float)
        margin = (2.0 * target.astype(float) - 1.0) * score
        losses = np.where(
            margin >= 1.0,
            0.0,
            np.where(margin >= -1.0, (1.0 - margin) ** 2, -4.0 * margin),
        )
        result = float(np.average(losses, weights=weight))
        if not np.isfinite(result):
            raise FloatingPointError("modified-Huber loss is non-finite")
        return result

    def train_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        features = self._features(patches)
        target, weight = self._target_weight(labels, weights, len(features))
        kwargs = {"classes": np.asarray([0, 1], dtype=np.uint8)} if not self._initialized else {}
        self.estimator.partial_fit(features, target, sample_weight=weight, **kwargs)
        self._initialized = True
        return self._loss(features, target, weight)

    def loss_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        if not self._initialized:
            raise RuntimeError("model must receive a training batch before validation")
        features = self._features(patches)
        target, weight = self._target_weight(labels, weights, len(features))
        return self._loss(features, target, weight)

    def decision_batch(self, patches: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        array = np.asarray(patches)
        logits = np.asarray(self.estimator.decision_function(self._features(array)), dtype=float)
        return logits.reshape(array.shape[0], *array.shape[-2:])

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        logits = self.decision_batch(patches)
        # Modified-Huber predict_proba follows sklearn's clipped score mapping;
        # use the estimator itself rather than silently treating scores as logits.
        array = np.asarray(patches)
        probability = self.estimator.predict_proba(self._features(array))[:, 1]
        return probability.reshape(array.shape[0], *array.shape[-2:])

    def predict_output(self, patches: np.ndarray) -> ModelOutput:
        target = self.task_spec.targets[0]
        return ModelOutput(raw={target: self.decision_batch(patches)}, transformed={target: self.predict_batch(patches)})


def build_model(task_spec: TaskSpec, **config: Any) -> FaultLocalHuber:
    return FaultLocalHuber(task_spec, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True)}
