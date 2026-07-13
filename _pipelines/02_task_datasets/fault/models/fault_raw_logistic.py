"""Weighted incremental logistic fault model using raw amplitude only."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from ml_framework.model_registry import register_model


class FaultRawLogistic:
    """Weighted pixel classifier whose sole input feature is raw amplitude."""

    description = "weighted_logistic_pixel_classifier_with_raw_amplitude_only"

    def __init__(self, seed: int = 2693) -> None:
        self.estimator = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            tol=None,
            random_state=seed,
            average=True,
        )
        self._initialized = False

    @staticmethod
    def _features(patches: np.ndarray) -> np.ndarray:
        patch_array = np.asarray(patches, dtype=np.float32)
        if patch_array.ndim != 4 or patch_array.shape[1] != 1:
            raise ValueError(f"expected [B,1,H,W] seismic patches, received {patch_array.shape}")
        features = patch_array[:, 0].reshape(-1, 1)
        if not np.isfinite(features).all():
            raise ValueError("raw-amplitude features must be finite")
        return features

    @staticmethod
    def _targets_and_weights(
        labels: np.ndarray, weights: np.ndarray, expected_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(labels, dtype=np.uint8).ravel()
        sample_weight = np.asarray(weights, dtype=np.float32).ravel()
        if targets.size != expected_size or sample_weight.size != expected_size:
            raise ValueError("labels and weights must match the patch voxel count")
        if not np.isin(targets, (0, 1)).all():
            raise ValueError("fault labels must be binary")
        if not np.isfinite(sample_weight).all() or np.any(sample_weight < 0):
            raise ValueError("sample weights must be finite and non-negative")
        if float(sample_weight.sum()) <= 0.0:
            raise ValueError("sample weights must have positive total weight")
        return targets, sample_weight

    def train_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        features = self._features(patches)
        targets, sample_weight = self._targets_and_weights(labels, weights, len(features))
        kwargs = {"classes": np.asarray([0, 1], dtype=np.uint8)} if not self._initialized else {}
        self.estimator.partial_fit(features, targets, sample_weight=sample_weight, **kwargs)
        self._initialized = True
        probabilities = self.estimator.predict_proba(features)[:, 1]
        return float(log_loss(targets, probabilities, sample_weight=sample_weight, labels=[0, 1]))

    def loss_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        if not self._initialized:
            raise RuntimeError("model must receive a training batch before validation")
        features = self._features(patches)
        targets, sample_weight = self._targets_and_weights(labels, weights, len(features))
        probabilities = self.estimator.predict_proba(features)[:, 1]
        return float(log_loss(targets, probabilities, sample_weight=sample_weight, labels=[0, 1]))

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        patch_array = np.asarray(patches)
        probabilities = self.estimator.predict_proba(self._features(patch_array))[:, 1]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("model produced non-finite probabilities")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities.reshape(patch_array.shape[0], *patch_array.shape[-2:])


@register_model("fault_raw_logistic")
def build_model(**kwargs) -> FaultRawLogistic:
    return FaultRawLogistic(**kwargs)
