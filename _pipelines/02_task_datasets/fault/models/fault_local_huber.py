"""Incremental modified-Huber fault model over simple local features."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel, uniform_filter
from sklearn.linear_model import SGDClassifier

from ml_framework.model_registry import register_model


class FaultLocalHuber:
    """Weighted modified-Huber pixel classifier over local seismic features."""

    description = "modified_huber_pixel_classifier_with_local_convolutional_features"

    def __init__(self, seed: int = 2693) -> None:
        self.estimator = SGDClassifier(
            loss="modified_huber",
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
        all_features: list[np.ndarray] = []
        for patch in np.asarray(patches, dtype=np.float32):
            if patch.ndim == 3 and patch.shape[0] == 1:
                array = patch[0]
            else:
                raise ValueError(f"expected [1,H,W] seismic patch, received {patch.shape}")
            if not np.isfinite(array).all():
                raise ValueError("seismic features must be finite")
            local_mean = uniform_filter(array, size=3, mode="nearest")
            local_sq_mean = uniform_filter(array * array, size=3, mode="nearest")
            local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
            grad_crossline = sobel(array, axis=0, mode="nearest") / 8.0
            grad_time = sobel(array, axis=1, mode="nearest") / 8.0
            all_features.append(
                np.column_stack(
                    (
                        array.ravel(),
                        local_mean.ravel(),
                        local_std.ravel(),
                        grad_crossline.ravel(),
                        grad_time.ravel(),
                    )
                ).astype(np.float32)
            )
        if not all_features:
            raise ValueError("expected at least one seismic patch")
        return np.concatenate(all_features)

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

    def _loss(self, features: np.ndarray, targets: np.ndarray, sample_weight: np.ndarray) -> float:
        scores = np.asarray(self.estimator.decision_function(features), dtype=np.float64)
        margins = (2.0 * targets.astype(np.float64) - 1.0) * scores
        losses = np.where(
            margins >= 1.0,
            0.0,
            np.where(margins >= -1.0, (1.0 - margins) ** 2, -4.0 * margins),
        )
        loss = float(np.average(losses, weights=sample_weight))
        if not np.isfinite(loss):
            raise RuntimeError("model produced a non-finite modified-Huber loss")
        return loss

    def train_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        features = self._features(patches)
        targets, sample_weight = self._targets_and_weights(labels, weights, len(features))
        kwargs = {"classes": np.asarray([0, 1], dtype=np.uint8)} if not self._initialized else {}
        self.estimator.partial_fit(features, targets, sample_weight=sample_weight, **kwargs)
        self._initialized = True
        return self._loss(features, targets, sample_weight)

    def loss_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        if not self._initialized:
            raise RuntimeError("model must receive a training batch before validation")
        features = self._features(patches)
        targets, sample_weight = self._targets_and_weights(labels, weights, len(features))
        return self._loss(features, targets, sample_weight)

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        patch_array = np.asarray(patches)
        probabilities = self.estimator.predict_proba(self._features(patch_array))[:, 1]
        if not np.isfinite(probabilities).all():
            raise RuntimeError("model produced non-finite probabilities")
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities.reshape(patch_array.shape[0], *patch_array.shape[-2:])


@register_model("fault_local_huber")
def build_model(**kwargs) -> FaultLocalHuber:
    return FaultLocalHuber(**kwargs)
