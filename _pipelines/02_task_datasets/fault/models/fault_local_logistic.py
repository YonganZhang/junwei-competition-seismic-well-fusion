"""Existing fault baseline packaged as a registered, swappable model adapter."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel, uniform_filter
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from ml_framework.model_registry import register_model


class FaultLocalLogistic:
    """Balanced logistic pixel classifier over local seismic features."""

    description = "balanced_logistic_pixel_classifier_with_local_convolutional_features"

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
        all_features: list[np.ndarray] = []
        for patch in np.asarray(patches, dtype=np.float32):
            if patch.ndim == 3 and patch.shape[0] == 1:
                array = patch[0]
            else:
                raise ValueError(f"expected [1,H,W] seismic patch, received {patch.shape}")
            # local_mean/std are additional convolutional model features; they
            # never replace the raw amplitude channel. Dataset preprocessing
            # explicitly uses denoise_identity, so no seismic input is smoothed.
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
        return np.concatenate(all_features)

    def train_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        x = self._features(patches)
        y = np.asarray(labels, dtype=np.uint8).ravel()
        sample_weight = np.asarray(weights, dtype=np.float32).ravel()
        kwargs = {"classes": np.asarray([0, 1], dtype=np.uint8)} if not self._initialized else {}
        self.estimator.partial_fit(x, y, sample_weight=sample_weight, **kwargs)
        self._initialized = True
        probabilities = self.estimator.predict_proba(x)[:, 1]
        return float(log_loss(y, probabilities, sample_weight=sample_weight, labels=[0, 1]))

    def loss_batch(self, patches: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
        if not self._initialized:
            raise RuntimeError("model must receive a training batch before validation")
        y = np.asarray(labels, dtype=np.uint8).ravel()
        sample_weight = np.asarray(weights, dtype=np.float32).ravel()
        probabilities = self.predict_batch(patches).ravel()
        return float(log_loss(y, probabilities, sample_weight=sample_weight, labels=[0, 1]))

    def predict_batch(self, patches: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        patch_array = np.asarray(patches)
        probabilities = self.estimator.predict_proba(self._features(patch_array))[:, 1]
        return probabilities.reshape(patch_array.shape[0], *patch_array.shape[-2:])


@register_model("fault_local_logistic")
def build_model(**kwargs) -> FaultLocalLogistic:
    return FaultLocalLogistic(**kwargs)
