"""Dependency-light per-pixel logistic facies baseline."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import SGDClassifier

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "facies_pixel_logistic"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_section"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class FaciesPixelLogistic:
    def __init__(self, task_spec: TaskSpec, *, num_classes: int, alpha: float = 1e-4, seed: int = 2693) -> None:
        if task_spec.task_type != "multiclass" or num_classes < 2:
            raise ValueError("facies_pixel_logistic requires multiclass TaskSpec and num_classes>=2")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.task_spec = task_spec; self.num_classes = int(num_classes)
        self.estimator = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=1, tol=None,
                                       random_state=seed, average=True)
        self._initialized = False

    @staticmethod
    def _features(sections: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        array = np.asarray(sections, dtype=np.float32)
        if array.ndim != 4 or array.shape[1] != 1 or not np.isfinite(array).all():
            raise ValueError(f"expected finite [B,1,H,W], received {array.shape}")
        batch, _, height, width = array.shape
        # Raw amplitude plus deterministic local coordinates; fold-level input
        # normalization remains the adapter's responsibility.
        yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing="ij")
        coords = np.column_stack((np.tile(yy.ravel(), batch), np.tile(xx.ravel(), batch)))
        return np.column_stack((array[:, 0].reshape(-1), coords)), (batch, height, width)

    def train_batch(self, sections: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray | None = None) -> float:
        features, shape = self._features(sections)
        target = np.asarray(labels, dtype=int).reshape(-1)
        valid = np.ones_like(target, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
        valid &= (target >= 0) & (target < self.num_classes)
        if target.size != len(features) or not valid.any():
            raise ValueError("labels/mask must align and contain valid pixels")
        kwargs = {"classes": np.arange(self.num_classes)} if not self._initialized else {}
        self.estimator.partial_fit(features[valid], target[valid], **kwargs)
        self._initialized = True
        probability = self.estimator.predict_proba(features[valid])
        chosen = np.maximum(probability[np.arange(valid.sum()), target[valid]], 1e-12)
        return float(-np.log(chosen).mean())

    def predict_output(self, sections: np.ndarray) -> ModelOutput:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        features, (batch, height, width) = self._features(sections)
        logits = np.asarray(self.estimator.decision_function(features), dtype=float)
        if logits.ndim == 1:
            logits = np.column_stack((-logits, logits))
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
        logits_4d = logits.reshape(batch, height, width, self.num_classes).transpose(0, 3, 1, 2)
        prob_4d = probability.reshape(batch, height, width, self.num_classes).transpose(0, 3, 1, 2)
        target = self.task_spec.targets[0]
        return ModelOutput(raw={target: logits_4d}, transformed={target: prob_4d})


def build_model(task_spec: TaskSpec, **config: Any) -> FaciesPixelLogistic:
    values = dict(config)
    num_classes = values.pop("num_classes", task_spec.metadata.get("num_classes"))
    if num_classes is None:
        raise ValueError("num_classes must be supplied in TaskSpec.metadata or config")
    return FaciesPixelLogistic(task_spec, num_classes=int(num_classes), **values)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True)}
