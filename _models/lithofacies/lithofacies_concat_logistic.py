"""Dependency-light concatenated multimodal lithofacies baseline."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import SGDClassifier

from _code.ml_framework.contracts import ModelOutput, TaskSpec


model_id = "lithofacies_concat_logistic"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["well_log_sequence", "seismic_patch"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class LithofaciesConcatLogistic:
    def __init__(self, task_spec: TaskSpec, *, num_classes: int = 9, alpha: float = 1e-4, seed: int = 2693) -> None:
        if task_spec.task_type != "multiclass" or num_classes < 2:
            raise ValueError("lithofacies_concat_logistic requires multiclass TaskSpec")
        self.task_spec = task_spec; self.num_classes = int(num_classes)
        self.estimator = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=1, tol=None,
                                       random_state=seed, average=True)
        self._initialized = False

    @staticmethod
    def _features(well_log_seq: np.ndarray, seismic_patch: np.ndarray) -> np.ndarray:
        logs = np.asarray(well_log_seq, dtype=float); seismic = np.asarray(seismic_patch, dtype=float)
        if logs.ndim < 2 or seismic.ndim < 2 or len(logs) != len(seismic):
            raise ValueError("modalities must be batched and aligned")
        features = np.concatenate((logs.reshape(len(logs), -1), seismic.reshape(len(seismic), -1)), axis=1)
        if not np.isfinite(features).all():
            raise ValueError("adapter must impute/scale fold-train features before the model")
        return features

    def train_batch(
        self, well_log_seq: np.ndarray, seismic_patch: np.ndarray, labels: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> float:
        features = self._features(well_log_seq, seismic_patch); target = np.asarray(labels, dtype=int).reshape(-1)
        valid = np.ones_like(target, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool).reshape(-1)
        valid &= (target >= 0) & (target < self.num_classes)
        if len(target) != len(features) or not valid.any():
            raise ValueError("labels/mask must align and contain valid samples")
        kwargs = {"classes": np.arange(self.num_classes)} if not self._initialized else {}
        self.estimator.partial_fit(features[valid], target[valid], **kwargs); self._initialized = True
        probability = self.estimator.predict_proba(features[valid])
        chosen = np.maximum(probability[np.arange(valid.sum()), target[valid]], 1e-12)
        return float(-np.log(chosen).mean())

    def predict_output(self, well_log_seq: np.ndarray, seismic_patch: np.ndarray) -> ModelOutput:
        if not self._initialized:
            raise RuntimeError("model has not been fitted")
        features = self._features(well_log_seq, seismic_patch)
        logits = np.asarray(self.estimator.decision_function(features), dtype=float)
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits); probability /= probability.sum(axis=1, keepdims=True)
        target = self.task_spec.targets[0]
        return ModelOutput(raw={target: logits}, transformed={target: probability})


def build_model(task_spec: TaskSpec, **config: Any) -> LithofaciesConcatLogistic:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("num_classes", 9)))
    return LithofaciesConcatLogistic(task_spec, num_classes=num_classes, **values)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True)}
