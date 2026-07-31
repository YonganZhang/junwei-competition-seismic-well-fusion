"""sktime MiniRocket plus Ridge thin adapter over both GM09 modalities."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import (
    NUM_CLASSES,
    fixed_schema_logits,
    require_dependency,
    standard_capabilities,
    validate_shapes,
    validate_task,
)


model_id = "minirocket_ridge_window"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="estimator", dependency_group="tabular-cpu")


def _multimodal_channels(well_log_seq: Any, seismic_patch: Any) -> np.ndarray:
    logs = np.asarray(well_log_seq, dtype=np.float32)
    seismic = np.asarray(seismic_patch, dtype=np.float32)
    if logs.ndim != 3 or logs.shape[1] != 26:
        raise ValueError(f"well_log_seq must be [B,26,L], got {logs.shape}")
    if seismic.ndim != 4 or tuple(seismic.shape[1:3]) != (3, 3):
        raise ValueError(f"seismic_patch must be [B,3,3,L], got {seismic.shape}")
    if logs.shape[0] != seismic.shape[0] or logs.shape[-1] != seismic.shape[-1]:
        raise ValueError("well-log and seismic inputs must align")
    values = np.concatenate((logs, seismic.reshape(len(seismic), 9, seismic.shape[-1])), axis=1)
    if not np.isfinite(values).all():
        raise ValueError("MiniRocket input contains NaN/Inf")
    return values


class MiniRocketRidgeAdapter:
    def __init__(self, *, num_kernels: int = 1000, seed: int = 2693, alpha: float = 1.0) -> None:
        require_dependency(model_id, "sktime")
        sklearn_linear = require_dependency(model_id, "sklearn.linear_model")
        rocket_module = require_dependency(model_id, "sktime.transformations.panel.rocket")
        self.transformer = rocket_module.MiniRocketMultivariate(
            num_kernels=int(num_kernels), random_state=int(seed), n_jobs=1
        )
        self.classifier = sklearn_linear.RidgeClassifier(alpha=float(alpha))

    def fit_stage1(
        self, well_log_seq: Any, seismic_patch: Any, labels: Any, *, class_counts: Any
    ) -> float:
        del class_counts
        values = _multimodal_channels(well_log_seq, seismic_patch)
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        features = np.asarray(self.transformer.fit_transform(values), dtype=np.float32)
        self.classifier.fit(features, target)
        logits = fixed_schema_logits(self.classifier.decision_function(features), self.classifier.classes_)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probability = np.exp(shifted)
        probability /= probability.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(probability[np.arange(len(target)), target], 1e-12, 1.0)).mean())

    def predict_logits(self, well_log_seq: Any, seismic_patch: Any) -> np.ndarray:
        values = _multimodal_channels(well_log_seq, seismic_patch)
        features = np.asarray(self.transformer.transform(values), dtype=np.float32)
        return fixed_schema_logits(self.classifier.decision_function(features), self.classifier.classes_)


def build_model(task_spec: TaskSpec, **config: Any) -> MiniRocketRidgeAdapter:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    values.pop("hidden_size", None)
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return MiniRocketRidgeAdapter(**values)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
