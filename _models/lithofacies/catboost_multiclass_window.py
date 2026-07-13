"""Thin CatBoost MultiClass adapter for multimodal GM09 windows."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import (
    NUM_CLASSES,
    fixed_schema_logits,
    multimodal_numpy_features,
    probability_loss,
    require_dependency,
    standard_capabilities,
    validate_shapes,
    validate_task,
)


model_id = "catboost_multiclass_window"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="estimator", dependency_group="tabular-cpu")


class CatBoostWindowAdapter:
    def __init__(self, *, iterations: int = 8, depth: int = 3, seed: int = 2693) -> None:
        module = require_dependency(model_id, "catboost")
        self.estimator = module.CatBoostClassifier(
            iterations=int(iterations),
            depth=int(depth),
            learning_rate=0.2,
            loss_function="MultiClass",
            classes_count=NUM_CLASSES,
            random_seed=int(seed),
            thread_count=1,
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
        )

    def fit_stage1(
        self, well_log_seq: Any, seismic_patch: Any, labels: Any, *, class_counts: Any
    ) -> float:
        features = multimodal_numpy_features(well_log_seq, seismic_patch)
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        counts = np.asarray(class_counts, dtype=np.float64)
        supported = counts > 0
        weights = np.zeros(NUM_CLASSES, dtype=np.float64)
        weights[supported] = 1.0 / np.sqrt(counts[supported])
        self.estimator.fit(features, target, sample_weight=weights[target])
        probability = self._fixed_probabilities(features)
        return probability_loss(probability, target)

    def _fixed_probabilities(self, features: np.ndarray) -> np.ndarray:
        partial = np.asarray(self.estimator.predict_proba(features), dtype=np.float64)
        classes = np.asarray(self.estimator.classes_, dtype=np.int64)
        fixed = np.zeros((len(features), NUM_CLASSES), dtype=np.float64)
        fixed[:, classes] = partial
        row_sum = fixed.sum(axis=1, keepdims=True)
        return np.divide(fixed, row_sum, out=np.zeros_like(fixed), where=row_sum > 0)

    def predict_logits(self, well_log_seq: Any, seismic_patch: Any) -> np.ndarray:
        features = multimodal_numpy_features(well_log_seq, seismic_patch)
        raw = np.asarray(self.estimator.predict(features, prediction_type="RawFormulaVal"))
        return fixed_schema_logits(raw, self.estimator.classes_)


def build_model(task_spec: TaskSpec, **config: Any) -> CatBoostWindowAdapter:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    values.pop("hidden_size", None)
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return CatBoostWindowAdapter(**values)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "depth": trial.suggest_int("depth", 3, 7),
        "iterations": trial.suggest_int("iterations", 20, 200),
    }
