"""Thin XGBoost ``multi:softprob`` adapter for multimodal GM09 windows."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import (
    NUM_CLASSES,
    multimodal_numpy_features,
    probability_loss,
    require_dependency,
    standard_capabilities,
    validate_shapes,
    validate_task,
)


model_id = "xgboost_multisoftprob_window"
DEFAULT_BASELINE_ROUNDS = 60
DEFAULT_BASELINE_MAX_DEPTH = 3
DEFAULT_BASELINE_ETA = 0.1


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="estimator", dependency_group="tabular-cpu")


class XGBoostWindowAdapter:
    def __init__(
        self,
        *,
        rounds: int = DEFAULT_BASELINE_ROUNDS,
        max_depth: int = DEFAULT_BASELINE_MAX_DEPTH,
        eta: float = DEFAULT_BASELINE_ETA,
        seed: int = 2693,
    ) -> None:
        require_dependency(model_id, "xgboost")
        self.rounds = int(rounds)
        self.max_depth = int(max_depth)
        self.eta = float(eta)
        self.seed = int(seed)
        self.booster: Any | None = None

    def fit_stage1(
        self, well_log_seq: Any, seismic_patch: Any, labels: Any, *, class_counts: Any
    ) -> float:
        features = multimodal_numpy_features(well_log_seq, seismic_patch)
        target = np.asarray(labels, dtype=np.int64).reshape(-1)
        counts = np.asarray(class_counts, dtype=np.float64)
        if len(features) != len(target) or counts.shape != (NUM_CLASSES,):
            raise ValueError("XGBoost Stage-1 labels/class counts do not match the fixed schema")
        supported = counts > 0
        class_weights = np.zeros(NUM_CLASSES, dtype=np.float64)
        class_weights[supported] = 1.0 / np.sqrt(counts[supported])
        weights = class_weights[target]
        xgboost = require_dependency(model_id, "xgboost")
        matrix = xgboost.DMatrix(features, label=target, weight=weights)
        self.booster = xgboost.train(
            {
                "objective": "multi:softprob",
                "num_class": NUM_CLASSES,
                "max_depth": self.max_depth,
                "eta": self.eta,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "tree_method": "hist",
                "seed": self.seed,
                "nthread": 1,
                "verbosity": 0,
            },
            matrix,
            num_boost_round=self.rounds,
        )
        return probability_loss(self.booster.predict(matrix), target)

    def predict_logits(self, well_log_seq: Any, seismic_patch: Any) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("XGBoost adapter has not been fitted")
        xgboost = require_dependency(model_id, "xgboost")
        features = multimodal_numpy_features(well_log_seq, seismic_patch)
        probabilities = np.asarray(
            self.booster.predict(xgboost.DMatrix(features)), dtype=np.float64
        )
        if probabilities.shape != (len(features), NUM_CLASSES):
            raise ValueError(f"XGBoost probabilities must be [B,9], got {probabilities.shape}")
        return np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32)


def build_model(task_spec: TaskSpec, **config: Any) -> XGBoostWindowAdapter:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    values.pop("hidden_size", None)
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return XGBoostWindowAdapter(**values)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "rounds": trial.suggest_int("rounds", 20, 200),
    }
