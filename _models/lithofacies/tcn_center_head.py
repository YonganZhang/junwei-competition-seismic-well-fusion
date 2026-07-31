"""Non-causal dual-encoder TCN adapter for centered GM09 windows."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import standard_capabilities, validate_shapes, validate_task
from _models.lithofacies.p5_torch_common import DualEncoderClassifier, TCNEncoder1D


model_id = "tcn_center_head"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="torch", dependency_group="torch-common")


def build_model(task_spec: TaskSpec, **config: Any) -> DualEncoderClassifier:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    hidden_size = int(values.pop("hidden_size", 32))
    dilations = tuple(int(value) for value in values.pop("dilations", (1, 2, 4)))
    if values:
        raise TypeError(f"unexpected tcn_center_head config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return DualEncoderClassifier(
        TCNEncoder1D(well_shape[0], hidden_size=hidden_size, dilations=dilations),
        TCNEncoder1D(seismic_shape[0] * seismic_shape[1], hidden_size=hidden_size, dilations=dilations),
        num_classes=num_classes,
    )


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64])}
