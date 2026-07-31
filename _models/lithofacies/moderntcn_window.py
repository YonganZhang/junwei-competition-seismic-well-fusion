"""Short-window ModernTCN-style adapter for the GM09 P leaderboard."""
from __future__ import annotations

from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import standard_capabilities, validate_shapes, validate_task
from _models.lithofacies.p5_torch_common import DualEncoderClassifier, ModernTCNEncoder1D


model_id = "moderntcn_window"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="torch", dependency_group="torch-common")


def build_model(task_spec: TaskSpec, **config: Any) -> DualEncoderClassifier:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    hidden_size = int(values.pop("hidden_size", 16))
    patch_size = int(values.pop("patch_size", 4))
    patch_stride = int(values.pop("patch_stride", 2))
    kernel_size = int(values.pop("kernel_size", 7))
    if values:
        raise TypeError(f"unexpected moderntcn_window config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    kwargs = {
        "hidden_size": hidden_size,
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "kernel_size": kernel_size,
    }
    return DualEncoderClassifier(
        ModernTCNEncoder1D(well_shape[0], **kwargs),
        ModernTCNEncoder1D(seismic_shape[0] * seismic_shape[1], **kwargs),
        num_classes=num_classes,
    )


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32]),
        "kernel_size": trial.suggest_categorical("kernel_size", [5, 7, 9]),
    }
