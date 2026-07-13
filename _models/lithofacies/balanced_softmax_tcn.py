"""TCN backbone with fold-train-only Balanced Softmax for GM09 long tails."""
from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as functional

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import standard_capabilities, validate_shapes, validate_task
from _models.lithofacies.p5_torch_common import DualEncoderClassifier, TCNEncoder1D


model_id = "balanced_softmax_tcn"


def capabilities() -> dict[str, Any]:
    result = standard_capabilities(lane="P", backend="torch", dependency_group="torch-common")
    result["training_mechanism"] = "balanced_softmax_fold_train_counts"
    return result


def build_model(task_spec: TaskSpec, **config: Any) -> DualEncoderClassifier:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    hidden_size = int(values.pop("hidden_size", 32))
    dilations = tuple(int(value) for value in values.pop("dilations", (1, 2, 4)))
    if values:
        raise TypeError(f"unexpected balanced_softmax_tcn config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return DualEncoderClassifier(
        TCNEncoder1D(well_shape[0], hidden_size=hidden_size, dilations=dilations),
        TCNEncoder1D(seismic_shape[0] * seismic_shape[1], hidden_size=hidden_size, dilations=dilations),
        num_classes=num_classes,
    )


def stage1_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, class_counts: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    del class_weights
    counts = class_counts.to(device=logits.device, dtype=logits.dtype)
    if counts.shape != (logits.shape[1],):
        raise ValueError("Balanced Softmax class counts must match the nine-logit schema")
    supported = counts > 0
    if not bool(supported[targets].all()):
        raise ValueError("a fold-train target has zero fold-train support")
    adjusted = logits + torch.log(counts.clamp_min(1)).unsqueeze(0)
    floor = torch.full_like(adjusted, -torch.finfo(adjusted.dtype).max / 4)
    adjusted = torch.where(supported.unsqueeze(0), adjusted, floor)
    return functional.cross_entropy(adjusted, targets)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64])}
