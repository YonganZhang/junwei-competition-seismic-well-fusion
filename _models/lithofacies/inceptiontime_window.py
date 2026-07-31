"""Thin tsai InceptionTime 1.0.1 adapter for the GM09 P leaderboard."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import (
    require_dependency,
    standard_capabilities,
    validate_shapes,
    validate_task,
)
from _models.lithofacies.p5_torch_common import seismic_as_channels, validate_multimodal_tensors


model_id = "inceptiontime_window"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="torch", dependency_group="tabular-cpu")


class TsaiInceptionTimeAdapter(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        validate_multimodal_tensors(well_log_seq, seismic_patch)
        return self.backbone(torch.cat((well_log_seq, seismic_as_channels(seismic_patch)), dim=1))


def build_model(task_spec: TaskSpec, **config: Any) -> TsaiInceptionTimeAdapter:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    nf = int(values.pop("nf", 8))
    kernel_size = int(values.pop("kernel_size", 31))
    values.pop("hidden_size", None)
    if values:
        raise TypeError(f"unexpected inceptiontime_window config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    require_dependency(model_id, "tsai")
    module = require_dependency(model_id, "tsai.models.InceptionTime")
    return TsaiInceptionTimeAdapter(
        module.InceptionTime(
            c_in=well_shape[0] + seismic_shape[0] * seismic_shape[1],
            c_out=num_classes,
            seq_len=well_shape[-1],
            nf=nf,
            ks=kernel_size,
        )
    )


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "nf": trial.suggest_categorical("nf", [8, 16, 32]),
        "kernel_size": trial.suggest_categorical("kernel_size", [15, 23, 31]),
    }
