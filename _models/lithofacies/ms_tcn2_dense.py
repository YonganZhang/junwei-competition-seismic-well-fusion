"""Compact MS-TCN++-style dense adapter for the separate GM09 S lane."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import standard_capabilities, validate_shapes, validate_task
from _models.lithofacies.p5_torch_common import seismic_as_channels, validate_multimodal_tensors


model_id = "ms_tcn2_dense"


def capabilities() -> dict[str, Any]:
    result = standard_capabilities(lane="S", backend="torch", dependency_group="torch-common")
    result["output_layout"] = "B,C,L"
    result["requires_real_md_order"] = True
    return result


class DualDilatedLayer(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.forward_dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.reverse_dilated = nn.Conv1d(channels, channels, 3, padding=2 * dilation, dilation=2 * dilation)
        self.fuse = nn.Conv1d(channels * 2, channels, 1)
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.fuse(torch.cat((self.forward_dilated(values), self.reverse_dilated(values)), dim=1))
        return values + self.activation(update)


class PredictionStage(nn.Module):
    def __init__(self, in_channels: int, feature_maps: int, num_classes: int, layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, feature_maps, 1)
        self.layers = nn.Sequential(
            *(DualDilatedLayer(feature_maps, 2**index) for index in range(layers))
        )
        self.classifier = nn.Conv1d(feature_maps, num_classes, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.layers(self.input_projection(values)))


class MSTCN2Dense(nn.Module):
    def __init__(self, *, num_classes: int, feature_maps: int = 16, layers: int = 4, stages: int = 2) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("stages must be positive")
        self.first_stage = PredictionStage(35, feature_maps, num_classes, layers)
        self.refinement = nn.ModuleList(
            PredictionStage(num_classes, feature_maps, num_classes, layers) for _ in range(stages - 1)
        )

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        validate_multimodal_tensors(well_log_seq, seismic_patch)
        logits = self.first_stage(torch.cat((well_log_seq, seismic_as_channels(seismic_patch)), dim=1))
        for stage in self.refinement:
            logits = stage(torch.softmax(logits, dim=1))
        return logits


def build_model(task_spec: TaskSpec, **config: Any) -> MSTCN2Dense:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    feature_maps = int(values.pop("feature_maps", values.pop("hidden_size", 16)))
    layers = int(values.pop("layers", 4))
    stages = int(values.pop("stages", 2))
    if values:
        raise TypeError(f"unexpected ms_tcn2_dense config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return MSTCN2Dense(
        num_classes=num_classes, feature_maps=feature_maps, layers=layers, stages=stages
    )


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "feature_maps": trial.suggest_categorical("feature_maps", [16, 32]),
        "stages": trial.suggest_int("stages", 2, 3),
    }
