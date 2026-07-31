"""Canonical Torch linear classifier over concatenated lithofacies inputs."""
from __future__ import annotations

from math import prod
from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec


model_id = "lithofacies_concat_linear"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["well_log_sequence", "seismic_patch"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class LithofaciesConcatLinear(nn.Module):
    def __init__(self, *, num_classes: int, well_log_shape: tuple[int, int], seismic_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.classifier = nn.Linear(prod(well_log_shape) + prod(seismic_shape), num_classes)

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        logs = torch.flatten(well_log_seq, start_dim=1)
        seismic = torch.flatten(seismic_patch, start_dim=1)
        return self.classifier(torch.cat((logs, seismic), dim=1))


def build_model(task_spec: TaskSpec, **config: Any) -> LithofaciesConcatLinear:
    if task_spec.task_type != "multiclass":
        raise ValueError("lithofacies_concat_linear requires multiclass TaskSpec")
    num_classes = int(config.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    config.pop("hidden_size", None)
    return LithofaciesConcatLinear(num_classes=num_classes, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del trial, task_spec
    return {}
