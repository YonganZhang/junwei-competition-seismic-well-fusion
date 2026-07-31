"""Canonical shallow fully convolutional facies baseline."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec


model_id = "facies_tiny_fcn"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_section"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class FaciesTinyFCN(nn.Module):
    def __init__(self, num_classes: int, hidden_channels: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"expected facies input [B,1,H,W], got {tuple(inputs.shape)}")
        return self.network(inputs)


def build_model(task_spec: TaskSpec, **config: Any) -> FaciesTinyFCN:
    if task_spec.task_type != "multiclass":
        raise ValueError("facies_tiny_fcn requires multiclass TaskSpec")
    num_classes = int(config.pop("num_classes", task_spec.metadata.get("num_classes", 0)))
    hidden_channels = int(config.pop("hidden_channels", 8))
    config.pop("base_channels", None)  # accepted by the historical generic factory
    if config:
        raise ValueError(f"unsupported facies_tiny_fcn config: {sorted(config)}")
    return FaciesTinyFCN(num_classes, hidden_channels)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"hidden_channels": trial.suggest_categorical("hidden_channels", [4, 8, 16])}
