"""Canonical Torch 1x1-convolution baseline for facies segmentation."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec


model_id = "facies_linear_pixel"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_section"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class FaciesLinearPixel(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(f"expected facies input [B,1,H,W], got {tuple(inputs.shape)}")
        return self.classifier(inputs)


def build_model(task_spec: TaskSpec, **config: Any) -> FaciesLinearPixel:
    if task_spec.task_type != "multiclass":
        raise ValueError("facies_linear_pixel requires multiclass TaskSpec")
    num_classes = int(config.pop("num_classes", task_spec.metadata.get("num_classes", 0)))
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    return FaciesLinearPixel(num_classes)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del trial, task_spec
    return {}
