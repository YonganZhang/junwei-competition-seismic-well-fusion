"""Canonical compact two-encoder MLP for lithofacies classification."""
from __future__ import annotations

from math import prod
from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec


model_id = "multimodal_mlp"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["well_log_sequence", "seismic_patch"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class MultimodalMLP(nn.Module):
    def __init__(
        self, *, num_classes: int, well_log_shape: tuple[int, int],
        seismic_shape: tuple[int, int, int], hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.log_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(prod(well_log_shape), hidden_size), nn.ReLU(), nn.Dropout(0.15)
        )
        self.seismic_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(prod(seismic_shape), hidden_size), nn.ReLU(), nn.Dropout(0.15)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        logs = self.log_encoder(well_log_seq)
        seismic = self.seismic_encoder(seismic_patch)
        return self.classifier(torch.cat((logs, seismic), dim=1))


def build_model(task_spec: TaskSpec, **config: Any) -> MultimodalMLP:
    if task_spec.task_type != "multiclass":
        raise ValueError("multimodal_mlp requires multiclass TaskSpec")
    num_classes = int(config.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    return MultimodalMLP(num_classes=num_classes, **config)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128])}
