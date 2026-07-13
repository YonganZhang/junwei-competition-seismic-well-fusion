"""Shallow late-fusion alternative for lithofacies classification."""
from __future__ import annotations

from math import prod

import torch
from torch import nn

from _code.ml_framework.model_registry import register_model


class LithofaciesLateFusion(nn.Module):
    """Encode logs and seismic independently before concatenated classification."""

    def __init__(
        self,
        *,
        num_classes: int,
        well_log_shape: tuple[int, int],
        seismic_shape: tuple[int, int, int],
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.well_log_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(well_log_shape), hidden_size),
            nn.ReLU(),
        )
        self.seismic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prod(seismic_shape), hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(
        self,
        well_log_seq: torch.Tensor,
        seismic_patch: torch.Tensor,
    ) -> torch.Tensor:
        well_log_features = self.well_log_encoder(well_log_seq)
        seismic_features = self.seismic_encoder(seismic_patch)
        return self.classifier(torch.cat((well_log_features, seismic_features), dim=1))


@register_model("lithofacies_late_fusion")
def build_model(
    num_classes: int,
    well_log_shape: tuple[int, int],
    seismic_shape: tuple[int, int, int],
    **kwargs: object,
) -> nn.Module:
    return LithofaciesLateFusion(
        num_classes=num_classes,
        well_log_shape=well_log_shape,
        seismic_shape=seismic_shape,
        hidden_size=int(kwargs.get("hidden_size", 32)),
    )
