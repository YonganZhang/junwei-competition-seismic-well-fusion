"""Linear lithofacies classifier over concatenated log and seismic inputs."""
from __future__ import annotations

from math import prod

import torch
from torch import nn

from _code.ml_framework.model_registry import register_model


class LithofaciesConcatLinear(nn.Module):
    """Flatten both modalities, concatenate them, and classify linearly."""

    def __init__(
        self,
        *,
        num_classes: int,
        well_log_shape: tuple[int, int],
        seismic_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        input_size = prod(well_log_shape) + prod(seismic_shape)
        self.classifier = nn.Linear(input_size, num_classes)

    def forward(
        self,
        well_log_seq: torch.Tensor,
        seismic_patch: torch.Tensor,
    ) -> torch.Tensor:
        well_log_flat = torch.flatten(well_log_seq, start_dim=1)
        seismic_flat = torch.flatten(seismic_patch, start_dim=1)
        return self.classifier(torch.cat((well_log_flat, seismic_flat), dim=1))


@register_model("lithofacies_concat_linear")
def build_model(
    num_classes: int,
    well_log_shape: tuple[int, int],
    seismic_shape: tuple[int, int, int],
    **_: object,
) -> nn.Module:
    return LithofaciesConcatLinear(
        num_classes=num_classes,
        well_log_shape=well_log_shape,
        seismic_shape=seismic_shape,
    )
