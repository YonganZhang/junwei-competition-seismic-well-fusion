"""Shallow fully convolutional network for facies segmentation."""
from __future__ import annotations

import torch
from torch import nn

from _code.ml_framework.model_registry import register_model


class FaciesTinyFCN(nn.Module):
    """Use two local feature layers followed by a per-pixel classifier."""

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
            raise ValueError(
                "facies_tiny_fcn expects input shaped [B, 1, H, W], "
                f"got {tuple(inputs.shape)}"
            )
        return self.network(inputs)


@register_model("facies_tiny_fcn")
def build_model(
    num_classes: int, hidden_channels: int = 8, **_: object
) -> nn.Module:
    """Build the registered shallow facies FCN."""
    return FaciesTinyFCN(
        num_classes=num_classes,
        hidden_channels=hidden_channels,
    )
