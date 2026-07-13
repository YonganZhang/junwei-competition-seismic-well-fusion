"""Independent per-pixel linear classifier for facies segmentation."""
from __future__ import annotations

import torch
from torch import nn

from _code.ml_framework.model_registry import register_model


class FaciesLinearPixel(nn.Module):
    """Classify each seismic pixel independently with a 1x1 convolution."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(1, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                "facies_linear_pixel expects input shaped [B, 1, H, W], "
                f"got {tuple(inputs.shape)}"
            )
        return self.classifier(inputs)


@register_model("facies_linear_pixel")
def build_model(num_classes: int, **_: object) -> nn.Module:
    """Build the registered per-pixel linear facies classifier."""
    return FaciesLinearPixel(num_classes=num_classes)
