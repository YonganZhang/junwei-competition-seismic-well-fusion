"""Small two-level U-Net facies baseline."""
from __future__ import annotations

import torch
from torch import nn

from _code.ml_framework.model_registry import register_model


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SmallUNet(nn.Module):
    """Same simple architecture used by the original facies baseline."""

    def __init__(self, num_classes: int, base_channels: int = 8) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(1, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.decoder2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.decoder1 = DoubleConv(base_channels * 2, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.decoder2(torch.cat((self.up2(bottleneck), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.classifier(dec1)


@register_model("small_unet")
def build_model(*, num_classes: int, base_channels: int = 8, **_: object) -> nn.Module:
    """Build the registered facies baseline model."""
    return SmallUNet(num_classes=num_classes, base_channels=base_channels)
