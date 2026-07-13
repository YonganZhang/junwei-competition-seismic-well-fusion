"""Canonical small two-level U-Net facies baseline."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec


model_id = "small_unet"


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["multiclass"],
        "input_modalities": ["seismic_section"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
    }


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SmallUNet(nn.Module):
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
        self.classifier = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.decoder2(torch.cat((self.up2(bottleneck), enc2), dim=1))
        dec1 = self.decoder1(torch.cat((self.up1(dec2), enc1), dim=1))
        return self.classifier(dec1)


def build_model(task_spec: TaskSpec, **config: Any) -> SmallUNet:
    if task_spec.task_type != "multiclass":
        raise ValueError("small_unet requires multiclass TaskSpec")
    num_classes = int(config.pop("num_classes", task_spec.metadata.get("num_classes", 0)))
    base_channels = int(config.pop("base_channels", 8))
    if config:
        raise ValueError(f"unsupported small_unet config: {sorted(config)}")
    return SmallUNet(num_classes, base_channels)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"base_channels": trial.suggest_categorical("base_channels", [4, 8, 16])}
