"""Small Torch building blocks used by source-locked P5 adapters."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def seismic_as_channels(seismic_patch: torch.Tensor) -> torch.Tensor:
    if seismic_patch.ndim != 4 or tuple(seismic_patch.shape[1:3]) != (3, 3):
        raise ValueError(f"seismic_patch must be [B,3,3,L], got {tuple(seismic_patch.shape)}")
    return seismic_patch.flatten(1, 2)


def validate_multimodal_tensors(well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> None:
    if well_log_seq.ndim != 3 or well_log_seq.shape[1] != 26:
        raise ValueError(f"well_log_seq must be [B,26,L], got {tuple(well_log_seq.shape)}")
    seismic = seismic_as_channels(seismic_patch)
    if well_log_seq.shape[0] != seismic.shape[0] or well_log_seq.shape[-1] != seismic.shape[-1]:
        raise ValueError("well-log and seismic tensors must align")
    if not torch.isfinite(well_log_seq).all() or not torch.isfinite(seismic_patch).all():
        raise ValueError("model input contains NaN/Inf")


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int, kernels: Sequence[int]) -> None:
        super().__init__()
        bottleneck_channels = max(branch_channels, min(32, in_channels))
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv1d(bottleneck_channels, branch_channels, kernel, padding=kernel // 2, bias=False)
            for kernel in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
        )
        self.norm = nn.BatchNorm1d(branch_channels * (len(kernels) + 1))
        self.activation = nn.ReLU()
        self.out_channels = branch_channels * (len(kernels) + 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        reduced = self.bottleneck(values)
        merged = torch.cat([*(branch(reduced) for branch in self.branches), self.pool_branch(values)], dim=1)
        return self.activation(self.norm(merged))


class InceptionEncoder1D(nn.Module):
    def __init__(self, in_channels: int, nf: int = 8, kernels: Sequence[int] = (31, 15, 7)) -> None:
        super().__init__()
        self.block1 = InceptionBlock1D(in_channels, nf, kernels)
        self.block2 = InceptionBlock1D(self.block1.out_channels, nf, kernels)
        self.out_channels = self.block2.out_channels

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(values))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class TCNEncoder1D(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int = 32, dilations: Sequence[int] = (1, 2, 4)) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, hidden_size, 1)
        self.blocks = nn.Sequential(
            *(ResidualTCNBlock(hidden_size, dilation=dilation) for dilation in dilations)
        )
        self.out_channels = hidden_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.input_projection(values))


class ModernTCNBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 7, expansion: int = 2) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2, groups=channels
        )
        self.norm = nn.BatchNorm1d(channels)
        self.feed_forward = nn.Sequential(
            nn.Conv1d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv1d(channels * expansion, channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.feed_forward(self.norm(self.depthwise(values)))


class ModernTCNEncoder1D(nn.Module):
    def __init__(
        self, in_channels: int, hidden_size: int = 16, patch_size: int = 4,
        patch_stride: int = 2, kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv1d(in_channels, hidden_size, patch_size, stride=patch_stride)
        self.block = ModernTCNBlock(hidden_size, kernel_size=kernel_size)
        self.out_channels = hidden_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(self.patch(values))


class DualEncoderClassifier(nn.Module):
    def __init__(self, log_encoder: nn.Module, seismic_encoder: nn.Module, *, num_classes: int) -> None:
        super().__init__()
        self.log_encoder = log_encoder
        self.seismic_encoder = seismic_encoder
        log_features = int(getattr(log_encoder, "out_channels"))
        seismic_features = int(getattr(seismic_encoder, "out_channels"))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(log_features + seismic_features, num_classes)

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        validate_multimodal_tensors(well_log_seq, seismic_patch)
        logs = self.pool(self.log_encoder(well_log_seq)).squeeze(-1)
        seismic = self.pool(self.seismic_encoder(seismic_as_channels(seismic_patch))).squeeze(-1)
        return self.classifier(torch.cat((logs, seismic), dim=1))
