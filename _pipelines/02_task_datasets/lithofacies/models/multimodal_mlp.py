"""Small registered MLP baseline for real log sequences and seismic patches."""
from __future__ import annotations

import torch
from torch import nn

from _code.ml_framework.model_registry import register_model


class MultimodalMLP(nn.Module):
    """Two shallow encoders followed by a compact fusion classifier."""

    def __init__(
        self,
        *,
        num_classes: int,
        well_log_shape: tuple[int, int],
        seismic_shape: tuple[int, int, int],
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        log_size = int(torch.tensor(well_log_shape).prod().item())
        seismic_size = int(torch.tensor(seismic_shape).prod().item())
        self.log_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(log_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.seismic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seismic_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        log_features = self.log_encoder(well_log_seq)
        seismic_features = self.seismic_encoder(seismic_patch)
        return self.classifier(torch.cat((log_features, seismic_features), dim=1))


@register_model("multimodal_mlp")
def build_model(
    *,
    num_classes: int,
    well_log_shape: tuple[int, int],
    seismic_shape: tuple[int, int, int],
    hidden_size: int = 64,
    **_: object,
) -> nn.Module:
    return MultimodalMLP(
        num_classes=num_classes,
        well_log_shape=well_log_shape,
        seismic_shape=seismic_shape,
        hidden_size=hidden_size,
    )
