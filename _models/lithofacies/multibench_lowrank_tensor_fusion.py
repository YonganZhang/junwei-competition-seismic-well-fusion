"""Low-rank tensor-fusion adapter inspired by the source-locked MultiBench implementation."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import standard_capabilities, validate_shapes, validate_task
from _models.lithofacies.p5_torch_common import TCNEncoder1D, seismic_as_channels, validate_multimodal_tensors


model_id = "multibench_lowrank_tensor_fusion"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(lane="P", backend="torch", dependency_group="torch-common")


class LowRankTensorFusion(nn.Module):
    def __init__(self, input_dims: tuple[int, int], output_dim: int, rank: int) -> None:
        super().__init__()
        self.log_factors = nn.Parameter(torch.empty(rank, input_dims[0] + 1, output_dim))
        self.seismic_factors = nn.Parameter(torch.empty(rank, input_dims[1] + 1, output_dim))
        self.rank_weights = nn.Parameter(torch.ones(rank, 1))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_normal_(self.log_factors)
        nn.init.xavier_normal_(self.seismic_factors)

    def forward(self, logs: torch.Tensor, seismic: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((len(logs), 1), dtype=logs.dtype, device=logs.device)
        augmented_logs = torch.cat((ones, logs), dim=1)
        augmented_seismic = torch.cat((ones, seismic), dim=1)
        log_projection = torch.einsum("bd,rdo->rbo", augmented_logs, self.log_factors)
        seismic_projection = torch.einsum("bd,rdo->rbo", augmented_seismic, self.seismic_factors)
        fused = (log_projection * seismic_projection) * self.rank_weights.unsqueeze(1)
        return fused.sum(dim=0) + self.bias


class LowRankFusionClassifier(nn.Module):
    def __init__(self, *, num_classes: int, hidden_size: int = 16, fusion_size: int = 16, rank: int = 2) -> None:
        super().__init__()
        self.log_encoder = TCNEncoder1D(26, hidden_size=hidden_size, dilations=(1, 2))
        self.seismic_encoder = TCNEncoder1D(9, hidden_size=hidden_size, dilations=(1, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fusion = LowRankTensorFusion((hidden_size, hidden_size), fusion_size, rank)
        self.classifier = nn.Linear(fusion_size, num_classes)

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        validate_multimodal_tensors(well_log_seq, seismic_patch)
        logs = self.pool(self.log_encoder(well_log_seq)).squeeze(-1)
        seismic = self.pool(self.seismic_encoder(seismic_as_channels(seismic_patch))).squeeze(-1)
        return self.classifier(torch.relu(self.fusion(logs, seismic)))


def build_model(task_spec: TaskSpec, **config: Any) -> LowRankFusionClassifier:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    hidden_size = int(values.pop("hidden_size", 16))
    fusion_size = int(values.pop("fusion_size", 16))
    rank = int(values.pop("rank", 2))
    if values:
        raise TypeError(f"unexpected multibench_lowrank_tensor_fusion config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return LowRankFusionClassifier(
        num_classes=num_classes, hidden_size=hidden_size, fusion_size=fusion_size, rank=rank
    )


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32]),
        "rank": trial.suggest_categorical("rank", [2, 4]),
    }
