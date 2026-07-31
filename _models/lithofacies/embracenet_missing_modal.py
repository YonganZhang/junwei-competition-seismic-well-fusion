"""EmbraceNet-style missing-modality fusion over real GM09 inputs."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import standard_capabilities, validate_shapes, validate_task
from _models.lithofacies.p5_torch_common import TCNEncoder1D, seismic_as_channels, validate_multimodal_tensors


model_id = "embracenet_missing_modal"


def capabilities() -> dict[str, Any]:
    return standard_capabilities(
        lane="P", backend="torch", dependency_group="torch-common",
        supports_modality_availability=True,
    )


class EmbraceNetClassifier(nn.Module):
    def __init__(self, *, num_classes: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.log_encoder = TCNEncoder1D(26, hidden_size=hidden_size, dilations=(1, 2))
        self.seismic_encoder = TCNEncoder1D(9, hidden_size=hidden_size, dilations=(1, 2))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        validate_multimodal_tensors(well_log_seq, seismic_patch)
        logs = self.pool(self.log_encoder(well_log_seq)).squeeze(-1)
        seismic = self.pool(self.seismic_encoder(seismic_as_channels(seismic_patch))).squeeze(-1)
        modalities = torch.stack((logs, seismic), dim=1)
        if modality_mask is None:
            availability = torch.ones(
                (len(logs), 2), dtype=logs.dtype, device=logs.device
            )
        else:
            availability = modality_mask.to(device=logs.device, dtype=logs.dtype)
            if availability.shape != (len(logs), 2) or not bool((availability.sum(dim=1) > 0).all()):
                raise ValueError("modality_mask must be [B,2] with at least one available modality")
        probabilities = availability / availability.sum(dim=1, keepdim=True)
        if self.training:
            selected = torch.multinomial(probabilities, logs.shape[1], replacement=True)
            gathered = torch.gather(
                modalities.permute(0, 2, 1), 2, selected.unsqueeze(-1)
            ).squeeze(-1)
            embraced = gathered
        else:
            embraced = (modalities * probabilities.unsqueeze(-1)).sum(dim=1)
        return self.classifier(embraced)


def build_model(task_spec: TaskSpec, **config: Any) -> EmbraceNetClassifier:
    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    hidden_size = int(values.pop("hidden_size", 16))
    if values:
        raise TypeError(f"unexpected embracenet_missing_modal config: {sorted(values)}")
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    return EmbraceNetClassifier(num_classes=num_classes, hidden_size=hidden_size)


def suggest_hparams(trial: Any, task_spec: TaskSpec) -> Mapping[str, Any]:
    del task_spec
    return {"hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64])}
