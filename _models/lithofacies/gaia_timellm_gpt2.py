"""GM09 lithofacies module with Gaia constraints and Time-LLM reprogramming."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from _code.foundation import CompactTimeLLMReprogrammer, enable_gpt2_lora
from _code.ml_framework.contracts import TaskSpec
from _models.lithofacies.p5_adapter_common import (
    standard_capabilities,
    validate_shapes,
    validate_task,
)


model_id = "gaia_timellm_gpt2"

DOMAIN_PROMPT = (
    "Petroleum GM09 genetic facies classification with nine fixed classes: "
    "marsh, mouth bar, offshore, lower shoreface, upper shoreface, tidal bar, "
    "tidal channel, tidal flat muddy, tidal flat sandy. Use observed well-log "
    "values and masks plus the local seismic patch. Never infer a class from a "
    "target-derived curve and preserve the fixed nine-class output schema."
)


def capabilities() -> dict[str, Any]:
    value = standard_capabilities(lane="P", backend="torch", dependency_group="torch-common")
    value.update(
        {
            "foundation_strategy": "frozen GPT-2 vocabulary reprogramming with optional LoRA",
            "domain_guidance": "Gaia-style GM09 ontology and leakage constraints",
            "full_parameter_finetuning": False,
        }
    )
    return value


class GaiaTimeLLMLithofacies(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        prompt_token_ids: torch.Tensor,
        lora_rank: int = 0,
        lora_last_blocks: int = 2,
        **model_config: Any,
    ) -> None:
        super().__init__()
        self.reprogrammer = CompactTimeLLMReprogrammer(
            backbone,
            input_channels=35,
            output_size=9,
            prompt_token_ids=prompt_token_ids,
            sequence_length=33,
            **model_config,
        )
        self.lora_modules = (
            enable_gpt2_lora(
                self.reprogrammer.backbone,
                last_blocks=lora_last_blocks,
                rank=lora_rank,
                alpha=float(2 * lora_rank),
            )
            if lora_rank > 0
            else []
        )

    def forward(self, well_log_seq: torch.Tensor, seismic_patch: torch.Tensor) -> torch.Tensor:
        if well_log_seq.ndim != 3 or tuple(well_log_seq.shape[1:]) != (26, 33):
            raise ValueError(f"well_log_seq must be [B,26,33], got {tuple(well_log_seq.shape)}")
        if seismic_patch.ndim != 4 or tuple(seismic_patch.shape[1:]) != (3, 3, 33):
            raise ValueError(f"seismic_patch must be [B,3,3,33], got {tuple(seismic_patch.shape)}")
        logs = well_log_seq.transpose(1, 2)
        seismic = seismic_patch.permute(0, 3, 1, 2).flatten(2)
        return self.reprogrammer(torch.cat((logs, seismic), dim=-1))


def build_model(task_spec: TaskSpec, **config: Any) -> GaiaTimeLLMLithofacies:
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    values = dict(config)
    num_classes = int(values.pop("num_classes", task_spec.metadata.get("class_count", 0)))
    well_shape = tuple(values.pop("well_log_shape"))
    seismic_shape = tuple(values.pop("seismic_shape"))
    backbone_path = Path(values.pop("backbone_path"))
    random_backbone = bool(values.pop("random_backbone", False))
    lora_rank = int(values.pop("lora_rank", 0))
    lora_last_blocks = int(values.pop("lora_last_blocks", 2))
    validate_task(task_spec, num_classes=num_classes)
    validate_shapes(well_shape, seismic_shape)
    if well_shape[-1] != 33:
        raise ValueError("the P6 lithofacies foundation adapter requires the frozen 33-point window")
    if not backbone_path.is_dir():
        raise FileNotFoundError(f"local backbone snapshot is missing: {backbone_path}")
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    prompt_ids = tokenizer(
        DOMAIN_PROMPT, add_special_tokens=True, truncation=True, max_length=72
    )["input_ids"]
    if random_backbone:
        backbone = AutoModel.from_config(
            AutoConfig.from_pretrained(backbone_path, local_files_only=True)
        )
    else:
        backbone = AutoModel.from_pretrained(backbone_path, local_files_only=True)
    return GaiaTimeLLMLithofacies(
        backbone,
        prompt_token_ids=torch.tensor(prompt_ids),
        lora_rank=lora_rank,
        lora_last_blocks=lora_last_blocks,
        numerical_width=int(values.pop("numerical_width", 64)),
        prototype_subset_size=int(values.pop("prototype_subset_size", 256)),
        prototype_count=int(values.pop("prototype_count", 64)),
        heads=int(values.pop("heads", 4)),
        key_width=int(values.pop("key_width", 16)),
        dropout=float(values.pop("dropout", 0.05)),
        **values,
    )
