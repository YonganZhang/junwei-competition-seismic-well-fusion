"""Property adapter using Gaia domain constraints and Time-LLM reprogramming."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from _code.foundation import CompactTimeLLMReprogrammer, enable_gpt2_lora
from _code.ml_framework.contracts import ModelBatch, ModelOutput, TaskSpec
from _models.property._p5_common import (
    PROPERTY_TARGETS,
    property_output,
    seed_torch_runtime,
    target_arrays,
    validate_property_task_spec,
)


model_id = "gaia_timellm_gpt2"

DOMAIN_PROMPT = (
    "Petroleum reservoir property task. Estimate PHIF porosity fraction, "
    "log1p KLOGH permeability, and SW water saturation from a nine-depth "
    "sequence of GR RT NPHI RHOB observed masks and local ST0202 seismic. "
    "Respect physical units and never use target curves as inputs."
)


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["well_log_sequence", "seismic_patch"],
        "supports_missing_mask": True,
        "supports_uncertainty": False,
        "foundation_strategy": "frozen GPT-2 vocabulary reprogramming",
        "domain_guidance": "Gaia-style petroleum schema and unit prompt",
        "full_parameter_finetuning": False,
    }


def _sequence(batch: ModelBatch) -> np.ndarray:
    logs = np.asarray(batch.inputs["well_log_sequence"], dtype=np.float32)
    seismic = np.asarray(batch.inputs["seismic_patch"], dtype=np.float32)
    if logs.ndim != 3 or logs.shape[1:] != (9, 8):
        raise ValueError(f"expected well logs [N,9,8], found {logs.shape}")
    if seismic.ndim != 4 or seismic.shape[1:] != (3, 3, 9):
        raise ValueError(f"expected seismic [N,3,3,9], found {seismic.shape}")
    seismic_sequence = seismic.transpose(0, 3, 1, 2).reshape(len(seismic), 9, 9)
    result = np.concatenate((logs, seismic_sequence), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("property sequence contains NaN/Inf")
    return result


class GaiaTimeLLMPropertyAdapter:
    """One-optimizer-step adapter compatible with the canonical model contract."""

    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        backbone_path: str,
        target_mean: list[float] | tuple[float, ...],
        target_std: list[float] | tuple[float, ...],
        device: str = "cpu",
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        seed: int = 2693,
        random_backbone: bool = False,
        lora_rank: int = 0,
        lora_last_blocks: int = 2,
        **model_config: Any,
    ) -> None:
        validate_property_task_spec(task_spec)
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        seed_torch_runtime(torch, seed)
        local = Path(backbone_path)
        if not local.is_dir():
            raise FileNotFoundError(f"local backbone snapshot is missing: {local}")
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True)
        prompt_ids = tokenizer(
            DOMAIN_PROMPT, add_special_tokens=True, truncation=True, max_length=64
        )["input_ids"]
        if random_backbone:
            backbone = AutoModel.from_config(AutoConfig.from_pretrained(local, local_files_only=True))
        else:
            backbone = AutoModel.from_pretrained(local, local_files_only=True)
        self.torch = torch
        self.device = torch.device(device)
        self.task_spec = task_spec
        self.module = CompactTimeLLMReprogrammer(
            backbone,
            input_channels=17,
            output_size=len(PROPERTY_TARGETS),
            prompt_token_ids=torch.tensor(prompt_ids),
            sequence_length=9,
            **model_config,
        )
        self.lora_modules = (
            enable_gpt2_lora(
                self.module.backbone,
                last_blocks=lora_last_blocks,
                rank=lora_rank,
                alpha=float(2 * lora_rank),
            )
            if lora_rank > 0
            else []
        )
        self.module.to(self.device)
        mean = np.asarray(target_mean, dtype=np.float32)
        std = np.asarray(target_std, dtype=np.float32)
        if mean.shape != (3,) or std.shape != (3,) or np.any(std <= 0):
            raise ValueError("target_mean/target_std must be positive three-vectors")
        self.target_mean = torch.as_tensor(mean, device=self.device)
        self.target_std = torch.as_tensor(std, device=self.device)
        self.optimizer = torch.optim.AdamW(
            (p for p in self.module.parameters() if p.requires_grad),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _prediction_standardized(self, batch: ModelBatch) -> Any:
        values = self.torch.as_tensor(_sequence(batch), dtype=self.torch.float32, device=self.device)
        return self.module(values)

    def fit(self, batch: ModelBatch) -> dict[str, Any]:
        target, mask = target_arrays(batch, self.task_spec)
        target_tensor = self.torch.as_tensor(target, dtype=self.torch.float32, device=self.device)
        mask_tensor = self.torch.as_tensor(mask, dtype=self.torch.float32, device=self.device)
        standardized = (target_tensor - self.target_mean) / self.target_std
        self.module.train()
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self._prediction_standardized(batch)
        denominator = mask_tensor.sum(dim=0).clamp_min(1.0)
        per_target = (((prediction - standardized) ** 2) * mask_tensor).sum(dim=0) / denominator
        loss = per_target.mean()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(
            [p for p in self.module.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()
        return {"loss": float(loss.detach().cpu()), "backward": True}

    def predict_array(self, batch: ModelBatch) -> np.ndarray:
        self.module.eval()
        with self.torch.no_grad():
            prediction = self._prediction_standardized(batch)
            prediction = prediction * self.target_std + self.target_mean
        return prediction.detach().cpu().numpy().astype(np.float64)

    def predict(self, batch: ModelBatch) -> ModelOutput:
        return property_output(self.predict_array(batch), self.task_spec)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        adapter_state = {
            key: value.detach().cpu()
            for key, value in self.module.state_dict().items()
            if not key.startswith("backbone.") or ".lora_" in key
        }
        self.torch.save(
            {
                "schema_version": 1,
                "model_id": model_id,
                "targets": PROPERTY_TARGETS,
                "checkpoint_scope": "adapter_only",
                "adapter_state": adapter_state,
                "optimizer_state": self.optimizer.state_dict(),
                "target_mean": self.target_mean.detach().cpu(),
                "target_std": self.target_std.detach().cpu(),
                "lora_modules": self.lora_modules,
                "backbone_path_persisted": False,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        value = self.torch.load(path, map_location=self.device, weights_only=False)
        if value.get("model_id") != model_id or tuple(value.get("targets", ())) != PROPERTY_TARGETS:
            raise ValueError("checkpoint identity mismatch")
        if value.get("checkpoint_scope") != "adapter_only":
            raise ValueError("checkpoint scope mismatch")
        missing, unexpected = self.module.load_state_dict(value["adapter_state"], strict=False)
        illegal_missing = [
            key
            for key in missing
            if not key.startswith("backbone.") or ".lora_" in key
        ]
        if illegal_missing or unexpected:
            raise ValueError(
                f"adapter checkpoint mismatch: missing={illegal_missing}, unexpected={unexpected}"
            )
        self.optimizer.load_state_dict(value["optimizer_state"])


def build_model(task_spec: TaskSpec, **config: Any) -> GaiaTimeLLMPropertyAdapter:
    return GaiaTimeLLMPropertyAdapter(task_spec, **config)
