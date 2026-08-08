"""Compact Time-LLM-style numerical reprogramming for frozen language models.

This is an original, dependency-light implementation of the architectural
idea in Time-LLM (Jin et al., ICLR 2024): numerical patches query a small set
of vocabulary-derived prototypes and are then passed through a frozen language
backbone.  It intentionally does not copy the upstream implementation and does
not perform full-parameter language-model fine-tuning.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def parameter_report(module: nn.Module) -> dict[str, int | float]:
    """Return an auditable trainable/total parameter report."""
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
        "trainable_fraction": float(trainable / total) if total else 0.0,
    }


class VocabularyPrototypeBank(nn.Module):
    """Learn compact mixtures of a deterministic frozen vocabulary subset."""

    def __init__(
        self,
        vocabulary_embeddings: torch.Tensor,
        *,
        subset_size: int = 256,
        prototype_count: int = 64,
    ) -> None:
        super().__init__()
        if vocabulary_embeddings.ndim != 2:
            raise ValueError("vocabulary embeddings must have shape [vocabulary, hidden]")
        vocabulary_size = int(vocabulary_embeddings.shape[0])
        if not 2 <= subset_size <= vocabulary_size:
            raise ValueError("subset_size must be between 2 and the vocabulary size")
        if not 2 <= prototype_count <= subset_size:
            raise ValueError("prototype_count must be between 2 and subset_size")
        indices = torch.linspace(0, vocabulary_size - 1, subset_size).round().long()
        selected = vocabulary_embeddings.detach().index_select(0, indices).float().clone()
        self.register_buffer("selected_embeddings", selected, persistent=True)
        initial = torch.full((prototype_count, subset_size), -8.0)
        anchors = torch.linspace(0, subset_size - 1, prototype_count).round().long()
        initial[torch.arange(prototype_count), anchors] = 8.0
        self.mixture_logits = nn.Parameter(initial)

    def forward(self) -> torch.Tensor:
        weights = torch.softmax(self.mixture_logits, dim=-1)
        return weights @ self.selected_embeddings


class ReprogrammingAttention(nn.Module):
    """Cross-attend numerical patch queries to vocabulary prototypes."""

    def __init__(
        self,
        *,
        numerical_width: int,
        language_width: int,
        heads: int = 4,
        key_width: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if numerical_width <= 0 or language_width <= 0 or heads <= 0 or key_width <= 0:
            raise ValueError("attention dimensions must be positive")
        self.heads = int(heads)
        self.key_width = int(key_width)
        joint = self.heads * self.key_width
        self.query_projection = nn.Linear(numerical_width, joint)
        self.key_projection = nn.Linear(language_width, joint)
        self.value_projection = nn.Linear(language_width, joint)
        self.output_projection = nn.Linear(joint, language_width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, numerical_tokens: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        batch, token_count, _ = numerical_tokens.shape
        prototype_count = prototypes.shape[0]
        queries = self.query_projection(numerical_tokens).view(
            batch, token_count, self.heads, self.key_width
        )
        keys = self.key_projection(prototypes).view(
            prototype_count, self.heads, self.key_width
        )
        values = self.value_projection(prototypes).view(
            prototype_count, self.heads, self.key_width
        )
        scores = torch.einsum("bthd,phd->bhtp", queries, keys) / math.sqrt(self.key_width)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        reprogrammed = torch.einsum("bhtp,phd->bthd", attention, values)
        return self.output_projection(reprogrammed.reshape(batch, token_count, -1))


class LoRAConv1D(nn.Module):
    """Low-rank residual for the Conv1D projection used by GPT-2."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not hasattr(base, "weight") or base.weight.ndim != 2:
            raise TypeError("LoRAConv1D expects a GPT-style two-dimensional Conv1D weight")
        input_width, output_width = map(int, base.weight.shape)
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(input_width, rank, bias=False)
        self.lora_b = nn.Linear(rank, output_width, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = float(alpha / rank)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.lora_b(self.lora_a(values)) * self.scale


def enable_gpt2_lora(
    backbone: nn.Module,
    *,
    last_blocks: int = 2,
    rank: int = 4,
    alpha: float = 8.0,
) -> list[str]:
    """Inject trainable LoRA projections into the last GPT-2 attention blocks."""
    blocks = getattr(backbone, "h", None)
    if blocks is None:
        raise TypeError("GPT-2 LoRA requires a backbone exposing transformer blocks as .h")
    if not 1 <= last_blocks <= len(blocks):
        raise ValueError("last_blocks is outside the backbone depth")
    replaced: list[str] = []
    for block_index in range(len(blocks) - last_blocks, len(blocks)):
        block = blocks[block_index]
        for name in ("c_attn", "c_proj"):
            base = getattr(block.attn, name)
            setattr(block.attn, name, LoRAConv1D(base, rank=rank, alpha=alpha))
            replaced.append(f"h.{block_index}.attn.{name}")
    return replaced


class CompactTimeLLMReprogrammer(nn.Module):
    """Frozen LLM plus trainable numerical tokenizer, reprogrammer, and head."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        input_channels: int,
        output_size: int,
        prompt_token_ids: torch.Tensor,
        sequence_length: int | None = None,
        numerical_width: int = 64,
        prototype_subset_size: int = 256,
        prototype_count: int = 64,
        heads: int = 4,
        key_width: int = 16,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or output_size <= 0:
            raise ValueError("input_channels and output_size must be positive")
        if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
            raise ValueError("prompt_token_ids must be a nonempty one-dimensional tensor")
        embeddings = backbone.get_input_embeddings()
        language_width = int(embeddings.weight.shape[1])
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone.eval()
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False
        self.backbone = backbone
        self.register_buffer("prompt_token_ids", prompt_token_ids.long().clone(), persistent=True)
        self.numerical_tokenizer = nn.Sequential(
            nn.Conv1d(input_channels, numerical_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(numerical_width, numerical_width, kernel_size=1),
        )
        self.prototype_bank = VocabularyPrototypeBank(
            embeddings.weight,
            subset_size=prototype_subset_size,
            prototype_count=prototype_count,
        )
        self.reprogrammer = ReprogrammingAttention(
            numerical_width=numerical_width,
            language_width=language_width,
            heads=heads,
            key_width=key_width,
            dropout=dropout,
        )
        self.sequence_length = sequence_length
        if sequence_length is None:
            self.head = nn.Sequential(
                nn.LayerNorm(language_width),
                nn.Dropout(dropout),
                nn.Linear(language_width, output_size),
            )
        else:
            if sequence_length <= 0:
                raise ValueError("sequence_length must be positive")
            self.head = nn.Sequential(
                nn.LayerNorm(language_width),
                nn.Dropout(dropout),
                nn.Flatten(start_dim=1),
                nn.Linear(language_width * sequence_length, output_size),
            )

    def train(self, mode: bool = True) -> "CompactTimeLLMReprogrammer":
        super().train(mode)
        # A frozen backbone is a deterministic feature operator; its dropout
        # must not silently become a second source of model variation.
        self.backbone.eval()
        return self

    def forward(self, numerical_sequence: torch.Tensor) -> torch.Tensor:
        if numerical_sequence.ndim != 3:
            raise ValueError("numerical_sequence must have shape [batch, length, channels]")
        if not torch.isfinite(numerical_sequence).all():
            raise ValueError("numerical_sequence contains NaN/Inf")
        numerical_tokens = self.numerical_tokenizer(
            numerical_sequence.transpose(1, 2)
        ).transpose(1, 2)
        prototypes = self.prototype_bank().to(dtype=numerical_tokens.dtype)
        reprogrammed = self.reprogrammer(numerical_tokens, prototypes)
        prompt_ids = self.prompt_token_ids.unsqueeze(0).expand(numerical_sequence.shape[0], -1)
        prompt_embeddings = self.backbone.get_input_embeddings()(prompt_ids)
        combined = torch.cat(
            (prompt_embeddings.to(dtype=reprogrammed.dtype), reprogrammed), dim=1
        )
        attention_mask = torch.ones(
            combined.shape[:2], dtype=torch.long, device=combined.device
        )
        hidden = self.backbone(inputs_embeds=combined, attention_mask=attention_mask).last_hidden_state
        numerical_hidden = hidden[:, -numerical_sequence.shape[1] :, :]
        if self.sequence_length is None:
            return self.head(numerical_hidden.mean(dim=1))
        if numerical_hidden.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected sequence length {self.sequence_length}, found {numerical_hidden.shape[1]}"
            )
        return self.head(numerical_hidden)
