from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2Model

from _code.foundation import CompactTimeLLMReprogrammer, enable_gpt2_lora, parameter_report


def test_compact_timellm_freezes_backbone_and_trains_adapter() -> None:
    backbone = GPT2Model(
        GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=2)
    )
    model = CompactTimeLLMReprogrammer(
        backbone,
        input_channels=5,
        output_size=3,
        prompt_token_ids=torch.tensor([1, 2, 3]),
        sequence_length=7,
        numerical_width=8,
        prototype_subset_size=16,
        prototype_count=8,
        heads=2,
        key_width=4,
        dropout=0.0,
    )
    output = model(torch.randn(4, 7, 5))
    assert output.shape == (4, 3)
    output.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    )
    report = parameter_report(model)
    assert 0 < report["trainable_parameters"] < report["total_parameters"]


def test_gpt2_lora_starts_as_an_exact_residual_and_is_trainable() -> None:
    backbone = GPT2Model(
        GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    )
    backbone.eval()
    values = torch.randn(2, 5, 16)
    with torch.no_grad():
        before = backbone(inputs_embeds=values).last_hidden_state
    names = enable_gpt2_lora(backbone, last_blocks=1, rank=2, alpha=4.0)
    with torch.no_grad():
        after = backbone(inputs_embeds=values).last_hidden_state
    assert names == ["h.1.attn.c_attn", "h.1.attn.c_proj"]
    assert torch.allclose(before, after)
    assert any("lora_" in name and parameter.requires_grad for name, parameter in backbone.named_parameters())
