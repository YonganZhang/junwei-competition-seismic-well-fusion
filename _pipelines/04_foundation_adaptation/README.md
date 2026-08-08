# P6 foundation adaptation

This lane treats Gaia V2 as the petroleum expert/control plane, not as a
numerical foundation checkpoint. Each track keeps a modality-appropriate
backbone and exposes its outputs to Gaia for schema, unit, geological-rule,
visualization, and explanation checks.

`track_routes.v1.json` routes all six tracks. Time-LLM-style language
reprogramming is used only where the input is genuinely sequential
(property, lithofacies and selected sweetspot targets). Fault, seismic facies
and 3-D reconstruction use vision/seismic/volume foundation families instead
of forcing numeric volumes through a text backbone.

The executable P6 pilots use only fixed P5 development folds and never accept
a test path. Each pilot compares the same adapter with:

1. an official pretrained GPT-2 backbone;
2. an architecture-matched randomly initialized GPT-2 backbone; and
3. a same-input simple baseline.

Both pilots use LoRA only on the last two GPT-2 attention blocks. The backbone
stays frozen, validation is not used for early stopping, and only adapter
weights are eligible for persistence.

## Reproduce

```bash
P5_TORCH_PYTHON=/path/to/torch-common/bin/python
GPT2_SNAPSHOT=/path/to/openai-community--gpt2/snapshot

"$P5_TORCH_PYTHON" _pipelines/04_foundation_adaptation/property_timellm_pilot.py \
  --fold /path/to/property/fixed_fold.npz \
  --backbone "$GPT2_SNAPSHOT" \
  --output-dir _pipelines/04_foundation_adaptation/_outputs/property_lora_multiseed \
  --variants pretrained_lora,random_lora \
  --seeds 2693,2701,2711 --update-steps 120 --batch-size 24 --learning-rate 3e-4

"$P5_TORCH_PYTHON" _pipelines/04_foundation_adaptation/lithofacies_timellm_pilot.py \
  --batch /path/to/lithofacies/fixed_fold.npz \
  --backbone "$GPT2_SNAPSHOT" \
  --output-dir _pipelines/04_foundation_adaptation/_outputs/lithofacies_lora_multiseed \
  --variants pretrained_lora,random_lora \
  --seeds 2693,2701,2711 --update-steps 180 --batch-size 32 --learning-rate 3e-4

"$P5_TORCH_PYTHON" _pipelines/04_foundation_adaptation/plot_p6_results.py
```

The tracked JSON files are compact audited evidence. Intermediate checkpoints,
partial results and runtime batches are intentionally ignored.
