# P6 foundation adaptation acceptance evidence

Date: 2026-07-26

## Scope

P6 adds a Gaia-guided foundation adaptation layer without changing frozen
P5 test contracts. Gaia is the petroleum expert/control plane; numerical
prediction uses a modality-appropriate foundation backbone.

## Source and backbone boundary

- Time-LLM reference: official author repository and ICLR 2024 paper.
- Local student TimeLLM project was inspected but contains exported
  configuration/results only; it does not contain a reproducible training
  implementation, raw training data or checkpoint.
- Backbone: `openai-community/gpt2`, revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- License boundary: MIT for GPT-2; Apache-2.0 for official Time-LLM code.
- Imported upstream code: none. P6 implements a compact local reprogramming
  adapter and records the architectural reference.

## Test firewall

Both runners require a single development-batch NPZ and expose no test-path
argument. The development batches contain only explicit train/validation
arrays. Validation is not used for early stopping. Checkpoints persist adapter
weights only.

## Audited results

| Evidence | SHA-256 |
|---|---|
| Property final JSON | `d85ac4175d5fbbae294c4ca6d27046762da4c014112282bb6b8e277cc854551f` |
| Lithofacies final JSON | `0df1c69cd860c88375746301945cf476757fdc6723d693da4720878242da5757` |
| Six-track route | `4e09f8e87e0f70ead593cf18b3d57cd65215816b66d0621cce65418bb584bca2` |
| Figure PNG | `5b07508d85cca53b7d3841d198a68404b8a935e48ce3d8d4e2ec602e53c9ff94` |

Property uses 192 train sequences from three well families and 81 validation
sequences from F-12. The historical F-15 frozen test family is excluded.
Lithofacies uses 315 train sequences from F-14/F-15/F-4 and 132 validation
sequences from 15/9-19. The historical F-5 frozen test family is excluded.

## Acceptance result

- Engineering gate: pass. The adapters are dynamically discoverable, train,
  emit finite outputs and preserve adapter-only checkpoints.
- Foundation ablation: pass. Pretrained GPT-2 outperforms the matched random
  GPT-2 at all three seeds for both tested tracks.
- Promotion gate: fail. Ridge remains better for property and logistic
  regression remains better for lithofacies.
- Scientific action: retain as P6 research evidence; do not replace P5 winners
  and do not consume frozen tests.

## Command gate

```bash
P5_TORCH_PYTHON=/path/to/torch-common/bin/python
"$P5_TORCH_PYTHON" -m pytest -q \
  _code/ml_framework/tests/test_timellm_reprogrammer.py \
  _pipelines/04_foundation_adaptation/tests/test_track_routes.py
"$P5_TORCH_PYTHON" _pipelines/04_foundation_adaptation/plot_p6_results.py
```

Expected unit result: 3 passed.

## Live / User Journey

The two pilot CLIs were run end to end against fixed development NPZ batches.
They loaded the local GPT-2 snapshot, trained adapter-only variants, emitted
per-seed metrics and checkpoint hashes, and then aggregated three seeds. The
plot CLI independently reloaded the final JSON evidence and emitted PNG, PDF
and figure-data JSON.

## Trace / SSDO audit

The auditable traces are the final JSON `results` arrays: each cell records
seed, parameter counts, LoRA module names, update budget, first/last/minimum
loss, validation metrics, runtime, peak CUDA memory and adapter checkpoint
SHA-256. Partial runtime logs and heavyweight checkpoints are intentionally
not treated as durable evidence. This is a structured-evidence downgrade from
a full experiment tracker, but it is sufficient to rerun and verify the stated
development-only conclusion.
