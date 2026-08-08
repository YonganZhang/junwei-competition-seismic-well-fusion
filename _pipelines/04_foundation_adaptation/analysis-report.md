# P6 Gaia-guided foundation adaptation: development results

## Decision

The foundation-model integration is technically runnable, and pretrained
GPT-2 contributes useful initialization in both tested sequence tracks.
However, neither Time-LLM-style adapter currently beats its same-input simple
baseline. The P6 models therefore remain research candidates and must not
replace the current track winners.

## What was tested

- Gaia V2 supplies petroleum ontology, allowed inputs, units, physical bounds,
  missing-modality rules and interpretation prompts.
- The numerical backbone is the official `openai-community/gpt2` snapshot.
- GPT-2 is frozen; LoRA is enabled only in the last two attention blocks.
- Each learned variant uses three fixed seeds: 2693, 2701 and 2711.
- Training has a fixed update budget and does not use validation for early
  stopping.
- All metrics are from fixed development folds. No frozen test input or
  historical test metric is accepted by either runner.

## Results

| Track | Same-input baseline | Pretrained GPT-2 + LoRA | Random GPT-2 + LoRA | Foundation contribution | Promotion |
|---|---:|---:|---:|---:|---|
| Property | Ridge RMSE 0.8646 | RMSE 1.0891 ± 0.0922 | RMSE 1.2592 ± 0.1847 | 13.50% lower RMSE than random | No |
| Lithofacies | Logistic accuracy 0.3939 | Accuracy 0.2778 ± 0.0554 | Accuracy 0.1136 ± 0.0750 | +0.1641 accuracy over random | No |

For property, lower is better. For lithofacies, higher is better. The
pretrained variant beat the random-backbone variant at every matched seed in
both tracks. That supports a real pretraining contribution, but it does not
establish task-level superiority over the baselines.

## Six-track route

| Track | Foundation route | Current state |
|---|---|---|
| Fault | 3-D vision/seismic encoder with parameter-efficient segmentation adapter | Blocked until verified negatives and legal development folds exist |
| Seismic facies | Pretrained vision/seismic encoder plus decoder adapter | Source integration ready; checkpoint license/weight approval still required |
| Property | Time-LLM-style numerical reprogramming + GPT-2 LoRA | Three-seed pilot complete; no promotion |
| Lithofacies | Well-log/seismic sequence reprogramming + GPT-2 LoRA | Three-seed pilot complete; no promotion |
| Sweetspot | Target-specific tabular/time-series foundation adapters | T1/T2 are next candidates; T3/T4 need objective/sample repair; T5–T7 remain gated |
| 3-D reconstruction | 3-D masked autoencoder, FNO or conditional diffusion | Route defined; Time-LLM is not scientifically appropriate |

## Interpretation

“Connect all six tracks to a large model” should mean a shared Gaia control
layer plus modality-appropriate foundation backbones. It should not mean
forcing every tensor into a text LLM. The present evidence supports continuing
parameter-efficient pilots for property, lithofacies and selected sweetspot
targets, while using 3-D or vision foundations for spatial tracks.

The next scientific gate is nested, family-isolated development CV with a
small pre-registered adapter search. Promotion requires beating the existing
track winner and passing the unchanged test firewall.

## Command Gate

The P6 unit gate passed 3/3 tests and the shared training-framework regression
passed 22/22 tests. Python compilation, JSON parsing, YAML parsing and
re-rendering the figure all completed with exit code zero.

## Live / User Journey

Both development pilots ran from prepared NPZ batches through backbone load,
adapter training, validation inference, checkpoint hashing, three-seed
aggregation and final figure generation. The resulting PNG was opened and
visually inspected, then published through the project card-render route.

## Trace / SSDO Audit

Durable trace evidence is stored in the two final pilot JSON files and
`figure_data.json`. Each learned cell records its seed, training budget, loss
trace summary, parameter boundary, validation metrics, resource use and
adapter-checkpoint hash. Full console logs and ignored checkpoint binaries are
not part of the durable audit package; this limitation is explicit.
