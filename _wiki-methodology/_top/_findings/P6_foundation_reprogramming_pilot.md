---
phase_id: P6
status: accepted
owner_col: col2
---

# P6 foundation reprogramming pilot

## Finding

Time-LLM-style GPT-2 reprogramming is technically compatible with the property
and lithofacies sequence contracts. Pretrained GPT-2 provides a consistent
advantage over an architecture-matched random GPT-2, but the adapters do not
yet beat simple same-input baselines.

## Decision

- Keep Gaia as the shared petroleum control and explanation layer.
- Keep modality-specific foundation models for numerical prediction.
- Continue only parameter-efficient, development-only studies.
- Do not promote the P6 adapters to current winners.
- Do not force Time-LLM into fault, facies or 3-D reconstruction.

## Evidence

- `_pipelines/04_foundation_adaptation/analysis-report.md`
- `_wiki-methodology/_tests/P6_foundation_adaptation_evidence.md`
- `_figures/p6_foundation_adaptation/fig1_foundation_pretraining_ablation.png`
