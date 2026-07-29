# Facies P10 model-results audit

## Conclusion

The corrected SAM2 environment ran the development-only repair audit for both facies datasets.
Real pretrained weights improve the same architecture over random initialization, so the foundation encoder is useful.
However, the pretrained adapter still trails the locked strong segmentation baselines, and the tested gated-residual fusion degrades the pretrained adapter.
The honest end-to-end decision therefore remains non_beneficial; no holdout was used for tuning.

## Interface audit matrix

| Item | Evidence | Result |
|---|---|---|
| Input channels | `_models/facies/sam2_semantic.py:forward`; summaries record `[B,1,H,W]` | one seismic channel is repeated to three channels before encoder normalization |
| Amplitude scaling | adapter forward path | clamp `[-5,5]`, rescale to `[0,1]`, then ImageNet mean/std |
| Native SAM2 preprocessing | audited source checkout and real checkpoint hash | official SAM2.1 Hiera-B+ encoder loaded with real pretrained weights |
| Prompt leakage | no prompt input in semantic adapter | no validation-label prompt path exists |
| Label mapping | `pipeline_contract.py` | F3 remains 10-class; Penobscot remains 8-class |
| Decoder / fusion | `p10_sam2_repair_audit.py` | frozen base logits plus a sigmoid-gated residual head was tested |
| PEFT / freeze policy | repair summaries | base adapter frozen; only residual head and scalar gate trained |
| Loss / postprocess | repair script and `p4_metrics.py` | weighted cross-entropy on raw logits; argmax only for evaluation |
| Evaluation parity | manifest hashes and fold rows | fixed development manifests, fixed sample caps/seeds, no frozen-test access |

## Development-only comparison

| Dataset | Strong baseline mIoU | Pretrained adapter mIoU | Random-init mIoU | Foundation gain | Gated repair mIoU | Repair vs pretrained |
|---|---:|---:|---:|---:|---:|---:|
| F3 | 0.122046 | 0.081087 | 0.020871 | +0.060216 | 0.031691 | -0.049396 |
| Penobscot | 0.156721 | 0.059506 | 0.009421 | +0.050084 | 0.025733 | -0.033773 |

## Per-fold evidence

### F3

| Fold | Seed | Strong baseline | Pretrained adapter | Random-init | Gated repair |
|---|---:|---:|---:|---:|---:|
| 0 | 2693 | 0.131036 | 0.067439 | 0.017070 | 0.019523 |
| 4 | 2697 | 0.113056 | 0.094735 | 0.024673 | 0.043860 |

### Penobscot

| Fold | Seed | Strong baseline | Pretrained adapter | Random-init | Gated repair |
|---|---:|---:|---:|---:|---:|
| 0 | 2693 | 0.157546 | 0.072871 | 0.001155 | 0.014827 |
| 4 | 2697 | 0.155897 | 0.046141 | 0.017688 | 0.036640 |

## Root cause and fix status

- The earlier blocker was an environment-selection error: `atom-sam2-py310` already contained Hydra and the audited SAM2 dependencies.
- Fix applied: execute both repair probes in that environment and replace the false blocker with measured development evidence.
- The tested gated-residual fusion is not a successful repair; it underfits and reduces mIoU relative to the pretrained adapter.
- Pretraining itself is beneficial versus same-architecture random initialization, but the current semantic adapter/head is still not competitive with the strong task-specific baselines.

## Evidence boundary

- Both summaries record `frozen_test_accessed=false`.
- Each probe used two fixed development folds, fixed seeds, at most 32 train samples and 16 validation samples per fold.
- No threshold tuning, seed selection, label remapping, or holdout reuse was performed.

## Residual risk

- This was a bounded repair audit, not a full backbone/head retraining campaign.
- A future attempt should test a properly trained multi-scale decoder or parameter-efficient fine-tuning strategy on the frozen development protocol; the current gated residual head should not be promoted.
