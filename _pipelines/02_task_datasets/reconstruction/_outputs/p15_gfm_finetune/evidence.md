# P15 GFM partial fine-tuning — development-only evidence

## Outcome

- PyKrige baseline pooled development RMSE: `0.028449728170`.
- Pretrained GFM partial-fine-tune gated mean-seed RMSE: `0.029071638337` (delta `+0.000621910167`, positive relative gain means improvement: `-2.185997%`).
- Matched random-init partial-fine-tune gated mean-seed RMSE: `0.029100154266`; pretrained minus random-init: `-0.000028515929`.
- Five independent spatial-fold outcomes for pretrained: 1 win / 3 loss / 1 tie.
- Decision: `VERIFIED_NO_PROMOTION`. No improvement is automatically attributed to the foundation weights.
- The optimization dynamics are fundamentally different from P14: P15 has genuine encoder gradients, directionally persistent gradient probes and parameter movement. The development generalization conclusion is still unchanged: it does not reverse the baseline result or establish a separated pretrained effect.

## Native continuous-volume window audit

- P14 did not resize each isolated `9×20×18` patch directly: it first assembled the development patches to `63×100×72`, then resized each `63×100` section to `400×160`. The K/time axis was still enlarged about 6.35×.
- P15 maps the OOF cells back to the original ST0202 SEG-Y (`385×605` traces, `1126` samples/trace, 4 ms sampling) and reads exact `400`-sample × `160`-adjacent-trace sections.
- The development cells occupy 84 native inline sections, crossline span 80 and time span 29; both fit inside one native GFM window.
- Resize/interpolation/padding applied: `False` / `False` / `False`.
- Three channels are forwarded separately: raw amplitude, 5-sample local RMS, and two-sample vertical gradient, all derived on the native window.

## Genuine partial fine-tuning

- GFM has 16 encoder blocks. Blocks 0-14 are frozen and their prefix tokens are cached as `float16` (a disclosed storage quantization); block 15 and the final LayerNorm execute in the training graph and receive gradients.
- Differential AdamW learning rates: encoder `1e-05`, new projection/regression head `0.0003`; weight decay `0.0001`, gradient clip `1.0`.
- Early stopping uses one inner spatial fold only. The selected update count is then refit from the identical initialization on all four outer-training folds before the untouched outer fold is predicted.
- Pretrained refit encoder gradient norm (cell means): `8.87261e-06`; encoder update L2: `0.00702539`.
- The bounded encoder-gradient probe has mean adjacent-step cosine `0.352014`; all 15/15 refits have positive mean direction consistency. This proves an optimization signal, not a useful pretrained effect.
- Nonzero gradients in every pretrained refit: `True`; encoder parameters moved in every refit: `True`.
- Matched random-init uses the same block 15, head, learning rates, update selection, weight decay and gate protocol. Its weights are initialized independently for the same three paired seeds.

## Five genuinely independent spatial units

The five locked folds are the inferential units. The three seeds are paired optimization pseudo-repeats inside each fold, not 15 independent samples.

| fold | baseline RMSE | pretrained RMSE | random-init RMSE | pretrained delta | outcome |
|---:|---:|---:|---:|---:|:---|
| 0 | 0.027239559 | 0.027224113 | 0.027207469 | -0.000015446 | win |
| 1 | 0.028627607 | 0.029426147 | 0.029437662 | +0.000798541 | loss |
| 2 | 0.018619291 | 0.018634894 | 0.018697948 | +0.000015603 | loss |
| 3 | 0.028605050 | 0.028605050 | 0.028605050 | +0.000000000 | tie |
| 4 | 0.036338338 | 0.038120940 | 0.038198064 | +0.001782602 | loss |

## Whole-fold bootstrap

- Pretrained minus PyKrige pooled RMSE point estimate: `+0.000621910167`; 95% interval `[-0.000004176718, +0.001309392151]`.
- Pretrained minus matched-random RMSE point estimate: `-0.000028515929`; 95% interval `[-0.000061113143, +0.000004015409]`.
- Bootstrap unit is the whole locked spatial fold. Seeds remain paired; voxels are never independently resampled.

## Holdout firewall

- `train.h5` reads are limited to patch metadata, `seismic_patch[3:6]` and `seismic_patch[8]`; no HDF5 label dataset is read.
- The continuous SEG-Y is used only for label-free seismic covariates. `test.h5`, frozen holdout paths, holdout labels and historical test metrics are neither opened nor probed.
- PORO targets and PyKrige predictions come only from the same five hash-verified development OOF archives as P11-P14.
