# P11 XGBoost–MOMENT cross-attention fusion evidence

## Objective result

Cross-attention fusion minus the strong XGBoost baseline on mean fixed-nine Macro-F1: `+0.006653`.

This is the overall system delta. It is not attributed to MOMENT. **Large-model contribution share awaits the next pretrained-versus-random encoder ablation.**

**大模型贡献占比待下一轮消融确认。**

## Strict LOGO4 variants

| variant | fixed-9 macro-F1 mean | std | accuracy | gate mean | mean abs residual |
|---|---:|---:|---:|---:|---:|
| baseline | 0.194938 | 0.013858 | 0.492678 | — | — |
| prior_calibrated | 0.202187 | 0.016854 | 0.511026 | — | — |
| cross_attention | 0.201590 | 0.017373 | 0.508418 | 0.021079 | 0.041222 |

## Component deltas

- Prior calibration − baseline: `+0.007249`.
- Cross-attention − prior calibration: `-0.000597`.
- Cross-attention wins over baseline in `12/12` fold/seed pairs.

## Architecture and optimization boundary

- XGBoost query tokens are real archived booster leaf IDs with shape `40 rounds × 9 classes`; no validation labels enter them.
- MOMENT key/value tokens are `13 log channels × 4 native patches`; there is no 33→512 interpolation.
- Prior calibration adds `0.25 × centered log(fold-train class count)` to partially undo inverse-sqrt class weighting. It is reported as a separate control.
- Encoder initialization is parameterized as `('pretrained', 'random')`. This evidence ran `pretrained` only.
- The fused output is a learned bounded residual over calibrated XGBoost logits, with exact zero-gate degeneration checked per cell.

## Development exploration log

- This is an explicitly exploratory LOGO development result, not an unbiased holdout estimate.
- Averaging the three XGBoost seed logits changed no argmax and gave `+0.000000`; it was not retained.
- Train-prior shrinkage candidates `0.20/0.25/0.30` produced mean fixed-nine Macro-F1 `0.201375/0.202187/0.202058`; `0.25` was frozen for the formal matrix.
- Depth-logit smoothing tested radii `1/2/3/4/6/8` and mixing `0.25/0.50/0.75/1.0`. Its best standalone delta was `+0.003821`, but it was unstable with prior calibration and was rejected.
- Cross-attention architecture/training constants were frozen after fold 0, repeat 0 smoke (`0.213580 → 0.227106 → 0.228214`) and then run unchanged.

## Every fold and seed

| fold | seed | baseline | prior calibrated | cross attention | gate |
|---:|---:|---:|---:|---:|---:|
| 0 | 1867973658 | 0.213580 | 0.227106 | 0.228214 | 0.021156 |
| 0 | 2137841944 | 0.213580 | 0.227106 | 0.227106 | 0.021119 |
| 0 | 3902865753 | 0.213580 | 0.227106 | 0.227106 | 0.021284 |
| 1 | 1867973658 | 0.183572 | 0.189821 | 0.184641 | 0.021182 |
| 1 | 2137841944 | 0.183572 | 0.189821 | 0.189821 | 0.021179 |
| 1 | 3902865753 | 0.183572 | 0.189821 | 0.189821 | 0.021381 |
| 2 | 1867973658 | 0.181276 | 0.186192 | 0.186192 | 0.020593 |
| 2 | 2137841944 | 0.181276 | 0.186192 | 0.186192 | 0.020598 |
| 2 | 3902865753 | 0.181276 | 0.186192 | 0.186192 | 0.020762 |
| 3 | 1867973658 | 0.201322 | 0.205629 | 0.202543 | 0.021215 |
| 3 | 2137841944 | 0.201322 | 0.205629 | 0.205629 | 0.021138 |
| 3 | 3902865753 | 0.201322 | 0.205629 | 0.205629 | 0.021337 |

## Leakage and preservation audit

- Only the immutable development LOGO4 folds were opened. Frozen holdout and every holdout-like path remain blocked.
- Calibration uses fold-train class counts only. Leaf IDs come from the archived fold-train-fitted XGBoost booster.
- The two earlier committed P11 evidence sets were verified byte-for-byte before this run.
- This development result cannot establish how much of the delta comes from pretrained MOMENT until the next encoder-init ablation.
