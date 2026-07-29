# P11 lithofacies gated residual-fusion evidence

## Decision

State: `NON_BENEFICIAL_KEEP_BASELINE`; default enabled: `false`.

## Strict LOGO4 ablations

| variant | fixed-9 macro-F1 mean | std | accuracy | gate mean | mean abs residual |
|---|---:|---:|---:|---:|---:|
| baseline | 0.194938 | 0.013858 | 0.492678 | — | — |
| direct | 0.051505 | 0.027386 | 0.213329 | — | — |
| pretrained | 0.196978 | 0.014964 | 0.497047 | 0.017773 | 0.022956 |
| random | 0.194120 | 0.015700 | 0.490009 | 0.017841 | 0.034227 |
| gate0 | 0.194938 | 0.013858 | 0.492678 | 0.000000 | 0.000000 |

## Paired checks

- pretrained − baseline: `+0.002040`.
- pretrained − random residual: `+0.002857`.
- pretrained − direct MOMENT: `+0.145473`.
- pretrained wins over baseline in `6/12` fold/seed pairs.
- gate0 − baseline: `+0.000000`; maximum logit error `0.0e+00`.

## Every fold and seed

| fold | seed | baseline | direct | pretrained | random | gate0 | pretrained gate | mean abs residual |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1867973658 | 0.213580 | 0.055556 | 0.218938 | 0.213580 | 0.213580 | 0.017806 | 0.024297 |
| 0 | 2137841944 | 0.213580 | 0.055556 | 0.217465 | 0.213580 | 0.213580 | 0.017729 | 0.019894 |
| 0 | 3902865753 | 0.213580 | 0.087314 | 0.217621 | 0.217621 | 0.213580 | 0.017826 | 0.028493 |
| 1 | 1867973658 | 0.183572 | 0.033689 | 0.180975 | 0.183572 | 0.183572 | 0.017716 | 0.015331 |
| 1 | 2137841944 | 0.183572 | 0.063371 | 0.190691 | 0.181197 | 0.183572 | 0.017778 | 0.022456 |
| 1 | 3902865753 | 0.183572 | 0.078250 | 0.190691 | 0.172193 | 0.183572 | 0.017779 | 0.024718 |
| 2 | 1867973658 | 0.181276 | 0.030234 | 0.181276 | 0.181276 | 0.181276 | 0.017777 | 0.021074 |
| 2 | 2137841944 | 0.181276 | 0.030234 | 0.181276 | 0.181450 | 0.181276 | 0.017723 | 0.023332 |
| 2 | 3902865753 | 0.181276 | 0.097029 | 0.181276 | 0.181450 | 0.181276 | 0.017722 | 0.022317 |
| 3 | 1867973658 | 0.201322 | 0.000000 | 0.199196 | 0.199196 | 0.201322 | 0.017812 | 0.025339 |
| 3 | 2137841944 | 0.201322 | 0.043614 | 0.201322 | 0.201322 | 0.201322 | 0.017765 | 0.022686 |
| 3 | 3902865753 | 0.201322 | 0.043210 | 0.203006 | 0.203006 | 0.201322 | 0.017841 | 0.025534 |

## Scientific boundary

- Inputs are the immutable four development mother-family folds; F-5 and every holdout-like path are rejected before opening.
- All trainable MOMENT lanes use the same 40-update, batch-32, fold-train-only class-weight contract.
- `pretrained` and `random` have the same bounded embedding-head residual architecture; only the frozen MOMENT feature extractor initialization differs.
- `direct` is the pretrained MOMENT classifier without XGBoost logits; `gate0` is the exact trained-residual degeneration.
- This evidence is development-only and cannot authorize a new frozen-test read or a competition claim.
