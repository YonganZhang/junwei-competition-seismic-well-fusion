# P29 lithofacies agent-action effect evidence

## Outcome

The bounded development verdict is **REJECT_AGENT**. A0 reproduced `0.2133487970` fixed-schema Macro-F1. Enhanced A2L produced `0.2047148654` (`-0.0086339317`).

This is adaptive development evidence and does not use a frozen holdout.

## Repaired observation and repeat contract

- The policy sees clipped deltas in 0.005 effect units, anonymous per-class train-support shares, and uncertainty from three disjoint inner LOGO folds.
- Raw metrics, labels, class names, group identities, sample identifiers, paths, residuals, and outer results remain hidden.
- The executor uses exactly one model seed `1867973658`. No duplicated deterministic seed is described as a replicate.
- The primary metric and split hash are unchanged.

## Arms

| arm | observation | normalized AUC@3 | outer mean | delta vs A0 | positive folds |
|---|---|---:|---:|---:|---:|
| A2L | safe normalized | 0.8740660224 | 0.2047148654 | -0.0086339317 | 0/4 |
| A2L categorical ablation | categorical only | 0.8740660224 | 0.2047148654 | -0.0086339317 | 0/4 |
| A2D | deterministic enhanced | 0.4176927230 | 0.2047148654 | -0.0086339317 | 0/4 |
| A3 | random median | 0.4864923389 | 0.2047148654 | -0.0086339317 | n/a |

## Action effects and transfer

All four non-baseline actions have different config hashes: **True**.

All four change an inner or outer prediction hash: **True**.

The exhaustive inner ceiling is `+0.0053158930`. A2L regret to that ceiling is `+0.0036975277`.

Across 16 action-by-outer pairs, inner-to-outer delta correlation is `0.19584841847480586`. 2 of 5 inner-positive pairs remain outer-positive.

The transfer matrix was computed only after all live decisions. It was not shown to a policy and did not select any legal endpoint.

## A2L outer results

| outer fold | selected action | inner delta | outer delta |
|---:|---|---:|---:|
| 0 | A0_DEPTH3_ETA01_ROUNDS60 | +0.0000000000 | +0.0000000000 |
| 1 | ACT_WEIGHT_EXP075_MEAN1 | +0.0064734613 | -0.0345357267 |
| 2 | A0_DEPTH3_ETA01_ROUNDS60 | +0.0000000000 | +0.0000000000 |
| 3 | A0_DEPTH3_ETA01_ROUNDS60 | +0.0000000000 | +0.0000000000 |

## Gates

- `a1_identity_replay`: **PASS**
- `all_live_actions_valid`: **PASS**
- `safe_observation_firewall`: **PASS**
- `single_seed_no_pseudo_replicates`: **PASS**
- `all_actions_change_prediction`: **PASS**
- `a2l_auc_above_categorical_ablation`: **FAIL**
- `a2l_auc_above_a2d`: **PASS**
- `a2l_auc_above_a3_median`: **PASS**
- `a2l_outer_mean_delta_at_least_0_005`: **FAIL**
- `a2l_outer_positive_on_at_least_3_folds`: **FAIL**
- `no_non_finite_class_metric`: **PASS**

## Boundary

No threshold, split, action, or result was changed after observing the pilot. The model remains disabled by default unless the preregistered retention gates pass.
