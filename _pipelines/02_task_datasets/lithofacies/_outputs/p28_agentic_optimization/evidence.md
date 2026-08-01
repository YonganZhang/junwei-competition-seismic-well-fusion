# P28 lithofacies Stage-1 execution-agent pilot evidence

## Outcome

The preregistered verdict is **REJECT**. A2L promotion changed fixed-schema nine-class Macro-F1 from `0.2133487970` to `0.2062263168` (`-0.0071224802`).

This is bounded adaptive development evidence, not a frozen-holdout estimate.

## Frozen contract and nested split

- A0: `depth3_eta01_rounds60`, frozen reference `0.2133487970`.
- Split SHA-256: `a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555`.
- Four outer anonymous promotion rotations; each remaining three folds form inner LOGO3 selection.
- Each execution-policy instance used exactly 3 actions without replacement and the same three model seeds.
- Promotion results were evaluated only after inner selection and were never returned to a policy.

## Agent and controls

| arm | role | normalized selection AUC@3 | promotion mean | delta vs A0 | positive outer folds |
|---|---|---:|---:|---:|---:|
| A2L | live_deepseek_execution_policy | 0.0020460825 | 0.2062263168 | -0.0071224802 | 0/4 |
| A2D | deterministic_diagnostic_policy | 0.0020460825 | 0.2062263168 | -0.0071224802 | 0/4 |
| A4 | fixed_offline_replay_control | 0.0037083983 | 0.2074603254 | -0.0058884717 | 1/4 |
| A3 | equal-budget random median (3 policy seeds) | 0.0024581503 | 0.1997713891 | -0.0135774079 | n/a |

## A2L per-fold observables

| outer fold | selected action | A0 Macro-F1 | A2L Macro-F1 | delta |
|---:|---|---:|---:|---:|
| 0 | ACT_WEIGHT_EXP05_MEAN1 | 0.2413555052 | 0.2215816002 | -0.0197739049 |
| 1 | A0_DEPTH3_ETA01_ROUNDS60 | 0.1967480000 | 0.1967480000 | +0.0000000000 |
| 2 | ACT_WELL_MASK_ONLY_858 | 0.1837172826 | 0.1750012667 | -0.0087160159 |
| 3 | A0_DEPTH3_ETA01_ROUNDS60 | 0.2315744004 | 0.2315744004 | +0.0000000000 |

## A2L per-class observable

| class id | A0 mean F1 | A2L mean F1 | delta |
|---:|---:|---:|---:|
| 0 | 0.0000000000 | 0.0000000000 | +0.0000000000 |
| 1 | 0.3754058328 | 0.3751693102 | -0.0002365225 |
| 2 | 0.0000000000 | 0.0000000000 | +0.0000000000 |
| 3 | 0.1842948718 | 0.1437246964 | -0.0405701754 |
| 4 | 0.0000000000 | 0.0000000000 | +0.0000000000 |
| 5 | 0.5545427021 | 0.5371040354 | -0.0174386666 |
| 6 | 0.8058957668 | 0.8000388095 | -0.0058569573 |
| 7 | 0.0000000000 | 0.0000000000 | +0.0000000000 |
| 8 | 0.0000000000 | 0.0000000000 | +0.0000000000 |

## Preregistered gates

- `a1_prediction_hash_equals_a0`: **PASS**
- `a1_metric_hash_equals_a0`: **PASS**
- `valid_action_rate_100_percent`: **PASS**
- `a2l_auc_above_a3_median`: **FAIL**
- `a2l_auc_above_a2d`: **FAIL**
- `a2l_positive_on_at_least_3_of_4_outer_folds`: **FAIL**
- `a2l_mean_promotion_delta_at_least_0_005`: **FAIL**
- `no_non_finite_class_metric`: **PASS**

## Leakage and attribution audit

- The live DeepSeek policy received only anonymous train-support buckets, `underfit|balanced|overfit`, action IDs/descriptions, and `improved|flat|worse` feedback.
- It never received a raw metric, class label, residual, family identity, sample identifier, filesystem path, or promotion result.
- `DEEPSEEK_KEY` was process-local and is absent from all artifacts.
- Only `train.h5` and the accepted development LOGO4 batch were loaded; frozen holdout and `test.h5` were not accessed.
- MOMENT pretrained/random remains a separate frozen paired-attribution lane and is excluded from the P28 agent AUC and promotion effect. 大模型贡献占比待下一轮消融确认。

## Interpretation

The P28 policy is retained only if every preregistered gate passes. No post-hoc action, threshold, or trial was added after observing results.
