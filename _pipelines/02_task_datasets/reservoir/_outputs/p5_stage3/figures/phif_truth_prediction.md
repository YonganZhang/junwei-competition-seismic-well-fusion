# PHIF Stage-3 truth prediction

## Meta
- Target: `PHIF` (fraction)
- Development-only winner shown: `extra_trees_regressor`
- Aggregated OOF samples: 324
- Repeats per sample: 3
- Frozen test access: false

## Quantitative Summary

| Rank | Model | Mean RMSE | 95% bootstrap CI | Worst-fold RMSE | Seed SD |
|---:|---|---:|---:|---:|---:|
| 1 | extra_trees_regressor | 0.02764054 | [0.020942565, 0.035282479] | 0.049790291 | 0.00049340882 |
| 2 | hist_gradient_boosting_regressor | 0.029849851 | [0.02341162, 0.037431973] | 0.049962799 | 0 |
| 3 | lightgbm_regressor | 0.030378838 | [0.024350153, 0.037258453] | 0.049741288 | 3.469447e-18 |

## Visual Description
This panel is reconstructed from the portable mean-over-repeat OOF aggregate for PHIF; it contains no frozen-test prediction or metric.

## Boundary
The rank-1 development model has mean RMSE 0.02764054. Stage-3 confirmation is not a frozen-test claim.
