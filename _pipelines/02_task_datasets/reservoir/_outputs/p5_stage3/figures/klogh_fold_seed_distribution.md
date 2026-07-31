# KLOGH Stage-3 fold seed distribution

## Meta
- Target: `KLOGH` (mD)
- Development-only winner shown: `extra_trees_regressor`
- Aggregated OOF samples: 324
- Repeats per sample: 3
- Frozen test access: false

## Quantitative Summary

| Rank | Model | Mean RMSE | 95% bootstrap CI | Worst-fold RMSE | Seed SD |
|---:|---|---:|---:|---:|---:|
| 1 | extra_trees_regressor | 542.90256 | [372.29251, 706.03936] | 967.62703 | 2.517119 |
| 2 | xgboost_regressor | 647.01641 | [466.31029, 827.2818] | 1067.03 | 2.3373669 |
| 3 | lightgbm_regressor | 654.39397 | [471.02535, 833.98492] | 1057.8455 | 0 |

## Visual Description
This panel is reconstructed from the portable mean-over-repeat OOF aggregate for KLOGH; it contains no frozen-test prediction or metric.

## Boundary
The rank-1 development model has mean RMSE 542.90256. Stage-3 confirmation is not a frozen-test claim.
