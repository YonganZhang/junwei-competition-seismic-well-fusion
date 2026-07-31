# SW Stage-3 truth prediction

## Meta
- Target: `SW` (fraction)
- Development-only winner shown: `xgboost_regressor`
- Aggregated OOF samples: 324
- Repeats per sample: 3
- Frozen test access: false

## Quantitative Summary

| Rank | Model | Mean RMSE | 95% bootstrap CI | Worst-fold RMSE | Seed SD |
|---:|---|---:|---:|---:|---:|
| 1 | xgboost_regressor | 0.17042689 | [0.15212235, 0.19165924] | 0.22622618 | 0.0012507703 |
| 2 | lightgbm_regressor | 0.1734948 | [0.14603691, 0.20549969] | 0.2639328 | 0 |
| 3 | hist_gradient_boosting_regressor | 0.18234953 | [0.15331398, 0.21567298] | 0.26942707 | 0 |

## Visual Description
This panel is reconstructed from the portable mean-over-repeat OOF aggregate for SW; it contains no frozen-test prediction or metric.

## Boundary
The rank-1 development model has mean RMSE 0.17042689. Stage-3 confirmation is not a frozen-test claim.
