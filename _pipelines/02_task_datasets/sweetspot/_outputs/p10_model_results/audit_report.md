# Sweetspot P10 model-results audit

## Conclusion

- T1/T2/T6/T7 retain their baseline-only results; no foundation route is claimed there.
- T3 Chronos-2 blend improves on the archived XGBoost baseline on the same development folds and remains the promoted foundation row.
- T4 Chronos-2 water-risk route does not beat the archived CatBoost baseline; it stays rejected / non-beneficial.
- T5 remains data-blocked / not feasible; no synthetic label or pseudo-split was introduced.
- The exact-calendar p8 Chronos diagnostic supports the contract audit but is not the promoted model row because the preregistered same-architecture random-init control is still missing.

## Before / after primary metric

### T3 productivity (MAE, lower is better)

| fold | baseline XGBoost | Chronos blend | Δabs | Δpct |
|---|---:|---:|---:|---:|
| fold 0 | 217.227 | 115.903 | 101.324 | 46.6% |
| fold 1 | 408.992 | 244.087 | 164.905 | 40.3% |
| fold 2 | 253.147 | 241.075 | 12.072 | 4.8% |
| fold 3 | 189.107 | 145.222 | 43.885 | 23.2% |

### T4 water breakthrough (average precision, higher is better)

| variant / split | baseline CatBoost | Chronos risk | Δabs | Δpct | note |
|---|---:|---:|---:|---:|---|
| dev fold 0 | 0.950 | 0.804 | -0.146 | -15.4% | cached dev/CV summary evidence |
| dev fold 1 | 0.869 | 0.567 | -0.303 | -34.8% | cached dev/CV summary evidence |
| dev fold 2 | 0.143 | 0.157 | 0.014 | 9.6% | cached dev/CV summary evidence |
| dev/CV macro | 0.654 | 0.509 | -0.145 | -22.2% | cached summary; rejected_no_gain |
| known-holdout confirmation | 0.654 | 0.131 | -0.523 | -79.9% | evidence-only; prior_test_consumed=true |
| chronos2_median_quantile repair | 0.654 | — | — | — | cached dev/CV summary evidence; macro_ap=blocked |
| chronos2_history_gate repair | 0.654 | — | — | — | cached dev/CV summary evidence; macro_ap=blocked |

## Root cause / fix applied

- No model-code defect was evidenced in the archived result set.
- The cached T4 dev/CV experiment is below the archived CatBoost baseline on folds 0 and 1 and at macro mean; fold 2 improves marginally, but not enough to change the rejected-no-gain decision.
- The p8 exact-calendar diagnostic is correctly kept separate from the promoted T3 row because it still lacks the preregistered same-architecture random-init control.
- The T4 Chronos route is non-beneficial versus CatBoost on the known holdout; that is a scientific outcome, not a pipeline bug.
- Median-quantile and history-gate repair variants are explicitly blocked because no archived dev/CV run is materialized for them.
- T5 is data-blocked because the required labels are not frozen.

## Files / tests / commit

- workbook: `_pipelines/02_task_datasets/sweetspot/_outputs/p10_model_results/track_model_metrics.xlsx`
- figure: `_pipelines/02_task_datasets/sweetspot/_outputs/p10_model_results/before_after_primary_metric.png`
- model rows: `41`
- T3 rows: `5`
- T4 rows: `7`
- T5 rows: `1`

## Residual risk

- The report is only as complete as the archived evidence set; no new training or split changes were introduced.
- Foundation promotion for T3 still depends on the missing same-architecture random-init control if future proofing is required.
- T5 remains blocked until the label contract itself is frozen.
