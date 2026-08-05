# P38 real-well direct-seismic PHIF fusion

Decision: `FEASIBLE_NO_PROMOTION`.

## Phase 0

Native published CPI PHIF remains distinct from PHIE and Eclipse PORO. All finite physical PHIF zeros were retained under the frozen non-performance rule.

| parent | physical PHIF | retained joint rows | coverage |
|---|---:|---:|---:|
| 15/9-19 | 1967 | 1962 | 99.745806% |
| 15/9-F-11 | 2017 | 1877 | 93.058999% |
| 15/9-F-15 | 9668 | 8846 | 91.497724% |

## Fixed LOGO3 result

| model | 15/9-19 RMSE | 15/9-F-11 RMSE | 15/9-F-15 RMSE | macro RMSE |
|---|---:|---:|---:|---:|
| well_only | 0.069284112717 | 0.064867699667 | 0.091791486719 | 0.075314433034 |
| seismic_only | 0.157498050272 | 0.074869648579 | 0.133676796163 | 0.122014831671 |
| raw_feature_fusion | 0.161661156974 | 0.037296788222 | 0.097092889155 | 0.098683611450 |
| same_architecture_random_init_moment_gfm_fusion | 0.119581656067 | 0.131170308770 | 0.144254927563 | 0.131668964134 |
| frozen_pretrained_moment_gfm_fusion | 0.094105663897 | 0.062553391376 | 0.082684630269 | 0.079781228514 |

Strongest budget-matched control: `well_only` with macro RMSE `0.075314433034`; frozen pretrained fusion is `0.079781228514` and same-architecture random-init is `0.131668964134`.

Paired 20 m depth-block bootstrap fusion-minus-control CI95: `[0.000535576019, 0.008768653251]`.

## Alignment and agent checks

Cyclic-well mismatch macro delta: `0.012225937720`; fixed +160 ms delta: `0.000458342539`.
Agent-selected-minus-fixed-default macro RMSE delta: `-0.008108196853`. Actions were selected only from outer-train inner-LOGO evidence.

## Boundary

P21/P30 remain Eclipse-PORO history and were not ranked against native PHIF. This three-parent Volve pilot does not establish field-wide generalization and does not disprove traditional geostatistics.

Exact commands are in `rerun_commands.json`; row-aligned evidence is in `predictions.npz`, and all durable hashes are in `artifact_manifest.json`.
