# P5 open-model Stage-3 acceptance evidence

> Date: 2026-07-14
> Scope: isolated branch `p5-model-benchmark-integration@9e5f501`; no merge to `master`, no push
> Interpretation: top-3 × three derived seeds × every scientifically valid development fold. Frozen test remains unopened.

## Acceptance summary

| Layer | Result | Boundary |
|---|---|---|
| Scientific execution | Five runnable tracks recorded 441 cells: 437 pass, 3 honest CatBoost failures and 1 PyKrige timeout. Fault has zero legal training folds and remains `not_rankable`. | Different datasets, tasks, modalities and strict/conditional modes remain separate boards. No missing cell was filled with a changed budget or temporary split. |
| Reproducibility | Root seed `2693`; repeat seeds `1867973658`, `2137841944`, `3902865753`; fold-local preprocessing/target transforms and immutable source/split/result hashes are archived. | Stage-2 candidate/configuration/budget was reused. Stage-3 did not perform HPO. |
| Test firewall | Every track summary reports no frozen-test archive/label/metric access. OOF manifests and specialized visualization manifests were independently read back. | These are development-CV results, not final blind-test claims. |
| Integration regression | Track-specific Stage-3 tests passed; full data/model regression was rerun by dependency group. | Shared environments were not modified. The old lithofacies mixed data+Torch test used a one-shot `legacy-isolated` environment. |

## Integrated commits

| Order | Track | Source commit | Integrated commit |
|---:|---|---|---|
| protocol | shared | — | `16bebd1` |
| 1 | fault | `c9ac3cf` | `2dbdcfa` |
| 2 | facies | `9a3bea5` | `85171a4` |
| 3 | property | `bcde22a` | `317ee5d` |
| 4 | lithofacies | `121f193` | `4bcef72` |
| 5 | sweetspot | `5a1fefe` | `5927558` |
| 6 | reconstruction | `b9cdab8` | `9e5f501` |

## Result matrix

| Track / task | Cells | Stage-3 winner and development metric | Acceptance boundary |
|---|---:|---|---|
| fault | 0 legal training cells | No winner; data-readiness figures only | No audited negative mask/unknown coverage, so no fabricated performance board. |
| facies / F3 | 45/45 pass | `smp_fpn_r18`, mean mIoU `0.131316` | F3 and Penobscot keep independent heads/manifests. |
| facies / Penobscot | 45/45 pass | `smp_deeplabv3plus_r18`, mean mIoU `0.132021` | OOF error/entropy/class metrics are dataset-specific. |
| property / PHIF | 36/36 pass | `extra_trees_regressor`, RMSE `0.027641`, R2 `0.8383` | Four development mother-well families, LOGO4. |
| property / KLOGH | 36/36 pass | `extra_trees_regressor`, RMSE `542.9026 mD`, R2 `0.3995` | Rankable but weak/unstable; do not describe as solved. |
| property / SW | 36/36 pass | `xgboost_regressor`, RMSE `0.170427`, R2 `0.7113` | Single MONAI 3D candidate remains a separate `not_rankable` lane. |
| lithofacies / GM09 P | 33/36 pass | `xgboost_multisoftprob_window`, fixed-nine macro-F1 `0.194938` | Three CatBoost fold-2 cells failed on NaN/Inf; completion `91.67%` remains above the frozen 80% ranking gate. |
| sweetspot / T1 | 27/27 pass | `lightgbm`, MAE `0.160379` | Per-target board only. |
| sweetspot / T2 | 27/27 pass | `catboost`, AP `0.984654` | Per-target board only. |
| sweetspot / T3 | 36/36 pass | `xgboost`, MAE `267.118`; worst fold `408.992` | Material fold instability is retained. |
| sweetspot / T4 | 27/27 pass | `catboost`, AP `0.654099`; worst fold `0.142853` | Highly unstable; mean score alone is not sufficient evidence. |
| sweetspot / T5–T7 | 0 training cells | T5 `not_feasible`; T6 PHIF/T7 KLOGH data-blocked | T6/T7 have labels but no development-only feature source; `test.h5` fallback is forbidden. |
| reconstruction / strict | 45/45 pass | `pykrige_ok3d`, RMSE `0.021207` | Buffered five-fold development CV. |
| reconstruction / conditional | 44/45 pass | `pykrige_ok3d`, RMSE `0.027932` | One honest 300 s timeout; lane remains rankable. |

The five runnable tracks therefore produced 441 attempted cells and 437 valid passes. Fault contributes one auditable gate record but no training cell.

## Independent regression evidence

- fault: Torch/P5 group `41 passed`; SEG-Y/data group `15 passed, 7 skipped, 2 subtests`.
- facies: full suite `47 passed, 1 skipped, 43 subtests`.
- property: full system-data suite `37 passed, 5 skipped`.
- lithofacies: P5 model suite `25 passed, 1 skipped, 26 subtests`; P4 HDF5 contract `8 passed, 4 skipped`; one-shot mixed legacy test `5 passed, 6 skipped, 3 subtests`.
- sweetspot: real-data contracts `21 passed`; P5 model suite `51 passed, 1 skipped`.
- reconstruction: full suite `55 passed, 10 skipped, 53 subtests`.

Skips above are explicit unavailable-real-artifact or declared integration gates. Missing packages in a deliberately narrow environment were not counted as code failures; commands were rerun in the documented dependency group without installing into the two shared environments.

## Decision after Stage-3

1. Freeze one winner per rankable task/lane using the archived Stage-3 board; retain worst-fold and stability warnings in the freeze record.
2. Only if a written preregistration justifies it, run a small Optuna search inside development CV around that winner. Otherwise skip HPO and refit directly on all development data.
3. Consume each frozen test exactly once through the existing lifecycle command; never use test metrics to reopen model selection.
4. Keep `p5-model-benchmark-integration` as the clean integration branch until the dirty `master` changes are attributed. Do not merge or push from an unattended run.
