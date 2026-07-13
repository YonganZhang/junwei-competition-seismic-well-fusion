# P5 open-model Stage-2 acceptance evidence

> Date: 2026-07-14
> Scope: isolated branch `p5-model-benchmark-integration@d46a7b5`; no merge to `master`, no push
> Interpretation: fixed-fold development pilots and structured scientific stops. These are not full CV, HPO or frozen-test rankings.

## Acceptance layers

| Evidence layer | Result | Boundary |
|---|---|---|
| Cross-track command gate | `torch-common`: 59 passed + 22 subtests. `tabular-cpu` applicable four-track suite: 40 passed. Stage-1 regression: 53 passed, 6 skipped, 77 subtests. | The tabular environment intentionally lacks `h5py`; reconstruction is verified in `torch-common`. |
| Real development journey | 140 preregistered cells produced 53 `development_piloted`/`PASS` results and 87 structured skip/blocked results; 0 failed, 0 timeout. | Every task used P4 fold 0, fixed budget and seed 2693. Small-fold scores are screening evidence only. |
| Test firewall / SSDO | Per-track portable JSONL/summary/leaderboards, result hashes, source locks, split hashes, CUDA lock evidence and clean commits were independently read back. | Frozen-test paths, labels and metrics were not accepted or consumed. Offline JSON plus immutable commits are the declared SSDO evidence. |

## Integrated commits

| Order | Track | Source commit | Integrated commit |
|---:|---|---|---|
| 1 | fault | `53db563` | `dd44c0a` |
| 2 | facies | `3c6c3b8` | `e98a870` |
| 3 | property | `cab82e5` | `df7f3a7` |
| 4 | lithofacies | `5302954` | `744c564` |
| 5 | sweetspot | `ad5fde8` | `a359ad3` |
| 6 | reconstruction | `15c4ae1` | `d46a7b5` |

## Result matrix and stop lines

| Track | Audited Stage-2 result | Development finding | Scientific boundary |
|---|---|---|---|
| fault | 10/10 `blocked/not_rankable`; 0 training updates | Engineering/data-gate evidence only | No audited negative mask or unknown coverage; no performance board. |
| facies | F3 6 pilots + 4 skips; Penobscot 6 pilots + 4 skips | F3 dev leader SMP-FPN R18 mIoU 0.128480; Penobscot dev leader SMP-DeepLabV3+ R18 mIoU 0.147194 | Separate label heads, manifests and boards; incomplete class support recorded, not resampled away. |
| property | 9 pilots + 1 license skip | Tabular dev leaders: PHIF ExtraTrees RMSE 0.022083/R2 0.880552; KLOGH LightGBM RMSE 227.877867 mD/R2 0.173771; SW LightGBM RMSE 0.117026/R2 0.822278 | Eight tabular candidates rank together; single MONAI 3D candidate is a separate `not_rankable` modality lane. |
| lithofacies | P lane 9 pilots; S lane 1 skip | XGBoost dev leader fixed-nine-class macro-F1 0.213580 | Supported-class macro-F1 is diagnostic only; fixed-nine-class score is primary. |
| sweetspot | 70 cells: 16 pilots + 54 skips | T1 LightGBM MAE 0.162587; T2 CatBoost AP 0.996217; T3 LightGBM MAE 193.223569; T4 three-way AP 0.95 tie | T4 validation n=5 is unstable. T5 remains not feasible. T6 PHIF/T7 KLOGH have versioned labels but no development-only feature source; `test.h5` fallback is forbidden. |
| reconstruction | strict 8 pilots + 2 skips; conditional 8 pilots + 2 skips | PyKrige OK3D dev RMSE: strict 0.024842, conditional 0.027240 | Modes remain independent. All 10 neural/operator pass records used `cuda:0` and `external flock -w 900`; no CPU neural result entered collation. |

## Stage-3 decision

- Eligible for Pareto screening: facies F3/Penobscot same-task candidates; property tabular lane; lithofacies P lane; sweetspot T1-T4; reconstruction strict/conditional.
- Remain blocked: fault, sweetspot T5, T6 and T7. Property MONAI 3D remains `not_rankable` until at least one comparable 3D candidate exists.
- Stage-3 may select at most three candidates per valid task/lane, then run three seeds over every scientifically valid fold. It must not force five folds when group/spatial support is smaller.
- No Stage-2 score authorizes frozen-test access. Only one configuration frozen after full development confirmation may consume the test once.
