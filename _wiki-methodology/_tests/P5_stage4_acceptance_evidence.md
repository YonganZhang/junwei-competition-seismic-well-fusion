# P5 open-model Stage-4 acceptance evidence

> Date: 2026-07-14
> Scope: isolated branch `p5-model-benchmark-integration@5af968c`; no merge to `master`, no push
> Interpretation: frozen Stage-3 winners were refit on all legal development data and evaluated on holdouts already consumed by P4/history.

## Acceptance summary

| Layer | Result | Boundary |
|---|---|---|
| Command gate | `178` tests passed, `6` subtests passed and one explicitly gated old P4 real-data smoke skipped. | Shared environments were not modified. Reconstruction was run in `torch-common`; property/lithofacies/sweetspot in `tabular-cpu`. |
| Live/user journey | Five runnable tracks executed the real CLI lifecycle from full-development refit through known-holdout prediction, metrics and task-specific figures; fault executed the real fail-closed data gate. | This is an offline research pipeline, so the equivalent live journey is the real-data CLI lifecycle rather than an interactive UI. |
| Trace/SSDO audit | Single-use state, source/split/config/checkpoint/prediction/figure hashes, portable manifests, clean commits and independent leader verification were read back. | No online trace service exists for these offline jobs; persisted manifests plus immutable commits are the declared SSDO downgrade. |
| Scientific execution | Five runnable tracks completed full-development refit and known-holdout confirmation. Fault remained `blocked/not_rankable` with zero legal folds and no fabricated winner. | Every metric is `evidence_class=previously_seen_reusable_holdout`, `prior_test_consumed=true`, `fresh_blind=false`. |
| Frozen decisions | Stage-3 winner, lane, preprocessing, transform, seed and budget were bound before holdout access. No Stage-4 metric reopened ranking, HPO, loss, threshold or feature selection. | Confirmation performance may be reported, but it is not an independent blind-test claim. |
| Reproducibility | Root seed `2693`; source/split/config/checkpoint/prediction/figure hashes and single-use state are archived per track. P4 lifecycle state was not reset or overwritten. | Large runtime artifacts remain track-private where declared; portable manifests preserve path/hash/shape boundaries. |
| Visual acceptance | F3, Penobscot, PHIF, lithofacies confusion, sweetspot T2 and strict reconstruction figures were visually inspected. Fault reused only hashed Stage-3 data-gate figures. | Weak performance and calibration defects remain visible; no figure was selected to hide a failed lane. |

## Integrated commits

| Order | Track | Source commit | Integrated commit |
|---:|---|---|---|
| protocol | shared | — | `b529048` |
| 1 | fault | `27c9333` | `15e1580` |
| 2 | facies | `cc1e615` | `7b1af1d` |
| 3 | property | `6766773` | `431e4c7` |
| 4 | lithofacies | `29600b3` | `a17de80` |
| 5 | sweetspot | `b2fd6aa` | `5e4f19a` |
| 6 | reconstruction | `e783066` | `5af968c` |

## Known-holdout result matrix

| Track / task | Frozen winner | Known holdout | Stage-4 result | Acceptance boundary |
|---|---|---:|---|---|
| fault | none | not accessed | `blocked/not_rankable`; zero refit, prediction and metric operations | Missing audited negatives/unknown coverage and legal buffered folds; no score manufactured. |
| facies / F3 | `smp_fpn_r18` | 445 slices / 7,290,880 pixels | Accuracy `0.238874`; mIoU `0.126495`; Macro-F1 `0.217813`; ECE `0.251950` | All ten classes supported, but accuracy and calibration remain weak. |
| facies / Penobscot | `smp_deeplabv3plus_r18` | 474 slices / 7,766,016 pixels | Accuracy `0.528572`; mIoU `0.125380`; Macro-F1 `0.170917`; ECE `0.293934` | Accuracy is dominated by class imbalance; mIoU/Macro-F1 remain the safer interpretation. |
| property / PHIF | `extra_trees_regressor` | F-15, 344 rows | MAE `0.006067`; RMSE `0.009319`; R2 `0.980711` | Previously seen F-15 only; interval coverage `0.994186` is diagnostic, not fresh calibration evidence. |
| property / KLOGH | `extra_trees_regressor` | F-15, 344 rows | physical MAE `145.640 mD`; RMSE `278.839 mD`; R2 `0.889133` | Very wide physical interval (`2879.75 mD` mean width) is retained; do not describe permeability uncertainty as solved. |
| property / SW | `xgboost_regressor` | F-15, 344 rows | MAE `0.064412`; RMSE `0.080550`; R2 `0.903155` | Previously seen family holdout; no post-confirmation retuning. |
| lithofacies / GM09 P | `xgboost_multisoftprob_window` | F-5, 120 rows | Accuracy `0.416667`; fixed-nine Macro-F1 `0.189153`; ECE `0.141489` | Supported-class Macro-F1 `0.243197` is diagnostic only; depth track remains `not_feasible`. |
| sweetspot / T1 reservoir quality | `lightgbm` | 11,936 rows | MAE `0.216162`; RMSE `0.279330`; R2 `0.896824`; Spearman `0.993129` | Target-specific regression board only. |
| sweetspot / T2 hydrocarbon pay | `catboost` | 12,081 rows | AP `0.999078`; Brier `0.022861`; F1@0.5 `0.977726`; net-thickness MAE `26.2 m` | Threshold remained frozen at 0.5; no holdout fit. |
| sweetspot / T3 productivity | `xgboost` | 132 rows | MAE `47.7020`; RMSE `68.3353`; R2 `-0.063574`; Spearman `0.635569` | Negative R2 is retained; top-k hit `0.428571` is diagnostic only. |
| sweetspot / T4 water breakthrough | `catboost` | 37 rows | AP `0.131463`; Brier `0.745222`; F1@0.5 `0.195122` | Small and weak holdout result; no claim of solved prediction. |
| sweetspot / T5 | none | not accessed | `not_feasible` | Missing approved labels/simulator-state contract. |
| sweetspot / T6 PHIF, T7 KLOGH | none | not accessed | `blocked` | No development-only feature source; `test.h5` fallback remained forbidden. |
| reconstruction / strict | `pykrige_ok3d` | 78,949 metric voxels | RMSE `0.035625`; MAE `0.027173`; R2 `-0.339679`; Pearson `-0.120046` | Strict result is poor and visibly over-smoothed; failure is not hidden. |
| reconstruction / conditional | `pykrige_ok3d` | 49,233 metric voxels | RMSE `0.021013`; MAE `0.015088`; R2 `0.013686`; Pearson `0.139364` | Conditional mode uses test-region constraints and is not strict holdout generalization. |

## Facies single-use recovery record

The first facies confirmation attempt raised a preprocessor deserialization/setup exception after
`TEST_ACCESS_STARTED`. The worker did not delete or reset that state, retrain, change the winner or
reuse partial metrics. A narrowly bound `resume-incomplete` path reused the exact checkpoint and
recorded that labels may already have been read. The final state is `CONFIRMATION_COMPLETE`, and the
recovery metadata explicitly records one recovery, no configuration change and no failed-attempt
prediction/metric reuse.

## Independent regression evidence

- fault Stage-3 + Stage-4: `22 passed`.
- facies full suite: `58 passed, 1 skipped`; skip requires the explicit old P4 real-data environment variable.
- property + lithofacies + sweetspot Stage-3/4 combined in `tabular-cpu`: `77 passed, 6 subtests passed`.
- reconstruction Stage-3/4 in `torch-common`: `21 passed`.
- facies artifact verifier: `13` portable artifacts and `2` runtime tasks, all hashes verified; machine paths absent.

One initial combined `tabular-cpu` collection included reconstruction and stopped because that environment lacks
`h5py`. No package was installed into the shared environment. The scientifically appropriate dependency groups
above were then run independently and all applicable tests passed.

## Decision after Stage-4

1. Preserve the frozen winners and Stage-4 outputs as known-holdout confirmation evidence; do not select a new model from these metrics.
2. Treat fault, sweetspot T5/T6/T7 and lithofacies continuous-depth output as unresolved data/contract gates.
3. Obtain a genuinely untouched external or organizer-hidden test set before claiming fresh-blind generalization.
4. Keep `p5-model-benchmark-integration` isolated until the dirty `master` changes are attributed; do not merge or push from this unattended run.
