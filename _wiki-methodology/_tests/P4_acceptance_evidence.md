# P4 training/evaluation baseline acceptance evidence

> Date: 2026-07-13
> Scope: isolated branch `p4-training-integration`; no merge to `master`, no push
> Interpretation: this accepts the reproducible training/evaluation **engineering baseline**. It does not claim that every simple model has useful scientific accuracy.

## Three evidence layers

| Evidence layer | Result | Boundary |
|---|---|---|
| Command gate | Shared 20; fault 29 (1 optional integration skip); facies 22 (1 skip); property 13 (4 skips); lithofacies 23 (7 skips); sweetspot 21; reconstruction 24 (2 skips). All executed suites exited 0. | Skips are explicit unavailable-data/artifact gates, not silent passes. |
| Live/user journey | Explicit real smokes passed for fault preflight, F3, Penobscot, property, lithofacies and reconstruction. Feasible cases then ran real CV/refit/single-use test/visualization as listed below. | This is a research pipeline, not an interactive UI; the equivalent live journey is the real-data CLI lifecycle. |
| Trace/SSDO audit | Lifecycle JSON, seed/environment report, split/config/checkpoint hashes, archived predictions/metrics/figures and manifests were read back. Hash audit: 7 manifests / 144 artifacts and 50 registry references verified. | No runtime trace service exists for these offline jobs; immutable lifecycle/artifact evidence is the declared SSDO downgrade. |

## Accepted engineering contracts

- Six canonical model namespaces live under `_models/{fault,facies,property,lithofacies,reconstruction,sweetspot}`; legacy track model paths are compatibility shims only.
- Shared `ModelBatch -> ModelOutput` envelope, while track-specific shapes, heads, masks, losses, inference activations and metrics remain explicit.
- Root seed `2693` fans out to Python/NumPy/framework/loaders/splitters and is persisted with environment evidence.
- Group/spatial split manifests are fixed before preprocessing; test data is excluded from HPO, threshold and calibration selection.
- Weighted reducers count samples or valid labels; checkpoints persist model/optimizer/scheduler/scaler/RNG/config/split state and validate resume envelopes.
- Lifecycle is `SPLIT_LOCKED -> SMOKE_PASSED -> CV_COMPLETE -> CONFIG_FROZEN -> REFIT_COMPLETE -> TEST_CONSUMED`; frozen test is single-use.
- Visualizers read archived predictions and metrics only. Artifact manifests bind config, split, checkpoint, prediction, metric and figure hashes.
- Optional HPO planning/backend is implemented for every plugin. Actual 8-sanity + 20-pilot selection was run for sweetspot targets 1–4; long neural HPO and top-3×3-seed campaigns were not run.

## Six-track evidence matrix

| Track | Real-data acceptance | Frozen-test result / status | Visualization | Honest boundary |
|---|---|---|---|---|
| ① fault | 3998 official fault points and real seismic coordinate/index preflight; tiny baseline smoke passed | formal blind test and CV `not_feasible`, requested 5 / effective 0 | entrypoint and contract tests pass | no audited coverage negatives; random non-fault patches are forbidden as fake negatives |
| ② facies / F3 | 5-fold OOF on 1548 development sections; 1-epoch refit; test opened once | 445 test sections; accuracy 0.1750, mIoU 0.02994, Macro-F1 0.05207 | archived-only class/confusion/calibration outputs generated | simple linear pixel baseline is very weak |
| ② facies / Penobscot | 5-fold OOF on 1398 development sections; 1-epoch refit; test opened once | 474 test sections; accuracy 0.4289, mIoU 0.07773, Macro-F1 0.11467 | archived-only outputs generated | simple linear pixel baseline is weak |
| ③ property | real P4 integration path passed; PHIF/KLOGH are independently trained below | PHIF and KLOGH frozen-test evidence recorded in targets 6/7 | per-target prediction/uncertainty artifacts | exact PHIE independently `not_feasible`; never substitute LFP_PHIE |
| ④ lithofacies | requested 5 -> effective 4 family folds; OOF 447 samples; refit on all development | frozen F-5: 120 samples, accuracy 0.2583, fixed-nine-class Macro-F1 0.05904, supported-class Macro-F1 0.07591 | confusion, per-class and calibration figures generated | depth track `not_feasible` because archived predictions lack real `center_md_m`; no interval midpoint fabrication |
| ⑥ reconstruction strict | 5 buffered spatial folds; 1-epoch ridge refit; test opened once | 78,949 voxels; RMSE 0.20962, MAE 0.20735, R² -45.385 | three planes, residual distribution and spectrum generated | no test-region constraints; result shows severe underfit |
| ⑥ reconstruction conditional | independent 5-fold case; 1-epoch ridge refit; test opened once | 49,233 voxels; RMSE 0.22435, MAE 0.22335, R² -111.432 | same archived diagnostics with conditional caveat | uses 90 test-region well constraints and is not strict holdout generalization; severe underfit |

## ⑤ Sweetspot: seven independent target cases

| No. | Task / truth scope | Frozen-test evidence | Status / interpretation |
|---:|---|---|---|
| 1 | reservoir quality, frozen RQI-style proxy | n=11,936; MAE 0.3842, RMSE 0.6593, R² 0.4252, Spearman 0.9177 | `proxy_feasible`; not direct reservoir quality truth |
| 2 | hydrocarbon pay / effective thickness, `SAND_FLAG` proxy | n=12,081; AP 0.9989, Brier 0.03120, F1 0.9728; net-thickness MAE 46.4 m | `proxy_feasible`; high sample metric must not be presented as hydrocarbon truth |
| 3 | future 30-day oil productivity | n=132; MAE 117.16, RMSE 131.51, R² -2.9388, Spearman 0.4307, top-k 0.3571 | feasible workflow, weak model |
| 4 | 7-day event / 30-day water-breakthrough risk proxy | n=37; AP 0.1554, Brier 0.2635, F1 0.2439 | `proxy_feasible`, tiny and very weak test |
| 5 | remaining oil / infill simulation case | no metrics | `not_feasible`: no tested Eclipse cell-state parser; realization/time/candidates/spacing/economics not frozen; never field truth |
| 6 | porosity PHIF | n=344 frozen test; MAE 0.01206, RMSE 0.01722, R² 0.9341; 90% residual interval coverage 0.9593 | complete; requested 5 -> effective 4 mother-family folds |
| 7 | permeability KLOGH trained in `log1p` domain | n=344; log R² 0.8687; physical MAE 226.48 mD, RMSE 544.60 mD, R² 0.5771 | complete; requested 5 -> effective 4 mother-family folds |

Registry: `_pipelines/02_task_datasets/sweetspot/targets/_outputs/registry_targets_1_to_7.json`. The hash audit verified 7 artifact manifests / 144 referenced artifacts and 50 registry references; `all_targets_independent=true`.

## What is complete vs still pending

Complete now:

- common reproducibility/SOP modules, canonical model discovery and track adapters;
- portable unit/contract/lifecycle suites and explicit real-data smokes;
- real group/spatial CV and single-use frozen-test campaigns where labels support them;
- track-specific archived visualizations, or explicit `not_feasible` evidence;
- seven sweetspot target statuses with independent task IDs and artifacts.

Still pending for P5:

- user-authorized merge/push from the isolated worktree;
- formal deep/multimodal architectures and performance targets;
- long-budget HPO, top-three configurations across three seeds and full-fold replication;
- new label/evidence acquisition for fault negatives, target 5 and exact PHIE;
- improvements for the weak facies, lithofacies, productivity, water-risk and reconstruction baselines.
