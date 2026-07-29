# ④ Volve lithofacies — real multimodal baseline

This track predicts the nine explicit depositional-facies classes in the eleven
Volve `Facies.xlsx` workbooks. It is a deliberately small pipeline baseline,
not a model-architecture upgrade.

## Frozen truth contract

The only target rows satisfy both:

- `* Source == GM09`
- `* Litho Crv Type == GENETIC FACIES`

The target is the interval field `Litho Class`, located by `Common Well Name`,
`* Top Depth (meters)`, and `* Base Depth (meters)`. The real scan contains 11
workbooks, 11 borehole tracks, 139 intervals, and these fixed IDs:

| ID | Exact class |
|---:|---|
| 0 | `F-MARSH` |
| 1 | `F-MOUTHBAR` |
| 2 | `F-OFFSHORE` |
| 3 | `F-LOWER SHOREFACE` |
| 4 | `F-UPPER SHOREFACE` |
| 5 | `F-TIDAL BAR` |
| 6 | `F-TIDAL CHANNEL` |
| 7 | `F-TIDAL FLAT MUDDY` |
| 8 | `F-TIDAL FLAT SANDY` |

`LITH`, `UNKNOWN`, `UNDEFINED`, interval-exterior depths, GR/VSH/PHIE
thresholds, RMS realizations, formations, and synthetic labels are never
targets.

## Leakage and split contract

Mother-well families are assigned before resampling, windowing, or fitting any
normalization statistic:

| Partition | Frozen mother families | Final usable families |
|---|---|---|
| train | `15/9-19`, `15/9-F-12`, `15/9-F-14`, `15/9-F-15` | `15/9-19`, `15/9-F-14`, `15/9-F-15` |
| guard | `15/9-F-4` | `15/9-F-4` |
| test | `15/9-F-5` | `15/9-F-5` |

Every F-15 sidetrack stays with F-15, and the 19 A/BT2/SR tracks stay in one
family. Guard samples are written into `lithofacies/train` with
`meta.partition=guard` because the shared dataset interface exposes only
`train/test`; the optimizer filters guard samples out and uses them only for
validation/checkpoint selection. Test is not loaded until the best guard-loss
checkpoint has been selected.

## Eleven-workbook audit to final intersection

The builder reads LAS members directly from the 7 GB ZIP without extracting
copies. Only depth-domain raw/basic measurements are eligible. F-12 is retained
in the audit but excluded because all four allowed LAS runs end above its GM09
intervals. One F-15B center lies beyond the last fully observed official
MD/TWT/XY pick and is excluded rather than extrapolated.

| Label well | Family / split | GM09 intervals | Allowed LAS | Candidate centers | Kept multimodal samples | Result |
|---|---|---:|---:|---:|---:|---|
| 15/9-19 A | 19 / train | 15 | 1 | 51 | 51 | usable |
| 15/9-19 BT2 | 19 / train | 16 | 1 | 71 | 71 | usable |
| 15/9-19 SR | 19 / train | 6 | 1 | 10 | 10 | usable |
| 15/9-F-12 | F-12 / train | 15 | 4 | 76 | 0 | no allowed LAS at label depth |
| 15/9-F-14 | F-14 / train | 16 | 3 | 101 | 101 | usable |
| 15/9-F-15 | F-15 / train | 14 | 3 | 55 | 55 | usable |
| 15/9-F-15 A | F-15 / train | 13 | 2 | 48 | 48 | usable |
| 15/9-F-15 B | F-15 / train | 3 | 1 | 6 | 5 | one center outside pick bracket |
| 15/9-F-15 C | F-15 / train | 13 | 2 | 19 | 19 | usable |
| 15/9-F-4 | F-4 / guard | 13 | 5 | 87 | 87 | usable |
| 15/9-F-5 | F-5 / test | 15 | 3 | 120 | 120 | usable |

Final sample counts are train 360, guard 87, and test 120.

## Real inputs and preprocessing

The log whitelist contains 13 measurement concepts: gamma ray, acoustic
slowness, caliper, bulk density, neutron porosity, deep resistivity, ROP, WOB,
RPM, flow, torque, standpipe pressure, and ECD. It accepts only explicit
mnemonic aliases declared in `pipeline_contract.py`. For the 19 A/BT2 tracks,
the only LAS source is the LFP package, so only its base GR/DT/CALI/RHOB/NPHI/RT
curves are admitted; all LFP model, mineral, sand, VSH, porosity-interpretation,
fluid, and synthetic curves are excluded. SR uses its independent composite
LAS. Raw F-well runs use only depth-domain MWD files; time-domain runs and
petrophysical/facies interpretation files are excluded.

Each center receives a `13×33` physical log window and a `13×33` observed mask.
Missing values are represented by normalized value zero plus mask zero, so the
stored `well_log_seq` has shape `26×33`. No smoothing is applied: the builder
calls shared `denoise_identity` explicitly.

The builder uses official `Well_picks_Volve_v1.dat` points only. MD is mapped
to TWT and XY by bracketed linear interpolation; it never extrapolates. XY is
converted through the measured Layer1 ST0202 affine index, then a true
`3×3×33` seismic patch is lazily read from the 1.1 GB SEG-Y.

Per-channel and seismic z-score statistics are fitted only on samples from
training mother families, then applied unchanged to guard/test. Shared
`NormStats`, `normalize`, and `denormalize` give a measured maximum round-trip
error of `2.842170943040401e-14`.

## Build and inspect

From the project root:

```bash
python3 _pipelines/02_task_datasets/lithofacies/build_dataset.py
python3 _code/dataset_io.py stats lithofacies/train
python3 _code/dataset_io.py stats lithofacies/test
```

Measured stats:

```text
lithofacies/train: n_samples=447, labels={0:11,1:104,2:7,3:40,4:27,5:124,6:127,7:6,8:1}
lithofacies/test:  n_samples=120, labels={0:2,1:31,2:2,3:17,4:3,5:24,6:41}
```

The train count is 360 optimizer samples plus 87 guard samples. The held-out
F-5 test family genuinely lacks classes 7 and 8; their support remains zero in
the fixed nine-class report.

## Environment, tests, and training

System Python already supplies NumPy, LAS, SEG-Y, HDF5, Excel, and plotting
dependencies. Create a local ignored PyTorch environment if needed:

```bash
uv venv --python python3 \
  --system-site-packages _pipelines/02_task_datasets/lithofacies/.venv
uv pip install \
  --python _pipelines/02_task_datasets/lithofacies/.venv/bin/python3 \
  --torch-backend cu128 torch
source _pipelines/02_task_datasets/lithofacies/.venv/bin/activate
```

The default no-artifact unit gate is safe before building. Artifact-dependent
tests are an explicit integration gate and skip when HDF5/checkpoint/history
artifacts are absent:

```bash
python3 -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests -v
```

Canonical reproducibility order is **build → train → integration audit**:

```bash
python3 _pipelines/02_task_datasets/lithofacies/build_dataset.py

python3 _pipelines/02_task_datasets/lithofacies/train_baseline.py \
  --model multimodal_mlp --epochs 80 --device cpu

LITHOFACIES_RUN_INTEGRATION=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests -v

python3 _pipelines/02_task_datasets/lithofacies/audit_pipeline.py
```

`models/multimodal_mlp.py` is dynamically imported and registered by name. It
uses two shallow MLP encoders and one fusion head. The script delegates the
entire epoch loop to shared `train_loop` using zero-argument DataLoader
factories; it saves `last.ckpt` every epoch and selects `best.ckpt` by minimum
guard loss.

Available swappable models follow the same two-input interface and are
dynamically discovered from a file whose name equals its registered name:

| Registered name | Architecture | Evidence status |
|---|---|---|
| `multimodal_mlp` | two shallow encoders plus fusion head | measured baseline below |
| `lithofacies_concat_linear` | flattened log and seismic tensors concatenated into one linear classifier | contract-tested only |
| `lithofacies_late_fusion` | independent shallow encoders followed by concatenated classification | contract-tested only |

The alternatives are interface baselines, not new reported experiments. They
have no formal train/test metrics in this repository; the measured results and
artifacts below remain exclusively those of `multimodal_mlp`.

## Measured honest baseline

Seed 2693, batch size 64, Adam `1e-3`, hidden size 64, CPU, 80 complete epochs:

| Best epoch | Best guard loss | Accuracy | Balanced accuracy (7 supported test classes) | Fixed-9 macro-F1 |
|---:|---:|---:|---:|---:|
| 2 | 1.843473 | 0.350000 | 0.209677 | 0.101521 |

The model predicts only mouthbar and tidal-bar classes on F-5. This is poor but
honest cross-family generalization, not a polished score. Train loss falls to
about 0.19 while guard loss rises above 11, so the complete curve exposes
severe overfitting and supports the early best checkpoint.

Artifacts:

- `_outputs/split_manifest.json`: full label/LAS/pick/split audit and every exclusion.
- `_outputs/normalization_stats.json`: train-family-only reversible statistics.
- `_outputs/multimodal_mlp/metrics.json`: all requested aggregate/per-class metrics and support.
- `_outputs/multimodal_mlp/confusion_matrix.png`: fixed-nine-class confusion matrix.
- `_outputs/multimodal_mlp/loss_curve.png`: all 80 train/guard losses and best epoch.
- `_outputs/multimodal_mlp/best_checkpoint_predictions.png`: real logs, masks, ST0202 patches, GT, and predictions.
- `_outputs/multimodal_mlp/run_manifest.json`: exact interpreter/arguments and checkpoint-selection contract.
- `_outputs/completion_audit.json`: content-level anti-fake PASS/FAIL audit over data and run artifacts.

## Residual risks

- Only five of six frozen mother families have a usable real multimodal
  intersection; F-12 needs a non-LAS parser/source extension and is not faked.
- The 19 A/BT2 base measurements are packaged in LFP LAS rather than raw MWD
  LAS. The whitelist prevents target-derived curves, but provenance differs
  from the F-well drilling logs.
- The 13-channel mask pattern varies strongly between well families, and the
  small sample/class distribution is highly imbalanced.
- The official well-pick mapping is sparse weak alignment, not a checkshot or
  synthetic-seismogram tie. Samples outside the observed pick bracket are
  rejected.
- The held-out test family has zero support for two frozen classes. Metrics
  report this directly; no same-well random split is used to fill the gap.

## P4 training, validation, and reproducibility SOP

The P4 implementation is a track-private plugin over the read-only shared
contracts in `_code/ml_framework`. It does not change the accepted GM09 label
schema or the measured baseline above. The existing models remain available
through a strict `TaskSpec` / `ModelBatch` / `ModelOutput` adapter because this
integration batch does not authorize writes to the canonical `_models/` tree.

### Frozen split and fold-local fitting

- F-5 remains the frozen test mother family.
- The requested fold count is five, but only four independent development
  families have real multimodal samples: 15/9-19, F-14, F-15, and F-4.
  P4 therefore records `effective_n_splits=4` and runs leave-one-family-out.
  It never splits rows from one mother family to manufacture a fifth fold.
- Every fold archives train/validation class support. When F-15 is held out,
  class 8 is absent from fold-train. The model still emits nine logits, its
  fold loss weight is zero for that unseen class, and both fixed-nine and
  observed-support metrics remain visible.
- Existing HDF5 values are first reversed with their stored statistics. Log,
  seismic, and class-weight statistics are then fitted only on the current
  fold-train families. A validation-only log channel is masked rather than
  normalized with validation data.
- New builds persist the actual `center_md_m` sampling coordinate. Legacy
  archives without it produce a `not_feasible_depth_facies_track.json`; an
  interval midpoint is never substituted.

### Lifecycle and commands

Use a run directory below this track, for example
`_pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1`. Commands have
separate responsibilities and must be executed in order:

```bash
python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py prepare \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1

python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py smoke \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1

python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py cv \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1

python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py hpo-plan \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1

python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py freeze \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1

python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py refit \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1

python3 _pipelines/02_task_datasets/lithofacies/p4_runner.py test \
  --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1
```

`prepare` is the only split-locking stage that indexes both HDF5 files. `cv`
and `refit` accept development data only. The `test` subcommand is the only
entry allowed to open F-5; it durably consumes the lifecycle before accessing
the HDF5 and rejects a second invocation for the same experiment.

Training consumes raw logits with cross-entropy. Softmax is applied only by
the inference/metric adapter. OOF logits fit one scalar temperature without
test labels. The primary HPO direction is to maximize development-fold
supported-class macro-F1, retaining fold mean/std/worst and calibration
guardrails. The archived plan is 8 random/sanity trials followed by 20
single-process TPE trials, `NopPruner` by default, and top-three configurations
confirmed with three registered seeds. `hpo-plan` records this contract but
does not launch HPO.

After OOF or frozen-test prediction archives exist, visualization is a
separate read-only operation:

```bash
python3 _pipelines/02_task_datasets/lithofacies/visualize_p4.py \
  --predictions _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1/frozen_test/predictions.json \
  --metrics _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1/frozen_test/metrics.json \
  --output-dir _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/gm09_v1/visualizations
```

It verifies that prediction and metric hashes are unchanged and produces the
depth facies/GT/prediction/confidence/error track, count plus row-normalized
confusion matrix, per-class precision/recall/F1/support, and calibration plot.

### P4 tests

The torch-free contract suite validates TaskSpec, F-5 isolation, honest LOGO-4
downgrade, fold class support, fold-local preprocessing, logits/softmax
separation, HPO direction, metric schemas, and read-only visualization. With
PyTorch available it additionally executes all three existing models, an
optimizer step, tiny-overfit, complete synthetic OOF/checkpoint/refit/single
test lifecycle, and checkpoint/artifact verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests \
  -p 'test_p4_contract.py' -v
```

The integration worktree intentionally does not carry the ignored HDF5 files.
The real-data one-step smoke gate therefore skips with an explicit reason until
those existing assets are mounted or the approved builder is rerun:

```bash
LITHOFACIES_P4_REAL_SMOKE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests \
  -p 'test_p4_contract.py' -v
```

If an integration worktree intentionally keeps ignored data elsewhere, point
the read-only gate at it with `LITHOFACIES_P4_DATASET_ROOT=/path/to/lithofacies`;
the test writes only to a temporary run directory and does not copy the HDF5.

No P4 score is a formal result until all four development folds complete,
configuration and epoch policy are frozen from OOF evidence, development is
refitted, and the single F-5 campaign is archived. The historical F-5 baseline
has already been observed, so this family is a frozen regression/final campaign
test rather than a previously unseen blind test.

## P5 open-model Stage-1 adapters

P5 adds contract-smoke adapters without changing the P4 GM09 schema, existing
models, split, baseline metrics, or frozen-test lifecycle. Exact upstream URLs,
revisions, licenses, dependency groups, weight policy, and smoke configurations
are frozen in `p5_source_lock.json`. Stage-1 uses scratch initialization only;
it never downloads a pretrained weight.

| Order | Model ID | Lane | Backend group |
|---:|---|:---:|---|
| 1 | `xgboost_multisoftprob_window` | P | `tabular-cpu` |
| 2 | `catboost_multiclass_window` | P | `tabular-cpu` |
| 3 | `minirocket_ridge_window` | P | `tabular-cpu` |
| 4 | `inceptiontime_window` | P | `tabular-cpu` (`tsai==1.0.1`) |
| 5 | `tcn_center_head` | P | `torch-common` |
| 6 | `balanced_softmax_tcn` | P | `torch-common` |
| 7 | `moderntcn_window` | P | `torch-common` |
| 8 | `ms_tcn2_dense` | S | `torch-common` |
| 9 | `embracenet_missing_modal` | P | `torch-common` |
| 10 | `multibench_lowrank_tensor_fusion` | P | `torch-common` |

P is the existing center-window classification contract (`[B,9]`). S is a
separate real-MD-ordered sequence-labeling contract (`[B,9,L]`) and never
shares a ranking with P. If the development archive lacks real
`center_md_m`, the S adapter is a structured `SKIP`; interval midpoints or a
center label repeated across a fabricated sequence are forbidden.

Every adapter consumes both inputs. The 26-row log tensor retains all 13
observed-value rows and all 13 missing-mask rows; the real `3x3x33` ST0202
patch is never discarded. Estimator adapters flatten all 1,155 values. Torch
adapters use explicit log and seismic paths and the tests perturb both the
seismic tensor and missing-mask rows to prove that each path affects logits.

### Stage-1 commands

Set the two variables to already approved shared interpreters; do not create an
environment or install dependencies from this runner. Runtime NPZ, reports,
and checkpoints live below the ignored `_outputs/p5_stage1/` directory.

```bash
TORCH_PYTHON="${TORCH_PYTHON:?set the approved torch-common interpreter}"
TABULAR_PYTHON="${TABULAR_PYTHON:?set the approved tabular-cpu interpreter}"
DATASET_ROOT="${LITHOFACIES_P5_DATASET_ROOT:?point to the existing lithofacies directory}"

"$TORCH_PYTHON" _pipelines/02_task_datasets/lithofacies/p5_stage1.py prepare-batch \
  --dataset-root "$DATASET_ROOT" \
  --batch-file _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/development_stage1.npz

"$TABULAR_PYTHON" _pipelines/02_task_datasets/lithofacies/p5_stage1.py smoke \
  --batch-file _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/development_stage1.npz \
  --models xgboost_multisoftprob_window,catboost_multiclass_window,minirocket_ridge_window,inceptiontime_window \
  --device cuda:0 \
  --output _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/tabular.json

"$TORCH_PYTHON" _pipelines/02_task_datasets/lithofacies/p5_stage1.py smoke \
  --batch-file _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/development_stage1.npz \
  --models tcn_center_head,balanced_softmax_tcn,moderntcn_window,ms_tcn2_dense,embracenet_missing_modal,multibench_lowrank_tensor_fusion \
  --device cuda:0 \
  --output _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/torch.json

"$TORCH_PYTHON" _pipelines/02_task_datasets/lithofacies/p5_stage1.py merge \
  --inputs \
    _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/tabular.json \
    _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/torch.json \
  --output _pipelines/02_task_datasets/lithofacies/_outputs/p5_stage1/summary.json
```

`prepare-batch` is development-only: its only HDF5 filename is `train.h5`, it
requires exactly the four approved development mother families, builds all
four LOGO folds, and records `frozen_test_accessed=false`. `smoke` performs a
fixed small fit/forward/loss/checkpoint round-trip; gradient models also run
backward and one AdamW step. It emits no formal metric or model ranking.

P5 contract tests, followed by the real-development batch gate:

```bash
PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests -p 'test_lithofacies_p5*.py' -v

LITHOFACIES_P5_REAL_BATCH=1 \
LITHOFACIES_P5_DATASET_ROOT="$DATASET_ROOT" \
PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests -p 'test_lithofacies_p5*.py' -v
```

## P5 Stage-2 fixed-budget development pilot

`lithofacies_p5_stage2.py` is the track-prefixed Stage-2 entry point. It keeps
the GM09 nine-class schema and the first valid P4 fold fixed: `15/9-F-14`,
`15/9-F-15`, and `15/9-F-4` train; `15/9-19` validates. The runner has no
frozen-test argument or loader. It consumes the existing Stage-1 NPZ envelope,
whose only loaded HDF5 basename is `train.h5`.

All P cells receive the same `26x33` log tensor (13 observed-value rows plus
13 missing-mask rows), `3x3x33` ST0202 patch, at most 320 fold-train samples,
and at most 160 validation samples. The real fixed fold uses all 315/132
available samples. Neural cells use batch size 32 and at most 40 parameter
updates, including a three-update finite/shape/backward tiny-overfit gate;
their wall limit is 600 seconds. XGBoost and CatBoost use 40 boosting
iterations, MiniRocket uses the source-locked 1,000-kernel transform, and each
CPU cell has a 300-second wall limit. Every seed is derived stably from root
seed 2693 and the model/component ID. Preprocessing and class counts come only
from the fold-train mother families.

The development archive has no finite `center_md_m` for any of its 447
samples. Consequently, the only S candidate (`ms_tcn2_dense`) remains a
structured `SKIP/not_rankable`; no interval midpoint, row order, or repeated
center label is used to fabricate a sequence, and S never enters the P board.

### Reproduce Stage-2

Use the approved shared environments. GPU commands must be wrapped by the
single frozen lock; the runner independently fails closed when a CUDA command
is not launched under that lock. Runtime NPZ/partials/checkpoints remain
ignored below `_outputs/p5_stage2/runtime/`; only the portable JSONL, summary,
and P leaderboard are versioned.

```bash
TORCH_PYTHON="${TORCH_PYTHON:?set the approved torch-common interpreter}"
TABULAR_PYTHON="${TABULAR_PYTHON:?set the approved tabular-cpu interpreter}"
DATASET_ROOT="${LITHOFACIES_P5_DATASET_ROOT:?point to the existing lithofacies directory}"
GPU_LOCK="${VOLVE_P5_GPU_LOCK:-$HOME/.cache/volve-p5/locks/gpu0.lock}"
STAGE2=_pipelines/02_task_datasets/lithofacies/lithofacies_p5_stage2.py
OUT=_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage2

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" "$STAGE2" prepare-batch \
  --dataset-root "$DATASET_ROOT" --batch-file "$OUT/runtime/development_fold0.npz"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" "$STAGE2" pilot \
  --batch-file "$OUT/runtime/development_fold0.npz" \
  --models xgboost_multisoftprob_window,catboost_multiclass_window,minirocket_ridge_window \
  --device cpu --output "$OUT/runtime/tabular_estimators.json"

flock -w 900 "$GPU_LOCK" env PYTHONDONTWRITEBYTECODE=1 \
  "$TABULAR_PYTHON" "$STAGE2" pilot \
  --batch-file "$OUT/runtime/development_fold0.npz" --models inceptiontime_window \
  --device cuda:0 --output "$OUT/runtime/tabular_inception.json"

flock -w 900 "$GPU_LOCK" env PYTHONDONTWRITEBYTECODE=1 \
  "$TORCH_PYTHON" "$STAGE2" pilot \
  --batch-file "$OUT/runtime/development_fold0.npz" \
  --models tcn_center_head,balanced_softmax_tcn,moderntcn_window,ms_tcn2_dense,embracenet_missing_modal,multibench_lowrank_tensor_fusion \
  --device cuda:0 --output "$OUT/runtime/torch_models.json"

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" "$STAGE2" finalize \
  --inputs "$OUT/runtime/tabular_estimators.json" \
    "$OUT/runtime/tabular_inception.json" "$OUT/runtime/torch_models.json" \
  --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests \
  -p 'test_lithofacies_p5*.py' -v
```

### Recorded Stage-2 evidence

All 10 preregistered cells have one portable record: 9 P `PASS`, 1 S
`SKIP`, 0 `FAIL`, and 0 `TIMEOUT`. The legal development-only P board is:

| Rank | Model ID | Fixed-9 macro-F1 (primary) | supported-class macro-F1 (diagnostic) |
|---:|---|---:|---:|
| 1 | `xgboost_multisoftprob_window` | 0.213580 | 0.274603 |
| 2 | `catboost_multiclass_window` | 0.167689 | 0.215600 |
| 3 | `inceptiontime_window` | 0.138188 | 0.177671 |
| 4 | `tcn_center_head` | 0.123678 | 0.159015 |
| 5 | `minirocket_ridge_window` | 0.123163 | 0.158352 |
| 6 | `embracenet_missing_modal` | 0.105848 | 0.136090 |
| 7 | `moderntcn_window` | 0.086816 | 0.111621 |
| 8 | `multibench_lowrank_tensor_fusion` | 0.055556 | 0.071429 |
| 9 | `balanced_softmax_tcn` | 0.007937 | 0.010204 |

This is a single-fold fixed-budget screening result, not CV confirmation and
not a frozen-test result. The primary metric and its worst-family guardrail
both use the fixed nine-class Macro-F1 required by the frozen Stage-2 matrix;
supported-class Macro-F1 is diagnostic only. The two fixed-nine values are
equal here because the pilot has exactly one validation mother family.
Canonical evidence lives in `_outputs/p5_stage2/` as
`p5_stage2_results.jsonl`, `p5_stage2_summary.json`, and
`p5_stage2_p_leaderboard.json`; it contains no host/worktree path or retained
checkpoint.

## P5 Stage-3 multiseed LOGO4 confirmation

`lithofacies_p5_stage3.py` reuses the accepted Stage-2 adapters, candidate
configuration, loss, preprocessing, context and update budgets. Its frozen
roster is exactly three P-lane models by four P4 LOGO folds by repeat seeds
`1867973658`, `2137841944`, and `3902865753` (36 cells). F-5 remains the
unopened frozen test family. The batch builder accepts the development dataset
directory but opens only `train.h5`; every fold fits normalization and class
weights on its three fold-train mother families before applying them to its
held-out family. Target IDs remain the fixed GM09 nine-class identity mapping,
and no post-hoc calibration is introduced.

The portable development result records all 36 cells: 33 `PASS`, 3 `FAIL`, 0
`SKIP`, and 0 `TIMEOUT`, for a 91.67% legal completion rate. XGBoost and
InceptionTime completed 12/12 cells. CatBoost completed 9/12 and is marked
`not_rankable` at 75%: fold 2 has zero fold-train support for class 8, and all
three CatBoost seeds returned non-finite logits instead of a legal fixed-nine
prediction. That failure is retained rather than merging classes or changing
the model. The eligible P leaderboard is therefore:

| Rank | Model ID | Fixed-9 macro-F1 mean | 95% cell-bootstrap CI | Worst fold | Seed-mean std |
|---:|---|---:|---:|---:|---:|
| 1 | `xgboost_multisoftprob_window` | 0.194938 | [0.187808, 0.202440] | 0.181276 | 0.000000 |
| 2 | `inceptiontime_window` | 0.078467 | [0.056435, 0.099647] | 0.055085 | 0.007954 |

The primary metric, bootstrap, worst-fold guardrail and ranking all use the
fixed nine-class Macro-F1. Supported-class Macro-F1 remains a diagnostic field
only. These are development OOF results, not frozen-test results.

Six committed figures cover the winning model's fixed-nine confusion matrix,
per-class precision/recall/F1, raw-softmax calibration, fold-by-seed scores and
missing-modality behavior. A continuous measured-depth facies track is
explicitly `not_feasible`: all 447 development samples have non-finite
`center_md_m`, so the runner emits an explanatory evidence panel and does not
fabricate depth from interval midpoints or row order. Full OOF predictions and
checkpoints remain ignored under `_outputs/p5_stage3/runtime/`; their portable
paths and hashes are recorded in the OOF manifest.

### Reproduce Stage-3

Run from the project root with the approved shared environments and an
existing development dataset. The finalizer writes all canonical artifacts but
returns non-zero when any cell is `FAIL`/`TIMEOUT`; that exit is expected for
the recorded three CatBoost failures and must not be interpreted as missing
artifacts.

```bash
TORCH_PYTHON="${TORCH_PYTHON:?set the approved torch-common interpreter}"
TABULAR_PYTHON="${TABULAR_PYTHON:?set the approved tabular-cpu interpreter}"
DATASET_ROOT="${LITHOFACIES_P5_DATASET_ROOT:?point to the existing lithofacies directory}"
GPU_LOCK="${VOLVE_P5_GPU_LOCK:-$HOME/.cache/volve-p5/locks/gpu0.lock}"
STAGE3=_pipelines/02_task_datasets/lithofacies/lithofacies_p5_stage3.py
OUT=_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" "$STAGE3" prepare-batch \
  --dataset-root "$DATASET_ROOT" --batch-file "$OUT/runtime/development_logo4.npz"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" "$STAGE3" run \
  --batch-file "$OUT/runtime/development_logo4.npz" \
  --models xgboost_multisoftprob_window,catboost_multiclass_window \
  --folds 0,1,2,3 --repeats 0,1,2 --device cpu \
  --output "$OUT/runtime/tabular_estimators.json"

flock -w 900 "$GPU_LOCK" env PYTHONDONTWRITEBYTECODE=1 \
  "$TABULAR_PYTHON" "$STAGE3" run \
  --batch-file "$OUT/runtime/development_logo4.npz" \
  --models inceptiontime_window --folds 0,1,2,3 --repeats 0,1,2 \
  --device cuda:0 --output "$OUT/runtime/inceptiontime.json"

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" "$STAGE3" finalize \
  --inputs "$OUT/runtime/tabular_estimators.json" "$OUT/runtime/inceptiontime.json" \
  --batch-file "$OUT/runtime/development_logo4.npz" --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests \
  -p 'test_lithofacies_p5*.py' -v
```

Canonical evidence is `_outputs/p5_stage3/p5_stage3_results.jsonl`,
`p5_stage3_summary.json`, `p5_stage3_gm09_p_leaderboard.json`,
`p5_stage3_oof_manifest.json`, `p5_stage3_visualization_manifest.json`, and the
six PNG files under `_outputs/p5_stage3/figures/`.

## P5 Stage-4 known-holdout confirmation

`lithofacies_p5_stage4.py` freezes the Stage-3 P winner
`xgboost_multisoftprob_window` at 40 boosting rounds, depth 2, seed 2693 and
the unchanged Stage-2 budget hash. This is not a fresh-blind campaign: the
repository already contains an earlier F-5 baseline. Every Stage-4 artifact
therefore records `prior_test_consumed=true`, `fresh_blind=false`, and
`evidence_class=previously_seen_reusable_holdout`.

The runner is independent of the Torch-only P4 lifecycle and never resets a
P4 run. Its single-use state sequence is `CONFIG_FROZEN` → `REFIT_COMPLETE` →
`KNOWN_HOLDOUT_CONSUMED` → `CONFIRMATION_COMPLETE`. The consumed transition,
including the frozen config and checkpoint hashes, is written durably before
the only Stage-4 function permitted to open F-5.

The two approved environments have complementary dependencies: `torch-common`
provides HDF5 reading and `tabular-cpu` provides XGBoost. Run the four commands
in order from the project root:

```bash
HDF5_PYTHON="${HDF5_PYTHON:?set the approved torch-common interpreter}"
TABULAR_PYTHON="${TABULAR_PYTHON:?set the approved tabular-cpu interpreter}"
DATASET_ROOT="${LITHOFACIES_P5_DATASET_ROOT:?point to the existing lithofacies directory}"
STAGE4=_pipelines/02_task_datasets/lithofacies/lithofacies_p5_stage4.py
OUT=_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage4_confirmation

PYTHONDONTWRITEBYTECODE=1 "$HDF5_PYTHON" "$STAGE4" prepare-development \
  --dataset-root "$DATASET_ROOT" --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" "$STAGE4" refit \
  --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$HDF5_PYTHON" "$STAGE4" prepare-holdout \
  --dataset-root "$DATASET_ROOT" --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" "$STAGE4" confirm \
  --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests \
  -p 'test_lithofacies_p5_stage[34].py' -v
```

Preprocessing and class weights fit all 447 samples from the four development
mother families. The unchanged Stage-2 cap then selects 320 deterministic,
class-balanced refit samples while retaining all four families. F-5 evaluation
uses all 120 samples, within the frozen validation cap of 160; no calibration
parameter is fitted on its labels.

The recorded confirmation completed with accuracy `0.416667`, fixed-nine
Macro-F1 `0.189153`, supported-class Macro-F1 `0.243197` (diagnostic only),
negative log-likelihood `1.702236`, multiclass Brier score `0.793780`, and ECE
`0.141489`. F-5 support is `[2,31,2,17,3,24,41,0,0]`. Preparation, refit,
holdout consumption and inference took `2.339s`, `30.641s`, `2.492s`, and
`4.255s`, respectively.

Portable evidence lives under `_outputs/p5_stage4_confirmation/`; ignored
runtime batches and the refit checkpoint are referenced by hash but are not
committed. The fixed-nine confusion, per-class precision/recall/F1/support and
raw-softmax reliability figures are committed. F-5 contains no finite
`center_md_m`, so the continuous depth track remains explicitly
`not_feasible`; interval midpoints and row order are never substituted.

## P5.1 R0/R1 split-mechanism audit

`lithofacies_p5_r01.py` is a development-only runner. R0 performs no fitting:
it freezes the original GM09 / `GENETIC FACIES` nine-class order, excludes
`UNKNOWN`, `UNDEFINED`, LITH and samples outside an explicit interpretation
interval, and records `fixed_schema_macro_f1` as the sole primary metric.
Supported-class Macro-F1 is diagnostic only. Modality (`W` well log or `M`
well log plus seismic) and task (`P` point/window classification or `S`
continuous sequence labeling) are orthogonal axes. Both S lanes remain
`not_rankable` because every development sample lacks finite `center_md_m`;
TWT, row order and interval midpoint are forbidden depth substitutes.

The sealed F-5 identity is recorded with `prior_test_consumed=true` and
`fresh_blind=false`, but R0/R1 has no physical-test command or loader. Its only
HDF5-facing command opens the explicit development `train.h5` and rejects any
family roster other than the four frozen development mother families.

R1 uses exactly one preregistered `SGDClassifier(loss="log_loss")`, seed 2693,
and 64 fixed iterations without HPO. It evaluates the same model budget in W
and M for four paired mechanisms: center-point random KFold4 versus LOGO4, and
full 33-point-window random KFold4 versus LOGO4. Random splits are
`diagnostic_only/not_rankable`; LOGO4 is a protocol result, not a model
leaderboard. Each fold fits log/seismic normalization, missing-channel masks
and square-root inverse class weights using fold-train only. Artifacts record
fixed-nine Macro-F1, accuracy, nine-class support and confusion, OOF coverage,
worst-family score, family/well/interval overlap and exact shifted-window
overlap. Any failed fold is retained as structured evidence and makes its
condition not rankable; no replacement split or label is allowed.

Run from the project root with the existing development asset and approved
interpreters. The first command needs HDF5 support; the second needs
scikit-learn. The compressed development envelope is ignored and only the four
portable JSON/JSONL evidence files are candidates for Git.

```bash
HDF5_PYTHON="${HDF5_PYTHON:?set an approved interpreter with h5py}"
TABULAR_PYTHON="${TABULAR_PYTHON:?set the approved tabular interpreter}"
DATASET_ROOT="${LITHOFACIES_P5_DATASET_ROOT:?point to the development dataset directory}"
R01=_pipelines/02_task_datasets/lithofacies/lithofacies_p5_r01.py
OUT=_pipelines/02_task_datasets/lithofacies/_outputs/p5_r01

PYTHONDONTWRITEBYTECODE=1 "$HDF5_PYTHON" "$R01" prepare \
  --dataset-root "$DATASET_ROOT" \
  --batch-file "$OUT/runtime/development.npz" \
  --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" "$R01" run \
  --batch-file "$OUT/runtime/development.npz" \
  --output-dir "$OUT"

PYTHONDONTWRITEBYTECODE=1 LITHOFACIES_R01_TINY=1 "$TABULAR_PYTHON" \
  -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests \
  -p 'test_lithofacies_p5_r01.py' -v
```

Canonical portable evidence is `_outputs/p5_r01/r0_contract.json`,
`r1_results.jsonl`, `r1_summary.json`, and `artifact_manifest.json`. R1 is a
split-protocol mechanism audit only. Fair ranking of at least ten models is a
later stage and must reuse the legal grouped protocol rather than its random
diagnostic counterpart.

## P11 conservative MOMENT residual fusion

`lithofacies_p11_residual_fusion.py` keeps the Stage-3 XGBoost logits as the
main route and lets frozen MOMENT embeddings contribute only through a
per-class bounded residual:

```text
fused_logits = baseline_logits
             + sigmoid(gate_logits)
             * 2 * tanh(linear(mean_MOMENT_embedding))
```

The runner evaluates the immutable LOGO4 folds and three frozen Stage-3 seeds.
Every fold/seed pair contains all five preregistered variants: `baseline`,
direct MOMENT, pretrained residual, same-architecture random residual, and
exact `gate0` degeneration. It rejects every test/holdout/frozen-like input
path before opening it and exposes no holdout command.

The complete 60-cell development matrix retained the XGBoost default. The
pretrained residual improved fixed-nine Macro-F1 by only `+0.002040` over the
baseline and `+0.002857` over the random control, with wins in `6/12` paired
cells. Those values miss all preregistered promotion thresholds, so the
recorded decision is `NON_BENEFICIAL_KEEP_BASELINE`. `gate0` is bit-identical
to the baseline logits.

### Reproduce P11

Run from the project root with the existing approved environments, cached
MOMENT snapshot, and ignored Stage-3 development runtime:

```bash
TABULAR_PYTHON="${P5_TABULAR_PYTHON:?set the approved tabular-cpu interpreter}"
TORCH_PYTHON="${P5_TORCH_PYTHON:?set the approved torch-common interpreter}"
STAGE3_RUNTIME="${LITHOFACIES_STAGE3_RUNTIME:?point to the Stage-3 development runtime}"
MOMENT_SNAPSHOT="${LITHOFACIES_MOMENT_SNAPSHOT:?point to the pinned local snapshot}"
GPU_LOCK="${VOLVE_P5_GPU_LOCK:-$HOME/.cache/volve-p5/locks/gpu0.lock}"
P11=_pipelines/02_task_datasets/lithofacies/lithofacies_p11_residual_fusion.py
STAGE3_RESULTS=_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3/p5_stage3_results.jsonl
OUT=_pipelines/02_task_datasets/lithofacies/_outputs/p11_residual_fusion

PYTHONDONTWRITEBYTECODE=1 "$TABULAR_PYTHON" "$P11" prepare-baseline \
  --development-batch "$STAGE3_RUNTIME/development_logo4.npz" \
  --stage3-results "$STAGE3_RESULTS" \
  --checkpoint-dir "$STAGE3_RUNTIME/checkpoints" \
  --prediction-dir "$STAGE3_RUNTIME/predictions" \
  --output-bundle "$OUT/runtime/baseline_logits.npz"

flock -w 900 "$GPU_LOCK" env PYTHONDONTWRITEBYTECODE=1 \
  "$TORCH_PYTHON" "$P11" run \
  --development-batch "$STAGE3_RUNTIME/development_logo4.npz" \
  --baseline-bundle "$OUT/runtime/baseline_logits.npz" \
  --snapshot "$MOMENT_SNAPSHOT" --output-dir "$OUT" --device cuda:0

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" "$P11" verify --output-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" -m pytest -q \
  _pipelines/02_task_datasets/lithofacies/tests/test_lithofacies_p11_residual_fusion.py
```

Canonical portable evidence is `_outputs/p11_residual_fusion/results.jsonl`,
`summary.json`, `evidence.md`, `primary_metric.png`, and
`artifact_manifest.json`. Runtime baseline logits, embedding caches, and
partial resume state remain ignored.

## P11 clean well-log native-context diagnostic

`lithofacies_p11_clean_well_native33.py` is an isolated diagnostic built on
the committed gated-residual harness. It leaves the original P11 runner and
60-cell artifacts unchanged. MOMENT now receives only the 13 real normalized
well-log curves at their native 33-point measured-depth resolution. The pinned
8-point patch length/stride yields four real tokens covering 32 samples; no
interpolation or other synthetic resampling is used. The 13 observation-mask
planes and 9 flattened seismic traces are deliberately excluded from both
MOMENT and the residual head in this clean-input phase.

The full strict LOGO4 four-fold by three-seed, five-variant matrix again keeps
the Stage-3 XGBoost baseline. Mean fixed-nine Macro-F1 is `0.194938` for
baseline, `0.073445` for direct native-context MOMENT, `0.197572` for the
pretrained residual, `0.194696` for the same-architecture random-init
residual, and `0.194938` for exact `gate0`. The diagnostic improvement over
random init is only `+0.002876`, below the existing `0.005` materiality
threshold and nearly unchanged from original P11's `+0.002857`. Pretrained
and random gate means are likewise close (`0.017800` versus `0.017848`).
The decision remains `NON_BENEFICIAL_KEEP_BASELINE`.

This result does not select a larger MOMENT model. The proposed later phase
that routes observation masks directly to the residual head and seismic
patches through a separate spatial CNN remains unrun pending an explicit
decision after this minimal representation diagnostic.

### Reproduce the native-context diagnostic

Run from the project root using the existing ignored P11 baseline-logit bundle:

```bash
TORCH_PYTHON="${P5_TORCH_PYTHON:?set the approved torch-common interpreter}"
STAGE3_RUNTIME="${LITHOFACIES_STAGE3_RUNTIME:?point to the Stage-3 development runtime}"
MOMENT_SNAPSHOT="${LITHOFACIES_MOMENT_SNAPSHOT:?point to the pinned local snapshot}"
GPU_LOCK="${VOLVE_P5_GPU_LOCK:-$HOME/.cache/volve-p5/locks/gpu0.lock}"
RUNNER=_pipelines/02_task_datasets/lithofacies/lithofacies_p11_clean_well_native33.py
OUT=_pipelines/02_task_datasets/lithofacies/_outputs/p11_clean_well_native33
BASELINE=_pipelines/02_task_datasets/lithofacies/_outputs/p11_residual_fusion/runtime/baseline_logits.npz

flock -w 900 "$GPU_LOCK" env PYTHONDONTWRITEBYTECODE=1 \
  "$TORCH_PYTHON" "$RUNNER" run \
  --development-batch "$STAGE3_RUNTIME/development_logo4.npz" \
  --baseline-bundle "$BASELINE" --snapshot "$MOMENT_SNAPSHOT" \
  --output-dir "$OUT" --device cuda:0

PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" "$RUNNER" verify \
  --output-dir "$OUT"
PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" -m pytest -q \
  _pipelines/02_task_datasets/lithofacies/tests/test_lithofacies_p11_clean_well_native33.py
```

Portable evidence is `_outputs/p11_clean_well_native33/results.jsonl`,
`summary.json`, `evidence.md`, `primary_metric.png`, and
`artifact_manifest.json`. Baseline logits, native embedding caches, resume
state, and raw predictions remain ignored.
