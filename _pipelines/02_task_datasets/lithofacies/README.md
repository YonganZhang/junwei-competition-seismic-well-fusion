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
  -s _pipelines/02_task_datasets/lithofacies/tests -p 'test_p5*.py' -v

LITHOFACIES_P5_REAL_BATCH=1 \
LITHOFACIES_P5_DATASET_ROOT="$DATASET_ROOT" \
PYTHONDONTWRITEBYTECODE=1 "$TORCH_PYTHON" -m unittest discover \
  -s _pipelines/02_task_datasets/lithofacies/tests -p 'test_p5*.py' -v
```
