# Track ⑥ — 3-D geological model reconstruction

This track builds a real-data porosity reconstruction task from the Volve
reservoir models.  It does not synthesize a reference volume when a proprietary
format cannot be decoded.

## P5.1 R0/R1 development-only mechanism audit

`reconstruction_p5_r01.py` implements the zero-training R0 provenance and
firewall audit plus one fixed-model R1 conditional-channel check.  It uses only
the conditional development I-blocks 0–3 from physical `train.h5`; its narrow
reader does not open physical `test.h5`, `well_log_seq`, known/frozen arrays,
predictions or metrics.  P4 conditional fold 2 is reused as the pseudo-test
block (K=4), with K=3/5 purged and K=0/1/2/6 retained for fitting.

Pseudo-well locations are frozen by geometry-only farthest-point sampling
before PORO values are read.  Their values are explicitly
`synthetic/reference-revealed` Eclipse target samples, not independently
measured PHIE.  One fixed `reconstruction_linear_sgd` model/checkpoint is used
for all three inference conditions on one common mask:

- `B0`: formal name `no_pseudo_test_PORO_condition`;
- `B1`: correct pseudo-test PORO values;
- `shuffled`: the same locations with a seed-2693 non-identity value rotation.

B0 still retains the fixed project-level weak MD→TWT well tie used to sample
the seismic volume.  It must never be described as containing no well-derived
information.  Exact pseudo-well cells are excluded from every condition, and
the report includes overall and pre-frozen distance-band RMSE/MAE/bias/R².  A
degenerate condition set, shuffle, distance band or insensitive model fails
closed.  R1 is `development_protocol_mechanism_only`, not HPO, model ranking,
fresh-blind evidence or field generalization.

Run the portable tests, then the bounded real-development entrypoint:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_reconstruction_p5_r01.py' -v

PYTHONDONTWRITEBYTECODE=1 python3 \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_r01.py run \
  --data-dir _data/processed/reconstruction \
  --output-dir _pipelines/02_task_datasets/reconstruction/p5_r01_evidence
```

Canonical portable evidence is `p5_r01_evidence/r0_manifest.json`,
`r1_results.json`, `P5_R01_REPORT.md` and `artifact_manifest.json`.  No HDF5,
prediction archive or checkpoint file is produced.

## P4 training/validation plugin

`p4_reconstruction.py` adds the frozen P4 TaskSpec, continuous spatial test,
buffered development CV, fold-train preprocessing/constraint filtering,
minimize-direction fixed baseline plan, full resumable checkpoint/artifact
contract and one-shot test lifecycle. `p4_visualize.py` renders inline,
crossline and time/depth truth/prediction/residual slices plus distribution and
spectrum diagnostics from archived predictions only.

Conditional and strict P4 runs are separate tasks and metric namespaces; their
artifacts cannot be mixed. The existing `baseline.py` and canonical result
JSON remain unchanged. See `P4_SOP.md` for commands, data provisioning and the
unit → contract → tiny → real-smoke → CV → refit → single-test order. Known
scientific limits are machine-readable in `not_feasible.json`.

## P5 known-holdout confirmation Stage 4

`reconstruction_p5_stage4.py` is a track-private finalization path for the
Stage-3 winner because the P4 runner does not support `pykrige_ok3d`. It does
not alter or impersonate the P4 single-test lifecycle. Both physical holdout
regions were scored by the historical Ridge baseline, so every Stage-4
artifact is fail-closed labelled `prior_test_consumed=true`,
`evidence_class=previously_seen_reusable_holdout`, and `fresh_blind=false`.
These results are known-holdout confirmation evidence, never a new blind test.

For each independent mode the runner verifies the Stage-3 leaderboard,
winner, split hash, prior result and canonical HDF5 hashes. It fits
preprocessing on every legal development cell, then applies the frozen
traditional-CPU cap: deterministic 512-point model fit, one update and 300
model-wall seconds. A timeout is retained as structured `budget_timeout`; no
configuration is relaxed. Conditional inference includes test-region well
constraints and excludes their exact 90 cells from metrics, while strict
inference receives no guard/test porosity constraints.

Use only the already-provisioned torch-common primary environment and
tabular/geostat auxiliary site-packages:

```bash
export VOLVE_P5_AUX_SITE_PACKAGES=/path/to/tabular-cpu/site-packages
export VOLVE_P5_AUX_DEPENDENCY_GROUP=tabular-cpu
export PYTHONDONTWRITEBYTECODE=1

timeout --signal=TERM 360s /path/to/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage4.py \
  run-mode --mode strict --data-dir /path/to/reconstruction \
  --output-dir _pipelines/02_task_datasets/reconstruction/p5_stage4_confirmation

timeout --signal=TERM 360s /path/to/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage4.py \
  run-mode --mode conditional --data-dir /path/to/reconstruction \
  --output-dir _pipelines/02_task_datasets/reconstruction/p5_stage4_confirmation

/path/to/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage4.py \
  collate \
  --output-dir _pipelines/02_task_datasets/reconstruction/p5_stage4_confirmation
```

Each mode persists its frozen config, refit audit/checkpoint, compact legal
holdout predictions, metrics, artifact hashes and a truth/prediction/residual
slice figure with ECDF and 3-D spectrum diagnostics. The ECDF is explicitly
diagnostic-only and contributes no selection or final score. `summary.json`
keeps strict and conditional metrics separate.

## P5 multiseed spatial-CV Stage 3

`reconstruction_p5_stage3.py` confirms only the frozen Stage-2 top three in
each independent lane. Strict uses `pykrige_ok3d`, `gpytorch_svgp` and
`gstools_krige_condsrf`; conditional uses `pykrige_ok3d`, `gpytorch_svgp` and
`scipy_rbf_neighbors`. Every model runs all five buffered P4 development
folds and repeat seeds `1867973658`, `2137841944`, and `3902865753`: exactly
90 expected cells. There is no HPO, temporary 20% split, frozen-test command,
or score backfill.

Cache preparation requires the read-only Stage-2 cache because its frozen
`split_manifest.json` files are the split source of truth. Stage 3 does not
rescan frozen-test metadata; it loads only logical development I-block arrays.
Fold preprocessing, identity target transform, conditional IDW constraints
and the explicit zero-well fallback are fit from that fold's effective train
IDs. The fixed 512 train / 2,048 validation-voxel maximum, model configs,
losses, tiny gate and update counts remain Stage-2 values.

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage3.py \
  prepare-cache --data-dir /path/to/reconstruction \
  --stage2-cache _tmp/p5_stage2_reconstruction/cache \
  --cache-dir _tmp/p5_stage3_reconstruction/cache
```

All cells use the torch-common interpreter. If its optional geostat modules
are provided by another already-provisioned shared environment, append that
environment's `site-packages` with `VOLVE_P5_AUX_SITE_PACKAGES`; core
Torch/NumPy/SciPy still resolve from torch-common. Do not install or copy
packages during the benchmark.

CPU geostat example:

```bash
VOLVE_P5_AUX_SITE_PACKAGES=/path/to/shared-geostat/site-packages \
VOLVE_P5_AUX_DEPENDENCY_GROUP=shared-geostat PYTHONDONTWRITEBYTECODE=1 \
  /path/to/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage3.py \
  run-cell --mode conditional --model scipy_rbf_neighbors \
  --fold-id 0 --repeat-id 0 \
  --cache-dir _tmp/p5_stage3_reconstruction/cache \
  --cell-root _tmp/p5_stage3_reconstruction/cells --device cpu
```

Every GPyTorch cell must run on `cuda:0` under the shared 900-second external
flock. The launcher records real queue wait time for the cell audit:

```bash
export VOLVE_P5_GPU_LOCK="${VOLVE_P5_GPU_LOCK:-$HOME/.cache/volve-p5/locks/gpu0.lock}"
export VOLVE_P5_LOCK_START_NS="$(date +%s%N)"
flock -w 900 "$VOLVE_P5_GPU_LOCK" bash -c '
  end_ns="$(date +%s%N)"
  export VOLVE_P5_GPU_LOCK_WAIT_SECONDS="$(awk -v start="$VOLVE_P5_LOCK_START_NS" \
    -v end="$end_ns" "BEGIN {printf \"%.9f\", (end-start)/1000000000}")"
  export VOLVE_P5_GPU_LOCK_HELD=1 CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
  exec /path/to/torch-common/bin/python \
    _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage3.py \
    run-cell --mode strict --model gpytorch_svgp --fold-id 0 --repeat-id 0 \
    --cache-dir _tmp/p5_stage3_reconstruction/cache \
    --cell-root _tmp/p5_stage3_reconstruction/cells --device cuda:0
'
```

After all 90 status files exist, collation fails closed on missing/duplicate
cells, wrong seeds or split hashes, budget/config drift, CPU neural results,
missing GPU wait/VRAM evidence, test access and cross-lane contamination. It
writes portable JSON/PNG evidence while checkpoints and sampled OOF archives
remain in ignored `_tmp/p5_stage3_reconstruction/`:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/torch-common/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage3.py \
  collate --cell-root _tmp/p5_stage3_reconstruction/cells \
  --output-dir _pipelines/02_task_datasets/reconstruction

PYTHONDONTWRITEBYTECODE=1 /path/to/torch-common/bin/python -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_reconstruction_p5_stage3.py' -v
```

The strict and conditional figures separately contain K/J/I truth,
prediction and residual views, CDF, radial spectrum, empirical variogram,
distance-to-fold-train-well error and fold-by-seed RMSE. These are sampled
development OOF diagnostics, not frozen-test or full-volume claims. A lane is
marked `not_rankable` when fewer than 80% of its frozen cells complete legally.

Budget audit: `gpytorch_svgp` declares `batch_representation=point`, so Stage 3
reuses the frozen Stage-2 `point_neural` budget verbatim: 100 formal training
updates, a 200-update protocol cap, and a 600-second model-wall cap. Its tiny
gate is recorded separately and is not added to the 100 updates. The
80-update cap belongs only to the 3-D volume neural/operator class (whose
frozen pilot actually uses 20 updates); it does not apply to `gpytorch_svgp`.

## P5 fixed-budget Stage 2

`reconstruction_p5_stage2.py` evaluates the ten preregistered reconstruction
candidates on the first valid P4 development fold with root seed 2693.  The
strict and conditional TaskSpecs, caches, result rows and leaderboards remain
independent.  Both representations are scored on the same 2,048 validation
voxels using RMSE, MAE and log-spectrum RMSE; this is a small fixed-fold pilot,
not complete CV or frozen-test evidence.

The fixed pilot exposes 512 fold-train targets.  Traditional interpolation
fits once with a 300-second cap, point neural models receive 100 updates within
the protocol's 200-update/600-second cap, and 3-D neural/operator models
receive 20 updates within the 80-update/900-second cap.  All trainable
neural/operator candidates must use `cuda:0`.  Their result is rejected unless
it records an acquired 900-second GPU flock; CPU neural scores and unresolved
`pilot_exception` records cannot enter collation.

Prepare the disposable development-only cache with an HDF5-capable shared
environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/hdf5-env/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage2.py \
  prepare-cache --data-dir /path/to/reconstruction \
  --cache-dir _tmp/p5_stage2_reconstruction/cache
```

Run CPU interpolation cells directly.  Run every trainable cell through the
single-GPU lock; the environment marker lets the child archive the external
lock evidence without trying to acquire the same lock twice:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/geostat-env/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage2.py \
  run-cell --mode strict --model pykrige_ok3d \
  --cache-dir _tmp/p5_stage2_reconstruction/cache \
  --cell-root _tmp/p5_stage2_reconstruction/cells --device cpu

LOCK="${VOLVE_P5_GPU_LOCK:-$HOME/.cache/volve-p5/locks/gpu0.lock}"
flock -w 900 "$LOCK" env VOLVE_P5_GPU_LOCK_HELD=1 CUDA_VISIBLE_DEVICES=0 \
  PYTHONDONTWRITEBYTECODE=1 /path/to/torch-env/bin/python \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage2.py \
  run-cell --mode strict --model neuralop_fno3d \
  --cache-dir _tmp/p5_stage2_reconstruction/cache \
  --cell-root _tmp/p5_stage2_reconstruction/cells --device cuda:0
```

After all twenty cells exist, `collate` writes only portable evidence into
this track directory: `p5_stage2_results.jsonl`, `p5_stage2_summary.json` and
the two mode-specific leaderboard JSON files.  MPS remains a structured skip
without a legal training image; tiny-cuda-nn remains a dependency skip.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 \
  _pipelines/02_task_datasets/reconstruction/reconstruction_p5_stage2.py \
  collate --cell-root _tmp/p5_stage2_reconstruction/cells \
  --output-dir _pipelines/02_task_datasets/reconstruction
```

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_reconstruction_p5_stage2.py' -v
```

## P5 open-model Stage 1

P5 adds ten dynamically discovered thin adapters under
`_models/reconstruction/`, with immutable upstream revisions and licenses in
`_models/reconstruction/source_lock.json`.  `p5_stage1.py` performs only
contract smoke: synthetic plus real **development** fit/forward/loss/backward
(where trainable), finite/shape checks, same-seed replay and checkpoint
round-trip.  It has no frozen-test command or loader.

Strict and conditional runs have different TaskSpecs and output directories.
Strict receives six seismic/coordinate features and zero target-derived well
values; conditional receives the P4 fold-train IDW feature as a seventh
feature.  MPSlib produces a structured `missing_legal_training_image` skip
unless an independently licensed training image is explicitly approved; the
Eclipse reference/test volume is forbidden as a training image.  Missing
optional packages likewise produce structured skips—Stage 1 never installs,
compiles or substitutes dependencies and never downloads weights.

Portable adapter/firewall tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_p5_stage1.py' -v
```

Explicit real-development smoke (use a pre-provisioned shared environment):

```bash
RECONSTRUCTION_DATA_DIR=/path/to/reconstruction \
PYTHONDONTWRITEBYTECODE=1 /path/to/shared/env/bin/python -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_p5_real_smoke.py' -v

PYTHONDONTWRITEBYTECODE=1 /path/to/shared/env/bin/python \
  _pipelines/02_task_datasets/reconstruction/p5_stage1.py \
  --mode both --models scipy_rbf_neighbors \
  --data-dir /path/to/reconstruction --output-root _tmp/p5-stage1-reconstruction
```

When HDF5 provisioning and optional model dependencies live in different
shared environments, first materialize a small development-only batch cache
with the HDF5-capable interpreter, then run each dependency group against that
cache.  The cache manifest hashes every batch and records
`frozen_test_i_blocks_loaded=[]`; it is a disposable `_tmp` artifact, not a
tracked dataset or a frozen-test export.

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/hdf5/env/bin/python \
  _pipelines/02_task_datasets/reconstruction/p5_stage1.py \
  --prepare-cache-only --data-dir /path/to/reconstruction \
  --batch-cache _tmp/p5-stage1-cache

PYTHONDONTWRITEBYTECODE=1 /path/to/model/env/bin/python \
  _pipelines/02_task_datasets/reconstruction/p5_stage1.py \
  --mode both --models neuralop_fno3d \
  --batch-cache _tmp/p5-stage1-cache \
  --output-root _tmp/p5-stage1-reconstruction --device cuda:0 \
  --fail-on-failed
```

The audited first execution is summarized without host paths or bulky
checkpoints in `p5_stage1_results.json`.  These are one-step Stage-1 contract
diagnostics, not CV scores, rankings or frozen-test metrics.

The legacy file names `train.h5` and `test.h5` are physical containers from
P3.  P5's scientific firewall is the mode-specific frozen I-block contract;
the runner asserts that selected arrays contain only development I-blocks and
archives `frozen_test_i_blocks_loaded=[]` for every result.

P4 strict is intentionally stricter than the historical strict baseline below:
because the available sparse porosity values were sampled from Eclipse target
cells, P4 strict excludes every well-porosity/IDW value and uses only seismic
attributes plus coordinates. The following protocol bullets document the
unchanged historical results and must not be read as the P4 input contract.

Two explicitly separate evaluation protocols are available:

- **Conditional reconstruction, NOT strict spatial holdout.**  Historical
  train I-blocks 0–3 and test I-blocks 4–5 use all 91 global constraints.  Of
  those, 90 lie inside the test region.  Exact well cells are excluded from
  metrics, while IDW intentionally propagates their values nearby.
- **Strict spatial holdout.**  A deterministic reverse split uses I-blocks 4–5
  for training, I-block 3 as an unused guard, and I-blocks 0–2 for testing.
  Constraint counts are 90/1/0 respectively; strict IDW receives only the 90
  train-region constraints.  This is one-well spatial extrapolation, not
  cross-well or cross-field generalization.

## What was actually parseable

- The Eclipse ZIP contains 65 files (1.70 GB uncompressed).  The useful static
  reference is fully available as `VOLVE_2016.GRID`, `VOLVE_2016.INIT`, the
  68.5 MB ASCII GRDECL grid, and ASCII `PHIF_NW`/`ACTNUM_2013` properties.
- `resdata==6.0.1` was tested in a temporary local install.  It reads the final
  grid as `108 x 100 x 63`, with 680,400 total and 183,545 active cells.  Its
  active `PORO` vector is identical to the result from the dependency-free
  Eclipse unformatted-record reader in `build_dataset.py`.
- `VOLVE_2016.h5` is valid HDF5, but contains simulator summary vectors rather
  than the static 3-D property grid, so it is not used as the reconstruction
  label.
- The RMS ZIP contains 5,325 files (9.80 GB uncompressed).  RMS project
  `realisation` files have a readable GEOMATIC text header and binary payload.
  The 508,622 big-endian float32 values in `merge_pp04b_PHIF_NW` have exactly
  the same sorted multiset as the non-zero Eclipse `PHIF_NW` values.
- RMS internal cell ordering/geometry was not reconstructed: it depends on the
  proprietary project index.  The `Resque.bin.66` export was identified by its
  `Rescue Geometry File` signature, but its spatial layout was not guessed.
  Therefore Eclipse supplies the validated spatial mapping and RMS is an
  independent property-value cross-check.

The complete machine-readable audit is in `model_inspection.json`.

## Dataset task

The target is final Eclipse porosity (`INIT/PORO`) on the active simulator
cells.  The `63 x 100 x 108` volume is tiled into `9 x 20 x 18` labels.  The
east one-third (two horizontal I-blocks) supplies a contiguous test-label set:
140 train patches / 134,222 active cells and 70 test patches / 49,323 active
cells.  This label split must not be described as a pure spatial generalization
holdout because the global well constraints cross that boundary.

Each `seismic_patch` has nine channels:

1. post-stack seismic amplitude at the mapped cell time;
2. local five-sample RMS;
3. vertical amplitude gradient;
4. normalized UTM X;
5. normalized UTM Y;
6. normalized depth;
7. sparse observed well porosity;
8. sparse well mask;
9. Eclipse active-cell mask.

`well_log_seq` holds the global sparse observation table.  Only 15/9-19 SR
genuinely intersects the final active simulator grid, yielding 91 unique cell
constraints.  Their positions follow the weak real-well trajectory, but their
porosity values are sampled from Eclipse reference cells rather than an
independent measured PHIE log; this is a controlled conditional benchmark.
15/9-19 A and BT2 are not forcibly snapped into the model.  The Layer-1 time
mapping is weak because its depth axis is MD while the model cell depth is
TVD-like; it is not a checkshot/VSP tie.

The currently saved HDF5 files predate the terminology correction and still
contain `blocked east holdout` in sample metadata.  They are treated as
immutable source containers.  Runtime mode audits and mode-specific results
are authoritative; the shared HDF5 is not rebuilt because this track's write
scope excludes `_data/`.

## Shared training skeleton

- Coordinate min-max normalization and model feature/target normalization use
  `ml_framework.preprocess`; every normalization is paired with
  `denormalize`, and the measured round-trip errors are recorded.
- Seismic input explicitly passes through `denoise_identity`.  Local RMS is a
  separate attribute, not smoothing applied in place of sharp amplitudes.
- The model lives in `models/ridge_linear.py` and registers `build_model()` as
  `ridge_linear`.  `get_model(..., models_package="models")` dynamically imports
  `models/<registered-name>.py`; adding that same-name file is the entire model
  swap, with no manual import or training-script edit.
- The available plugins are the canonical `ridge_linear`, lightly regularised
  NumPy `reconstruction_linear_sgd`, and single-hidden-layer NumPy
  `reconstruction_tiny_mlp`.  The two alternatives share the same adapter and
  dynamic-import contract, but have not been assigned formal track metrics;
  the results below remain the unchanged Ridge evidence.
- Both structural and seismic variants call `ml_framework.train.train_loop`
  for 600 complete epochs without early stopping.  A central K-depth block is
  used for validation while all four training I-blocks remain represented.
- The fixed factory API is used as
  `train_batches_fn/val_batches_fn = lambda: [(features, target)]`; every call
  constructs a fresh iterable.  The pre-fix code passed ordinary reusable
  lists, not one-shot generators, so it was not exposed to the exhausted-
  generator/zero-loss bug.
- Best and last checkpoints, full histories and shared-framework loss plots
  are stored under `_outputs/<model>/<evaluation-mode>/`.  The plots mark the minimum-validation
  epoch selected for evaluation.  A second zoomed validation-loss plot shows
  the neighborhood of that minimum.  The curve audit requires every epoch in
  the final sustained 50-epoch window to remain more than 1% above the best
  validation loss while mean train loss remains below its value at the best
  epoch; a single endpoint or anomalous spike cannot pass it.

## Dependencies and data provision

Run entrypoints from the project root with `python3`.  Runtime packages are
NumPy, SciPy, Matplotlib, h5py and Pillow; rebuilding the source dataset also
requires `segyio`.  The shared project environment, rather than a vendored
`.venv` or local `site-packages`, must provide them.

The portable Git evidence does not contain raw/model data or checkpoints.
Provision these paths before running data-dependent checks:

- `_sandbox/volve_data/` with the Volve model ZIPs and extracted seismic;
- `_pipelines/01_common_preprocess/outputs/seismic_index.npz` and
  `well_tie_weak.npz`;
- `_data/processed/reconstruction/train.h5` and `test.h5` through the shared
  `dataset_io` workflow;
- mode checkpoint paths recorded in `results_conditional.json` and
  `results_strict.json` when checkpoint inference is required.

Checkpoints, HDF5, `_tmp`, caches and local dependency targets are ignored by
Git.  Canonical portable evidence consists of mode results/manifests,
preprocess/history JSON, loss plots, and conditional/strict prediction figures.
The historical unsuffixed `results.json`, root checkpoint tree and
`prediction_visualization.png` remain local noncanonical snapshots.

The saved HDF5 predates the terminology correction and contains an old
`blocked east holdout` string.  Code and tests must never use that prose field
to define current evaluation semantics.  `MODE_I_BLOCKS` plus the
mode-specific result audit are authoritative; the HDF5 is not rewritten merely
to update wording.

## Verification order

The default contract gate needs no HDF5, Layer1 files or checkpoints and must
pass in a clean checkout after Python dependencies are installed:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_contract.py' -v
```

After data provision, verify the unified dataset and then run the explicit
integration/data-dependent gate.  Missing assets produce one clear `SKIP`, not
a false failure or a silently weakened test:

```bash
python3 _code/dataset_io.py stats reconstruction/train
python3 _code/dataset_io.py stats reconstruction/test
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_dual_evaluation.py' -v
```

Portable entrypoint smoke checks are read-only:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 _pipelines/02_task_datasets/reconstruction/build_dataset.py --help
PYTHONDONTWRITEBYTECODE=1 python3 _pipelines/02_task_datasets/reconstruction/baseline.py --help
PYTHONDONTWRITEBYTECODE=1 python3 _pipelines/02_task_datasets/reconstruction/visualize_prediction.py --help
```

## Full reproduction commands

These commands write datasets, train models or regenerate figures; run them
only after data provision when a full reproduction is intentionally requested:

```bash
python3 _pipelines/02_task_datasets/reconstruction/build_dataset.py --inspect-only
python3 _pipelines/02_task_datasets/reconstruction/build_dataset.py
python3 _pipelines/02_task_datasets/reconstruction/baseline.py --model ridge_linear --evaluation-mode conditional --epochs 600
python3 _pipelines/02_task_datasets/reconstruction/baseline.py --model ridge_linear --evaluation-mode strict --epochs 600
python3 _pipelines/02_task_datasets/reconstruction/visualize_prediction.py --evaluation-mode both
```

The baseline uses 3-D inverse-distance interpolation from the mode-allowed
sparse constraints, followed by train-only ridge regression with seismic and
coordinate features.  `results_conditional.json` and `results_strict.json`
contain separate measured comparisons and machine-readable constraint audits.
The historical `results.json` and original checkpoints remain untouched.

For **conditional** evaluation on 49,233 unobserved active test cells, the train-mean baseline has
RMSE 0.023887.  Sparse-well IDW alone is worse (0.032544); ridge with IDW and
coordinates reaches 0.021751; adding the three real seismic attributes reaches
0.021589.  This is a 9.62% RMSE improvement over the train mean, but only 0.74%
over the structural ridge.  Shuffling test seismic gives RMSE 0.021711,
confirming that the present weak time tie yields only a small seismic
contribution.  The seismic model's validation minimum is epoch 203; validation
loss remains at least 1% above that minimum throughout the final 50 epochs,
while mean training loss over the same sustained window remains lower than at
epoch 203.

For **strict** evaluation on 78,949 active test cells with no test-region well
values, train-mean RMSE is 0.034892, IDW is 0.031470, structural ridge is
0.033090, and seismic ridge is 0.032047 (MAE 0.024732, Pearson -0.082431,
R² -0.084070).  Shuffled-test-seismic RMSE is 0.032472.  The strict seismic
validation minimum is epoch 207.  These weak/negative generalization metrics
are reported as-is; they are not evidence of cross-well or cross-field skill.

Constant-prediction Pearson correlation is mathematically undefined.  New mode
results store it as `null` with `pearson_r_defined=false` and a reason instead
of the historical fake `0.0` convention.  Every defined numeric metric is
asserted finite.

## Reproducibility and evidence

- `run_manifest_<mode>.json` records the exact command, environment versions,
  Git HEAD, source hashes, HDF5 hashes and checkpoint/history/plot hashes.
- `_tests/test_dual_evaluation.py` verifies unified schema, region zero-overlap,
  strict guard/test constraint exclusion, train-only normalization,
  best-checkpoint selection, finite/explicitly-undefined metrics and real
  checkpoint visualization reproduction.
- `_outputs/prediction_visualization_conditional.png` and
  `_outputs/prediction_visualization_strict.png` show real Eclipse reference
  slices, best-checkpoint predictions, loss curves, metrics and mode caveats.

After shared-framework fix `62d68fa`, the same 600-epoch run was repeated end
to end.  Both train/validation histories are bit-for-bit identical to the
pre-fix evidence (maximum absolute difference 0.0); best epochs remain 202
(structural) and 203 (seismic), and all test metrics and artifact checksums are
unchanged.
