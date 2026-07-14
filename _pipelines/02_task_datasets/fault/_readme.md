# Fault detection — track ①

This track builds a reproducible real-Volve fault-stick segmentation dataset
and runs the deliberately simple `fault_local_logistic` pipeline baseline. The
baseline validates data and training plumbing; it is not the final architecture.

## Preservation contract

The pre-enhancement hashes are frozen in `historical_baseline_manifest.json`.
Historical checkpoints, models and duplicate images remain optional local
integration assets and are not part of the portable merge payload. If the full
historical bundle is present, audited commands still verify every hash; a
partial bundle fails loudly. Verify it explicitly with:

```bash
python3 _pipelines/02_task_datasets/fault/verify_historical_assets.py
```

The shared HDF5 files at `_data/processed/fault/{train,test}.h5` are rebuilt by
the audited builder because `dataset_io` defines those as the active dataset
paths. Their old hashes remain in the historical manifest; each new run records
the active HDF5 hashes in both build and training reports.

## Real-data preparation

Real Volve data are intentionally not stored in Git. Before an end-to-end run,
prepare the project data workspace described by `_meta/_data_registry.yml` and
run Layer1 common preprocessing so these inputs exist:

- `_pipelines/01_common_preprocess/outputs/fault_points.npz`;
- `_pipelines/01_common_preprocess/outputs/seismic_index.npz`;
- exactly one extracted ST0202 stack SEG-Y under the project `_sandbox/volve_data/`
  layout expected by `build_dataset.py`.

The builder then writes the ignored active HDF5 files through the shared
`dataset_io.save_split("fault", ...)` interface. A clean source checkout can run
the default unit gate without these assets; the real-data gate below reports a
clear skip until they are available.

## End-to-end commands

Run from the project root with the project Python:

```bash
python3 _pipelines/02_task_datasets/fault/build_dataset.py --run-name audited_v2
python3 _pipelines/02_task_datasets/fault/baseline.py --model fault_local_logistic --run-name audited_v2 --epochs 120
python3 _pipelines/02_task_datasets/fault/visualize_predictions.py --run-name audited_v2
python3 _code/dataset_io.py stats fault/train
python3 _code/dataset_io.py stats fault/test
```

Portable default unit gate (no real HDF5 or historical binaries required):

```bash
python3 -m unittest discover -v -s _pipelines/02_task_datasets/fault -p 'test_fault_pipeline.py'
```

Explicit data-dependent integration gate (runs all strong assertions when the
local assets exist; otherwise reports one clear skip):

```bash
python3 -m unittest discover -v -s _pipelines/02_task_datasets/fault -p 'test_fault_real_data.py'
```

Use a new `--run-name` for another audited run. Never reuse a name when the
intent is to preserve a previous audited result.

For an exact reproducibility check, repeat the three pipeline commands with a
second run name and compare them:

```bash
python3 _pipelines/02_task_datasets/fault/verify_reproducibility.py --reference audited_v2 --candidate audited_v2_repro
```

The verifier requires identical input/data hashes, split manifest,
normalization, rasterization audit, threshold, metrics, model/checkpoint hashes,
loss curve and prediction visualization. It writes
`audited_v2/reproducibility_report.json` and fails on any mismatch.
The duplicate candidate run is a local integration artifact and is ignored by
Git; regenerate it when an exact reproduction audit is required.

## Split and leakage contract

- Samples are 2-D crossline/time patches at one fixed inline.
- Default ranges are derived deterministically from the full inline grid:
  train, an eight-inline guard band, then test.
- The builder asserts pairwise-zero train/guard/test inline overlap, rejects
  duplicate centres, and writes every centre to `split_manifest.json`.
- Training makes a second spatial fit/validation split with a two-inline guard.
- Test is not used for class weights, checkpoint selection, or threshold
  selection. The threshold is selected from validation predictions only.
- Named geological faults can extend across both spatial blocks. The build
  report counts these entities explicitly. The guard proves sample-grid
  separation but does not pretend that a long geological fault becomes a new
  entity at the boundary.

## Label contract

- Layer1's historical `stick_no` field contains source vertex codes
  `1=start, 2=middle, 3=end`; it is not a persistent stick identifier.
- Consecutive vertices within each complete stick are connected at nearest 3-D
  voxel centres. The run records point/stick/fault counts, TWT snapping error,
  rasterization method, and voxel count.
- Dilation radii are explicitly `{inline: 0, crossline: 0, time: 0}`. No fault
  thickness or surface interpolation between sticks is invented.
- The raw target is therefore a sparse interpreted-stick skeleton, not a dense
  fault surface. Annotation-free patches are weak negatives: they contain no
  rasterized stick voxel but may still contain an uninterpreted fault.
- Positive and negative patch centres are balanced. Negatives are restricted to
  the annotated time support so late-time padding cannot become an easy cue.

## Preprocessing and metrics

- `ml_framework.preprocess.denoise_identity` is called explicitly. Sharp
  amplitudes may be real geology, so no smoothing is applied by default.
- One z-score `NormStats` is fitted only from the spatial train-fit subset,
  stored in every sample, and reused unchanged for validation and test. Neither
  validation nor test is refitted. Build and training must use matching
  `--val-fraction` and `--val-guard-inlines` values.
- Every sample is checked through `normalize -> denormalize`; non-finite values
  or excessive round-trip error fail loudly.
- Patch predictions are averaged at repeated physical `(inline, crossline,
  time_index)` voxels before scoring. Metrics therefore do not count overlapping
  patch coverage multiple times.
- Reports include precision, recall, F1/Dice, IoU, average precision, PR-AUC,
  physical-coverage counts, and an all-negative reference.

## Model and shared training contracts

Models live under `models/`. `models/<name>.py` exports a decorated
`build_model()` and is discovered dynamically with
`get_model(name, models_package="models")`; no central import list is required.

Three deliberately simple model adapters are selectable through `--model`:

- `fault_local_logistic` (default) retains raw amplitude, local 3×3 mean/std,
  and Sobel features with an incremental logistic classifier;
- `fault_raw_logistic` is a weighted incremental logistic classifier using only
  the raw amplitude at each voxel;
- `fault_local_huber` uses the same kind of local amplitude/statistic/gradient
  features with a modified-Huber incremental classifier.

The two alternatives are interface-verified options, not new audited benchmark
claims. The formal `audited_v2` metrics remain those of `fault_local_logistic`.
`baseline.py` still delegates all epochs to `ml_framework.train.train_loop()`
using zero-argument reusable `train_batches_fn` and `val_batches_fn` factories.
Empty batches fail loudly, train/validation loss is recorded every epoch, and
minimum-validation-loss `best.ckpt` is distinct from `last.ckpt`.

## Audited run outputs

`_outputs/runs/<run-name>/` contains:

- `build_summary.json` — inputs, versions, hashes, split, normalization,
  rasterization and label-risk audit;
- `split_manifest.json` — exact train/test sample centres and kinds;
- `checkpoints/{best,last}.ckpt` and `checkpoints/history.json`;
- `baseline_model.joblib`, `baseline_metrics.json`, `loss_curve.png`;
- `prediction_visualization.png` and `visualization_report.json`.
- `reproducibility_report.json` after an exact second-run comparison.

Git keeps only the portable canonical `audited_v2` metrics, build/split reports,
loss curve, prediction image and small audit reports. Checkpoints, joblib models,
HDF5 files, historical duplicate outputs, noncanonical runs and
`audited_v2_repro` remain local. Consequently, best-checkpoint inference is an
explicit integration gate rather than a clean-checkout unit requirement.

Canonical reports intentionally retain the source hashes that produced the
recorded real run. Later portability-only changes to asset gating, tests,
documentation or ignore rules do not rewrite those hashes and do not claim a
new training run.

Architecture upgrades such as FaultSeg3D/3-D U-Net, focal loss, or semi-supervised
learning are intentionally out of scope until the owner chooses a model route.

## P4 sparse-label adapter

The P4 adapter is implemented in `p4_contract.py`, `p4_split.py`,
`p4_workflow.py`, and `p4_visualization.py`; the operational contract is
documented in `P4_SOP.md`. It preserves the three historical SGD baselines while
changing their scientific role:

- rasterized fault-stick voxels are the only default valid positives;
- all other voxels remain unknown with `valid_label_mask=false`;
- historical annotation-free samples are separately marked `proxy_mask` and
  cannot become formal negatives or enable a CV fold;
- a real negative requires explicit complete-annotation coverage evidence.

Run the fast P4 gate from the project root:

```bash
python3 -m unittest discover -v -s _pipelines/02_task_datasets/fault -p 'test_fault_p4.py'
python3 _pipelines/02_task_datasets/fault/p4_smoke.py
```

The smoke first audits real coordinates and label coverage. If an unconsumed,
contiguous, completely annotated block exists, it freezes that block before any
model run. Otherwise it writes an auditable `blind_test_not_feasible.json`,
keeps `audited_v2` as regression evidence only, and forbids frozen-test
consumption. The current canonical data contain no complete-annotation coverage
record, so lack of a blind test is reported rather than hidden.

Development uses requested five-fold contiguous spatial blocks with global
inline buffers. The adapter reduces folds only for independent-block or valid
binary-label support and writes the reason; proxy counts never satisfy the
negative-label gate. Development CV and fixed-trial HPO APIs expose no test
argument. The default smoke records the optional HPO plan but does not run HPO.

The legacy baseline adapter returns finite `[B,D,H,W]` raw logits plus sigmoid
probabilities. Unknown voxels have zero official training weight; absent audited
negatives make official training fail loudly. Full resumable checkpoint,
artifact manifest, exact OOF, and one-shot test lifecycle entry points live in
`p4_workflow.py`.

`p4_visualization.py` only reads archived prediction/metric files. It cannot
load a model or select a threshold, and produces the required masked
input/GT/probability/confusion, orthogonal, PR/threshold, and
boundary/components figures with source hashes.

## P5 Stage-4 confirmation gate

The deterministic Stage-4 gate consumes only the frozen Stage-3 manifests and
hash-verifies their existing readiness SVGs. Because fault has zero legal
development folds, no audited negatives, and no frozen winner, it records a
portable `blocked/not_rankable` confirmation without refitting a model or
opening a holdout:

```bash
python3 _pipelines/02_task_datasets/fault/fault_p5_stage4.py
python3 -m unittest discover -v -s _pipelines/02_task_datasets/fault -p 'test_fault_p5_stage4.py'
```

Tracked confirmation evidence lives under `_outputs/p5_stage4_confirmation/`.
It references the Stage-3 data-gate visualization by path and SHA-256; it does
not copy those figures or fabricate prediction artifacts.
