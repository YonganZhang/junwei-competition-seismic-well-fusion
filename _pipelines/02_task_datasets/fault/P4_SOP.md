# Fault P4 training, evaluation, and reproducibility SOP

This SOP applies only to track ① fault segmentation. The shared P4 contract is
read-only here; track code adapts its scientific semantics to that contract.

## 1. Preflight before any model run

From the project root:

```bash
python3 _pipelines/02_task_datasets/fault/p4_smoke.py
```

The command is intentionally fast. It reads the real Layer1 fault coordinates,
seismic index, and canonical `audited_v2` reports; it does not read a test
loader, rebuild HDF5, train on real data, run HPO, or consume a frozen test. It
also runs a tiny deterministic baseline only on synthetic explicitly verified
positive/negative labels.

Expected portable outputs live in `_outputs/p4_preflight/`:

- `task_spec.json`: strict task/loss/metric/mask/figure contract;
- `blind_test_manifest.json` or `blind_test_not_feasible.json`, never both;
- `buffered_cv_plan.json`: requested/effective folds and buffer evidence;
- `fixed_baselines.json` and `hpo_plan.json`: plans, not claimed trials;
- `tiny_baseline_smoke.json`: synthetic contract evidence only;
- `smoke_report.json` and `manifest.json`.

## 2. Frozen label semantics

- Rasterized fault-stick voxels are positive and valid.
- An unlabelled voxel is unknown and has `valid_label_mask=false`; its stored
  target zero is only a masked storage placeholder.
- A negative becomes valid only inside a separately audited complete annotation
  coverage region.
- Historical `non_fault` samples are weak negatives. Their unlabelled voxels are
  marked in `proxy_mask` but remain outside `valid_label_mask`.
- Proxy supervision is available only through the explicitly named
  `proxy_regression` mode. Its metrics cannot be promoted to formal CV/test.

Official loss, thresholding, AP/PR, Dice/IoU, boundary, and component metrics
must index `valid_label_mask`. Proxy and unknown voxels cannot enable a fold.

## 3. Blind-test audit

`audit_blind_test()` searches explicit annotation-coverage evidence before a
model runs. A candidate must be contiguous, completely audited, contain both
positive and verified-negative labels, and be disjoint from historical training
and observed-test exposure.

If no block meets all conditions, the pipeline writes
`blind_test_not_feasible.json`. The old `audited_v2` test stays regression
evidence and the frozen-test lifecycle cannot be consumed. Do not relabel the
old test as blind or use unknown/proxy voxels as formal negatives.

## 4. Buffered development CV and HPO

The requested fold count is five. `build_buffered_spatial_cv()` partitions
contiguous inline blocks and globally excludes samples inside boundary buffers.
It searches downward from five and records an honest reduction when independent
block or binary label support is insufficient. Fewer than two supported blocks
returns `status=not_feasible`, not random patch K-fold.

`run_buffered_development_cv()` has no test argument. Every retained development
sample must receive exactly one archived OOF prediction. The primary AP score is
weighted by valid-label count; fold mean/std/worst are also archived.

`run_fault_fixed_trials()` likewise has no test argument and fixes direction to
`maximize`. `fault_hpo_plan()` records the optional 8/20 sanity/TPE budget, but
the preflight smoke never executes HPO. Run a real HPO campaign only after the
blind split, valid negatives, fold-specific preprocessing, budget, and recovery
directory are frozen.

## 5. Baselines, output, checkpoint, and lifecycle

The three existing SGD baselines are wrapped, not rewritten. Their probability
output is converted to finite `[B,D,H,W]` logits/probabilities through
`ModelOutput`; sigmoid remains an inference transform in the TaskSpec.
Unknown voxels receive zero training weight. Official baseline training fails if
either verified positives or verified negatives are absent. Weak negatives need
the explicit proxy mode and retain the proxy result label.

`FaultRunContext` writes the TaskSpec, seed/environment reports, split/blind
audit, lifecycle, full resumable checkpoint, and content-addressed artifact
manifest. It follows the shared one-way lifecycle:

```text
SPLIT_LOCKED -> SMOKE_PASSED -> CV_COMPLETE -> CONFIG_FROZEN
-> REFIT_COMPLETE -> TEST_CONSUMED
```

`TEST_CONSUMED` is permitted once and only when the blind audit status is
`frozen`; hashes must match the locked split, frozen config, and refit
checkpoint.

## 6. Archived-prediction visualization

The visualizer accepts only `--prediction`, `--metrics`, and `--output-dir`:

```bash
python3 _pipelines/02_task_datasets/fault/p4_visualization.py \
  --prediction <run>/frozen_test/predictions.npz \
  --metrics <run>/frozen_test/metrics.json \
  --output-dir <run>/visualizations
```

The NPZ must contain `amplitude`, `target`, `valid_label_mask`, `proxy_mask`, and
`probability` volumes. Metrics must contain a fixed `pooled_oof` threshold and
config/split/checkpoint provenance hashes. The command never loads a model or
selects a threshold. It creates:

- input / masked GT / probability / TN-TP-FP-FN;
- three orthogonal probability views;
- PR and threshold diagnostics using the already fixed threshold;
- valid-region boundary and connected-component diagnostics.

## 7. Fast acceptance commands

```bash
python3 -m unittest discover -v \
  -s _pipelines/02_task_datasets/fault -p 'test_fault_pipeline.py'
python3 -m unittest discover -v \
  -s _pipelines/02_task_datasets/fault -p 'test_fault_p4.py'
python3 -m unittest discover -v \
  -s _pipelines/02_task_datasets/fault -p 'test_fault_real_data.py'
python3 _pipelines/02_task_datasets/fault/p4_smoke.py
```

The third command is the historical binary/HDF5 integration gate and skips
clearly when those optional local assets are absent. The fourth command is the
new read-only real-coordinate P4 smoke and does not require active HDF5.
