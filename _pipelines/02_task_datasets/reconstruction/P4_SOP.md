# P4 SOP — Volve 3-D reconstruction

This SOP adds a new experiment contract around the existing Ridge+IDW,
linear-SGD and tiny-MLP baselines. It does not replace or reinterpret the
historical `results_conditional.json` and `results_strict.json` evidence.

## Independent tasks

`conditional` and `strict` are separate TaskSpecs, labels, input whitelists,
split manifests, metric namespaces and run roots.

- Conditional: development I-blocks 0–3; frozen test I-blocks 4–5. Frozen-test
  inference may use test-region sparse well constraints, exact constrained
  cells are excluded from metrics, and every result must say this is not strict
  holdout generalization.
- Strict: development I-blocks 4–5; guard I-block 3; frozen test I-blocks 0–2.
  Guard/test targets, future fields, reference-derived features and test-region
  well values are forbidden inputs. The current sparse porosity values were
  sampled from Eclipse reference cells rather than an independent measured
  PHIE log, so strict P4 excludes every such value and has no IDW feature; it
  uses seismic attributes plus coordinates only.

Both use an explicit MSE regression loss and identity target/output transform.
The primary metric is the mode-specific RMSE and its direction is `minimize`.

## Split and CV order

1. Scan HDF5 metadata without loading labels.
2. Freeze the continuous I-block test interval.
3. Exclude the strict guard block.
4. Partition development K-blocks into requested five contiguous folds.
5. Purge one neighboring K-block on either side of each validation block.
6. Fit normalization from `fold.purge.effective_train_sample_ids` only. For
   conditional mode only, construct sparse-well IDW from that same fit subset;
   strict mode never constructs or consumes IDW.
7. Produce exactly one OOF patch prediction per development patch.

The shared SplitManifest requires candidate train plus validation IDs to cover
development. Therefore the reconstruction plugin records the actual post-
buffer fit subset in `fold.purge.effective_train_sample_ids`; tests verify it
is disjoint from both validation and the buffer.

Conditional development contains only one real sparse constraint. A fold that
holds out or purges it cannot perform IDW. Such a fold uses one declared
neutral feature value—the fold-train porosity mean—and records
`zero_constraint_fallback=fold_train_target_mean`. It never reads frozen-test
constraints during CV/HPO.

## Commands

Run from the project root. The normal data location is
`_data/processed/reconstruction/`; a read-only external provision can be
selected with `--data-dir` or `RECONSTRUCTION_DATA_DIR`.

Portable unit/contract/tiny tests (discovery is required because a directory
component starts with a digit):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_p4_reconstruction.py' -v
```

Explicit real-data smoke; this reads metadata/development arrays but performs
no model training and no frozen-test inference:

```bash
RECONSTRUCTION_DATA_DIR=/path/to/reconstruction \
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _pipelines/02_task_datasets/reconstruction/_tests \
  -p 'test_p4_real_smoke.py' -v
```

Inspect contracts and plans without writing:

```bash
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py task-specs --mode both
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py plan --mode strict
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py real-smoke
```

An intentional experiment uses one mode-specific run root:

```bash
RUN=_pipelines/02_task_datasets/reconstruction/_outputs/runs/reconstruction/strict/example
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py prepare --mode strict --run-root "$RUN"
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py cv --mode strict --model ridge_linear --epochs 20 --run-root "$RUN"
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py refit --mode strict --model ridge_linear --epochs 20 --run-root "$RUN"
python3 _pipelines/02_task_datasets/reconstruction/p4_reconstruction.py test --mode strict --run-root "$RUN"
python3 _pipelines/02_task_datasets/reconstruction/p4_visualize.py \
  --predictions "$RUN/frozen_test/predictions.npz" \
  --metrics "$RUN/frozen_test/metrics.json" \
  --output "$RUN/visualizations/strict_volume_diagnostics.png"
```

The `test` command durably advances the lifecycle to `TEST_CONSUMED` before it
loads any test-block array. A second invocation fails. `p4_visualize.py` only
accepts matching archived prediction/metric artifacts; it has no model or
dataset inference path.

## Artifacts

Each run archives TaskSpec, seed/environment reports, split manifest, HPO plan,
per-fold preprocessing/constraint audits, full resumable checkpoints, OOF
predictions/metrics, frozen config/refit state, one-shot frozen-test prediction
and metrics, visualizations and a SHA-256 manifest.

The dedicated visualization contains:

- inline truth/prediction/residual;
- crossline truth/prediction/residual;
- time/depth truth/prediction/residual;
- voxel-error histogram;
- standardized seismic-amplitude/reference/prediction distributions;
- radial FFT spectrum comparison.

Conditional and strict archives are rejected if mixed.

## HPO boundary

The frozen plan is 8 sanity plus 20 sequential TPE trials, top three configs,
three confirmation seeds, NopPruner and minimize direction. Fixed baselines run
without Optuna. No long HPO is part of the implementation smoke or this change.

## Scientific boundary

The legacy strict block has already been observed and reported. It is a useful
spatial regression test, not a new blind campaign. One Volve volume and one
intersecting physical well cannot establish cross-field or cross-well
generalization; see `not_feasible.json`.
