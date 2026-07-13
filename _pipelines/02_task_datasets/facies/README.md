# Seismic facies reference pipeline

`facies_f3` and `facies_penobscot` are independent segmentation tasks. Their
integer IDs do not share geological semantics, so data, normalization,
checkpoints, confusion matrices, and metrics are never combined.

## Fixed label schemas

| Task | Valid IDs | Classes | Ignore index |
|---|---:|---:|---:|
| `facies_f3` | 0–9 | 10 | none |
| `facies_penobscot` | 0–7 | 8 | none |

The F3 release declares ten classes. Penobscot's published `dataset-log.txt`
declares `num_classes=8`, and the HDF5 values are exactly 0–7; this pipeline
therefore treats all eight IDs as valid. The Zenodo landing-page prose says
seven classes, which conflicts with the downloadable log and is recorded here
rather than silently changing ID 0 into a void class. `num_classes` always
comes from `pipeline_contract.py`; test labels are never used to infer model
width.

## Leakage controls and data QA

Only inline sections are used. F3 crosslines are excluded because orthogonal
views intersect the same voxels. Both datasets use lower train, middle guard,
and upper test regions:

| Task | Saved train | External guard | Test | Saved train/test patches |
|---|---:|---:|---:|---:|
| F3 | 100–586 | 587–619 | 620–750 | 1948 / 445 |
| Penobscot | 1000–1448 | 1449–1479 | 1480–1600 | 1766 / 474 |

The saved train split is partitioned again before fitting or model selection:

| Task | Model train | Validation guard | Validation |
|---|---:|---:|---:|
| F3 | 100–463 | 464–488 | 489–586 |
| Penobscot | 1000–1335 | 1336–1358 | 1359–1448 |

All boundaries are asserted ordered and disjoint. The test HDF5 is not loaded
by `train_baseline.py` until the minimum-validation-loss checkpoint has already
been selected.

Patch validity is deterministic and label-independent. A raw patch is filtered
when its peak-to-peak amplitude is at most `1e-6`, or when one exact amplitude
value occupies at least 50% of its pixels (mostly fill-valued/no-data patch).
The rule is identical for every split. It removed F3 train/test `0/79` and
Penobscot train/test `30/10`; all retained inlines and configured classes remain
represented.

There is no default smoothing. The builder explicitly calls
`denoise_identity`, because sharp seismic events may be real geology. One
global z-score is fitted only on valid model-train patches and reused unchanged
for saved-train, validation, and test samples:

| Task | Fit patches | Mean | Std | Max inverse error |
|---|---:|---:|---:|---:|
| F3 | 1456 | 0.926528 | 2515.237793 | 0.001953125 |
| Penobscot | 1314 | -0.592729 | 1468.020020 | 0.001953125 |

The local integration-only `data_build_manifest.json` records split ranges,
label histograms, filter counts, normalization parameters, source/processed
SHA-256 hashes, and build configuration. It may contain machine-specific data
provisioning paths and is intentionally excluded from Git. The portable
`dataset_audit.json` is canonical evidence produced by independently reading
all saved samples and failing on overlap, schema violations, inconsistent
normalization, duplicate coordinates, or retained fill-valued patches.

## Default contract tests

The default test suite is asset-free: it does not open local HDF5 files or
checkpoints. From the project root, with the environment activated:

```bash
python3 -m unittest discover \
  -s _pipelines/02_task_datasets/facies/tests -v
```

## External data provisioning and integration audit

Raw data is never committed. Provision an external directory with this
layout, then pass it explicitly with `--data-root`:

```text
<FACIES_DATA_ROOT>/
├── f3demo/
│   ├── inlines.zip
│   └── masks.tar.gz
└── penobscot/
    └── dataset.h5
```

Example from the project root:

```bash
export FACIES_DATA_ROOT=/path/to/f3_penobscot

python3 _pipelines/02_task_datasets/facies/build_dataset.py \
  --data-root "$FACIES_DATA_ROOT" --normalization zscore

python3 _pipelines/02_task_datasets/facies/audit_dataset.py

python3 _code/dataset_io.py stats facies_f3/train
python3 _code/dataset_io.py stats facies_f3/test
python3 _code/dataset_io.py stats facies_penobscot/train
python3 _code/dataset_io.py stats facies_penobscot/test
```

Real checkpoint verification is an explicit integration entry, not part of the
default seven contract tests. It requires locally retained processed HDF5 and
both ignored `best.ckpt`/`last.ckpt` files:

```bash
python3 _pipelines/02_task_datasets/facies/verify_artifacts.py
```

## Model swapping and training

The unchanged baseline architecture is `models/small_unet.py`. Models are
dynamically discovered; adding a model requires only a new
`models/<name>.py` containing decorated
`@register_model("<name>") build_model(...)`, then selecting `--model <name>`.
There are no manual model imports.

Available models:

- `small_unet`: the unchanged two-level U-Net used for the reported baseline.
- `facies_linear_pixel`: a 1x1 per-pixel linear classifier with no spatial
  context.
- `facies_tiny_fcn`: a shallow three-convolution FCN with local spatial
  context.

Only `small_unet` produced the canonical metrics below; the alternatives are
swappable architecture references and do not change the reported results.

Training uses the shared `ml_framework.train.train_loop` with repeatable
zero-argument DataLoader factories. It records train and validation loss every
epoch, writes `best.ckpt` on minimum validation loss and `last.ckpt` every
epoch, never early-stops, and plots the full loss curve with the best marker.

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> \
  python3 \
  _pipelines/02_task_datasets/facies/train_baseline.py --task facies_f3

CUDA_VISIBLE_DEVICES=<free_gpu> \
  python3 \
  _pipelines/02_task_datasets/facies/train_baseline.py --task facies_penobscot
```

Defaults are 40 epochs for F3 and 120 for Penobscot. Penobscot was extended
from 80 because epoch 80 was still the best point; the 120-epoch run exposes a
real post-best rise without changing data, architecture, optimizer, or other
hyperparameters.

## Final real-data results

Both tasks use SmallUNet (`base_channels=8`), batch size 16, Adam at `1e-3`,
inverse-square-root model-train class weights, horizontal/vertical flips, and
seed 2693. Metrics include every configured class and no ignored pixels.

| Task | Epochs | Best epoch | Best / last val loss | Accuracy | mIoU | Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| F3 | 40 | 32 | 0.329916 / 0.349452 | 0.591454 | 0.436795 | 0.576185 |
| Penobscot | 120 | 112 | 0.167969 / 0.246452 | 0.913217 | 0.690189 | 0.809002 |

Each `metrics.json` also contains per-class support/IoU/F1, confusion matrix,
evaluated/ignored pixel counts, full history, split ranges, normalization,
command, Git HEAD, and Python/NumPy/PyTorch/CUDA/cuDNN versions. All aggregate
and per-class metrics are required to be finite.

Portable canonical Git evidence is deliberately small:

- `_outputs/leakage_fixed_v2/dataset_audit.json`
- `_outputs/leakage_fixed_v2/artifact_verification.json`
- `_outputs/leakage_fixed_v2/prediction_visualization_evidence.json`
- `<task>/small_unet/{history.json,loss_curve.png,metrics.json}`
- `_outputs/prediction_visualization.png`

Local integration assets are retained on disk but ignored by Git:
`data_build_manifest.json`, all `*.ckpt`/`*.pt`, processed/raw HDF5, the
pre-fix `_outputs/facies_*` trees, and `_outputs/legacy`. The canonical metrics
and figures retain the scientific result without committing machine-local
data, duplicate historical outputs, or model binaries.

## Real prediction visualization

```bash
CUDA_VISIBLE_DEVICES=<free_gpu> \
  python3 \
  _pipelines/02_task_datasets/facies/visualize_predictions.py
```

The script loads each real `best.ckpt`, reads real held-out test samples through
`dataset_io`, runs inference, inverse-normalizes seismic input, and writes the
required `_outputs/prediction_visualization.png`. Selection uses deterministic
interior quantiles among test patches containing at least two ground-truth
classes; predictions and prediction quality are never used for selection. The
sidecar records checkpoint path, original test index, inline, GT IDs, and the
annotated Accuracy/mIoU/Macro-F1.

## Environment and reproducibility note

The tested environment uses Python 3.10.12, NumPy 2.2.6, PyTorch 2.11.0+cu128,
CUDA 12.8, and cuDNN 9.19. A local ignored environment can be recreated with:

```bash
uv venv \
  --python python3 --system-site-packages \
  _pipelines/02_task_datasets/facies/.venv

uv pip install \
  --python _pipelines/02_task_datasets/facies/.venv/bin/python3 \
  --torch-backend cu128 \
  -r _pipelines/02_task_datasets/facies/requirements.txt

source _pipelines/02_task_datasets/facies/.venv/bin/activate
python3 -m unittest discover \
  -s _pipelines/02_task_datasets/facies/tests -v
```

CUDA 2-D cross-entropy warns that its kernel is not bitwise deterministic on
this GPU. Seeds and deterministic settings are still fixed, but results should
be compared with a small numeric tolerance rather than claimed bit-identical.
