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
