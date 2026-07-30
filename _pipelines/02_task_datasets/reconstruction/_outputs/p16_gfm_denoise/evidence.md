# P16 GFM masked seismic reconstruction — development-only evidence

## Outcome

- PyKrige baseline pooled development RMSE: `0.028449728170`.
- Best pretrained masked-reconstruction head (`fixed_ridge10`) gated mean-seed RMSE: `0.028588136732` (delta vs PyKrige `+0.000138408562`, positive relative gain means improvement: `-0.486502%`).
- Same encoder-decoder architecture with random initialization and identical masks: `0.028569634453`; pretrained minus random-init `+0.000018502280`.
- Direct raw no-foundation structural route: `0.028539761187`; pretrained reconstruction minus raw `+0.000048375545`.
- Five independent spatial-fold outcomes versus PyKrige: 2 win / 3 loss / 0 tie.
- Interpretation: masked reconstruction did not improve on the direct raw structural route.
- Decision: `VERIFIED_NO_PROMOTION`. No gain is automatically attributed to pretrained GFM weights.

## Genuine GFM masked-interpolation path

- Each of the three seismic attributes is forwarded separately through the real GFM encoder and all 12 decoder blocks. The checkpoint is frozen and receives no PORO supervision.
- Exactly `40/160` traces (25%) are masked per native section and seed (`len_keep=120`). Pretrained and random-init use identical paired masks.
- Output follows the upstream interpolation tutorial exactly: decoder predictions replace masked traces, while visible traces are copied from the input. Every visible OOF value passed the bitwise no-change check.
- This is masked trace interpolation, not supervised denoiser fine-tuning. The base checkpoint is not claimed to remove every possible noise process.
- The six existing P11 structural fields remain present. P16 supplements them with three reconstructed values, three reconstruction deltas, and one masked-trace indicator; Ridge preprocessing remains outer-train-only.

## Native seismic and patch alignment

- P16 reuses P15's audited mapping from development patch cells to the original continuous ST0202 SEG-Y. It reads exact `400`-sample × `160`-adjacent-trace windows; resize, interpolation and padding are all false.
- The OOF cells span 84 native inline sections, 80 crosslines and 29 samples.
- Native-input versus raw `seismic_patch[0:3]` Pearson correlations by amplitude/RMS/gradient: `0.997691`, `0.992288`, `0.997162`.
- Native-input versus fold-standardized Stage-3 structural correlations: `0.996604`, `0.990984`, `0.995175`. The 0.98 alignment floor passed for every channel.

## Five genuinely independent spatial units

The five locked outer folds are the inferential units. The three mask/model seeds are paired pseudo-repeats inside each fold, not 15 independent observations.

| fold | PyKrige RMSE | pretrained reconstruction | random-init reconstruction | raw structural | pretrained delta | outcome |
|---:|---:|---:|---:|---:|---:|:---|
| 0 | 0.027239559 | 0.027908911 | 0.027852308 | 0.027770139 | +0.000669352 | loss |
| 1 | 0.028627607 | 0.029538434 | 0.029540401 | 0.029575893 | +0.000910827 | loss |
| 2 | 0.018619291 | 0.017153951 | 0.017165903 | 0.017231110 | -0.001465340 | win |
| 3 | 0.028605050 | 0.028909131 | 0.028928278 | 0.028977340 | +0.000304081 | loss |
| 4 | 0.036338338 | 0.036124580 | 0.036072416 | 0.035917662 | -0.000213758 | win |

## Whole-fold bootstrap

- Pretrained reconstruction minus PyKrige RMSE: `+0.000138408562`; 95% interval `[-0.000551410540, +0.000694066784]`.
- Pretrained reconstruction minus random-init reconstruction: `+0.000018502280`; 95% interval `[-0.000012588985, +0.000045005567]`.
- Pretrained reconstruction minus raw structural: `+0.000048375545`; 95% interval `[-0.000061518419, +0.000150893378]`.
- Bootstrap resamples whole locked spatial folds. Seeds stay paired and voxels are never resampled independently.

## Provenance and holdout firewall

- Model: `thinkonward/geophysical-foundation-model`, snapshot `d4a33965730a506cfdb4c85fa2a0a344c53216a2`, Apache-2.0. Weight SHA-256 `c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446`; vendor source SHA-256 `ce6f97b5be889231107716d71175fe4482a678f8eb4e608c274265ca57612904`.
- PORO targets and PyKrige predictions come only from the same five hash-verified development OOF archives as P11-P15.
- `train.h5` reads are limited to patch metadata, `seismic_patch[0:3]`, `seismic_patch[3:6]`, and `seismic_patch[8]`; no HDF5 label dataset is read.
- The continuous SEG-Y supplies label-free development seismic covariates only. `test.h5`, frozen holdout paths, holdout labels and historical test metrics are neither opened nor probed.
- Gate=0 is bitwise identical to the strong PyKrige OOF baseline.
