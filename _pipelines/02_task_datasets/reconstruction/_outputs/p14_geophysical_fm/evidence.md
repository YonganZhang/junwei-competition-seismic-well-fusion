# P14 geophysical foundation model — development-only evidence

## Outcome

- PyKrige baseline pooled development RMSE: `0.028449728170`.
- Best observed pretrained GFM head (`fixed_ridge10`) gated RMSE: `0.028621657173`.
- Relative change versus baseline (positive means improvement): `-0.604326%`.
- Same-architecture random-init gated RMSE: `0.028601027108`; pretrained minus random-init: `+0.000020630065`.
- No-foundation structural gated RMSE: `0.028539761187`.
- Five independent spatial-fold outcomes: 2 win / 3 loss / 0 tie.
- The experiment does not automatically attribute any overall change to GFM pretrained weights; the matched random-init result is reported separately.
- Decision: `VERIFIED_NO_PROMOTION`. The domain-matched pretrained encoder did not produce positive pooled development gain under the locked P11 protocol.

## 3-D to 2-D slice design

- The real `train.h5` patch shape is `[K,J,I]=[9,20,18]`; 140 non-overlapping patches assemble exactly to `[63,100,72]`.
- Orientation: `KxJ vertical section at fixed I` with source shape `[63, 100]` resized bilinearly to `[400, 160]`.
- The 10,240 OOF rows touch 18 distinct fixed-I slices; no unused slice is forwarded.
- Rationale: K is retained as the vertical trace-sample axis; J provides 100 neighboring traces, closer to the pretrained width 160 than the orthogonal I width 72, reducing interpolation distortion while changing only the encoder route.
- GFM uses `patch_size=[400,1]`, so every token represents one complete vertical trace. Each OOF voxel receives the nearest resized trace token for its J coordinate plus the slice CLS token.
- All three seismic attributes are forwarded separately. Each slice is z-scored over its own active cells; inactive pixels are zero. No PORO label or cross-sample fitted normalization is used.

## Five genuinely independent spatial units

The five locked outer folds are the inferential units. The three seeds are paired optimization pseudo-repeats inside each fold, not 15 independent observations.

| fold | baseline RMSE | pretrained GFM RMSE | delta | outcome |
|---:|---:|---:|---:|:---|
| 0 | 0.027239559 | 0.028529280 | +0.001289721 | loss |
| 1 | 0.028627607 | 0.029041911 | +0.000414304 | loss |
| 2 | 0.018619291 | 0.017313699 | -0.001305593 | win |
| 3 | 0.028605050 | 0.028623353 | +0.000018303 | loss |
| 4 | 0.036338338 | 0.036325986 | -0.000012352 | win |

## Whole-fold bootstrap

- Pooled RMSE delta (candidate - baseline): `+0.000171929003`.
- 95% interval: `[-0.000505904071, +0.000773142136]`.
- Bootstrap unit is the whole locked spatial fold. Seeds stay paired and voxels are never resampled independently.

## Model provenance and protocol

- Model: `thinkonward/geophysical-foundation-model`, snapshot `d4a33965730a506cfdb4c85fa2a0a344c53216a2`, Apache-2.0 model card.
- Weight SHA-256: `c905945267bbbc58f0e1848106d182f40b5dc61273959b666a49b384cfcb7446`; vendor source SHA-256: `ce6f97b5be889231107716d71175fe4482a678f8eb4e608c274265ca57612904`.
- The encoder uses all 160 trace tokens (`len_keep=160`), so no trace is masked during feature extraction.
- Residual regression, alpha grid, inner-OOF gate, gate bounds, PyKrige baseline and fold/seed identities are reused unchanged from the committed P11 diagnostic harness.
- The random-init control instantiates the exact same GFM architecture independently for each paired seed and never loads pretrained weights.

## Holdout firewall

- Only `train.h5` metadata, `seismic_patch[0:3]` and `seismic_patch[8]` were read. PORO targets and baseline predictions came from hash-verified development OOF archives.
- `test.h5`, frozen holdout paths, historical test metrics and holdout labels were neither opened nor probed.
