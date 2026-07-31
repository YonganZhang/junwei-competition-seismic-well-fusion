# P17 foundation-informed nonstationary geostatistics

## Result

- PyKrige pooled development OOF RMSE: `0.028449728170`.
- Selected pretrained-GFM kernel RMSE: `0.028319907650`.
- Selected candidate: `gfm_metric_f0.05_s0.10_k128_blend_0.75`.
- Delta (candidate - PyKrige): `-0.000129820520`.
- Independent spatial-fold outcomes: 3 wins / 2 losses / 0 ties.
- Whole-fold bootstrap delta CI95: `[-0.000589605334, +0.000128806422]`.
- Decision: `DEVELOPMENT_SIGNAL`; default remains disabled.

## Method

The genuine frozen ThinkOnward GFM encodes input seismic traces. Within every outer spatial fold, its representations are scaled and reduced by PCA using only the 512 legal training points.  Physical coordinates, local seismic attributes and the reduced GFM coordinates jointly define a nonstationary neighbourhood metric.  Inverse-distance kernel interpolation then estimates porosity and is conservatively blended with the unchanged PyKrige prediction.

## Boundary

- Exactly 512 training labels and 2,048 validation rows are used per fold.
- No frozen test or `test.h5` path exists in the CLI.
- Encoder inputs are target-free seismic and coordinates only.
- This phase does not run a matched random-init/no-foundation ablation; causal attribution to pretraining is therefore not claimed.
- Any positive result is development evidence, not a final test claim.
