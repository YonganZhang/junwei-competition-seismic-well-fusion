# P18 CIG-Bench RGT external-drift kriging

## Result

- Committed PyKrige development OOF RMSE: `0.028449728170`.
- CIG-Bench RGT KED development OOF RMSE: `0.028632416369`.
- KED minus PyKrige RMSE: `+0.000182688199` (relative `+0.642144%`).
- Five genuinely independent spatial-fold outcomes: `2 win / 3 loss / 0 tie`; there are no random-seed pseudo-repeats.
- Whole-spatial-fold bootstrap 95% CI for KED minus PyKrige RMSE: `[-0.000351312161, +0.000765203628]`.
- Decision: `NO_ROBUST_DEVELOPMENT_GAIN`; the candidate remains disabled and the frozen holdout remains sealed.

## RGT construction and geological rationale

- The 20 non-overlapping development patches are reassembled into the real contiguous `63 x 100 x 72` K/J/I volume. Only `seismic_patch[0]` is sent to RGT-Est because it is seismic amplitude; channels 1/2 are already engineered local-RMS and vertical-gradient attributes, not independent seismic acquisitions.
- RGT is aligned as `(T,H,W)=(K,J,I)`: K is the vertical/depositional sampling axis and J/I are the two lateral trace axes. The returned RGT is resized back to the exact development grid before KJI lookup.
- Upstream recommends `400 x 512 x 512`; this run used `[128, 256, 256]`. Canonical-shape status is `False`.
- The persisted canonical feasibility probe status is `CUDA_OUT_OF_MEMORY`. A fallback shape is allowed only after the same weight and real development volume produce a CUDA out-of-memory result; it is fixed from resource limits, never chosen by target RMSE.
- UniversalKriging3D receives one specified RGT drift. Coordinates, 512 labels/fold, linear variogram, `nlags=4`, and 2,048 validation rows/fold are unchanged from the OrdinaryKriging3D control.
- The drift is fixed before reading any fold target and is never fitted to porosity. It expresses a mature KED prior: cells at similar relative geological time may share a trend even when Euclidean depth differs.

## Control and provenance checks

- Drift-disabled OK3D recomputation max absolute difference from the committed OOF archive: `7.450e-09`.
- Installed `cig-bench==0.2.0`; wheel metadata and official upstream declare MIT. Weight SHA-256: `e328f5534bc90e53d4dbd8e5eeb75e43a03c072db17a2a60f3347b4f6ef8b3ec`.
- The installed wheel metadata homepage is a placeholder; provenance is therefore locked to the official `douyimin/CIG-bench` URL, imported source-file hashes, ModelScope model ID, and downloaded weight hash.

## Fold breakdown

| fold | PyKrige RMSE | RGT-KED RMSE | delta | outcome |
|---:|---:|---:|---:|:---|
| 0 | 0.027239559 | 0.026775971 | -0.000463588 | win |
| 1 | 0.028627607 | 0.028726939 | +0.000099332 | loss |
| 2 | 0.018619291 | 0.018090217 | -0.000529074 | win |
| 3 | 0.028605050 | 0.028669189 | +0.000064139 | loss |
| 4 | 0.036338338 | 0.037519439 | +0.001181101 | loss |

## Scope and limitations

- These are development-only results from five spatial units. The 20,000 bootstrap draws resample folds, not voxels, and do not create 20,000 independent observations.
- RGT quality has no ground-truth horizon labels in this dataset; the volume is a pretrained target-free structural prior, not a validated geological-age label.
- PropertyPredictor was not run: its documented interface is conditional on a sparse property/well volume, so treating it as a standalone zero-shot porosity predictor would not be a fair or documented comparison. KED was the preregistered priority.
- No claim is made that any metric change is caused by pretrained knowledge without a same-architecture random-init RGT ablation.
- Only `train.h5`, audited P5 development caches, and development OOF artifacts were opened. No frozen evaluation surface was read.
