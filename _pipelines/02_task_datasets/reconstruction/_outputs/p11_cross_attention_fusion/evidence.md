# P11 cross-attention fusion — development-only evidence

## Result

- PyKrige baseline pooled development RMSE: `0.028449728170`.
- Cross-attention consensus-gated mean-seed pooled RMSE: `0.028449728170`.
- The less conservative adaptive gate RMSE was `0.028881520098`; it is retained as a diagnostic, not selected as the safe route.
- Absolute RMSE change (candidate - baseline): `+0.000000000000`; relative change (positive means improvement): `+0.000000%`.
- No positive development RMSE change was observed; the four-of-four inner-fold consensus guard rejected every correction and preserved the baseline exactly.
- 大模型贡献占比待下一轮消融确认。This run only evaluates the pretrained encoder mode; it does not attribute the overall change to OpenMind.

## Five genuinely independent spatial units

The five locked outer folds are the inferential units. The three seeds are paired optimization pseudo-repeats inside each fold, not 15 independent samples. These are five genuinely independent spatial units.

| fold | baseline RMSE | cross-attention RMSE | delta | outcome |
|---:|---:|---:|---:|:---|
| 0 | 0.027239559 | 0.027239559 | +0.000000000 | tie |
| 1 | 0.028627607 | 0.028627607 | +0.000000000 | tie |
| 2 | 0.018619291 | 0.018619291 | +0.000000000 | tie |
| 3 | 0.028605050 | 0.028605050 | +0.000000000 | tie |
| 4 | 0.036338338 | 0.036338338 | +0.000000000 | tie |

Independent-fold outcomes: **0 win / 0 loss / 5 tie**.

## Block-bootstrap uncertainty

Whole spatial folds were resampled; voxels and seed rows were never sampled as independent observations.
- Mean-seed pooled RMSE delta point estimate: `-0.000000000000`.
- 95% block-bootstrap interval: `[-0.000000000000, +0.000000000000]`.

## Method and firewall

- Each normalized seismic channel was encoded separately. The structured query cross-attends to three channel tokens.
- Standardization and PCA were fitted inside each training split only. The attention residual and bounded benefit gate were both cross-fitted by the original spatial folds. The safe route additionally requires all four inner folds to win before any correction can leave gate=0.
- Encoder weight loading is parameterized as `pretrained` or `random_init`; this run used only `pretrained`. The next matched random-initialization ablation can reuse this harness.
- Only `train.h5` metadata, `seismic_patch[0:3]`, and `seismic_patch[8]` were available to OpenMind extraction. Labels came from hash-verified development OOF archives.
- `test.h5`, frozen holdout paths, historical test metrics, and holdout labels were not read or probed.

## Interpretation boundary

- This is a development diagnostic, not promotion evidence: the same-architecture random-init contribution control was intentionally deferred by the current task.
- Any overall gain can reflect the full fusion harness, regularization, PCA, or gating. It must not be described as an OpenMind-specific gain.
