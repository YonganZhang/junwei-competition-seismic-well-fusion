# P10 lithofacies model-results audit

## Conclusion

Status: non_beneficial. The cached MOMENT-1-base foundation path remained below the fixed-nine XGBoost LOGO4 baseline on the development pairs.
Baseline fixed-schema macro-F1 mean: 0.194938.
MOMENT pretrained fixed-schema macro-F1 mean: 0.046308.
MOMENT random-init fixed-schema macro-F1 mean: 0.041112.
Primary-metric delta (pretrained - baseline): -0.148630.

## Before/after by fold and seed

| fold | seed | baseline | pretrained | random-init | delta(pretrained-baseline) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1867973658 | 0.213580 | 0.055556 | 0.048652 | -0.158025 |
| 1 | 1867973658 | 0.183572 | 0.033689 | 0.049832 | -0.149883 |
| 2 | 1867973658 | 0.181276 | 0.030234 | 0.053227 | -0.151041 |
| 3 | 1867973658 | 0.201322 | 0.000000 | 0.018713 | -0.201322 |
| 0 | 2137841944 | 0.213580 | 0.055556 | 0.048652 | -0.158025 |
| 1 | 2137841944 | 0.183572 | 0.063371 | 0.064163 | -0.120202 |
| 2 | 2137841944 | 0.181276 | 0.030234 | 0.030234 | -0.151041 |
| 3 | 2137841944 | 0.201322 | 0.043614 | 0.018713 | -0.157709 |
| 0 | 3902865753 | 0.213580 | 0.087314 | 0.048652 | -0.126267 |
| 1 | 3902865753 | 0.183572 | 0.078250 | 0.064163 | -0.105322 |
| 2 | 3902865753 | 0.181276 | 0.097029 | 0.005128 | -0.084247 |
| 3 | 3902865753 | 0.201322 | 0.043210 | 0.043210 | -0.158113 |

## Root cause and fix status

- No reproducible integration defect was found in the development-only path.
- The frozen fixed-nine / LOGO4 / train-fold-only preprocessing contract held.
- MOMENT pretrained improved over random initialization but did not beat the strong XGBoost baseline.
- Therefore the honest outcome is non_beneficial, not a repaired win.

## Evidence and hashes

- development batch: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-stage3-lithofacies/_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3/runtime/development_logo4.npz`
- development batch sha256: `b6817ae218dee12bc72a1551e76423a61eb4ff11faf70dcb0444b01ba422f51c`
- MOMENT snapshot: `/mnt/data/yongan-admin-2/.cache/huggingface/hub/models--AutonLab--MOMENT-1-base/snapshots/5e44b0ea26376a176360f87831124e018f876d96`
- MOMENT snapshot sha256: `1a436826ffe618273ec62b9656dc4cab8edc470364f104e90542a4ebc14fb825`
- split hash: `a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555`
- code commit: `a58433d0645899093316a0b4c4a087367160eb8d`

## Traceable contract evidence

- Depth-window / stride / direction: `p9_moment_effect._inputs()` keeps the fixed 33-position LOGO4 window and reshapes the 26 well-log + 9 seismic channels to `[B,35,33]`; the audit uses the cached development batch from `_outputs/p5_stage3/runtime/development_logo4.npz`.
- Padding / mask: `p4_contract.apply_fold_preprocessor()` preserves the 26 physical channels, appends the 13-channel missing mask, and records `fit_scope = fold_train_mother_families_only`.
- Fold-train-only normalization: `p4_contract.fit_fold_preprocessor()` derives `log_stats`, `seismic_stats`, and `class_weights` only from the fold-train mother families; validation uses the immutable train statistics only.
- MOMENT embedding / input channels: `_models/lithofacies/moment_depth.py` requires `n_channels=35`, interpolates the 33-position input to 512 internally, and uses the cached `momentfm.MOMENTPipeline`.
- Frozen / PEFT / head / output classes: the MOMENT audit uses `freeze_encoder=True`, `freeze_embedder=True`, `freeze_head=False`; `build_model(..., num_class=9)` is a fixed-nine classifier head.
- Fixed-nine label mapping: `p4_contract.CLASS_NAMES` and `classification_metrics_from_logits()` operate on the frozen GM09 nine-class schema; `fixed_schema_macro_f1` is the primary metric and `supported_class_macro_f1` remains diagnostic only.
- Class imbalance: `fit_fold_preprocessor()` computes fold-train-only `class_weights` and the runner passes them into `torch.nn.functional.cross_entropy`; the stage3 baseline uses locked `sqrt_inverse_frequency_weighted_*` contracts.
- LOGO fold / sample universe: `build_lithofacies_split_manifest()` freezes the four development mother families plus the F-5 test family; the development batch reports `split_hash = a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555` and `frozen_test_accessed = False`.
- Seed / metric direction: the stage3 development audit uses seeds `1867973658`, `2137841944`, `3902865753`; the score direction is maximize for `fixed_schema_macro_f1` and minimize for calibration, NLL, and Brier only.

## Residual risk

- The audit is limited to the cached MOMENT-1-base snapshot and the fixed development contract.
- No frozen-test / known-holdout evidence was consumed.
- MOMENT pretrained showed a small foundation gain over random initialization (`0.046308` vs `0.041112`) but remained far below XGBoost (`0.194938`), so the end-to-end conclusion stays `non_beneficial`.
- No HPO or split changes were performed.
