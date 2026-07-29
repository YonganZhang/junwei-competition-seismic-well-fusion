# P10 reconstruction model-results audit

- code_commit: `5685933c382b698c4cc4003780d66277f24f21a3`
- workbook: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/track_model_metrics.xlsx`
- figure: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/before_after_primary_metric.png`
- row_count: `64`
- sheet_name: `模型指标`
- evidence_only_boundary: `True`

## Conclusion

The OpenMind/ResEnc-L lane has a real foundation gain but remains end-to-end non_beneficial.
Pretraining reduces the same-architecture random-init RMSE from 1.052412481992174 to
0.5415301607840952, but the same strict-development sample universe still sits far above the
PyKrige reference at 0.02120691345759842. That means the foundation effect is real, but it is
not enough for promotion against the strong baseline.

No verified bug in the archived evidence justified a code fix that would close the gap.

## Foundation gain vs end-to-end outcome

- random-init macro RMSE: `1.05241248199`
- pretrained macro RMSE: `0.541530160784`
- foundation gain vs random-init: `0.510882321208`
- foundation gain vs random-init (%): `48.5439245495`
- PyKrige reference macro RMSE: `0.0212069134576`
- end-to-end delta vs PyKrige: `-0.520323247326`
- strict ridge_linear RMSE: `0.0320466299129`
- conditional ridge_linear RMSE: `0.0215892758155`
- Stage-4 strict known-holdout RMSE: `0.0356249253592`
- Stage-4 conditional known-holdout RMSE: `0.0210128882524`

## Audited implementation checkpoints

| check | evidence / value | status |
| --- | --- | --- |
| 3D patch shape and axis order | `PATCH_SHAPE = (9, 20, 18)` in `build_dataset.py`; tiled as `k,j,i` over grid `[63, 100, 108]` | passed |
| coordinate / scale | `coordinate_bounds` = x `[432851.25, 437175.25]`, y `[6477478.5, 6480275.0]`, depth `[2800.718505859375, 3543.77587890625]`; `mapping` = `nearest active Eclipse cell in anisotropic (50m,50m,2m) coordinates` | passed |
| observation mask conditioning | `n_observation_rows=91`, `n_unique_cells=91`, `n_wells_with_constraints=1`; strict supplies 90 constraints, conditional supplies 91 | passed |
| train/eval mask mutual exclusion | strict: `n_direct_well_cells_excluded_from_metrics=0`; conditional: `n_direct_well_cells_excluded_from_metrics=90`; both have `test_patch_blocks_disjoint_from_train_and_validation=True` | passed |
| normalization / inverse transform | `results_strict.json` and `results_conditional.json` report `framework=ml_framework.preprocess` and `all_roundtrip_checks_passed=True`; `build_summary.preprocessing.coordinate_roundtrip` shows zero max abs error for x/y/depth | passed |
| adapter / decoder output | `p9_openmind_effect.py` and summary constrain `trainable_scope='attribute_projection_and_decoder'`; output metric is scalar RMSE on same sample universe | partially checked; internal tensor width not exposed in archived summaries |
| fold / sample universe | OpenMind summary: `folds=[0, 1, 2, 3, 4]`; `same_validation_sample_universe_as_strong_baseline=True`; strict/conditional results preserve separate train/test patch blocks | passed |
| metric direction | `rmse/mae` lower-is-better, `r2/pearson_r` higher-is-better, encoded in workbook rows | passed |

## Evidence and scope audit

- `build_summary.json` says `grid_shape_kji=[63, 100, 108]` and `n_active_cells=183545`.
- `results_strict.json` strict train/test split: train i-blocks `[4, 5]`, test i-blocks `[0, 1, 2]`.
- `results_conditional.json` conditional train/test split: train i-blocks `[0, 1, 2, 3]`, test i-blocks `[4, 5]`.
- `results_strict.json` leakage check: `test_patch_blocks_disjoint_from_train_and_validation=True`.
- `results_conditional.json` leakage check: `direct_well_observation_cells_excluded_from_test_metrics=True`.
- `p9_openmind_effect.py` is a strict-development-only comparison; its summary records `frozen_test_accessed=False` and `guard_accessed=False`.
- `p5_stage4_confirmation/summary.json` records `prior_test_consumed=True` and `fresh_blind=False`.

## Root cause assessment

The evidence points to model-inductive-bias / capacity mismatch, not a broken pipeline. The OpenMind lane
is helped by pretraining, but it still loses decisively to the PyKrige reference on the same split universe.
The available archives are enough to label the lane non_beneficial without inventing an unsupported bug fix.

## Evidence-only boundary

- Existing legal-dev evidence was sufficient: `build_summary.json`, `results_strict.json`, `results_conditional.json`, `p9_openmind_effect/summary.json`, `p5_stage4_confirmation/summary.json`, `model_inspection.json`.
- No frozen holdout or tuning run was used for this audit bundle.
- The report intentionally does not claim a newly improved production model.

## Files written

- `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/track_model_metrics.xlsx`
- `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/figures_manifest.csv`
- `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/tables_manifest.csv`
- `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/audit_report.md`
- `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/before_after_primary_metric.png`

## Residual risk

The current report is evidence-only. It does not re-run the model or change split/protocol choices,
so it cannot prove a new production model. It only documents that the better-performing baseline is
still the PyKrige reference.
