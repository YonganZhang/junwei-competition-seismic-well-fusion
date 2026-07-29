# P10 reconstruction model-results audit

- code_commit: `e4fd5d8a6371c2b0db6ba2258a41349ec6cfb4f7`
- workbook: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/track_model_metrics.xlsx`
- figure: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p10-results-reconstruction/_pipelines/02_task_datasets/reconstruction/_outputs/p10_model_results/before_after_primary_metric.png`
- row_count: `64`
- sheet_name: `model metrics`

## Conclusion

The OpenMind/ResEnc-L lane is non_beneficial on the same strict development split: pretraining
reduces the random-init error, but the pretrained model still stays far above the PyKrige
reference. No verified bug in normalization, observation masking, or adapter wiring was found,
so there is no justified code fix to promote.

## Key before/after facts

- random-init macro RMSE: 1.05241248199
- pretrained macro RMSE: 0.541530160784
- PyKrige reference macro RMSE: 0.0212069134576
- strict ridge_linear RMSE: 0.0320466299129
- conditional ridge_linear RMSE: 0.0215892758155
- Stage-4 strict known-holdout RMSE: 0.0356249253592
- Stage-4 conditional known-holdout RMSE: 0.0210128882524

## Root cause assessment

The evidence points to model-inductive-bias / capacity mismatch, not a broken pipeline:

1. The strict split, sample universe, and observation-mask handling are already fenced in the archived summaries.
2. The pretrained OpenMind adapter only trains the attribute projection and decoder.
3. The same-architecture random-init run is much worse than the pretrained run, proving pretraining helps,
   but the PyKrige reference remains dramatically stronger on the same split universe.
4. Conditional reconstruction is explicitly not strict holdout, so it is not promoted as blind generalization.

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
