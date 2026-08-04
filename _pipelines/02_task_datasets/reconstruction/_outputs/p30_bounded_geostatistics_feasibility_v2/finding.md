# P30 bounded reconstruction geostatistics finding

## Finding

P21 remains the reconstruction default. The matched-budget physical
anisotropic ordinary-kriging and seismic regression-kriging candidates
both fail promotion. Classical well-log co-kriging is blocked by the
absence of an independently aligned well-log secondary variable in the
P21 folds, not by a missing numerical solver.

Old P29 outputs are retained only as historical policy evidence. They
must not be used for a new promotion because the old replay interface
misread scalar ensemble weights and silently zero-filled query-side
seismic/GFM covariates. The repaired code is locked by an A0-to-P21
identity check and query-side fail-closed tests.

P30 is a sparse Eclipse-grid proxy under fixed five-fold, 512-label
training and 2,048-row validation budgets per fold. It is not a real
well-log-driven kriging or co-kriging experiment. The covariance-form
variance now uses `C(0) - w^T c - mu`; this sign correction changes no
weights, mean predictions, RMSE values, or promotion decision.

## Matched five-fold result

| fold | P21 RMSE | anisotropic OK RMSE | regression-kriging RMSE |
|---:|---:|---:|---:|
| 0 | 0.026804535898 | 0.027374617416 | 0.027102193025 |
| 1 | 0.028579678784 | 0.033271859969 | 0.032631974742 |
| 2 | 0.016645911572 | 0.023631858050 | 0.021637016601 |
| 3 | 0.027712695506 | 0.031618691657 | 0.030936404681 |
| 4 | 0.035575505088 | 0.035466659615 | 0.036105582030 |

Pooled P21 RMSE is `0.027734374378`;
anisotropic ordinary kriging is `0.030569516403`
(1 wins / 4 losses), and
regression kriging is `0.030093884156`
(0 wins / 5 losses).

## Direction-cone audit

The requested direction cosine was 0.8. Only fold 0 depth required
relaxation to 0.6, for both the target and regression-residual
variograms; both fits are flagged `relaxed_low_resolution`. Every
other axis/fold fit used 0.8. No threshold used validation targets.

## Residual risks and next input

The current depth is TVD-like Eclipse cell-centre depth, not MD or
seismic TWT. The existing weak well tie mixes MD with TVD-like depth,
and only one intersecting LFP well supplies sparse constraints. Future
well-log plus seismic foundation fusion therefore requires the exact
machine contract in `fusion_io_contract.json`: aligned KJI and metre
coordinates, an identified MD-to-TVDSS/TWT transform, declared curve
units and quality masks, support weights, explicit missing-modality
masks, outer-fold roles, source/weight hashes, and PORO mean/variance
plus modality-ablation outputs.

## Reproduction

```bash
PYTHONPYCACHEPREFIX=_pipelines/02_task_datasets/reconstruction/_tmp/p30_pycache /usr/bin/python3 -m py_compile _pipelines/02_task_datasets/reconstruction/p29_agent_action_effect_repair.py _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py _pipelines/02_task_datasets/reconstruction/_tests/test_p29_agent_action_effect_repair.py _pipelines/02_task_datasets/reconstruction/_tests/test_p30_bounded_geostatistics_feasibility.py
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p29_agent_action_effect_repair.py' -v
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -m unittest discover -s _pipelines/02_task_datasets/reconstruction/_tests -p 'test_p30_bounded_geostatistics_feasibility.py' -v
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py --data-dir .claude/worktrees/track-reconstruction/_data/processed/reconstruction --stage3-root .claude/worktrees/p5-stage3-reconstruction/_tmp/p5_stage3_reconstruction --feature-cache .claude/worktrees/p10-results-reconstruction/_tmp/p17_foundation_geostatistics/gfm_point_features.npz --build-summary .claude/worktrees/track-reconstruction/_pipelines/02_task_datasets/reconstruction/build_summary.json --p29-summary _pipelines/02_task_datasets/reconstruction/_outputs/p29_agent_action_effect_repair_v2/summary.json --output-dir _pipelines/02_task_datasets/reconstruction/_outputs/p30_bounded_geostatistics_feasibility_v2
PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 _pipelines/02_task_datasets/reconstruction/p30_bounded_geostatistics_feasibility.py --verify-only
git diff --check
```
