# P18 CIG-Bench Property feasibility

Verdict: **BLOCKED_DATA_OR_API**

## API / package inspection
- cig_bench version: `0.2.0`
- modelscope version: `1.39.0`
- torch version: `2.13.0`
- PropertyPredictor signature: `(restore_path: Optional[str] = None, device: str = 'cuda', use_autocast: bool = True, use_tanh: bool = True, model_cls=None, model_id: Optional[str] = None, file_path: Optional[str] = None, cache_dir: Optional[str] = None, revision: Optional[str] = None)`
- property registry entry: `('douyimin/CIG-Bench', 'CIG-Bench-Property.pth')`

## Development inputs actually inspected
- train.h5: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/track-property/_data/processed/reservoir/train.h5`
  - sha256: `b1962a89b049dd2c23ff2fbf857b5daf69de8b40c2f1f5166205f9bc3df70ab2`
  - sample_count: `1135`
  - sample_key: `sample_0000000`
  - seismic_patch shape: `(3, 3, 9)`
  - well_log_seq shape: `(9, 8)`
  - label shape: `(3,)`
  - meta target_names: `['PHIF', 'log1p(KLOGH)', 'SW']`
- guard.npz: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/track-property/_pipelines/02_task_datasets/reservoir/_outputs/guard.npz`
  - sha256: `67b3866d1975f0ddb32b7016d3e7ba6fa595a5cbbd03cb6b0231a74851ca77fe`
  - sample_count: `81`
  - seismic_patch shape: `(3, 3, 9)`
  - well_log_seq shape: `(9, 8)`
  - label shape: `(3,)`

## Baseline reference already present in repo
- model: `tiny_mlp`
- framework: `NumPy small one-hidden-layer MLP via shared ml_framework.train_loop`
- split families: `{'15/9-F-11': 'train', '15/9-F-1': 'train', '15/9-19': 'train', '15/9-F-12': 'guard', '15/9-F-15': 'test'}`
- PHIF RMSE/MAE/R2/Pearson: `{'RMSE': 0.020900944868432036, 'MAE': 0.015648417414317217, 'R2': 0.9029771182086087, 'R2_reason': None, 'Pearson': 0.9590147489015269, 'Pearson_reason': None}`
- log1p(KLOGH) RMSE/MAE/R2/Pearson: `{'RMSE': 1.128879382159456, 'MAE': 0.8932959697611352, 'R2': 0.7762172881275564, 'R2_reason': None, 'Pearson': 0.9330526367782634, 'Pearson_reason': None}`
- SW RMSE/MAE/R2/Pearson: `{'RMSE': 0.23263274302747222, 'MAE': 0.185149644883243, 'R2': 0.1922326878837205, 'R2_reason': None, 'Pearson': 0.718124802322107, 'Pearson_reason': None}`

## Smoke / blocker probe
- smoke status: `BLOCKED_DATA_OR_API`
- smoke reason: `property weight download / predictor init failed`
- smoke error: `HTTPError("[E3020] [404] 获取模型文件失败，文件内容为空 (request_id=62ebe4da-3bbd-449a-8cbd-9508ea24fa54) | code=10990101007\n  Request: GET https://modelscope.cn/api/v1/models/douyimin/CIG-Bench/repo?Revision=master&FilePath=CIG-Bench-Property.pth\n  Response: {'Code': 10990101007, 'Message': '获取模型文件失败，文件内容为空', 'RequestId': '62ebe4da-3bbd-449a-8cbd-9508ea24fa54', 'Success': False}")`

## Blocker reasons
- API probe returned BLOCKED_DATA_OR_API: property weight download / predictor init failed
- ModelScope default checkpoint download for douyimin/CIG-Bench / CIG-Bench-Property.pth failed with a reproducible HTTP 404.
- The reservoir development tensors are sample-level seismic patches plus well_log_seq feature sequences; they do not provide a legal sparse target-property volume matching CIG-Bench PropertyPredictor's seismic+ sparse property contract without target leakage.

## Commands run
- `python3 -m pip install cig_bench`
- `python3 - <<'PY' ... PropertyPredictor(device='cpu', use_autocast=False) ... PY`

## Notes
- No frozen holdout / test.h5 was opened.
- No candidate metrics were fabricated.
- Because the API/weight contract is blocked, there is no honest dev-only baseline comparison to report.
