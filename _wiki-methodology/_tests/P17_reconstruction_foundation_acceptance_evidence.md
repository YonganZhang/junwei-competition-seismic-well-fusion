# P17 三维重建基础模型验收证据

日期：2026-07-31
工作树：`p10-results-reconstruction`
根种子：`2693`

## 验收结论

P17 真实使用冻结 ThinkOnward geophysical foundation model 的地震表征，
并将其用于非平稳邻域度量，而非直接孔隙度回归。候选在锁定的 5 折
开发 OOF 上出现正向信号，通过代码合同、哈希、行对齐、数据防火墙和独立复算。
因整折 bootstrap 95% 区间跨 0，验收状态为 `DEVELOPMENT_SIGNAL`，
`default_enabled=false`。

## 定量结果

| 项目 | PyKrige | P17 | 变化 |
|---|---:|---:|---:|
| OOF RMSE | 0.028449728170 | 0.028319907650 | -0.000129820520 (-0.4563%) |
| OOF MAE | 0.021413486381 | 0.021200329887 | -0.000213156494 |
| OOF bias | -0.001857953993 | -0.000870598245 | 绝对偏差下降 |

- 选中候选：`gfm_metric_f0.05_s0.10_k128_blend_0.75`；
- 独立空间折：3 胜 / 2 负 / 0 平；
- 整折 bootstrap：20,000 次，P(P17 更优)=`0.7668`；
- RMSE 差值 95% CI：`[-0.000589605334, +0.000128806422]`。

## 数据与泄漏门

- 每折仅 512 条 `point_train` 标签，每折 2,048 条验证行；
- 标准化和 PCA 只在当前外层训练点拟合；
- GFM 仅读 `train.h5` 的地震属性、坐标和 active mask，不读目标数据集；
- CLI 没有 `--test` 或 `--holdout` 参数；`test.h5` 与冻结 holdout 未打开；
- 早期使用其他 OOF 标签扩大至 8,192 训练行的试验已判定为预算不等价，
  不进入任何结果表。

## Command Gate

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python3 \
  -m unittest -v \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p14_geophysical_fm.py \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p15_gfm_finetune.py \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p16_gfm_denoise.py \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p17_foundation_geostatistics.py
```

结果：`39 tests`, `OK`。P17 单独为 `8 tests`, `OK`。

## Live / User Journey

本交付是科学计算 CLI，没有 UI 或交互式用户旅程。实时路径验收由真实
GFM 缓存加载、5 折计算、产物落盘和随后的 `verify` 命令代替；不使用截图作为通过证据。

## Trace / SSDO Audit

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python3 \
  _pipelines/02_task_datasets/reconstruction/p17_foundation_geostatistics.py verify
```

独立复算状态：`PASSED`；10,240 行、5 个折的 RMSE、样本数、预测归档 SHA-256
均与 `summary.json` 匹配。
结构化追踪位于 `summary.json` 的 `sample_audit`、`foundation_feature_audit`、
`fit_audits`、`holdout_firewall` 和 `runtime`；独立防伪完成证据位于 `verification.json`。
其中任一样本预算、行数、哈希、分折指标或 holdout 状态不一致都会失败关闭。

## 结论边界

- 本阶段按用户要求不做消融；不归因具体改善来源；
- 不声称预训练 GFM 已统计显著超过 PyKrige；
- 不在本阶段开启默认路线，不消费冻结 holdout。

## 真源

- `code:reconstruction_foundation_geostatistics`
- `test:P17_reconstruction_foundation_acceptance_evidence`
- `finding:P17_reconstruction_foundation_role_mismatch`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p17_foundation_geostatistics/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p17_foundation_geostatistics/verification.json`
