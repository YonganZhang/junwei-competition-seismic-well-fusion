# P18 赛道⑥各向异性基础模型地统计验收

> **已由 P19 取代。** P19 发现其余折训练子集与当前验证折存在 24--58 个
> 标签坐标重叠，原“仅排除当前折排名指标”的嵌套协议仍可能间接使用当前折
> 标签。完成元选择训练坐标去重后的当前真源为
> `P19_reconstruction_training_diagnostics_acceptance_evidence.md`；修正 RMSE
> `0.027751397628`，5/5 折改善，未推翻方向性结论。

日期：2026-07-31

## 验收对象

P18 修正 P17 的同 OOF 选优偏差，并在严格保持每折 512 条训练标签、2,048 条
验证样本和冻结 holdout 防火墙的前提下，引入垂向各向异性与嵌套 LOFO top-3
选型。

## 科学门

- 候选空间在最终报告前固定为 `5×3×3×3×3×3=1,215`；
- 每个被报告折的候选排名指标仅使用其余四折；P19 后续证明这里还缺少对其余
  折训练子集的当前验证坐标去重，因此本条不足以构成完整标签防火墙；
- GFM 标准化/PCA 只在当前外层折 512 个训练点拟合；
- CLI 没有 `--test` 或 `--holdout` 参数；
- 不做消融、不声称预训练因果贡献、候选默认关闭。

## 结果

| 路线 | RMSE | 相对 PyKrige | 空间折结果 |
|---|---:|---:|---:|
| PyKrige | 0.028449728170 | — | — |
| P17 原同 OOF 选优 | 0.028319907650 | -0.4563% | 3/5 胜；已取代 |
| P17 候选族嵌套 top-1 | 0.028599146195 | +0.5252% | 同口径修正 |
| P17 候选族嵌套 top-3 | 0.028534404074 | +0.2976% | 评估修正 |
| P18 各向异性嵌套 top-3 | 0.027752680679 | -2.4501% | 5/5 胜 |

P18 完整空间折 bootstrap RMSE 差值 95% 区间为
`[-0.001140994782, -0.000353924655]`，20,000 次抽样中候选更优概率为 `1.0`。
由于只有 5 个粗粒度空间单元，该区间主要概括已观察到的逐折方向，不作大样本
显著性解释。部分入选超参数位于网格边缘，已登记为预注册外部验证事项。

## Command Gate

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python3 \
  -m unittest -v \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p17_foundation_geostatistics.py \
  _pipelines/02_task_datasets/reconstruction/_tests/test_p18_anisotropic_foundation_geostatistics.py
```

结果：`16 tests`, `OK`。

## Live / User Journey

本交付是科学计算 CLI，没有网页或交互式用户旅程。真实路径已从锁定 Stage-3
开发归档读取 5 个空间折，复用哈希锁定的 GFM 特征，完成 1,215 候选嵌套
选型并生成 `summary.json`、`prediction_errors.npz` 和 `evidence.md`；冻结
holdout 未进入命令参数或文件访问记录。

## Trace / SSDO Audit

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python3 \
  _pipelines/02_task_datasets/reconstruction/p18_anisotropic_foundation_geostatistics.py verify
```

独立复算状态：`PASSED`。10,240 行、5 个逐折指标、P17 嵌套修正值、预测归档
SHA-256、summary SHA-256 和选型防火墙全部匹配。

## 证据边界

P18 是 `ROBUST_DEVELOPMENT_SIGNAL`，不是冻结测试结论，也不是基础模型消融
结论。真实预训练 GFM 确实参与候选方案，但具体提升来源留待后续消融。

## 真源

- `_pipelines/02_task_datasets/reconstruction/_outputs/p18_anisotropic_foundation_geostatistics/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p18_anisotropic_foundation_geostatistics/verification.json`
- `finding:P18_reconstruction_anisotropy_recovers_robust_signal`
