---
phase_id: P18
status: superseded
severity: major
owner_col: COL2
source: runtime
created_at: 2026-07-31
closed_at: 2026-07-31
closure_evidence: _wiki-methodology/_tests/P19_reconstruction_training_diagnostics_acceptance_evidence.md
superseded_by: P19_reconstruction_meta_purge_and_gradient_diagnosis
---

# 赛道⑥通过各向异性与嵌套选型获得稳健开发改善

> P19 发现其余折训练标签与当前验证折存在坐标重叠，已用元选择去重协议完整
> 复算并取代本 finding；P18 的方向性结论保留，原协议表述不再作为当前真源。

## Local Case

P18 在不增加每折 512 条训练标签、不打开冻结 holdout 的条件下，将 PyKrige
RMSE 从 `0.028449728170` 降至 `0.027752680679`，相对改善 `2.4501%`，五个
空间折全部改善，完整空间折 bootstrap 95% 区间完全低于零。

## Class Pattern

小样本三维地学任务中，大模型表征必须服从地质方向性与严格的模型选择协议。
先把垂向/横向相关尺度写入距离，再让冻结表征提供非平稳修正，通常比直接微调
高自由度解码器更稳健；同一 OOF 上选优仍会制造虚假改善，必须嵌套选型。

## Evidence

- `_pipelines/02_task_datasets/reconstruction/_outputs/p18_anisotropic_foundation_geostatistics/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p18_anisotropic_foundation_geostatistics/verification.json`
- `_wiki-methodology/_tests/P18_reconstruction_anisotropic_acceptance_evidence.md`

## Impact

赛道⑥已从不稳定的 `DEVELOPMENT_SIGNAL` 推进为
`ROBUST_DEVELOPMENT_SIGNAL`。本轮没有消融，因此只对完整方案负责，不把提升
因果归于预训练权重；完整空间折只有 5 个，且部分超参数位于网格边缘，因此
不作大样本显著性或最终泛化声明。冻结 holdout 仍封存，默认仍关闭。

## Prevention Rule (candidate)

超过一个候选的开发 OOF 搜索必须嵌套选择；三维插值在加入复杂特征前必须显式
检查垂向/横向各向异性，不能只用池化最优值掩盖空间折退化。

## Links

- task_plan: ../_task_plan.md
- method: ../../_wiki/_methods/explorations/003-p18-reconstruction-anisotropic-foundation-geostatistics.md
