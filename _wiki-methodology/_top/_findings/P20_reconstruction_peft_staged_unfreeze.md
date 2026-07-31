---
phase_id: P20
status: accepted
severity: major
owner_col: COL2
source: runtime
created_at: 2026-07-31
---

# 赛道⑥ LoRA 可修复梯度失配，但当前监督目标与 P19 表征高度冗余

## Local Case

P20 在不改变五折、每折 512 个合法训练标签和 2,048 个验证点的前提下，使用
432 个无缩放、无填充、无插值的 `400×160` 连续 SEG-Y 原生窗口，依次比较非零
小初始化头部、rank-4 LoRA、瓶颈 Adapter 和分阶段 LoRA。分阶段路线按头部热身、
LoRA、末端 LayerNorm、最后一 Transformer 块低学习率解冻推进。

四条路线对 PyKrige 均为 5/5 折改善。32 步时 RMSE 分别为
`0.027814382567`、`0.027789635296`、`0.027814441628` 和
`0.027789615700`。最佳分阶段 LoRA 相对 PyKrige 改善 `2.3203%`，但仍比已接受
P19 的 `0.027751397628` 高 `0.000038218072`，不能晋级。延长到 80 步后 inner
calibration 继续下降，外折 RMSE 却变为 `0.027791517166`，排除了“只是训练步数
不够”的解释。

## Class Pattern

LoRA/Adapter 的梯度非零和参数实际移动，只证明优化路径打通，不等于新增可泛化
信息。P20 与 P19 的 OOF 误差相关系数为 `0.9992037`；固定融合网格的最优权重是
P20 权重 `0.0`。这说明当前有监督 PEFT 主要复现了 P19 已利用的空间—地震邻域
结构，而没有获得独立于 P19 的语义或地质约束。

## Evidence

- `_pipelines/02_task_datasets/reconstruction/_outputs/p20_peft_staged_unfreeze/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p20_peft_staged_unfreeze/verification.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p20_peft_staged_unfreeze/predictions.npz`
- `_wiki-methodology/_tests/P20_reconstruction_peft_acceptance_evidence.md`

## Impact

P20 验证了 P19 建议的非零初始化、低秩适配和分阶段解冻均已正确实现，且张量、
梯度、更新量均可审计；但当前默认继续使用 P19，P20 保持
`default_enabled=false`。下一步若继续，应改变监督信号而不是只扩大 PEFT 容量：
例如地震重建/对比式辅助目标、层位或断层约束、多尺度体素一致性，或学习 P19
无法表达的残差；不能在相同标量回归目标上继续堆 LoRA rank。

## Prevention Rule (candidate)

小样本 PEFT 必须同时通过三道门：梯度与参数更新真实、严格外折优于现有最佳、
误差相对现有最佳具有互补性；只通过第一道门不得宣称基础模型已产生有效提升。

## Links

- task_plan: ../_task_plan.md
- method: ../../_wiki/_methods/explorations/005-p20-reconstruction-peft-staged-unfreeze.md
