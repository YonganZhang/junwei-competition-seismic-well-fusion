---
phase_id: P19
status: accepted
severity: major
owner_col: COL2
source: runtime
created_at: 2026-07-31
---

# 赛道⑥元选择去重后保持稳健改善，尾块微调存在梯度失配

## Local Case

P19 发现 P18 的其余折训练子集含有当前验证折的 24--58 个标签坐标。删除这些
坐标并重新拟合元选择候选后，RMSE 为 `0.027751397628`，相对 PyKrige 改善
`2.4546%`，5/5 空间折改善，说明 P18 主信号没有由该漏洞制造。冻结 holdout
仍未打开，方案继续 `default_enabled=false`。

真实 P15 路径的输出层零初始化导致首步编码器梯度为零；随后编码器每参数梯度
RMS 约为 `3e-9`，三步相对更新约 `2e-5`，而头部相对更新约 `9.4e-3`。最后
一个 GFM 块包含 17,298,000 个可训练参数，相对当前小样本监督预算过重。

## Class Pattern

小样本基础模型适配中，“梯度非零”不等于“有效微调”。必须同时记录逐层张量、
激活分布、每参数梯度 RMS 和相对参数更新；全块解冻若比任务头慢数百倍，优先
缩小适配面或分阶段解冻。跨折候选排名还必须清除被报告折标签通过其他训练子集
间接进入元选择的路径。

## Evidence

- `_pipelines/02_task_datasets/reconstruction/_outputs/p19_training_diagnostics/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p19_training_diagnostics/verification.json`
- `_wiki-methodology/_tests/P19_reconstruction_training_diagnostics_acceptance_evidence.md`

## Impact

当前可信开发结果更新为 `ROBUST_DEVELOPMENT_SIGNAL`、RMSE
`0.027751397628`。普通激活替换、小 MLP、扩大网格、回归克里金和 K/J/I
层序距离均未通过更严格口径，不晋级。后续训练优化应预注册低秩适配器、非零小
初始化输出层、头部热身后逐层解冻和梯度比率门；消融仍按用户要求后置。

## Prevention Rule (candidate)

嵌套空间验证不仅要排除当前折的排名指标，还要从所有元选择模型的训练子集中
删除当前验证折坐标；微调验收不得只检查 `grad != 0`，还要检查相对更新尺度。

## Links

- task_plan: ../_task_plan.md
- method: ../../_wiki/_methods/explorations/004-p19-reconstruction-meta-purge-and-training-diagnostics.md
