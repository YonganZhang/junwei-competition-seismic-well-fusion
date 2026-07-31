---
phase_id: P17
status: superseded
severity: major
owner_col: COL2
source: runtime
created_at: 2026-07-31
closed_at: 2026-07-31
superseded_by: P18_reconstruction_anisotropy_recovers_robust_signal
closure_evidence: _wiki-methodology/_tests/P18_reconstruction_anisotropic_acceptance_evidence.md
---

# 赛道⑥的基础模型问题主要是角色不匹配，而非接线失效

## Local Case

P14 冻结特征直接回归、P15 局部微调和 P16 遮蔽重建都没有超过 PyKrige。
P17 保留同一真实预训练 GFM，并将它从直接目标拟合改为非平稳邻域构建。
它曾在同一 pooled OOF 上选出 `0.028319907650`，但 P18 的嵌套复核表明原
候选族 RMSE 为 `0.028534404074`，没有超过 PyKrige。原改善已被取代。

## Class Pattern

小样本科学建模中，预训练表征并不必然适合直接微调为目标解码器。当传统模型已经
编码了任务中的强先验时，更有效的融合位置可能是距离、核、协方差或正则化项，
而不是替代强基线。

## Evidence

- `finding:P17_reconstruction_foundation_role_mismatch`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p17_foundation_geostatistics/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p17_foundation_geostatistics/verification.json`
- `_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md`

## Impact

P17 仍证明“调整基础模型角色”是合理方向，但不能用其同 OOF 最优值证明提升。
后续结论统一引用 P18 的嵌套选型与各向异性结果。

## Prevention Rule (candidate)

基础模型直接预测失败后，应先审计“融合位置是否匹配强先验”，再决定是否更换模型；
不能用额外 OOF 标签扩大法定训练预算来制造改善。

## Links

- task_plan: ../_task_plan.md
- method: ../../_wiki/_methods/explorations/002-p17-reconstruction-foundation-geostatistics.md
- superseding finding: P18_reconstruction_anisotropy_recovers_robust_signal.md
