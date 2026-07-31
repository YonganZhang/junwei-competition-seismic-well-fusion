---
phase_id: P24
status: accepted
severity: major
owner_col: COL2
source: runtime
created_at: 2026-08-01
---

# 冻结 P21 在未使用的同场区历史属性版本上保持孔隙度重建增益

## Local Case

现有 `test.h5` 已在 P5 Stage-4 使用，不能再次称为首次盲测。P24 因此在
读取目标指标前提交预注册协议，固定使用 RMS 工程中未被现有管线使用的
`pp04phif/realisation.1`。该属性与最终 `merge_pp04b_PHIF_NW` 位于同一
`pp04bxg03postf11a` 网格。通过 Eclipse KJI→IJK 转置与正值掩膜，最终
RMS 属性的 508,622 个值可与 Eclipse 属性逐元素完全一致，因而历史属性
可无歧义地映射回现有折的 KJI 坐标。

协议保持 P5/P21 的 5 个空间折、每折 512 条训练标签和 2,048 条验证标签，
并锁定 P21 的三个 foundation 核及 PyKrige 1.7.3 基线。预注册提交为
`1a3a056`，执行代码提交为 `4ee7a32`。唯一一次目标评估得到：

- PyKrige RMSE/MAE：`0.028235410003 / 0.021293578592`；
- 冻结 P21 RMSE/MAE：`0.027825182663 / 0.020826337775`；
- RMSE 相对改善：`1.4529%`；
- 空间折结果：4 胜、1 负；
- 整折 bootstrap 候选更优概率：`0.94595`，95% 区间仍跨 0。

结果通过预注册的“至少 1% pooled RMSE 改善、最多 1 折退步”门槛。

## Class Pattern

冻结的预训练表征若只在单一开发目标上改善，仍可能是目标版本特有的选择结果。
在同一输入几何和同一标签预算下更换未参与开发的历史属性版本，可以检验邻域
度量是否对目标版本扰动具有迁移性，同时避免重新搜索超参数。

## Evidence

- `_pipelines/02_task_datasets/reconstruction/p24_historical_transfer_preregistration.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p24_historical_transfer/summary.json`
- `_wiki-methodology/_tests/P24_reconstruction_historical_transfer_evidence.md`
- `_wiki-methodology/_wiki/_methods/explorations/014-p24-reconstruction-historical-version-transfer.md`

## Impact

P21 现在同时具有开发 OOF 支持和同场区历史版本迁移支持。该结果增强了“真实
预训练 GFM 表征在固定邻域构造中有用”的证据，但不构成跨场区、赛事隐藏测试
或首次盲测结论。P21 继续作为赛道⑥当前默认模型；下一次性能结论只接受新的
外部油田、官方隐藏测试或预先封存的新井/新目标面。

## Prevention Rule (candidate)

已使用的测试面不得更名为独立盲测；替代目标必须在开指标前锁定来源哈希、
空间映射、基线、候选参数和成功门槛，开指标后不得在同一目标上调参。

## Links

- task_plan: ../_task_plan.md
- method: ../../_wiki/_methods/explorations/014-p24-reconstruction-historical-version-transfer.md
- test: ../../_tests/P24_reconstruction_historical_transfer_evidence.md
