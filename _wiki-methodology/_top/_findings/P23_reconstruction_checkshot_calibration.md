---
phase_id: P23
status: accepted
severity: major
owner_col: COL2
source: runtime
created_at: 2026-08-01
---

# 赛道⑥ Checkshot 独立校验修正了时深关系，但未产生稳定的下游误差收益

## Local Case

项目早期将稀疏分层点的 MD–TWT 插值作为弱井震标定，并误认为本地
缺少 VSP/checkshot。资产复核发现 `Volve_Seismic_VSP.zip` 实际包含 5 口井
的 checkshot。P23 固定用 19A/19BT2/19SR 拟合，将 F11T2/F15A 完全留作
独立校验。在两口独立井共 80 个目标储层点上，旧弱标定的 TWT MAE
为 `633.1867 ms`，checkshot 候选降至 `8.7389 ms`，2/2 校验井均改善。

下游五折复测保持 P21 的模型、512 标签预算和参数不变。全体积替换与
直接观测支持门控的 RMSE 分别为 `0.027768546911` 和 `0.027790989240`，
均未超过 P21 的 `0.027734374378`。

## Class Pattern

独立校验应针对一个明确命题。本轮已证明“时深标定更准”，但这不等于
“当前下游估计器会利用这个改正”。当地震权重较小、核与训练协议固定时，
输入层的物理校正可能仅改变局部邻域，而不会自动转化为稳定的目标 RMSE 收益。

## Evidence

- `_wiki-methodology/_wiki/_methods/explorations/011-p23-reconstruction-checkshot-target-tie.md`
- `_wiki-methodology/_wiki/_methods/explorations/012-p23-reconstruction-checkshot-aligned-gfm.md`
- `_wiki-methodology/_wiki/_methods/explorations/013-p23-reconstruction-checkshot-support-gate.md`
- `_wiki-methodology/_tests/P23_reconstruction_checkshot_calibration_evidence.md`

## Impact

Checkshot 时深关系作为研究阶段已验证的标定资产保留；旧弱标定仅为
已有管线复现而保留。P21 仍为赛道⑥默认模型，P23 不声称孔隙度预测提升。
真正的独立孔隙度结论仍需要从未参与开发的新地质实现、新目标井或赛事
隐藏测试集。

## Prevention Rule (candidate)

井震校正必须先在未参与拟合的 checkshot 井上验证，再独立评估下游目标；
不得用下游指标退步否定标定精度，也不得用标定精度代替下游性能证据。

## Links

- task_plan: ../_task_plan.md
- method: ../../_wiki/_methods/explorations/011-p23-reconstruction-checkshot-target-tie.md
- test: ../../_tests/P23_reconstruction_checkshot_calibration_evidence.md
