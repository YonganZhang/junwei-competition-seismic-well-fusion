# 012: P23 Checkshot 对齐地震属性与 GFM 复测

status: L3_validated_reject_full_replacement | created: 2026-08-01 | updated: 2026-08-01

## Hypothesis

011 在两口独立井上将目标储层 TWT MAE 从 633.19 ms 降到 8.74 ms。
若 P21 中的地震属性和 GFM 确实受到错误时窗限制，则用三口拟合井的
checkshot TVDSS/TVD–TWT 曲线重采样后，固定 P21 核应在同口径 OOF 上改善。

## Frozen protocol

- 只读 reconstruction `train.h5`；`test.h5` 与所有 holdout 不打开。
- 标签、五个外层空间折、每折 512 训练标签、2048 验证行、PyKrige 基线、
  P21 `z=4/f=0.1/s={0,0.1,0.2}/k=64/p=1.5/blend=0.75` 全部固定。
- 只替换前三个地震属性通道，并在更正后三维属性上重新提取真实
  pretrained GFM 特征；目标、坐标、井约束和 mask 不改。
- 拟合只用 19A/19BT2/19SR checkshot。F11T2/F15A 继续封存，不进入训练体。
- 对 checkshot/trajectory 末端之外、但仍在储层网格中的深部单元，只允许使用
  最后 20 个 checkshot 点的线性趋势做最多 300 m 外推；不回退到 MD 弱标定。
- 主指标仍为 10,240 行 pooled development OOF RMSE。

## Acceptance

候选 RMSE 必须严格低于 P21 `0.027734374378067677`，且相对 P21 最多
1 个空间折退步。否则不进入正式流水线。

## Implementation

沙箱入口：`_sandbox/p23_checkshot_aligned_gfm/run.py`。

## Result

全体积 Checkshot 对齐版 RMSE 为 `0.02776854691114144`，相对 P21
`0.027734374378067677` 变差 `+0.000034172533073764666`，折结果为
3 胜 / 2 负。fold 0/1/4 分别改善 `0.00010957/0.00012527/0.00005274`，
fold 2/3 分别退步 `0.00052943/0.00014995`。

输入审计显示：134,222 个活跃单元全部有至少两口井可用，但 71,163 个单元
至少依赖一条最深 300 m 的 checkshot/轨迹末端线性外推。全体积替换因此将
已独立验证的插值区与未独立验证的深部外推区混在同一个候选中。

## Verdict + Reason

REJECT FULL REPLACEMENT。候选虽在 3/5 折改善，但 pooled RMSE 和“最多 1 折
退步”均未达标，不正式化。这不推翻 011 的独立标定结论；它只表明不应
对缺乏直接 checkshot 支持的深度做全体积强制替换。

## Sub-explorations triggered

013：仅当至少 2/3 拟合井在该深度上有直接 checkshot 与轨迹观测支持时，
才使用对齐后地震属性；否则回退到已验收 P21 输入。
