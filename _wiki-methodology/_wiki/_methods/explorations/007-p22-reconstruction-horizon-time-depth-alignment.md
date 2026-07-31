# 007: P22 官方层位约束的井震时深重对齐

status: L3_validated_reject | created: 2026-08-01 | updated: 2026-08-01

## Hypothesis

当前重建网格通过弱井震标定被投影到约 1884--2396 ms，而同一区域的官方
Hugin、Shetland、Ty 与 BCU 成对深度/时间层位指向更深的储层时窗。先用官方层位
重建局部深度--时间映射，再抽取地震与基础模型特征，应比继续扩大 LoRA/Adapter
容量更有效。

## Motivation

P20 的 LoRA/Adapter 均有非零梯度但未超过 P19；P21 的对比残差也未跨空间折迁移。
数据审计显示，问题更可能是基础模型输入窗口的地质错位，而不是参数没有更新。

## Source

- `_meta/_data_registry.yml` 登记的 Volve Geophysical Interpretations；
- 官方深度层位与 TWT 层位：Ty、Shetland、Hugin Top/Base、BCU；
- `_pipelines/01_common_preprocess/step_03_load_fault_horizon.py` 的已验证解析器。

## Cost

先做 CPU 级沙盒消融；只有对齐坐标与标量地震属性在同一 OOF 协议下有效，才重建
GFM 特征缓存并进入 GPU 阶段。

## Predicted impact

预期修正地震时窗后，地质相邻关系与 GFM 表征的任务相关性提高；目标是严格低于
P21 pooled OOF RMSE `0.027734374378067677`，且不以新增折损失换取均值微降。

## Implementation

计划入口：`_sandbox/p22_horizon_alignment/run.py`。固定五个空间外折、每折 512 条
训练标签、2048 条验证记录；所有标准化只在当前外折训练行拟合。不得打开
`test.h5`、known holdout 或 frozen test。

## Result

P21 identity RMSE 为 `0.027734374378067677`。官方层位校正使请求点 TWT 中位数
增加约 `647.586 ms`，但固定协议下各路线均退化：仅层位时间坐标
`0.02785355988820412`，仅重采样标量地震属性 `0.02777198671753563`，二者联合
`0.027887281923140557`。在校正后的完整训练体上重新运行真实预训练 GFM 后，
仅替换 GFM 特征为 `0.02778201813325268`，五折全部劣于 P21；完全联合为
`0.027929371081559726`。

## Verdict + Reason

REJECT。时深错位审计是真实的，但官方层位来自 ST10010/解释版本，而目标与输入体
是 Eclipse 2016 + ST0202；当前证据不足以证明这些解释可以无偏替换弱井震标定。
标量和真实预训练 GFM 均未获益，因此不调权重、不晋级、不写入正式 predictor。

## Sub-explorations triggered

- 已完成真实预训练 GFM 的对齐缓存复算，结果仍为负；
- 转入 008：保持空间隔离不变，扩大现有训练区内的合法标签预算。
