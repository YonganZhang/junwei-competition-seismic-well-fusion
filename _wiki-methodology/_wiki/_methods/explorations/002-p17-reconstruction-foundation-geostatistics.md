# P17 三维重建基础模型的非平稳地统计路线

日期：2026-07-31

## 问题

赛道⑥的 PyKrige 基线在稀疏井点约束下很强，而 P14 冻结 GFM 特征回归、P15
局部微调和 P16 遮蔽重建均未提升 OOF RMSE。这不说明预训练模型没有信息，
更可能说明“用大模型直接拟合孔隙度残差”与任务结构不匹配。现有标签预算
每折只有 512 点，无法支撑高自由度解码器可靠地重建三维连续场。

## 方法转换

P17 将预训练 GFM 从“目标预测器”改为“局部几何先验”：

1. 用真实 ThinkOnward GFM 编码目标无关的地震道；
2. 每个外层空间折仅在 512 个合法训练点上拟合标准化和 PCA；
3. 将物理坐标、局部地震属性和弱加权 GFM 坐标组成非平稳距离；
4. 在该距离下做局部反距离插值，并与不变的 PyKrige OOF 保守融合。

这样的分工与数据规模一致：地统计模型负责稀疏标签插值，基础模型只负责
表达地震结构相似性。

## 开发集设计

- 外层单元：5 个已锁定空间折；
- 每折训练预算：512 条 `point_train` 标签；
- 每折验证：2,048 条原 PyKrige OOF 样本；
- 候选空间：13 组正的 GFM/地震权重、3 个邻域尺度、4 个融合权重，
  共 156 个有界候选；
- 主指标：池化开发 OOF RMSE；不确定性按完整空间折 bootstrap；
- 防火墙：CLI 无 test/holdout 参数，不打开 `test.h5`，编码器不读目标。

## 结果

最佳候选为 `gfm_metric_f0.05_s0.10_k128_blend_0.75`。PyKrige RMSE 为
`0.028449728170`，P17 为 `0.028319907650`，相对变化 `-0.4563%`。MAE 从
`0.021413486381` 降至 `0.021200329887`，偏差绝对值也降低。五个独立空间折
为 3 胜 2 负；整折 bootstrap 中候选更优的概率为 `0.7668`，95% RMSE 差值
区间为 `[-0.000589605334, +0.000128806422]`，仍跨 0。

## 解释边界

该结果证明“基础模型参与非平稳邻域构建”已出现合法的开发集信号，但不证明
它已稳定超过 PyKrige，也不能将改善因果归因于预训练权重。按用户要求，
本阶段不做 no-foundation/random-init 消融；候选保持 `default_enabled=false`，冻结
holdout 保持封存。

## 入口

- code: `_pipelines/02_task_datasets/reconstruction/p17_foundation_geostatistics.py`
- test: `_pipelines/02_task_datasets/reconstruction/_tests/test_p17_foundation_geostatistics.py`
- evidence: `_pipelines/02_task_datasets/reconstruction/_outputs/p17_foundation_geostatistics/`
- acceptance: `_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md`
