# P18 三维重建的各向异性基础模型地统计路线

日期：2026-07-31

## 问题修正

P17 在同一 pooled OOF 上搜索并报告 156 个候选中的最优值，`-0.4563%` 因而
带有选择后乐观偏差。对原候选族改用嵌套留一空间折选型后，单候选 RMSE 为
`0.028599146195`，top-3 平均 RMSE 为 `0.028534404074`，均不及 PyKrige。
P17 的头条改善据此被 P18 取代，不能继续作为有效增益引用。

## 方法

P18 保留真实冻结 ThinkOnward GFM 和每折 512 条标签预算，但同时修正模型与
评估协议：

1. 每折只用自身 512 个训练点拟合 GFM 标准化和 16 维 PCA；
2. 将垂向坐标乘以独立各向异性系数，再与横向坐标、三通道地震属性和 GFM
   潜坐标组成局部距离；
3. 在该距离下进行不同邻域、距离幂次的反距离插值，并与 PyKrige 保守融合；
4. 对每个待报告空间折，只在其余四折上排名 1,215 个预先限定候选，平均 top-3
   后预测该折。

## P19 口径修正

P19 进一步检查发现：虽然每折自身 512 个训练点与 2,048 个验证点完全分离，
五组验证点也互不重叠，但其余折的训练子集中仍有 24--58 个坐标落在当前被
报告折的验证集合。因而“被报告折标签不参与选型”的原表述过强：标签没有直接
进入排名指标，却可能通过其余折的模型拟合间接影响候选排名。P19 已在每次元
选择时从其余四折训练子集中删除这些坐标后完整复算，P18 数值与协议由 P19
取代。修正后的 RMSE 为 `0.027751397628`，没有推翻主结论。

## 结果

- PyKrige RMSE：`0.028449728170`；
- P18 嵌套 RMSE：`0.027752680679`；
- 相对 RMSE 变化：`-2.4501%`；
- MAE：`0.021413486381 → 0.020830115995`；
- 五个空间折：`5 胜 / 0 负`；
- 20,000 次完整空间折 bootstrap 的 RMSE 差值 95% 区间：
  `[-0.001140994782, -0.000353924655]`，候选更优概率为 `1.0`。

Claude 建议的 PLS 目标感知投影也完成了合法嵌套评估，最佳稳定方案 RMSE 为
`0.027811249487`，未超过 P18。更激进的各向异性精修达到
`0.027732695244`，但只有 4/5 折改善，因稳定性不足未采纳。

## 解释边界

P18 证明完整的“预训练 GFM + 地震 + 各向异性地统计”方案在锁定开发折上有
稳定改善；本轮没有 no-foundation/random-init 消融，不能把 `2.4501%` 因果
归于预训练 GFM。完整空间折只有 5 个，bootstrap 主要概括已观察到的 5/5
方向一致性，不能解释为大样本显著性。部分入选超参数位于有界网格边缘；未来
只能预注册扩展网格后在新外部数据上一次性验证，不能用冻结评估集继续调参。
冻结 holdout 仍未打开，候选保持 `default_enabled=false`。

## 入口

- code: `_pipelines/02_task_datasets/reconstruction/p18_anisotropic_foundation_geostatistics.py`
- test: `_pipelines/02_task_datasets/reconstruction/_tests/test_p18_anisotropic_foundation_geostatistics.py`
- evidence: `_pipelines/02_task_datasets/reconstruction/_outputs/p18_anisotropic_foundation_geostatistics/`
- acceptance: `_wiki-methodology/_tests/P18_reconstruction_anisotropic_acceptance_evidence.md`
- review: `_wiki-methodology/_top/_external_reviews/P18_claude_p17_metric_review_20260731.md`
- correction: `_wiki-methodology/_tests/P19_reconstruction_training_diagnostics_acceptance_evidence.md`
