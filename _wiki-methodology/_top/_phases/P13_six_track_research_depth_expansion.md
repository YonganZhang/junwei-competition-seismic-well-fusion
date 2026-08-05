# P13 六赛道科研深度与基础模型扩展

> 状态：2026-07-31 完成。  
> 目标：把六赛道从“模型结果汇总”扩展为可复现的科学问题、受控基础模型对照、空间结构诊断与论文式技术报告。

## 统一研究设计

六赛道均保留强科学基线，并把大模型限制为可检验的候选分支。凡使用预训练权重，必须与同架构随机初始化和原有强基线在同一划分上比较；只有主要指标与领域结构诊断同时改善，候选分支才可晋级。完整合同见
`_pipelines/05_research_visualization_expansion/foundation_model_experiment_contract.json`。

## 主要扩展

| 赛道 | 科学基线 | 基础模型候选 | 新增结构诊断 |
|---|---|---|---|
| 断层预测 | 多尺度不连续属性 + Logistic | SAM-Med3D | 真实 SEG-Y 剖面与断层解释叠合、三维断层棒连续性 |
| 地震相分割 | FPN / DeepLabV3+ | SAM2.1 Hiera | F3 连续切片与十类解释、跨调查区边界连续性 |
| 储层物性 | ExtraTrees / XGBoost | TabICL | 逐井深残差、PHIF–KLOGH 物理一致性 |
| 岩相分类 | 固定九类 XGBoost | MOMENT-1-base | TWT 岩相序列、错误置信度分离 |
| 甜点评价 | 目标专属 GBDT / 历史均值 | Chronos-2 | 因果时间轴、实测–预测散点、残差 Q–Q |
| 三维重建 | OK3D 普通克里金 | OpenMind-MAE | MAPAXES 三维体、正交切片、方向经验变差函数 |

## 关键认识

- 三维重建的点值 RMSE 不能代替空间结构评价。当前条件重建在约 16.6 m 的 K 向距离处只保留真值约
  \(5.4\times10^{-5}\) 的变差，在约 612 m 的 I–J 向距离处只保留约 0.0066，主要失败模式是过度平滑。
- 岩相分类已知留出集共 120 条记录，其中 70 条误分类；错误与正确样本的平均最大类别概率分别约
  0.506 与 0.501，现有置信度不能有效识别错误。
- 甜点 T3 已知留出集的 \(R^2=-0.0636\)。Chronos-2 的开发期收益不能替代冻结时间留出评价。
- 储层物性中 PHIF 与 \(\log(1+\mathrm{KLOGH})\) 的相关系数由观测 0.704 提高到预测 0.790；这说明预测保持主趋势，但也可能压缩真实散布，需与残差和区间覆盖联合判断。

## 交付

- 图件流水线：`_pipelines/05_research_visualization_expansion/`
- 图件清单：`_outputs/research_visualization_expansion/v1/artifact_manifest.json`
- LaTeX 真源：`_paper/technical_report/src/`
- 技术报告：`_paper/technical_report/build/junwei_six_track_technical_report.pdf`
- Claude 交接：`_reports/handoffs/20260731_six_track_pdf_handoff_to_claude.md`
- LaTeX 源码包：`https://share.yongan.site/junwei-six-track-latex-source/junwei_six_track_latex_source_latest.zip`
- 外部审阅：`_reports/_foreign_aid/20260731T023009__gemini__1801638/result.md`

## 科学边界

F3/Penobscot 与 Volve 不混合配准；没有 XYZ 的岩相只展示 TWT 序列；残差不称为不确定性；生成式架构图只解释信息流，不作为性能证据。尚未完成随机初始化或独立开发折的赛道，只陈述研究假设和待验证条件。
