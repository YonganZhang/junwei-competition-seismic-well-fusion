# 六赛道模型总表汇总

- 统一工作簿：`_outputs/model_results_summary/six_track_model_metrics.xlsx`
- 单 Sheet：`模型指标`
- 行数：1456
- 列数：26
- 六赛道：facies, fault, lithofacies, property, reconstruction, sweetspot

结论：
- fault：data_blocked，当前没有可直接交付的有效 fault 开发分数。
- facies：SAM2 预训练路径优于随机初始化，但仍低于强基线；修复后的门控残差也没有把它推过基线。
- property：TabICL 的三目标开发集结果是有效的，PHIF / KLOGH / SW 的 RMSE 分别约改善 4.9%、13.6%、25.2%，但仍按 effect_supported_not_promoted 和重现阻断分开保留。
- lithofacies：MOMENT 小幅优于随机初始化，但仍低于 XGBoost 基线。
- sweetspot：T3 有提升，T4 总体不利，T5 blocked。
- reconstruction：预训练比随机初始化约好 48.5%，但仍低于 PyKrige。

校验：
- 所有非空 evidence_path 已通过存在性检查；总表包含 6 个赛道。
- 组合后的状态分布：{'reference_only': 7, 'data_blocked': 2, 'ranked': 632, 'non_beneficial': 136, 'control': 59, 'effect_supported_not_promoted': 14, 'reference': 258, 'confirmed_holdout': 12, 'evidence_only': 36, 'beneficial': 119, 'production_reference': 12, 'diagnostic_only': 84, 'proxy_feasible': 16, 'promote': 5, 'rejected_no_gain': 4, 'blocked': 2, 'complete': 11, 'passed': 31, 'negative_control': 8, 'known_holdout_confirmation': 8}

说明：
- 这里只合并已发布的赛道产物，不改科学数值，不读 holdout，不重训。
