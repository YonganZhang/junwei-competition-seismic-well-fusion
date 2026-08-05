---
phase_id: P41
status: accepted
severity: major
owner_col: COL3
source: experiment
created_at: 2026-08-05
closed_at: 2026-08-05
closure_evidence: _pipelines/02_task_datasets/reservoir/_outputs/p41_property_crossmodal_qualification/artifact_manifest.json
---

# 储层物性井震双基础模型资格门

## 结论

P41 完成了③储层物性赛道的开发集井震双基础模型资格门。冻结 MOMENT 测井特征与 GFM 地震特征真实进入门控残差头，但未达到预注册晋级条件。因此，本轮结论为 `R0_STOP_NO_ATTRIBUTABLE_SIGNAL`，不进入 LoRA、Adapter 或分阶段解冻。

## 实验结果

实验以 P5 Stage-3 强基线为 B0。该基线对 PHIF 和 KLOGH 使用 ExtraTrees，对 SW 使用 XGBoost，输入为 81 维地震 patch、36 维测井值和 36 维缺失掩码。四个开发井族采用外层 LOGO，并在外层训练井族内完成全部归一化、降维和残差训练。冻结测试集没有读取入口。

F1 双基础模型融合将等权复合 RMSE 从 `0.427225121813` 降至 `0.426497477630`，相对改善 `0.1703%`。该变化只有 `2/4` 个外层井族获胜，三个目标中只有一个满足非退化条件。配对 bootstrap 改善区间为 `[-0.005133873428, 0.006637890596]`，仍跨越零。`fusion_off` 与 B0 的最大绝对误差为 `0.0`，说明残差接口可以精确回退；预训练与随机、打乱和错位控制也确实产生了不同预测，但这些变化没有形成稳定泛化增益。

## 决策边界

本结果只说明当前四井族、单种子和冻结表征下没有达到可归因晋级门。它不等于井震融合无效，也不等于后续参数高效微调没有潜力。下一步只有在新增独立井族、改善井震配对质量或获得更明确的任务对齐预训练信号后，才应重新开启适配器训练。

## 复现入口

- runner: `_pipelines/02_task_datasets/reservoir/p41_property_crossmodal_qualification.py`
- foundation adapter: `_pipelines/02_task_datasets/reservoir/p41_foundation_features.py`
- summary: `_pipelines/02_task_datasets/reservoir/_outputs/p41_property_crossmodal_qualification/summary.json`
- independent metric check: `_pipelines/02_task_datasets/reservoir/_outputs/p41_property_crossmodal_qualification/independent_metric_check.json`
- artifact manifest: `_pipelines/02_task_datasets/reservoir/_outputs/p41_property_crossmodal_qualification/artifact_manifest.json`
