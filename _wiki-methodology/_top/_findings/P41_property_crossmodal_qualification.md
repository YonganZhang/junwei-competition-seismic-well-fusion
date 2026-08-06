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

## 结论修正（2026-08-06，Codex 独立复核后收窄）

原始 `R0_STOP_NO_ATTRIBUTABLE_SIGNAL` 结论把负结果解释为"井震融合在③赛道无可归因信号"，**这个外推范围过宽，已被收窄**。独立复核（用户质疑触发，Claude 子智能体 + Codex 各自独立用真实数据复核，均重新提取过 embedding 并从原始 SEG-Y 重建过 patch）确认了与④赛道 P40 同一类结构性接口缺陷：1216 个样本的 MOMENT 测井侧 embedding 各不相同，但 GFM 地震侧 embedding 只有 211 种取值（=211 个不同的 `section_id+trace_id`），1201/1216 行落在"同道重复组"里；组内真实物性目标仍有明显差异（PHIF/log1p(KLOGH)/SW 组内标准差中位数分别为 0.02448/0.62695/0.05558，最大达 0.09891/3.61959/0.43732），但地震 embedding 对这些不同深度点完全相同。根因同样是 GFM 只保留了整道 CLS token，没有把查询点 time_idx 传入。

这不是数据/标签/坐标对齐 bug（两次独立复核均确认张量非空、无 NaN、真实进入模型、标签与原始 SEG-Y 重建逐元素零误差），而是**接口设计缺陷**，把 trace-level 表示误当作 depth-local 表示使用。因此本轮 `0.1703%` 的微弱改善及"不晋级"结论**只能证明"整道级、无深度分辨率的地震表征提供不了足够信号"，不能证明"井震融合本身在③赛道无效"**。正确定性应为 `R0_INCONCLUSIVE_DEPTH_BLIND_SEISMIC_INTERFACE`，下一步应换用能分辨深度的局部地震 token（参考 ⑥赛道 P39 的 query-local 波形 token 设计）重新测试。

## 原始结论（保留存档，范围已收窄见上）

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
- 深度分辨率独立复核笔记: `_sandbox/p37_p41_data_integrity_audit/reservoir_p41_notes.md`
