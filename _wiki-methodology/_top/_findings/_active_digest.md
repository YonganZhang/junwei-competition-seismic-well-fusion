---
phase_id: index
status: accepted
owner_col: COL2
source: navigation
created_at: 2026-08-07
---

# Active Findings 摘要

新会话接手时先读这一页，再按需展开单个 finding。这里只列 `status: accepted|pending` 的条目；
`superseded` 的历史结论不在此处，避免沿着已被取代的路径继续研究。

> 本页是导航层，不是真源。每条结论的证据以对应 finding 文件为准。
> 赛道当前冠军与已否决路线另见各赛道自己的真源，⑤甜点为
> `_pipelines/02_task_datasets/sweetspot/_outputs/incumbent/incumbent.json`。

## 按赛道快速定位

🔴 **读这张表前先分清两个层级，混淆它们会得出完全错误的项目判断：**

- **赛道层**：这条赛道是否跑通、有默认模型、相对基线有改善。**六条赛道全部成立。**
- **路线层**：某条具体的增强路线是否通过晋级门。部分未通过。

P42 的原话：*「拒绝表示候选真实执行但没有超过当前默认，**不表示 Pipeline 失败**。」*
一条路线未晋级 ≠ 该赛道失败。护栏挡住一次假晋级，本身是赛道健康的证据。

### 赛道层：六条全部成功

| 赛道 | 当前默认模型 | 相对基线的改善 | 展开 |
|---|---|---|---|
| ① 断层 | 局部 baseline + A2D 决策治理 + P30 连续三维体 | pipeline 七阶段 PASS，三维开发体与 group-isolated split 已集成 | P25.1、P30 |
| ② 地震相 | P32 混合优化（F3 联合配置 + Penobscot A0）| 等均 mIoU `0.301022749 → 0.325721341`；F3 `+0.049397`，Penobscot 不降 | P32 |
| ③ 储层物性 | P32 候选生成 + 确定性调度 | 复合 RMSE 改善 `4.2696%`，3/3 seed 全胜 | P32 |
| ④ 岩相 | XGBoost `depth=3/eta=0.1/rounds=60` | Macro-F1 `0.194938 → 0.213349` | P33 |
| ⑤ 甜点 | T3 = Chronos-2 `F1_chronos2_train_blend`；T1/T2/T4 = P5 小模型 | T3 MAE `267.118 → 186.572`（`-30.15%`），正式 `PROMOTE` | P43、P44 |
| ⑥ 三维重建 | P21 固定三核集成 | 开发 OOF RMSE 相对 PyKrige 改善 `2.5144%`；P24 同场区迁移再改善 `1.4529%` | P19–P24 |

### 路线层：这些具体路线未通过晋级门

未晋级不等于失败，等于「真跑过、没赢过当前默认」。重开需要新证据。

| 路线 | 涉及赛道 | 结果 |
|---|---|---|
| 冻结双基础模型融合 | ③④⑥ | ③改善 `0.1703%` 区间跨零；④`0.161312` < B0 `0.213349`；⑥`0.075908484` 弱于 well-only `0.075314433` |
| 直接 LLM 数值裁决 | 六赛道 | 均无可归因的稳定 endpoint 优势 |
| CIG-Bench 断层评测 | ① | 高召回、相对先验有 lift，但精度不足 |
| PEFT / 对比残差 / checkshot 下游 | ⑥ | 均未超过 P21 |

跨赛道有效的是**混合智能体**（LLM 提候选 + 确定性执行 + 独立晋级门），在②③取得可归因增益。

## 全部 active 条目

| Phase | Severity | 结论 | 文件 |
|---|---|---|---|
| P2.2 | - | 纠正：③④赛道数据量足够，之前 finding 把"最窄子集"误当"全部数据" | `P2.2_volve_well_log_scale_correction.md` |
| P2.3 | - | Volve 数据集完整逐层扫描：修正岩心覆盖结论 + 记录命名陷阱 | `P2.3_volve_full_inventory_scan.md` |
| P2.4 | - | Volve 数据内容级程序化验证（非抽样）：真实通过率与已知问题清单 | `P2.4_volve_content_level_validation.md` |
| P2.5 | - | Codex+Claude 联合审查：数据完整性确认无遗漏，算法源头盘点，registry 漂移修正 | `P2.5_joint_data_and_algorithm_source_audit.md` |
| P2.6 | - | 六赛道 worktree 合并就绪审计：全部 MERGE_READY=NO，核心阻塞是 0 commits | `P2.6_six_tracks_merge_readiness_audit.md` |
| P16.1 | - | 基础模型迁移诊断：完整方案取得提升，赛道⑥需要改变基础模型职责 | `P16.1_foundation_model_transfer_diagnosis_and_recovery.md` |
| P19 | major | 赛道⑥元选择去重后保持稳健改善，尾块微调存在梯度失配 | `P19_reconstruction_meta_purge_and_gradient_diagnosis.md` |
| P20 | major | 赛道⑥ LoRA 可修复梯度失配，但当前监督目标与 P19 表征高度冗余 | `P20_reconstruction_peft_staged_unfreeze.md` |
| P21 | major | 赛道⑥改变 PEFT 学习信号仍未形成空间可迁移残差，固定基础模型核更稳 | `P21_reconstruction_contrastive_residual.md` |
| P23 | major | 赛道⑥ Checkshot 独立校验修正了时深关系，但未产生稳定的下游误差收益 | `P23_reconstruction_checkshot_calibration.md` |
| P24 | major | 冻结 P21 在未使用的同场区历史属性版本上保持孔隙度重建增益 | `P24_reconstruction_historical_transfer.md` |
| P25.1 | - | ①断层 CIG-Bench：两个真实 bug 修复后产出有效对比，但须降级为"对齐后可测但较弱的信号" | `P25.1_cigbench_fault_st10010_alignment_recovery.md` |
| P31 | major | 智能体优化机制与六赛道独立 Pipeline 注册审计 | `P31_agent_optimizer_and_six_pipeline_registration_audit.md` |
| P32 | major | 混合智能体优化结果：②mIoU +0.024699、③复合 RMSE -4.2696% | `P32_hybrid_agent_optimizer_results.md` |
| P33 | major | ④岩相现默认护栏阻止一次错误晋级 | `P33_lithofacies_incumbent_guard_prevents_false_promotion.md` |
| P35 | major | 断层与重建接口收口 | `P35_fault_reconstruction_interface_closeout.md` |
| P37 | major | 三口父井无法同时满足原生 PHIE 与合法 development 支持，未启动模型小试 | `P37_real_well_seismic_cross_modal_foundation.md` |
| P38 | major | 双预训练融合 RMSE 0.079781229，未超过 well-only 0.075314433 | `P38_real_well_phif_direct_seismic_fusion.md` |
| P39 | major | 固定共同基线后双预训练 0.075908484，区间与错位门均未通过 | `P39_query_local_well_seismic_foundation_fusion.md` |
| P41 | major | 物性双基础资格门：改善仅 0.1703%，2/4 外层胜出且区间跨零，不晋级 | `P41_property_crossmodal_qualification.md` |
| P42 | major | 六赛道最新进度与接手说明；⑤甜点列为第二优先级，瓶颈在数据与标签门 | `P42_six_track_progress_and_claude_handoff.md` |
| P43 | major | ⑤甜点 T6/T7 卡在样本身份不可逆而非缺特征源；T2 是 SAND_FLAG 代理任务 | `P43_sweetspot_seven_target_gate_root_cause.md` |
| P44 | **critical** | ⑤甜点 T1/T2/T6/T7 的标签均为 CPI 解释产物，单条 RHOB 即达 R² 0.9696；T6/T7 的 `is_proxy=False` 标注与证据不符 | `P44_sweetspot_label_provenance_collapse.md` |

## 不要重启的路线

以下已有明确否定证据，除非出现新数据或新对照，否则不重开：

- 直接让 LLM 做数值裁决 / 直接选最终配置 —— 六赛道均无稳定优势（P28、P29、P31）
- 赛道⑥的 LoRA、分阶段解冻、对比残差 —— 均未超过固定三核 P21（P20、P21）
- 冻结双基础模型融合用于③④⑥ —— 三条赛道均未通过晋级门（P39、P40、P41）
- ⑤甜点 T1/T2/T6/T7 的模型优化 —— 标签是解释产物，无待学未知量（P44）
