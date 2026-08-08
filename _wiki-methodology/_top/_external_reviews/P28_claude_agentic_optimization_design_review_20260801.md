---
phase_id: P28
status: accepted
owner_col: COL2
source: external
created_at: 2026-08-01
status_basis: 文内含「已采纳修改」段
---

# P28 Claude 独立方法审查

日期：2026-08-01  
审查任务：`20260801T144127__claude__600989`  
状态：`COMPLETE / ERROR_CLASS=NONE`  
原始结果：`_reports/_foreign_aid/20260801T144127__claude__600989/result.md`

## 结论

1. 主识别对照必须是同 observation、动作表、trial 数和预算下的 `A2L LLM 决策 - A3 随机策略`；`A2L-A0` 只能说明自适应搜索是否有用。
2. `A1 advice-only` 是泄漏探针，预测数组与指标必须逐位等于 A0；不相等即为实现污染。
3. 选动作和最终晋升必须使用不相交的 selection-dev 与 promotion-dev 折，避免 dev/OOF winner's curse；重建赛道继续采用 purged spatial folds。
4. 基础模型 on/off 不得混入智能体主效应；在同一比较中固定，或作为独立因子报告。
5. ②地震相与④岩相适合直接比较 LLM 决策和确定性诊断策略；③物性、⑤甜点、⑥重建更适合混合智能体，由 LLM 选路线、诊断分支和停止条件，确定性策略/BO 调数值。
6. ①断层当前缺 verified-negative voxel、连续 3D block 与 group-isolated split，保持 `DATA_GATE_BLOCKED`，不得生成伪 3D IoU 增益。

## 已采纳修改

- P28 增加 `A2L_llm_agent_execute` 与 `A2D_deterministic_agent`。
- 统一主判据改为 best-of-budget、optimization AUC 和 `A2L-A3`。
- LLM observation 仅含 fold-train 聚合诊断与离散的 `improved|flat|worse` 反馈；原始 development metric 只进入冻结执行器日志。
- 进入 Stage 2 前要求 A1/A0 哈希一致、A2L 动作合法、selection/promotion 折不相交、无 frozen-test 访问。

## 边界

本次为只读设计审查，没有运行六赛道新训练，也没有把已有 development 信号写成盲测结论。最终是否保留 LLM 智能体，仍须由 P28 在线 pilot 和负责人独立复核决定。
