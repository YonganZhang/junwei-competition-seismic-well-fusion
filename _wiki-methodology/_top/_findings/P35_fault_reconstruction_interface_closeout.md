---
phase_id: P35
status: accepted
severity: major
owner_col: COL4
source: user
created_at: 2026-08-04
closed_at: 2026-08-04
closure_evidence: _wiki-methodology/_tests/P35_fault_reconstruction_interface_closeout.md
---

# 断层与重建接口收口

## 结论

断层与三维重建的主要遗留问题已经从“散落实验脚本”收口为 Pipeline 内的显式阶段。断层最终评估固定使用 ST10010 连续三维开发体，并把子体构建与 CIG-Bench 公平评估绑定为一个命令；三维重建则将基础模型特征缓存、查询侧地震特征、查询侧基础模型嵌入和混合基线全部改为显式输入。旧 P29 v1 结果不再具有晋级资格，新的 P29 v2 与 P30 v2 成为权威证据。

## 断层结果

CIG-Bench 在 guard 集上的 precision lift 为 `1.179665` 倍，average-precision lift 为 `1.423894` 倍，说明预训练模型并非完全无效；但其最优阈值仅 `3.64e-13`，预测正例覆盖率为 `69.378%`，半径 2 体素容差 F1 为 `0.020238`。因此它只保留为高召回诊断候选，不替代当前断层默认模型。该判断同时满足 development-only、group-isolated split 和 frozen-holdout 未访问约束。

## 重建结果

P29 v2 在五个 held folds 上完成真实 DeepSeek 决策，10 次 provider 调用全部成功。修复后的 A0 RMSE 为 `0.027734374378`，与 P21 一致；策略仅在 2/5 折改善，平均 signed RMSE delta 为 `+0.000181521`，因此确定性门禁保留 P21。P30 v2 再次确认各向异性普通克里金 RMSE 为 `0.030569516403`，回归克里金代理为 `0.030093884156`，均未超过 P21。

未晋级不再被解释为接口错误。当前证据表明：接口已经正确传递训练侧与查询侧模态，智能体也确实执行了真实动作；在现有动作空间和数据预算下，P21 仍是更强的默认模型。

## 跨模态准备

P30 v2 生成 `fusion_io_contract.json`，要求后续井震融合显式提供 KJI、物理坐标、地震体、地震基础模型嵌入、测井观测、测井基础模型嵌入、MD→TVDSS/TWT 对齐、缺失模态掩码、split 身份及来源/权重哈希。输出至少包括孔隙度均值、方差、条件化审计、模态消融和 provenance。该合同是下一阶段“测井基础模型 + 地震基础模型 + 井震跨模态基础模型”的接口起点，但不等于跨模态模型已经训练或获得提升。

## Prevention Rule

任何基础模型特征、时深对齐、查询侧协变量或基线预测都必须作为 Pipeline 的显式参数或注册产物；禁止依赖当前工作目录、隐式 worktree、零填充查询模态或历史输出文件名推断运行状态。

## Links

- task_plan: ../_task_plan.md
- codebook: ../../../_codebook/six_track_pipelines.md
- fault adapter: ../../../_pipelines/02_task_datasets/fault/pipeline_adapter.py
- reconstruction adapter: ../../../_pipelines/02_task_datasets/reconstruction/pipeline_adapter.py
- closure evidence: ../../_tests/P35_fault_reconstruction_interface_closeout.md
