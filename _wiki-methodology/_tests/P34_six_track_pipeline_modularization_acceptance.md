# P34 六赛道 Pipeline 模块化验收证据

## 验收结论

六个赛道已分别注册为六条 Pipeline，并通过同一入口完成发现、依赖规划、执行前预检和逐阶段证据核验。公共层不复制模型实现，也不把旧产物存在误写成训练完成；它把每条赛道的真实预处理、基线、智能体优化、晋级、重训和核验入口固定在 adapter 中。

本阶段未重新训练模型、未调用外部 LLM、未读取冻结测试集。`verify` 验证的是已有科学证据及接线一致性；实际执行前仍必须通过 `preflight --intent execute`。

## 六条 Pipeline

| 赛道 | Adapter | 智能体在优化阶段的作用 | 当前执行边界 |
|---|---|---|---|
| 断层 | `fault/pipeline_adapter.py` | 有界门控与动作效应治理 | 数值路线仍受数据门限制 |
| 地震相 | `facies/pipeline_adapter.py` | 生成有界融合候选，确定性调度器控制预算和晋级 | 可执行，但须提供真实数据和设备参数 |
| 物性 | `reservoir/pipeline_adapter.py` | 生成有界候选，确定性调度器完成同预算选择 | 可执行，但须提供训练集与 guard 资产 |
| 岩相 | `lithofacies/pipeline_adapter.py` | 提议超参数候选，强 incumbent 护栏决定是否保留 | 当前默认仍受已验证 A0 保护 |
| 甜点 | `sweetspot/pipeline_adapter.py` | 只在标签合同和数据来源获批后参与动作选择 | prepare、baseline、optimize 明确标为 manual |
| 三维重建 | `reconstruction/pipeline_adapter.py` | 在开发折上提出或选择数值策略，晋级由确定性护栏裁决 | 可执行，但须提供 P19/P21 数据与资产 |

六条 Pipeline 均固定为：`validate → prepare → baseline → optimize → promote → refit → verify`。`plan --through <stage>` 只返回依赖闭包前缀，不能直接跳到下游阶段。

## 自动化核验

| 核验项 | 结果 |
|---|---|
| 统一运行时单元测试 | 14/14 通过 |
| 原 lifecycle 回归测试 | 3/3 通过 |
| 六赛道 manifest/adapter 数量与引用 | 恰好 6 条，通过 |
| 七阶段顺序、依赖、真实 prepare 入口 | 通过 |
| optimize 智能体角色、决策者、候选源、晋级护栏、回退 | 六赛道齐全 |
| `preflight --intent verify --track all` | PASS |
| `verify --track all --through verify` | 6 条 Pipeline、42 个阶段全部 PASS |
| `preflight --intent execute --track all` | 按预期返回 2，在任何训练或网络调用前列出缺失输入、参数和甜点人工阻断 |
| 便携 trace | 使用 `{python}` 与项目相对路径，不含 worktree 绝对路径 |
| Claude 独立终审 | 独立复跑 14 项运行时测试、3 项 lifecycle 回归和 42 阶段核验；六条 manifest 的 stale 验签已全部刷新 |

结构化核验记录：`_wiki-methodology/_tests/_artifacts/P34_six_track_pipeline_verify.json`。

## 防漏约束

1. 下游阶段请求自动带出全部前置阶段，避免忘记预处理或智能体优化。
2. manifest 与 adapter 的赛道名、七阶段顺序、依赖和 registry 引用发生漂移时立即失败。
3. 执行意图先汇总六赛道的缺参数、缺输入和 manual 阶段；预检未通过时不启动任何子进程。
4. 单体优化脚本包含 promote/refit 时必须写明 `included_in`，不伪造不存在的独立入口。
5. 智能体负责候选或策略判断，最终模型晋级仍由固定开发集、预算和确定性门槛裁决。

## 已知边界

- 公共 CLI 当前不直接调度长训练，只提供计划、完整预检和证据核验；真实命令由 adapter 明示，人工确认资源后再执行。
- 执行预检对“可由计划内前置阶段产生”的中间输出按依赖闭包判定，不把这些未来产物误写成已经落盘；最终完成状态仍以阶段执行证据和 `verify` 为准。
- 甜点赛道尚无获批的 canonical 标签合同，且旧 P29 优化器仍依赖隐式 worktree 资产，因此保持 fail closed，不能表述为已自动化。
- 本阶段证明调用结构完整、接线可审计，不证明智能体本身带来新的指标提升。
