# P34 六赛道 Pipeline 模块化

## 目标

将六个赛道收口为六条可发现、可规划、可预检、可逐段核验的 Pipeline。保留各赛道的科学实现，不搬运或重写训练代码；统一层只负责阶段合同、赛道适配、依赖闭包、智能体角色和反伪完成证据。

## 统一阶段

每条 Pipeline 固定经过七段：`validate → prepare → baseline → optimize → promote → refit → verify`。

- `validate`：检查数据、标签、split 和冻结集边界。
- `prepare`：调用真实公共预处理或赛道数据构建入口。
- `baseline`：建立本赛道当前默认参照。
- `optimize`：由智能体提出或选择候选，确定性调度器控制预算。
- `promote`：只用隔离的 development/promotion 证据决定是否晋级。
- `refit`：按冻结配置重训或确认既有默认模型。
- `verify`：逐段复核证据、哈希、产物和最终结论。

## 实施边界

1. 六个根 manifest 仍是六条 Pipeline 的接线真源。
2. 六个 `pipeline_adapter.py` 只描述各赛道如何接入公共接口，不复制科学实现。
3. 统一运行时必须先做完整 preflight；参数、入口或前置阶段缺失时，在任何训练或网络调用前失败。
4. `prepare` 不得退化为只看旧 summary；必须指向真实数据构建模块。
5. `optimize` 必须声明智能体角色、最终决策所有者、晋级护栏和失败回退。
6. 单体优化脚本内含的 promote/refit 必须显式标记 `included_in`，不得伪装成独立执行器。

## 验收标准

- 恰好注册六个赛道，且每条都有一致七阶段合同。
- `plan --through <stage>` 自动包含所有前置阶段，不能跳过 `prepare` 或 `optimize`。
- `preflight` 能在动作发生前报告缺参数、缺入口、缺智能体合同和不安全的 manual/delegated 阶段。
- `verify --track all` 逐赛道逐阶段运行现有证据校验并生成可选结构化 trace。
- adapter、manifest、registry、CodeBook、codemap、测试 gate 和 TOP 相互可达。
- 六条 Pipeline 重新盖机器印记，`closeout` 与 `doctor` 通过。

## 不在本阶段做

- 不重新训练六个模型，不调用外部 LLM，不打开冻结测试集。
- 不以统一接口为理由强迫分割、分类、回归和三维重建共用内部张量形状。
- 不把尚未包装为安全命令的科学入口包装成“已自动执行”；这类入口必须 fail closed。

## 实施结果

- 新增 `_code/six_track_pipeline/` 公共运行时，提供 `list / plan / preflight / verify`。
- 新增六个赛道 adapter，显式登记真实预处理、基线、智能体优化、晋级、重训和核验入口。
- 六个根 manifest 已从同一生命周期校验器引用改为各自 adapter 引用；公共 lifecycle 继续作为底层证据核验器。
- `verify --track all` 已顺序核验 6 条 Pipeline 的 42 个阶段，全部通过并生成便携 trace。
- `preflight --intent execute --track all` 已按预期 fail closed：集中报告数据、参数和⑤甜点人工门禁，且未启动训练或网络调用。
- Claude 独立终审复跑了 14 项运行时测试、3 项 lifecycle 回归和 42 阶段核验；其发现的六条 stale pipeline 验签已全部刷新。关于甜点 `python -m` 的疑问经本机实测排除，模块帮助入口正常退出。

⑤甜点是当前唯一不能无人值守执行的赛道。原因不是统一层缺失，而是 canonical 标签合同尚未获批、数据构建入口只支持审计、旧优化器仍含隐式 worktree 依赖；三段均以 `manual` 明示，后续补齐科学真源后再解除。

验收证据见 `_wiki-methodology/_tests/P34_six_track_pipeline_modularization_acceptance.md`。
