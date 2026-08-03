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
