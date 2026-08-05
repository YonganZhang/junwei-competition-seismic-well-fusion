# P6 Gaia/DAGT 六赛道迁移、微调与胜出验收合同（Claude 审查修订版）

独立审查：
`_reports/_foreign_aid/20260727T000318__claude__3249627/result.md`
（`COMPLETION_STATUS: COMPLETE`、`VERDICT: PASS`；R1–R5 已通过干净 P6 worktree、
飞行员先验、逐目标 gate 和强制 C1/C2 合同修订处理，未发现新的硬性阻断项）。

## 1. 目标与结论口径

本阶段把上游 Gaia/DAGT 已验证的“石油领域 LLM Agent 结构化推理 + 基础模型参数高效微调”
迁移为本项目的公共能力，并分别接入六个赛道：

1. 断层预测；
2. 地震相分类；
3. 储层物性预测；
4. 岩相预测；
5. 甜点预测；
6. 三维模型重建。

本阶段不承诺数学意义上的“六赛道一定超过所有简单模型”。正式完成口径是：

- 每个赛道都得到可运行、可复测、可消融的 Agent + foundation-model 结果；
- 只有在预冻结的盲测协议上超过当前最强可比基线，才标记 `verified_gain`；
- 若在限定实验预算内未胜出，则保留原基线并标记 `no_verified_gain`，必须交付失败原因、
  诊断证据和下一步建议，禁止挑 seed、换测试集或只报最好一次；
- 没有可用真实数据或不满足赛道物理前提时，标记 `blocked_by_data`，禁止用合成数据或假三维
  结果冒充真实胜出。

## 2. 上游来源与允许复用范围

上游仓库：

- 路径：`/mnt/data/yongan-admin-2/projects/自己-全球分辨率填充研究-9a20ac`
- 审查时提交：`0987684d6ecb7409bade73a35555593b678d70e1`
- 重点来源：
  - `dagt/agents/prompt_templates.py`：石油工程 V1/V2 提示词；
  - `dagt/agents/agentize.py`：真 API Agent、缓存、反事实条件、泄漏字段排除；
  - `scripts/finetune_chronos2.py`：Chronos-2 参数高效微调范例；
  - `_pipelines/_steps.yml`、`build_domain.yml`、`predict.yml`：管线结构。

上游锁定文件 SHA256：

- `prompt_templates.py`：
  `355513fb7dffb65f90566dee5bfc44d5115305bdd8ab175ff61c1d856d61e1c8`
- `agentize.py`：
  `a68f716858d5e147a2ec9889cf0c43fe426ea500eaf0ff6861a5bd2575711296`
- `finetune_chronos2.py`：
  `90536b6c47abed5209f112fefcd21ed003a325fc936c776fb1f594fac4f661e6`
- `_steps.yml`：
  `d7454d75e2a97c5909ac26e30df963822eb4bddcbee899f19c4bb1e8623591d0`
- `build_domain.yml`：
  `365f57d473c92a73f60a9865e0e4b9ee7cdf77f490da2391c52bd392e509d149`
- `predict.yml`：
  `b80ffb0b67b4a314aa7d7958ed3f9cc825cf9e32d72db35dc16bf7513bd2c79b`

迁移原则：

- 先登记上游提交、文件 SHA256 和本地迁移清单，再改接口；
- 保留提示词版本、Agent 模型名、API base URL、缓存键和输出 schema 的可追溯信息；
- 禁止直接复制上游过时 README、mock/stub 路径、固定 AC 目标、固定五条曲线或合成油藏假设；
- `inferred_ac_offset` 等近似答案字段不得进入模型条件输入；
- API 不可用时必须明确失败或走预先声明的 `agent_unavailable` 消融，禁止静默退回规则模型后仍称
  为“大模型 Agent”；
- Agent LLM 本身的 SFT 不作为首轮必选项。首轮主路径是“通用 LLM Agent 结构化推理 +
  下游基础模型 LoRA/adapter”；只有具备足够石油指令监督数据时才启动 Agent SFT。

### 2.1 已有 P6 飞行员是负结果先验，不得重跑冒充新实验

分支 `p6-foundation-reprogramming-pilot`、提交 `e4ea27baede1e902d8737c92d1f09b2adec33e92`
已经完成 development-only 的三 seed 飞行员：

- ③物性：Ridge RMSE `0.8646`；预训练 GPT-2 + LoRA RMSE `1.0891 ± 0.0922`，
  未超过简单基线；但较随机 GPT-2 + LoRA 的 `1.2592 ± 0.1847` 改善 `13.50%`；
- ④岩相：Logistic accuracy `0.3939`；预训练 GPT-2 + LoRA accuracy
  `0.2778 ± 0.0554`，未超过简单基线；但较随机 GPT-2 + LoRA 的
  `0.1136 ± 0.0750` 提升 `0.1641`；
- 该结果证明“预训练有信号”，不证明“任务模型胜出”，且不是冻结测试结论。

新实验不得原样重跑上述 GPT-2 飞行员。③④必须从失败诊断、真实 Agent 控制、合适的数值/井曲线
基础模型或更合理的 adapter 入手，并把上述数值作为 B2 先验对照。

### 2.2 Agent 输入来源可用性门

当前项目内未找到可确认与每个预测样本一一对应的原始地质解释/完井文字报告。Volve 的主要输入是
SEG-Y、LAS 和 Eclipse 网格；上游中文报告提示词不能未经验证直接套用。

每赛道必须先将 Agent 模式标为以下一种：

- `predictive_text_agent`：存在真实、样本级、训练时合法可见的文字来源，AgentEvidence 可进入模型；
- `supervisory_qc_agent`：Agent 只读 schema、单位、物理边界和无标签的数据摘要，用于 QC、告警和
  实验编排，不进入预测张量，也不能申报 Agent 精度增益；
- `agent_unavailable`：没有可靠输入或 API 不可用，只运行明确的无 Agent 控制。

禁止把数字特征或模型预测自动改写成自然语言，再让 Agent 复述后作为“新增信息”进入同一模型；
这只能算确定性特征变换，容易形成泄漏或循环论证。六赛道都可以拥有监督/QC 智能体，但只有
`predictive_text_agent` 才能进入 F2/C1/C2 的“Agent 增益”比较。

## 3. 公共可移植接口

公共代码建议落在独立集成分支的 `_models/gaia_dagt/`，先形成单独可 cherry-pick 的提交，再由
六赛道适配。不得在六个赛道各复制一份核心代码。

### 3.1 `TrackSpec`

每个赛道必须声明：

- `track_id`、`task_type`、`modality`；
- 输入字段、目标字段、单位、缺失值规则；
- 合法 train/validation/test 切分和禁止访问的数据；
- 主指标、次指标、方向和容许退化阈值；
- 当前最强可比基线提交、结果文件 SHA256 和复测命令；
- 基础模型、微调方式、参数预算、随机种子；
- Agent 提示词版本、允许使用的原始文本字段和禁止字段。

### 3.2 `AgentEvidence`

统一输出：

- `prompt_version`、`agent_model`、`source_text_hash`；
- `structured_priors`；
- `confidence`；
- 可人工核查的 `evidence`；
- `warnings`、`cache_key`、`provenance`。

Agent 输出只作为条件变量或先验，不得携带标签、测试统计量、由真值反算的数值或文件路径。
`AgentEvidence` 必须内嵌 `prohibited_fields`，至少拒绝：
`label`、`target`、`test_stat`、`test_metric`、`inferred_ac_offset`、真值派生距离/残差及其别名。

### 3.3 `ModelBatch` 与 `ModelOutput`

`ModelBatch` 至少包含：

- 原始模态输入、mask、物理坐标/深度；
- 可选 Agent condition；
- 仅训练阶段可见的 target；
- 样本、井、体、切分和变换 provenance。

`ModelOutput` 至少包含：

- prediction/logits/规则体素 volume；
- uncertainty（若模型确实可估计）；
- Agent 消融标识；
- 模型、权重、数据和配置 provenance。

若模型没有 ensemble、后验样本、MC dropout 或显式分布输出，`uncertainty` 必须为 `null`；
禁止用重构残差、跨样本方差或色散图冒充预测不确定性。

### 3.4 每赛道插件接口

每个适配器实现：

- `build_agent_condition`
- `build_model`
- `fit`
- `predict`
- `evaluate`
- `render_sci`

公共层不得假设所有任务都是时序预测；每个赛道按模态选择基础模型。

## 4. 六赛道首轮技术路线

| 赛道 | Agent 作用 | 首选基础模型/微调 | 首轮不允许的捷径 |
|---|---|---|---|
| ① 断层预测 | 从解释报告、构造样式与属性描述生成空间先验和置信度 | 现有 3D/2D 地震分割 backbone 的 LoRA/adapter 或条件头 | 把 fault label、断层距离真值或测试解释写进 prompt |
| ② 地震相分类 | 提取沉积相、反射结构、邻域连续性先验 | 视觉/地震分割 foundation backbone 的 LoRA/adapter | 随机切片造成相邻泄漏；把类别名直接写入样本 prompt |
| ③ 储层物性 | 将岩性、物性、含烃、邻井描述转为条件协变量 | Chronos/序列或适配的空间属性 foundation model LoRA | 原封不动保留上游 AC、五曲线和合成 bump 假设 |
| ④ 岩相预测 | 提取层序、岩性语义锚点和缺失模态说明 | 序列分类/井曲线 foundation model adapter | 用同井相邻深度跨 train/test；Agent evidence 引用标签 |
| ⑤ 甜点预测 | 汇总地质、工程、含烃证据形成可审计 proxy trace | 多任务表格/时序 foundation model adapter | 先造一个综合甜点评分替代七个冻结目标 |
| ⑥ 三维重建 | 生成约束、边界、物理 QC 和缺失区域先验 | 3D volume/implicit-field foundation model adapter | 用点云或重复二维面冒充完整属性体；用条件开发分数冒充 strict 胜出 |

当前 P6 起始状态：

- ①缺 verified negatives 与合法 development fold，预计为 `blocked_by_data`，readiness 仍需核实；
- ②数据管线可用，但预训练 checkpoint 必须通过许可证和本地权重可用性核查；
- ③④已有 GPT-2/Time-LLM 负结果，禁止原样重跑；
- ⑤只允许已满足数据合同的目标进入训练；T3–T7 若 objective/sample 仍不满足则逐目标
  `blocked_by_data`，不阻止 T1/T2 先交付；
- ⑥只走 3D volume/implicit-field 路线，不复用 Time-LLM。

## 5. 迁移前接口测试

公共层必须先通过以下测试，未通过不得启动六赛道训练：

1. 上游文件登记和 SHA256 锁定；
2. `TrackSpec` schema 校验及 JSON round-trip；
3. `AgentEvidence` 缓存键确定性和 prompt-version pin；
4. 分类、回归、分割、多任务、三维体五类 dummy batch 的 shape/NaN/mask 测试；
5. 无 Agent 条件时的中性输入测试；
6. 真实、乱序、随机、反事实 Agent condition 的可切换测试；
7. 标签字段、测试统计量、真值派生字段的 deny-list 测试；
8. API key 缺失、API 超时、坏 JSON、缓存损坏的显式失败测试；
9. 相同配置、种子和缓存下的可复现性 smoke test；
10. 一个最小端到端 dry run：raw report → AgentEvidence → adapter → prediction → metric → SCI 图。

dummy batch 最小约定：

- 分类/回归：`features [B, L, C]` 或 `features [B, C]`，mask 与样本/序列维可广播；
- 2D 分割：`features [B, C, H, W]`、`prediction [B, K, H, W]`；
- 3D 分割/体重建：`features [B, C, D, H, W]`、输出保持物理体素索引映射；
- 多任务：每个目标拥有独立 mask、metric 和可行性状态，禁止缺失标签被填成负类或零。

## 6. 公平比较与反作弊矩阵

每个赛道在同一冻结切分、同一预处理、同一训练预算和同一评估脚本上至少运行：

- `B0`：当前最强已验证基线；
- `B1`：赛道简单模型；
- `F0`：冻结 foundation backbone + 轻量头；
- `F1`：foundation model LoRA/adapter，无 Agent；
- `F2`：foundation model LoRA/adapter + 真实 Agent；
- `C1`：F2 + 样本间乱序 Agent；
- `C2`：F2 + 反事实/随机 Agent；
- `C3`：若可行，固定模型仅替换 Agent condition 的干预测试。

F2/C1/C2 只适用于 `predictive_text_agent`。若赛道是 `supervisory_qc_agent`，仍须交付 Agent 的
结构化 QC 结果和准确的工作流状态，但模型比较停在 F1，结论不得含 `agent_signal`。

公平性硬约束：

- split、样本 ID、预处理和测试指标在训练前固化并保存 SHA256；
- 测试集只在模型和超参数冻结后运行一次；调试与选择只看 train/validation；
- 同一比较使用相同 seed 集合，首轮至少 3 个 seed；小样本时补 bootstrap 置信区间；
- 总可训练参数、训练步数、GPU 时长和峰值显存全部记录；
- 调参上限预注册，不因结果不好临时扩大某一方法预算；
- 结果必须同时报告均值、离散度、逐 fold/逐目标指标和失败 cell；
- `C1/C2` 与 `F2` 无差异时，只能说明 Agent 未被有效利用，不能宣称“大模型增益”；
- `C1/C2` 优于 `F2` 或出现反常高分时，优先判定泄漏、prompt 标签暗示或 split 问题。

## 7. 胜出门槛

每个赛道预检时冻结一个主指标。建议起点如下，最终以现有协议和数据可用性为准：

- ①断层预测：fault-class F1 / mIoU，并以边界质量和校准作为 guardrail；
- ②地震相分类：macro-F1 / mIoU，并按独立空间块报告；
- ③储层物性：每个冻结目标的 RMSE/MAE/R²，禁止只报跨目标综合分；
- ④岩相预测：固定类别集合的 macro-F1 / balanced accuracy；
- ⑤甜点预测：七个目标逐项使用预注册分类或回归指标，禁止用新造综合分替代；
- ⑥三维重建：strict 空间留出的体素/剖面误差、结构一致性和井约束吻合；没有 strict 数据则不得
  宣称胜出。

`verified_gain` 必须同时满足：

1. `F2` 在 untouched test 的主指标优于 `max(B0, B1)`；
2. 提升跨 seed/fold 稳定，置信区间或配对统计支持不是单次偶然；
3. 次指标、物理约束、校准和资源 guardrail 无重大退化；
4. `F2` 优于 `F1`，且真实 Agent 优于 `C1/C2`；
5. 独立复测可从锁定输入与提交重现；
6. 结果图与指标文件 hash 一致。

如只满足“foundation model 胜出”但 Agent 不胜出，则标记 `foundation_gain_only`；如 Agent 有增益但仍
未超过 B0，则标记 `agent_signal_but_no_baseline_win`。

## 8. 有界自动实验策略

每赛道先跑 smoke，再跑首轮 3-seed 正交矩阵；仅允许围绕验证集进行有界迭代：

- 数据/标签/切分 bug 修复；
- 学习率、adapter rank、上下文长度等少量预注册参数；
- 一次架构适配修正；
- 一次提示词 schema/证据约束修正。

每赛道最多两轮正式迭代。两轮后没有 `verified_gain` 也必须结束并交付真实结论，禁止无限搜索。
GPU 任务通过项目锁串行或按显存安全分组；CPU 数据准备、测试与作图可并行。

GPU 锁必须是可审计的互斥文件锁或项目现有 `VOLVE_P5_GPU_LOCK` 等价机制；worker 不得只凭
`nvidia-smi` 瞬时空闲就启动。锁记录 job、赛道、GPU、PID、开始/结束时间。未完成显存 smoke
前默认一张 GPU 只允许一个训练 job。

## 9. Goal 两阶段派发

本任务满足长周期、目标稳定、可独立验收和需要自动 continuation 的条件，因此六个赛道都应使用
`/goal`，但不得直接盲发。

### Phase A：只读 readiness

六窗口先分别读取：

- 本合同；
- 自己赛道的当前 HEAD、现有 strongest baseline、数据合同、测试防火墙和 SCI 图管线；
- 公共 Gaia/DAGT 核心及契约测试；
- 自己可写范围和 GPU/数据限制。

为避免破坏旧窗口现场，Phase A/B 不在六个原 `track-*` worktree 直接实施。负责人从以下已独立
验收提交分别建立新的 `p6-gaia-*` 干净 worktree：

- ① `0ff064d8e8850611df2f6eea0eab1aeebf03721c`
- ② `450f3d5840ac8d25bc8f3d5e9029753f1cdbe591`
- ③ `65740d49b479b1716f8c1a1d807ca4915d7a1dba`
- ④ `e5d5cbce2d2e26ce8479d4e6731b32d0bec16362`
- ⑤ `d008a62da3d29b28be19d992e7173bfa8181d010`
- ⑥ `7e35b45540d4391dbdf79097b1dd23a6db283102`

这同时解决旧 `track-property` 的未跟踪文件和旧 `track-lithofacies` 分支漂移问题：两处现场原样
保留，不删除、不切换、不带入 P6。

统一返回：

- `UNDERSTANDING`
- `SOURCES_READ`
- `BASELINE_LOCK`
- `METRIC_AND_SPLIT_LOCK`
- `FOUNDATION_MODEL_CHOICE`
- `AGENT_INPUT_AND_DENYLIST`
- `AGENT_MODE_AND_SOURCE_PROOF`
- `C1_C2_IMPLEMENTATION_PATH`（若为 predictive Agent；否则说明为何不适用）
- `PILOT_NEGATIVE_RESULT_DIAGNOSIS`
- `ASSUMPTIONS`
- `BLOCKING_QUESTIONS`
- `DRAFT_GOAL_COMMAND`
- `READINESS`

### Phase B：正式 Goal

仅对 `READINESS=READY` 或有明确可交付子目标的 `PARTIAL_READY` 赛道发送一次经负责人审定的
`/goal`。例如⑤可只实施 T1/T2，同时诚实锁住 T3–T7。Bus 合同记录完整 objective、acceptance、
authority、write ownership、时限和资源约束；`goal achieved` 仅是 worker 自报，仍须 collect、
独立复测、核对 hash 后才能 close。

Phase B 对 predictive Agent 必须明确写入：F2、C1、C2 同批同 seed 运行，不允许先看 F2 后决定
是否补控制；若 C1/C2 挂钩不存在，先实现并通过契约测试，不能把控制组留到事后。

## 10. 每赛道交付物

每个赛道必须交付：

- 锁定的 `TrackSpec`；
- 赛道适配器与测试；
- B0/B1/F0/F1/F2/C1/C2 结果表；
- 逐 seed/fold/目标明细；
- 资源与失败日志；
- 一个结论 JSON，状态只能是：
  `verified_gain`、`foundation_gain_only`、`agent_signal_but_no_baseline_win`、
  `no_verified_gain`、`blocked_by_data`；
- SCI Plot：预测、真值、残差/错误、不确定性（仅在真实可估计时）、Agent 消融与基线对比；
- 适用三维任务提供真实规则属性体或诚实的 `not_feasible`，禁止假体；
- 一个干净、可审计的本地提交。

## 11. 独立验收与停止线

负责人逐赛道独立执行：

- baseline 复测；
- 新模型测试与最小预测；
- split/泄漏审计；
- Agent shuffle/counterfactual 审计；
- 结果文件和图件 hash 审计；
- worktree/提交范围审计；
- 石油流程和可视化合理性复核。

任一硬条件失败即进入一次 rework；第二次仍失败则按真实失败状态关闭，不得包装成胜出。
