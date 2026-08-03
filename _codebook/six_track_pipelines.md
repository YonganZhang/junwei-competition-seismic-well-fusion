---
title: 六赛道统一 Pipeline
user_workflow: 从一个入口查看、规划、预检和核验六个赛道，按固定阶段运行且不能漏过预处理或智能体环节
owner_refs:
- code:six_track_pipeline_cli
- pipeline:fault_agentic_optimization
- pipeline:facies_agentic_optimization
- pipeline:property_agentic_optimization
- pipeline:lithofacies_agentic_optimization
- pipeline:sweetspot_agentic_optimization
- pipeline:reconstruction_agentic_optimization
- test:p34-six-track-pipeline-contract
tracked_paths_hint:
- _code/six_track_pipeline/cli.py
- _pipelines/02_task_datasets/fault/pipeline_adapter.py
- _pipelines/02_task_datasets/facies/pipeline_adapter.py
- _pipelines/02_task_datasets/reservoir/pipeline_adapter.py
- _pipelines/02_task_datasets/lithofacies/pipeline_adapter.py
- _pipelines/02_task_datasets/sweetspot/pipeline_adapter.py
- _pipelines/02_task_datasets/reconstruction/pipeline_adapter.py
- _pipelines/fault_agentic_optimization.yml
- _pipelines/facies_agentic_optimization.yml
- _pipelines/property_agentic_optimization.yml
- _pipelines/lithofacies_agentic_optimization.yml
- _pipelines/sweetspot_agentic_optimization.yml
- _pipelines/reconstruction_agentic_optimization.yml
last_verified_hash: 9bebcc53ce8423ea3801bccdfe004f1ea04bfdc1d08172f4979b981c45ff88d5
validator_version: 1
---

# 六赛道 Pipeline

六个赛道共用一个入口：`python3 _code/six_track_pipeline/cli.py`。它统一负责发现、规划、执行前预检和证据核验；真正的训练命令、参数和产物仍由各赛道 adapter 声明，不再靠临时记忆拼接零散脚本。

> 本页描述 P34 的统一接口。CLI 与 adapter 正在集成时，应以最终 registry、测试 gate 和 `--help` 为准；本页不代表任何实验已经执行，也不声明任何指标结果。

## 基本用法

先用 `list` 查看可用赛道，不会启动训练：

```bash
python3 _code/six_track_pipeline/cli.py list
```

用 `plan` 展开从起点到目标阶段的完整依赖链。下面的命令表示为六个赛道规划到 `verify`：

```bash
python3 _code/six_track_pipeline/cli.py plan \
  --track all \
  --through verify
```

用 `preflight --intent execute` 在任何训练或网络调用前检查 manifest、adapter、参数和数据前置条件。`--param` 可重复提供，但参数必须先由相应赛道声明：

```bash
python3 _code/six_track_pipeline/cli.py preflight \
  --intent execute \
  --track <id|all> \
  --through <stage> \
  --param key=value
```

用 `verify` 核验已产生的阶段证据和依赖关系。只有显式提供 `--output` 时才持久化结构化 trace；未提供时只输出到终端：

```bash
python3 _code/six_track_pipeline/cli.py verify \
  --track <id|all> \
  --through <stage> \
  --output <path>
```

这四个命令承担不同职责：`list` 回答“有哪些赛道”，`plan` 回答“将按什么顺序做”，`preflight` 回答“现在能否安全开始”，`verify` 回答“已完成的阶段是否有可核验依据”。当前统一入口不直接启动长训练或外部 LLM；它先把可执行命令与阻断原因完整暴露。不要用 `verify` 代替实际训练，也不要用已有旧产物绕过 `preflight`。

## 七个固定阶段

六个赛道都遵循同一顺序：`validate → prepare → baseline → optimize → promote → refit → verify`。赛道可以实现不同算法，但不能改变阶段的责任边界。

| 阶段 | 学术含义 | 工程责任 |
|---|---|---|
| `validate` | 明确任务、数据范围、标签可用性和评价协议 | 检查输入、访问边界、split 约束及必要证据 |
| `prepare` | 把原始数据变成可比较的训练样本 | 固定预处理、特征模式、单位、掩码、划分和缓存指纹 |
| `baseline` | 在同一协议下建立当前参照方法 | 记录默认配置、随机种子、指标方向和 incumbent 身份 |
| `optimize` | 在开发集预算内提出并执行候选方案 | 让智能体给出有约束的建议，由确定性调度器执行真实候选并记录 trace |
| `promote` | 判断候选是否足以替代当前方法 | 用独立证据比较候选与 incumbent，执行非退化、预算和数据隔离护栏 |
| `refit` | 用已确认的配置完成最终拟合 | 只使用协议允许的数据重训，不重新选择超参数或偷看冻结测试集 |
| `verify` | 确认结论可以从证据复查 | 核对产物哈希、阶段依赖、决策记录、参数回放和失败状态 |

### 为什么不能跳过 `prepare`

`prepare` 决定模型实际看到的输入，也决定后续指标是否可比较。归一化方式、深度或时间单位、类别映射、空间掩码、训练划分和缓存版本只要有一项不同，baseline 与候选就可能不再处于同一实验条件。

因此，调用下游阶段时必须同时具有与本次计划一致的 `prepare` 证据。旧缓存即使文件名相同，只要参数、上游数据或 adapter 版本不一致，也不能被当作有效前置结果。

## 六个赛道的接线

调用方只使用下表中的 track id。adapter 负责把统一阶段翻译为赛道内的科学实现；manifest 负责声明阶段顺序与验收条件。科学入口可以随研究迭代更新，但公共调用方式保持不变。

| 赛道 | Track id | Adapter | Manifest | 当前科学入口 |
|---|---|---|---|---|
| ① 断层识别 | `fault` | `_pipelines/02_task_datasets/fault/pipeline_adapter.py` | `_pipelines/fault_agentic_optimization.yml` | `_pipelines/02_task_datasets/fault/fault_p29_agent_action_effect.py` |
| ② 地震相识别 | `facies` | `_pipelines/02_task_datasets/facies/pipeline_adapter.py` | `_pipelines/facies_agentic_optimization.yml` | `_pipelines/02_task_datasets/facies/p32_hybrid_agent_optimizer.py` |
| ③ 储层物性预测 | `property` | `_pipelines/02_task_datasets/reservoir/pipeline_adapter.py` | `_pipelines/property_agentic_optimization.yml` | `_pipelines/02_task_datasets/reservoir/p32_hybrid_agent_optimizer.py` |
| ④ 岩相识别 | `lithofacies` | `_pipelines/02_task_datasets/lithofacies/pipeline_adapter.py` | `_pipelines/lithofacies_agentic_optimization.yml` | `_pipelines/02_task_datasets/lithofacies/lithofacies_p33_hybrid_agent_optimizer.py` |
| ⑤ 甜点评价 | `sweetspot` | `_pipelines/02_task_datasets/sweetspot/pipeline_adapter.py` | `_pipelines/sweetspot_agentic_optimization.yml` | `_pipelines/02_task_datasets/sweetspot/p29_agent_action_effect.py` |
| ⑥ 三维重建 | `reconstruction` | `_pipelines/02_task_datasets/reconstruction/pipeline_adapter.py` | `_pipelines/reconstruction_agentic_optimization.yml` | `_pipelines/02_task_datasets/reconstruction/p29_agent_action_effect_repair.py` |

`property` 是公共 track id，而磁盘上的历史目录名是 `reservoir`。这个差异只由 adapter 处理，用户不应把目录名当作新的 track id。

## 智能体如何参与优化

智能体只在 `optimize` 阶段提供受约束的候选建议，例如选择可搜索的参数、模型分支或训练策略。它不能直接改写数据划分、评价指标、预算和 promotion 门槛，也不能把自然语言判断写成实验结论。

候选必须经过确定性执行器真实运行。调度器记录候选配置、预算、随机种子、输入证据和输出摘要，使“智能体选了什么”与“代码实际跑了什么”可以逐项对照。智能体无响应、输出非法或建议超出动作空间时，Pipeline 回退到事先登记的确定性候选或 incumbent，而不是猜测一个配置继续运行。

最终是否晋级由 `promote` 的确定性 guard 决定。guard 至少比较候选与当前 incumbent，并检查主指标方向、任务级非退化、预算一致性和 selection/promotion 数据隔离。智能体可以引导搜索，但不能给自己的候选签发晋级结论。

## 何时会 fail closed

统一执行路径遇到下列情况必须停止，并留下可定位的失败记录，而不是跳过阶段继续：

- track id、阶段或参数未登记，或者 manifest 与 adapter 的阶段声明不一致；
- 前置阶段缺失、失败或指纹过期，特别是 `prepare` 证据与本次数据或参数不匹配；
- 未显式提供 `--output`，输出目录不可写，或 adapter 试图把正式产物写到未声明位置；
- 智能体返回无法解析的动作、越过允许范围，或真实 executor 没有产生对应 trace；
- baseline 身份不明确，候选与 incumbent 预算不一致，或 promotion 没有做 incumbent 对照；
- 选择数据与晋级数据发生禁止的重叠，或运行触碰冻结测试集；
- 验收所需的配置、哈希、指标方向或阶段证据缺失、矛盾、不可回放。

“拒绝候选并保留 incumbent”是有效的科学结果；“缺少证据却继续执行”不是。

## 输出与 trace

`verify --output <path>` 只控制统一核验 trace 的位置。模型、数据集和智能体实验产物仍写入 adapter 的 `expected_outputs` 所列赛道目录；调用前应在 `plan` 和 `preflight` 中确认这些位置，不能把临时缓存当作正式证据。

需要复跑时，应保留原输出作为只读审计材料，并为新的核验 trace 提供新路径，避免新旧记录相互覆盖。trace 中命令使用 `{python}` 和项目相对路径，不固化本机 worktree 地址。

## 推荐操作顺序

第一次接触项目时，先 `list` 确认 track id，再用 `plan` 阅读完整阶段链。准备执行前运行 `preflight`；实验完成后运行 `verify`。如果任何一步返回失败，应先修复其指出的前置条件，再重新生成计划，不要直接进入下游科学脚本。
