---
phase_id: P31
status: accepted
severity: major
owner_col: COL4
source: user
created_at: 2026-08-01
closed_at: 2026-08-03
closure_evidence: _wiki-methodology/_tests/P31_six_track_agentic_pipeline_acceptance_evidence.md
---

# 智能体优化机制与六赛道独立 Pipeline 注册审计

## 结论

当前没有赛道证明“直接由大模型选择数值动作”能够稳定超过确定性策略。主要原因不是大模型缺少一般推理能力，而是比较协议把它放在了不具优势的位置：动作空间很小，确定性控制器可以近似穷举；智能体只有少量试次，且只看到压缩后的折级反馈。此时，确定性策略拥有更完整的数值证据，智能体却承担最终选型责任，稳定胜出并不符合该实验结构。

下一阶段不应继续只改提示词。更合理的路线是把大模型改为候选生成器和先验提供者，再由贝叶斯优化或多保真调度器完成数值选择。六个赛道也应注册为六条独立 pipeline，但注册必须指向主仓中的稳定入口、门禁和产物合同，不能指向隔离 worktree 或仅有函数库的伪入口。

## 文献依据

- LLAMBO 将大模型用于零样本 warm start、候选采样和 surrogate 建模，并强调其优势主要出现在观测稀疏的搜索早期；该方法是可嵌入传统贝叶斯优化的模块，而不是要求大模型单独替代数值优化器。来源：<https://arxiv.org/abs/2402.03921>。
- AgentHPO 让智能体持续读取任务信息和历史试验，再迭代提出超参数；其有效性依赖完整试验轨迹，而不是一次性分类式选项。来源：<https://arxiv.org/abs/2402.01881>。
- OPRO 将历史方案及其得分放入优化轨迹，说明大模型可以通过反馈生成新方案；这一结论支持“多轮候选生成”，不能推出大模型在有限离散动作表上必然超过穷举算法。来源：<https://arxiv.org/abs/2309.03409>。
- HPO 的结论对搜索空间和预算敏感；不同搜索子空间甚至会反转算法优劣，因此策略比较必须使用一致的预算和可比的搜索域。来源：<https://arxiv.org/abs/2102.03034>。
- 小数据和重复读取验证分数会造成 HPO 层面的 overtuning。近期研究报告约 10% 的案例会选出比默认配置或首个配置泛化更差的超参数，因此本项目必须保留嵌套选择和冻结测试边界。来源：<https://arxiv.org/abs/2506.19540>。

## 本地证据

P29 已修复六赛道的 `observation → action → prediction → metric → promotion` 因果链，并排除了 split、基线口径、held-fold 提示泄漏、恒假 gate 和 no-op 证据不足等问题。修复后，赛道②保留确定性数据集条件化 hybrid，赛道③保留确定性 A2D；赛道④、⑤、⑥拒绝直接智能体数值优化，赛道①仍没有合法 prediction endpoint。该结果说明代码问题确实存在，但修完执行链仍不足以证明直接 LLM 决策优于确定性控制。

P29 之后出现了两项必须纳入 P31 的新证据。赛道①在 `track-fault` 建立了合法连续 3D development 体、verified background/unknown mask 和 group-isolated split；CIG-Bench FaultPredictor 已能真实运行，但修正体素尺度后 guard F1 为 `0.003555`，低于 audited_v2 baseline 的 `0.017641`，因此数据门解除不等于候选晋级。赛道④在 `p11-residual-lithofacies@1e1915f` 将纯 XGBoost `depth=3/eta=0.1/rounds=60` 提升为分支默认，development Macro-F1 从 `0.194938` 提升到 `0.213349`；该提升与 MOMENT 或 LLM 无关，后续所有智能体增益必须相对新 A0 计算。

进一步审计显示，P29 的智能体通常只在很小的离散动作表中运行，并只获得 2--4 次试验预算。确定性对照能够按规则枚举、排序或组合相同动作，因此它拥有更低的搜索 regret。赛道②的单一全局动作还同时作用于 F3 与 Penobscot，数据集异质性使一个动作很难对两者同时最优；确定性 per-dataset hybrid 获胜正好印证了这一点。

项目当前的代码组织也放大了这一问题。`_meta/_registry.yml` 只将 `_pipelines/02_task_datasets/` 注册为一条六赛道总管线；正式 pipeline manifest 只有可视化交付和科研可视化扩展两条。`_codemap.md` 虽列出六赛道入口，但赛道①、②、③的 P4 文件主要是函数库，赛道⑤只有目标 registry；只有赛道④和⑥已经具有较完整的生命周期 CLI。测试门存在，不等于各赛道已经具备可发现、可执行、可验收的独立 pipeline。

## 根因判断

1. **角色错配**：智能体被要求直接给出最终数值动作，而不是利用领域知识生成高价值候选或先验。
2. **预算不对等**：确定性方法可覆盖小动作空间，智能体试次更少，比较的是覆盖率而不是推理质量。
3. **反馈不足**：提示只提供离散或压缩指标，缺少不确定性、成本、约束、失败类型和可迁移的历史轨迹。
4. **缺少数值代理**：当前没有共享 surrogate、acquisition 或 calibrated uncertainty，智能体无法稳定估计 expected improvement。
5. **异质性未建模**：跨数据集或跨折使用单一动作，掩盖了条件化策略的优势。
6. **评估容易过调**：开发折反复用于决策和晋级，若没有嵌套验证与匹配预算，少量正增益可能只是选择噪声。
7. **流程不可发现**：六赛道缺少统一 manifest、stamp 和 P29 门禁，智能体每次都要重新猜入口、状态和产物语义。

## 推荐优化架构

每个赛道采用相同的外层优化合同，同时保留任务专属模型、损失、指标与数据约束：

1. pipeline 输出冻结的 data card、model card、objective card、action-space card 和当前预算；
2. 大模型读取上述合同与合法历史轨迹，提出 `K` 个候选、候选理由和必要约束；
3. 候选进入确定性去重与合法性检查，不合法候选立即拒绝；
4. 贝叶斯优化、TPE 或 ASHA 根据 surrogate、不确定性和成本选择实际运行次序；
5. 低保真试验先筛选，只有候选达到预注册阈值才进入完整训练；
6. 每次运行回写配置、预测、指标、成本、置信度和失败类型，供下一轮智能体归纳；
7. 最终晋级只看嵌套 development promotion，不向智能体暴露冻结测试；
8. 所有策略在相同试验次数、训练步数和数据访问权限下比较。

该架构将智能体的优势放在跨实验归纳、候选扩展和领域约束表达上，将确定性优化器的优势放在数值排序和预算调度上。目标也应从“每次决策都超过确定性策略”改为“在匹配预算下，LLM 增强优化器的中位 regret、胜率或达到目标阈值的成本优于纯确定性优化器”。

## 六条 Pipeline 注册方案

建议在主仓建立以下六个 manifest：

| Pipeline ID | 赛道 | 当前入口状态 | 注册前必须补齐 |
|---|---|---|---|
| `track_fault` | 断层识别 | P30 已补合法连续 3D development endpoint，训练与评测入口仍分散 | 统一 lifecycle CLI；注册 P30 mask/split 数据合同和候选不晋级 verdict |
| `track_facies` | 地震相分类 | 训练与 P5 脚本分散 | 统一 F3/Penobscot 条件化入口、独立指标和 promotion 合同 |
| `track_property` | 储层物性 | 数据、训练与 P4/P5 入口分散 | 统一 family-disjoint split、三目标 reducer 和 checkpoint 入口 |
| `track_lithofacies` | 岩相预测 | 已有较完整 P4 CLI | 补 agent optimizer 步骤、P29 gate 与标准 stamp |
| `track_sweetspot` | 甜点预测 | 七目标 registry 与多阶段 runner 分散 | 建立七目标总控 CLI；未批准目标继续 fail closed |
| `track_reconstruction` | 三维重建 | 已有完整 P4 CLI 与 P17--P24 入口 | 统一当前 P21 默认模型、历史研究入口和 agent optimizer 路由 |

每个 manifest 至少注册 `validate → prepare → baseline → optimize → promote → refit → verify` 七个步骤，并持久化输入哈希、split 哈希、配置哈希、预测哈希、主指标、预算、provider 状态和最终 verdict。P29 目前仍在隔离分支，因此不能把 manifest 指向 `.claude/worktrees/`。正确顺序是先决定并集成 P29 主线实现，再补稳定 CLI，最后生成六个 manifest、registry 条目和门禁 stamp。

## 最小验证计划

首轮只在赛道②和③开展，因为两者已经显示出可利用的条件化信号：

- 对照组：A0、纯确定性 A2D、当前直接 A2L；
- 新策略：LLM warm-start + BO，以及 LLM candidate proposal + BO/ASHA；
- 公平约束：相同训练起点、相同数据、相同总 trial/epoch/GPU 预算和相同 primary metric；
- 重复方式：多个预注册 seed 或 outer folds，不把确定性重复伪装为独立重复；
- 主结果：匹配预算下的 median regret、达到晋级阈值的成本和跨折 win rate；
- 晋级条件：新策略在多数独立折获胜，置信区间不由单一折主导，且冻结测试仍未读取。

只有这一小范围试验通过后，才把新优化器扩展到其余赛道。赛道①已经具备 development prediction endpoint，但当前 CIG-Bench 候选未超过 baseline；赛道⑤必须先确认目标可训练性。赛道④必须以 `0.213349` 的新 XGBoost A0 为起点，赛道④和⑥还应扩大为有科学意义的连续或结构化动作空间，否则智能体仍只是在小列表中做低价值选择。

## 影响

六 pipeline 注册是必要的工程基础，但它本身不会提高指标。它解决的是入口、状态、反馈和证据合同不清晰的问题；真正提高智能体效果的是角色重构、匹配预算、条件化搜索、数值 surrogate 和嵌套验证。两项工作应按“集成 P29 → 补六条稳定 CLI/manifest → 赛道②③小范围混合优化试验 → 通过后扩展”的顺序推进。

## 2026-08-03 落地状态

P29 六赛道修复、P30 断层连续三维证据与④新 XGBoost 默认已选择性集成，并快进合入 `master@0942cf5`。注册名称最终为 `fault_agentic_optimization`、`facies_agentic_optimization`、`property_agentic_optimization`、`lithofacies_agentic_optimization`、`sweetspot_agentic_optimization` 和 `reconstruction_agentic_optimization`。六条线都引用主仓内的 `_pipelines/02_task_datasets/track_lifecycle.py`，不依赖 `.claude/worktrees/` 路径。

六条 pipeline 均已由 `sixone-cli verify-pipeline` 生成内容哈希印记，TOP doctor 报告 8 条已注册 pipeline 全部 `fresh`，无 `stale` 或 `broken`。六赛道 P29/P30/default 聚焦回归与新 lifecycle 回归共 61 项通过。断层 ignored joblib 的五维局部特征逻辑回归参数已转为哈希绑定的可移植系数检查点，在完整 P30 development 体上重算 fit/guard 的 precision、recall、F1、IoU 和 threshold，均与归档结果在 12 位小数内一致。

P31 验收时的默认结论为：①保留断层局部 baseline 和 A2D 治理，CIG/LLM 不晋级；②保留数据集条件化确定性 hybrid；③保留 A2D `reservoir_linear`；④升级纯 XGBoost `depth=3/eta=0.1/rounds=60`；⑤保留冻结 A0；⑥保留 P21 固定三核集成。这些结论区分“确定性优化有效”与“直接 LLM 决策有效”，不再把两者混合归因。后续 P32 已将②③升级为“LLM 有界候选生成 + 确定性预算调度”的混合优化器；P31 的直接 LLM 负面结论仍然有效，新的混合结果见 `P32_hybrid_agent_optimizer_results.md`。

## Prevention Rule

凡比较智能体与确定性优化器，必须使用相同搜索域、相同信息权限和相同计算预算；大模型生成的候选必须经过真实 executor 和独立 promotion，不能用 oracle、冻结测试或确定性枚举结果回写提示后再归因给智能体。

## Links

- task_plan: ../_task_plan.md
- P29 phase: ../_phases/P29_agent_action_effect_repair.md
- registry: ../../../_meta/_registry.yml
- codemap: ../../../_codemap.md
- gates: ../../_tests/_gates.yml
- P31 acceptance: ../../_tests/P31_six_track_agentic_pipeline_acceptance_evidence.md
