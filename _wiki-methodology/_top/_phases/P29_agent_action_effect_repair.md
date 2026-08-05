# P29 六赛道智能体动作效应修复与再验证

状态：`COMPLETE_VERIFIED`  
日期：2026-08-01  
目标：区分“LLM 决策错误”“提示信息不足”“动作执行未接线”“比较或评测错误”和“动作空间确实无有效信号”，先修协议与代码，再以小预算、同起点、同口径实验复核智能体能否带来预测指标增益。

## 1. 触发原因

P28 把六赛道统一概括为“智能体真实执行，但没有 prediction endpoint 增益”。重新逐行审计后，这个总括不够准确：部分赛道确有正向候选，部分赛道的 LLM 在其可见信息下做出了合理决定，但反馈口径、对照公平性、动作空间或晋级逻辑掩盖了结果。因此 P29 不预设“智能体无效”，也不预设“必须提升”，而是逐赛道验证完整因果链：

`observation → prompt → selected_action → effective_config → optimizer/executor → prediction_hash → primary_metric → promotion → endpoint`

## 2. 已确认根因

| 赛道 | 已确认事实 | 当前主因 | P29 修复方向 |
|---|---|---|---|
| ①断层识别 | P28 只做四个门禁场景的动作分类；动作没有进入训练或预测；A2L 4/4 动作正确但 evidence F1 为 0 | 任务定义不是 prediction optimization；evidence schema 提示不足 | 修正证据字段和 validator；保持 `DATA_GATE_BLOCKED`，不伪造预测提升；补“门禁动作确实改变状态”的契约测试 |
| ②地震相分类 | 五个动作均改变配置、梯度与预测；A2L 首轮选中最佳 `FAC_GATE_050`；相对 A0 等权 mean mIoU `+0.026226`，但 Penobscot `-0.008510` | 单一全局动作无法处理双数据集冲突；4/5 动作预算使各策略终点必然收敛；观察信息过粗 | 改为数据集条件化动作/双头 gate，小预算比较 sample efficiency；保留 guard，不用均值掩盖回退 |
| ③储层物性 | A0 checkpoint 的原训练集合已含 P28 selection/promotion 井族；候选仅从零训练约 2 epochs，而 A0 为 60 epochs；global gate 又包含永远不可能提升 1% 的 A1 | split 泄漏、不同起点/预算的不公平比较、晋级逻辑恒假、primary/secondary metric 混用 | 重建合法 family-disjoint split；同一 A0 checkpoint 继续训练或同预算重训；逐策略 gate 排除 A0/A1；统一 primary metric |
| ④岩相预测 | XGBoost 动作与预测真实变化；A2L 接近 inner ceiling，但 outer 0/4 提升；三 seed 因完全确定性配置得到相同 hash | inner→outer transfer、动作空间上限、伪重复 seed、提示过度离散化 | 使用真实随机 subsampling 或取消伪重复；给安全归一化幅度与类别支持；按 outer fold 预注册迁移策略 |
| ⑤甜点预测 | 候选 selection MAE 错与历史 Stage3 总体基线比较，导致四项全标 `worse`；DeepSeek 因而合理停止；同场 promotion 中已有候选 MAE `132.050` 优于 A0 `144.083` | 决定性的 feedback-baseline 口径 bug；控制组缺独立性；selection→promotion 转移弱 | 基线改为 P28 同 fold、同执行器 A0；重新生成 prompt；控制组使用独立固定序列/随机种子；明确只以合法 selection 选优 |
| ⑥三维重建 | 9 个动作均产生不同预测；全开发诊断最优可从 RMSE `0.0277344` 降到 `0.0276690`（约 `0.236%`，仅作上限诊断）；A2L promotion 仍略差于 A0 | categorical-only prompt、低杠杆单因子动作、折间噪声、缺少可部署 predictor endpoint | 做 prompt 信息消融；扩展 foundation/vertical/seismic 等高杠杆动作；补获胜配置 replay endpoint 合同 |

## 3. P29 统一实验约束

1. 不读取冻结测试标签，不更改任务标签、官方 metric 或既有 split 真源；发现 split 污染时先修 split，再运行任何比较。
2. 同一比较必须同初始 checkpoint、同训练步数、同数据、同预处理与同 primary metric；A1 只做 identity replay，不参加“全部策略均提升”的 global gate。
3. Prompt 可以看到 development-only 的归一化相对增益、跨折 win/loss、方差/置信区间、剩余预算和晋级阈值；仍不得看到逐样本标签、残差、冻结测试指标或路径。
4. 每个动作必须记录 config hash、参数更新或执行器状态、prediction hash 和 primary-metric delta；至少有一条测试断言“不同有效动作产生不同配置，并在适用时产生不同预测”。
5. 先做 2--4 个动作的小范围反事实测试；确认提示或执行链有辨识度后才扩大。不得为了得到正数而读取 promotion 结果回写 prompt。
6. 报告同时给出 A0、A1、A2L、A2D、A3 和 oracle ceiling（oracle 仅用于诊断动作空间上限，不得作为合法模型选择）。

## 4. 分赛道 Goal 交付合同

每个赛道在自己的隔离 worktree 中完成：

1. `root_cause.md`：逐项回答 prompt、action executor、prediction、metric、promotion 和 endpoint 是否接线；
2. 修正后的 P29 runner 或 P28 最小补丁，不覆盖 P28 既有证据；新产物写入 `_outputs/p29_agent_action_effect/`；
3. `action_effects.json`：A0 与每个动作的 config/prediction hash、primary delta、合法选择可见性；
4. 至少一项 prompt 信息消融和一项 action-effect/no-op 回归测试；
5. 小预算结果与 honest verdict：`RETAIN_AGENT`、`RETAIN_HYBRID`、`REJECT_AGENT` 或 `DATA_GATE_BLOCKED`；
6. 仅提交本赛道 P29 文件，保持既有无关改动不变，返回 commit、测试、产物哈希和遗留风险。

## 5. 验收标准

- 若 LLM 看到的信息足以区分动作，且 selected action 确实改变 prediction endpoint，再讨论智能体是否带来增益。
- 若同一动作空间的合法 oracle ceiling 不高于 A0，则结论是动作空间不足，不归咎于 LLM。
- 若 oracle 有提升而 A2L 没选到，比较 prompt-enhanced、A2D 和 A3 的 regret/sample efficiency；只有 A2L 可重复优于对照才保留直接决策角色。
- 若候选提升只来自数据泄漏、不同训练预算或 promotion 回看，则全部作废并重跑。
- Claude 独立只读审查与负责人复跑测试通过后，才允许改变 `default_enabled` 或 champion/registry。

## 6. 最终结果

| 赛道 | 修复后结论 | 归因与保留策略 | 验收提交 |
|---|---|---|---|
| ①断层识别 | `DATA_GATE_BLOCKED` | 门禁动作、证据白名单和执行器状态已经可验证；A2L 决策准确率低于 A2D，且当前不存在合法 prediction endpoint，因此只保留确定性门禁控制，不声称模型指标提升 | `91e5243` |
| ②地震相分类 | `RETAIN_HYBRID` | 直接 A2L 晋级增益为 0，`retain_agent=false`；独立 A4 按数据集选择 F3=`FAC_GATE_050`、Penobscot=`A0`，promotion mean mIoU 相对 A0 增加 `0.0304809775`，只归因于确定性混合策略 | `47c5f81` |
| ③储层物性 | 直接 A2L 不保留，保留 A2D | 重建同模型、同 split、同 2-step 预算的因果 A0；A2L 与 A2D 均优于 A0，但 A2D 略优，故 `keep_llm=false`，历史 `best.ckpt` 只作非因果诊断 | `5574709`、`54cae62` |
| ④岩相预测 | `REJECT_AGENT` | 使用真实三个 inner folds 估计不确定性，取消伪重复 seed；A0 Macro-F1=`0.2133487970`，A2L=`0.2047148654`，差值 `-0.0086339317`，且增强提示相对分类提示无收益 | `c9be2e9` |
| ⑤甜点预测 | `REJECT_AGENT` | feedback baseline 已修为同折、同执行器 A0；A2L 看到所有非基线候选均为非正增益后合法停止，确定性控制独立运行，未发现可晋级的 LLM 数值优化作用 | `3654066` |
| ⑥三维重建 | `RETAIN_FROZEN_BASELINE` | 五个外折均采用 held-fold 排除的实时 DeepSeek 决策；safe policy 的平均 signed RMSE delta=`+0.000176898`，仅 2/5 折为正，故不保留直接智能体优化；oracle 只作动作空间上限诊断 | `4742637`、`3786bd7`、`c0cee0d` |

统一结论不是“智能体一定有效”或“智能体完全无效”，而是：六条赛道的动作链现已能够区分配置变化、预测变化与最终指标变化；原先确有 split、基线口径、晋级 gate、held-fold 提示泄漏和 no-op 证据不足等代码或协议问题。修复后，②和③存在可保留的确定性决策/混合策略，但没有赛道取得足以单独归因给 LLM 的稳定 endpoint 优势。正确决策仍可能因动作空间上限、跨折迁移或双数据集冲突而不提升，因而必须把“动作真实生效”和“策略优于对照”分开检验。

## 7. 验收记录与边界

- 两个负责人批次已逐赛道 collect/verify；补充批次 `junwei-p29-agent-action-repair-successor-20260801` 于 2026-08-01 完成关闭。
- 独立复测：① 5 项、② 13 项、③ 7 项、④ 11 项、⑤ 4 项、⑥ 6 项聚焦测试通过；各赛道提交后工作树与产物哈希按本赛道合同核验。
- ②的 F3 promotion 与 Penobscot selection 各有一个配置类别在该折无样本；类别支持向量已持久化并哈希。因此 `+0.0304809775` 只作为当前 development promotion 证据，不外推为所有类别、所有数据集均提升。
- 所有赛道保持冻结测试关闭；oracle、promotion 或已知 holdout 结果均不得回写 prompt。
- Claude 静态独立复核见 `_reports/_foreign_aid/20260801T210450__claude__2459352/result.md`；其结论用于确认问题分类，最终数值以修复后 runner、测试与哈希产物为准。
- P29 提交仍位于各隔离赛道分支；本阶段完成的是因果链修复与验收，不自动替用户执行跨分支集成。
