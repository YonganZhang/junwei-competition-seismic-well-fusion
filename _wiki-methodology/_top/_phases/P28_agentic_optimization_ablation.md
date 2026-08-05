# P28 六赛道执行型智能体优化与消融协议

状态：`STAGE1_CLOSED_HONEST_NEGATIVE`  
日期：2026-08-01  
目标：把六赛道现有“分析/质检智能体”升级为受约束的实验决策者，让它真正选择并执行优化动作；用等预算消融判断智能体是否有效，有效才保留。
独立审查：

- 设计预审：`_wiki-methodology/_top/_external_reviews/P28_claude_agentic_optimization_design_review_20260801.md`
- 首轮终审：`_reports/_foreign_aid/20260801T162948__claude__1249571/result.md`
- 因果修正终审：`_reports/_foreign_aid/20260801T170829__claude__1594905/result.md`
- 重建精确同一性复核：`_reports/_foreign_aid/20260801T173049__claude__1769727/result.md`
- 项目内摘要：`_wiki-methodology/_top/_external_reviews/P28_claude_agentic_pilot_review_20260801.md`

## 1. 核心问题

本阶段不回答“调用了大模型没有”，而回答：

> 在固定数据划分、评测口径、算力预算和候选动作空间时，智能体能否比固定方案、只写建议和随机选择更快找到更好的 development 配置？

智能体只能读取训练/开发证据，包括训练曲线、梯度范数、特征统计、模型复杂度、历史 trial 和资源消耗；不得读取冻结测试标签、原始测试预测或测试指标，不得修改 split、metric、预处理真源或评测函数。

## 2. 统一消融

| 组别 | 决策者 | 是否调用大模型 | 是否执行建议 | 作用 |
|---|---|---:|---:|---|
| `A0_static_baseline` | 冻结人工配置 | 否 | 否 | 当前强基线 |
| `A1_advice_only` | LLM | 是 | 否 | 隔离“只生成文字建议”；预测必须等同 A0 |
| `A2L_llm_agent_execute` | 受约束 LLM 智能体 | 是 | 是 | 从批准动作表选择、执行、读取反馈并继续决策 |
| `A2D_deterministic_agent` | 规则/诊断智能体 | 否 | 是 | 同 observation、动作表和预算的确定性决策策略 |
| `A3_random_policy` | 固定随机策略 | 否 | 是 | 同动作表、同 trial 数、同预算，判断收益是否只是“多试了几次” |
| `A4_deterministic_search` | 网格/TPE/贝叶斯优化 | 否 | 是 | 最终保留前的等预算数值优化对照；小试阶段可先离线 replay |

所有执行组从同一初始状态开始，使用相同 action IDs、trial 数、单 trial wall-clock 上限、GPU 上限和早停规则。主识别对照是 `A2L-A3`；`A2L-A0` 只回答“自适应是否有用”，不能单独归因于 LLM 决策质量。`A1` 是泄漏探针：预测数组和指标必须逐位等于 A0，否则判为实现污染并立即停止。主结果同时报告：

1. primary metric 的最终 best-of-budget；
2. 每一步 best-so-far 与其离散积分（running-max path score；不是 ROC-AUC）；
3. 无效 JSON、越界动作、crash、超时和显存峰值；
4. 各折/各数据集方向一致性。

## 3. 高效执行顺序

### Stage 0：离线契约与 replay

- 复用已有 P13/P17/P19--P24 候选结果，隐藏 candidate metric、随机打乱 action ID，让各策略选择后再揭晓结果。
- 目的只验证 observation/action/schema、决策质量和 replay，不把离线赢家冒充新实验提升。
- 每赛道预计分钟级，无 GPU 或只需轻量 CPU。

### Stage 1：低成本在线 pilot

- 每赛道只允许 2--4 个真实 development trial。
- 先跑不改配置的 A0；再按相同预算跑 A2L、A2D 和 A3。A1 只验证建议已生成且预测哈希等于 A0。
- A2L 未优于 A0 或随机策略中位数、动作越界、发生 leakage、两次连续无效决策时立即停止，不扩大训练。
- 选动作使用 selection-dev 折，最终晋升使用未参与选动作的 promotion-dev 折；同一 OOF 既选优又晋升的结果只记探索证据，不得默认启用。

### Stage 2：确认性消融

- 只对 Stage 1 通过者增加 A4、重复 policy seed 和完整合法 development folds。
- 最终保留要求：A2L 在 best-of-budget 或 running-max path score 上优于 A3，并相对 A2D/A4 仍有可解释优势；否则把数值调参交给 A2D/A4，只保留 LLM 的高层路线选择、故障诊断与停止决策职责。

## 4. 六赛道首轮动作空间

动作必须离散化为允许列表；LLM 只能返回 `action_id`，不能直接生成 shell 或任意代码。

| 赛道 | 首轮 observation | 首轮允许动作 | primary metric | pilot gate |
|---|---|---|---|---|
| ①断层识别 | 合法 development mask 状态、正负例覆盖、连续 3D block 与 group-isolated split 状态 | 本轮无模型动作；仅数据门四项修复任务 | average precision；3D IoU 仅数据门通过后启用 | verified-negative voxel=0 或仅有 2D 切片时记 `DATA_GATE_BLOCKED`；不得产生伪 3D 排名 |
| ②地震相分类 | F3/Penobscot 分任务 mIoU、attention entropy、fusion scale、梯度、更新范数 | fusion-scale 初值、SAM2 frozen/top-2、fusion LR 档、Dice 权重、融合开关 | F3 与 Penobscot 等权 mean mIoU | mean mIoU 提升且两任务无不可接受退化；必须与继续训练 CNN 对照 |
| ③储层物性 | PHIF/KLOGH/SW 训练侧诊断、guard 漂移、特征相关、模型容量、TabICL/CIG 可用性 | LLM 只选模型路线/诊断分支；路线内部的容量、正则和融合权重交给 A2D/A4 | train-std normalized composite RMSE | composite 改善且任一目标不得灾难性退化；family split 不变 |
| ④岩相预测 | LOGO4 macro-F1、per-class F1、类别支持、MOMENT 贡献、校准、模型复杂度 | XGBoost depth/eta/rounds、class-weight 档、MOMENT gate、prior calibration、特征 lane | LOGO4 fixed-schema macro-F1 | 至少 3/4 折方向不差，平均提高达到预注册阈值 |
| ⑤甜点预测 | 各目标可行性、T3 训练侧时序诊断、Chronos/XGBoost 可用性 | LLM 只选 T3 模型路线/是否继续；blend weight 与 XGBoost 数值交给 A2D/A4；T5--T7 保持既有门禁 | T3 causal macro-fold MAE | 至少 3/4 promotion-dev 折改善且因果特征边界不变；不得为阻塞目标造标签 |
| ⑥三维重建 | P21/P24 训练侧各向异性、邻域统计、foundation 核、CIG RGT 可用性 | LLM 只选 P21/CIG-KED 路线与诊断分支；核参数交给嵌套 CV/BO | held-label-purged nested spatial-fold RMSE | 严格优于 P21，0 个明显 fold loss；迁移证据单列，不混入同场 OOF |

具体阈值由各赛道在预检中依据现有指标分辨率和噪声预注册，看到新结果后不得修改。

## 5. 执行型智能体合同

每轮固定为：

1. `observe`：执行器生成脱敏、development-only JSON；LLM 只看 fold-train 聚合诊断和 selection-dev 结果的离散状态（`improved|flat|worse`），不看标签、逐样本残差或原始 metric 数值；
2. `decide`：LLM 接收 observation 与 allowlist，严格返回一个 action ID、置信度、理由和 stop 标志；
3. `validate`：schema、allowlist、预算、split hash 和 deny-list 全部通过才执行；
4. `act`：执行器根据 action registry 调用固定 Python 入口，LLM 不生成命令；
5. `measure`：由冻结评测器计算 primary metric；原始 selection-dev 数值只进入执行日志和控制器，不进入 LLM prompt；
6. `learn`：将 trial 结果加入下一轮 observation；
7. `stop/promote`：按预注册门槛停止、扩大或拒绝。

LLM 请求失败、返回越界 action、JSON 不合法或超时必须 fail closed；不得静默改用人工最优动作。允许显式回退为 `A0`，并把回退记入日志。

## 6. 结构化证据

每个 trial 至少记录：

```text
schema_version, run_id, track_id, policy_id, policy_revision,
provider, model, model_revision, policy_seed,
split_hash, metric_id, action_space_hash, observation_hash,
observation, candidate_action_ids, selected_action_id,
decision_confidence, decision_rationale, stop_requested,
executor_entrypoint, config_hash, wall_clock_budget_s,
exit_status, runtime_s, peak_vram_mb,
baseline_metric, candidate_metric, delta, fold_metrics,
prediction_hash, frozen_test_accessed, leakage_checks,
artifact_paths, artifact_hashes
```

另记录 `selection_fold_ids` 与 `promotion_fold_ids`，并断言两者不相交；基础模型是否开启在同一组主比较中保持固定，避免把基础模型本身的增益误归因为智能体增益。

原始日志采用 JSONL；每赛道另产出 `summary.json`、`evidence.md`、`artifact_manifest` 和最小单测。`prediction_hash` 用于证明 A1 与 A0 完全一致。

## 7. 晋升与保留

### 进入 Stage 2

- A2L 至少成功执行两个合法、不同的动作；
- strict JSON/allowlist 通过率 100%；
- 没有 frozen-test、split、metric 或 train-only fit 泄漏；
- A2L 优于 A0，且至少优于 A3 多个固定随机种子的中位数；
- 提升不是单个数据集、单个目标或单个折的灾难性换取。

### 最终保留

- 重复 policy seeds 或合法独立折方向稳定；
- A2L 的 best-of-budget 或 running-max path score 优于 A3；
- 与 A2D/A4 比较后仍有价值。若确定性策略更适合数值参数，则采用混合智能体：LLM 决定模型路线、动作空间、诊断分支和停止条件，A2D/A4 在选定分支内调连续/离散数值；
- 由独立 Claude 只读复核 observation、decision、action、result 与 leakage；
- 负责人独立重跑测试和哈希验收后才允许 `default_enabled=true`。

## 8. 调度

- 当前 P18 CIG-Bench 批次占用 ①③⑥，先完成并验收，不进行双重派活。
- ②④⑤先做 `GOAL_PREFLIGHT`：只读真源、预注册动作表与门槛，不执行长训练；Claude 设计审查已于 2026-08-01 完成并通过，预检验收后再派低成本 online pilot。
- ①③⑥在 P18 收口后使用同一合同补齐。
- 第一批验收完成前不展开完整六赛道多种子训练。

## 9. Worker 返回格式

每个赛道按以下顺序返回：

1. `CONCLUSION`：是否适合进入在线 pilot；
2. `SOURCES_READ`：实际读取的代码、数据合同和证据；
3. `FROZEN_BASELINE`：primary metric、split hash、评测入口；
4. `OBSERVATION_SCHEMA`；
5. `ACTION_ALLOWLIST`：2--8 个动作及单 trial 预算；
6. `ABLATION_MATRIX`：A0、A1、A2L、A2D、A3、A4 如何公平运行；
7. `LEAKAGE_FIREWALL`；
8. `PROMOTION_GATE`；
9. `DRAFT_GOAL_COMMAND`；
10. `READINESS=READY|BLOCKED`。

预检阶段禁止修改赛道文件、调用冻结测试、跑长训练或先行提交。

## 10. Stage 1 实验结论

六个赛道都完成了 `observe → decide → validate → act → measure → stop` 的受约束执行闭环，说明“智能体能够实际执行优化任务”已经实现；但等预算消融没有发现任何赛道的 LLM 执行策略 `A2L` 对最终 promotion 指标具有可归因的正增益。执行能力成立，不等于预测增益成立。

| 赛道 | 实际执行与关键结果 | Stage 1 结论 | 保留的智能体职责 |
|---|---|---|---|
| ①断层识别 | A2D 完成数据门和证据动作；A2L 的 evidence F1 为 0；当前仍无合格连续 3D 正负例门 | 拒绝 A2L，`DATA_GATE_BLOCKED` | 保留确定性数据门、证据核验与停止决策；不进入预测调参 |
| ②地震相分类 | A2L、A2D、A3 等预算真实执行；三者 promotion mean mIoU 均为 `0.2967678500`，A2L 的 Penobscot guard 失败 | 拒绝 A2L | 数值调参交给确定性策略；LLM 最多提供高层路线建议，不进入预测路径 |
| ③储层物性 | A1 精确 replay，A4 确定性搜索有效执行；现有证据不足以支持 primary metric 上的 LLM 因果增益 | 拒绝直接 A2L | LLM 仅作路线/停止顾问；容量、正则和融合权重交给 A2D/A4 |
| ④岩相预测 | A2L 与 A2D 的部分轨迹不同，但最终 promotion macro-F1 均为 `0.2062263168`，低于 A0 `0.2133487970` | 拒绝 A2L；瓶颈为 inner-to-promotion transfer | 不保留预测型 LLM；优先修正内层选择向 promotion 的迁移问题 |
| ⑤甜点预测 | A2L 严格 fail-closed/STOPPED；完整 locked-env 测试通过，未产生优于确定性策略的 endpoint 证据 | 拒绝直接 A2L | LLM 只做目标可行性、路线和停止顾问；数值优化保持确定性 |
| ⑥三维重建 | fresh A0/A1 独立执行后数组逐位相等、哈希同为 `494618d4…9ecd`；A2L RMSE `0.0277598422`，差于 A0 `0.0277343744` | 保留冻结 A0，拒绝 A2L | LLM 仅作 P21/CIG-KED 路线确认与诊断顾问；核参数继续由嵌套 CV/BO 决定 |

因此没有赛道进入“直接 LLM 数值优化器”的 Stage 2，也没有设置 `A2L default_enabled=true`。后续若继续研究智能体，应该预注册新的信息优势或更合适的高层决策问题，而不是继续扩大相同动作表的调用次数。

## 11. 修正提交与验收

| 赛道 | 最终修正提交 | 验收状态 |
|---|---|---|
| ①断层识别 | `76c4464d01f6a4372fc1b48b75e4765f66c6b2ee` | 8/8 unittest；selection/promotion 不相交；工作树干净 |
| ②地震相分类 | `30cabbed7e8d48b9b5c2ab3727169b04b618b68d` | focused/adjacent tests 与 artifact verifier 通过；工作树干净 |
| ③储层物性 | `f6a4943fc1ddddcaeabdf10d0ce297750eec1400` | 6/6 pytest；P28 范围通过；工作树中保留了既有 P18/模型文件，未混入提交 |
| ④岩相预测 | `c4e238755bb3a7def7a6cd0075556898917f0d17` | 11 pytest + 3 subtests；artifact verifier 通过；工作树干净 |
| ⑤甜点预测 | `002afed9ce654c96e28e8d65ffb496e1f8d9a2a5` | locked env 5/5；路径与 manifest 扫描通过；工作树干净 |
| ⑥三维重建 | `0bb71fc27b84802fa8b5ac41939b35da2010e214` | locked env 12/12；fresh A0=A1 精确同一；测试不再污染 canonical 输出；工作树干净 |

Claude 最终窄复核确认⑥此前的 A1 容差同一性和测试污染两个问题均已修复，无 blocker/major；仅留下 `strict_owner=False` 属于测试便利性的 minor 观察，不影响本阶段结论。
