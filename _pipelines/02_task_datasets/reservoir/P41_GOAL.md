# P41 Goal：物性井震双基础模型条件增量资格门

## 目标

在③储层物性预测赛道上，用一次低成本、可归因、严格多井隔离的开发集实验回答：测井基础模型与地震基础模型的冻结特征，是否能够解释当前强基线的剩余误差。只有资格门通过，后续才允许进入 LoRA/Adapter 与分阶段解冻。

本 Goal 不以“做出一个融合网络”为验收，而以“相对同一强基线获得可重复、配对特异的增益”为验收。若没有信号，必须停机并指出失败来自数据、配对、表征还是学习器，不得降低基线、替换指标或读取冻结测试集来制造提升。

## 写域与 Git

- 独立 worktree：当前 `p41-property-crossmodal-qualification`。
- 仅修改 `_pipelines/02_task_datasets/reservoir/**`，以及一份 `_wiki-methodology/_top/_findings/P41_*.md`（如确有必要）。
- 不触碰当前 `track-property` 工作树中的未提交文件，不 merge、不 push。
- 最终一次干净提交；报告 commit、测试、产物路径和 worktree 状态。

## Phase 0：先冻结事实

1. 核对并哈希当前开发数据、井族划分、样本数、目标 `PHIF / log1p(KLOGH) / SW`、测井窗口、地震窗口、深度—时间—inline/xline 对齐字段。
2. 只允许使用现有 development families；`test.h5`、known/frozen holdout 及其标签严禁打开、枚举或用于归一化、PCA、早停、选择和阈值。
3. 从现有开发证据中选定当前可重放的最强确定性基线。必须在同一解释器、同一 split、同一 seed 池重放；不能用更弱的临时模型作为 B0。当前强基线真源是 P5 Stage-3 LOGO4×3seed：`PHIF/KLOGH=extra_trees_regressor`、`SW=xgboost_regressor`，其 153 维输入为 81 维地震 patch、36 维测井值和 36 维 mask。实现须记录 P5 runner、budget、split 与 leaderboard 哈希，并在 P41 分折上重新拟合；严禁使用 P29 TinyMLP/ridge 降级基线。若无法合法重放 P5 强基线，状态必须是 `BLOCKED_STRONG_BASELINE_REPLAY`，不能继续融合。
4. 输出 `phase0_freeze.json`、`aligned_pair_manifest.jsonl`、split/hash/asset audit；明确每个样本的 well/family/MD/TVDSS/TWT/inline/xline 与配对质量。
5. 若真正可用的独立开发井族少于 4，或有效配对样本不足以做外层井族留出，状态必须是 `BLOCKED_DATA_GATE`，不要伪造交叉验证。

## R0：便宜的条件增量探针

采用 outer LOGO（井族留出）+ outer-train 内层 LOGO；固定种子至少 `2693`，计算可承受时用 `2693/2694/2695`。所有归一化、PCA/adapter、目标标准化和超参数选择仅在当前 train families 拟合。

必须包含且共用完全相同 B0 预测的变体：

- `B0`：冻结强基线。
- `W1`：仅测井基础模型特征的零初始化 gated residual。
- `S1`：仅地震基础模型特征的零初始化 gated residual。
- `F1`：测井+地震双分支 gated residual，输出锚定在 B0；gate=0 时逐元素等于 B0。
- `A5`：同架构、双随机初始化 foundation control；仍锚定同一 B0。
- `A6`：训练集内、同井族内打乱井震配对；验证标签与 B0 不动。
- `A7`：fusion-off，必须逐元素、逐样本、逐目标等于 B0。
- `A8`：时间/深度小幅错位（仅训练输入或固定干预，绝不能根据验证结果调偏移），用于验证正确对齐是否必要。

优先复用已经锁定的本地 MOMENT/GFM（或项目中已验证的测井/地震 foundation）资产，保留原生 token 后在 outer-train 内做确定性降维/小头；不得沿用 P38 的 768/1200→16 随机投影。若③的 4×9 测井窗口不符合 foundation 输入，允许使用标签无关、写入审计的 pad/resample+mask；不允许偷偷换成普通 MLP 后仍称为基础模型。

训练时记录：每层输入/输出张量维度、参数量/可训练参数量、首末 loss、gate/梯度范数、特征方差、预训练与随机特征最大差、残差贡献幅度。固定训练预算，不用外层 held 做 early stopping。

## 主指标与晋级门

主指标沿用现有赛道定义：三目标在 outer-train 标准差归一化空间中的等权复合 RMSE，越低越好；同时报告每目标 RMSE/MAE/R²/Pearson、最差井族及 bootstrap 置信区间。

只有以下条件同时成立才标记 `R0_SIGNAL_PASS_NEXT_STAGE_ADVICE_ONLY`：

1. `F1` 相对同一 `B0` 的复合 RMSE 至少降低 0.5%；
2. 至少 3/4 外层井族胜出，并且三个目标中至少两个不恶化；
3. `F1` 比最佳单模态 `W1/S1` 再降低至少 0.3%；
4. `F1` 比 `A5`、`A6`、`A8` 各至少好 0.5%；
5. 配对 bootstrap 的改善区间不跨 0；
6. `A7 == B0` 逐元素精确成立，且所有泄漏/哈希/对齐门通过。

任一项不满足则 `R0_STOP_NO_ATTRIBUTABLE_SIGNAL`，本 Goal 立即停止，不做 LoRA、不看测试集。不要为了过门调低阈值。

## 产物与验证

输出到 `_outputs/p41_property_crossmodal_qualification/`：

- `summary.json`、`results.jsonl`、`predictions.npz`
- `phase0_freeze.json`、`aligned_pair_manifest.jsonl`
- `foundation_audit.json`、`feature_or_adapter_audit.json`
- `artifact_manifest.json`、`rerun_commands.json`、`evidence.md`

提供 `--verify-only`，校验 schema、split/hash、资产、禁用 holdout、变体完整、B0 共用、A7 精确回退和产物哈希。补覆盖数据门、外层拟合隔离、同井族 shuffle、gate=0、指标方向与晋级逻辑的测试。独立重算主指标后再提交。
