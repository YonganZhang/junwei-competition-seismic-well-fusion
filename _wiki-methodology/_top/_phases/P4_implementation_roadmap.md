# P4 实施路线图

> 状态：待用户启动 `/goal`  
> 前置：五窗口调研已验收；架构决策与 SOP 已冻结  
> 原则：先安全集成和公共合同，再赛道插件，再正式 CV/test；不得在脏 master 上直接叠加。

## 实施批次

### A. 安全集成审计

目标：把五条已有赛道的 baseline 与简单模型候选放进 clean integration worktree，保留用户主仓脏改动。

动作：

1. 记录主仓 `git status`、当前 HEAD 和各 track 的 `master..track` 完整 commit range。
2. 核对最新已验收分支头：
   - fault `db84205163e2954060cf6905fe88fe56ecbc0657`
   - facies `25edb4097fab36bd10663dda93791c606a510de0`
   - property `d35bbf0b4b85a1cc88149c5515d5f3322c959fb2`
   - lithofacies `86281c6c5002c2302e9b58bacfba169de59c3218`
   - reconstruction `7e97f33947dc329acfcddb4e5d2113af20334e21`
3. 不只 cherry-pick “最后一个 SHA”；必须根据祖先关系集成每个分支的完整已验收 commit range。
4. 在 clean integration worktree 逐条合入、逐条跑便携 contract tests；冲突只解决本批范围，不覆盖主仓用户改动。
5. 形成 integration report；未获明确授权不 merge 回 master、不 push。

退出门：五条赛道在一个 clean integration branch 共存，测试通过，模型/标签/split/指标未被静默改写。

### B. 公共训练合同

目标：实现 `_models/<track>/`、统一 envelope、seed、split manifest、artifact 和 test firewall。

建议模块：

```text
_code/ml_framework/
  contracts.py
  reproducibility.py
  split_manifest.py
  experiment.py
  reducers.py
  checkpointing.py
  hpo.py
  artifacts.py
  test_firewall.py
```

必须先修：

- sample/valid-label weighted reducer。
- checkpoint 包含 optimizer/scheduler/scaler/RNG/config hash。
- CV/HPO API 不接受 test loader。
- 模型目录迁移有兼容 shim，但只有一个代码真源。
- 每个公共模块有 unit/contract tests。

退出门：一个合成分类任务和一个合成回归任务能走通 `smoke -> cv -> freeze -> refit -> test -> visualize`，且人为泄漏用例会失败。

### C. 五赛道插件并行实施

| 窗口 | 责任 | 先做 | 后做 |
|---|---|---|---|
| fault | ①断层 | buffered splitter、logits/loss/metric、完整块 inference | 三视图/边界/连通/三维面、HPO |
| facies | ②地震相 | F3/Penobscot 独立折内预处理、CE baseline、逐类 metric | 稠密剖面、confidence/calibration、HPO |
| property | ③物性 + ⑤目标6/7公共 adapter | PHIF/KLOGH/SW 分离 mask、4-fold LOGO、raw/bounded 输出 | 逐井/空间/不确定性图、孔隙度/渗透率独立 baseline |
| lithofacies | ④岩相 | 4-fold family CV、固定9类/支持类双口径、概率输出 | 连续井深、校准、embedding、模态消融、HPO |
| reconstruction | ⑥重建 | strict/conditional 拆分、buffered block CV、Huber baseline | SSIM/频谱/变差/等值面消融和图件 |

每个窗口只写自己的赛道插件/模型/测试；共享框架只由指定公共 owner 修改，避免并行冲突。

退出门：五赛道都通过 unit、contract、tiny-overfit、real-data smoke；CV 尚可先用小预算，但 split manifest 必须真实。

### D. ⑤甜点七目标

按目标逐个建立 feasibility gate，不把七个目标绑成“一次全有标签才训练”：

1. `reservoir_quality`：冻结组合/排序定义与泄漏白名单。
2. `hydrocarbon_pay`：优先 CPI flag；阈值代理标 `proxy`。
3. `productivity`：冻结 30/90 天或 PI 等 horizon，井+时间因果划分。
4. `water_breakthrough`：定义事件、持续天数、删失与预测 cutoff。
5. `remaining_oil_infill`：先做 simulation case，固定 realization 与时刻；不称 field truth。
6. `porosity`：独立 PHIF/PHIE 回归。
7. `permeability`：独立 `log1p(KLOGH)` 回归与 raw-space 诊断。

每个目标交付：`task_spec`、数据/标签证据、split、简单 baseline、metric、图件或 `not_feasible.json`。

退出门：7 个 task ID 均有可审计状态；目标 6/7 至少完成真实 smoke 和独立测试路径；目标 1–5 不使用未来或标签构造字段泄漏。

### E. HPO、CV 与冻结测试

1. 先按赛道跑 8–12 sanity trials；修复失败/搜索空间问题。
2. 再跑 20–30 TPE pilot；昂贵模型可降低 pilot 预算但要记录。
3. top 3 配置 × 3 seeds × 全部有效 folds。
4. pooled OOF 固定阈值/校准；冻结 config hash。
5. development 全量 refit。
6. frozen test 单独 campaign，一次性生成指标与专属图。

退出门：每个完成任务都有逐 fold/seed/OOF/test 证据；任何折数降级有理由；test 未参与选择。

### F. 独立验收与交付

- 五个赛道窗口交叉验收，不由原作者只验自己。
- 负责人复核 split 零交叉、seed/config hash、模型动态发现、图件来源、物理/类别口径。
- 主仓 dirty 状态与用户改动保持不变；只有用户授权后才执行最终 merge/push。
- 更新 TOP、codemap、模型目录索引与最短运行说明。

## 工作量控制

第一轮不要求每个任务都跑昂贵完整神经网络。允许先用现有简单模型验证统一 SOP，但必须把接口、划分、指标、图件和 test 防火墙做真。深度模型替换发生在统一合同之上，不需要重写数据/评价管线。

优先级：

1. 防泄漏与独立 test。
2. 统一 seed/manifest/artifact/reducer。
3. 赛道专属 metric/visualization。
4. 有限预算 HPO 和多 seed。
5. 更复杂模型与结构损失。

## 风险与止损

- 独立 group 不足：降折，不拆组。
- 真实标签不足：`not_feasible`，不伪造。
- GPU/时间不足：缩小 HPO pilot，不削 test 隔离。
- 分支冲突：停止该批，保留 clean integration worktree，不在 master 强解。
- 结构 metric 实现不可靠：先保留体素/逐类主指标，结构项只作诊断。
- 旧 test 已被观察：明确标记 regression test；新增数据到位后再建立真正 blind test。

## 完成定义

- 六赛道共享统一模型目录和外层 I/O；内部任务 shape/头保持差异。
- 五已有赛道与⑤七目标均有明确 task status、真实测试路径和专属图件。
- 所有模型选择只使用 development CV；折数不足时诚实降级。
- seed、环境、数据、代码、split、config、checkpoint、预测、指标、图全部可追溯。
- 简单模型和未来深度模型可通过替换模型文件运行，不改公共数据/评价协议。
