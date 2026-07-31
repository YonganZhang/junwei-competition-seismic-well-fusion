# P4 最终实施 `/goal`

> 状态：可复制使用  
> 生成日期：2026-07-13  
> 用法：在项目根目录的新会话中完整复制下方代码块。该 Goal 授权在隔离分支/worktree 内实施和验证，但不授权覆盖主仓脏改动、merge 回 master、push 或发布。

```text
/goal 在项目 /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5 中，完整实施 P4 统一训练、验证、自动调参、复现、独立测试与赛道专属可视化体系，并把⑤甜点赛道实现为七个独立任务（1储层品质、2含油气/有效厚度、3产能、4见水风险、5剩余油/加密井潜力、6孔隙度、7渗透率）。

开始前必须完整读取并以其为真源：
1. AGENTS.md 和其要求的全局 rules；
2. _wiki-methodology/_top/_decisions/P4.1_training_validation_reproducibility_architecture.md；
3. _wiki-methodology/_wiki/_methods/training-evaluation-reproducibility-contract.md；
4. _wiki-methodology/_top/_phases/P4_implementation_roadmap.md；
5. _wiki-methodology/_top/_phases/P4_sweetspot_seven_target_contract.md；
6. _wiki-methodology/_top/_phases/P4_goal_acceptance_draft.md；
7. _wiki-methodology/_top/_external_reviews/P4_five_track_training_research_20260713.md；
8. _wiki-methodology/_top/_task_plan.md 及相关数据/算法实体文档。

使用 share-agent-bus / Secretary Bus 建一个负责人 leader，调度并持续监控以下五个现有窗口：
- volve-worker-fault：①断层；
- volve-worker-facies：②地震相；
- volve-worker-property：③储层物性，并负责⑤目标6孔隙度/7渗透率的共享物性 adapter；
- volve-worker-lithofacies：④岩相；
- volve-worker-reconstruction：⑥三维重建。

负责人负责公共框架、⑤目标1–5的操作定义收敛、集成、交叉验收和 TOP 更新。若公共实现需要独立写 owner，必须明确 write ownership；五个窗口不得并发修改同一共享文件。窗口先交研究/实现计划和 write set，负责人确认后再写。

按下列批次推进，不得跳过验收门：

A. 安全集成
- 先记录主仓 dirty 状态和用户改动；不要清理、reset、checkout 或覆盖它们。
- 建 clean integration worktree/branch。
- 审计 master..track 的完整 commit range，不得只 cherry-pick 最后一个 SHA。最新已验收 heads：fault db84205163e2954060cf6905fe88fe56ecbc0657；facies 25edb4097fab36bd10663dda93791c606a510de0；property d35bbf0b4b85a1cc88149c5515d5f3322c959fb2；lithofacies 86281c6c5002c2302e9b58bacfba169de59c3218；reconstruction 7e97f33947dc329acfcddb4e5d2113af20334e21。
- 把各分支完整已验收提交安全集成到隔离分支并逐条复测。未获用户明确授权，不 merge 回 master、不 push。

B. 公共合同
- 建规范模型目录 _models/<track>/<model_id>.py；每个模型动态发现并导出 build_model(task_spec, **config) 与 capabilities；可选 suggest_hparams。旧赛道 models/ 只留兼容 shim，禁止双真源。
- 实现 TaskSpec、ModelBatch、ModelOutput 外层 envelope；不同任务内部张量/shape/head 保持差异。
- 实现统一 root_seed=2693（可覆盖）及 split/cv/model/loader/sampler/augmentation/hpo/diagnostic seed 派生；覆盖 Python、NumPy、Torch CPU/CUDA、DataLoader worker/generator，并输出 determinism report。
- 实现 split/fold manifest、test firewall、sample/valid-label weighted reducer、完整 checkpoint/resume、artifact envelope、OOF/refit/frozen-test 生命周期。
- CV/HPO API 在结构上不得接收 test loader；visualize 只读已归档 prediction/metric。
- Optuna 是可选默认 HPO 后端；无 Optuna 的固定 baseline 必须能运行。当前不整体迁移 Lightning/Ray。

C. 五赛道插件
- 断层：完整连续空间块 test；buffered spatial CV；raw logits；比较 BCEWithLogits、BCE+Dice、Focal、Tversky；AP/PR、Dice/IoU、boundary/连通；完整块滑窗、三视图、TP/FP/FN、PR阈值、三维断层面。
- 地震相：F3/Penobscot 分开 schema/训练/测试；每折仅用 fold-train 重拟合归一化和类别权重；比较 CE、Focal、CE+Dice/Lovasz；稠密剖面、逐类指标、混淆、confidence/entropy/calibration。
- 储层物性：PHIF、log1p(KLOGH)、SW 拆分独立 label mask；当前仅4个非test母井家族，诚实做4-fold LOGO；比较 MSE/Huber/MAE和可选概率回归；identity与物理约束输出都报告 raw/约束指标和越界率；逐井深度、散点、残差、空间与不确定性图。
- 岩相：GM09固定九类；F-5 frozen test；当前4-fold leave-one-family-out；每折重拟合预处理；固定9类和 observed-support 双口径；CE/Focal/class-balanced候选；井深轨、逐类PR/F1、混淆、校准、embedding、模态缺失/消融。
- 三维重建：conditional 与 strict 分开；strict frozen test；buffered K/I spatial CV，每折过滤井约束；Huber默认，MSE/MAE对照，SSIM/频谱/变差/几何项只作二阶段消融；三视图、差值体、CDF、频谱、变差、等值面、距井误差。

D. ⑤甜点七目标
- 七个目标必须各自拥有 task_id、标签/输入白名单、label mask、split、baseline、metric、checkpoint、图件和状态；目标6孔隙度、7渗透率不得只藏在目标1综合评分中。
- 目标1冻结组合或排序定义并做权重敏感性；目标2优先用CPI pay/reservoir flag，阈值代理必须标proxy；目标3冻结30/90天或PI等horizon并按井+因果时间切分；目标4定义稳定见水事件、持续天数、删失和预测cutoff；目标5先作为simulation case固定realization与预测/评价时刻，不冒充field truth；目标6用PHIF/PHIE独立回归；目标7用log1p(KLOGH)独立回归并报告原空间。
- 构造标签或未来结果的字段不得进入同任务推理输入。数据不足时生成可审计 not_feasible.json，不能伪造标签或假指标凑七项。

E. 训练、HPO与正式验证
- 每个任务依次通过 unit、contract、tiny-overfit、real-data smoke、development CV、config freeze、development refit、frozen-test、reproduce。
- 默认争取5-fold，但 effective_n_splits 必须受独立group/有效类别/正例块约束；不足时写明降级理由，禁止拆同井、随机patch或取消空间buffer凑五折。
- 所有预处理、类别/正例权重、采样、target transform、阈值和校准都必须逐折只看fold-train/OOF。
- HPO先8–12 sanity/random trials，再20–30单进程TPE pilot；小数据/噪声大先NopPruner，有证据后才保守MedianPruner。primary score之外保存fold std、worst fold、guardrail和成本。
- top 3配置至少3个预注册seed复验。最终epoch规则由CV best-epoch分布预先冻结；test只跑一次，不参与任何选择。

F. 独立验收与文档
- 每个窗口完成后先collect，再由负责人按测试、split manifest、hash、指标、图件和clean worktree独立verify；失败要具体修复，不接受只看窗口自述。
- 五个窗口交叉审查，原作者不能成为唯一验收人。
- 运行 git diff --check、便携测试、真实数据smoke与必要CV；长训练/HPO要有预算、超时、产物目录和中断恢复。
- 更新 TOP task plan、决策、method、codemap/模型索引、最短运行说明和最终证据矩阵。
- 保留用户主仓脏改动。未获用户明确授权，不执行最终merge到master、push、发布或删除worktree。

硬性拒收：用test调参；同井/相邻空间/未来事件泄漏；只有训练损失；按batch mean等权聚合；静默clip；伪造sweetspot标签；为了五折拆group；所有任务强用同一loss/激活/图；图无法追溯checkpoint/config/prediction；绝对路径或未登记模型文件；覆盖用户已有改动。

完成标准：六赛道共享统一可替换模型目录与外层I/O，五条已有赛道和⑤七目标都有可审计状态、合理CV/真实测试路径、全局复现合同、自动调参入口、赛道专属图件和分层自测；简单模型能跑通全SOP，未来替换深度模型时不改数据/评价协议。持续推进直至全部验收或遇到确需用户新增授权/数据的真实阻塞；若阻塞，提交证据化not_feasible/blocked清单，不用假结果绕过。
```

## 启动后首先应看到的证据

1. leader ID、五窗口 task map、write ownership、验收标准与 deadline。
2. 主仓 dirty snapshot 和 clean integration worktree 路径。
3. 五个 track 的完整 commit range，而非五个孤立 SHA。
4. 公共合同文件清单和避免共享文件并发冲突的 owner。
5. 每个赛道/甜点目标的 `requested_n_splits`、可用 group 数与预期 `effective_n_splits`。
