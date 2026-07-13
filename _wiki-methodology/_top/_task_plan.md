# 军伟的比赛（地震+测井多模态融合识别有利油气目标） — 任务计划

> 创建: 2026-07-08
> **🧭 当前: [P4工程基线已实施并验收] 六赛道已在隔离 `p4-training-integration` worktree 共存；
>   公共训练合同、统一 seed/split/checkpoint/artifact/test firewall、规范 `_models/`、赛道插件与专属可视化入口已完成。
>   ⑤甜点已落地七个独立案例：1–4、6、7已有真实数据 baseline 与冻结测试，5诚实 `not_feasible`。
>   当前结果仅证明端到端训练/评价系统可用；多个简单模型精度很低，正式深度模型、长 HPO 与 top-3×3-seed 属 P5。
>   尚未 merge 回主仓、未 push，等待用户明确授权。**
> Done Criteria: Volve/F3-Demo/Penobscot 三批数据下载完整校验通过(已达成)；六赛道baseline pipeline候选
>   各自端到端验收通过(已达成，见下方P3/P4)；③④标签/目标定义已落地；
>   ⑤七目标均有可审计状态；主仓集成策略待授权；正式高性能模型尚待P5。
> 目标交付物: 模型 + 模型参数 + 数据预处理代码 + 模型运行代码（无文字报告要求）
> 距交付: TBD（官方未给最终截止时间，待军伟电话确认）

| Phase | 状态 | 进度 | 关键 finding |
|---|---|---|---|
| P0 问题界定 | ✅ | 1/1 | 赛题核心=地震+测井多模态融合预测有利油气目标，见下方赛题信息 |
| P1 方法论 / 方案 | 🔄 | 0/1 | pipeline 骨架已定（5阶段），具体算法待补 |
| P2 数据 / 实现准备 | ✅ | 3/3 | Volve全套(含ST10010)+F3+Penobscot下载完成；③④赛道数据量核实足够(纠正见P2.2) |
| P3 探索 / 实现 | ✅ | 6/6 | 六赛道baseline pipeline候选在各自隔离worktree端到端验收通过+可移植性收口完成(独立verify)；⑤分支1个已验收commit，①②③④⑥分支各2个已验收commit（第二个commit各含2个简单备选模型），尚未合并进主仓master，见P2.6与下方SHA登记 |
| P4 验证 | ✅ | 2/2 | 公共合同+五赛道插件+⑤七目标已在隔离集成分支实施；便携测试、真实 smoke、可行任务 CV/refit/frozen-test 与产物哈希已验收，证据见 `_tests/P4_acceptance_evidence.md` |
| P5 合成 / 交付 | 🔄 | 0/1 | 待授权 merge/push；简单模型只是工程 baseline，正式深模型、长 HPO、多 seed 复验与最终性能达标尚未完成 |

## 协作与决策权边界

- **军伟**：顶层设计决策者 + 技术路线（多模态算法/最新AI模型魔改，军伟自己负责，不外包）。
- **永安（本人）**：数据获取、pipeline 骨架梳理、文献调研协调。
- **孟洋师兄等其他人**：建议权，非决策权——军伟判断有用才采纳。
- **原则**：不闭门造车，尽量复用现成轮子（开源代码/预训练模型/已发表pipeline）。

## 赛题信息（官方，摘要不含具体建模方法）

### 核心任务
利用地震数据和测井数据做多模态融合，识别地下有利油气目标；重点证明"地震+测井融合"相比"仅地震"有指标提升。

### 输入
1. **地震数据**：三维叠后地震体/图像（SEG-Y / numpy / inline-crossline-time 切片等格式）
2. **测井数据**：LAS格式，常规九线（岩性：SP/GR/CAL；三孔隙度：AC/CNL/DEN；电阻率：MSFL/LLS/LLD）
3. **井位与井轨迹**：井口坐标、井轨迹、MD/TVD/TVDSS、补心海拔、地面海拔（单独文件）——**不含**时深表/校验炮/VSP/合成记录/速度模型
4. **专家标签**：断层/有利储层等目标标签，三维网格对齐优先，允许二维切片/井点/弱标签

### 可选辅助数据（仅可用于训练增强，推理阶段不可强依赖）
时深表、速度模型、合成记录、地层分层、岩性解释、储层参数解释、地震属性体

### 输出
断层概率图 / 有利储层概率图 / 有利油气目标概率体 或 二值分割结果（具体类型待与官方确认）

### 评估指标
- 分割类：Dice / IoU / Precision / Recall / F1
- 概率类：ROC-AUC / PR-AUC / Top-K命中率
- 断层专项：连续性 / 边界F1 / 骨架匹配
- **核心对比实验**：地震单模态 baseline vs 地震+测井融合模型，看指标是否提升——这是证明多模态融合有效性的关键，比赛评判重点

### 提交要求
模型 + 模型参数 + 数据预处理代码 + 模型运行代码；需在多个开源下游分割任务数据集上测试。**不要求文字报告**。

### 已知信息缺口
- 比赛主办方"勘探院"流程仓促，缺测试数据提交样例（军伟原话）
- 具体输出类型、最终截止时间待军伟电话确认后更新

## 当前门槛（P2 阶段）

- [x] Layer1公共预处理(`_pipelines/01_common_preprocess`)4个step已用真实Volve数据(ST0202地震体
      +06.LFP测井LAS+Official_Faults.dat+Horizons_TWT层位+Well_picks井分层点)跑通验证，
      2026-07-10。细节见该目录`_readme.md`与`_meta/_registry.yml`的`common_preprocess`条目。
- [ ] Volve 数据（~5TB全套，含2套3D地震+4D地震+测井+生产+钻井）下载完整校验通过（进行中，见 `data:volve-north-sea`；`GeoScience_OW_Archive.zip`此前截断827MB已修复）
- [ ] F3 Demo（荷兰北海，~8GB）下载 + 对应 Zenodo ML切片包（下载中，见 `data:f3-demo-netherlands`）
- [ ] Penobscot（加拿大Nova Scotia，~13GB）下载 + 对应 Zenodo ML切片包（下载中，见 `data:penobscot-canada`）
- [ ] 军伟周五前完成文献综述（1~5号pipeline阶段对应文献 + 其他领域可迁移方法文献包）
- [ ] 传统石油行业标准pipeline调研（军伟负责问询），确认能否补全"传统方法先后顺序"作为本项目pipeline骨架的参照
- [x] 六赛道数据可行度盘点（断层/相分类/储层物性/岩相/甜点/三维重建），Excel已发用户；③④两条经核实岩心样本量不足，
      需军伟决策是否降级为"仅校准用途"，见 `_findings/P2.1_volve_core_data_scale_gap.md`
- [x] 2026-07-11 **纠正上条结论**：之前只查了`06.LFP`/`09.CORE`(3口井)就判定"数据不够"，漏查了同一
      压缩包里`02.LWD_EWL`(24口井全覆盖原始测井)和`05.PETROPHYSICAL INTERPRETATION`(21口井CPI)。
      Volve历史钻井24口，测井曲线覆盖21-24口井，样本量足够支撑③④赛道，**不再需要军伟决策降级**，
      可直接开工；岩心数据(30-47样本)改作校准/精度验证用途。见 `_findings/P2.2_volve_well_log_scale_correction.md`
- [x] 2026-07-11 用户拍板：①②⑥三条赛道用git worktree隔离，各开一个Codex worker(leader mode由本Claude
      会话主线负责)。①⑥已用**简单baseline模型**(逻辑回归/岭回归，非深度学习)跑通完整pipeline验证数据管道，
      ②因F3/Penobscot标签空间不兼容(军伟决策：不强行统一，同一模型架构分开训练/测试)重新派活中。
      **🔴 当前baseline模型仅用于验证数据管道，不是最终模型**——用户已明确后续要换成深度学习/大模型，
      模型架构选型仍是军伟的决策权，worker不得擅自升级模型架构。见 `.claude/rules/leader-mode.md`
- [x] 2026-07-12 Volve/F3/Penobscot数据全貌整合为统一权威文档，取代逐个翻查P2.1~P2.4：
      见 `_wiki-methodology/_wiki/_entities/volve-dataset.md`（9大类→14zip映射、24井×数据源
      覆盖矩阵、井名命名陷阱、内容级程序化验证结果、③④⑤赛道数据可行性结论）。
- [x] 2026-07-13 Codex(直连Databricks远端字节级比对)+Claude Workflow(8路并行网络搜索)联合审查：
      **Volve/F3/Penobscot三批数据确认完整无遗漏**(与官方远端逐文件同名同字节数)；同时修正
      registry里4处文字漂移(总量7.7TB→实测4.566TB压缩/5.437TB标称、"9大类"→官方11类文件夹/
      本项目归并9组、F3标注类别数、Penobscot真实num_classes=8而非页面写的7)。产出①②③④⑥赛道
      均有权威公开代码仓库(15个repo均`git ls-remote`验证可访问)，**⑤甜点预测仅有论文级方法，
      无作者公开仓库，且Volve无现成甜点真值**——加固了⑤仍需军伟定义代理标签的既有结论。
      见 `_findings/P2.5_joint_data_and_algorithm_source_audit.md`、
      `_wiki-methodology/_wiki/_entities/algorithm-baselines-6tracks.md`。
- [x] 2026-07-13 军伟决策落地(通过tmux直接下发给③④worker)：③储层物性目标定为PHIF/log1p(KLOGH)/SW
      三输出+地震测井多模态+Hugin储层段+母井家族隔离；④岩相定为9类GENETIC FACIES(非36码LITH)+
      多模态+母井家族隔离；⑤仍未定标签，worker只交付了validate-only审计基础设施(未生成任何标签/
      数据集)。①-⑥六条赛道baseline pipeline全部完成并独立verify通过(独立重跑测试+dataset_io
      stats+哈希核验，非只信worker自述)。
- [x] 2026-07-13 六赛道worktree只读合并就绪审计：**全部MERGE_READY=NO**，核心共性阻塞是6个track
      分支目前都是0 commits(HEAD与master相同)，直接`git merge`会合入零内容；次要问题是集成测试
      无条件依赖被gitignore的产物(需skip guard)、各赛道独立.venv体积6.5-6.9GB级别、部分JSON残留
      worktree绝对路径。见 `_findings/P2.6_six_tracks_merge_readiness_audit.md`。
- [x] 2026-07-13 六赛道可移植性修复：按P2.6审计的REQUIRED_PRE_MERGE_ACTIONS逐条修复，范围严格限定
      README/测试门控/路径序列化/.gitignore(禁止改模型/标签/split/指标)。全部独立verify通过：
      ①fault候选收敛21文件/378KB；②facies候选从6.9GB(.venv)压到521KB/23文件；③reservoir新增
      `--run-integration`显式门控(默认7 passed 2 skipped，完整9 passed)；④lithofacies同样门控
      模式(默认5 passed 6 skipped，integration 11/11)；⑤sweetspot清理绝对路径+扩充.gitignore，
      标签仍是draft未approve；⑥reconstruction候选从~394MB(含_tmp 392MB)压到34文件/1.07MB。
      所有指标/split/模型值均确认未变。**下一步（未执行，需先确认方向）**：各worktree按各自
      INCLUDE清单做选择性`git add`+commit，再评估合并进主仓策略。
- [x] 2026-07-13 六赛道各自完成1个已验收commit(按各自INCLUDE清单选择性提交，非全量`git add -A`)，
      仍未合并进主仓master(各分支ahead=1/behind=0)。真实SHA：
      ①fault `09abe70905f988f4585debbd2216a6a2542e5dfd`；②facies `5f70a33e23227f45bb401d077630ff594d477b14`；
      ③reservoir `ea3b0f35c0dd05173b38be83834f87b82315f875`；④lithofacies `84839c2421143804e7c923c30500cf03eefc7078`；
      ⑤sweetspot `b13d876349c55834abc36550864d6b6f19bda9cd`；⑥reconstruction `1a142f3787b18999123a9750fcd8bea30001d822`。
      **下一步（未执行，需先确认方向）**：待军伟/负责人拍板合并策略后再执行merge进master。
- [x] 2026-07-13 按军伟“先做数个简单模型”的指令，①②③④⑥各新增2个同名文件动态注册的备选模型，
      未改共享`_code/ml_framework`、数据、标签、split、既有baseline或正式指标。新增commit：
      ①fault `db84205163e2954060cf6905fe88fe56ecbc0657`（raw logistic/local Huber，独立复测15/15）；
      ②facies `25edb4097fab36bd10663dda93791c606a510de0`（pixel linear/tiny FCN，独立复测9/9）；
      ③reservoir `d35bbf0b4b85a1cc88149c5515d5f3322c959fb2`（linear/ridge，独立复测7 passed、2 integration skipped）；
      ④lithofacies `86281c6c5002c2302e9b58bacfba169de59c3218`（concat linear/late fusion，独立复测5 passed、6 integration skipped）；
      ⑥reconstruction `7e97f33947dc329acfcddb4e5d2113af20334e21`（linear SGD/tiny MLP，独立复测14/14）。
      每个新模型均位于本赛道`models/<registered_name>.py`并导出`build_model()`，可由既有`get_model`动态发现，
      不需改训练主循环。**边界**：目前固定的是“各赛道内部”的模型输入输出适配；五条赛道的张量/方法签名仍因任务类型
      （分割、分类、回归、重建）不同而异，尚没有跨赛道统一`ModelBatch -> ModelOutput`合同，不能误报为全项目统一I/O已完成。
      **下一步（未执行）**：待军伟确认⑤目标语义，并另行决定是否设计跨赛道统一I/O合同与五个分支的merge顺序。
- [x] 2026-07-13 军伟进一步拍板：⑤甜点采用“一赛道七任务”，在原拟五个目标外独立增加⑥孔隙度、⑦渗透率；
      同时要求六赛道补齐独立测试、合理train/validation/test划分、训练+验证内5-fold、全局随机种子、自动调参、
      loss/输出激活比较、赛道专属可视化及分层自测。当前先执行五窗口**只读调研批**，不改代码、不跑长训练；
      五窗口已通过 Secretary Bus 完成并由负责人逐一verify，leader `junwei-p4-training-research-20260713` 已关闭为completed。
      研究合同见 `_phases/P4_training_validation_system_research.md`，合并调研见
      `_external_reviews/P4_five_track_training_research_20260713.md`，七目标合同见
      `_phases/P4_sweetspot_seven_target_contract.md`，架构决策见
      `_decisions/P4.1_training_validation_reproducibility_architecture.md`，统一SOP见
      `../_wiki/_methods/training-evaluation-reproducibility-contract.md`，实施路线见
      `_phases/P4_implementation_roadmap.md`，最终可复制 `/goal` 见 `_phases/P4_goal_prompt.md`。本轮只做调研与文档，未实施模型/CV/HPO代码。
- [x] 2026-07-13 P4 Goal 已在隔离 `p4-training-integration` 分支实施：安全集成五赛道候选，
      完成 `_code/ml_framework` 公共合同、`_models/<track>/` 唯一模型真源、全局 seed=2693、
      group/spatial split、加权 reducer、checkpoint/resume、HPO 接口、artifact manifest 与一次性冻结测试门。
      ⑤七目标中 1–4、6、7 已训练并消费冻结测试，5 因缺 Eclipse 状态解析器与冻结时点/候选井/经济约束而
      `not_feasible`；精确 PHIE 也独立 `not_feasible`，没有用 LFP_PHIE 偷换。断层真实数据有 3998 个正例点，
      但缺覆盖已审核负例，因此正式 blind/CV 也诚实 `not_feasible`。可行赛道已补齐真实 smoke；
      F3/Penobscot、岩相、strict/conditional 重建已走通 CV→freeze→refit→single-use test→archived visualization。
      验收证据见 `_wiki-methodology/_tests/P4_acceptance_evidence.md`。**未 merge 回 master，未 push**。

## 数据资产索引

详见 `_meta/_data_registry.yml`（单一真源，本文件不重复数据字段细节）。

## Pipeline 骨架

详见 `_pipelines/_readme.md`（元思维层：5阶段骨架 + 接口关系；不含具体算法/模型选型——由军伟后续用最新AI论文模型替换填充）。

## 决策入口

- 关键决策放 `_decisions/`；这里只留当前阶段必须看的链接。

## 证据入口

- 结论 / 类层模式放 `_findings/`。
- 操作过程放 `_logs/`。
- 外援意见放 `_external_reviews/`。
