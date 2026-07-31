# P5 ④岩相：测井序列岩性/岩相分类与未见井泛化直接基准调研

> 调研日期：2026-07-14（Asia/Hong_Kong）
> 调研边界：只纳入“以沿深度测井为输入、以岩性/岩相为目标、并以整井或井组留出检验未见井”的工作。普通时间序列分类、普通表格多分类、纯地震相分类和纯岩芯图像分类不计为直接基准。
> 本地基线：分支 `p5-stage3-lithofacies`，调研开始时 HEAD `29600b33884092b92ba6043bca418bb26c41bb62`，`git status --short --untracked-files=all` 无输出。
> 执行限制：本次没有训练、重建数据、读取新的 F-5 指标、安装依赖或下载权重；仅生成本报告。

## 1. 结论先行

1. **GM09 固定九类必须保留。** 文献中的 SEG 九类只是类别数恰好相同，语义是砂岩—粉砂岩—碳酸盐岩组合，不能映射或合并为 GM09 的九类沉积相。固定 schema 的 Macro-F1 作为主排名，比按每折“碰巧出现的类”计算 supported-class Macro-F1 更能约束稀有类退化。
2. **母井家族隔离是正确且强于常见基准的做法。** 直接基准从 Hall 的单盲井、SEG/FORCE 的秘密井，发展到按井 Leave-One-Well-Out（LOOW）和 fivefold well-group CV。将母井和侧钻视为一个 family，再在窗口化和归一化前划分，是对“按井划分”的必要强化。
3. **F-5 只能称为已见过的可复用 holdout，不能再称 fresh blind。** 当前 Stage-4 已明确记录 `prior_test_consumed=true`、`fresh_blind=false`。今后的模型选择只能依赖 development LOGO4；F-5 可保留为历史确认结果，不能反复打开追分。
4. **75% 完成率不排名是合理的 fail-closed 工程规则，但不是某篇岩相论文规定的行业阈值。** 当前实际阈值是 80%，且要求四折、三 seed 都覆盖；因此 9/12=75% 的 CatBoost 不排名。建议保留并在协议中明确“预注册覆盖门槛”，不要把它宣传为外部标准。
5. **P/S lane 分离符合直接基准。** 中心窗口输出一个中心标签属于 P（sequence-to-point/context-window）；连续井段输出逐深度标签或整井 HMM 解码属于 S（sequence labeling）。当前归档没有有限 `center_md_m`，所以 S lane `not_rankable` 是正确结论；不得用行号、区间中点或重复中心样本伪造序列。
6. **当前 P4 合同存在一处真实漂移。** `p4_contract.py` 仍把 `supported_class_macro_f1` 写成 TaskSpec/HPO 主指标，而 Stage2–4 已按 `fixed_schema_macro_f1` 排名。下一次允许改代码时应统一；本报告不修改它。
7. **现任务是测井+井旁地震多模态，而直接公开基准几乎都是测井单模态。** 正式结果应保留当前多模态输入，但必须另报同 split、同预算的 well-log-only 消融，且不得把两个输入条件混在一个榜中。否则无法与 SEG/FORCE/LOOW 文献做方法层面的公平比较。

## 2. 纳入标准、证据等级和记号

### 2.1 直接性

- **D1（强直接）**：沿真实井深的测井曲线预测显式岩性/岩相；测试单位是从未参与拟合的整井或井组。
- **D2（有条件直接）**：任务和井留出直接，但测试井无标签特征被半监督/转导方法消费，或工作同时包含岩芯图像而测井只是其中一支。可借鉴，不能与纯归纳 holdout 等价。
- **不纳入**：只做随机深度点拆分而没有独立盲井；普通时间序列/表格 benchmark；纯地震相；只用岩芯图像且没有测井支路。

### 2.2 来源等级

- **P1**：原始论文、官方数据/竞赛页、作者或官方组织 GitHub。
- **P2**：机构论文索引、作者公开手稿，用于补足 P1 页面未展开的实现细节。
- **NR**：已查主源但没有报告；不按常识补写。
- **代码许可**与**论文/数据许可**分开记录；论文 CC BY 不自动给 GitHub 代码授权。

本报告的 10 项优先工作均有 P1；其中 9 项可直接访问原始论文或官方竞赛/数据页，7 项有可核对代码。GitHub revision 均在 2026-07-14 通过默认分支 ref 实扫，不以浮动 `main/master` 代替。

## 3. 十项优先直接工作：身份、直接性、来源与许可

| ID | 工作与直接性 | 一手论文/官方页 | 代码与精确 revision | 许可结论 |
|---|---|---|---|---|
| B1 | **Hall (2016) 教程 + Hall & Hall (2017) SEG Facies Classification Challenge**，D1。教程有 SHANKLE 整井，正式竞赛有 STUART/CRAWFORD 两口秘密井。 | [教程 DOI](https://doi.org/10.1190/tle35100906.1)、[SEG Wiki](https://wiki.seg.org/wiki/Facies_classification_using_machine_learning)、[结果论文 DOI](https://doi.org/10.1190/tle36030267.1) | [SEG 官方仓库 @ `160430c8`](https://github.com/seg/2016-ml-contest/tree/160430c84659785c7cfc74fd380ffa05c387b8cb) | GitHub 元数据为 Apache-2.0，但仓库 README 明说各参赛代码由作者自行决定条款；教程/竞赛材料 CC BY；当时数据明确“not openly licensed”。复用具体队伍代码前需逐目录核权。 |
| B2 | **Bestagini, Lipari & Tubaro (2017)**，D1。测井梯度/交互特征 + boosting，在按井留出开发切分和 SEG 双盲井上验证。 | [原始论文 DOI](https://doi.org/10.1190/segam2017-17729805.1)、[机构记录](https://ricerca.ogs.it/handle/20.500.14083/25096) | [官方竞赛仓库 ISPL @ `160430c8`](https://github.com/seg/2016-ml-contest/tree/160430c84659785c7cfc74fd380ffa05c387b8cb/ispl) | 仓库根显示 Apache-2.0，但 README 对参赛者代码保留“author terms”；ISPL 子目录无独立 LICENSE，故代码许可记为**有歧义，需作者确认**。 |
| B3 | **Tschannen et al. (2017) Inception ConvNet**，D1。真实短深度窗口输入、中心点分类，并在两口竞赛盲井评价。 | [arXiv 原始论文](https://arxiv.org/abs/1706.00613) | [作者仓库 @ `dbc85f0`](https://github.com/vts21/2016-ml-contest/tree/dbc85f03750307384a7d0740de9cebaca7d3c676/itwm) | 作者仓库 SPDX Apache-2.0；底层 SEG 数据沿用 B1 的限制。 |
| B4 | **Feng (2020) ANN–HMM**，D1。点级 ANN emission 与整井 HMM/Viterbi 序列解码结合，在两口未训练盲井验证。 | [OUP 原始全文/DOI](https://academic.oup.com/gji/article/221/3/1484/5807722) | 未发现作者发布的可运行代码 | 论文按出版社条款免费阅读；无代码、无代码许可。 |
| B5 | **Dunham, Malcolm & Welford (2020) label propagation + self-training**，D2。仅一口井有标签，预测其余九口井；但无标签井特征参与转导训练。 | [原始论文 DOI](https://doi.org/10.1190/geo2019-0238.1)、[作者单位出版目录](https://www.esd.mun.ca/~kwelford/publications.html) | 未发现作者官方代码仓库 | SEG 论文版权；无公开代码许可。 |
| B6 | **FORCE 2020 Lithology Competition 官方基准**，D1。98 口发布训练井、10 口公开测试井、10 口最终盲井；最终盲井决定排名。 | [官方 Zenodo 数据 DOI](https://doi.org/10.5281/zenodo.4351156)、[挪威海洋管理局结果页](https://www.sodir.no/en/force/Previous-events/2020/results-of-the-FORCE-2020-lithology-competition/) | [官方赛后代码归档 @ `c8d01ee`](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition/tree/c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1)、[Equinor 参赛仓库 @ `649c9f9`](https://github.com/equinor/force-ml-2020-wells/tree/649c9f9762b1d203a7fca1a6a4dc9d1a5bb0687c) | 测井 NLOD 2.0，标签 CC BY 4.0；两个代码仓库均无根 LICENSE（GitHub `NOASSERTION`），不能因“官方开放”推断代码许可。 |
| B7 | **Olawale Ibrahim FORCE 2020 冠军方案**，D1。XGBoost + 邻域/梯度特征，在整井验证集和最终 10 口盲井上验证。 | [作者技术复盘](https://ibrahim-olawale13.medium.com/force-2020-machine-learning-lithology-predictionwinning-solution-8cbf78290b41)、[FORCE 官方结果](https://www.sodir.no/en/force/Previous-events/2020/results-of-the-FORCE-2020-lithology-competition/) | [作者仓库 @ `b55438d`](https://github.com/olawaleibrahim/2020_FORCE_Lithology_Prediction/tree/b55438d16eec36db04aca67294fbdaff8d27523a) | 代码 Apache-2.0；数据许可同 B6。 |
| B8 | **Martin, Meyer & Jobe (2021) Q204 cored wells**，D2。测井 XGBoost 支路按井留出；另有 CCL/岩芯图像序列支路，后者不能冒充测井序列基准。 | [Frontiers 原始全文/DOI](https://doi.org/10.3389/feart.2021.659611) | [作者仓库 @ `d638a00`](https://github.com/rgmyr/coremdlr/tree/d638a00339bda75afa8811fe004fafa38f285e71) | 论文 CC BY；仓库无 LICENSE，代码法律状态为未授权复用。 |
| B9 | **Nguyen, Nguyen & Mai (2025) attention-residual network**，D1。八次宣称 LOOW；模型对同一深度的异构特征建模，不是沿深度 Transformer。 | [PeerJ 原始论文 DOI](https://doi.org/10.7717/peerj-cs.2977)、[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12453817/) | [数据/代码仓库 @ `da775f5`](https://github.com/mardani72/Facies-Classification-Machine-Learning/tree/da775f5b3f5b23991292c6324b5193fcb4737c14)；论文另附小型 supplement | 论文 CC BY 4.0；GitHub 仓库无 LICENSE，不能把论文许可外推到代码。 |
| B10 | **Carvalho et al. (2026) well-log lithology benchmark**，D1。FORCE/Geolink，按井 fivefold CV，同时评估点级和逐深度序列输出。 | [Springer 原始全文/DOI](https://doi.org/10.1007/s11004-026-10300-1) | [作者仓库 @ `11d9e34`](https://github.com/uai-ufmg/well-log-lithology-classification/tree/11d9e3460165ae2767f64b58d7ee9dcf07372230) | 论文 CC BY 4.0；仓库无 LICENSE，代码不可默认再分发。 |

## 4. 证据矩阵 A：曲线、类别 schema 与窗口/序列构造

| ID | 曲线/输入 | 类别 schema | 窗口或序列构造 | 任务类型 |
|---|---|---|---|---|
| B1 | GR、`ILD_log10`、PE、DeltaPHI、PHIND；另有 NM_M、RELPOS 两个地质约束；半英尺深度采样。 | SEG 九类：SS、CSiS、FSiS、SiSh、MS、WS、D、PS、BS。 | Hall SVM 是单深度点；正式参赛者可自行做邻域特征/平滑。 | 点级分类；官方竞赛不要求统一序列模型。 |
| B2 | 同 B1 的 5 条测井 + 2 个约束。PE 在部分井缺失。 | 同 SEG 九类。 | 先按井深排序，计算相邻深度梯度，再做二阶多项式/交互特征；验证代码不是端到端序列网络。 | 上下文增强的 sequence-to-point（P）。 |
| B3 | 同 B1。论文写 5 条 wireline + 2 约束；代码对缺失 PE 做回归插补。 | 同 SEG 九类。 | 每井构造 31 点（15.5 ft）窗口、边界零填充、重叠 30%，窗口输出中心岩相；Inception 1D CNN。 | 真正的短窗口 sequence-to-point（P），不是逐井序列标注。 |
| B4 | GR、RE、PE、DiffPHI、AvgPHI 五个输入特征。 | 同 SEG 九类。 | ANN 独立输出每点 softmax/emission；HMM 使用由训练井扫描得到的转移矩阵，Viterbi 沿整口井向上解码。 | 整井序列解码（S）。 |
| B5 | 同 SEG 的 5 条 wireline + NM_M、RELPOS。 | 同 SEG 九类，并利用 NM_M 将 marine/non-marine 流形分开。 | 对样本特征图做 label propagation/self-training；没有沿 MD 的卷积/RNN 序列。 | 转导式点级分类，不是 S lane。 |
| B6 | WELL、MD、XYZ、CALI、RDEP、RHOB、DRHO、SGR、GR、RMED、RMIC、NPHI、SP 等，最多约 20 个物理/质量曲线；不同井缺失模式不同。 | 12 类：sandstone、sandstone/shale、shale、marl、dolomite、limestone、chalk、halite、anhydrite、tuff、coal、basement。 | 官方竞赛没有统一窗口；参赛代码多为点级树模型和手工邻域特征。 | 固定整井盲测的点级 benchmark。 |
| B7 | B6 的多曲线 + GROUP、FORMATION、WELL 编码；删除公开测试中全缺或低覆盖曲线。 | FORCE 12 类。 | 复制 Bestagini 风格的上下邻点与梯度增强；XGBoost 对增强后的中心点分类。 | 上下文增强 P。 |
| B8 | 井间曲线套件不同：GR、PEF、SP、深电阻率、NPOR、DTC、RHOB 等。另有 CCL 与岩芯图像。 | 5 岩性（sandstone、muddy sandstone、sandy mudstone、oil-stained sandstone、mudstone）；另有 6 个沉积过程相。 | **测井支路**为点级 XGBoost；CCL 的 WaveNet 与图像 DeepTEN 才是上下文序列，输入模态不是普通测井。 | 测井 P + 岩芯/CCL S，必须分开解读。 |
| B9 | 论文称 5 条 wireline + 2 指示变量；列出的字段为 GR、`ILD_log10`、PE、DeltaPHI、PNHIND、NM_M，字段数叙述不完全一致。 | 同 SEG 九类。 | categorical embeddings + feature-wise Transformer 与 numerical residual MLP 融合；论文明确是非序列特征交互，不依赖深度顺序。 | 点级 P；“Transformer”名称不能使其进入 S。 |
| B10 | 统一选 GR、NPHI、RHOB、DTC 四曲线。FORCE 有 12 类，Geolink 有 26 类。 | 保留各公开数据原 schema，不为平衡合并类。 | 浅层模型用长度 1；深度模型用井内连续、无缺失、非重叠序列，研究 1/10/20/50，推荐 1 与 50；长度 N 输出 N 个逐深度标签。 | 点级 P 与真正 sequence-to-sequence S 都覆盖。 |

### 对“序列”的直接证据判断

- B2/B3 是最接近当前 `P_CONTEXT_LENGTH=33` 的直接先例：窗口含上下文，但一个样本只有一个中心目标。
- B4/B10 才是 S lane 的直接先例：模型或解码器必须看到有物理顺序的连续同井段，并输出整段标签。
- B9 只在“字段维”做 attention。把它叫序列模型会把 feature token 与 depth token 混为一谈。
- B8 的 WaveNet/DeepTEN 证明上下文有用，但其输入是 CCL/岩芯图像；不能作为“普通测井 S 模型已验证”的证据。

## 5. 证据矩阵 B：split、盲井、group CV 与泄漏

| ID | 开发 split | blind well / group CV | 测试协议 | 对未见井结论的限制 |
|---|---|---|---|---|
| B1 | Hall 教程先扣出 SHANKLE，但内部调参把其余深度点随机 90/10；Scaler 在随机拆分前拟合。 | 教程 SHANKLE 整井；正式竞赛 STUART/CRAWFORD 两口秘密井。无统一 group CV。 | 竞赛允许所有发布井训练，秘密井标签不可见；随机模型取 100 次 realization 的中位 F1-micro。 | 整井盲测有效；教程内部随机点 CV 明显乐观，不能用来选最终泛化协议。 |
| B2 | `LeavePGroupsOut(2)` 按井划分，只保留 train/val 都覆盖九类的组合。RobustScaler 在 fold-train 拟合。 | 最终仍测 STUART/CRAWFORD。 | 多个按井开发 split 平均 F1，之后全发布井 refit；100 次随机 realization。 | 方向正确；但 PE mean imputer 在生成 split 前对全部发布井拟合，CV 内存在预处理泄漏。按“验证集也必须九类”筛 split 还会产生选择偏差。 |
| B3 | 生成大量重叠窗口后随机抽 10% 作内部 test；窗口间共享大量相邻测点。代码还对每口井单独 `fit_transform` RobustScaler。 | STUART/CRAWFORD 两盲井。无 group CV。 | 盲井 F1/混淆；作者承认过拟合和小样本问题。 | 最终盲井仍有价值，但内部随机重叠窗口和 per-test-well scaling 是转导性信息使用，不可复制到严格 GM09 选择流程。 |
| B4 | 2379 对随机 60/20/20 用于 ANN 训练/验证/内部测试。 | 另有两口完全未训练盲井；无 group CV。 | 在每口盲井上比较 ANN-HMM、ANN、HMM、SVM。 | 盲井结论有效；模型选择阶段仍不是按井 CV。HMM 转移矩阵必须只从训练井估计。 |
| B5 | 每次只把一口井当有标签井，其余九口当无标签井；超参 CV 细节主源未充分报告。 | 九口目标井标签不用于训练，但它们的 X 进入 label propagation/self-training。 | 重复不同 labeled-well 场景，比较半监督与监督模型。 | 是“未见标签井”，不是纯归纳 blind：测试域特征参与训练。当前 F-5 firewall 不应采用这种转导协议。 |
| B6 | 竞赛固定 98 train、10 public test、10 hidden blind；不做统一 group CV。 | 最终 10 口 hidden blind 决定名次。 | 官方记录 329 队、148 队提交、2200 次盲井评分；盲井分布与 train/public test 不同。 | 是强整井外推证据，但公开 leaderboard 被反复使用；只有 hidden blind 可视作独立证据。 |
| B7 | 作者另设 78 训练井 + 两组各 10 口随机整井验证集；最终模型内部使用 10-fold `StratifiedKFold`，实码是逐样本而非 GroupKFold。 | FORCE hidden 10 口最终盲井。 | 手工调参同时比较整井 validation 和 public leaderboard，最后一次 hidden 评分。 | 获胜结果是真盲井；开发 10-fold 不是井隔离，且 public leaderboard 参与调参。不能照搬其 CV。 |
| B8 | 岩性用预定义 3/6/9 口训练井组合并多次整井测试；只有 5 口有 facies，故 facies 做五次 LOOW。 | 每次测试井完全与训练井分开；无单独永久盲井。 | 按井重复测试，明确反对随机深度点划分。 | 强支持 group split；不是一次冻结的外部 blind。测井与厘米级标签分辨率失配导致性能接近多数类基线。 |
| B9 | 宣称八次 LOOW；其余井数据再按 9:1 做 train/val，但 validation 是否再按井分组为 NR。 | 每次一口整井 test。 | 五次随机训练、validation AUPRC early stop、held-out well 指标。 | LOOW 方向正确；论文结果表只列 7 口井且缺 Recruit F9，和“八次”叙述不一致，不能把均值当完整 8-well OOF。 |
| B10 | 对 well-name 列表执行 `KFold(n_splits=5)`，再按井名选取全部序列；Scaler 每折只 fit train wells。 | fivefold held-out wells；无永久外部 blind。 | FORCE 与 Geolink 的全部已发布井进入 grouped CV，准确索引随代码发布。 | 是当前最清楚的按井 CV 直接 benchmark；但把竞赛已发布 blind/hidden 井重新混入 CV 后，不再具有 fresh-blind 属性。 |

## 6. 证据矩阵 C：不均衡、loss/采样、指标、缺失曲线和许可风险

| ID | 类不均衡与 loss/采样 | 指标/图 | 缺失曲线处理 | 关键风险 |
|---|---|---|---|---|
| B1 | SVM 无显式 class weight/采样。 | F1-micro；教程给逐类 precision/recall/F1、普通与 adjacent confusion。 | 教程版本未给通用缺失策略。 | adjacent-F1 可作地质容错诊断，但不能替代固定 schema Macro-F1。 |
| B2 | GradientBoosting exponential/deviance 候选；无 class weight。 | CV F1-micro、平均 confusion、双盲井分数。 | PE 全局均值插补；该 imputer 在 group split 前拟合。 | 预处理泄漏；只保留全类覆盖 split 会掩盖真实 unseen-well 缺类。 |
| B3 | softmax + cross-entropy；无 class balancing；代码做 8% adjacent-label perturbation，这不是不均衡采样。 | 盲井平均 F1 0.574、逐类 P/R/F1 与 confusion、置信度井深轨。 | 训练缺 PE 用 KRR 插补；边界零填充。 | 重叠窗口随机拆分、测试井自标准化、标签扰动都需单独消融。 |
| B4 | ANN cross-entropy；HMM 显式使用训练类先验与转移频率，无 class weight/重采样报告。 | 两盲井 MCC 0.4531/0.5264；逐井深轨与 confusion。Macro-F1 为 NR。 | NR。 | HMM 会平滑薄层；转移矩阵若由全井或测试井标签估计会直接泄漏。 |
| B5 | 类先验进入 self-training；显式重加权/重采样为 NR。 | accuracy/adjacent accuracy 与按井比较；Macro-F1、逐类 F1 完整报告为 NR。 | NR。 | 转导测试特征、超参选择和停止条件使它不适合作 frozen-test 主榜。 |
| B6 | 官方主指标是地质 penalty/cost matrix，不规定统一 loss。 | 官方 blind score、总体/逐类 confusion；不是 Macro-F1 benchmark。 | 缺失普遍存在，各队自定删除、插补或树模型缺失路由。 | 不同曲线套件、区域和 12 类 schema 与 GM09 不可直接比绝对分数。 |
| B7 | `multi:softprob` + `mlogloss`，无 class weight；100 trees、depth 10、lambda 1500。 | penalty score、accuracy、作者代码 weighted-F1；无固定九类 Macro-F1。 | `fillna(-999)`，无 missing mask；删低覆盖曲线。 | WELL/GROUP/FORMATION 编码可能记忆区域/井身份；public leaderboard 调参有偏。 |
| B8 | balanced class weighting；XGBoost 原生缺失分支。 | accuracy、precision、recall、F1、confusion 和井深结果；重点报告 weighted F1。 | 不同井曲线套件可不同；测井 XGBoost 接受缺失。 | 核心标签来自岩芯图像并插值到低分辨率测井，存在尺度失配；代码无许可证。 |
| B9 | cross-entropy；未报告 class weights/过采样；以 AUPRC 关注不均衡。 | AUROC、AUPRC、ACC、F1；未给完整逐类 P/R/F1/confusion 主表。 | 数值均值插补、类别 unknown embedding；均值/标准化 fit scope 为 NR。 | “五 seed”与“七个 trial/井”表述混杂；训练/测试样本总数叙述不一致；代码无许可证。 |
| B10 | 深模用 CE；weighted 变体的 class weights 仅从 fold train 计算，浅模用 fold-train sample weights。 | 9 项：accuracy、balanced accuracy（代码名写作 weighted accuracy）、MCC、macro/weighted P/R/F1、时间；井深图。 | 浅模删除任一选中曲线缺失的行；深模丢弃含任一缺失值的完整序列，无 mask/插补。 | complete-case 选择偏差明显；无独立 fresh blind；代码无许可证。 |

## 7. 十项工作的逐项判断

### B1 SEG 教程/竞赛：整井盲测是基准，随机点 CV 不是

- 教程代码明确把 SHANKLE 留整井，但对其余数据用随机 `test_size=0.1`，且 StandardScaler 在拆分前拟合。教程自己也承认同井随机验证约 0.71，而更合理的盲井更差。
- 正式竞赛后来允许使用所有发布井，只在 STUART/CRAWFORD 秘密标签上计分；这部分才是未见井证据。
- 该基准支持：整井 firewall、逐类 confusion、相邻岩相诊断。它不支持：随机点调参、用 adjacent-F1 替代严格 Macro-F1。

### B2 Bestagini：最接近 P 窗口树模型，但 CV 预处理不够严格

- `LeavePGroupsOut(2)` 是直接的按井 group CV 先例；梯度和交互项是当前树模型 flatten-window 的合理祖型。
- 代码先对全发布数据拟合 PE 均值插补，再建 group split；在现代合同下应改成每折 train-only。
- 它还筛掉 validation 缺类的井组合。GM09 只有四个 development family，不能照此筛 fold；真实缺类必须保留并用固定九类口径计零。

### B3 Inception ConvNet：P lane 的直接深度窗口先例

- 31 点窗口和当前 33 点窗口量级一致；softmax 概率、CE、置信度井深轨都可直接借鉴。
- 它把重叠窗口随机拆分，且 `generate_sequences` 对每口井单独 `fit_transform`，因此内部分数不是严格未见井估计。
- 结论是“中心窗口 CNN 可测”，不是“深度网络必胜”；作者报告深模低于 boosted trees，并明确指出数据量不足。

### B4 ANN-HMM：S lane 的最直接先例

- HMM 转移矩阵提供真实沿深度约束，Viterbi 输出整井序列；这与中心点模型在任务定义上不同。
- 转移矩阵、类先验、emission calibration 都必须 fit fold-train。用 held-out family 的标签统计转移就是泄漏。
- 薄层被 HMM 平滑掉既可能去噪，也可能是地质错误；应报告边界/薄层误差，而非只看总体分数。

### B5 半监督 label propagation：未见标签不等于 fresh blind

- 目标九口井标签不参与训练，但 X 参与图传播和 self-training。这是合法的转导设定，却不是当前 F-5 的归纳 firewall。
- 若未来业务允许“部署前看完整目标井无标签曲线”，可设独立 transductive lane；不得与不看目标井分布的 inductive P/S 榜混排。

### B6 FORCE 官方基准：最强的大规模整井域偏移证据

- 官方记录表明 public test 与 hidden blind 差异明显，顶队在 blind 上显著掉分；这正是冻结最终井族的价值。
- penalty matrix 有地质意义，但依赖 FORCE 12 类的代价定义。GM09 未经军伟确认不能编造相邻类代价矩阵，更不能替代固定九类 Macro-F1。

### B7 FORCE 冠军：XGBoost 是强基线，但开发 split 不能原样复制

- 胜者用整井 validation 检查泛化，最终也在 hidden wells 获胜，支持把 XGBoost 保留为首要 P baseline。
- 最终 notebook 的 10-fold 是逐样本 `StratifiedKFold`，而不是按井 GroupKFold；同时使用 public leaderboard 调参。
- `WELL_encoded` 对新井泛化有明显域记忆风险。当前 GM09 禁止把井 ID 作为预测特征是正确的。

### B8 Q204：按井拆分与缺失曲线的正例，也是标签分辨率的警告

- 作者明确指出随机行拆分会泄漏，并对 facies 做五次 LOOW；XGBoost 可处理井间不同曲线套件。
- 标准测井预测厘米级岩芯标签接近多数类基线，说明标签尺度必须与测井分辨率相容。GM09 只应用专家区间覆盖真实采样点，不应把区间边界无限细化成虚假高分辨率标签。

### B9 attention-residual：LOOW 有价值，但不是序列 Transformer

- 论文明确 attention 在同一深度的字段之间工作，独立于 sequential order。因此只能作为 P/表征融合候选。
- 八次 LOOW 的描述值得借鉴；但结果表仅呈现七口井，Recruit F9 缺席，且数据总数叙述存在版本不一致。复现时必须先锁定数据 commit 和完整 OOF coverage。

### B10 2026 benchmark：group CV 与 P/S 定义最接近当前协议

- 代码先对 well-name 列表 KFold，再取整井数据；Scaler 每折 train-only，直接支持母井族 LOGO4 和 fold-train preprocessing。
- 它明确规定长度 N 序列输出 N 个标签，支持当前“无真实 MD 就不运行 S”。
- 其 complete-case 策略会丢弃缺曲线段，和当前 mask 方案不同。GM09 应保留 mask，同时另报每 fold/类的有效支持，避免完整曲线选择偏差。
- 它把所有已发布 FORCE/Geolink 井重新 CV，没有永久 blind。其 CV 可作 Protocol A，不可替代 Protocol B。

## 8. 未进入优先十项的边界案例

| 工作 | 处理决定 | 原因 |
|---|---|---|
| [Dubois, Bohling & Chakrabarti (2007)](https://pubs.usgs.gov/publication/70029702) | 只作为 Hall 数据/任务的历史来源，不计入 10 项未见井基准。 | USGS 摘要只证明 3600 个样本被分 train/test，未证明按整井留出、group CV 或 blind well。 |
| [Mukhamediev et al. (2024)](https://doi.org/10.3390/app14177779) | 作为补充方法证据，不占优先十项。 | 是直接的 96 口 uranium well、90/10 按井 split 和上下浮动窗口研究，支持“窗口必须在井内且 split 先于窗口”；但矿种、AR/SP/GR 曲线和 9 类岩性语义与 GM09 更远，代码仅 Dropbox 且无明确许可。 |
| 只写“CNN-LSTM/Transformer”但未说明连续 MD、输出对齐或井级 split 的论文 | 不计。 | 模型名称不能证明序列标注，也不能证明未见井泛化。 |
| 通用 InceptionTime、MiniRocket、TCN、MS-TCN2、EmbraceNet、MultiBench 原论文 | 不计为直接岩相 benchmark。 | 可作为本项目算法候选，但原任务不是测井岩性/岩相未见井泛化。 |
| 纯地震 facies benchmark、纯岩芯图片分类 | 不计。 | 输入模态不满足本报告边界；可另做多模态/图像扩展调研。 |

## 9. 当前 GM09 Pipeline：保留、修改、废弃

### 9.1 保留

1. **真实标签与固定 schema。** `pipeline_contract.py:12-28` 固定 `Source=GM09`、`Litho Crv Type=GENETIC FACIES` 和九类：`F-MARSH`、`F-MOUTHBAR`、`F-OFFSHORE`、`F-LOWER SHOREFACE`、`F-UPPER SHOREFACE`、`F-TIDAL BAR`、`F-TIDAL CHANNEL`、`F-TIDAL FLAT MUDDY`、`F-TIDAL FLAT SANDY`。`UNKNOWN/UNDEFINED` 排除见 `pipeline_contract.py:141-149`。
2. **母井家族先分组。** `pipeline_contract.py:66-74,101-108` 在采样前冻结 family，并把 `15/9-19`、`15/9-F-15` 的侧钻归到母井。直接文献只做到 well-level，本方案更严格。
3. **F-5 firewall 的身份。** `split_manifest.json` 把 `15/9-F-5` 固定为 test；development 是 `15/9-19`、`15/9-F-14`、`15/9-F-15`、`15/9-F-4`，请求 5 折因只有四个独立 family 诚实降为 LOGO4。
4. **fold-train-only 预处理、类权重和 softmax 推理。** `p4_contract.py:59-78` 的 fit scope 与 B2/B10 的正确部分一致。
5. **固定九类 Macro-F1 主排名。** Stage2/3/4 已执行固定九类；supported-class 只诊断。F-5 对后两类支持为 0 时仍保留九类 schema，不随机打散同井补齐。
6. **P/S lane 隔离。** README `431-445` 明确 P 输入为 `26x33`（13 值 + 13 mask）和 `3x3x33` seismic patch，S 因 447 个 development 样本没有有限 `center_md_m` 而 `not_rankable`。
7. **缺失 mask。** 相比 B7 的 `-999` 和 B10 的 complete-case 删除，显式 mask 更能保留真实井间采集差异；应继续作为全部模型的固定输入合同。
8. **75% 不排名。** `lithofacies_p5_stage3.py:86,912-916` 要求完成率至少 0.80，且必须覆盖所有四折和三 seed。CatBoost 9/12 缺一折，因此诚实保留失败而不改类。
9. **完整诊断。** 固定九类 confusion、逐类 P/R/F1/IoU/support、校准和缺模态图都应保留；B1/B3/B4/B10 支持井深轨和 confusion 的必要性。

### 9.2 修改（后续另立代码任务，本轮不改）

1. **统一旧 P4 metric contract。** `p4_contract.py:80-104,154,179` 仍把 supported-class Macro-F1 作为 TaskSpec/HPO 主指标和首选规则；应改为 fixed-nine Macro-F1，worst-family 也用固定九类，supported-class 只作诊断。这是合同漂移，不否定 Stage2–4 已按正确口径生成的结果。
2. **把 F-5 的文案统一为 known holdout confirmation。** README `602-615` 和 Stage-4 artifact 已正确；其他说明若仍写 frozen/blind test，应加 `previously_seen_reusable_holdout`，避免独立性误读。
3. **增加 well-log-only（W）与 multimodal（M）成对消融。** W 只用真实测井值+mask，供直接基准比较；M 使用完全相同样本/split/budget 再加 ST0202。两榜分开，不以 M 对 W 的资源优势冒充模型优势。
4. **为 P/S 写死 I/O 语义。** P：`[B,2C,L] (+ seismic)` → `[B,9]`；S：同一口井的连续 `[B,2C,L]` → `[B,L,9]` 并带 position/valid mask。输出维度而非模型名字决定 lane。
5. **S 建序列前增加物理连续性门。** 要求同井、有限 MD、严格单调、采样间隔容差、无跨大缺口/跨井窗口；先 family split，再构序列。不能从 Excel 区间中点或 HDF5 行序恢复 MD。
6. **明确 completion denominator。** 在运行前冻结 legal cell roster；依赖缺失、zero-support、非有限 logits、timeout 都保留结构化状态。模型只有达到 80% 且全 fold/seed coverage 才可排名；75% 明确不排名。若要改变阈值只能在下一 campaign 预注册，不能看到结果后改。
7. **增加类覆盖双报告。** 每 fold 同时给 train/val 的九类 support；零 train support 的 fold 标 `not_feasible` 或按预注册模型能力 fail，不合并类别。固定九类 Macro-F1 仍按 9 类计算。
8. **增加边界/薄层诊断。** 参考 B4，报告相变边界附近误差、预测段长度分布和过度平滑；只作诊断，不改变主排名。
9. **校准 fit scope 落到可验证 artifact。** 温度或其他 calibrator 只能 fit development OOF，不能 fit F-5；同时报告 raw softmax 与 calibrated ECE/NLL。

### 9.3 废弃/禁止

1. 随机打散深度点、重叠窗口或同井样本来补足类别。
2. 用 supported-class Macro-F1、weighted-F1、accuracy 或地质相邻类容错分替代固定九类 Macro-F1 主排名。
3. 为五折合并 GM09 类、删除稀有类、从 F-5 抽样补 train，或把母井侧钻拆到不同 partition。
4. 反复使用 F-5 做 HPO、阈值选择、imputer/Scaler fit、calibration 或模型选择。
5. 使用目标井无标签 X 做 label propagation、自标准化或 transductive imputation，却仍把结果称为 inductive blind。
6. 用行顺序、区间中点、formation 名或人工重复中心样本制造 S 序列。
7. 把通用 TSC/表格/多模态论文当作“已有测井岩相未见井 SOTA”证据。
8. 将 SEG 九类、FORCE 12 类、Geolink 26 类与 GM09 九类的绝对分数横向排名。
9. 把 M（测井+地震）与 W（仅测井）模型混到同一 leaderboard。

## 10. 推荐协议 A：GM09 development 的可重复未见母井估计

**用途：** 模型选择和内部泛化估计；不声称 fresh blind。

1. **冻结对象**：GM09 九类及 class-id 顺序、139 个来源区间、样本 provenance、四个 development family、F-5 身份、输入曲线白名单、窗口长度 33、source-lock、模型预算、seed 和 legal-cell roster 均哈希冻结。
2. **外层评估**：四个 development family 做 LOGO4；每折恰好一个 family 作 validation，其余三个作 train。不能因 validation 缺类删折。
3. **fit scope**：曲线标准化、缺失插补（若有）、类权重、target transform、HMM 转移、特征选择、校准候选都只 fit fold-train。mask 本身来自观测事实，不由 test 统计构造。
4. **P 构造**：先 split 后在各 family 内构 33 点中心窗口；窗口不可跨井、跨 family、跨非法 MD 缺口。输出中心标签。
5. **S 构造**：只有真实有限 MD 且可验证连续时才创建；输出逐深度九类 logits。当前 archive 不满足，故 S 维持 `not_rankable`。
6. **重复**：模型比较使用同一预注册 seed 集；任何候选必须覆盖同一 fold×seed roster。失败保留，不换 seed 追分。
7. **主指标**：每 cell 固定九类 Macro-F1；模型主排名为 cell/fold 聚合的 fixed-nine Macro-F1，第一 tie-break 为 worst-family fixed-nine Macro-F1，再看 seed 稳定性、NLL/ECE、复杂度/时间。
8. **次指标**：accuracy、fixed-nine balanced accuracy、MCC、逐类 precision/recall/F1/IoU/support、count/row-normalized confusion、NLL、Brier、ECE；supported-class Macro-F1 只作诊断。
9. **完成门**：预注册 legal cells 中完成率 `<0.80` 或任何 fold/seed 轴不完整即 `not_rankable`。因此 75% 必须不排名。正式胜者最好要求 100% cells；80% 是“可进榜”的最低门，不是科学性能指标。
10. **可视化**：每折 OOF 井深轨（曲线、GT、预测、置信度、错误）、固定九类 confusion/PRF、reliability、fold×seed 热图、missing-modality/support 图；无真实 MD 时 depth track 明确 `not_feasible`。
11. **模态榜**：W 与 M 使用完全相同的 sample IDs、folds、seeds、模型容量/更新预算，分别排名；报告 `M-W` 的 paired fold 差值，但不把它当独立样本做夸大显著性检验。
12. **F-5**：不参与 A 的任何选择。现有 Stage-4 指标只作为冻结 winner 的历史 known-holdout confirmation 引用，不再次消费追分。

## 11. 推荐协议 B：真正 fresh-blind 的未来确认

**用途：** 对外声称“未见井泛化”的独立确认。

1. 新取得一个或多个从未被任何开发者、模型、图件或统计消费的 GM09 同定义母井家族；在标签揭盲前冻结数据 hash、标签 schema、winner、预处理、预算和一次性命令。
2. 用 Protocol A 选出的单一 winner/config，在全部 development family 上 refit；所有预处理和类权重仍只 fit development。
3. 先持久化 checkpoint/config/hash，再一次性打开新 family；推理后不可回滚状态或继续调参。
4. 按固定九类报告，即使 blind family 缺类也保留 support=0；同时给 supported-class 诊断，不能借后者重排。
5. 如果近期没有新 GM09 family：只报告 Protocol A 的 grouped-CV estimate 和 F-5 的 known-holdout confirmation；明确 `fresh_blind=false`。重复 grouped CV 不能改名为 blind test。
6. SEG/FORCE/Geolink 的标签语义不同，只能做外部预训练/域偏移实验，不能充当 GM09 Protocol B test。

## 12. 现有十模型的公平测试设计

当前 source lock 正好有 10 个 adapter。它们可以作为**本项目候选**，但只有部分有直接地学模型先例；通用原论文不计入第 3 节的直接基准。

| 顺序 | 当前 model_id | lane | 直接地学对应 | 公平测试要求与定位 |
|---:|---|:---:|---|---|
| 1 | `xgboost_multisoftprob_window` | P | B2、B6、B7、B8、B10 | 必备强基线；固定 `multi:softprob`、同 33 点输入、同 mask/地震条件；不得加 WELL/family ID。 |
| 2 | `catboost_multiclass_window` | P | B6/B7 的竞赛候选 | 与 XGB 同输入、round/时间上限；zero-train-support 导致非有限输出必须保留失败。 |
| 3 | `minirocket_ridge_window` | P | 无直接岩相论文 | 探索性 P；只能说明通用卷积特征在 GM09 的结果，不能称复现直接基准。 |
| 4 | `inceptiontime_window` | P | 结构思想接近 B3，但并非 B3 原实现 | 33 点中心窗口、九类中心 logits；与 B3 的 31 点结果只作机制参照，不比绝对分。 |
| 5 | `tcn_center_head` | P | B3/B10 的上下文思想 | 非因果中心窗口；输出 `[B,9]`，不能进入 S。 |
| 6 | `balanced_softmax_tcn` | P | 类不均衡机制无直接地学主源；B8/B10 支持 fold-train weighting | 只允许使用 fold-train 九类计数；与 #5 配对作为 loss 消融。 |
| 7 | `moderntcn_window` | P | 无直接岩相原作 | 探索性 P；同预算，不因现代架构增加窗口/updates。 |
| 8 | `ms_tcn2_dense` | S | 任务思想接近 B4/B10，原 MS-TCN2 不是地学模型 | 只有真实连续 MD 时运行，输出 `[B,L,9]`；当前结构化 skip，不进入 P 榜。 |
| 9 | `embracenet_missing_modal` | P | B6/B8 证明缺曲线问题，但 EmbraceNet 非直接岩相 | 多模态缺失鲁棒性消融；必须保留每模态 availability mask，W/M 分榜。 |
| 10 | `multibench_lowrank_tensor_fusion` | P | 无直接岩相原作 | 多模态交互扩展；参数量、输入和 update cap 与其他神经 P 候选同时报告。 |

### 12.1 固定的 apple-to-apple 条件

- **数据**：同一 GM09 sample IDs、LOGO4、九类 schema；F-5 不参与模型测试。
- **P 输入**：所有 P 模型同一 `26x33` log tensor（13 observed + 13 mask）和同一 `3x3x33` seismic patch；W 消融用相同 adapter contract 中的显式 seismic-absent mask，不删样本。
- **预算**：沿用 Stage2 source lock：每折 train 最多 320、validation 最多 160；神经模型 batch 32、最多 40 parameter updates；XGB/CatBoost 40 rounds；MiniRocket 1000 kernels；同时报告 wall time、CPU/GPU、峰值内存和参数量。
- **随机性**：相同预注册 root/repeat seeds；不能为失败模型追加“幸运 seed”。
- **loss**：候选的 frozen loss 不变；所有 class frequency 只来自 fold-train。不得把不同比例的重采样和不同 loss 一起偷偷改变。
- **输出**：训练接收 logits/loss；推理统一 softmax → argmax。概率校准只能在 development OOF 单独 fit。
- **排名**：P-W、P-M、S-W、S-M 至少按 lane/模态分开。当前 S 无合法数据，所以合法榜只有 P-W/P-M；不得为凑“十模型同榜”把 #8 当中心模型。
- **覆盖**：每个候选预注册 4 folds × repeats；`not_feasible`、`skip`、`timeout`、非有限 logits 分开计，不能用临时 20% split 替代失败 cell。

### 12.2 为什么不能只做一个“十模型总榜”

P 评估的是中心点，S 评估的是连续段；W 与 M 的信息量也不同。把 10 个模型塞入一个总榜会同时混合任务、输入和可运行性。公平的含义应是“每个合法 lane 内同数据同预算”，而不是“强行同榜”。当前应保留 9 个 P 的对比和 1 个 S 的结构化 `not_rankable`，直到真实 MD 序列可用。

## 13. 最小复现实验（后续执行建议，本轮未运行）

### R0：零训练合同复现

目标是先证明比较问题定义正确。

1. 校验 139 个 GM09 区间、九类顺序、UNKNOWN/UNDEFINED/区间外排除。
2. 校验每个 sample 的 label trace、well/family、partition、log/mask/seismic shape。
3. 校验四个 development family 与 F-5 零交集，且 split hash 与 source-lock hash 固定。
4. 校验任何 P 窗口不跨 family；任何 S 序列没有 finite MD 则 fail-closed。
5. 校验 fixed-nine metric 在零支持类上给有限 0，而不是删类或 NaN。

通过标准：不访问 F-5 内容、不写模型 artifact；所有合同测试通过。

### R1：公开 SEG 的泄漏差异最小复现

只用公开 SEG 数据、固定一个 SVM/XGBoost 和一个 31 点 Inception 小模型，比较：

1. Hall 原式随机深度点 90/10；
2. 按井 LOOW/Leave-P-Groups-Out，所有预处理 fold-train-only；
3. 最终 STUART/CRAWFORD 盲井（若许可和标签访问条件允许）。

只比较 split 带来的差异，不做 HPO。预期用途是量化随机点/重叠窗口的乐观偏差，不把 SEG 分数映射为 GM09 分数。

### R2：GM09 单折 development smoke

1. 固定 fold 0、seed 2693、P context 33、train≤320、val≤160。
2. 依 source lock 依次运行 9 个 P adapter；每个记录 fit/forward/loss/checkpoint roundtrip、fixed-nine Macro-F1、support 和资源。
3. `ms_tcn2_dense` 只执行 sequence availability gate；当前应得到结构化 `not_rankable`，不训练。
4. W/M 两个输入条件使用完全相同 sample IDs；先完成 W，再 M；禁止访问 F-5。

通过标准：10 个候选都有 PASS/FAIL/SKIP 记录，P 榜只含合法 P cell，S 不混榜，任何非有限值 fail-closed。

### R3：正式 Protocol A

R0–R2 通过后才运行 LOGO4 × 预注册 seeds；冻结 cell roster、预算和 hash。输出 OOF、井深图、fixed-nine confusion/PRF、校准和缺模态诊断。模型完成率低于 80%或缺 fold/seed 轴即 `not_rankable`。

### R4：Protocol B

仅在得到新、从未消费的 GM09 母井家族后执行一次。F-5 不可充当 R4。

## 14. 对当前五个重点问题的最终核查

| 问题 | 结论 | 证据与边界 |
|---|---|---|
| GM09 固定九类是否符合直接基准？ | **符合，而且必须保留。** | B1–B5 保持 SEG 九类，B6/B10 保持各自原 schema；没有主源支持为 fold coverage 临时合并标签。GM09 与 SEG 只是同为 9 类，语义不能互换。 |
| 母井家族 split 是否合适？ | **合适，优于普通按井 split。** | B2/B8/B9/B10 直接支持整井/按井 CV；侧钻共享地质和采集条件，按母井 family 合并更保守。 |
| F-5 holdout 是否仍是 blind？ | **不是 fresh blind。** | 本地 Stage-4 `summary.json` 明确 `prior_test_consumed=true`、`fresh_blind=false`、`evidence_class=previously_seen_reusable_holdout`。只能作为已知 holdout 确认。 |
| 75% 完成率不排名是否合理？ | **合理但属本项目预注册门槛。** | Stage3 最低 80%且要求全 fold/seed coverage；CatBoost 9/12=75% 缺 fold 2，不能算 worst-family，也不应排名。文献没有给出“75%”这一行业阈值。 |
| 序列模型 lane 是否正确？ | **定义正确，当前数据状态下不运行也正确。** | B3 是 P 中心窗口；B4/B10 是 S 逐井序列。现 archive 447 个 development 样本都无 finite `center_md_m`，因此 S `not_rankable`；模型名不能越过数据合同。 |

## 15. 本地证据索引

- 标签/schema/母井：`_pipelines/02_task_datasets/lithofacies/pipeline_contract.py:12-28,66-74,101-108,141-149`
- 冻结 split 与样本/类计数：`_pipelines/02_task_datasets/lithofacies/_outputs/split_manifest.json`
- P4 TaskSpec/HPO 指标漂移：`_pipelines/02_task_datasets/lithofacies/p4_contract.py:80-104,154,179`
- 十模型 source lock：`_pipelines/02_task_datasets/lithofacies/p5_source_lock.json`
- P/S 输入与 S 数据阻断：`_pipelines/02_task_datasets/lithofacies/README.md:431-445`
- Stage3 80% 门与全轴覆盖：`_pipelines/02_task_datasets/lithofacies/lithofacies_p5_stage3.py:78-86,912-966`
- Stage3 当前榜：`_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage3/p5_stage3_gm09_p_leaderboard.json`
- Stage4 F-5 已见 holdout：`_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage4_confirmation/summary.json`

## 16. 来源核验备注

1. Bestagini 的机构 PDF 直链在命令行返回 403，因此论文身份/摘要以 DOI 和机构记录核验，算法细节以 SEG 官方仓库固定 commit 的 notebook 实码交叉核验。
2. Dunham 论文主页面受出版社访问限制；任务、one-labeled/nine-unlabeled 设计来自原始 DOI/作者公开手稿与作者单位出版目录。未看到的 loss、缺失处理和完整逐类指标均写 NR。
3. GitHub “无 LICENSE”按未授权处理；不因 README 写 open source、论文 CC BY 或包依赖本身有许可证而推断仓库代码许可。
4. 未把论文摘要中的“context”“attention”“time series”自动解释为沿井深序列；只有能核实 MD 顺序、窗口构造和输出对齐的工作才进入 P/S 证据。
5. 所有网页和 commit 在 2026-07-14 可访问；本轮没有执行第三方代码或复算其指标。
