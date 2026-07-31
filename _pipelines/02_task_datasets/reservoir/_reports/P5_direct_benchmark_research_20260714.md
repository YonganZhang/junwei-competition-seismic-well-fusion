# P5 储层物性直接基准调研：PHIF/PHIE、KLOGH 与 SW

日期：2026-07-14

调研范围：仅纳入用测井，或用地震与测井联合预测孔隙度、渗透率、含水饱和度的石油地学工作。普通表格回归、UCI、房价等非油气任务不纳入。

本地审计基线：`67667739739203703ee5c95a3c8aeb7613b21bf8`（开始调研时工作树 clean）。

结论口径：本报告是有界证据调研，不声称系统综述。`NR` 表示在已核验的一手页面、论文正文或官方仓库中未报告，不能据常识补写。

## 结论先行

1. 当前 Stage3 的“母井家族 LOGO、三目标独立 mask/排名、fold-train 拟合预处理、KLOGH 可逆 `log1p`、物理域与变换域同时报告、worst-family 证据”比多数已检索工作更严格。很多论文仍采用随机深度点切分，或者没有说明标准化是否仅拟合训练折。
2. F-15 不是新鲜盲测。它已被历史 baseline、P4 和 Stage4 消费；Stage4 也已正确标成 `previously_seen_reusable_holdout`、`prior_test_consumed=true`、`fresh_blind=false`。它可以继续做回归确认和漂移诊断，但不能再为模型选择、HPO 或“外部泛化”提供独立证据。
3. Stage4 的高分不能和论文数值横向排位。当前标签是 Volve 解释曲线，许多论文的标签是岩心、NMR 或 MICP 实验；标签来源、单位、深度相关性、测试盲性和井数均不一致。特别是 KLOGH，物理域长尾会让 RMSE 对极端值非常敏感。
4. 必须补一份从未被任何 P4/P5 模型、阈值、HPO 或图表消费的外部验证：最低要求是一个新母井家族；可信的跨域结论还需要一个有明确许可和标签谱系的可比油田。若只能找到 PHIE、岩心 K 或 MICP-SW，必须作为独立 label version/case，不能与 PHIF/KLOGH/解释 SW 混值。
5. 十模型不能排成一个总榜。八个 scratch tabular 模型可在同一 tabular lane 排名；TabICLv2 使用官方发布且许可明确的回归 checkpoint，应在重新登记官方 source/version/hash 后进入独立 pretrained lane，而不是沿用本地旧许可 gate；MONAI DenseNet3D 属于 seismic-3D lane，单候选时只能记录 pilot、`not_rankable`。

## 检索与纳入规则

- 优先级：原始论文/出版社正文、官方数据或竞赛页、作者官方 GitHub；搜索聚合页只用于发现，不作为技术结论的最终依据。
- 直接性 A：目标就是 PHIF/PHIE/其他明确孔隙度、渗透率或 SW，输入为测井或地震+测井。
- 直接性 B：目标是与上述物性紧邻的同域代理，例如 NPHI；只能作为方法学旁证，不能当作 PHIF 数值基准。
- 排除：目标不直接相关、仅有数据卡而无有效 benchmark、无法确认油气地学语境、合成玩具数据、或只有论文名而没有可核验方法正文。
- 共筛查 16 项，保留 10 项直接工作；其中 10 项均有原始论文页面，另有 1 项作者官方代码仓库。满足“至少 6 项一手来源”。

## 证据矩阵 A：数据、标签与验证设计

| ID | 一手来源与直接性 | 油田/规模 | 输入 | 标签来源与单位 | 切分、外部井、CV 与盲性 |
|---|---|---|---|---|---|
| B1 | [Helle, Bhatt & Ursin, *Geophysical Prospecting* (2001)](https://doi.org/10.1046/j.1365-2478.2001.00271.x)，A：直接预测孔隙度和渗透率 | 北海砂岩；样本数在可访问摘要中 NR | PHI 网络：声波、密度、电阻率；K 网络：密度、GR、中子孔隙度、声波 | 氦气岩心孔隙度，fraction；岩心渗透率，mD | 两个独立网络。井级切分、外部井、CV、真盲测试均 NR，因此不能证明跨井泛化 |
| B2 | [Wood, *Journal of Petroleum Science and Engineering* (2020)](https://doi.org/10.1016/j.petrol.2019.106587)，A：同一方法直接估计有效孔隙度、渗透率、SW | 阿尔及利亚 Hassi R’Mel；一个 100 m 复合储层剖面 | 8 条测井加岩性/层序索引，共 10 变量 | 有效孔隙度 EP，fraction；有效渗透率 Ke，mD；SW，fraction | 同剖面最近邻/数据匹配；未见井级留出、外部井或 CV。测试不是独立母井盲测 |
| B3 | [Urang et al., *Journal of Applied Geophysics* (2020)](https://doi.org/10.1016/j.jappgeo.2020.104207)，A：直接预测岩心孔隙度与渗透率 | 尼日尔三角洲；后续应用于 4 口井 | PHI：RHOB；K：RHOB+SW | 岩心孔隙度；岩心渗透率；单位在已核验摘要中 NR | 报告 train/validation/test，但未证明按井切分；4 井结果未证明各井完全排除于训练。CV、母井家族隔离 NR；不能视为真盲跨井测试 |
| B4 | [Tamoto, Gioria & Carneiro, *Journal of Petroleum Science and Engineering* (2023)](https://doi.org/10.1016/j.petrol.2022.111169)，A：直接预测 NMR 总/有效/自由流体孔隙度 | 巴西 Santos 盆地 Búzios 盐下碳酸盐岩；6 井、123,713 深度点 | 18 个辅助测井特征，包含 GR、POTA、THOR、URAN、RHOB、PE、深浅电阻率、DTC 和元素产额 | NMR PhiT、Phie、自由流体孔隙度，fraction | 论文区分 3 个 test wells，较随机点切分更接近外部井；未见 nested CV 或每折预处理声明。测试井是否完全未参与全部 HPO 需复现时再核验 |
| B5 | [Zhang et al., *Applied Sciences* (2024)](https://doi.org/10.3390/app14103956)，A：直接预测岩心孔隙度和渗透率 | 鄂尔多斯盆地庆阳气田；94 井、3,389 个岩心实验点 | AC、CAL、CNL、DEN、SP、RLLS、PE、GR、RT 等常规测井 | 岩心 POR 与 PERM；论文数据表中的单位需复现时逐表核验 | 80/20 随机行切分，不是井级切分；无外部井 CV。随机搜索、手工网格和最终报告共用同一测试框架，真盲性不足 |
| B6 | [Davari & Kadkhodaie, *Scientific Reports* (2024)](https://doi.org/10.1038/s41598-024-73846-2)，A：直接预测岩心渗透率 | 可比碳酸盐岩油田；A 井 86 点、B 井 301 点，C 井为盲井 | GR、RT、PHIE、RHOB、DT、NPHI 的 57 种组合 | 岩心渗透率；物理单位按文中渗透率口径，复现前应从数据表锁定 | A/B 内部随机深度点 70/20/10；C 井训练时未使用，是本批少数明确外部井证据。无母井家族 CV；只有一个外部井 |
| B7 | [Gohari Nezhad & Emami Niri, *Journal of Petroleum Exploration and Production Technology* (2025)](https://doi.org/10.1007/s13202-025-01995-9)，A：直接预测 SW | 伊朗 South Pars 碳酸盐气田；4 井、10,674 点 | Depth、CGR、CAL、DTCO、NPHI、PEF、PHIE、RHOB、RLA3/RLA5、矿物/页岩比例及井/层带索引 | 由 MICP 岩心 Pc-SW 与饱和度高度模型校正得到的实验约束 SW，fraction | A/B/C 随机 85/15 建模；空间分离的 D 井在 Part I 中未参与建模/HPO，是真盲井。Part II 将 4 井用于 10-fold CV，之后 D 不再是新鲜盲测 |
| B8 | [Liu et al., *Journal of Geophysics and Engineering* (2023)](https://doi.org/10.1093/jge/gxad063)，A：地震到孔隙度 | 中国非均质碳酸盐储层；W1–W17 建模，W18/W19 外部井 | 叠前角度道集、反演 P 波阻抗、Vp/Vs、低频孔隙度模型 | 平滑并降采样到地震尺度的孔隙度测井；单位在已核验正文段落中 NR | 100 次实验，每次从 W1–W17 随机留 4 井；W18/W19 始终完全排除，且低频模型也不使用其井数据，是真外部井设计。没有母井家族概念 |
| B9 | [Corrales, Hoteit & Ravasi, Seis2Rock](https://doi.org/10.1029/2023EA003301)；[作者官方 GitHub](https://github.com/DeepWave-KAUST/Seis2Rock)，A：地震到孔隙度/SW | Volve 实例和 Smeaheia 合成实例；Volve 使用 15/9-19 BT2、15/9-19 A | 测得/岩石物理生成的弹性-物性日志、合成 AVO、叠前/叠后地震 | Volve 物性包括孔隙度；Smeaheia 包括孔隙度、SW 与 4D SW | Volve 基底先用 BT2 后加入 A，剖面沿这两井展示；未形成第三口独立外部井。Smeaheia 是合成验证，不能替代真实外部井 |
| B10 | [Zhang et al., *Geofluids* (2022)](https://doi.org/10.1155/2022/9443955)，A：同一工作直接预测孔隙度、渗透率和 SW | 鄂尔多斯盆地 Jiyuan 油田 Chang 8 砂泥岩；北/南子区，第一实验测试集 420 点 | 常规测井序列；可访问主文未完整列出全部 mnemonic，故逐曲线记 NR，而不从二手材料补写 | 岩心测量向量：孔隙度 `%`、K `mD`、SW `%` | 三组实验含大/小样本及南区 transfer case；优化内使用 K-fold，但主文未证明按井或母井家族切分，420 点 test 也未证明外部井。不能视为真盲跨井测试 |

## 证据矩阵 B：建模、变换、指标与可复现性

| ID | 预处理是否逐折拟合 / K 变换 | 模型、loss 与 HPO | 报告指标与物理越界 | 代码、权重与许可状态 | 主要风险 |
|---|---|---|---|---|---|
| B1 | 标准化/逐折拟合 NR；K 变换 NR | 两个独立反向传播 ANN；loss/HPO 细节在摘要中 NR | 平均孔隙度差 `<0.01` fraction；平均 K 差约 400 mD；边界检查 NR | 未找到作者官方代码/权重；论文受出版社版权约束 | 老工作、样本和切分不可审计；只能证明“独立 PHI/K 网络”是已有做法 |
| B2 | 同一复合剖面数据匹配；逐折拟合不适用/NR；K 不变换 | 优化最近邻 `TOB` 数据匹配，不是梯度学习；HPO 为匹配参数优化 | 100 m/10 cm 时约 K 15 mD、SW 0.1、EP 0.01 RMSE；局部 10 m/1 cm 时约 K 1.3 mD、SW 0.003、EP 0.0006；越界 NR | 未找到代码/权重；Elsevier 论文许可，不等同开源代码许可 | 近邻深度泄漏、单剖面、分辨率变化会显著改变误差，不能当跨井 benchmark |
| B3 | 报告缩放/归一化；是否只拟合 train NR；K 变换 NR | Bayesian-framework MLP；MSE；另有 Gauss–Newton 曲线拟合 | PHI 测试 MSE `1.37204e-6`；K 测试 MSE `0.483728`；4 井相关系数 PHI `.99987/.99099/.99474/.83749`，K `.97584/.83594/.97002/.83512`；边界 NR | 未找到官方代码/权重；出版社论文许可 | K 模型把 SW 当输入，对本项目会构成跨目标解释曲线泄漏；切分很可能高估跨井性能 |
| B4 | 论文给出训练/测试统计；是否按每个训练折拟合 NR；非 K 任务 | MLP、AdaBoost、XGBoost、CatBoost；CatBoost 最佳；HPO 细节需全文复现实验锁定 | 最佳 adjusted R² 约 `.87`、RMSE `<0.01`；报告差异多小于 5%；原始预测越界率 NR | 数据需 ANP 许可；未找到官方代码/权重；无可直接复用代码许可 | 目标是 NMR PhiT/Phie/FF，不能与 PHIF 混成一个标签；测试井少且数据非公开 |
| B5 | 比较 `LOG(PERM)`、`arctan(PERM)`，最终结合 K-means 分组；逐折拟合不成立 | XGBoost；穷举特征组合、随机搜索、手工网格 | 变换后 K test R² 约 `.26`/`.19`；分组后 PHI R² `.73/.85`、K R² `.58/.85`；未报告严格物理越界 | 论文为 CC BY 4.0；未找到官方代码/权重/数据发布 | 行随机切分、用同一框架选特征/超参、井间重复地层会乐观；但支持 K 长尾变换必要性 |
| B6 | 全输入/输出 min-max 到 `[0,1]`；scaler 是否只拟合 train NR；K 变换 NR | ELM、RF、GB、KNN、MLP；57 输入组合，共 285 预测 | C 盲井 R²：GB `.92`、RF `.90`、ELM `.73`，KNN/MLP 更低；可访问证据未锁定 C 井 RMSE；越界 NR | 数据按请求且受 IRD 许可；无官方代码/权重；文章 CC BY-NC-ND 4.0 | 输入含 PHIE：若与 PHI/K 联合赛道共用会泄漏；随机深度切分仍用于开发选择 |
| B7 | 对 RLA3/RLA5 做 log；StandardScaler+MinMaxScaler，fit scope NR；非 K 目标 | XGBoost grid search；CNN/LSTM/CNN-LSTM 用遗传算法；MSE loss、MAE 监控、early stopping | 真盲 D：CNN-LSTM R² `.859`、MAE `.095`、MSE `.013`；1D CNN `.801/.112/.017`；LSTM `.765/.123/.021`；XGB `.632/.170/.037`；Archie `.322/.192/.068`；越界 NR | 数据保密；无公开代码/权重；文章 CC BY-NC-ND 4.0 | 输入 PHIE，且 SW 标签受孔隙度/渗透率建模影响；不可与纯原始测井输入直接比 |
| B8 | 对全部数据标准化；是否每次仅用 13 训练井拟合统计 NR；非 K 任务 | 监督 CNN；NNI/TPE HPO；卷积、ReLU、Adam | 100 次随机盲井实验平均 RMSE 从约 `4.09%` 降至 `2.66%`；W18/W19 相对 RMSE 改善约 32%/30%；绝对外部井误差和越界率 NR | 未找到作者官方代码/权重；出版社开放访问条款需复用前单独审计 | 若低频模型含测试井会隐性泄漏；该文明确排除 W18/W19，是本项目地震 lane 应保留的防火墙 |
| B9 | SVD 基底、正则化反演；不是 fold-train scaler；K 不涉及 | Seis2Rock 最优基底投影+叠后反演；无监督 ML loss/HPO | 论文以剖面和反演结果为主；已核验证据中无可与本项目直接比较的 PHI/SW MAE/RMSE/R²；越界处理需代码复现 | 官方仓库 commit `6996b0e5aebf3a1620c153d7aaa448ddde3dc2e4`，MIT；无需预训练权重 | Volve 两口相关井参与基底/展示，不是独立 F-15 式盲测；合成结果不能替代真实井 |
| B10 | 先异常值检测和 normalization，再建基本数据集；是否只 fit train/fold NR；K 指标使用对数值 RMSE | KNN、SVR、RF、CRBM-Bayes-LightGBM；squared loss；CRBM 降维、Bayesian HPO、K-fold、transfer learning | LightGBM 在三目标均取得表中最小 RMSE；PHI/SW 用物理域 RMSE，K 用 log-K RMSE；可访问 HTML 未稳定呈现 Table 6 数值，故不转抄；R²和越界率 NR/未完整报告 | 数据向通讯作者申请；未找到官方代码/权重或代码许可；论文开放获取不等于实现开源 | 预处理先于切分的流程存在全局统计泄漏风险；子区 transfer 不是已证明的外部井盲测 |

## 十项证据的可用结论

### B1：北海岩心 PHI/K 的两个独立 ANN

它直接支持“孔隙度与渗透率可以共享数据源但应保持独立目标建模”的设计。论文使用的输入集合也表明 GR、声波、密度、中子和电阻率是常见物性预测输入。由于未核实到井级留出、CV 和代码，不能用其误差作为 Stage3/4 的数值基准。

### B2：Hassi R’Mel 三物性同时估计

这是十项中少数同时覆盖有效孔隙度、渗透率和 SW 的工作，但方法在一个复合剖面上做最近邻数据匹配。误差会随取样间隔从 10 cm 到 1 cm 大幅下降，说明局部深度自相关本身就能制造很漂亮的指标。它反向支持本项目坚持母井家族先切分、再插值/窗口化。

### B3：尼日尔三角洲岩心 PHI/K

目标直接，但其 K 输入含 SW。对当前三目标协议，这种设计不合法：SW 及其派生解释曲线不得进入 KLOGH 输入。论文的随机点 train/validation/test 和后续 4 井相关系数也不足以证明外部井独立。

### B4：Búzios NMR 孔隙度

6 井和 3 test wells 是较有价值的跨井证据，也说明 CatBoost/XGBoost 对多种常规与元素测井有效。其 Phie 是独立实验标签版本；如果以后获得合法数据，应建立 PHIE 独立 manifest、指标和模型选择，不能把它补入 PHIF 缺失值。

### B5：庆阳气田岩心 PHI/K

该文直接验证 K 长尾变换和分组建模可能改善拟合，因此当前 `log1p(KLOGH)` 有文献动机。但 94 井仍采用行级 80/20 切分，模型看到同井邻近地层，不能作为跨井证据。其 R² 不应与本项目母井 LOGO 的 R²直接比较。

### B6：碳酸盐岩盲井 K

Well C 是清楚的外部井，这是比随机点切分更可信的设计。但开发阶段仍在 A/B 内随机分深度点，且输入使用 PHIE。当前项目可借鉴“外部井只使用一次”，不能照搬其输入白名单。

### B7：South Pars 盲井 SW

Well D 在 Part I 明确未用于建模和 HPO，是真盲设计；其 CNN-LSTM 在 D 井的 R²/MAE/MSE 是本批较完整的外部井指标。Part II 把四井都用于 10-fold 后，D 的 single-use 状态已被消费，这与当前 F-15 的历史完全相似。其 PHIE 输入和实验约束 SW 标签谱系与当前解释 SW 不同，数值只能定性对照。

### B8：地震+井的外部井孔隙度

W18/W19 连低频孔隙度模型都不允许使用，避免了“标签没进训练，但测试井进入低频背景”的隐性泄漏。当前 seismic patch lane 若做外部井验证，应复制这一约束：任何井震标定、低频模型、归一化、空间插值或邻井选择都必须在 test 井冻结前完成，且不得消费 test 井标签。

### B9：Volve Seis2Rock

这是最直接的 Volve 地震-物性官方代码证据，且仓库为 MIT。它证明基于实测/岩石物理日志构造最优基底再反演孔隙度/SW 是可运行路线，但 Volve 例子使用 15/9-19 BT2 与 A 建基底和展示，没有独立第三井。因此它是方法候选或物理 baseline，不是 F-15 独立性背书。

### B10：Jiyuan 三目标 LightGBM 与 transfer learning

这是另一项同时覆盖孔隙度、渗透率和 SW 的直接工作，并明确用 log-K 误差处理数量级跨度。它支持当前三目标分别度量以及 KLOGH 双域指标。其异常值检测、归一化、CRBM 降维和 Bayesian HPO 的顺序没有提供 fold-train 哈希，且没有证明按井切分，因此不能把其 transfer case 等同于外部井泛化。

## 排除项与原因

| 来源 | 处理 | 原因 |
|---|---|---|
| [Kaggle: Drilling Log Dataset](https://www.kaggle.com/datasets/ahmedelbashir99/drilling-log-dataset) | 排除出 10 项 benchmark；只记为暴露风险 | 页面指向 Volve F-15/CPO/钻井数据并含 PHIF、VSH、SW，但没有可审计模型、井级 split 或可靠源许可。更重要的是 F-15 已是公开数据对象，不能假设其长期保持外部盲性 |
| [Kaggle: Well Logs](https://www.kaggle.com/datasets/sahasourav17/well-logs/data) | 排除 | 数据卡只有通用曲线描述，缺少可信油田谱系、井 ID、直接 PHIF/K/SW benchmark 和盲测协议；数据卡许可证不能替代上游数据许可 |
| [Kanfar et al.: Real-Time Well Log Prediction From Drilling Data Using Deep Learning](https://arxiv.org/abs/2001.10156) | 排除 | Volve 目标是 NPHI/RHOB/DTC，不是 PHIF/PHIE/KLOGH/SW；且滑窗后抽 5% 测试造成 train/test 窗口重叠，只能作为切分风险旁证 |
| FORCE 2020 Well Log Challenge 及镜像 | 排除 | 主要目标是岩性分类，不是 PHIF/PHIE、KLOGH 或 SW 连续预测 |
| 合成油藏属性表格数据 | 排除 | 没有真实井/震观测和真实标签谱系，不能回答本次跨井泛化问题 |
| [Volve F-12 孔隙压力预测](https://doi.org/10.1038/s41598-025-89199-3) | 排除 | 目标是孔隙压力；PHI/K/SW 是输入而非本次目标。若纳入会把“输入相关”误当“目标直接” |

## 与当前 Stage3/4 的逐项对照

### 本地证据

- Stage3 总结：[`_outputs/p5_stage3/p5_stage3_summary.json`](../_outputs/p5_stage3/p5_stage3_summary.json)
- Stage3 PHIF/KLOGH/SW 独立榜：[`_outputs/p5_stage3/`](../_outputs/p5_stage3/)
- Stage3 冻结 split：[`_outputs/p5_stage3/p5_stage3_split_manifest.json`](../_outputs/p5_stage3/p5_stage3_split_manifest.json)
- Stage4 合同及历史暴露哈希：[`reservoir_p5_stage4_contract.json`](../reservoir_p5_stage4_contract.json)
- Stage4 已见 holdout 总结：[`_outputs/p5_stage4_confirmation/summary.json`](../_outputs/p5_stage4_confirmation/summary.json)
- 历史 F-15 指标：[`_outputs/metrics.json`](../_outputs/metrics.json)

### 当前输入与标签口径

Stage4 合同锁定的科学输入是 `ST0202_seismic_patch`、GR、RT、NPHI、RHOB 及四条曲线各自的 observed mask；PHIF、PHIE、LFP_PHIE、KLOGH/KLOGH_NEW/KLOGV、SW、BVW、SWIRR、VSH/LFP_VSH 均为 forbidden inputs。当前 tabular adapter 实际使用 153 维融合平铺视图：`3×3×9=81` 个真实地震 patch 值，加 `9×8=72` 个测井序列/显式 mask 值。因此这里的 `tabular_cpu` 是**模型接口 lane**，科学模态仍是“地震+测井”，不能把其分数冒充纯测井 benchmark。MONAI lane 消费原始 patch 张量，信息表示和计算预算不同。

当前 PHIF、KLOGH、SW 来自 Volve 解释曲线构建链，而 B1/B3/B5/B6 的主要标签是岩心、B4 是 NMR、B7 是 MICP/饱和度高度约束。即使单位相同，也不是同一测量真值。当前 Stage4 还对 PHIF/SW 做 `[0,1]` clip，对 K 做 `expm1(clamp(log1p_prediction,0))`；所以必须另存 clip 前输出和越界率，才能与无边界后处理的论文公平比较。

### 当前数值，只能作为内部证据

| 目标 | Stage3 唯一胜者 | Stage3 development LOGO OOF（4 家族×3 seed） | worst mother-family | Stage4 已见 F-15（n=344） |
|---|---|---|---|---|
| PHIF | ExtraTrees | RMSE `0.02764054`，MAE `0.01765264`，R² `0.838256` | RMSE `0.04979029` | RMSE `0.00931924`，MAE `0.00606697`，R² `0.980711` |
| KLOGH | ExtraTrees | physical RMSE `542.90256 mD`，MAE `206.29050 mD`，R² `0.399466`；`log1p` RMSE `1.204951` | physical RMSE `967.627 mD` | physical RMSE `278.8388 mD`，MAE `145.6398 mD`，R² `0.889133`；`log1p` RMSE `0.704012` |
| SW | XGBoost | RMSE `0.17042689`，MAE `0.14083380`，R² `0.711261` | RMSE `0.226226` | RMSE `0.08055030`，MAE `0.06441160`，R² `0.903155` |

Stage4 显著优于 development LOGO OOF，但这不能自动解释为泛化能力提升。F-15 可能更容易，也已被历史 baseline/P4 观察；一次较容易的已见 holdout 不等于多个独立外部井。KLOGH 的 Stage4 区间覆盖为 `1.0`，但平均物理区间宽约 `2879.75 mD`，说明“覆盖高”不等于“不确定性有用”，还必须报告区间宽度、校准曲线和按井覆盖。

### 符合前人做法且应保留

1. **三目标独立建模和独立排名。** B1 用独立 PHI/K 网络，B4 区分三种 NMR 孔隙度；当前独立 target mask 比把多种解释曲线混成标签更稳妥。
2. **KLOGH 先做可逆长尾变换。** B5 直接比较 log/arctan 变换；当前训练 `log1p(KLOGH)`、反演回 mD 并同时报告两域指标是合理做法。
3. **母井家族在任何插值/窗口化前冻结。** B2、B5 暴露了邻深度和行随机切分的乐观风险；当前协议明确避免这一点。
4. **development 用 LOGO、test 只一次。** B6、B7、B8 的真正外部井是最可信证据；当前 Stage3 不访问 F-15 的防火墙与其原则一致。
5. **地震和表格分 lane。** B8/B9 与纯测井模型的输入信息量不同，不应跨 lane 排名。当前 MONAI 3D 单候选 `not_rankable` 是正确结论。
6. **worst-family、重复 seed 和完整失败记录。** 多数论文只给平均数；当前 worst-family 更能暴露跨井退化。

### 必须修改的结论或下一阶段协议

1. **把 F-15 永久降级为 known-holdout regression confirmation。** 不得在摘要、图题或 leaderboard 中再称 blind、external test 或 final test；不得再用它选模型、阈值、特征、区间或 HPO。
2. **新增真正未见的测试单元。** 最低是一个新母井家族，推荐至少 2 个外部井/家族；只有 1 个测试家族时不能稳定估计跨井方差。跨油田验证必须单列，不与 Volve 内部指标平均。
3. **锁定标签谱系。** PHIF、NMR PHIE、岩心 porosity 是不同 case；解释 KLOGH 与岩心 permeability 是不同 case；解释 SW 与 MICP/SCAL 约束 SW 是不同 case。每个 case 独立 manifest/normalization/metrics。
4. **增加原始越界率。** 在任何 clip 前报告 `PHI<0 or >1`、`SW<0 or >1`、`K<0` 的比例、最大越界和按井统计；clip 后的 RMSE 只能作为附加诊断。
5. **证明所有变换逐折拟合。** 包括缺失值策略、scaler、特征筛选、目标变换参数、校准和不确定性；B4/B6/B7/B8 都没有足够清楚地报告这一点，当前项目应把 fit-row hashes 写入 fold manifest。
6. **不再仅用密集深度点置信区间。** bootstrap/显著性检验应以母井家族为重采样单元；同井相邻深度点不是独立样本。
7. **增加标签来源匹配的简单基线。** SW 至少对比可合法构造的 Archie/经验关系；K 至少对比训练井内拟合的线性/Kozeny-Carman 类基线；但任何用 PHIF/SW 解释曲线的物理式不能偷偷进入“原始输入”模型。

### 应废弃的做法或声明

- 废弃“Stage4 F-15 是最终盲测”及“Stage4 高 R² 证明外部泛化”的声明。
- 废弃按深度点随机划分 train/test、先全数据插值/标准化再切分、滑窗后切分。
- 废弃把 PHIF、PHIE、NPHI 或不同实验来源 K/SW 填成同一标签列。
- 废弃用 PHIF/KLOGH/SW 或其派生解释曲线预测另一个目标，除非建立明确、单独命名的有条件任务；这类任务不能进入当前原始测井白名单榜。
- 废弃把 tabular、预训练 tabular 和 seismic-3D 放入同一个 leaderboard。
- 废弃跨论文直接比较 R²/MAE/RMSE 排名；除非标签、单位、split、盲性和测试井完全一致。
- 废弃只报 clipped 指标或只报 K 的 log 域指标；物理 mD 必须是主结果之一。

## 推荐的下一版验证协议

### 1. 数据与 single-use 状态

建立三层、不可回退的状态机：

1. `development_only`：当前四个 Volve development 母井家族，只用于 LOGO、模型选择和 HPO。
2. `known_holdout_consumed`：F-15，只做固定赢家的回归确认；禁止参与任何选择。
3. `fresh_blind_external`：新母井家族或新油田，manifest 在训练前登记 SHA256、标签版本、母井归属、单位、可用输入和唯一开启人；开启后立即写 single-use 状态，不能“重新变盲”。

如外部油田的井名/侧钻关系不完整，先 fail-loud，不得用字符串井名猜家族。若没有可合法使用的外部标签，本阶段结论只能停在“Volve development 内部跨家族验证完成”。

### 2. 每个目标独立执行

- PHIF primary case：只用 PHIF。PHIE 若覆盖足够，另建 `porosity_phie` case，不填补 PHIF。
- KLOGH case：原始物理单位锁为 mD；训练用 `log1p`，预测用 `expm1`。同时报告物理域 MAE/RMSE/R²/Pearson 和 log 域诊断；记录负值率及 p50/p90/p99 误差。
- SW case：锁定 fraction；报告 MAE/RMSE/R²/Pearson、`[0,1]` 越界率，并按 SW 区间分层统计。
- 每个 case 独立 target mask、split manifest、fold hashes、赢家、checkpoint 和开启状态；一个目标缺标签不能改变另两个目标的样本或 test 选择。

### 3. development、HPO 与 final test

1. 在 development 中做 mother-family LOGO；所有预处理只 fit fold-train。
2. 若做 HPO，使用 nested family CV 或固定内部 family folds；HPO 方向、搜索空间和预算先写入 manifest。外层 OOF 和 fresh test 不回流。
3. 以母井家族为单位汇总平均、最差家族和 family-bootstrap 区间；不能把数万相邻深度点当独立重复。
4. 在 Stage3 赢家和配置冻结后，才允许单次打开 fresh test。若中途因代码错误重跑，必须证明预测语义未变并保留失败记录。
5. 外部井的井震标定、低频模型、空间插值和归一化不能使用其标签；若使用无标签井坐标/地震，必须预先在 transductive policy 中声明。

### 4. 指标、图与域偏移

- 主指标：每目标物理域 RMSE；次级为 worst-family RMSE、MAE、R²、Pearson 和 seed/family 方差。
- KLOGH 另报 log1p RMSE/MAE；不能用 log 域改善掩盖 mD 极端误差。
- 图：逐井深度 GT-pred、预测-真值、残差-真值、残差-深度、worst-family、fold×seed 分布；外部井单独成图。
- 不确定性：同时报 coverage、平均/中位区间宽、按井 coverage、误差-区间宽相关；不得只报 coverage。
- 域偏移：报告每条输入曲线在 train 与 external test 的缺失率、分位数漂移、超出 train range 比例；seismic lane 另报振幅/频谱/采样率漂移。
- 越界：保存未裁剪预测并统计，裁剪只作为明确标注的后处理敏感性分析。

## 十模型同预算评测方法

当前 source lock 中的 10 个 model_id 都纳入“执行清单”，但排名必须按信息模态和预训练状态分 lane；不存在科学有效的全局 1–10 名。

### TabICLv2 官方证据与本地 gate 纠正

| 层次 | 一手证据与精确身份 | 审计结论 |
|---|---|---|
| 代码 | 官方 [`soda-inria/tabicl` README](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/README.md) 在 commit `46b91961db4f8873dd049ec09990698a435e1e29` 明确称项目为 permissive license；根 [`LICENSE`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/LICENSE) 是 BSD-3-Clause（`src/tabicl/forecast` 的衍生代码另列 Apache-2.0，不影响本次 regressor） | TabICLv2 回归代码允许按 BSD-3-Clause 使用；不能再写“代码许可不清” |
| 官方发布 checkpoint | README 的 Available models 表明确列出 `tabicl-regressor-v2-20260212.ckpt`；官方 [`TabICLRegressor` source](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/regressor.py#L88-L101) 将默认来源锁为 Hugging Face `jingang/TabICL`。官方模型仓库快照 [`4dcd344ece2c00be9e831fdd35bed57b5ad83e19`](https://huggingface.co/jingang/TabICL/tree/4dcd344ece2c00be9e831fdd35bed57b5ad83e19) 中的[精确文件](https://huggingface.co/jingang/TabICL/blob/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-regressor-v2-20260212.ckpt)为 114,324,594 bytes，LFS SHA256 `0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a`；[官方 API 元数据](https://huggingface.co/api/models/jingang/TabICL?blobs=true)标记 `private=false`、`gated=false`、模型卡 `license=bsd-3-clause` | 这是可明确识别、公开、非 gated 且许可明确的官方回归权重；不存在需要用相反模型卡证据才能解除的许可阻塞 |
| v2 回归预训练 | README 明确提供三阶段 regressor recipe：[stage 1](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/scripts/train_v2_reg_stage1.sh)、[stage 2](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/scripts/train_v2_reg_stage2.sh)、[stage 3](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/scripts/train_v2_reg_stage3.sh)，使用 quantile regression/pinball loss | 官方提供了回归预训练代码；但 README 同时声明迁移后的 v2 脚本尚未端到端重现原始预训练，故证据级别应区分“官方 checkpoint 可用”和“从零预训练已复现” |
| 本地 source lock 与 cache | 当前 [`source_lock.json`](../../../../_models/property/source_lock.json) 仍登记 `tabicl==2.0.0`、code revision `f719c886…`，却把 weight source 写为 GitHub、`license_status=unconfirmed`、`sha256=null`；当前 [`tabiclv2_regressor.py`](../../../../_models/property/tabiclv2_regressor.py) 又要求显式 approved local checkpoint 且 `allow_auto_download=False`。本次审计机默认 Hugging Face cache 未发现 `jingang/TabICL` snapshot | 本地 gate 已过时，但本报告不能代替代码/source-lock 变更。后续应把 checkpoint 来源、HF snapshot、文件 SHA256 和 BSD-3-Clause 分层登记；本地文件缺失或哈希不符只能记为 `artifact_unavailable`/`integrity_mismatch`，不能再记 `license_unconfirmed` |

因此，TabICLv2 的正确状态是：**官方代码和官方回归 checkpoint 许可已核实；本地 metadata/cache 尚未按官方身份重新登记。** 它应进入独立 `tabular-pretrained` lane，并在显式供应且哈希核验通过后运行正式 development cells；不得与 scratch tabular lane 跨 lane 排名。报告修改不改变当前 runner 的 fail-closed 行为，也不授权自动下载。

| model_id | lane | 资格 | 同预算执行口径 | 排名状态 |
|---|---|---|---|---|
| `catboost_regressor` | tabular-scratch | 合法 | 同一 fold rows、特征、mask、seed；固定 32 estimator/update 配置并记录实际 wall time | 可排名 |
| `lightgbm_regressor` | tabular-scratch | 合法 | 同上 | 可排名 |
| `tabm_regressor` | tabular-scratch | 合法 | 同上；记录实际 optimizer updates，不能把 epoch 与 tree 数假装完全等价 | 可排名 |
| `xgboost_regressor` | tabular-scratch | 合法 | 同上 | 可排名 |
| `extra_trees_regressor` | tabular-scratch | 合法 | 同上 | 可排名 |
| `hist_gradient_boosting_regressor` | tabular-scratch | 合法 | 同上 | 可排名 |
| `realmlp_regressor` | tabular-scratch | 合法 | 同上 | 可排名 |
| `ft_transformer_regressor` | tabular-scratch | 合法 | 同上 | 可排名 |
| `tabiclv2_regressor` | tabular-pretrained | 官方代码/checkpoint 许可已核实；本地 source lock 待按官方 snapshot/hash 重登记 | 使用显式本地 checkpoint，先核对 HF snapshot、文件名及 SHA256；固定相同 target/fold/seed，记录 `n_estimators`、forward 次数、wall time 和显存；不自动下载 | 可产生正式 pretrained-lane 指标；不与 scratch lane 跨 lane 排名 |
| `monai_densenet3d_regressor` | seismic-3D-scratch | 合法但信息模态不同 | 使用相同目标/split/seed 和独立 GPU 时间/显存上限；输入真实 3D patch | 单候选 `not_rankable`；至少再有一个同 lane 候选才排名 |

### 公平预算的具体冻结项

1. 固定目标、development 母井家族、4 个 LOGO folds、3 个 repeat seeds、训练行和输入白名单。
2. 固定为每 cell 32 个注册的 estimator/update 单位，同时设置 lane 内相同 wall-time、CPU 线程、RAM 上限；报告真实更新数和耗时。`32 trees` 与 `32 optimizer steps` 不是等量计算，因此结论必须同时给“固定更新预算榜”和“固定资源预算敏感性”，不能只用一个数字宣称公平。
3. tabular-scratch lane 的每个合法模型都跑 PHIF/KLOGH/SW：理论为 `8 models × 3 targets × 4 folds × 3 seeds = 288` 个合法 cells。
4. TabICLv2 在完成官方 source/version/hash 的本地重登记并供应校验通过的 checkpoint 后，单独产生 `1×3×4×3=36` 个合法 pretrained-lane cells。若本地 artifact 缺失或哈希不符则结构化保留相应运行状态，但不得再使用 `license_unconfirmed`；也不能把 pretrained 结果追加入 scratch 旧榜形成跨 lane 排名。
5. MONAI 产生 `1×3×4×3=36` 个 seismic-lane cells，但只有一模型时只报 pilot 完成率与指标，不授名次。
6. 失败/超时/非有限输出原样保留；合法完成率 `<80%` 的 model-target 不排名。禁止用临时 20% 行切分补 cell。
7. 目标内以 mean physical RMSE 为主排序；依次用 worst-family RMSE、seed 标准差、wall time、model_id 作预注册 tie-break。K 的 log 指标是诊断，不取代物理主榜。
8. leaderboard 只消费 development OOF。F-15 和 fresh external test 均不参与模型排序；final test 只评价冻结的唯一赢家。

## 最小复现实验

### R0：合同与防火墙（秒级，无科学结论）

- 验证 10 个 model_id 可动态发现；依赖缺失、checkpoint 未供应或哈希不符者按真实原因结构化 SKIP。TabICLv2 的官方许可已核实，不再把它归入“权重许可不合法”。
- 构造每目标独立 mask，验证 PHIF/KLOGH/SW 的样本数可不同且不会互相填值。
- 验证 KLOGH `x -> log1p(x) -> expm1(x)` round-trip，负 K fail-loud。
- 验证代码无法解析或加载 F-15/fresh-test path；所有统计 fit-row hashes 只来自 fold-train。
- finite output、shape、checkpoint round-trip 和固定 seed replay。

### R1：真实 development tiny smoke（分钟级，不排名）

- 每个 development 家族抽固定、已登记的小批次，只用训练侧数据。
- 每个可用模型至少执行一次 fit/predict 或 forward/loss/backward/checkpoint。
- 核验真实缺失 mask、输入 shape、物理反变换和未裁剪越界统计。
- MONAI 必须读取真实 seismic patch；tabular 模型不得把 patch 展平特征后混入 tabular 榜，除非另建明确 fusion lane。

### R2：同预算完整 development 排名

- 运行上述 288 个 tabular-scratch cells；固定 folds、3 seeds、32 更新预算和资源上限。
- 输出每目标独立 OOF、逐家族指标、worst-family、完成率、耗时和便携图。
- TabICLv2 在完成本地 source-lock/hash 重登记后运行独立 pretrained lane；MONAI 按 seismic lane 规则处理。任何 artifact 缺失、哈希失败或运行失败均保留真实状态，不伪造数值，也不跨 lane 排名。
- 冻结每目标唯一赢家、配置和 artifact hashes 后停止模型开发。

### R3：外部验证 dry-run 与单次开启

1. 在不读标签的情况下验证外部数据许可、井族、曲线单位、采样、标签版本和输入白名单；生成 manifest hash。
2. 用全部合法 development rows 重拟合冻结赢家；所有 scaler/target transform 仍只用 development。
3. 单次开启 fresh external 标签并生成物理指标、逐井图、域偏移和不确定性报告。
4. 立即把状态改为 consumed；任何后续模型改动都不得复用该集合宣称 fresh blind。

## 对后续外部验证的最低验收

当前最缺的不是更复杂模型，而是独立数据。下一次泛化声明至少同时满足：

- 一个从未被当前仓库历史指标、P4、Stage3、Stage4、Kaggle notebook 或人工调参观察消费的新母井家族；最好再加一个可比油田。
- 明确数据许可证和上游来源，不能用来源不明的 Kaggle 镜像替代。
- 标签来源、单位和深度配准可审计；PHIF/PHIE、解释 K/岩心 K、解释 SW/MICP-SW 分 case。
- 所有井震标定、低频模型、空间插值和标准化遵守 test 防火墙。
- 至少两个 external families 才报告跨井离散度；只有一井时明确写 `single-well external confirmation`。
- frozen winner 一次测试；不因结果不好更换模型、阈值、输入或后处理。

在满足这些条件前，最准确的 completion claim 是：**已完成 Volve 四个 development 母井家族的多 seed LOGO 选择，并在已消费的 F-15 上完成 known-holdout 确认；尚未完成 fresh-blind 或跨油田外部验证。**

## 一手来源清单

1. Helle, H. B., Bhatt, A. & Ursin, B. (2001). *Porosity and permeability prediction from wireline logs using artificial neural networks: a North Sea case study*. [DOI](https://doi.org/10.1046/j.1365-2478.2001.00271.x).
2. Wood, D. A. (2020). *Predicting porosity, permeability and water saturation applying an optimized nearest-neighbour, machine-learning and data-mining network of well-log data*. [DOI](https://doi.org/10.1016/j.petrol.2019.106587).
3. Urang, J. G. et al. (2020). *A new approach for porosity and permeability prediction from well logs using artificial neural network and curve fitting techniques: A case study of Niger Delta, Nigeria*. [DOI](https://doi.org/10.1016/j.jappgeo.2020.104207).
4. Tamoto, H., Gioria, R. dos S. & Carneiro, C. de C. (2023). *Prediction of nuclear magnetic resonance porosity well-logs in a carbonate reservoir using supervised machine learning models*. [DOI](https://doi.org/10.1016/j.petrol.2022.111169).
5. Zhang, J. et al. (2024). *A Machine Learning Method for Predicting Reservoir Porosity and Permeability Based on XGBoost and K-Means*. [Publisher/DOI](https://doi.org/10.3390/app14103956).
6. Davari, M. A. & Kadkhodaie, A. (2024). *Comprehensive input models and machine learning methods to improve permeability prediction*. [Nature/DOI](https://doi.org/10.1038/s41598-024-73846-2).
7. Gohari Nezhad, L. & Emami Niri, M. (2025). *Enhancing water saturation predictions from conventional well logs in a carbonate gas reservoir with a hybrid CNN-LSTM model*. [Springer/DOI](https://doi.org/10.1007/s13202-025-01995-9).
8. Liu, J. et al. (2023). *Porosity prediction from prestack seismic data via deep learning: incorporating a low-frequency porosity model*. [Oxford Academic/DOI](https://doi.org/10.1093/jge/gxad063).
9. Corrales, M., Hoteit, H. & Ravasi, M. (2024). *Seis2Rock: A Data-Driven Approach to Direct Petrophysical Inversion of Pre-Stack Seismic Data*. [Paper DOI](https://doi.org/10.1029/2023EA003301), [official GitHub](https://github.com/DeepWave-KAUST/Seis2Rock), verified repository commit `6996b0e5aebf3a1620c153d7aaa448ddde3dc2e4`, MIT.
10. Zhang, S. et al. (2022). *Petrophysical Regression regarding Porosity, Permeability, and Water Saturation Driven by Logging-Based Ensemble and Transfer Learnings: A Case Study of Sandy-Mud Reservoirs*. [Wiley/DOI](https://doi.org/10.1155/2022/9443955).
