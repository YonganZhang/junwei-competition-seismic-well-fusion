# P5 甜点七目标直接基准调研（2026-07-14）

## 1. 范围、基线与判定口径

- 仓库基线：分支 `p5-stage3-sweetspot`，调研开始时 HEAD 为 `b2fd6aaa2617abc5df986720d7c7167afe903a6b`，工作树 clean。
- 研究对象仅限七个已冻结目标：T1 储层品质、T2 含油气/有效厚度、T3 产能、T4 见水风险、T5 剩余油/加密井潜力、T6 孔隙度、T7 渗透率。
- 纳入来源仅为原始论文、官方数据/竞赛页面、Kaggle 官方页面或作者官方 GitHub。泛化分类、推荐系统、非油气风险预测不计入直接证据。
- “直接证据”要求论文或数据任务的被预测对象与本目标一致；只有相似模型而没有目标对应关系者不纳入。T6 中 NPHI 论文被保留为“孔隙度测井响应”的直接负面对照，但明确不能替代 PHIF 真值。
- 本报告完成了 14 项去重直接证据核验，每个目标 2 项。未发现合格的目标专属 Kaggle 竞赛；只找到一个与 T7 论文对应的作者代码仓库，但仓库没有软件许可证。其余来源本次有界检索未找到作者公开实现。
- 本次只调研并提出合同建议，没有修改现有 label、split、Stage3/4 结果，没有读取 frozen test、训练或复算指标。

## 2. 结论摘要

1. **七目标不能合并成一个甜点标签。** 文献中的 field truth 分属岩心/解释物性、层段试井、未来生产、见水事件、动态监测或数值模拟，观测尺度和因果时间均不同。
2. **当前可保留的是独立目标框架，不是所有当前标签语义。** T1 的 RQI、T2 的 SAND_FLAG、T4 的连续正水量都是可审计 proxy；它们不能分别直接宣称“储层甜点真值”“含烃有效层真值”“水突破真值”。
3. **T3/T4 必须是因果时间任务。** 输入截止到预测时点，目标位于未来固定日历窗口；窗口重叠要 purge，T4 还必须定义右删失。随机行切分或同日未来产量/含水特征会造成泄漏。
4. **T5 目前维持 `not_feasible` 是正确的。** 公开工作通常用数值模拟的未来增量产油/NPV 作为优化代理；实际油田 field truth 更接近特定监测日期的剩余油饱和度，而不是“加密井是否成功”。两类真值不可混合。
5. **T6/T7 可以合法复用物性赛道的合同和原始构建逻辑，但不能复用 test 数据或联合头结论。** 必须从正式原始输入重建 development-only 特征，保留 mother-well split、样本身份和 hash；T6 PHIF 与 T7 KLOGH 继续独立 estimator/head。
6. **当前 Stage4 只能称为 known-holdout confirmation。** T1–T4 holdout 已有 P4 暴露记录，不能作为 fresh-blind 独立 test；T5–T7 没有合法 Stage3 winner。

## 3. 七目标证据覆盖表

| 目标 | 直接一手证据 | 公开竞赛/数据基准 | 作者代码与许可 | 证据覆盖结论 |
|---|---:|---|---|---|
| T1 储层品质 | 2 | 未发现目标专属公开竞赛 | 两项均未找到作者实现 | 有岩心/RQI 分类定义；缺可运行、跨井、概率校准基准 |
| T2 含油气/有效厚度 | 2 | Volve 官方数据可用，但不是带冻结 pay label 的竞赛 | 两项均未找到作者实现 | 有层段生产/MDT 佐证；公开标签生成细节不足，SAND_FLAG 不能替代 pay |
| T3 产能 | 2 | Volve 官方生产数据可用，无目标专属竞赛 | 两项均未找到作者实现 | 有时间序列预测论文；未来窗口、同时刻协变量和 test 复用风险突出 |
| T4 见水风险 | 2 | 未发现目标专属 Kaggle 竞赛 | 两项均未找到作者实现 | 有模拟水突破与现场突破时间；删失、阈值和概率校准普遍不足 |
| T5 剩余油/加密井 | 2 | OLYMPUS 是官方模拟优化挑战，不是 field-truth 竞赛 | 两项均未找到作者实现 | 模拟代理定义清楚；真实 field truth 仅有岩心/监测饱和度证据，经济成功标签空缺 |
| T6 孔隙度 | 2 | Volve 官方数据不含现成冻结 PHIF 竞赛标签 | 两项均未找到作者实现 | 有跨井岩心孔隙度研究；NPHI 预测不能冒充 PHIF，当前 dev-only 特征仍缺 |
| T7 渗透率 | 2 | 未发现目标专属公开竞赛 | 1 个作者仓库，commit 已锁定但无许可证 | 有岩心渗透率及盲井测试证据；当前 dev-only 特征仍缺，外部研究常使用当前合同禁用的 PHIE |
| **合计** | **14** | **合格 Kaggle 竞赛 0** | **作者代码 1；具明确软件许可证 0** | 达到“至少 10 项且每目标独立覆盖”的要求 |

## 4. 直接证据逐项审计

### T1 储层品质

#### T1-E1：Abuamarah & Nabawy (2021)，基于 RQI/FZI 的储层品质分级

- 主源：[A proposed classification for reservoir quality assessment of hydrocarbon-bearing sandstone and carbonate reservoirs](https://doi.org/10.1016/j.jngse.2021.103807)。
- 目标与窗口：静态岩石储层品质，无生产时间窗；输出为 tight/poor/fair/good/very good/excellent 六级品质。
- 标签生成：185 个砂岩和 64 个碳酸盐岩样品，以有效孔隙度、基质气测渗透率、RQI/FZI 和部分 MICP 孔喉信息建立等级。它是岩心尺度静态分类，不是井段生产甜点真值。
- 输入：岩心孔隙度、渗透率及孔喉表征；不是纯测井推理任务。
- split/独立 test：论文是经验分级与跨岩性对比，未报告可复用的井级 train/validation/test 协议。
- 删失/未来泄漏：无因果时间问题；主要风险是把同一岩心推导出的 RQI 与其组成变量同时作为输入和标签。
- loss/指标：非监督训练任务，未定义训练 loss；报告品质等级和岩石物理关系，不含 Top-K、概率校准或独立盲井指标。
- 代码/许可：本次未找到作者代码或数据仓库；论文受出版社许可约束，不能据此推定软件许可。
- 对本项目的含义：支持把 RQI 当作**独立、显式标注为 proxy 的连续物性目标**，但不支持把通用分级阈值直接移植到 Volve 或宣布为综合甜点真值。

#### T1-E2：Zhao et al. (2024)，五井 RQI 储层分类

- 主源：[Approaches of Combining Machine Learning with NMR-Based Pore Structure Characterization to Improve Reservoir Classification in Low-Permeability Sandstones](https://doi.org/10.3390/su16072774)。
- 目标与窗口：静态储层等级；四类标签由 RQI、渗透率、孔隙度及 NMR 孔隙几何共同定义，无生产时间窗。
- 标签生成：论文公布了四类经验范围，例如 Class I 使用较高 RQI/渗透率区间；这些数值来自该研究区，属于外部证据，不是本项目可自行采用的阈值。
- 输入：NMR/常规孔隙结构与岩石物理变量；数据超过 7,000 个样本、来自 5 口井。
- split/独立 test：4 井训练、1 井测试，优于随机深度行切分；但只有一个外部井，区域迁移仍未验证。
- 删失/未来泄漏：无时间删失；若标签定义变量同时进入特征，则存在确定性标签回算风险，部署输入必须排除标签组成量或明确任务是公式复现。
- loss/指标：比较 RF/SVM/XGBoost；主要指标为分类准确率，XGBoost 约 97%。未报告概率校准、Top-K 决策、成本敏感 loss。
- 代码/许可：论文为 CC BY 4.0；本次未找到作者官方代码，数据公开状态不足以直接复现。
- 对本项目的含义：支持 mother-well holdout 和训练区阈值校准；不支持把论文阈值复制为 Volve 标签合同。

### T2 含油气/有效厚度

#### T2-E1：Masoudi et al. (2012)，井测/生产佐证的 productive-zone 分类

- 主源：[Application of Bayesian in determining productive zones by well log data in oil wells](https://doi.org/10.1016/j.petrol.2012.06.028)。
- 目标与窗口：Sarvak 层段 productive zone 分类；标签与层段试井/产量证据关联，而非单独用砂层标志。
- 标签生成：论文把无烃、低于和高于生产门槛的层段区分为不同类别。可公开访问的主源摘要未完整给出门槛数值和层段对齐表，因此不足以直接重建标签。
- 输入：由 CGR、声波、深浅电阻率、中子和密度等常规测井生成的岩石物理特征。
- split/独立 test：训练井内各类随机 70/30，另展示 generalization well；井内随机深度切分仍可能因相邻采样点泄漏而高估表现。
- 删失/未来泄漏：静态任务无未来窗口；真正风险是用目标井投产后信息反向生成历史测井标签，而部署合同未说明是否允许该信息。
- loss/指标：Bayesian likelihood/classification；公开主源未给出可复现实验 loss、完整概率校准或 Top-K 规则。
- 代码/许可：未找到作者代码或公开标签表；出版社论文许可不等于软件许可。
- 对本项目的含义：说明 direct pay 至少需要层段级试井/生产证据；SAND_FLAG 最多是 net-reservoir 辅助标签。

#### T2-E2：Otchere et al. (2022)，Volve net reservoir/pay 分类

- 主源：[Data analytics and Bayesian optimised XGBoost for net reservoir and net pay classification](https://doi.org/10.1016/j.asoc.2022.108680)。
- 目标与窗口：静态 net reservoir/net pay 分类与净厚度解释，无预测时间窗。
- 标签生成：从常规测井推导页岩体积、孔隙度、渗透率和含水饱和度，再以油田特定 cut-off 形成类别，并用新井 mobility/MDT 证据对解释结果作外部核对。公开主源没有提供可直接移植的完整标签表与批准 cut-off。
- 输入：Volve 15/9-F-1 的五类常规测井（论文明确包含声波、GR、中子等）及推导岩石物理量。
- split/独立 test：论文报告 train/test 与新井部署，但公开摘要不足以确认是否先按整井隔离再做全部调参与阈值选择。
- 删失/未来泄漏：无因果时间问题；若以 PHI/Sw/Vsh cut-off 生成 label，则这些变量或其确定性变换不能同时作为普通特征而不声明“规则复现”。MDT 是验证证据，不应在推理时泄漏。
- loss/指标：Bayesian optimization 的 XGBoost，报告 accuracy 0.93、precision 0.94、recall 0.92、F1 0.93；未报告 Brier、校准曲线或固定 Top-K。
- 代码/许可：数据应回到 [Equinor Volve 官方页面](https://www.equinor.com/energy/volve-data-sharing)；本次未找到作者官方实现，论文也未授予软件许可。
- 对本项目的含义：支持“binary pay + net thickness diagnostic”的双输出；反证当前 SAND_FLAG 不能被重命名为含烃有效层。

### T3 产能

#### T3-E1：Ng, Ghahfarokhi & Nait Amar (2022)，Volve 井产量预测

- 主源：[Well production forecast in Volve field: application of rigorous machine learning techniques and metaheuristic algorithm](https://doi.org/10.1016/j.petrol.2021.109468)。
- 目标与窗口：油井产量随时间预测；公开主源未给出与本项目一致的固定 30/90 日 forecast-origin 合同。
- 标签生成：Volve 日生产记录中的油量；不是人工甜点等级。
- 输入：时间、开井小时、井下压力/温度、油嘴、井口压力/温度及气/水日产量等。
- split/独立 test：报告 train/validation/test 和多模型比较，但公开主源没有足够信息证明 test 是完全未参与模型选择的独立未来段或独立井。
- 删失/未来泄漏：把与目标油量同日的气量、水量、开井小时和压力用于“未来预测”会把任务变成 nowcast；只有全部协变量在预测起点前已知时才可用于 forecast。
- loss/指标：SVR/FNN/RNN 与 PSO；报告 train/validation/test 的 R²（测试预测能力大于 0.94，LSTM 最佳）。未报告按井误差、Top-K、区间覆盖或校准。
- 代码/许可：论文页面标为开放许可，具体论文许可不构成代码许可；本次未找到作者官方代码。
- 对本项目的含义：模型家族可作 baseline 参考，但不能为当前 `future-30-calendar-row` 的因果合法性背书。

#### T3-E2：Hosseini & Akilan (2023)，五井顺序切分的产量时间序列

- 主源：[Advanced Deep Regression Models for Forecasting Time Series Oil Production](https://arxiv.org/abs/2308.16105)。
- 目标与窗口：Volve 五口生产井 2008–2016 的油量时间序列；论文以序列预测为目标，但没有冻结成本项目所需的统一日历预测窗。
- 标签生成：官方生产表中的油量序列。
- 输入：每井历史生产序列及论文构造的时间序列窗口；公开实现未提供，无法审计精确 shape 和所有协变量。
- split/独立 test：每井按时间顺序 70% 训练、30% 测试；这优于随机行切分，但没有 leave-one-well-out。文稿流程中用 test 选择最佳模型的表述造成 test 再利用风险。
- 删失/未来泄漏：必须确保窗口只向前看；停井/缺测及边界处样本处理未形成可移植合同。
- loss/指标：LSTM 与 1D CNN；MAE、R²，LSTM 报告 MAE 111.16、R² 0.98。未报告 Top-K、概率/区间校准或独立外部井。
- 代码/许可：论文按 arXiv 页面许可分发；本次未找到作者官方代码或软件许可证。
- 对本项目的含义：支持顺序时间切分，但本项目还需要 prediction-origin、精确日历窗、horizon purge 和 held-well 双重约束。

### T4 见水风险

#### T4-E1：Bai & Tahmasebi (2021)，Egg 模型未来水突破/含水轨迹

- 主源：[Efficient and data-driven prediction of water breakthrough in oil fields](https://doi.org/10.1007/s10596-020-10005-2)。
- 目标与窗口：预测未来含水/水突破轨迹。Egg 合成油藏使用 30 天时间步，约 100 个历史步预测之后 100 个步；这是模拟场景窗口，不能直接成为 Volve 阈值。
- 标签生成：100 个地质实现、8 注 4 采、400 种注入方案的数值模拟轨迹，包括含水饱和度、电阻率、油量和水量/含水等动态量。
- 输入：固定长度历史动态序列；论文比较输入窗长度，最优示例约 10 个时间步。
- split/独立 test：前 90% 时间点用于训练、其余 10% 未来段测试，并从训练段抽验证集；另报告 84 个预测起点后发生突破的测试案例。
- 删失/未来泄漏：它只分析最终发生突破的模拟案例，不能自然代表现场右删失井；归一化必须仅 fit 训练段。随机抽训练段验证会弱化时间外推检验。
- loss/指标：ANN/RNN/GRU/LSTM，报告 MAPE/MAE/RMSE；deep LSTM 示例 MAPE 2.52、MAE 0.31、RMSE 0.42。没有事件概率 Brier、校准或 survival 指标。
- 代码/许可：本次未找到作者代码；论文受 Springer 出版许可约束，模拟数据也未作为带软件许可的数据包发布。
- 对本项目的含义：支持把 T4 作为未来轨迹/事件问题，不支持把“连续 7 天水量大于零”直接当作通用 field breakthrough。

#### T4-E2：Kamari et al. (2014)，裂缝性油藏见水时间预测

- 主源：[Prediction of breakthrough time of water coning in fractured reservoirs by a new approach](https://doi.org/10.1016/j.fuel.2013.09.071)。
- 目标与窗口：北波斯湾现场裂缝性油藏的 water-coning breakthrough time，是连续 time-to-event，而不是固定 30 日二分类。
- 标签生成：由实际油田见水时刻/生产资料形成突破时间。可访问主源没有披露可逐样本重建的事件阈值、持续期和观察截止规则。
- 输入：井/油藏/生产相关描述量；公开主源不足以核验完整字段列表和推理时可用性。
- split/独立 test：比较 LSSVM、ANN 和 HFKGA；公开主源未充分报告井级/因果时间 split，因而不能确认真正独立 test。
- 删失/未来泄漏：未明确处理观察期内未见水井的右删失；若只保留已发生事件井，会产生选择偏差。
- loss/指标：主要报告 AARD% 等点预测误差；没有 Brier、校准曲线、time-dependent C-index 或 integrated Brier score，也没有 Top-K 干预规则。
- 代码/许可：未找到作者代码或公开标签数据；无软件许可证。
- 对本项目的含义：直接支持为 T4 增加 survival lane；若继续固定窗分类，也必须由领域专家批准见水阈值、持续期和删失规则。

### T5 剩余油/加密井潜力

#### T5-E1：Chu et al. (2020)，模拟场加密井位置优化

- 主源：[Determination of an infill well placement using data-driven multimodal convolutional neural network](https://doi.org/10.1016/j.petrol.2019.106805)。
- 目标与窗口：已有井生产 10 年后选择垂直加密井位置，预测之后 10 年累计产油；10 年是该模拟实验设定，不是本项目可自行采用的窗口。
- 标签生成：修改的 SPE10 河道化合成油藏。504 个候选情景经全物理模拟得到未来累计油量，搜索空间约 13,200 个位置；因此标签是 simulator proxy，不是现场成败真值。
- 输入：候选点附近的 3D 渗透率、孔隙度、压力、含油饱和度四模态体。
- split/独立 test：10-fold CV；论文没有真实油田或独立模拟器/地质分布外 test。
- 删失/未来泄漏：模拟状态必须冻结在决策时点；压力和含油饱和度若来自评价期之后会泄漏。训练/验证候选空间相邻时还会有强空间相关。
- loss/指标：CNN 训练 1,000 epochs，以平方误差/RMSE 类目标优化；报告 RMSE、R²。四模态示例 RMSE 0.261 MMSTB、R² 0.725。
- Top-K/校准：用模型筛 top-20 候选，再用全物理模拟复核；这是有价值的决策协议。未报告概率校准，最终可信度来自 simulator rerun。
- 代码/许可：未找到作者官方代码/数据；无可复用软件许可证。
- 对本项目的含义：T5 可行的第一条路径是**明确标记 simulation-only** 的增量产油/NPV 排序，而不是制造 field label。

#### T5-E2：Roueche & Karacan (2018)，水驱后剩余油带/饱和度

- 主源：[USGS: Zone identification and oil saturation prediction in a waterflooded field—Residual oil zone, East Seminole Field](https://www.usgs.gov/publications/zone-identification-and-oil-saturation-prediction-a-waterflooded-field-residual-oil)。
- 目标与窗口：特定监测/取心时点的 main pay、residual-oil zone 和原位含油饱和度；没有未来加密井经济窗口。
- 标签生成：三口井岩心、测井和饱和度资料建立分区及深度上的概率性原位含油饱和度，是少见的现场观测 truth，但只覆盖取心/解释位置。
- 输入：井日志、岩心及饱和度相关变量；ANN expert system 和 CART 用于分区/饱和度预测。
- split/独立 test：官方摘要报告训练/测试相关性，但未公开可审计的整井隔离和空间 buffer 细节。
- 删失/未来泄漏：没有事件删失；风险是把只在打井后取得的岩心/饱和度信息当作规划期可用输入。未取心区不能自动当负样本。
- loss/指标：zone identification 成功率超过 90%；含油饱和度相关系数约为 test 0.6、train 0.8。无 Top-K 加密井 regret、经济指标或概率校准。
- 代码/许可：USGS 官方出版物；本次未找到对应代码/可训练数据包。出版物可公开访问不等于输入数据具有统一软件/数据许可。
- 对本项目的含义：真实 field truth 可定义为“评价时点的剩余油饱和度/ROZ 概率”，但不能据此生成“加密井成功”标签。

### T6 孔隙度

#### T6-E1：He et al. (2025)，72 井岩心孔隙度预测

- 主源：[Porosity prediction of tight reservoir rock based on machine learning](https://www.nature.com/articles/s41598-025-95578-7)。
- 目标与窗口：静态岩心/解释孔隙度，无生产时间窗；105,411 个样本、72 口鄂尔多斯探评井。
- 标签生成：实测/岩心孔隙度与井深对齐，不是 NPHI 曲线本身。
- 输入：AC、CAL、CNL、DEN、GR、RT、SP 等测井。
- split/独立 test：先进行样本级约 80/20 切分，并增加两口未见井的验证。盲井比随机深度 split 更可信，但模型/超参数是否完全未见盲井仍需代码核验。
- 删失/未来泄漏：无时间问题；同井相邻深度随机切分会泄漏层序/井况。标准化、特征选择和 PSO 必须仅 fit fold-train。
- loss/指标：GBDT/RF/XGBoost/MLP 与 PSO，主要 R²；未报告统一物理 MAE、置信区间覆盖、Top-K。
- 代码/许可：论文为 CC BY-NC-ND 4.0；数据需向 PetroChina No.12 申请，本次未找到作者代码。
- 对本项目的含义：支持整井外推和原始测井输入；不能解除当前 portable development-only feature source 的阻塞。

#### T6-E2：Kanfar et al. (2020)，Volve 钻井参数预测 NPHI

- 主源：[Real-Time Well Log Prediction From Drilling Data Using Deep Learning](https://arxiv.org/abs/2001.10156)。
- 目标与窗口：实时预测 NPHI、密度和声波测井响应；NPHI 是中子孔隙度响应，**不是当前 T6 冻结的 PHIF 标签**。
- 标签生成：Volve 12 井的目标测井曲线，共约 89,549 个深度点。
- 输入：深度、ROP、WOB、流量、MSE 等钻井变量；增广后的示例窗口 shape 为 `(16680, 50, 5)`。
- split/独立 test：约 5% 窗口作测试；允许窗口重叠，只避免完整重复，没有 held-well test。
- 删失/未来泄漏：无生产时间；滑窗随机切分会让相邻、部分重叠窗口跨 split，导致强空间/序列泄漏。目标井盲测未完成。
- loss/指标：CNN/CNN-TCN 使用 MSE；论文报告 NPHI 测试 MSE 约 0.11/0.40、相关系数约 0.60/0.47。无物理 PHIF MAE、校准或外井 test。
- 代码/许可：论文按 arXiv 页面许可分发；本次未找到作者官方代码或软件许可证。
- 对本项目的含义：可参考实时钻井输入形式，但不能把 NPHI 结果复用为 PHIF 标签或指标。

### T7 渗透率

#### T7-E1：Matinkia et al. (2022/2023)，测井到岩心渗透率的混合 MLP

- 主源：[Prediction of permeability from well logs using a new hybrid machine learning algorithm](https://doi.org/10.1016/j.petlm.2022.03.003)；作者仓库：[mmehrad1986/Hybrid-MLP](https://github.com/mmehrad1986/Hybrid-MLP)，核验 commit `80830983a106619d8152bf1c161218dff447a45e`。
- 目标与窗口：伊朗 Fahlian 储层静态岩心渗透率，无生产时间窗。
- 标签生成：岩心渗透率与测井深度配准。
- 输入：常规岩石物理测井；MLP 结合 SSD，并与 PSO/GA 优化比较。
- split/独立 test：论文报告 test R² 0.9928，但公开主源不足以证明整井盲测；随机相邻深度切分风险高。
- 删失/未来泄漏：无时间删失；若孔隙度/解释渗透率等派生曲线进入输入，会与当前 T7 推理期 whitelist 冲突。
- loss/指标：MLP 回归误差，报告 R² 等点预测指标；未报告 mD 尺度 MAE、log/physical 双尺度误差、校准或外部井。
- 代码/许可：仓库只含两个 RAR 归档，未见 `LICENSE`；因此是 source-available，不是获准复用的软件。论文的开放许可不能替代仓库软件许可证。
- 对本项目的含义：能复现的门槛仍不满足；若未来联系作者取得许可，也必须改为 mother-well split 并遵守 T7 禁用目标派生输入。

#### T7-E2：Davari & Kadkhodaie (2024)，多组合渗透率模型与盲井

- 主源：[Comprehensive input models and machine learning methods to improve permeability prediction](https://www.nature.com/articles/s41598-024-73846-2)。
- 目标与窗口：静态岩心渗透率，无生产时间窗。
- 标签生成：岩心渗透率与井日志配准；比较 57 个输入组合和 ELM/RF/GB/KNN/MLP，共 285 个预测配置。
- 输入：GR、RT、PHIE、RHO、DT、NPHI 等。注意 PHIE 在当前 T7 推理合同中若属于目标/解释派生量，则不能直接照搬。
- split/独立 test：A/B 井内随机 70/20/10 train/test/validation，另用完全未参与建模的 C 井作 blind test；C 井 GB R² 约 0.92、RF 约 0.90。
- 删失/未来泄漏：无时间问题；A/B 随机深度切分有邻接泄漏，但 C 井盲测提供较强外推证据。所有 57 组合的选择若看过 C 井会削弱其独立性，论文未提供代码审计。
- loss/指标：回归训练，报告 R²、RMSE；未报告 mD MAE、log1p 误差、分层校准或 Top-K。
- 代码/许可：Scientific Reports 开放论文；数据需经 IRD/作者许可申请，本次未找到作者代码。
- 对本项目的含义：支持保留盲井评测和物理尺度指标；不支持把 PHIE 作为 T7 推理输入，也不解除 dev-only 构建阻塞。

## 5. 官方数据、竞赛、Kaggle 与代码审计

### 5.1 官方数据与模拟挑战

- [Equinor Volve Data Sharing](https://www.equinor.com/energy/volve-data-sharing) 是 Volve 的首选数据源，覆盖约 40,000 个文件及 2008–2016 生产历史，并附 Open Data Licence。第三方 Kaggle 副本不能替代其来源 hash、授权条件和成员清单。
- [OLYMPUS Field Development Optimization Challenge](https://www.isapp2.com/optimization-challenge/optimization-challenge-download-files.html) 是 TNO/ISAPP 的合成油田开发优化挑战。它可支持 T5 的 simulation-only 路线，但不是现场剩余油/加密井 field truth；下载需登记，文件再分发受原始说明和版权头约束。

### 5.2 Kaggle 检索结论

本次对七个目标分别检索 Kaggle，未找到同时满足“目标直接对应、标签生成可审计、split/独立 test 明确、官方竞赛或官方作者发布”的合格竞赛。发现的条目主要是用户转存 Volve 生产/钻井日志、无冻结 label 的 well-log 表或合成油藏表。它们不计入 14 项证据，也不应用于替代 Equinor 官方源。

### 5.3 作者代码与许可结论

- 唯一找到并能锁定版本的作者仓库是 T7-E1 的 `Hybrid-MLP`，但只含 RAR，且没有 `LICENSE`，所以不能列为“许可可复用”。
- T1、T2、T3、T4、T5、T6 均未找到与纳入论文一一对应、由作者维护且带软件许可的公开实现。
- “论文开放获取”“官方数据可下载”“GitHub 可浏览”是三种不同权利状态；只有明确软件/数据许可证才能支持代码或数据再分发。

## 6. 推荐标签合同（仅建议，不改变当前 spec）

以下合同均要求 `approved=true`、批准人、版本、源 hash、允许字段、fit 域和 split 完整后才可 build。本文不选择任何新阈值。

### T1 储层品质

- **推荐真值**：优先连续 RQI/FZI 或经岩心校准的连续品质量；可选等级输出必须由领域专家基于 Volve/目标区训练井批准阈值。
- **当前 proxy 的位置**：`RQI=0.0314*sqrt(KLOGH/PHIF)` 可保留为 `reservoir_quality_proxy`，明确单位和公式版本；不得同时把 PHIF/KLOGH 或其确定性变换作为普通推理特征。
- **尺度与负样本**：岩心/解释深度点或批准井段；未解释区为 unlabeled，不是差储层。
- **评测**：整井 holdout 的 MAE/RMSE/Spearman、每井误差；只有实际存在优选井段决策时才增加预先冻结的 Top-K recall/regret。

### T2 含油气/有效厚度

- **推荐真值**：层段级 `pay / non-pay / unlabeled`，由批准的 PHI/Vsh/Sw 规则与 MDT/DST/试井/投产证据共同生成；同时输出每井/层的 net-pay thickness。
- **当前 proxy 的位置**：SAND_FLAG 只能保留为 `net_reservoir_proxy`，不得称为含烃 pay。
- **负样本与模糊样本**：只有被同等观测机制证实的 non-pay 才是负样本；无 MDT、无试井或未测井层段为 unlabeled。
- **评测**：held-well PR-AUC、Brier、F1（阈值仅由 fold-train/OOF 冻结）和净厚度绝对误差；报告 calibration curve。

### T3 产能

- **推荐真值**：批准 prediction origin `t0` 后精确 30 个日历日、90 日或 PI/累产之一；当前 30 日语义若保留，必须从“calendar-row”改为确切日历时间并冻结停井/缺测规则。
- **允许输入**：时间戳 `<=t0` 的静态地质、完井和历史生产；同日/未来油气水量、未来开井小时、未来注入和后验修井信息禁止。
- **删失**：观测尾部不满 horizon 的样本删失/排除，不得补零；停井是否计零由合同批准。
- **评测**：held-well + rolling-origin，训练窗和目标窗间按 horizon purge；MAE/RMSE/Spearman、井组误差和预先冻结的 Top-K 产能命中/后悔值。

### T4 见水风险

- **推荐真值 A**：在 `t0` 后固定窗口内，water cut 超过领域批准阈值并持续批准天数的事件概率。
- **推荐真值 B**：从 `t0` 到首次持续突破的 time-to-event，并记录右删失时间。A/B 是独立任务，不混合排行榜。
- **当前 proxy 的位置**：首次连续 7 个正水量日可作为 `positive-water proxy`；“大于零”的仪器噪声敏感定义不得冒充正式突破。
- **评测**：分类用 AP/Brier/校准/F1；survival 用 time-dependent C-index、integrated Brier score 和 horizon calibration。只用 `t0` 前历史并做 temporal purge/held-well test。

### T5 剩余油/加密井潜力

- **合同 A（field residual-oil）**：冻结监测日期，以岩心/饱和度测井/4D 地震/历史匹配后的剩余油饱和度或 ROZ 概率为标签；只在被观测位置监督，未观测区域为 unlabeled。
- **合同 B（simulation infill proxy）**：冻结模拟器与版本、deck/realization hash、历史截止时点、候选井轨迹、井距/作业/经济约束、评价 horizon 和 no-infill baseline；标签为增量累产或增量 NPV，并永久标记 `simulation_proxy=true`。
- **split**：A 按井/空间块及监测时间隔离；B 按地质 realization 和候选空间块隔离，禁止相邻候选跨 split。
- **评测**：A 用饱和度误差/空间校准；B 用 Top-K hit/regret、rank correlation，最终候选必须由未用于训练的全物理模拟 rerun。两者都不能被称为现场加密井成功，除非有独立未来钻后结果。

### T6 孔隙度

- **推荐真值**：保留当前已冻结的 PHIF/PHIE 明确版本，本项目当前为 `target6-phif-cpi-v1`；NPHI 不能替代 PHIF。
- **合法复用**：只复用物性赛道的正式原始数据 builder、样本 ID、source hash 和 mother-well split，从 development 原始输入重建特征；禁止 `test.h5` 回填 development。
- **独立性**：T6 使用独立 TaskSpec/estimator/head；预处理只 fit fold-train。
- **评测**：每井 MAE/RMSE/R²、残差随深度/井分布及预测区间覆盖；fresh test 必须是未参与 P4/P5 选择的新井或新数据。

### T7 渗透率

- **推荐真值**：保留 `target7-klogh-cpi-v1`，物理单位 mD；训练可用 fold-train 确定的 `log1p(KLOGH_mD)`，报告时还原物理尺度。
- **合法复用**：与 T6 相同，只从正式 raw development 源重建；不复用 test、联合多输出结论或看过 test 的 checkpoint。
- **允许输入**：遵守当前推理 whitelist；外部论文常用 PHIE，但若合同禁止目标/解释派生曲线，就不能因文献高分放宽边界。
- **评测**：log 尺度 MAE/RMSE/R² + 物理 mD MAE/分位误差，按井报告；T7 与 T6 不共享 head 或排名。

## 7. 统一训练、验证与测试协议

1. **合同先于数据。** 每目标独立版本化 label spec，列明语义、输出类型、公式/阈值、fit 域、时间窗、空间尺度、正/负/unlabeled、推理输入、指标、批准人和源 hash；未批准 fail-closed。
2. **样本身份先于特征。** 先以 mother-well/井段/时间戳生成不可变 sample ID，再做任何窗口、标准化、重采样和特征选择。
3. **空间任务按整井隔离。** T1/T2/T6/T7 使用 frozen mother-well split；同层相邻深度点不得随机跨 fold。若是网格/候选井任务，增加空间 buffer。
4. **动态任务按因果时间隔离。** T3/T4 定义 prediction origin，所有特征时间 `<=t0`，目标严格位于未来；按 horizon purge 重叠窗口，并同时报告 held-well 与 rolling-origin 结果。
5. **fold-train-only fit。** 缺失填补、缩放、类别权重、target transform、阈值、概率校准和 early stopping 均只能 fit fold-train；OOF 后才冻结部署阈值。
6. **删失与观测机制显式化。** T2 未测层段、T4 未发生但观察结束、T5 未取心区域、T3 不满 horizon 都不是自动负样本或零值。
7. **任务独立。** 七目标各自 estimator/head、metrics、leaderboard 和图件，不输出未经批准的综合甜点分数。
8. **test firewall。** 当前 Stage4 的 T1–T4 是 `previously_seen_reusable_holdout`，只可作确认性结果；不得用于阈值、模型或合同选择。新的 final claim 需要 fresh-blind 外井/外期。T5–T7 当前不得访问 test。
9. **可审计产物。** 记录代码 commit、label/split/source/config hash、依赖和 seed；保存紧凑 OOF/预测 hash，而非把大 checkpoint 纳入 Git。

## 8. 对当前 Stage3/4 的逐目标去留

| 目标 | 当前状态 | 结论 | 必须修改/保留的边界 |
|---|---|---|---|
| T1 | Stage3 RQI proxy 可排名；Stage4 LightGBM known holdout | **修改** | 保留 RQI proxy lane 和独立回归；废弃“等于综合储层甜点真值”的表述。新阈值须批准，新 test 须 fresh-blind |
| T2 | Stage3 SAND_FLAG near-binary；Stage4 CatBoost known holdout | **废弃当前 direct-pay 解释，保留辅助 proxy** | 不能用近乎确定性的 SAND_FLAG 高分宣称含烃 pay；待 MDT/DST/Sw/试井合同后重建 direct label，并同时评净厚度 |
| T3 | Stage3 future-30-calendar-row；Stage4 XGBoost known holdout | **修改** | 模型 smoke 可保留；标签改成精确日历窗，冻结停井/缺测/删失，输入只到 `t0`，双重 held-well + rolling-origin |
| T4 | Stage3 7 日正水 proxy；Stage4 CatBoost known holdout | **修改，正式风险解释暂废弃** | proxy lane 可留；正式任务需 water-cut/持续期或 survival 合同、右删失和校准。当前确认结果不能作为 fresh test |
| T5 | Stage3/4 `not_feasible` | **保留** | 在 field residual-oil 或 simulation infill 合同二选一获批、解析器与时点/经济/候选约束冻结前，不造标签 |
| T6 | PHIF label/version 冻结；P5 因无 dev-only 特征源 blocked | **保留合同与 blocked** | 可复用物性赛道 raw builder/split/hash；禁止用 `test.h5` 回填 development，独立 T6 head |
| T7 | KLOGH label/version 冻结；P5 因无 dev-only 特征源 blocked | **保留合同与 blocked** | 同 T6；log 与物理尺度双报告，禁止引入当前合同不允许的 PHIF/PHIE 派生输入 |

补充：Stage4 已记录的 T1–T4 模型分别为 LightGBM、CatBoost、XGBoost、CatBoost，但其 holdout 均已有 P4 暴露，证据等级只能是 known-holdout confirmation。T2 的极高分类分数更像 SAND_FLAG 规则复现；T4 的低 AP/高 Brier 也提示当前 proxy 与可校准见水风险之间仍有明显语义差距。本文不复算这些数值，也不据此重选模型。

## 9. 关键缺口

1. **T1**：缺 Volve/目标区岩心标定的品质阈值、跨井概率校准和具软件许可的实现。
2. **T2**：缺深度对齐的 MDT/DST/试井/投产层段真值与批准 cut-off；官方 Volve 数据可用不等于存在公开 pay label。
3. **T3**：缺冻结 prediction origin、精确日历 horizon、停井/缺测规则和未见外井/外期 test。
4. **T4**：缺批准的 water-cut 阈值/持续期、观察截止和右删失；公开论文多为模拟或只纳入已发生事件。
5. **T5**：缺 field residual-oil 监测数据或获批模拟器解析链、决策时点、候选井/井距/经济约束。公开模拟代理不能冒充现场真值。
6. **T6/T7**：P4 合同与 split 已有，但 P5 当前没有便携、可 hash 的 development-only 特征包；`test.h5` 明确不是替代物。
7. **复现与许可**：14 项中只有 1 项找到作者代码且仍无仓库许可；没有合格的公开 Kaggle 竞赛可提供冻结 blind leaderboard。

## 10. 最小复现实验（下一步建议，不在本次执行）

| 实验 | 前置条件 | 最小设计 | 成功判据 |
|---|---|---|---|
| T1 proxy 基线 | 当前批准 RQI proxy 不变 | development-only LOGO；LightGBM/线性基线；每井 MAE/Spearman | 无目标组成量泄漏，fold/hash 可复建；只称 proxy |
| T2 direct-pay 对照 | 领域专家批准 pay/negative/unlabeled 与 MDT/DST 对齐 | held-well；SAND_FLAG 仅作 auxiliary；输出概率和 net thickness | PR/Brier/F1 与厚度误差均可审计，校准只 fit OOF |
| T3 因果 30 日 | 批准精确日历窗口、停井与删失规则 | 1 个 held well + rolling origins；LightGBM/XGBoost；horizon purge | 所有特征 `<=t0`，外期 MAE/Spearman/Top-K 可复跑 |
| T4 分类/生存双 lane | 批准事件阈值/持续期/观察截止 | 相同 prediction origins；分类与 survival 分榜；右删失 | AP/Brier/calibration 或 C-index/IBS 完整，不丢 censored wells |
| T5 simulation smoke | 批准模拟器、deck/realization、时点、候选与经济约束并具解析器 | 少量 realization × 候选；预测增量油/NPV；top candidates simulator rerun | 明确 `simulation_proxy=true`，held-realization Top-K regret 可重建 |
| T6 PHIF | 从官方 raw 重建 development-only bundle | 复用 frozen mother-well split/sample ID；独立轻量回归 | source/bundle/split hash 一致；全程不读 test |
| T7 KLOGH | 同 T6，保持独立 head | fold-train `log1p` transform；报告 log + mD 指标 | 不用 PHIF/PHIE 等禁用派生输入；全程不读 test |

## 11. 最终研究判定

- 14 项直接一手证据达标，但“公开可运行代码”远弱于“论文有结果”；不能把论文分数直接转成项目可复现基准。
- T1–T4 目前都有真实 development 实验基础，但只有 T1 的 RQI、T2 的 SAND_FLAG、T4 的正水事件 proxy 语义；T3 还需把行窗口收紧为真正因果日历窗口。Stage4 全部是已见 holdout，不是 blind test。
- T5 继续 `not_feasible`，直到 field 与 simulation 两种真值路径之一被明确批准。
- T6/T7 的标签合同可合法复用物性赛道，数据边界不能复用：必须重建 development-only 特征，严禁读取 `test.h5` 回填。
- 因此下一步不是再训练，而是由军伟/领域专家批准 T1 阈值语义、T2 pay 证据、T3 时间/停井规则、T4 事件与删失、T5 field/simulation 路线，并为 T6/T7 提供可审计 development-only 原始构建入口。
