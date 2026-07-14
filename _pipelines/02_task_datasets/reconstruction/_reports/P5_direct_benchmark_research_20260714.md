# P5 三维地质/储层属性体直接基准调研

日期：2026-07-14

审计基线：`e783066f817c5ae79ee279479e8f699d7fcd38cb`

赛道：⑥ 三维模型重建（reconstruction）

范围：只纳入“由稀疏井、井震约束或部分地震属性重建三维地质/储层属性体”的工作；普通图像修复、通用 3D 生成、无地学约束插值不计入直接基准。
证据口径：论文结论优先原始论文/出版社页；数据与代码状态优先官方数据页、作者仓库和仓库许可证。网页均于 2026-07-14 核对。

## 1. 执行结论

1. 当前将任务拆成 `strict` 与 `conditional` 两条独立 lane 是必要且科学的。`strict` 必须禁止测试空间块的目标、井值及其派生量；`conditional` 可以在推理时接收声明过的测试区井约束，但必须排除精确井格点，并明确它不是严格空间泛化。
2. 当前 `strict` 的连续 I-block 留出、I-block guard、development 内 K-block 五折与一块 buffer，方向上优于文献中常见的随机点切分或仅留一口井。不过，这一 holdout 已被 P4/P5 历史流程消费，只能称“已见留出复核”，不能再称 fresh blind test。
3. 当前 `conditional` 的精确井格点排除是正确的，但它和 `strict` 使用不同物理测试块，不能直接用两者分数差证明“加入测试区井约束有效”。应在同一测试块、同一训练数据、同一模型下做 paired no-well / with-well 消融，并按距井距离分层报告。
4. 当前胜者 `pykrige_ok3d` 是合法而重要的地统计基线，但实际实现只读三维坐标。它忽略地震属性，也忽略 conditional 的 IDW 井约束特征；因此应命名为“坐标普通克里金/开发区标签插值基线”，不能作为“井震融合模型”或“conditional 井约束收益”的证据。
5. 当前严格留出的 `R²=-0.3397` 是有效失败信号：SSE 比“用该测试块真值均值作常数预测”的 SSE 高 33.97%。结合 Pearson `-0.1200` 与正偏差 `+0.0175`，不能解释为仅有尺度偏移；该结果不支持严格空间泛化主张。
6. 未核实到与本任务等价的 Kaggle 竞赛。F3 seismic 切片、Volve 单井 CSV 和 FORCE 1D 岩性任务都缺少“稀疏井/地震输入—对齐 3D 属性真值—空间盲测”三件套，不应列为直接 benchmark。

## 2. 本报告的操作性定义

### 2.1 strict：严格空间泛化

- 先冻结连续空间测试块或外部区块，再做任何预处理、CV、HPO 和模型选择。
- 测试块内的目标体、井约束、由目标采样得到的井值、未来量和任何派生真值均不得进入输入。
- 训练期只访问 development；所有归一化、变差函数/核参数、类别权重、校准均只 fit fold-train。
- development CV 必须按空间组切分，并用与空间相关长度相称的 buffer/purge。
- frozen test 只允许一次最终消费。若已历史消费，必须标为 `previously_seen_reusable_holdout`，而不是 fresh blind。

### 2.2 conditional：给定测试区井约束的条件重建

- 推理时允许测试区已声明、现实中可测的井值进入模型；不得引入井外测试真值。
- 精确井格点必须从主指标中排除，避免硬条件点的恒等命中抬高分数。
- 仍需报告距井误差带，因为 IDW/kriging 会把井值传播到附近格点。
- 最有解释力的设计是在同一空间测试块上做 `no_test_well` 与 `with_test_well` 配对消融；它衡量条件信息的增益，而不是跨区严格泛化。

### 2.3 直接性等级

- **D3（直接）**：输出三维储层/地质属性体，输入明确包含井约束、井震约束或三维/部分地震属性。
- **D2（近直接）**：输出三维体，但验证只在井位/合成 realization，或输入条件与本项目并不完全一致。
- **不纳入**：二维普通图像任务、通用 3D 生成、仅 1D 井曲线预测、没有地学条件或没有三维目标体。

## 3. Volve 真值与数据血缘

Equinor 的[官方 Volve 数据共享页](https://www.equinor.com/energy/volve-data-sharing)说明该开放数据覆盖完整的地下与生产资料；其[官方目录/许可说明 PDF](https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf)明确列出 Eclipse 动态模型和 RMS 静态地质模型。两者支持本项目把 Eclipse porosity 作为真实工程参考体，而不是合成图片。

但这里的“ground truth”应理解为**参考地质/数模解释结果**，不是不可争议的原位真值：RMS/Eclipse 本身由地震、井和地质解释构建。因而协议必须记录标签来源、模型版本与网格转换，避免把参考模型对输入资料的既有利用误称为独立物理真值。

## 4. 十项直接工作证据矩阵

下表十项均有原始论文或作者/官方实现作为一手来源；没有仅凭二手综述收录的候选。`未报告` 表示本次可访问的一手材料没有足够信息，不能据常规做法猜测。

| ID | 一手来源与直接性 | 目标体与约束 | strict / conditional 映射 | split、buffer 与盲测 | 方法、loss 与指标 | 代码与许可 |
|---|---|---|---|---|---|---|
| B01 | Yao & Journel (2000)，[原始论文](https://www.sciencedirect.com/science/article/pii/S0920410500000681)，DOI `10.1016/S0920-4105(00)00068-1`；**D3**，问题表述与本赛道最接近的经典井震融合之一 | 西德州碳酸盐储层 3D 孔隙度；硬井孔隙度 + 垂向分辨率较低的 2D 地震属性图 | **conditional**：输出显式条件于硬井数据和软地震信息；不是 strict | 论文公开摘要未给空间块 holdout、buffer 或独立盲测 | 先估计 2D 垂向平均孔隙度，再用 block kriging 与 direct sequential simulation 构造 3D；约束是重建体的垂向平均精确复现 2D 条件；未见 ML loss、RMSE/SSIM | 未核实作者代码；代码许可不适用/未知 |
| B02 | Leite & Vidal (2011)，[出版社论文](https://www.sciencedirect.com/science/article/pii/S0098300410002682)，DOI `10.1016/j.cageo.2010.08.001`；**D3** | post-stack 3D 振幅经稀疏脉冲/递归反演得到阻抗；井密度、声波、GR 约束；输出 3D 有效孔隙度 | 工作流整体为**井震条件预测**；并未定义本报告意义的 strict/conditional 双 lane | NN 在井位训练、验证、测试，随后应用到全体；未证实连续空间块或 buffer；是否真正盲井需保守记为未报告 | 低频阻抗由井日志 kriging，feed-forward NN 学习 GR+阻抗到孔隙度；公开材料未披露可复核 loss、体素 RMSE/SSIM/频谱/变差函数 | 未核实作者代码；许可未知 |
| B03 | Chaki et al. (2014)，[作者预印本](https://arxiv.org/abs/1509.07079)，DOI `10.1016/j.petrol.2014.06.019`；**D3** | 3D 地震阻抗、瞬时振幅、瞬时频率 + 8 口井；输出全深度砂岩比例并可扩展到体 | 训练时不使用被留井目标，属 **strict-like blind-well**；不是测试空间块 strict，也不是测试井条件 conditional | 7 井训练，1 井测试；按 well tops 分 3 个 zone；未报告空间 buffer | 分区 modular ANN；公开材料列 CC、RMSE、绝对误差均值和耗时；无 SSIM/频谱/变差函数 | 预印本文本可读；未核实可运行官方代码，代码许可未知 |
| B04 | Verma et al. (2014)，[原始论文](https://www.sciencedirect.com/science/article/pii/S0926985114002912)及[作者预印本](https://arxiv.org/abs/1509.07074)，DOI `10.1016/j.jappgeo.2014.10.005`；**D3** | 3D post-stack 阻抗、振幅、瞬时频率 + 6 口井；输出砂岩比例体 | 井震监督预测；公开材料只说明 unseen validation at well control，无法确认 strict block；不是测试区井条件生成 | train/test/validation 细节和 buffer 在公开摘要中不足；不能据此视为独立空间盲测 | 三类 ANFIS 与 ANN；指标 CC、RMSE、AEM、scatter index；无频谱/变差函数/结构连通性 | 未核实官方代码；许可未知 |
| B05 | Azevedo, Grana & Amaro (2019)，[期刊原文](https://academic.oup.com/gji/article/216/3/1728/5222649)及[作者预印本](https://arxiv.org/abs/1810.06552)，DOI `10.1093/gji/ggy511`；**D3** | 真实北海 3D 体；井日志 + pre/partial-stack 反射 + 岩石物理；联合输出孔隙度、页岩体积、流体饱和度和相 | 属**条件地统计反演**，不是 strict 学习式 holdout | 与另一 geostatistical AVA inversion 比较；未报告连续空间块 blind test 或 buffer | stochastic sequential simulation 扰动，facies-dependent rock physics，cross-over genetic optimizer；评价强调地震匹配、属性/相和不确定性，未给本项目式统一 RMSE/SSIM | 未核实随文官方代码；许可未知 |
| B06 | Jo et al. (2021)，[原始预印本](https://arxiv.org/abs/2111.13581)；**D3（合成 realization）** | 3 个 10/20/30 Hz 谱分解地震体输入，输出 `64×64×32` 3D 孔隙度；训练 realization 的孔隙度生成过程受井数据约束，但推理不接收新测试井 | **strict-like across realizations**，不是同一场内 block strict，也不是本项目 conditional | 100 个合成 realization：70 train / 30 validation；无空间 buffer；没有独立真实场 blind test | 3D ResUNet++，min-max 到 `[-1,1]`，MSE，1000 epochs；报告 MSE、MAE、R²及噪声/方向 stress test；无变差函数，结构指标有限 | 未核实与论文完全对应的官方可运行仓库；预印本许可不等于代码许可，代码许可未知 |
| B07 | Putra et al. (2024)，[出版社论文](https://link.springer.com/article/10.1007/s12145-024-01240-7)，DOI `10.1007/s12145-024-01240-7`；**D3** | post-stack 地震属性经 RF 选特征，GP 预测总孔隙度并提供后验不确定性 | 被留井完全不进训练，属 **strict-like blind-well**；不是空间块 strict | 轮流完整移除一口井作 blind test；未报告距井 buffer 或整块盲体 | Gaussian process + RF feature selection；公开页报告 blind-test 表现并讨论 MSLL/相关性；未见 SSIM、频谱、变差函数 | 未核实官方代码；许可未知 |
| B08 | Strebelle (2002)，[原始论文 DOI](https://doi.org/10.1023/A:1014009426274)；可运行实现为 [MPSlib 官方仓库](https://github.com/AUProbGeo/mpslib)与[官方文档](https://mpslib.readthedocs.io/en/latest/)；**D3（离散相）** | 3D 河流相/复杂几何；训练图像提供高阶空间模式，hard/soft field data 提供条件 | 典型 **conditional simulation**；若测试井作为 hard data 输入，不是 strict | 原论文关注条件模拟而非 ML holdout；无空间块 blind test；MPS 搜索模板/半径不是数据 split buffer | SNESIM/序贯多点统计；不以 MSE 为核心。应评估硬数据 honoring、相比例、indicator variogram、连通性/几何分布和多 realization 不确定性 | MPSlib 可运行，LGPL-3.0；必须另有独立、授权且不含 frozen-test 真值的 training image |
| B09 | Song et al. (2022) GANSim-3D，[原始论文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031865)，DOI `10.1029/2021WR031865`；[作者官方仓库](https://github.com/SuihongSong/GeoModeling_GANSim-3D_For_large_arbitrary_reservoirs)；**D3（离散相）** | 3D 稀疏井相 + 3D 概率图（可由地球物理解释得到）+ 全局特征，输出 3D cave facies；训练 cube `64³`，场应用 `64³` 与 `336×256×96` | 明确 **conditional**；不应包装为 strict | 有合成 train/validation/test cube 与场应用，但不是把真实场连续空间块作为独立真值盲测；未见本项目式 buffer | progressive 3D GAN，adversarial loss + well/probability condition losses；评价应含条件符合率、IoU/几何/多样性，不能只用孔隙度 RMSE | 官方代码/数据/权重存在；许可证是混合的：上游 Progressive GAN 部分 CC BY-NC 4.0，作者新增材料/权重 MIT，商业使用需逐文件审计 |
| B10 | Mosser, Dubrule & Blunt (2018)，[原始预印本](https://arxiv.org/abs/1802.05622)；**D3（条件 3D 生成）** | Maules Creek 3D 储层相受一维井段约束；另有微 CT 体受二维切片约束 | 明确 **conditional**：优化 latent 使 realization honor 已知井/切片；不是 strict | 重点是条件 realization 采样；没有空间块盲测或 buffer | 预训练 3D GAN 作为先验，latent optimization 使用 masked content loss + discriminator perceptual loss；需要条件命中、形态/两点统计与多样性，不适合只报 RMSE | 未核实论文对应的完整官方代码/许可证；许可未知 |

### 4.1 证据覆盖统计

- 核心候选：10。
- 一手来源：10/10；其中 10 项有原始论文/预印本，B08/B09 另有官方实现，满足“至少 6 项一手来源”。
- 真正报告连续空间块 + buffer 的外部论文：0/10。现有地学文献更常见留井、合成 realization 或条件模拟，这恰好说明本项目的空间协议需要单独冻结和公开。
- 官方可运行代码且许可证可核对：B08；B09 有官方代码但混合/非商业条款需要细审。其余不能把“论文可访问”误写成“代码可复现”。

## 5. 补充近邻证据与排除项

### 5.1 近邻但不计入核心十项

- [Latent Diffusion Model for Conditional Reservoir Facies Generation](https://arxiv.org/abs/2311.01968)及[官方 MIT 仓库](https://github.com/ML4ITS/Latent-Diffusion-Model-for-Conditional-Reservoir-Facies-Generation)：确实是稀疏井条件储层相生成，代码/5000 样本可用，但公开实现的核心任务是二维相图，不能冒充本赛道三维直接 benchmark。
- [DiffSIM](https://arxiv.org/abs/2603.07383)：含一个 3D point-bar 案例，使用井位置/相指示和 mask-based denoising，并报告比例、变差函数和几何特征；很适合借鉴结构指标，但本次未核实官方代码与许可证，暂不列为十模型可执行基准。
- [blockCV 方法论文](https://doi.org/10.1111/2041-210X.13107)：不是储层重建模型，却是一手空间验证方法来源；支持按空间相关长度选择 block/buffer，防止相邻样本泄漏。

### 5.2 明确排除

| 排除对象 | 原因 |
|---|---|
| 普通 2D/3D 图像 inpainting、医学体补全、通用 NeRF/3D diffusion | 没有井、地震或储层结构条件；视觉相似不等于地质合理 |
| 数字岩心无条件 3D 生成 | 尺度、目标和约束不同；除非显式受井/切片条件且用于储层体，否则不作为本赛道直接基准 |
| 纯 IDW/RBF/Kriging 示例数据 | 算法可作基线，但若没有真实地质目标与合法空间验证，不能作为外部 benchmark 证据 |
| Kaggle F3 seismic 切片数据 | 有地震/相背景，但未核实到对齐的“稀疏井输入—3D 属性真值—空间盲测”竞赛合同 |
| Kaggle Volve F-9-A 单井 CSV/WITSML 子集 | 只有单井资料，不是三维属性体重建 |
| FORCE 井日志岩性任务 | 目标是 1D well-log/lithology prediction，不是 3D 空间体重建 |

本次“未发现直接 Kaggle 竞赛”是有界检索结论，不是证明全网绝不存在；在没有官方赛题页和可核验数据合同前，不应为凑数收录。

## 6. 当前项目合同的原始证据审计

### 6.1 两条 lane 与空间拆分

本地原始合同位于 `p4_reconstruction.py`：

- conditional：development I-block `(0,1,2,3)`，test `(4,5)`，没有 I-block guard，输入含 `conditional_idw_porosity`（`p4_reconstruction.py:70-82`）。
- strict：development `(4,5)`，guard `(3,)`，test `(0,1,2)`，不含 IDW/井输入（`p4_reconstruction.py:83-95`）。
- 两者拥有不同 `task_id`、target name、label version、input whitelist 和 metric prefix；这个物理隔离应保留。
- 测试 I-block 在 development CV 之前冻结；development CV 按连续 K-block 分组，`buffer_blocks=1`，purged fold-train IDs 单独记录（`p4_reconstruction.py:410-504`）。
- fold-train preprocessing 使用 z-score，target/output 为 identity；主指标是 RMSE，次指标有 MAE、bias、R²、Pearson 与 spectral log RMSE（`p4_reconstruction.py:230-266, 690-720`）。
- validation/final 统一使用 `~observed_mask`，因此 exact well cells 不进入指标（`p4_reconstruction.py:679-688, 1389-1406`）。
- Stage4 产物已明确 `prior_test_consumed=true`、`fresh_blind=false` 和 `evidence_class=previously_seen_reusable_holdout`（`p5_stage4_confirmation/summary.json:2-8`）。

这套合同在“模式隔离、空间 CV、防精确井点抬分、历史测试诚实标记”上符合推荐方向；不足之处见下节。

### 6.2 PyKrige 的真实输入语义

`_models/reconstruction/pykrige_ok3d.py:45-62` 的 `_fit_backend` 与 `_predict_backend` 均使用 `features[:, -3:]`。在当前 feature 顺序下这只是归一化坐标：

- strict 的 3 个地震通道未被使用；
- conditional 的 IDW 井约束通道和 3 个地震通道都未被使用；
- Stage4 JSON 虽记录 `test_constraints_used=90`（`p5_stage4_confirmation/conditional/metrics.json:8-18`），这表示条件特征在 pipeline 中被构造/供应，并不证明胜者后端消费了该列；adapter 的实际切片是更强的模型级证据。

同时 Stage2/Stage4 的 PyKrige fit 点来自固定预算的 development reference labels（Stage4 为 512 个），不是只来自实际稀疏井。它对 strict 测试块没有直接目标泄漏，但科学名称应是：

> 用 512 个 development 参考标签拟合的、只依赖坐标的 3D ordinary kriging 空间外推基线。

这仍然是公平模型预算下有价值的 baseline，但与 B01/B08 那类“硬井数据条件模拟”不是同一观测合同。后续报告必须给出“声明可用通道”和“模型实际读取通道”的 truth table。

### 6.3 当前已见留出结果

| lane | winner | metric voxels | exact well cells excluded | RMSE | MAE | bias | R² | Pearson | spectral log RMSE | 可支持的结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| strict | pykrige_ok3d | 78,949 | 0 | 0.0356249 | 0.0271728 | +0.0175436 | -0.339679 | -0.120046 | 1.479849 | 已见空间块上的坐标克里金失败/弱基线；不支持严格泛化 |
| conditional | pykrige_ok3d | 49,233 | 90 | 0.0210129 | 0.0150876 | +0.0004605 | 0.013686 | 0.139364 | 1.103390 | 已见、不同测试块上的条件 lane 分数；因后端忽略条件列，不能证明井约束收益 |

两行不能互减：其 test I-block、目标分布、development 规模和精确井点数量都不同。

## 7. 当前设计：保留、修改、废弃

### 7.1 保留

1. strict/conditional 各自独立的 TaskSpec、标签、input whitelist、split hash、leaderboard 和图件。
2. strict 禁止 test/guard/future/derived truth，且 reference-derived sparse well values 不作 strict 输入。
3. 连续空间块先冻结，再做 development 内 buffered spatial CV；fold-train-only preprocessing。
4. conditional 精确井格点从所有主指标排除。
5. RMSE 为连续孔隙度主指标，MAE/bias/R²/Pearson/频谱为互补诊断；所有最终指标做有限值检查。
6. PyKrige、GSTools、RBF 作为简单、可解释的空间插值基线。
7. 已消费 holdout 永久标为 known/previously seen，不覆写为 fresh blind。

### 7.2 修改

1. **把 conditional 做成配对实验**：同一测试块、同一训练集合、同一模型/seed，分别禁用和启用测试区井约束；报告增量及 block-bootstrap CI。
2. **增加 conditional 边界控制**：至少报告从 development/test 边界向内的距离分层；如要讨论空间外推，再引入与变差函数相关长度对应的 I-block guard。
3. **增加距井指标**：`exact=excluded`，再报告 `(0,r]`、`(r,2r]`、`>2r`，其中 `r` 由 development residual variogram range 或预注册物理距离确定。
4. **模型输入审计**：结果中写明 declared/constructed/actually consumed 三层通道。当前 PyKrige/GSTools/RBF 必须标 `coordinates_only`。
5. **区分监督学习与地统计观测预算**：统一 512 development labels 是模型公平预算，但不是“只有稀疏井”的观测合同。建议同时设 fixed-label-budget 与 well-only 两个子协议。
6. **结构指标补齐**：连续体加入 residual variogram、directional spectrum、按物理采样间距的频率轴与可选 3D/切片 SSIM；离散相加入 indicator variogram、连通性、相比例、几何对象尺度和条件符合率。
7. **不确定性基线**：PyKrige/GSTools/GP 报告预测区间 coverage、width 与 calibration，而不是丢弃 variance。
8. **新盲测**：若要恢复严格泛化主张，冻结一个从未被 P4/P5 看过的空间区/外部场；不能重用当前 holdout 名义“再次盲测”。

### 7.3 废弃

1. 废弃把 strict 与 conditional 两张榜直接比较成“井约束提升”的叙述。
2. 废弃把 PyKrige 当前分数称为井震融合、conditional-aware 或 seismic-aware。
3. 废弃只因精确井点被 mask 就宣称不存在条件泄漏；邻井传播是任务允许的信息，但必须分距离量化。
4. 废弃 voxel-iid 随机切分、voxel-iid 置信区间和随机像素 bootstrap。
5. 废弃用单一 RMSE 排名连续孔隙度、离散相生成和多 realization 生成模型的混合大榜。
6. 废弃把已见 holdout、留井验证或合成 realization validation 称为外部场 fresh blind test。

## 8. 推荐协议 A：严格空间泛化

### 8.1 数据与防火墙

1. 预注册一个从未访问的新连续 I-block、独立 reservoir sector 或外部 field；生成 `split_hash` 后锁定。
2. test 区不得有 target、well target、由 target 采样的 synthetic well、IDW/kriging from test truth、future simulation response 或任何同源派生量进入输入。
3. seismic 属性若由全体地震无监督/确定性计算可以使用，但需证明计算不访问 target；任何监督属性变换只 fit development。
4. guard 的物理宽度由 development 的目标/残差 directional variogram range 预注册；若块数不足，诚实降低折数，不缩 buffer 追分。

### 8.2 训练、选择与测试

1. development 内 5-fold contiguous spatial CV；每 fold 的预处理、模型、变差函数、校准只 fit effective fold-train。
2. primary = pooled unique-voxel RMSE；同时报告 MAE、bias、R²、Pearson/Spearman、normalized RMSE、directional spectrum、residual variogram。
3. HPO 只看 development OOF；winner/config/seed budget 冻结后，refit 全部 legal development。
4. frozen test 原子消费一次。若 test 读取前状态机/哈希不一致，fail closed。
5. test 结论按全体、I/K/depth block 和距 development 边界分层；CI 用空间 block bootstrap。

### 8.3 结论门槛

- `R²<=0`、相关性非正或显著残差空间结构时，不允许“泛化成功”表述，即使 RMSE 数字有限。
- 至少优于 development-mean、coordinate trend、IDW/RBF、PyKrige/GSTools 等预注册基线，且改进在空间 block CI 下稳定。
- 当前 Stage4 只可作为 known-holdout historical evidence，不满足 fresh blind 条件。

## 9. 推荐协议 B：测试区井约束条件重建

### 9.1 配对设计

1. 使用与 Protocol A 相同的物理测试块，或另冻一个 conditional 专用块；关键是 no-well 与 with-well 必须共享同一测试体。
2. 固定一组现实可获得的测试区井位置/值，记录来源、数量、测量噪声和 hash。
3. 同一 frozen 模型分别推理：
   - `B0`: 不给测试区井约束；
   - `B1`: 给测试区井约束；
   - 可选 `B2`: 打乱井值/位置的负对照，验证模型不是忽略条件。
4. exact well cells 永久排除；不得因为 hard honoring 把井点本身计为成功。

### 9.2 指标与声明

- 主指标仍为相同 metric mask 上的 RMSE；报告 `ΔRMSE = RMSE(B1)-RMSE(B0)` 及 spatial block bootstrap CI。
- 距井 bands、距测试边界 bands、每口井 leave-one-conditioning-out sensitivity 必须报告。
- 检查模型 actual feature use：若 B0/B1 预测 bitwise 相同或近似相同，标 `condition_unaware`，不得进入 conditional-aware 排名。
- 结论固定写“conditional reconstruction given test-region well constraints, NOT strict spatial holdout generalization”。

## 10. 当前十模型的公平测试要求

### 10.1 实际输入与排名资格

| model_id | 家族/representation | 当前实际或预期消费 | strict 资格 | conditional-aware 资格 | 额外要求 |
|---|---|---|---|---|---|
| `scipy_rbf_neighbors` | RBF，point | 当前实现只用末三列坐标 | 是，坐标基线 | 否；除非实现显式 consume 井条件或仅标 ablation | 固定 kernel/neighbors/smoothing，报告坐标单位 |
| `pykrige_ok3d` | Ordinary Kriging，point | 当前只用末三列坐标和 development labels | 是，坐标地统计基线 | 当前否；供应 IDW 不等于消费 | 报 variogram/nlags、variance calibration；区分 well-only 与 512-label budget |
| `gstools_krige_condsrf` | Kriging/conditional SRF，point | 当前只用末三列坐标 | 是，坐标地统计基线 | 当前否 | 报 covariance/len-scale、realization seed 和 uncertainty |
| `mpslib_snesim3d` | MPS，categorical volume | 合法 training image + hard/soft conditions | 仅当 TI 与 test 独立 | 是，天然 hard-condition | 无独立授权 TI 时结构化 SKIP；不得从 frozen Eclipse test 制 TI |
| `gpytorch_svgp` | sparse GP，point neural | 全 feature vector，strict 6/conditional 7 列 | 是 | 是，须通过 B0/B1 sensitivity | 固定 inducing budget/updates，报 posterior calibration |
| `monai_basicunet3d` | 3D CNN，volume | 全体 volume channels | 是 | 是，须保证 constraint channel/mask 明确 | 与 SegResNet/FNO 共享 patch、target voxel 和 GPU budget |
| `monai_segresnet3d` | 3D CNN，volume | 全体 volume channels | 是 | 是 | 同上；不能把 CPU neural 分数混入 GPU 正式榜 |
| `neuralop_fno3d` | neural operator，volume | 全体 volume channels | 是 | 是 | 相同频率 modes、updates、patch 与 GPU 锁证据 |
| `tcnn_hashgrid_inr` | hash-grid INR，point | 全 feature vector | 依赖可用后是 | 依赖可用后是 | 缺合法 tiny-cuda-nn build 时结构化 SKIP，不现场编译追分 |
| `siren_inr` | sinusoidal INR，point | 全 feature vector | 是 | 是 | 固定 omega/层宽/updates；报告频谱偏置与 seed 方差 |

### 10.2 Apple-to-apple 规则

1. **相同数据**：同 lane、fold、seed 使用相同 active cells、metric mask、标签版本、split hash；禁止 candidate-specific 临时切分。
2. **相同观测预算**：point 模型共享相同 512 train voxels 与 validation voxels；volume 模型共享等量 target mask。另设 well-only 协议时，所有地统计模型只能读同一井集。
3. **相同选择规则**：相同 OOF primary metric、minimize 方向、固定更新/墙钟上限；不以 frozen test 选择模型。
4. **分 representation/lane 排名**：point、volume、categorical-generative 不混成单榜。统一连续孔隙度 RMSE 可作横向摘要，但须同时显示计算预算和输入模态。
5. **条件消费验证**：conditional-aware 模型必须通过井通道置零/打乱/真值三组 sensitivity；坐标模型标作 condition-unaware ablation。
6. **结构与频谱定义冻结**：FFT 前的 mask/fill/window、物理 voxel spacing、方向轴和归一化写入 manifest；当前 mean-fill spectral RMSE 可保留，但需增加有物理频率轴的 directional 版本。
7. **SSIM 定位**：仅作切片/局部结构诊断；冻结 data range、window、mask 和三方向聚合，不取代 RMSE。
8. **变差函数**：连续体报告 truth/prediction/residual 的 directional experimental variogram 及 range/sill/nugget 偏差；离散相用 indicator variogram。
9. **盲测状态**：development OOF、known holdout、fresh blind 三类证据分栏，禁止混报。
10. **许可/依赖**：每个外部实现锁定 source commit/tag 与 license；MPS TI、GAN 权重和数据的许可证单独核验。

## 11. R² 为负：解释与必做测试

### 11.1 正确解释

当前代码计算：

`R² = 1 - sum((prediction - target)²) / sum((target - mean(target))²)`。

[scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)明确说明：对非恒定 `y_true`，R² 可以为负；预测真实 `y` 均值的常数模型得到 0，负值表示模型可比该参照更差。因而：

- strict `R²=-0.3396787` 表示 `SSE_model / SST_test = 1.3396787`；模型在该已见空间块上比 oracle test-mean 常数基准多 33.97% 平方误差。
- oracle test mean 只用于解释 R²，不是可部署 baseline，因为推理时不知道测试目标均值。
- Pearson `-0.1200` 说明空间排序也略反向；bias `+0.01754` 说明整体高估。三者合看，问题不只是均值校准。
- conditional `R²=0.01369` 只解释约 1.37% 的测试方差，仍是很弱的空间解释力，不能仅凭 RMSE 小于 strict 就宣称有效。

### 11.2 最小审计测试

1. 用独立实现或 `r2_score(..., force_finite=False)` 复算，并断言 target 非恒定、所有输入有限。
2. 同时报告 `SSE`、`SST`、target mean/std/range、prediction mean/std/range 和 `RMSE / std(target)`。
3. 比较合法可部署基线：development mean、development linear coordinate trend、RBF、IDW、PyKrige/GSTools；不能把 test mean 当部署基线。
4. 按 I-block、K/depth、距井、距 train/test 边界报告 RMSE/R²；小分组 target 近恒定时 R² 标 undefined，不强制替换为 0。
5. 报 calibration slope/intercept、bias、Pearson、Spearman；画 truth-pred scatter/CDF 与三视图 residual。
6. 对 residual 计算 directional variogram 或空间自相关，识别未建模趋势/结构。
7. 用空间 block 或 well-level bootstrap 给 RMSE/R²/ΔRMSE 置信区间；禁止把数万相关 voxel 当 iid。
8. 预注册 fail gate：fresh strict test 上 `R²<=0` 或相关性非正时，只能报告失败/弱基线，不得写“泛化成功”。

## 12. 最小复现实验

目标不是长训练，而是用最小代价验证协议和当前 PyKrige 语义。

### 12.1 Development-only 预检

1. 从冻结 manifest 读取 strict/conditional 的 development fold 0；断言 frozen test arrays 未加载。
2. 固定 seed `2693` 和同一 512-point train budget、2048-point validation budget。
3. strict 构建 6 列、conditional 构建 7 列；记录列名/hash。
4. 对 PyKrige 分别运行：
   - 原始特征；
   - 地震列随机置换；
   - conditional IDW 列随机置换；
   - 坐标列随机置换。
5. 预期原实现对前两种置换预测完全不变，对坐标置换变化。若成立，自动标 `coordinates_only`。
6. 在同一 fold 比较 development-mean、RBF、PyKrige、GP；报告 RMSE/MAE/R²/spectral/residual variogram 与墙钟。

### 12.2 Conditional paired smoke

1. 仅在 development 内模拟一个“pseudo-test spatial block”，不打开 frozen test。
2. 从 pseudo-test 内选预注册稀疏井；B0 不给井，B1 给井，exact points 从 metric mask 去掉。
3. 使用真正消费条件通道的 GP/线性/MLP/INR 或显式 hard-conditioned kriging；同一 frozen fit 比较 B0/B1。
4. 报总体与距井 bands 的 ΔRMSE；加入 shuffled-well B2 负对照。
5. PyKrige 当前 adapter 作为 condition-unaware ablation；若 B0/B1 相同，测试应通过但其 conditional-aware rank 必须拒绝。

### 12.3 新 strict blind 的进入条件

只有在找到从未被 P4/P5 读取的新空间区/外部 field、冻结 split 和 buffer、完成 development-only 选择后，才能执行一次 final test。当前已见 Stage4 holdout 不参与这一步，也不能通过重命名恢复“盲”。

## 13. 主要风险与证据边界

1. **单一参考体**：Volve Eclipse/RMS 是工程模型，不是多个独立 field；同体空间块泛化不等价于跨场泛化。
2. **标签谱系**：参考体可能已经吸收同源井/地震解释。输入和标签相关是任务本身所需，但应披露，不应称完全独立测量。
3. **空间相关**：相邻 block、无 guard 的 conditional 边界会使有效样本数远小于 voxel 数；必须用空间 CI。
4. **表示不等价**：连续孔隙度回归、离散相模拟和多 realization 生成不能靠单一 RMSE 排同榜。
5. **论文复现性**：核心十项中多数没有可核验官方代码/许可证，适合做协议证据，不等于可直接接入的十模型实现。
6. **GANSim-3D 许可**：官方仓库声明上游部分 CC BY-NC 4.0、作者新增部分 MIT；需逐文件 provenance，不能笼统标 MIT。
7. **MPS training image**：无独立授权 training image 时结构化 SKIP 是正确选择；禁止从 frozen Eclipse test 体制作 TI。

## 14. 最终建议

当前最可信的科学叙述是：

> 本赛道在同一 Volve 参考体上维护两套互不混报的合同。strict 用连续空间块评估不含测试区井/真值的外推，但现有 holdout 已历史消费且当前 PyKrige 的负 R² 不支持泛化成功；conditional 允许测试区稀疏井并排除精确井点，但当前 PyKrige 实际忽略条件通道，且两 lane 测试块不同，所以尚未形成井约束增益的配对证据。

下一轮不应先增加更复杂模型，而应先完成三件事：冻结同块 B0/B1 conditional 消融；把 declared/actual feature consumption 纳入 fail-closed 测试；为 strict 准备一个从未消费的新空间区或外部场。完成后，再按第 10 节的统一预算与结构指标评测十模型。

## 15. 主来源清单

1. Equinor. [Volve data sharing](https://www.equinor.com/energy/volve-data-sharing)；[官方数据目录与许可说明](https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf)。
2. Yao, T., & Journel, A. G. (2000). [Integrating seismic attribute maps and well logs for porosity modeling](https://www.sciencedirect.com/science/article/pii/S0920410500000681). DOI `10.1016/S0920-4105(00)00068-1`。
3. Leite, E. P., & Vidal, A. C. (2011). [3D porosity prediction from seismic inversion and neural networks](https://www.sciencedirect.com/science/article/pii/S0098300410002682). DOI `10.1016/j.cageo.2010.08.001`。
4. Chaki, S., et al. (2014). [Well tops guided prediction of reservoir properties using modular neural network](https://arxiv.org/abs/1509.07079). DOI `10.1016/j.petrol.2014.06.019`。
5. Verma, A. K., et al. (2014). [Quantification of sand fraction from seismic attributes using Neuro-Fuzzy approach](https://www.sciencedirect.com/science/article/pii/S0926985114002912). DOI `10.1016/j.jappgeo.2014.10.005`。
6. Azevedo, L., Grana, D., & Amaro, C. (2019). [Geostatistical rock physics AVA inversion](https://academic.oup.com/gji/article/216/3/1728/5222649). DOI `10.1093/gji/ggy511`。
7. Jo, H., et al. (2021). [Machine learning-based porosity estimation from spectral decomposed seismic data](https://arxiv.org/abs/2111.13581)。
8. Putra, M. H. R., et al. (2024). [Reservoir porosity assessment and anomaly identification from seismic attributes using Gaussian process](https://link.springer.com/article/10.1007/s12145-024-01240-7). DOI `10.1007/s12145-024-01240-7`。
9. Strebelle, S. (2002). [Conditional Simulation of Complex Geological Structures Using Multiple-Point Statistics](https://doi.org/10.1023/A:1014009426274)；[MPSlib 官方仓库](https://github.com/AUProbGeo/mpslib)。
10. Song, S., et al. (2022). [GANSim-3D for Conditional Geomodeling](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021WR031865)；[作者官方仓库](https://github.com/SuihongSong/GeoModeling_GANSim-3D_For_large_arbitrary_reservoirs)。
11. Mosser, L., Dubrule, O., & Blunt, M. J. (2018). [Conditioning of three-dimensional GANs for pore and reservoir-scale models](https://arxiv.org/abs/1802.05622)。
12. Roberts, D. R., et al. (2017). [Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://doi.org/10.1111/2041-210X.13107)。
13. scikit-learn. [`r2_score` 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)。
