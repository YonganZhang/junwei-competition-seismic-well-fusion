# P5 地震相直接基准调研：F3 Netherlands、Penobscot 与语义等价 3D 像素分割

日期：2026-07-14

范围：只纳入 F3 Netherlands、Penobscot，或任务语义等价的 3D 地震相、盐体、地层相像素级分割。普通自然图像分割与仅讲通用 2D 分割的材料不纳入候选。

本地基线：`p5-stage3-facies`，审计起点 `cc1e615e8a35a6da1040a08ca7009a41ca4c4a6b`，调研开始时工作树 clean。

## 1. 结论先行

1. **没有证据表明本项目误用了与当前标签完全匹配的公开标准 split。** Silva F3 的十个整数区间（0..9）和 Baroni Penobscot 的八个整数区间（0..7）原始发布均没有给出可直接复用的、带冻结 test 的空间 benchmark split。现有连续 inline + guard 的外层切分是 **track-defined split**，不是文献 canonical split；其空间隔离反而比原发布论文中的随机/交错 section 切分更严格。
2. **不得把当前 F3 十类结果与 Alaudah 六类 F3 benchmark 横向比较。** Alaudah 对地层合并后只保留六类，并规定两个空间 test block；它是最清楚的 F3 公共 benchmark，但标签版本不同。当前 split 也不复现其 NW-train / SW-test1 / east-test2 体块。
3. **不得把当前 Penobscot 八类结果与合并 facies 的论文数字横向比较。** Chevitarese/Civitarese 等工作把薄层类别合并为七类或更少，并主要使用相邻 inline 的 block 内 70/30 划分；EarthAdaptNet 的三类域适配更不是完整八类任务。
4. **随机切片和随机重叠 patch 会产生实质泄漏。** 相邻 inline/crossline 高度相关；inline 与 crossline 还在真实体素处相交；先提取重叠 patch 再随机切分会把近重复纹理分到 train/val。Gutierrez 等 2025 直接指出随机 60/20/20 patch 切分会夸大地震相模型表现。
5. **当前空间 split/CV、防 test 读取、fold-train 统计、逐类指标与校准图应保留。** 但 Stage3 的 32 个训练 patch、16 个验证 patch、40 次更新只是 compute-matched pilot；Stage4 又只评估保存的 patch 样本，且 holdout 已被历史流程消费。其数字可用于内部确认，不能冒充 full-volume、fresh-blind 或公共 benchmark 结果。
6. 下一轮最有价值的补实验不是再换自然图像 backbone，而是：在冻结空间 split 上增加充分训练预算；补 full-section/full-volume 推理；比较 weighted CE、CE+Dice、CE→Lovász；再以严格体块方式比较 2D、2.5D、3D。所有选择只能由 development OOF 决定。

## 2. 研究方法与证据口径

- 一手来源优先级：原始论文/出版社页面、数据集或竞赛官方页面、作者官方 GitHub。下表十项均有原始论文；其中五项同时有官方数据、官方竞赛或作者代码。
- “直接”要求：输入必须是地震剖面或地震体，输出必须是逐像素/逐体素地震相、盐相或地层相标签。只做整图分类、目标检测、通用自然图像分割的工作不计入。
- 没有在原始材料中找到的字段记为 **NR（not reported）**，不根据常见做法猜测。
- “标准 split”仅指发布者明确冻结、可按索引复建且标签定义相同的切分。论文中的一次随机划分、模糊比例或不同标签合并，不称为当前任务的标准 split。
- 许可证分开记录数据、代码与权重。论文开放获取不代表代码或模型权重可自由使用。

## 3. 数据与标签版本先对齐

| 数据/版本 | 原始体或切片 | 标签 schema | 是否等价于本项目 |
|---|---|---|---|
| Silva F3 Netherlands | 原体约 651 inline × 951 crossline × 462 time；发布 TIFF/PNG section 对 | 9 条层位把像素分成 ID 0..9 的十个区间；Zenodo 页面摘要又写“9 classes”且 section 数与论文不完全一致 | **是，本项目 `facies_f3` 固定 0..9**；需在 artifact 中锁定实际文件清单/hash，不能只引用页面摘要 |
| Alaudah F3 benchmark | 裁剪体 IL 100..700、XL 300..1200、depth 1005..1877 | 合并成 6 类：Zechstein、Scruff、Rijnland/Chalk、Lower/Middle/Upper North Sea | **否**；只能作为单独 `F3-Alaudah6` 复现任务 |
| Baroni Penobscot | 601 inline × 481 crossline × 1501 depth | 7 条层位形成 ID 0..7 的八个区间 | **是，本项目 `facies_penobscot` 固定 0..7** |
| Chevitarese/Civitarese Penobscot | 清理后主要使用 459 个 inline | 合并薄层，常见为 7 类或更少 | **否**；只能另建 merged-label reproduction lane |
| EarthAdaptNet 域适配 | F3 六类、Penobscot 八类源数据，但域适配只取语义可对应的 3 类 | 三个反射/地层语义近似类别 | **否**；完整多类 segmentation 与三类 DDA 不应混榜 |

## 4. 十项直接工作证据矩阵 A：数据、标签与空间切分

| # | 工作与直接性 | 数据切片/体 | 标签 schema | train/val/test 与空间间隔 | 2D/2.5D/3D |
|---|---|---|---|---|---|
| 1 | **Silva et al. 2019, Netherlands Dataset**：直接发布 F3 像素级地层相标签，是当前十类任务的源头。[论文](https://arxiv.org/abs/1904.00770)；[官方数据](https://doi.org/10.5281/zenodo.1422787) | F3 原体约 651 IL × 951 XL × 462 time；论文称 1,602 seismic TIFF + 1,602 mask PNG | 9 horizons → 10 个整数区间 0..9；官方页面的“9 classes”文字与文件/论文计数存在版本冲突 | **无发布方固定 split。** 论文建议用户自己留 25% test、再从余下取 25% val；分类示例用奇偶 section 交错。未定义 buffer | 发布物是 2D inline/crossline；可由体坐标重建 3D 关系 |
| 2 | **Baroni et al. 2019, Penobscot Dataset**：直接发布 Penobscot 像素级相标签，是当前八类任务的源头。[论文](https://arxiv.org/abs/1903.12060)；[官方数据](https://doi.org/10.5281/zenodo.1324463) | 601 IL × 481 XL × 1501 depth；1,083 seismic TIFF + 1,083 mask | 7 horizons → 8 区间，ID 0..7 | **无固定空间 benchmark split。** 分类数据先按 section 文件划分再 shuffle；若 IL/XL 同时随机进入两侧，会在交点共享体素。未定义 buffer | 2D section/tiles；原 HDF5 是对齐 3D 体 |
| 3 | **Alaudah et al. 2019, ML benchmark for facies classification**：直接 F3 六类 dense segmentation，且给出最明确公共体块 benchmark。[论文](https://arxiv.org/abs/1901.07659)；[数据](https://doi.org/10.5281/zenodo.3755060)；[作者代码](https://github.com/yalaudah/facies_classification_benchmark) | 裁剪体 IL 100..700、XL 300..1200、depth 1005..1877 | 6 类；Rijnland/Chalk 合并 | train：IL 300..700, XL 300..1000；test1：IL 100..299, XL 300..1000；test2：IL 100..700, XL 1001..1200。零重叠但边界相邻，**无 guard**；论文要求 test 仅最终使用 | 2D patch 与 2D full-section 两条模型；推理融合 IL/XL 预测，相当于双视角 2.5D 弱融合但非 3D 卷积 |
| 4 | **Chevitarese et al. 2018, Seismic Facies Segmentation Using Deep Learning**：同时直接处理 F3/Penobscot 像素相分割。[论文](http://www.searchanddiscovery.com/documents/2018/42286chevitarese/ndx_chevitarese.pdf) | 去除边缘低质量 section 后约 Pen 459 IL、F3 591 IL；训练 tiles 80×120，分类预训练 tiles 40×40 | Pen 原 8 类后合并两对薄层；F3 原标签也合并一对，故与当前 schema 不同 | 把 inline 分成 10 个空间块，每块前 70% train、后 30% val；未报告独立冻结 test；块内边界相邻、无 buffer | 2D，两阶段 patch 分类预训练→FCN segmentation |
| 5 | **Civitarese et al. 2019, Semantic Segmentation of Seismic Images**：直接 Penobscot FCN/U-Net 相分割。[论文](https://arxiv.org/abs/1905.04307) | 主要使用清理后的 459 IL；80×120 或 128×128、50% overlap tiles | Pen 8 类合并 2/3 后为 7 类 | 文中描述 n 个块内 70/30 train/val，同时又提 40 张 test，独立来源不够清楚；无空间 buffer | 2D Danet-FCN2/3、FCN、U-Net |
| 6 | **Nasim et al. 2022, EarthAdaptNet**：直接做 F3/Penobscot 地震相 segmentation，并做跨数据域三类适配。[论文](https://arxiv.org/abs/2011.10510)；[出版社 DOI](https://doi.org/10.1109/TGRS.2022.3151883) | F3 采用 Alaudah 裁剪体；Penobscot 用 IL/XL section；segmentation patches 99×99、半步长 | F3 6 类；Pen 8 类；跨域 DDA 仅取 3 个可对应类别 | F3 沿用 Alaudah 体块。Pen 报告 train IL 1000..1500、test 1500..1600，若端点均含则 IL1500 重叠；patch 提取后随机留 20% val，重叠 patch 存在近重复泄漏风险；无 buffer | 2D segmentation 与 patch-level DDA |
| 7 | **Gutierrez et al. 2025, Performance Evaluation of DL Models for Seismic Facies Segmentation**：直接 F3/Parihaka 公平比较，重点研究 split 与指标偏差。[论文](https://doi.org/10.1111/1365-2478.70104) | F3 采用 Silva 数据但合并到 Alaudah 六类；输入 512 crop；同时评估 IL/XL | F3 6 类；Parihaka 6 类 | F3 遵循 Alaudah 体块 split；论文明确批评先随机 60/20/20 切重叠 patch，因为相邻地震相近似重复。未额外加 guard | 2D DeepLabV3/V3+、SegFormer、Segmenter、SETR |
| 8 | **Sheng et al. 2023, Seismic Foundation Model (SFM)**：不使用 F3/Pen 标签，但直接做 Parihaka 多类地震相及 TGS 盐像素分割，任务语义等价。[论文](https://arxiv.org/abs/2309.02791)；[作者代码](https://github.com/shenghanlin/SeismicFoundationModel) | 192 个 3D surveys 预训练得到约 2.286M 2D 图；Parihaka 590 XL；TGS 101×101 | Parihaka 6 类；TGS salt 二类 | Parihaka 前 500 XL train、后 90 val，再每 5 条采样为约 100/17；边界无 guard、无冻结 test。TGS 约 3500/500 train/val，空间坐标未知 | 2D MAE/ViT 预训练与 2D segmentation fine-tune |
| 9 | **Shi et al. 2019, SaltSeg**：真正 3D 地震盐体逐体素分割，并在 F3 做定性迁移，属于语义等价 3D lane。[论文](https://doi.org/10.1190/INT-2018-0235.1)；[作者托管 PDF](https://cig.ustc.edu.cn/_upload/tpl/05/cd/1485/template1485/papers/shi2019saltSeg.pdf) | SEAM 合成 3D 体，随机 128³ subvolume；F3 子体 651×700×240 仅定性 | salt / non-salt 二类 | SEAM volume 空间对半 train/validation，论文抽取文本未给精确索引；不是随机 section。无独立公开 test；F3 无 GT 指标 | 3D U-Net，滑窗重叠 3D inference |
| 10 | **Karchevskiy et al. 2018, TGS Salt**：真实地震图像二类盐相逐像素分割，竞赛提供隐藏 test。[论文](https://arxiv.org/abs/1812.01429)；[官方竞赛](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview)；[作者代码](https://github.com/K-Mike/Automatic-salt-deposits-segmentation) | 4,000 张 101×101 labelled train，约 18k hidden test；有 depth 元数据，未发布采集空间坐标 | salt / non-salt 二类 | 官方 hidden test；训练内部 5-fold。因缺空间坐标，无法审计不同图像是否地质相邻；不能把随机 image fold 迁移为 F3/Pen 的合规方法 | 2D U-Net + SE-ResNeXt50 |

## 5. 十项直接工作证据矩阵 B：训练、指标、推理与许可

| # | 增强 | loss 与类别不均衡 | 指标与报告口径 | 测试时策略 | 代码/数据/权重许可 |
|---|---|---|---|---|---|
| 1 | 分割增强 NR；分类通过从每类抽同量 tiles 做平衡 | 分割 loss NR；分类以 dominant-pixel tile 标签，设阈值控制纯度 | 先行 segmentation 工作报告约 90–98% mIoU，但类别合并和 split 细节不足，不能当当前十类 benchmark | NR | 数据 CC BY 4.0；未识别官方训练代码，代码许可 NR |
| 2 | 分类 tile 采样允许最多 30% 其他类 | 合并薄层并按类平衡 tile 数 | 先行语义分割宣称 >97% IoU，但标签合并且无严格 test 细节 | NR | 数据 CC BY 4.0；未识别官方训练代码，代码许可 NR |
| 3 | 小角度旋转、水平翻转、Gaussian noise | 普通分类/segmentation 目标；直接报告严重不均衡：六类约 1.48/3.17/6.53/48.44/11.89/28.49% | PA、逐类 CA、MCA、FWIU；不报告标准 per-class IoU/mIoU。最佳 section+aug+skip 约 PA .905、MCA .817、FWIU .832 | patch 重叠输出平均；section 模型对 IL/XL 预测融合 | 代码 MIT；Zenodo 数据许可以记录页为准；预训练权重无独立许可声明时不得推定 |
| 4 | NR | 先以 dominant facies 40×40 patch 做分类预训练，后转 FCN；通过合并薄层缓解不均衡；具体 segmentation loss NR | 主要看 validation mIoU/像素准确率；抽象称像素准确率 >97%，无当前 schema 的逐类可复核表 | 2D tiles 拼接/section 可视化；TTA NR | 未找到官方可运行代码；代码/权重许可 NR |
| 5 | NR | RMSProp、weight decay；loss 在论文抽取文本中 NR；通过合并 2/3 降低薄层困难 | per-class IoU、图像级 mIoU，再对图像 mIoU 求平均（mmIoU）；与全局 confusion 汇总 mIoU 不等价；报告可达 >99% | 无 overlap test tiles 组装 section；按 validation mIoU 选模型 | 未找到作者官方代码；代码/权重许可 NR |
| 6 | ≤10° rotation、blur、flip、shift、noise | segmentation CE；域适配为任务 loss + CORAL；类别映射只保留三类 | PA、CA、MCA、IoU/mIoU、FWIU；EAN segmentation mIoU 约 .62，三类 DDA 主要报 accuracy | segmentation 用无 overlap patch；早停/dropout；TTA NR | 未找到作者官方 repo；代码/权重许可 NR |
| 7 | random crop、horizontal flip | **weighted CE**，权重来自训练分布；明确讨论 rare class 与 accuracy/FWIU 的掩蔽问题 | 以 mIoU 排名，并报 PA、MCA、macro-F1、per-class IoU/confusion；F3 六类最佳 SETR mIoU .7849 | IL/XL 依照体块评估；softmax/TTA 细节 NR | 文章开放获取；未定位明确官方代码仓和代码许可，故代码许可 NR |
| 8 | 下游增强细节 NR | fine-tune 使用 CE；预训练为 MAE reconstruction；未报告显式类权重 | Parihaka IoU/CPA/mIoU/accuracy；SFM fine-tune mIoU .7294，较大输入版本 .7980 | 2D fine-tune；TTA NR | 作者 GitHub 代码 MIT；checkpoint 虽可下载但未单列权重许可，故本项目继续 scratch/许可 gate |
| 9 | 3D random crop；旋转等增强 | BCE；每个 subvolume z-score，未报告显式 class weight | accuracy .9609、precision .9004、recall .9468、F1 .923；未报 IoU | overlapping 128³ 滑窗，以 Gaussian 衰减权重融合；阈值 .5；F3 再做 salt-indicator 后处理 | 数据需向作者申请；未找到公共官方代码，代码/权重许可 NR |
| 10 | horizontal flip、brightness、horizontal shift、小旋转；作者报告 vertical flip/大旋转有害 | 先 BCE，后 0.1 BCE + 0.9 Lovász；depth/CoordConv；不以像素频率 class weight 为主 | Kaggle 指标为 IoU 阈值 .50:.05:.95 的 mean precision，不等于 mIoU；竞赛排名 top 1% | 5-fold snapshot ensemble、horizontal-flip TTA、validation 选 threshold、连通域后处理 | 作者代码 MIT；竞赛数据受 Kaggle competition rules 约束；ImageNet encoder 权重许可须另核 |

## 6. F3/Penobscot “标准 split”审计

| 问题 | 一手证据 | 当前 Stage3/4 | 判定 |
|---|---|---|---|
| 当前 F3 十类是否有官方固定 split？ | Silva 只发布 section+mask 并建议用户自划 train/val/test；没有索引 manifest | development IL 100..586；guard 587..619；holdout 620..750；只用 inline，避免 IL/XL 体素交点泄漏 | **无标准可误用。** 当前是更严格的自定义 split，必须标注 `track-defined-f3-10class` |
| 当前 F3 是否复现 Alaudah？ | Alaudah 是六类，train NW、test1 SW、test2 east | 当前为十类、只按 inline 连续切，且本地 patch 范围超出 Alaudah 的裁剪定义 | **不复现，也不应声称复现。** 若比较论文数字，必须新建六类任务并用其精确体块 |
| 当前 Pen 八类是否有官方固定 split？ | Baroni 无冻结空间 benchmark；分类示例会 shuffle section | development IL 1000..1448；guard 1449..1479；holdout 1480..1600 | **无标准可误用。** 当前 guard 比多数论文更保守，命名应为 `track-defined-penobscot-8class` |
| 当前 Pen 是否复现 EarthAdaptNet？ | EarthAdaptNet 报 train 1000..1500、test 1500..1600，端点可能重合；三类 DDA 又改变 schema | 当前在 1449..1479 留 guard，test 从 1480；完整八类 | **不复现。** 可能受其范围启发，但不能引用其 split 名义或三类分数 |
| holdout 还是 blind test 吗？ | 公共数据标签始终可见；本项目 P4/Stage4 已计算同一 holdout 的指标 | Stage4 明确写 `prior_test_consumed=true`、`fresh_blind=false`、`evidence_class=previously_seen_reusable_holdout` | **不是 fresh blind。** 可作固定 reusable holdout，但再次试验不能称独立盲测 |

### 6.1 随机切片泄漏的四条路径

1. **邻近 section 泄漏**：IL `k` 和 `k+1` 只相隔一个采样间距，构造、断层和相带大面积相同。随机 section split 会把近复制地质结构放到两侧。
2. **正交视图交点泄漏**：一个 inline 与一个 crossline 在整条 time/depth 轴上共享真实体素。若同一体的 IL 与 XL 分别进入 train/test，即使文件名不同也不是独立样本。
3. **重叠 patch 泄漏**：先用半步长或 50% overlap 切 patch、后随机分 train/val，会直接共享大量像素；EarthAdaptNet 和 Civitarese 类流程存在这一风险。
4. **选择/统计泄漏**：在 test 上选阈值、后处理、epoch、代表样本，或从全体数据拟合 normalization/class weight，同样消费 test 信息。softmax 本身不是泄漏，但校准器、温度或阈值只能 fit fold-train/OOF。

现有 pipeline 通过“只用 inline + 空间连续外层 holdout + guard + fold buffer + fold-train preprocessing”阻断了前四条中的前三条与统计泄漏；Stage4 的问题不是 train/test 空间重叠，而是同一 holdout 已在历史阶段被查看，因此证据类别降低为 reusable holdout。

## 7. 当前 Stage3/4 与直接基准的 Pipeline 差异

| 环节 | 当前实现/产物证据 | 直接文献做法 | 决策 |
|---|---|---|---|
| 任务身份 | F3 0..9、Pen 0..7，独立 head/榜单；见 `p5_stage3_summary.json` | 公开结果常把 F3 合为 6 类、Pen 合为 7/3 类 | **保留。** 标签版本进入 task ID 与每个 artifact；不同版本永不混榜 |
| 外层 test | 连续 inline，F3 guard 33 条、Pen guard 31 条；不读 crossline | Alaudah 有 6 类空间块但无 guard；原始 10/8 类无标准 | **保留。** 但统一称 track-defined；Stage4 以后称 known/reusable holdout |
| development CV | 5 个 group/spatial folds，train/val 有 buffer，OOF core 全覆盖 | 多数旧论文随机 patch 或块内 70/30 且无 buffer | **保留并作为主协议。** 这是当前最重要的防泄漏优势 |
| patch 构建 | 每 inline 4 个确定性 128×128 patch；split 前按 section 归组 | 文献常 50% overlap、99×99、80×120、full section 或 3D 128³ | **补实验。** patch pilot 可保留，但正式报告加 full-section/full-volume sliding evaluation |
| Stage3 预算 | 每 fold/seed 32 train +16 val patch、40 updates，5 folds×3 seeds；90/90 cells | 公共结果常 200–1000 epochs或约 10k crops | **推翻其“充分训练/最终模型”含义。** 保留为公平 Stage1/3 pilot，但不能据此否决模型能力 |
| loss/激活 | fold-train class weight 的 CE on raw logits；softmax 仅 inference | weighted CE 有直接支持；Salt/TGS 常用 Dice/Lovász 类边界/IoU surrogate | **保留为固定 baseline；补 CE+Dice、CE→Lovász 受控消融。** 不把 softmax 放进 CE 前 |
| 不均衡指标 | accuracy、global confusion mIoU、macro-F1、per-class support/IoU/F1，缺类记 0；ECE/reliability | 旧文常只报 PA/FWIU；2025 研究主张 mIoU/macro/per-class | **保留。** 同时明确 absent-class 规则、再补 frequency-weighted 指标仅作辅助 |
| 模型选择 | development OOF 排名；F3 winner `smp_fpn_r18`，Pen winner `smp_deeplabv3plus_r18` | 文献可用 validation mIoU；test 不应反复选择 | **保留。** Stage4 结果不能反向改 winner、loss、阈值或 split |
| Stage4 refit | 对 legal development 统计，但固定 40×2 draws，F3/Pen 实际只见 78/79 个唯一优化样本；评估保存 test patches | 公共 benchmark 多评估完整 section/volume | **补充而非替代。** 当前结果仅 known-holdout confirmation，不是最终 full-volume benchmark |
| Stage4 证据 | F3 acc .238874/mIoU .126495/macro-F1 .217813；Pen .528572/.125380/.170917 | 文献高分多来自不同标签、不同 split、充分训练、不同 mIoU 聚合 | **禁止直接数值比较。** 差异主要是任务和预算，不足以推断架构优劣 |
| 2.5D/3D | 当前正式竞赛均为 2D；`monai_unet3d` 因缺 contiguous-block adapter 结构化 skip | Alaudah 融合 IL/XL；SaltSeg 真 3D 滑窗 | **补实验。** 新 adapter 必须从同一空间块取邻片/子体，不能跨 guard/fold |
| 校准与图 | raw softmax、ECE、可靠性图、GT/pred/confidence/error | 旧 benchmark 常不报校准 | **保留。** 任何 temperature/threshold 只能由 fold-train/OOF fit，test 只 apply 一次 |

本地证据路径：

- 数据与外层切分：`_pipelines/02_task_datasets/facies/build_dataset.py`
- P4 空间 CV：`_pipelines/02_task_datasets/facies/p4_spatial.py`
- Stage3 执行与汇总：`_pipelines/02_task_datasets/facies/facies_p5_stage3.py`、`_outputs/p5_stage3/p5_stage3_summary.json`
- Stage4 known-holdout：`_pipelines/02_task_datasets/facies/facies_p5_stage4.py`、`_outputs/p5_stage4_confirmation/p5_stage4_summary.json`
- Stage4 状态声明：`_outputs/p5_stage4_confirmation/stage4_state.json`

## 8. 推荐协议 A：本项目原生标签的防泄漏主协议

目标是回答“在 Silva F3 十类和 Baroni Penobscot 八类各自标签空间内，模型对空间外推的能力如何”。

1. **任务冻结**：`facies_f3_track10_v1` 固定 0..9；`facies_penobscot_track8_v1` 固定 0..7；无 `ignore_index`。原始文件、section 坐标、标签直方图和发布版本全部 hash。
2. **评估单元冻结**：训练可用 patch，但外层指标在完整 held-out section 或完整 holdout subvolume 汇总。global confusion 先累计全部像素，再算 Accuracy、mIoU、macro-F1 和每类 support/IoU/F1；另报 section-level 分布，不能用“每图 mIoU 再平均”替代主指标。
3. **空间 test 先冻结**：沿一个轴留连续 development/guard/test，并证明 sample、section、体素三层零重叠。若同时用 IL/XL，必须按 3D 体块归属，不能按图像文件独立随机分。
4. **development 5-fold**：沿空间 group 做 blocked CV；每个 fold 的 train/val 间保留物理 buffer。若某 rare class 无法在 5 个 val core 都出现，诚实降折或报告 support=0，不通过读取 test 调整。
5. **fold-local fit**：z-score、class weight、采样权重、target mapping、温度/阈值只 fit fold-train；val/test 只 transform。augmentation 在线生成，不能让同一原 patch 的增强副本跨 split。
6. **训练与选择**：固定更新数与有效像素预算；保存 min-val-loss checkpoint，同时报告 OOF mIoU/macro-F1。HPO 仅在 development nested/OOF 上；选定配置后写死并 refit all development。
7. **推理**：full section/volume 滑窗，重叠区使用预先固定的 uniform 或 Gaussian blending；softmax 只在 logits 之后。阈值/后处理不得在 outer holdout 调整。
8. **当前 holdout 的定位**：因 P4/Stage4 已消费，继续可作 `previously_seen_reusable_holdout` 回归集，但不能产生 `fresh_blind=true`。若需要真正盲测，必须由数据 owner 冻结一个从未查看的新空间体块或外部 survey；不能重置状态文件。

## 9. 推荐协议 B：文献可比的独立复现协议

协议 B 不覆盖协议 A，而是新建**不可混榜**的 benchmark task。

### B1. F3-Alaudah6

- 用 Alaudah Zenodo 的六类标签版本和裁剪体；精确复建 train NW、test1 SW、test2 east 三个空间块。
- development validation 只能从 NW train 内再做带 buffer 的 blocked folds。test1/test2 不参与 normalization、class weight、epoch、阈值或 model selection。
- 主报 Alaudah 的 PA、CA、MCA、FWIU以便历史对照；同时新增 global mIoU、macro-F1、per-class support/IoU/F1，明确两组定义。
- 先分别报 IL-only、XL-only，再报预先固定的 IL/XL logit averaging；不选择对 test 最有利的融合权重。

### B2. Penobscot published-reproduction lanes

- Penobscot 没有与八类完全匹配的 canonical frozen split。若复现 Civitarese，应建立 `Penobscot7-merged`，记录 2/3 合并和清理后的 459 IL 清单；若复现 EarthAdaptNet 三类 DDA，应建立 `Penobscot3-DDA`。
- 严格照论文 split 只用于“published-protocol reproduction”；另加一个带 guard 的 sensitivity split。两者分别报告，不能把无 buffer 的高分冠以独立 blind test。
- 若目标仍是完整八类，则回到协议 A；不得为了复现高分临时合并 rare class。

## 10. 首批十模型的公平测试要求

当前十个 `model_id`：`smp_unet_r18`、`smp_deeplabv3plus_r18`、`smp_unetpp_r18`、`smp_fpn_r18`、`torchvision_lraspp_mbv3`、`deepseismic_patch_skip`、`deepseismic_seresnet_unet`、`hf_segformer_b0`、`sfm_base_facies`、`monai_unet3d`。

### 10.1 所有模型共同硬门

1. F3/Pen 分别建 head、checkpoint、preprocessor、OOF、leaderboard；禁止跨任务预处理或合并排名。
2. 同 task/lane 使用同一冻结 split manifest、fold core、buffer、三 seed、训练样本/有效像素预算、更新数、验证频率和 wall-clock 上限。结构化 SKIP 不可用另一个模型补位。
3. 输入 seismic amplitude 的 fold-train z-score 相同；不默认平滑。标签 identity transform；raw logits 进入 loss，softmax 只在推理/校准。
4. 相同 loss lane 内使用相同 weighted CE；若测试 Dice/Lovász，另开 loss-ablation lane，所有模型一起切换，不能只给某模型特权。
5. 参数量、训练 wall time、峰值 VRAM、best update、每 fold/seed 指标、失败原因都保留。排名以 task 内 OOF global mIoU 为主，macro-F1 为辅，完成率 <80% 不排名。
6. 预训练模型单列 pretrained lane，必须锁 source commit、checkpoint hash、权重许可和预训练数据；不得与 scratch 模型直接合并一个榜。
7. full-section/full-volume inference 采用同一 patch size/stride/blending/TTA。默认无 TTA；若做 flip TTA，所有兼容 2D 模型一起做预注册消融。
8. test/known-holdout 只执行冻结 winner。非 winner 不得读取 test 以补表或挑代表图。

### 10.2 结构差异的公平 lane

| lane | 模型 | 公平边界 |
|---|---|---|
| 2D scratch | 四个 SMP、LRASPP、SegFormer | 相同单 section 输入 `[B,1,H,W]`、scratch、weighted CE、更新/像素预算；这是可直接主排名的六模型 |
| seismic legacy scratch | `deepseismic_patch_skip`、`deepseismic_seresnet_unet` | 按作者结构精确移植并锁 source；若依赖或结构无法重现则 SKIP。可与 2D scratch 报并列表，但同时标注 legacy training 机制是否被统一 |
| seismic pretrained | `sfm_base_facies` | 仅在权重许可与 hash 明确后进入；scratch 与 pretrained 分榜。否则保持许可 gate SKIP |
| true 3D scratch | `monai_unet3d` | 输入必须是连续 `[B,1,D,H,W]` 空间子体；按相同 development/test 体块取样，子体不得跨 buffer。按“处理体素数 + updates + wall time”对齐，不能伪装成 2D |

## 11. 最小复现实验（不触碰现有结果的新增 development-only 计划）

### E0：零训练 split/标签审计

- 对两个 task 生成 section、patch 和 voxel-coordinate 三层集合；断言 train/val/buffer/test 交集为 0。
- 若未来启用 crossline，显式计算 IL/XL 体素交点归属；任一跨 split 交点即 fail closed。
- 对每 fold 只从 fold-train 扫描 normalization 与 class support；输出 rare/absent class，不为补类重切。

### E1：先验证当前 40-update pilot 是否欠拟合

- 只用当前两个 Stage3 winner；每 task 选固定 fold-0，seed 2693。
- 同一数据与 optimizer 比较 40、400、1,000 updates；不得 early-stop，保存完整 train/val loss 与 min-val checkpoint。
- 只看 development val，报告 unique samples seen、有效像素数、mIoU/macro-F1/per-class。若 400→1,000 已平台或过拟合，再冻结正式预算；不读取 known holdout。

### E2：类别不均衡/loss 小消融

- 在 E1 冻结的充分预算下，对两 winner 比较三种预注册 loss：weighted CE；weighted CE + soft Dice；weighted CE 预热后 Lovász-Softmax。
- 2 tasks × 1 winner × 3 losses × 3 seeds × 1 locked fold = 18 cells；class weight 仅 fit fold-train。
- 以 rare-class per-class IoU、macro-F1 和 global mIoU 联合解释；不能只按 accuracy 选择。若 Lovász 实现/许可不清楚则结构化 SKIP。

### E3：2D/2.5D/3D 信息量消融

- 2D：中心 inline；2.5D：同一 split 内的 `k-1,k,k+1` 三通道；3D：连续小体块。边界邻片落入 buffer/另一 split 时丢弃该样本，不用复制 test 邻片。
- 使用等有效体素预算；至少 3 seeds、同一 locked fold。只有 adapter、shape、显存测试通过后扩到 5 folds。

### E4：patch 指标与完整 section 指标差异

- 在 development OOF 上同时评估当前固定 128×128 样本和完整 val section 滑窗；固定 stride/blending，无 TTA。
- 输出 global confusion、每 section 指标分布、逐类 support/IoU/F1。该实验直接量化 Stage3/4 patch 采样对类别分布与分数的偏差。

推荐执行顺序为 **E0 → E1 → E4 → E2 → E3**。E0/E1 能先判断数据合同与训练不足，避免在明显欠拟合的 40-update 条件下对十模型或复杂 loss 作错误淘汰。

## 12. 可保留、应推翻与应补实验的最终清单

### 保留

- F3 与 Penobscot 独立 task、独立 label version/head/榜单。
- 连续 inline 外层切分、guard、development blocked 5-fold、fold 内 buffer 与 OOF 全覆盖。
- 只从 fold-train 拟合 z-score/class weight；`denoise_identity`；raw logits + CE、softmax inference-only。
- Accuracy + global mIoU + macro-F1 + per-class support/IoU/F1 + confusion/ECE/reliability/代表剖面。
- checkpoint hash、source lock、seed、预算、GPU 和 test-firewall 证据；缺依赖/许可时结构化 SKIP。

### 推翻或降级表述

- 推翻“当前 split 是 F3/Penobscot 公共标准 split”的任何表述；应写 track-defined spatial split。
- 推翻“Stage3 40 updates 是模型最终能力比较”；它只是小预算可运行性/初筛。
- 推翻“Stage4 是 fresh blind test”；它是 previously-seen reusable holdout confirmation。
- 推翻把当前十类/八类 mIoU 与 Alaudah 六类、Civitarese 七类、EarthAdaptNet 三类或 TGS 二类数字直接排表比较。
- 推翻仅凭 accuracy/FWIU 或 image-averaged mIoU 下结论；rare facies 必须看逐类 support/IoU/F1。

### 补实验

- 充分训练曲线、完整 section/volume 推理、loss 消融、2D/2.5D/3D 空间输入消融。
- 若需要公共可比性，新增 F3-Alaudah6 独立复现 task；若需要 Pen 文献复现，新增清晰命名的 merged-label lane。
- 若需要 fresh-blind 结论，由 owner 冻结从未查看的新空间体块或外部 survey；不能通过改状态文件把旧 holdout “洗回” blind。

## 13. 局限与残余风险

- F3 Silva 论文、Zenodo 版本摘要和本地文件计数存在文字/版本差异；正式复现必须引用具体 Zenodo record、文件清单和 SHA，而不能只写“F3”。
- 多篇早期论文未公开代码、精确 split 索引、loss 或许可证；本报告把这些字段标为 NR，不能将论文高分视为可复现事实。
- 当前 test 由保存的 patch 组成，不覆盖完整空间体；采样会改变类别 support。Stage4 的低 mIoU 与文献高分的差距混合了标签、split、预算、评估单元和模型差异。
- guard 以 inline 数而非米制/相关长度定义。建议在不读取 test 标签的情况下，用 development seismic amplitude 的空间自相关长度解释 buffer 是否足够；不得据 test 表现反调 guard。
- TGS 图片缺采集平面坐标，随机 image CV 的独立性不可审计；它只支持 loss/TTA 的候选性，不支持 F3/Pen split 设计。

## 14. 一手来源索引

1. Silva et al., *Netherlands Dataset: A New Public Dataset for Machine Learning in Seismic Interpretation*: [arXiv](https://arxiv.org/abs/1904.00770), [Zenodo](https://doi.org/10.5281/zenodo.1422787).
2. Baroni et al., *Penobscot Dataset: Fostering Machine Learning Development for Seismic Interpretation*: [arXiv](https://arxiv.org/abs/1903.12060), [Zenodo](https://doi.org/10.5281/zenodo.1324463).
3. Alaudah et al., *A Machine-Learning Benchmark for Facies Classification*: [arXiv](https://arxiv.org/abs/1901.07659), [Zenodo](https://doi.org/10.5281/zenodo.3755060), [author GitHub](https://github.com/yalaudah/facies_classification_benchmark).
4. Chevitarese et al., *Seismic Facies Segmentation Using Deep Learning*: [AAPG Search and Discovery PDF](http://www.searchanddiscovery.com/documents/2018/42286chevitarese/ndx_chevitarese.pdf), DOI 10.1306/42286Chevitarese2018.
5. Civitarese et al., *Semantic Segmentation of Seismic Images*: [arXiv](https://arxiv.org/abs/1905.04307).
6. Nasim et al., *Seismic Facies Analysis: A Deep Domain Adaptation Approach*: [arXiv](https://arxiv.org/abs/2011.10510), [IEEE DOI](https://doi.org/10.1109/TGRS.2022.3151883).
7. Gutierrez et al., *On the Performance Evaluation of Deep Learning Models for Seismic Facies Segmentation*: [Geophysical Prospecting DOI](https://doi.org/10.1111/1365-2478.70104).
8. Sheng et al., *Seismic Foundation Model (SFM)*: [arXiv](https://arxiv.org/abs/2309.02791), [author GitHub](https://github.com/shenghanlin/SeismicFoundationModel).
9. Shi et al., *SaltSeg: Automatic 3D Salt Segmentation Using a Deep Convolutional Neural Network*: [Interpretation DOI](https://doi.org/10.1190/INT-2018-0235.1), [author-hosted PDF](https://cig.ustc.edu.cn/_upload/tpl/05/cd/1485/template1485/papers/shi2019saltSeg.pdf).
10. Karchevskiy et al., *Automatic Salt Deposits Segmentation*: [arXiv](https://arxiv.org/abs/1812.01429), [Kaggle official competition](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview), [author GitHub](https://github.com/K-Mike/Automatic-salt-deposits-segmentation).

补充交叉核查但未计入十项主矩阵：Zeng et al. 的 SEAM salt segmentation（[arXiv](https://arxiv.org/abs/1812.01101)）支持 CE→Lovász、5-fold ensemble 与 IoU 报告；SaltISNet3D（[Remote Sensing DOI](https://doi.org/10.3390/rs15092319)）因需要交互提示而改变了任务输入合同，只作为 3D/interactive 旁证，不纳入公平十模型主榜。
