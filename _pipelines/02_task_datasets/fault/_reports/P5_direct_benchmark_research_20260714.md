# P5 断层直接基准调研：真实地震、合成到真实迁移与不完整标注

- 调研日期：2026-07-14
- 本地基线：`p5-stage3-fault@27c9333a9c90f835c1da5fa3dfc67492b0f3d92c`
- 范围：只纳入直接研究三维地震断层检测/分割、真实地震迁移、稀疏或不完整断层标注的工作；普通图像分割、医学分割和泛化空间交叉验证不作为核心证据。
- 证据口径：核心矩阵含 13 项去重工作/官方挑战，13 项均至少有论文原文、官方数据/挑战页或作者仓库这一类一手来源。未能从一手来源确认的字段统一写为“未发现”，不以综述或博客补齐。
- 搜索结论：未发现 Kaggle 官方举办的同任务三维地震断层分割竞赛。Kaggle 上可找到 FaultSeg3D 合成数据的社区镜像，但它不是独立竞赛，也不计入下列 13 项。直接同任务的官方盲测证据来自 ThinkOnward/FORCE 与 Dark Side of the Volume。

## 1. 执行摘要

当前 Stage3/Stage4 的 `blocked/not_rankable` 结论应保留。所审计的 13 项直接工作没有一项证明“仅有 fault-stick 正例时，stick 之外的真实体素可以无条件视为可靠负例”。前人实际采用了四种不同、不可混同的负例来源：

1. 合成生成器给出的稠密真值；这是本次证据中唯一不依赖人工覆盖假设的可靠负例来源。
2. 人工稠密解释的公开体；但 Thebe 明确只解释特定位移尺度和深度范围，模型预测到未标注细小断层时会被表面上记为假阳性，因此“0”仍只在经过覆盖审计的区域内可视为确认负例。
3. 稀疏人工切片：切片之间明确设为 `unknown/-1` 并屏蔽梯度；切片内的 0 仍可能是假负例。Masked/Dynamic loss 只能降低错误标注影响，不能把它升级为审计负例。
4. 相干体、RANSAC 或阈值产生的“非断层”样本；这些只是 proxy negative，不能进入正式监督指标或解锁最终排名。

因此，本项目可以立即做“合成预训练”或“只在有效标签上监督、在 unknown 上做一致性约束”的开发性研究，但在取得覆盖审计负例和合法空间/体级开发折之前，只能作为独立的弱监督/迁移 lane，继续 `not_rankable`，不能 refit、不能选择 frozen winner、不能访问 frozen test。

## 2. 本地 Stage3/Stage4 证据基线

### 2.1 原始证据与哈希

| 本地证据 | SHA-256 | 已核对事实 |
|---|---|---|
| `_outputs/p5_stage3/p5_stage3_data_manifest.json` | `66f7f922fad93dfb8adc218cfcbebd65cbc0b7e4f061dca99a1feb3e884083a1` | 32 positive、0 verified negative、2016 unknown，共 2048 体素；`requested_folds=5`、合法有效折为 0；无原始体或 test label 读取。 |
| `_outputs/p5_stage3/p5_stage3_summary.json` | `52351babd768dc89dcf86b753a15e9392193ca140ce4aedecd020506bc8f6985` | `reason_code=NO_VALID_FAULT_DEVELOPMENT_FOLDS`；未训练、未 HPO、未临时切分、未访问 frozen test。 |
| `_outputs/p5_stage4_confirmation/p5_stage4_confirmation.json` | `a7eb96b8734bad002621cae34e566660979556852a2b439cbc86eca2ce79cfdc` | `frozen_winner=null`、`refit_executed=false`、`holdout_accessed=false`；历史 test 只作 regression evidence。 |
| `_outputs/p5_stage4_confirmation/p5_stage4_visualization_reuse.json` | `dde5a8532aeae829d08963800430d4314a84739c9a41a373d87d83611a679138` | 只复用并校验 Stage3 readiness/negative/unknown 覆盖图，没有伪造预测。 |

以上路径均相对于 `_pipelines/02_task_datasets/fault/`。当前阻塞码为：

- `AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING`
- `COVERAGE_AUDITED_UNKNOWN_MASK_MISSING`
- `DEVELOPMENT_SPLIT_NOT_FEASIBLE`
- `NO_FROZEN_STAGE3_TOP3`

### 2.2 当前科学红线

- fault-stick 像素为正例；未标注体素默认 unknown，`valid_label_mask=false`。
- `proxy_mask` 与 verified-negative mask 分离；proxy 不得启用正式 loss、metric 或 fold。
- 请求 5-fold；只能按独立空间块/体及标签支持诚实降折，不得临时随机 20% 切分。
- 开发 API 不接收 test；阈值只能由 pooled OOF 冻结；历史 `audited_v2` 仅是回归证据。
- 当前没有合法 fold，故 Stage4 必须保持 blocked，不得 refit 或打开 holdout。

这些红线比多数已发表工作更严格，但与已知不完整标注事实相符，并非过度保守。

## 3. 去重证据矩阵

| ID | 工作/年份 | 直接相关性 | 真实/合成与标注问题 | 一手来源 | 可复现状态 | 对本项目的核心证据 |
|---|---|---|---|---|---|---|
| S01 | FaultSeg3D, Wu 等, 2019 | 直接做 3D 断层概率分割 | 200+20 个稠密合成体；多个真实体仅定性 | [论文](https://doi.org/10.1190/geo2018-0646.1)、[作者代码](https://github.com/xinwucwp/faultSeg) | 代码、预训练入口可用；非商业许可 | 合成稠密标签能给可靠负例，但不能证明真实体未标注区为负例。 |
| S02 | FaultNet3D, Wu 等, 2019 | 3D CNN 同时预测断层存在、走向、倾角 | 90 万合成小体；真实应用定性 | [论文](https://doi.org/10.1109/TGRS.2019.2925003)、[作者 PDF](https://cig.ustc.edu.cn/_upload/tpl/05/cd/1485/template1485/papers/wu2019faultNet3d.pdf) | 未发现作者代码/权重 | 可靠负类来自合成生成过程；几何多任务不解决真实 unknown。 |
| S03 | Cunha/Pochet 等, 2020 | 明确研究 synthetic-to-real 地震断层迁移 | 合成预训练；F3 单解释剖面少量真实 patch | [论文](https://doi.org/10.1016/j.cageo.2019.104344) | 未发现代码/权重 | 单剖面随机分层 5-fold 不能证明空间独立，未解释区存在假负例风险。 |
| S04 | CNN for Fault Interpretation / Thebe, An 等, 2021 | 真实三维地震体上的 2D/3D断层解释基准 | Thebe 真实体；专家范围受限的不完整标注 | [论文](https://doi.org/10.1016/j.cageo.2021.104776)、[官方数据](https://doi.org/10.7910/DVN/YBYGBK)、[作者代码](https://github.com/anyuzoey/CNNforFaultInterpretation) | 数据、代码、checkpoint 可得 | 连续空间切分优于随机切片，但无 purge；公开 0 标签并非天然审核负例。 |
| S05 | Attention weak supervision, Dou 等, 2021 | 专门用稀疏 2D 切片训练 3D 断层网络 | Shengli 真实体每 30 inline 标一张；切片间 unknown | [论文](https://arxiv.org/abs/2105.03857)、[正式 DOI](https://doi.org/10.1109/TGRS.2021.3113676) | 未发现官方代码/权重 | 明确 `-1` 屏蔽 unknown，是当前 mask 语义的一手支持；验证样本取自测试区是反例。 |
| S06 | Automatic label generation + transfer, Yan 等, 2021 | 直接为真实地震断层自动造标签并迁移 | 合成预训练；相干属性/NMS/RANSAC 生成真实 proxy 标签 | [论文全文](https://doi.org/10.3390/en14123650) | 未发现代码/权重 | 阈值产生的“无断层”是 proxy，不是人工覆盖审计负例。 |
| S07 | Focal-loss transfer, Wei 等, 2022 | F3 真实地震断层 patch 检测与合成迁移 | 合成 + 部分真实标签，严重类别不平衡 | [论文](https://doi.org/10.1016/j.cageo.2021.104968)、[作者代码](https://github.com/weixiaoli125/fault-detection) | 代码和 checkpoint 可得；软件许可证未发现 | 源码把中心非 1 或无正例 patch 当负类，在不完整标注下会引入假负例。 |
| S08 | MD Loss, Dou 等, 2022 | 专门处理稀疏切片、unknown 与切片内假负例 | Shengli 真实体 + FaultSeg3D 合成体 | [论文](https://arxiv.org/abs/2110.05319)、[正式 DOI](https://doi.org/10.1109/TGRS.2022.3196810) | 未发现官方代码/权重 | unknown mask + 对切片内 false-negative 降权可用作弱监督机制，但不生成可靠负例。 |
| S09 | FaultSSL, Dou 等, 2024 | 多真实地震体稀疏标签半监督 3D 断层检测 | 每体 1–3 张标注切片；更多无标签真实体 | [论文](https://arxiv.org/abs/2309.02930)、[正式 DOI](https://doi.org/10.1190/geo2023-0550.1) | 未发现官方代码/权重 | unlabeled 用一致性/伪标签而非硬负类，最接近当前 unknown 红线；独立盲测仍不足。 |
| S10 | Seismic Fault SAM, Guo 等, 2024 | 将 SAM 适配到 Thebe 真实断层分割 | 5 相邻 crossline 输入、中央切片输出 | [论文](https://arxiv.org/abs/2407.14121) | 未发现官方 fault 代码/权重 | 复用 Thebe 连续切分；OIS/ODS 在测试标签上寻阈值，不符合 frozen-test 规则。 |
| S11 | FORCE 2020 ML Competition | 官方真实地震断层盲测挑战 | 合成训练，未知真实 Ichthys 最终体，专家主观评分 | [官方挑战页](https://thinkonward.com/app/c/challenges/force-seismic)、[公开方案 1](https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition)、[公开方案 2](https://github.com/satyakees/FaultNet) | 挑战资源和社区方案可得 | 证明 synthetic-only 到真正未见真实体可做盲评；但无稠密真值，不能产生本项目正式负例。 |
| S12 | Dark Side of the Volume | 官方 3D 断层掩码盲测挑战 | 400 个 3D 体和稠密 mask；官方页未说明真实/合成来源 | [官方挑战页](https://thinkonward.com/app/c/challenges/dark-side)、[官方获胜方案](https://huggingface.co/thinkonward/challenges/tree/dark-side) | 六个获胜方案和权重可得 | 稠密标签、独立 final holdout 与服务器限时验证了可审计生命周期；数据来源未知，不能直接外推到稀疏真实 stick。 |
| S13 | Large benchmark, Quesada 等, 2025/2026 | 统一比较 FaultSeg3D、CRACKS/F3、Thebe 的域迁移 | 合成 + 两套真实断层数据 | [论文](https://arxiv.org/abs/2505.08585)、[官方代码](https://github.com/olivesgatech/large-bench-geo) | 代码可得；软件许可证未发现 | 训练集定 ODS 阈值后测测试集是可保留做法；也直接报告未标注真断层会被误判为 FP。 |

## 4. 逐项证据卡

除非另有说明，“未发现”均指截至 2026-07-14 未能从上述一手来源确认。

### S01. FaultSeg3D

- **为什么直接相关**：以简化 3D U-Net 对三维地震体逐体素输出断层概率，是此后多项地震断层工作的合成预训练源。
- **数据与输入体**：200 个训练、20 个验证合成体；每体 `128×128×128`，单通道振幅；论文另在 F3、Clyde、Costa Rica、Campos、Kerry、Opunake 等真实体上展示结果。
- **标签、负例与 unknown**：断层面与地震响应同步生成，得到稠密二值 mask；非断层体素可视为生成器内可靠负例。无 unknown mask。真实展示体没有用于定量监督的稠密真值。
- **切分、buffer/purge**：200/20 按独立生成体切分；未发现独立 frozen test。独立生成体之间不需要空间 buffer，但验证集被用于训练选择，不是盲测。
- **loss、采样、增强**：binary cross-entropy；batch 1；每体 z-score；源码含翻转增强；Adam `1e-4`、100 epochs。类别采样策略未发现。
- **指标、阈值、盲测**：合成验证体报告 PR/ROC；推理经 sigmoid。固定二值阈值的完整规定未发现。真实体为定性展示，不是盲测。
- **代码、权重与许可**：作者仓库 commit `0ab8ba1c10cb9e5748b0129bfe1a2fe3031b80fe`；README 提供预训练模型入口。README 声明 CC BY-NC 4.0、个人/研究使用，不能默认商业复用。
- **对本项目的判断**：可作为 Protocol B 的合成预训练基线；不能用其合成 0 标签替代真实数据的覆盖审计。

### S02. FaultNet3D

- **为什么直接相关**：输入三维地震小体，联合预测断层概率、走向和倾角，并重建 fault cells/skins。
- **数据与输入体**：约 900,000 个合成体；输入为 48 个垂向样点、32 inline、32 crossline。577 类为无断层一类加 576 个走向/倾角组合。
- **标签、负例与 unknown**：无断层类和几何类均由合成过程给出，负例可靠；无真实 unknown 建模。
- **切分、buffer/purge**：训练/验证/测试的精确体数及 purge 未发现。
- **loss、采样、增强**：七层 CNN；精确 loss、类别采样和增强未发现。
- **指标、阈值、盲测**：网络输出类别概率，随后用各向异性高斯核叠加并生成断层面；精确阈值和独立盲测未发现。真实案例为应用展示。
- **代码、权重与许可**：未发现作者官方代码、权重或软件许可证。不能把第三方同名 FaultNet 仓库当作其官方实现。
- **对本项目的判断**：几何辅助任务可以列为以后架构升级项，但不会解除真实负例/unknown 数据门禁。

### S03. Synthetic-to-real transfer on F3

- **为什么直接相关**：题目和实验均直接研究由合成地震断层 CNN 向真实 F3 数据迁移。
- **数据与输入体**：合成 patch 预训练；真实端来自 F3 的一张人工解释剖面，构造小型且不平衡的 patch 数据。精确 patch shape 未发现。
- **标签、负例与 unknown**：论文说明人工只标出部分断层；真实负 patch 的覆盖审计规则和 unknown mask 未发现。因此未标注像素不能据此视为确认背景。
- **切分、buffer/purge**：对同一剖面的 patch 做 stratified random 5-fold；未发现按空间块、体切分或邻块 purge。相邻 patch 的相关性使其不能作为本项目的空间独立证据。
- **loss、采样、增强**：比较全量 fine-tune 与 CNN 特征提取后接 SVM；精确 loss、增强和采样规则未发现。
- **指标、阈值、盲测**：在该真实数据上比较分类表现；完整指标、阈值未发现。不是未知体盲测。
- **代码、权重与许可**：未发现作者官方代码、权重或软件许可证。
- **对本项目的判断**：支持“合成预训练可降低真实标注需求”，但其随机 patch CV 不可复制为正式验证协议。

### S04. Thebe 真实数据与 CNN 基准

- **为什么直接相关**：发布大规模真实地震体、fault sticks、binary masks 和 train/validation/test 切片，并专门讨论相邻切片泄漏。
- **数据与输入体**：Thebe 体约为 1803 crosslines、1537 samples、3174 inlines；Harvard Dataverse v4 约 53.09 GB、78 个文件，含原始 sticks、掩码及切分资产。
- **标签、负例与 unknown**：解释者主要标记位移大于约 20 m、深度约 2–4 km 的断层，浅部、深部和细小断层可能未标。发布 mask 仍以 1/0 表示，未提供 unknown mask。论文观察到模型找出未标注细小断层，说明部分“FP”实际处于 unknown 语义。
- **切分、buffer/purge**：前 900 crosslines 训练、随后 200 验证、最后 703 测试；作者明确反对随机切片，因为相邻切片近乎相同。三个连续区之间未发现 buffer/purge。
- **loss、采样、增强**：裁 `96×96` patch、stride 48；过滤正例像素少于 3% 的 patch，得到约 181,029 train、64,317 validation patch。分割网络 BCE；HED 使用 weighted CE；sigmoid 推理；使用图像增强，完整参数清单未发现。
- **指标、阈值、盲测**：边缘式 OIS/ODS/F 类指标；测试标签随数据公开，非真正封存盲测。精确阈值选择规则应在复现时逐版本核对。
- **代码、权重与许可**：作者仓库 commit `58eae5db2312feca003b9eb179fd9172beeeab5d`，GPL-3.0，含 checkpoint。Dataverse 页面为 CC0 加附加/修改条款，部分文档又标 CC BY 4.0；重分发前必须逐文件核对，不应笼统写为单一许可。
- **对本项目的判断**：可采用“连续空间切分”，但必须补 purge、coverage/unknown 审计；正例富集 patch 采样不可定义负例真实性。

### S05. Attention-based weak supervision with sparse slices

- **为什么直接相关**：目标正是用极少二维人工断层切片训练三维断层分割网络。
- **数据与输入体**：Shengli 六个真实工区；一个工区训练、其余工区测试。人工约每 30 个 inline 标一张，一个体约 12 张。取 `64³` patch、stride 25，删除断层体素少于 64 的 patch后随机保留 5000 个。
- **标签、负例与 unknown**：标注切片中断层为 1、背景为 0；切片之间为 `-1`，在 λ-BCE/λ-smooth-L1 中梯度为 0。该机制明确承认体内大部分区域 unknown，但没有证明标注切片内所有 0 都是可靠负例。
- **切分、buffer/purge**：真实实验从被称作测试的工区 Mode B 随机取 500 个样本作 validation，再据此选 checkpoint，构成测试区信息参与选择。合成 patch 另做随机 5-fold；未发现空间 buffer/purge。
- **loss、采样、增强**：λ-BCE、λ-smooth L1 与 attention activation module；真实训练 batch 32、Adam `1e-3`、35 epochs；随机旋转。正例富集采样如上。
- **指标、阈值、盲测**：Precision、Recall、IoU、Dice、Hausdorff；sigmoid 推理；固定阈值未发现。真实结果以定性为主，不是严格盲测。
- **代码、权重与许可**：未发现作者官方代码、权重或软件许可证。
- **对本项目的判断**：`unknown=-1` 屏蔽 loss 可直接借鉴；从 test 工区抽 validation 的做法必须废弃。

### S06. Attribute/RANSAC 自动标签迁移

- **为什么直接相关**：专门研究如何用合成数据预训练，再从真实地震属性自动构造断层标签进行 fine-tune。
- **数据与输入体**：合成 2D patch 为 `48×32`；32,000 fault、18,000 non-fault 用于训练，5,000 validation。真实第一体为 `751×440×201`，另在中国东部与 Kerry 体展示迁移。
- **标签、负例与 unknown**：真实伪标签由 forward/backward filter 的 coherence、NMS、阈值/半径种子和 RANSAC 线段生成。全零“非断层”patch 也由属性阈值筛选，属于 proxy negative；无 unknown mask，也无人工覆盖审计。
- **切分、buffer/purge**：第一真实体产生超过 10,000 fault 与 20,000 non-fault 候选，随机取 1000/2000 fine-tune、另随机取 1000 validation。未发现空间块切分或 purge；同一体又作为主要展示对象，不是独立测试。
- **loss、采样、增强**：U-Net、sigmoid、balanced cross-entropy；合成 Adam `1e-4`、batch 256、20 epochs；迁移 `5e-5`、batch 128、10 epochs。真实采样如上；完整增强未发现。
- **指标、阈值、盲测**：主要用可视化和属性对比；独立真实定量指标、固定阈值与盲测未发现。
- **代码、权重与许可**：未发现官方代码、权重或软件许可证；论文网页为 CC BY 4.0，不等于软件许可。
- **对本项目的判断**：可把属性/RANSAC 结果放进 `proxy_mask` 做辅助 loss 或采样，但绝不能写入 verified-negative mask。

### S07. Focal-loss synthetic-to-real transfer

- **为什么直接相关**：对 F3 真实断层 patch 做检测，显式以 focal loss 处理断层/非断层不平衡并从合成数据迁移。
- **数据与输入体**：2D `45×45` patch 分类；合成地震预训练，F3 部分真实标注迁移。
- **标签、负例与 unknown**：公开源码将不含 `label==1` 的 patch 视为负例，迁移预处理还把中心像素非 1 作为负类；没有 coverage/unknown mask。因此在不完整解释下会把漏标断层当负例。
- **切分、buffer/purge**：源码按合成图像前 80%/后 10%/末 10% 组织资产，迁移数组又作约 90/10 train/validation；未发现体级独立 test、空间块或 purge。
- **loss、采样、增强**：focal loss，源码 `gamma=2`，synthetic `alpha=0.75`、transfer `alpha=0.85`；正负 patch 采样和形态学/Hough 后处理。完整增强参数未发现。
- **指标、阈值、盲测**：Accuracy、Precision、Recall、F1、Specificity、AUC；二类 softmax argmax 相当于 0.5 决策。真实展示非盲测。
- **代码、权重与许可**：作者仓库 commit `5c24aee162c439507f1bc7c5303437dac262eb27`，含 checkpoint；仓库未发现明确软件许可证，因此默认不可假定再分发权利。
- **对本项目的判断**：focal loss 可作为模型内类别不平衡候选，但不能修复错误负标签；未审计 0 标签仍必须 mask 掉。

### S08. Missing-label Dynamic Loss

- **为什么直接相关**：针对稀疏二维断层解释训练三维 CNN，并专门讨论两类缺标：切片间无标签和切片内 false negative。
- **数据与输入体**：Shengli 三个工区 A/B/C；约每 30 inline 标一张。取 `128³` patch、stride 35，移除断层体素少于 128 的 patch，保留约 300 个真实样本；与 FaultSeg3D 合成体混训。
- **标签、负例与 unknown**：标注切片为 1/0，切片间为 `-1` 并 mask；MD Loss 对标注切片内可能的 false-negative 动态降权。它是鲁棒化，不会产生已审核负例。
- **切分、buffer/purge**：采用 A+B 训练/C 测试等跨工区组合；验证始终为合成数据。合成 200/20 体再上采样/随机裁为约 600 train/120 validation。未发现工区内 buffer/purge。
- **loss、采样、增强**：Dice 预训练约 200 epochs，再用对比 loss 约 200 epochs；Adam `1e-3`、batch 10；`0/90/180/270°` 随机旋转。
- **指标、阈值、盲测**：合成验证报告 IoU，阈值 0.5；真实工区主要定性。跨工区比同体随机 patch 更好，但没有独立封存的真实定量 test。
- **代码、权重与许可**：未发现作者官方仓库、权重或软件许可证。
- **对本项目的判断**：可在 Protocol B 作为“切片内假负例鲁棒性”消融；不得用 MD 权重后的 0 反向声明为 audited negative。

### S09. FaultSSL

- **为什么直接相关**：直接针对三维地震断层的少标签/无标签半监督，联合真实与合成体。
- **数据与输入体**：有标签真实体包括 Kerry、F3、Poseidon、Canning、Opunake、Costa Rica、Clyde、Niuzhuang，每体仅 1–3 张标注切片；另有 Parihaka、Ichthys、Adele、Sinopec 无标签体；合成端沿用 FaultSeg3D 200 体。
- **标签、负例与 unknown**：标注切片使用 MD mask；无标签体通过 Mean Teacher、Panning Consistency 和 Patching Consistency 生成一致性/伪标签信号，不直接把 unlabeled 变负类。伪标签的置信度并不等同人工审核真值。
- **切分、buffer/purge**：一手论文未发现可复算的体级 train/validation/test manifest、buffer 或 purge；部分报告体同时参与适配和定性展示，不能当作独立盲测。
- **loss、采样、增强**：HRNet；合成预训练约 30k steps，半监督约 160k steps，AdamW `1e-3`；三轴翻转、不同平面旋转、强度/滤波等增强；一致性和 MD 监督组合。
- **指标、阈值、盲测**：真实结果主要定性与消融；可审计的固定阈值、独立真实数值 test 未发现。
- **代码、权重与许可**：未发现作者官方代码、权重或软件许可证。
- **对本项目的判断**：这是采用 masked weak/SSL 的最直接一手依据；只能在开发 lane 使用，伪标签不得进入正式 GT 或 frozen-test 选择。

### S10. Seismic Fault SAM

- **为什么直接相关**：在 Thebe 真实地震断层数据上把 SAM 改为自动 2.5D 断层分割器。
- **数据与输入体**：5 个相邻 crosslines 作输入、中央切片作输出；沿用 Thebe 前 900/中 200/后 703 连续切分。
- **标签、负例与 unknown**：沿用 Thebe 二值 mask，无新 coverage 或 unknown 定义；因此继承 Thebe 的不完整解释限制。
- **切分、buffer/purge**：连续空间切分但未发现边界 purge；公开 test 标签可见。
- **loss、采样、增强**：冻结大部分 SAM-B/H 参数，只训练约 2% adapter/decoder 参数；300 epochs、A6000；水平翻转和随机仿射旋转/缩放/平移。分割 loss 的完整精确定义未发现。
- **指标、阈值、盲测**：OIS 为每张测试图寻找最佳阈值，ODS 为整个测试集寻找全局最佳阈值；测试图使用过 0.7 阈值。该阈值流程读取 test label，不符合本项目 OOF-only 规则。测试集公开且参与阈值选择，非盲测。
- **代码、权重与许可**：未发现作者官方 fault 适配代码、fault 权重或软件许可证；基础 SAM 许可不能自动覆盖未发布适配实现。
- **对本项目的判断**：可作为未来架构候选，不能把其 OIS/ODS 数值直接列为本项目 frozen-test 可比基准。

### S11. FORCE 2020 blind real-seismic challenge

- **为什么直接相关**：官方任务要求从合成训练资源出发，在最后一天才收到未见过的真实 Ichthys 3D 地震体并返回断层概率。
- **数据与输入体**：训练期强烈鼓励合成；最终真实 SEG-Y 与训练调查在频率、噪声和地理位置上有域偏移。提交完整概率体及指定 inline/crossline。
- **标签、负例与 unknown**：官方没有发布稠密真实 GT；负例/unknown 的精确定义未发现，也不适用于主观专家评分。各队合成 mask 内可有可靠负例。
- **切分、buffer/purge**：最终体在最后一天发放，15 小时内返回，构成真正未见体测试；开发空间 buffer 由各队自行决定，官方统一规则未发现。
- **loss、采样、增强**：依方案而异，官方没有统一固定 loss、采样或增强。
- **指标、阈值、盲测**：结构地质专家按 1–6 主观评分，不设传统 leaderboard；概率输出 0–1。是真盲测，但不是可复算的稠密定量负例评估。
- **代码、权重与许可**：官方提供的部分生成器/资源及提交条款含 CC BY 4.0；公开社区方案 `bolgebrygg` commit `c8d01ee92c1c8e1ecba36f96cca6ea7b689338a1`、`satyakees/FaultNet` commit `e129f0ed974d20446da90bf2fc6f5971742a9fbf`，两仓库均未发现清晰软件许可；实际采用时仍应重新核对 HEAD 和依赖许可。
- **对本项目的判断**：支持 Protocol B 的“合成训练→未知真实体一次盲评”，但不能代替 Protocol A 的审核负例与数值 CV。

### S12. Dark Side of the Volume

- **为什么直接相关**：官方 3D 断层二值分割挑战，输入为三维地震数组，输出同形状断层 mask。
- **数据与输入体**：400 个训练 3D 体及对应 NumPy mask；提交 50 个数组。官方页面未发现数据为真实、合成或混合的明确说明，因此本报告不将其当作“真实稀疏标注”证据。
- **标签、负例与 unknown**：发布稠密 mask，按竞赛语义 0 为负、1 为断层；无 unknown mask。标签生成/人工覆盖流程未发现。
- **切分、buffer/purge**：官方服务器在未见 final holdout 上运行容器，约 90% 总分来自该 holdout；开发切分、buffer/purge 由参赛者自行实现，未发现统一要求。
- **loss、采样、增强**：因获胜方案而异，挑战未统一规定；应从各方案配置逐项复核，不能写成一个共同配置。
- **指标、阈值、盲测**：3D Dice；最终 holdout 在 `g5.12xlarge` 限时约 4 小时，是有效盲测生命周期。固定阈值规则未发现。
- **代码、权重与许可**：ThinkOnward 官方 Hugging Face `dark-side` 分支发布六个获胜方案/权重，根仓库标 Apache-2.0；实际采用仍需检查每个模型和上游权重的独立许可。
- **对本项目的判断**：可借鉴容器化一次性 blind holdout 和稠密 mask 契约；不能用其来源未明的稠密 0 来证明本项目 stick 外为负例。

### S13. Large benchmark for seismic fault segmentation

- **为什么直接相关**：同一协议比较 FaultSeg3D、CRACKS/F3 和 Thebe，覆盖 synthetic-to-real、fine-tune、joint training 与 domain adaptation。
- **数据与输入体**：FaultSeg3D 200/20 合成体；CRACKS/F3 含由 32 名 crowd interpreters 与专家形成的约 400 inline 标签；Thebe 真实数据。为统一断层厚度，采用 skeletonize 后 rank-3 dilation。
- **标签、负例与 unknown**：统一为二值 mask，无 explicit unknown。论文直接指出真实专家标签主观且不完整，正确预测未标断层会被计作 FP。
- **切分、buffer/purge**：CRACKS 以首 30+末 30 作为 test、中间 340 train；边界未发现 purge。Thebe 使用 400 sections train 和其他位置 100 test，精确坐标/缓冲未发现。FaultSeg3D 按独立生成体 200/20。
- **loss、采样、增强**：覆盖八类架构和 200 余配置，具体 loss/采样/增强按配置变化，不能归结为单一组合；仓库提供配置真源。
- **指标、阈值、盲测**：Dice、modified Hausdorff、双向 Chamfer 及结构描述符；ODS 阈值在 train 上确定后固定应用于 test，这是本项目可保留的方向。真实 test 标签公开，不是封存盲测。
- **代码、权重与许可**：官方仓库 commit `c547014aefa4d68e5e8e41c4b342266d905a5cbc`；未发现明确软件许可证，权重可用性未发现。
- **对本项目的判断**：支持多种结构/边界指标和 train-only 阈值；也说明 synthetic-to-real 可能负迁移或灾难性遗忘，必须有 zero-shot/from-scratch 对照。

## 5. 前人如何获得“负例”：证据分级

| 负例来源 | 直接例证 | 可靠性 | 在本项目中的合法用途 |
|---|---|---|---|
| 合成生成器稠密 mask 的 0 | S01、S02、S08、S11 | 对该合成分布可靠；对真实分布没有自动效力 | 可用于 synthetic-only 预训练、校验代码和合成指标；必须单独 lane。 |
| 完整覆盖审计区域内人工 mask 的 0 | 理想契约；S12 的竞赛稠密 mask 接近该形式，但来源流程未公开 | 只有标注范围、解释标准和质检均可审计时才可靠 | 唯一可进入真实 formal loss/metric 的负类。 |
| 公开真实二值 mask 的 0，但解释目标受限 | S04 Thebe、S10、S13 | 混合了真实背景与漏标细断层；默认不可靠 | 未补 coverage 审计前应转为 unknown，或只报告数据集传统回归指标并明确偏差。 |
| 稀疏标注切片内的 0 | S05、S08、S09 | 比切片外信息强，但仍可能有漏标；MD Loss 只降低风险 | 可作为弱标签并单列 mask/权重；不能称 verified negative。 |
| 切片之间、未解释体块 | S05、S08、S09 | unknown | 监督 loss 和正式 metric 必须排除；可做一致性/无监督目标。 |
| 属性阈值、NMS/RANSAC、“无正例 patch” | S06、S07 | proxy；同时受属性失效和人工漏标影响 | 只能进入 `proxy_mask`、采样或辅助 loss，并单独报告 proxy/regression 指标。 |
| 模型高置信伪标签 | S09 | 随模型和域偏移变化；不是 ground truth | 可做 teacher-student consistency；不得回写正式标签或解锁 test。 |

**结论**：没有直接地震断层一手证据支持把 positive sticks 的补集当作可靠负类。Stage3 的 `verified_negative_labels=0` 应保持原样，直到出现逐体/逐空间块的覆盖审计和显式 verified-negative mask。

## 6. 三类可行方案与不自欺边界

### 6.1 Positive-unlabeled（PU）

本次直接来源中未发现一项采用带可审计 class prior、无偏/非负 PU risk estimator，并在独立真实块上验证的地震断层工作。S05/S08 是 masked weak supervision，S09 是 consistency/pseudo-label SSL；它们不等于统计意义的 PU 学习。

因此可以新增“PU exploratory”研究 lane，但必须同时满足：

- `P` 仅为明确 stick/解释像素；`U` 绝不作为负例。
- PU class prior 只能由 fold-train 的审计子集或预注册灵敏度范围估计，不能从 validation/test 反推。
- 训练和模型选择仍需要至少两个彼此独立、含审核正负例的开发块；否则只能检查 loss 是否运行，不能给可排名性能。
- 报告 prior 敏感性、预测阳性率、calibration 与 unknown 覆盖，不以漂亮的 Dice 替代标签可识别性。

当前数据不满足上述评估条件，所以 PU 只能列为设计候选，不能解除 Stage3/4。

### 6.2 稀疏切片/weak supervision

可直接采用 S05/S08/S09 共同支持的三层 mask：

- `positive_mask`：人工 stick/切片明确断层。
- `valid_label_mask`：仅覆盖审计认可可监督的像素；切片外为 false。
- `proxy_mask`：属性、伪标签或弱负例，永不与 verified negative 合并。

监督项仅在 `valid_label_mask` 上计算；unknown 上可用 teacher/student、平移或 patch 一致性。MD-style 权重可作为切片内漏标鲁棒性消融，但其输出不能改变标签 provenance。没有审核负例时，所有弱监督模型仍 `not_rankable`。

### 6.3 合成预训练 + 真实微调

S01、S03、S06、S07、S09、S11、S13 共同证明这是直接可研究路线，但不能保证正迁移。合法执行应包括：

1. 在 FaultSeg3D 类稠密合成体上预训练，合成 split 按独立生成体冻结。
2. 在真实开发块只用 masked labels 微调；unknown 不进入 BCE/Focal/Dice 的负类项。
3. 同时报告 synthetic-only zero-shot、real from-scratch、synthetic-pretrained/frozen encoder、full fine-tune。
4. 所有 checkpoint、预处理、类别权重、伪标签阈值和校准只看 fold-train；阈值由 pooled OOF 冻结。
5. 只有 Protocol A 数据门禁通过、开发 winner 冻结后，才允许一次 frozen-test 推理。

FORCE 证明这种路线可在真正未知真实体上接受专家盲评；S13 同时提醒域偏移和灾难性遗忘可能使迁移变差，因此必须保留 zero-shot/from-scratch 对照。

## 7. 推荐协议 A/B

### Protocol A：审核监督、可正式排名

**目标**：建立可选择 frozen winner 的正式真实数据基准。

**数据门禁**：

1. 每个开发体/空间块有原始 volume hash、坐标范围、解释者/版本、解释目标与最小断层尺度。
2. 显式 `positive_mask`、`verified_negative_mask`、`unknown_mask`、`proxy_mask`；四者关系可机检，verified negative 与 positive 不重叠。
3. 负例必须来自声明为“在该范围内完整寻找目标尺度断层”的人工覆盖审计，而不是 `1-positive`、属性阈值或模型预测。
4. buffer 后至少存在 2 个合法开发折；目标为 5-fold，不足时只按独立体/块及双方类别支持诚实降折。
5. frozen test 的路径、标签、预测和指标对开发 API 不可见；历史 test 继续只作 regression evidence。

**切分**：优先按独立地震体；单体只能按连续空间块，并在 train/validation 两侧 purge 至少 8 inline（沿用当前 P4 合同），实际宽度取 `max(8, patch overlap/receptive-field correlation audit 所需宽度)`。任何重叠 patch、相邻切片或同一增强源不得跨折。

**训练与选择**：预处理、类别权重、target transform、采样器、伪标签/校准仅 fit fold-train。主指标 AP；辅报 Dice、IoU、Precision、Recall、boundary/Hausdorff、component/continuity，全部只在 `valid_label_mask` 上。阈值由 pooled OOF 一次冻结，禁止 OIS/ODS 在 test 上寻优。

**test 生命周期**：固定候选、固定 seeds、冻结 winner、完成 refit manifest 后才执行一次 test；失败不得换模型重试。若没有未消费连续标注块，则明确 `blind_test:not_feasible`，不得伪造 holdout。

### Protocol B：合成预训练 + masked weak/SSL，开发性不排名

**目标**：在 Protocol A 数据门禁尚未满足时，研究是否能用合成和 unlabeled 真实体降低标注需求，而不冒充正式性能。

**B0 合成基线**：FaultSeg3D 类独立生成体 train/validation；3 seeds；记录 synthetic AP/Dice 与真实 unlabeled 的定性 zero-shot，但不把真实 unknown 当 GT。

**B1 masked real adaptation**：只在 positive/valid labels 上做监督；unknown 上只做一致性。MD-style 仅为切片内假负例鲁棒性机制。属性/RANSAC/teacher 结果写入 proxy mask，单独加权。

**B2 控制组**：synthetic-only zero-shot、real from-scratch、frozen encoder、full fine-tune、去掉 proxy、去掉 consistency。候选、更新次数、预处理和 loss 预算预先冻结，不因结果追分。

**评价边界**：如果没有 Protocol A 的审核负例和合法开发折，B0/B1/B2 只报告运行证据、覆盖率、proxy/regression 指标和预注册定性图，统一 `not_rankable`；不得 frozen winner/refit/test。只有 A 门禁通过后，B 模型才可在相同开发折中与 A 做 apple-to-apple 比较。

## 8. 最小复现实验

### E0：数据可识别性审计（第一优先，当前唯一合法步骤）

- 对每个真实体生成只读 manifest：shape、坐标系、inline/crossline/time 范围、源文件 SHA-256、标注版本和解释范围。
- 分别统计 positive、verified negative、unknown、proxy 的 voxel/slice/block 覆盖；生成三正交 coverage 图，而不是预测图。
- 检查是否存在从未被任何模型/阈值/人工调参消费的连续标注块；若有才可候选 blind test，立即只存 hash/坐标并隔离内容。
- 验收：每个开发块中 mask 互斥/完备关系可复算；负例 provenance 非空；buffer 后至少 2 折双方都有正例和 verified negative。未通过则保持当前 Stage3/4 blocked。

### E1：合成-only 可重复基线

- 模型：保留现有简单 baseline，再选一个 FaultSeg3D 兼容 3D CNN；不下载大权重。
- 数据：独立生成体切分；不得从同一父体裁 patch 后随机跨折。
- 固定 3 seeds、相同 patch/更新次数；输出 masked loss、AP/Dice、checkpoint hash 与环境 manifest。
- 目的：验证训练/保存/恢复和 synthetic 上限，不声称真实效果。

### E2：真实 masked weak adaptation（仅在有合法开发折后）

- 同一组冻结 folds 比较 from-scratch、synthetic zero-shot、frozen encoder、full fine-tune、MD-style、SSL consistency。
- unknown 对监督 loss 权重严格为 0；proxy 单独开关并在结果记录其覆盖和权重。
- 每 fold×seed 都输出 OOF；完成率低于协议阈值则 not-rankable，不补临时切分。

### E3：防自欺消融

- 随机 patch CV 对比 buffered spatial/volume CV，只用于量化泄漏膨胀，不采用前者作结论。
- 测试 test-optimized threshold 与 OOF-frozen threshold 的差异，但前者只能在公开 regression 数据上做方法学演示，不能接触 frozen test。
- 对 verified negative、slice-zero weak negative、attribute proxy 分层报告，禁止合并成一个“背景”指标。
- 报告 synthetic→real 相对 from-scratch 的均值、方差和负迁移次数。

### E4：一次 frozen test

只有 E0 通过、E2 合法完成、winner/threshold/refit manifest 全冻结后才存在合法命令。当前没有合法命令，Stage4 `holdout_accessed=false` 必须保持。

## 9. 现 Pipeline 保留/修改/废弃清单

### 保留

- 保留 Stage3/Stage4 fail-closed 的 `NO_VALID_FAULT_DEVELOPMENT_FOLDS`、`blocked/not_rankable`、`frozen_winner=null`。
- 保留 fault-stick 为 positive、未标注为 unknown/invalid、proxy 单列的语义。
- 保留 requested 5-fold、基于独立体/空间块诚实降折和禁止临时 20% 切分。
- 保留 test firewall、历史 test regression-only、OOF-only threshold、一次性 frozen-test 生命周期。
- 保留 source hash、readiness/negative/unknown 覆盖图和只读归档。
- 保留当前简单 baseline，作为以后 Protocol A/B 的共同可移植基线。

### 取得数据契约后修改/新增

- 增加逐体 annotation-scope manifest：解释深度、目标最小位移、解释者、完整性声明、覆盖多边形/切片和审计版本。
- 增加 verified-negative mask 的独立文件、provenance、计数、hash 和与 positive 的不相交测试。
- 增加合成-only、weak/SSL、formal-supervised 三个 lane，排行榜和结论不得跨 lane 污染。
- buffer 宽度由当前最小 8 inline 与 patch 重叠/感受野/空间相关审计的较大值决定。
- loss 接受 `valid_label_mask`，proxy 辅助项接受独立 mask/权重；所有 mask 流经 checkpoint/artifact manifest。
- 指标增加 boundary/Hausdorff、components/continuity，并同时报告有效标签覆盖、unknown/proxy 覆盖。
- 合成迁移增加 zero-shot/from-scratch/frozen/full fine-tune 控制，防止只报告正迁移案例。

### 废弃/禁止

- 废弃“stick/mask 之外全部为负例”的补集标签生成。
- 禁止把 coherence、RANSAC、teacher 高置信或“无正例 patch”叫作 audited negative。
- 禁止同一体随机 patch/slice 交叉验证、重叠 patch 跨折、连续块边界无 purge。
- 禁止从 test 工区抽 validation、在 test 上做 OIS/ODS/阈值选择或反复挑 checkpoint。
- 禁止把真实展示体、公开静态测试集或历史 `audited_v2` 宣称为新鲜 blind test。
- 禁止在 unknown/proxy 上计算正式 Dice/AP 后据此解锁 winner。
- 禁止用 MD/Focal/Dice 等 loss 名称掩盖标签 provenance；loss 不能把不可靠负例变可靠。

## 10. 最小解除阻塞数据契约

要把当前 `effective_folds=0` 升级为可训练、可排名，至少需要：

1. **源体契约**：每个真实开发体的不可变 ID、shape/坐标、源 SHA-256、采集域和访问许可。
2. **覆盖契约**：逐体/块说明人工实际检查过的空间范围、深度、最小目标尺度、解释者和质检；不是只给 sticks。
3. **四掩码契约**：positive、verified negative、unknown、proxy；明确互斥关系、来源和 hash。`unknown = not valid_label_mask`，proxy 永不自动 valid。
4. **切分契约**：至少两个独立开发体或 buffer 后彼此独立的空间块；每折 train/validation 两侧均含 positive 和 verified negative。目标 5 折，数据不支持时诚实降折。
5. **test 契约**：存在从未被训练、可视化、阈值或人工选择消费的连续标注块则冻结其 hash/坐标并隔离；不存在就继续 `not_feasible`。
6. **metric 契约**：AP 主指标；OOF 固定阈值；mask-aware Dice/IoU/Precision/Recall 和 boundary/component 指标；同步报告每类 mask 覆盖。

仅增加模型、loss、合成样本、proxy 标签或更多 positive sticks，都不能单独满足该契约。

## 11. 来源清单与版本锁

1. Wu et al., FaultSeg3D, DOI: https://doi.org/10.1190/geo2018-0646.1 ；作者仓库：https://github.com/xinwucwp/faultSeg ，核对 commit `0ab8ba1c10cb9e5748b0129bfe1a2fe3031b80fe`。
2. Wu et al., FaultNet3D, DOI: https://doi.org/10.1109/TGRS.2019.2925003 ；作者 PDF：https://cig.ustc.edu.cn/_upload/tpl/05/cd/1485/template1485/papers/wu2019faultNet3d.pdf 。
3. Cunha/Pochet et al., synthetic-to-real transfer, DOI: https://doi.org/10.1016/j.cageo.2019.104344 。
4. An et al., real-data CNN benchmark, DOI: https://doi.org/10.1016/j.cageo.2021.104776 ；Thebe Dataverse：https://doi.org/10.7910/DVN/YBYGBK ；作者仓库：https://github.com/anyuzoey/CNNforFaultInterpretation ，核对 commit `58eae5db2312feca003b9eb179fd9172beeeab5d`。
5. Dou et al., attention weak supervision, arXiv: https://arxiv.org/abs/2105.03857 ；正式 DOI：https://doi.org/10.1109/TGRS.2021.3113676 。
6. Yan et al., automatic labels and transfer, DOI/full text: https://doi.org/10.3390/en14123650 。
7. Wei et al., focal-loss transfer, DOI: https://doi.org/10.1016/j.cageo.2021.104968 ；作者仓库：https://github.com/weixiaoli125/fault-detection ，核对 commit `5c24aee162c439507f1bc7c5303437dac262eb27`。
8. Dou et al., MD Loss, arXiv: https://arxiv.org/abs/2110.05319 ；正式 DOI：https://doi.org/10.1109/TGRS.2022.3196810 。
9. Dou et al., FaultSSL, arXiv: https://arxiv.org/abs/2309.02930 ；正式 DOI：https://doi.org/10.1190/geo2023-0550.1 。
10. Guo et al., Seismic Fault SAM, arXiv: https://arxiv.org/abs/2407.14121 。
11. FORCE 2020 official challenge: https://thinkonward.com/app/c/challenges/force-seismic ；公开参赛方案：https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition 与 https://github.com/satyakees/FaultNet 。
12. Dark Side of the Volume official challenge: https://thinkonward.com/app/c/challenges/dark-side ；官方获胜方案：https://huggingface.co/thinkonward/challenges/tree/dark-side 。
13. Quesada et al., large benchmark, arXiv: https://arxiv.org/abs/2505.08585 ；官方仓库：https://github.com/olivesgatech/large-bench-geo ，核对 commit `c547014aefa4d68e5e8e41c4b342266d905a5cbc`。

补充发现但不计入核心 13 项：Kaggle 社区镜像 https://www.kaggle.com/datasets/malik9/synth-seis-data 只是 FaultSeg3D 合成资产镜像；未发现官方 Kaggle 同任务竞赛。由于不是新的方法、数据真源或盲测协议，本报告不将其计数。

## 12. 最终判断

- **可靠负例**：真实数据中只能由明确覆盖范围内的完整人工审计产生；合成稠密 0 只对合成域可靠；弱标签、属性和伪标签均不能升级为正式负例。
- **何时允许训练**：synthetic-only 或 masked weak/SSL 快速研究可以在独立开发 lane 运行；正式真实模型训练/选择至少需要两个 buffer 后合法开发折及双方类别支持。当前 Stage3 的 0 合法折意味着正式训练仍不允许。
- **PU/弱监督**：可研究，但 unknown 永不作负类，proxy 单列；本次未找到直接地震断层工作对正式 PU risk 做独立验证，因此不得借“PU”名义解锁排名。
- **合成到真实**：有多项直接证据支持预训练价值，也有直接 benchmark 提示域偏移和负迁移；必须以 zero-shot/from-scratch 对照、独立空间/体折和 OOF 阈值验证。
- **测试**：最可信生命周期来自最后时刻未见真实体或服务器 final holdout；Thebe/SAM 的公开 test 与 test-optimal OIS/ODS 不满足当前 frozen-test 标准。当前仓库应继续 `blocked/not_rankable`，直到第 10 节数据契约真正满足。
