---
phase_id: P2.1
reviewer: codex
status: pending
review_date: 2026-07-13
scope: data-integrity-and-algorithm-provenance
---

# 数据完整性与算法源头审查（2026-07-13）

## 1. 审查结论

本轮使用本机真实文件系统、压缩包目录、官方校验值、HDF5 实读和当前官方在线页面交叉核验，结论如下：

| 数据资产 | 包级完整性 | 官方清单覆盖 | 结论 |
|---|---:|---:|---|
| Volve Data Village | ✅ | ✅ | `_sandbox/volve_data/` 中 14 个 ZIP 与当前 Databricks Marketplace 官方卷逐文件同名、同字节数；11 个官方文件夹类别全部覆盖，无遗漏、无新增 ZIP |
| Netherlands F3 Interpretation Dataset | ✅ | ✅ | Zenodo 1471548 当前版本的 8 个文件全部在本地，文件名、大小和 MD5 全部一致；无未下载的 ML 切片包/子集 |
| Penobscot Interpretation Dataset | ✅ | ✅ | Zenodo 3924682 当前版本的 5 个文件全部在本地，文件名、大小和 MD5 全部一致；无未下载的 ML 切片包/子集 |

需要区分两个概念：**官方下载包齐全**不等于**包内每种专业格式都已做内容级、全量语义验证**。Volve 的本轮结论是“14 个官方 ZIP 的包级清单和大小完整，中央目录可读”；没有对 4.566 TB 压缩数据做全量解压 CRC，也不能替代 SEG-Y、DLIS、WITSML、OpenWorks 等格式各自的全量业务校验。

本轮还发现 4 项登记/文档漂移，但它们不是缺包：

1. `_meta/_data_registry.yml` 的 Volve `~7.7TB` 与实测不符：14 个 ZIP 合计 **4,566,064,614,182 bytes = 4.566 TB = 4.153 TiB**；`unzip -l` 汇总的包内未压缩标称大小约 **5.437 TB**，也不是 7.7 TB。
2. 项目 Wiki 所称“官方 9 大类”是本项目自行归并口径；Equinor 许可证 Extract 1 实际列出 **11 个官方文件夹类别**，其中 Seismic 再拆成 4 个 ZIP，最终形成当前 14 个 ZIP。
3. F3 registry 写“9 类标注”，但 [Zenodo 1471548](https://zenodo.org/records/1471548) 当前说明是 **9 个解释层位分隔 10 个地震相区间/类别**。
4. F3、Penobscot 的 registry 状态已是 `downloaded`，但 note 末尾仍写“下载中”；Penobscot 页面正文说 7 classes，而本地官方 `dataset-log.txt` 明确记录 `num_classes: 8`，HDF5 `label` 实读为 0–7，推测页面的“7”指 7 个解释层位而生成数据包含 8 个层间区间，后续建模应以数据文件的 8 个整数标签为准。

## 2. 核查方法与边界

### 2.1 实际执行的本地命令

```bash
du -sh _sandbox/volve_data \
  _sandbox/f3_penobscot/f3demo \
  _sandbox/f3_penobscot/penobscot

find _sandbox/volve_data -maxdepth 1 -type f -name '*.zip' \
  -printf '%f\t%s\n' | sort

for f in _sandbox/volve_data/*.zip; do
  unzip -l "$f" | tail -1
done

md5sum _sandbox/f3_penobscot/f3demo/* \
  _sandbox/f3_penobscot/penobscot/*

databricks fs ls \
  dbfs:/Volumes/equinor_asa_volve_data_village/public/volvezipfiles \
  --long
```

实测摘要：

```text
4.2T  _sandbox/volve_data
1.6G  _sandbox/f3_penobscot/f3demo
2.2G  _sandbox/f3_penobscot/penobscot

Volve ZIP count: 14
Volve ZIP bytes: 4566064614182
partial/part/tmp files: 0
```

磁盘核查时 `/mnt/data` 为 73T 总量、64T 已用、5.4T 可用、使用率 93%。这不影响完整性结论，但后续若全量解压大体积地震包，必须先重新评估空间。

### 2.2 在线权威源

- [Equinor — Volve field data set](https://www.equinor.com/energy/volve-data-sharing)：当前仍称这是 complete set，约 40,000 个文件，官方入口为 Databricks Marketplace。
- [Equinor — Volve Open Data Licence PDF](https://www.equinor.com/content/dam/statoil/documents/what-we-do/Equinor-HRS-Terms-and-conditions-for-licence-to-data-Volve.pdf)：Extract 1 给出 11 个官方文件夹类别。
- [Databricks Marketplace — Equinor ASA / Volve Data Village](https://marketplace.databricks.com/details/5c3558ef-315c-44dd-baef-7062ac301f22/Equinor-ASA_Volve-Data-Village)：本轮通过已授权 Databricks CLI 直接读取其官方卷，而不是只看项目 registry。
- [Zenodo 1471548 — Netherlands F3 Interpretation Dataset](https://zenodo.org/records/1471548)，Version 2.0.0。
- [Zenodo 3924682 — Penobscot Interpretation Dataset](https://zenodo.org/records/3924682)，Version 3.0.0。

### 2.3 完整性判据

- Volve：官方远端卷与本地的文件名集合一致；每个 ZIP 的字节数一致；14/14 的 `unzip -l` 返回 0；无 `.part`、`.partial`、`.tmp` 残留。
- F3/Penobscot：Zenodo 当前页面文件集合与本地一致；逐文件 MD5 一致；ZIP/TAR 中央目录可列出；Penobscot HDF5 能被 `h5py` 打开并抽样读取。
- 本轮没有修改 `_meta/_data_registry.yml` 或任何现有代码、数据。

## 3. Volve 官方类别逐项核验

### 3.1 官方 11 类 → 本地 14 ZIP

| # | Equinor 官方文件夹类别 | 当前 Marketplace ZIP | 本地大小（bytes） | 状态 |
|---:|---|---|---:|---:|
| 1 | Geophysical Interpretations | `Volve_Geophysical_Interpretations.zip` | 103,908,838 | ✅ |
| 2 | GeoScience_OW_Archive | `Volve_GeoScience_OW_Archive.zip` | 58,668,428,829 | ✅ |
| 3 | Production data | `Volve_Production_data.zip` | 2,011,517 | ✅ |
| 4 | Reports | `Volve_Reports.zip` | 169,470,945 | ✅ |
| 5 | Reservoir_Model-Eclipse_model | `Volve_Reservoir_Model-Eclipse_model.zip` | 409,380,233 | ✅ |
| 6 | Reservoir_Model-RMS_model | `Volve_Reservoir_Model-RMS_model.zip` | 2,267,692,060 | ✅ |
| 7a | Seismic / ST0202 | `Volve_Seismic_ST0202.zip` | 1,289,281,970,411 | ✅ |
| 7b | Seismic / ST10010 | `Volve_Seismic_ST10010.zip` | 2,842,647,518,663 | ✅ |
| 7c | Seismic / ST0202 vs ST10010 4D | `Volve_Seismic_ST0202vsST10010_4D.zip` | 354,769,640,397 | ✅ |
| 7d | Seismic / VSP | `Volve_Seismic_VSP.zip` | 99,961,401 | ✅ |
| 8 | Well_logs | `Volve_Well_logs.zip` | 7,449,415,582 | ✅ |
| 9 | Well_logs_pr_WELL | `Volve_Well_logs_pr_WELL.zip` | 7,496,450,438 | ✅ |
| 10 | Well_technical_data | `Volve_Well_technical_data.zip` | 222,167,142 | ✅ |
| 11 | WITSML Realtime drilling data | `Volve_WITSML Realtime drilling data.zip` | 2,476,597,726 | ✅ |

**远端差集结果：**缺失 `[]`，本地多余 ZIP `[]`，同名大小不一致 `[]`。截至 2026-07-13，当前官方 Marketplace 卷没有第 15 个 ZIP，也没有超出上述 11 类的新类别。

### 3.2 ZIP 中央目录可读性

| ZIP | `unzip -l` 返回码 | 条目数 | 包内标称未压缩字节 |
|---|---:|---:|---:|
| GeoScience OW Archive | 0 | 5 | 59,108,560,766 |
| Geophysical Interpretations | 0 | 56 | 405,708,541 |
| Production data | 0 | 3 | 2,343,915 |
| Reports | 0 | 4 | 187,239,137 |
| Eclipse model | 0 | 67 | 1,703,985,968 |
| RMS model | 0 | 8,069 | 9,799,424,396 |
| Seismic ST0202 | 0 | 286 | 1,565,887,966,361 |
| Seismic 4D | 0 | 45 | 401,821,295,060 |
| Seismic ST10010 | 0 | 474 | 3,345,347,272,498 |
| Seismic VSP | 0 | 53 | 124,147,492 |
| WITSML realtime drilling | 0 | 23,501 | 19,337,743,378 |
| Well logs | 0 | 2,920 | 16,223,217,950 |
| Well logs per well | 0 | 2,933 | 16,275,037,416 |
| Well technical data | 0 | 5,568 | 410,188,371 |

### 3.3 内容级边界和已知风险

- ✅ 14 个 ZIP 的官方下载完整性已由远端字节数和中央目录验证。
- ✅ 项目既有 inventory 显示 DLIS 607/607 可打开，抽查/登记的 PDF 可打开；必要的 SEG-Y、解释数据和测井子集已能供 Layer 1 使用。
- ⚠️ LAS 既有内容检查中记录过 251 个成功、2 个真实失败、48 个超时；这属于包内源文件/解析兼容性问题，不是下载缺包。
- ⚠️ WITSML 既有验证是抽样而非 18,814 个 XML 全量语义验证。
- ⚠️ OpenWorks `.bri/.hts/.dszip/.bck` 为专有/归档格式，当前只能证明存在、大小和容器结构，不能声称业务内容已被完整解析。
- ⚠️ 本轮没有执行 4.566 TB 全量 `unzip -t`，因此不能把 `unzip -l` 描述成逐字节 CRC 验证。

## 4. F3-Demo 与 Penobscot 的 Zenodo 子集核验

### 4.1 F3 — Zenodo 1471548

Zenodo 当前版本列出 8 个文件；本地 8/8，MD5 8/8 匹配：

| 文件 | 本地字节数 | Zenodo MD5 | 状态 |
|---|---:|---|---:|
| `crosslines.zip` | 639,101,595 | `32e4e6228c44995cce218a8f8936bedb` | ✅ |
| `examples_crossline_tiles.png` | 289,379 | `35d6a1208aacd072ebf17cb6dc0313f3` | ✅ |
| `examples_inline_tiles.png` | 288,843 | `7fd2694ccbe500870f2e6020e88000f9` | ✅ |
| `horizons.tar.gz` | 95,320,034 | `30b40f0426d95c5f26878bcd398fc853` | ✅ |
| `inlines.zip` | 637,187,326 | `fb5b0d16ca27f7c8c3e19930a28eedbe` | ✅ |
| `masks.tar.gz` | 4,800,574 | `42e42e9955ec2d957901339f49db720f` | ✅ |
| `tiles_crosslines.tar.gz` | 140,979,246 | `8e255eab8a223c1a55d856ee2556948f` | ✅ |
| `tiles_inlines.tar.gz` | 139,644,717 | `eae2198d5e9e1f16b3042957fa01176b` | ✅ |

容器目录均可读：`crosslines.zip` 1,906 条目、`inlines.zip` 1,306、`horizons.tar.gz` 20、`masks.tar.gz` 3,206、两个 tiles 包分别 188,804 和 189,444 条目。

**遗漏判断：✅ 无遗漏。** Zenodo 页面没有额外的机器学习切片包、mask 包或隐藏子集需要补下载。

### 4.2 Penobscot — Zenodo 3924682

Zenodo 当前版本列出 5 个文件；本地 5/5，MD5 5/5 匹配：

| 文件 | 本地字节数 | Zenodo MD5 | 状态 |
|---|---:|---|---:|
| `dataset-log.txt` | 1,947 | `d1fd8bdc6fa7818401326819c9130387` | ✅ |
| `dataset.h5` | 2,268,562,702 | `e53af020d42f49dba7a1a5988eccc829` | ✅ |
| `horizons.zip` | 36,213,721 | `42c104fafbb8e79695ae23527a91ee78` | ✅ |
| `how-to-read.ipynb` | 633,270 | `3595ff238e927171ae6065532c77a7aa` | ✅ |
| `penobscot-examples.png` | 10,649,865 | `ed415cd77672fb31a4f17c2b58fd67a3` | ✅ |

`horizons.zip` 中央目录可读（8 条目）。`dataset.h5` 实读结果：

```text
keys = column, direction, features, label, line_number, pixel_depth
features.shape = (601, 1501, 481, 1), dtype=float32
label.shape    = (601, 1501, 481),    dtype=uint8
sampled feature slices finite = true
observed label values = [0,1,2,3,4,5,6,7]
```

**遗漏判断：✅ 无遗漏。** Zenodo 页面没有额外的 ML 切片包/子集；该发布本身已把训练用体数据和标签装入单一 HDF5。

## 5. 六条赛道的算法/基线源头

### 5.1 评价口径

- **代码**：✅ 作者/机构公开仓库；⚠️ 公开但归档、老框架或复现材料不完整；❌ 本轮没有找到作者公开代码。
- **文档质量**：高 = 安装、数据、训练/推理、示例较完整；中 = README 可运行但环境/数据说明有限；低 = 主要依赖论文自行重建。
- “未找到代码”仅表示在本轮对论文页、作者/机构页和 GitHub 的定向检索中没有定位到权威仓库，不等于证明互联网上绝对不存在非官方实现。
- 表中标为 ✅/⚠️ 的 15 个 GitHub 仓库均在 2026-07-13 以 `git ls-remote --heads <repo>` 实测可访问（15/15 成功）；⚠️ 表示维护/复现风险，不表示仓库链接失效。

### 5.2 ① 断层预测

| 源头 | 论文/代码 | 代码 | 文档 | 对本赛道的价值 |
|---|---|---:|---:|---|
| FaultSeg3D (Wu et al., GEOPHYSICS 2019) | [论文](https://doi.org/10.1190/geo2018-0646.1) / [作者仓库](https://github.com/xinwucwp/faultSeg) | ✅ | 中 | 3D U-Net 风格端到端断层体分割的经典公开基线，含合成训练数据、预训练模型和预测 notebook；适合作为 Volve 稀疏断层棒标签基线的结构参照 |
| Seismic fault detection using CNNs with focal loss | [论文](https://doi.org/10.1016/j.cageo.2021.104968) / [作者仓库](https://github.com/weixiaoli125/fault-detection) | ⚠️ | 中偏低 | 直接处理断层/非断层严重不平衡，给出 focal loss、迁移训练和 F3 实测案例；但依赖 TensorFlow 1.14，现代环境复现成本较高 |
| FaultSSL | [论文](https://arxiv.org/abs/2309.02930) | ❌ | 低 | Mean Teacher + 相邻切片/patch 一致性，特别适合“少量解释切片 + 大量未标注地震体”的 Volve 场景；本轮未确认到作者公开代码，宜作为研究方向而非首个可复现 baseline |

### 5.3 ② 地震相分类

| 源头 | 论文/代码 | 代码 | 文档 | 对本赛道的价值 |
|---|---|---:|---:|---|
| A Machine-Learning Benchmark for Facies Classification | [论文](https://arxiv.org/abs/1901.07659) / [作者仓库](https://github.com/olivesgatech/facies_classification_benchmark) | ✅ | 中 | F3 全标注 3D 地质模型、固定划分和两个反卷积网络 baseline，是本项目 F3 地震相赛道最直接的可比源头 |
| Microsoft DeepSeismic | [文档](https://microsoft.github.io/seismic-deeplearning/) / [微软仓库](https://github.com/microsoft/seismic-deeplearning) | ⚠️ | 高 | 提供 F3/Penobscot 数据转换、训练、测试、notebook 和配置化实验；仓库自 2023-11-16 起归档，只适合做强复现参考，不宜原样作为长期依赖 |
| EarthAdaptNet / EAN-DDA | [论文](https://arxiv.org/abs/2011.10510) | ❌ | 低 | 直接以 F3 为源域、Penobscot 为目标域，用 CORAL 做无监督域适配；对跨区块泛化很有参考价值，但本轮未确认到与论文一致的作者训练仓库 |

### 5.4 ③ 储层物性预测

| 源头 | 论文/代码 | 代码 | 文档 | 对本赛道的价值 |
|---|---|---:|---:|---|
| Joint Learning for Seismic Inversion | [论文](https://arxiv.org/abs/2104.02750) / [作者仓库](https://github.com/thelearningcurves/SEG-2020-Joint-learning-with-spatial-context-for-inversion) | ✅ | 中 | 用少量井日志约束地震到声阻抗回归，并借跨数据集联合学习和空间上下文减少过拟合；最贴近“地震 + 测井稀疏标签”的多模态主线 |
| OpenFWI / InversionNet | [项目与数据文档](https://openfwi-lanl.github.io/) / [LANL 仓库](https://github.com/lanl/OpenFWI) | ✅ | 高 | 大规模 FWI 数据与 InversionNet、VelocityGAN 等基线，适合检验速度/阻抗体反演网络和 2D→3D 扩展；其公开主基准以合成数据为主，不能代替 Volve 实井校准 |
| GeostatsPy / GSLIB baseline | [仓库与文档](https://github.com/GeostatsGuy/GeostatsPy) | ✅ | 高 | 提供变差函数、克里金、协同/条件模拟及 3D SGSIM/SISIM，可作为神经网络之外的稀疏井点到 3D 属性体传统强基线和不确定性对照 |
| ADSeismic.jl | [仓库](https://github.com/kailaix/ADSeismic.jl) / [反演文档](https://kailaix.github.io/ADSeismic.jl/dev/backward_inversion/) | ✅ | 高 | 自动微分声学/弹性正演与 FWI，可把物理一致性作为纯数据驱动模型的对照或正则；Julia/ADCME 技术栈与项目现有 Python 栈不同 |

### 5.5 ④ 岩相预测

| 源头 | 论文/代码 | 代码 | 文档 | 对本赛道的价值 |
|---|---|---:|---:|---|
| SEG 2016 Machine Learning Contest | [SEG 仓库](https://github.com/seg/2016-ml-contest) / [SEG 教程](https://wiki.seg.org/wiki/Facies_classification_using_machine_learning) | ✅ | 高 | 经典测井岩相分类数据、按井交叉验证和大量公开 notebook，可直接规范“不能随机打散同一口井泄漏到测试集”的 baseline 协议 |
| FORCE 2020 Well Log Lithofacies Competition | [Equinor 仓库](https://github.com/equinor/force-ml-2020-wells) / [官方结果页](https://www.sodir.no/en/force/Previous-events/2020/results-of-the-FORCE-2020-lithology-competition/) | ⚠️ | 高 | 98 训练井 + 10 盲测井，公开 XGBoost/CatBoost 流程及参赛结果；非常适合 Volve 多井测井岩相的树模型强基线，仓库已于 2025-03-27 归档 |
| Well-log Lithology Classification Benchmark (2026) | [论文](https://doi.org/10.1007/s11004-026-10300-1) / [作者仓库](https://github.com/uai-ufmg/well-log-lithology-classification) | ✅ | 高 | 在统一协议下比较传统 ML 与利用纵向连续性的深度模型，代码、环境和实验结构公开，可作为更现代的复现标准 |

### 5.6 ⑤ 甜点预测

| 源头 | 论文/代码 | 代码 | 文档 | 对本赛道的价值 |
|---|---|---:|---:|---|
| Intelligent prediction and integral analysis of shale oil and gas sweet spots | [开放论文](https://link.springer.com/article/10.1007/s12182-018-0261-y) | ❌ | 中（论文） | 用井约束筛选地震属性，再以模糊评价/机器学习外推甜点属性，是“井上定标、地震横向推广”的直接方法论源头 |
| Prediction of sweet spots from well logging and 3D seismic | [论文](https://doi.org/10.1177/0144598716679961) | ❌ | 中（论文） | 明确串联测井解释、岩石物理敏感参数、叠前反演和多指标甜点分级，可用于定义本项目代理标签和可解释特征链 |
| Shale-gas sweet-spot parameter prediction with CNN | [开放全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470566/) | ❌ | 中（论文） | 1D CNN 在井上学习 PHI/TOC/GAS，再逐地震道生成 3D 参数体，提供了可落地的“测井监督→地震体推理”范式 |
| BHO-LCE interpretable sweet-spot prediction | [论文](https://academic.oup.com/jge/article/23/3/843/8466409) | ❌ | 中（论文） | 2026 年工作把贝叶斯调参、局部级联集成和 SHAP 用于少井条件下的甜点类型预测，可作为 RF/XGB/HGB 之后的高级对照 |

**赛道⑤关键判断：**本轮没有找到上述权威甜点方法对应的作者公开端到端仓库；公开材料质量主要停留在论文级。赛道当前最大风险不是“缺一个网络”，而是 Volve **没有现成甜点真值**。必须先由项目 owner 明确定义地质甜点、工程甜点或生产甜点，以及代理标签如何由孔隙度/渗透率、岩相、断层距离、产量和井位组合；否则不同论文中的“甜点”目标不可直接比较。

### 5.7 ⑥ 三维模型重建

| 源头 | 论文/代码 | 代码 | 文档 | 对本赛道的价值 |
|---|---|---:|---:|---|
| GeoINR 1.0 | [GMD 论文](https://gmd.copernicus.org/articles/16/6987/2023/) / [作者仓库](https://github.com/MichaelHillier/GeoINR) | ✅ | 中 | 用隐式神经表示从界面、法向和地层约束恢复连续 3D 地质体，是“稀疏井/层位约束→连续模型”的直接神经基线 |
| GemPy v3 | [项目主页/教程](https://www.gempy.org/) / [仓库](https://github.com/gempy-project/gempy) | ✅ | 高 | 基于界面点和产状的隐式 3D 构造建模，支持断层、不整合和概率建模；适合作为学习式重建的传统、可解释强基线 |
| LoopStructural | [论文](https://gmd.copernicus.org/articles/14/3915/2021/) / [仓库](https://github.com/Loop3D/LoopStructural) / [用户指南](https://loop3d.org/LoopStructural/user_guide/index.html) | ✅ | 高 | 支持断层运动学、褶皱、多种隐式插值和多格式导出，适合把 Volve 断层/层位/井分层点融合成结构一致的 3D 模型 |
| DeepSDF | [论文](https://arxiv.org/abs/1901.05103) / [Meta 作者仓库](https://github.com/facebookresearch/DeepSDF) | ✅ | 高 | 通用连续 SDF 重建/补全源头，可参考隐式表示和稀疏采样训练；它不是地质专用模型，必须加入地层顺序、断层和井约束后才有地学意义 |

## 6. 建议的 baseline 优先级

1. **①断层：**先以 FaultSeg3D/小型 3D U-Net 为可复现基线，再评估 focal loss；FaultSSL 放在有可靠未标注一致性实验后。
2. **②地震相：**以 F3 benchmark 固定划分和 DeepSeismic 的数据协议为首选；Penobscot 作为跨区块外测，避免同一 inline/crossline 邻域泄漏。
3. **③物性：**用“树模型/TCN 回归 + 井留一验证”建立底线，同时用 GeostatsPy 克里金/条件模拟作非神经对照；OpenFWI 仅作反演架构预训练/方法参考。
4. **④岩相：**先复现 SEG 2016 与 FORCE 2020 的按井划分树模型，再比较 1D CNN/序列模型，指标采用 macro-F1 和按井混淆矩阵。
5. **⑤甜点：**在任何训练前冻结标签定义和专家规则；当前不建议仅凭产油井位置自动制造“真值”。
6. **⑥三维重建：**先用 GemPy/LoopStructural 建立可解释结构基线，再比较 GeoINR；若赛题目标其实是地震速度体而不是地层结构体，应改用 OpenFWI/InversionNet3D 方向，二者不能混称同一任务。

## 7. 需补充下载与后续动作

### 7.1 数据下载

- Volve 官方类别/ZIP：**无需补下载**。
- F3 Zenodo 1471548 ML 包：**无需补下载**。
- Penobscot Zenodo 3924682 ML 包：**无需补下载**。
- 不建议现在全量解压 3 个超大 Volve Seismic ZIP；当前仅 5.4T 可用，且解压后的标称总量约 5.437 TB，还需要留出训练产物和共享盘安全余量。

### 7.2 登记修正（本轮只审查，未修改）

- 把 Volve `size_approx` 改成同时注明“压缩包 4.566 TB / 中央目录标称未压缩 5.437 TB”，不要继续写无证据的 `~7.7TB`。
- 把“官方 9 大类”修正为“Equinor 官方 11 个文件夹类别，本项目可归并为 9 个业务组，远端为 14 ZIP”。
- 删除 F3/Penobscot note 中残留的“下载中”。
- F3 标注改为“9 个层位、10 个层间类别”；Penobscot 建模说明改为“7 个解释层位、发布 HDF5 的 `num_classes=8`/标签 0–7”，并记录页面文字与产物元数据差异。

## 8. 最终判定

- **数据包完整性：通过。** 三批已登记资产均未发现漏下载文件或远端新增包。
- **内容级可用性：有条件通过。** F3/Penobscot 已有强校验；Volve 包级完整，但专有格式、WITSML 全量和少数 LAS 的内容级验证仍有边界。
- **算法源头可获得性：①②③④⑥通过，⑤部分通过。** ⑤有权威论文链但缺少经确认的作者端到端公开仓库，且本项目标签定义尚未冻结。
- **本轮改动范围：**仅新增本报告；没有修改现有代码、registry 或数据。

## Adoption Decision

- [ ] accepted
- [ ] partially accepted
- [ ] rejected

Reason: 待项目 owner 审阅并决定是否采纳第 7.2 节的 registry/Wiki 修正及赛道⑤标签定义工作。
