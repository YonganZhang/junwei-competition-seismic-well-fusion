# 六赛道算法/Baseline源头 — 权威档案

> 更新: 2026-07-13
> 定位：军伟做具体模型选型前的**起点参考清单**，不是最终技术路线决策(路线选型仍是军伟的决策权)。
> 来源：Codex(secretary_web:28窗口，逐仓库`git ls-remote`验证15/15可访问) + Claude Workflow(6路并行网络搜索)交叉整合。

## 评价口径

- **代码**：✅ 作者/机构公开仓库；⚠️ 公开但已归档/老框架/复现材料不完整；❌ 本轮未找到作者公开代码，仅论文级。
- "未找到代码"只表示本轮定向检索未命中，不代表互联网上绝对不存在非官方实现。

## ① 断层预测

| 源头 | 链接 | 代码 | 对本赛道价值 |
|---|---|---|---|
| FaultSeg3D (Wu et al., GEOPHYSICS 2019) | [论文](https://doi.org/10.1190/geo2018-0646.1) / [仓库](https://github.com/xinwucwp/faultSeg) | ✅ | 3D U-Net端到端断层体分割经典基线，含合成训练数据+预训练模型，最省事的起点 |
| CNN+focal loss断层检测 | [论文](https://doi.org/10.1016/j.cageo.2021.104968) / [仓库](https://github.com/weixiaoli125/fault-detection) | ⚠️ | 处理断层/非断层严重不平衡的focal loss思路；依赖TF1.14，复现成本较高 |
| FaultSSL(半监督) | [论文](https://arxiv.org/abs/2309.02930) | ❌ | Mean Teacher一致性正则，适合"少量解释切片+大量未标注地震体"场景，作为研究方向而非首个可复现baseline |
| large-bench-geo(统一benchmark) | [仓库](https://github.com/olivesgatech/large-bench-geo) | ✅ | 整合FaultSeg3D/CRACKS/Thebe三套数据集，UNet/UNet++/DeepLabV3+/SegFormer多架构对比，适合选模型/调参参照 |

**建议**：先复现FaultSeg3D，再用large-bench-geo做架构横向对比。

## ② 地震相分类

| 源头 | 链接 | 代码 | 对本赛道价值 |
|---|---|---|---|
| F3 Facies Classification Benchmark | [论文](https://arxiv.org/abs/1901.07659) / [仓库](https://github.com/olivesgatech/facies_classification_benchmark) | ✅ | F3全标注3D地质模型+固定划分+两个deconvnet baseline，本项目F3赛道最直接可比源头 |
| Microsoft DeepSeismic | [文档](https://microsoft.github.io/seismic-deeplearning/) / [仓库](https://github.com/microsoft/seismic-deeplearning) | ⚠️ | 提供F3/Penobscot数据转换+训练+测试+notebook全流程；仓库已归档(2023-11-16起)，只适合强复现参考 |
| SEAM AI Facies Identification Challenge | [官网](https://www.aicrowd.com/challenges/seismic-facies-identification-challenge) | — | Parihaka三维地震体(1006×782×590)，6类相标签，F3之外最权威公开benchmark |
| EarthAdaptNet/EAN-DDA(域适配) | [论文](https://arxiv.org/abs/2011.10510) | ❌ | 以F3为源域、Penobscot为目标域做CORAL无监督域适配，对本项目"F3/Penobscot标签空间不兼容"问题有直接参考价值 |

**建议**：优先复现F3 benchmark；Penobscot作跨区块外测，注意避免同一inline/crossline邻域数据泄漏。

## ③ 储层物性预测

| 源头 | 链接 | 代码 | 对本赛道价值 |
|---|---|---|---|
| Joint Learning for Seismic Inversion | [论文](https://arxiv.org/abs/2104.02750) / [仓库](https://github.com/thelearningcurves/SEG-2020-Joint-learning-with-spatial-context-for-inversion) | ✅ | 少量井日志约束地震到声阻抗回归+跨数据集联合学习，最贴近"地震+测井稀疏标签"多模态主线 |
| Direct Multi-Modal Inversion (Alyaev&Elsheikh 2022) | [论文](https://arxiv.org/abs/2201.01871) / [仓库](https://github.com/alin256/multi-mode-prediction-with-mtp-loss) | ✅ | 混合密度DNN+多轨迹预测(MTP)损失，一次前向给出多个可能解及概率 |
| OpenFWI/InversionNet | [项目](https://openfwi-lanl.github.io/) / [仓库](https://github.com/lanl/OpenFWI) | ✅ | 大规模FWI数据+InversionNet/VelocityGAN基线，适合检验速度/阻抗体反演网络；以合成数据为主，不能代替Volve实井校准 |
| GeostatsPy/GSLIB | [仓库](https://github.com/GeostatsGuy/GeostatsPy) | ✅ | 变差函数/克里金/条件模拟，神经网络之外的传统强基线和不确定性对照 |
| SPWLA PDDA SIG 2021-2022竞赛 | [方案汇总](https://www.researchgate.net/publication/377913259) | — | 行业权威benchmark，测井曲线预测孔隙度/饱和度，可对标评测指标 |

**建议**：树模型/TCN回归+井留一验证建立底线，GeostatsPy做非神经对照。

## ④ 岩相预测

| 源头 | 链接 | 代码 | 对本赛道价值 |
|---|---|---|---|
| SEG 2016 Machine Learning Contest | [仓库](https://github.com/seg/2016-ml-contest) / [教程](https://wiki.seg.org/wiki/Facies_classification_using_machine_learning) | ✅ | 经典测井岩相分类数据+按井交叉验证规范，可直接借鉴"不能把同一口井随机打散到训练/测试集"的协议 |
| FORCE 2020 Lithology Prediction | [仓库](https://github.com/equinor/force-ml-2020-wells) / [结果页](https://www.sodir.no/en/force/Previous-events/2020/results-of-the-FORCE-2020-lithology-competition/) | ⚠️ | 98训练井+10盲测井，公开XGBoost/CatBoost流程；仓库已归档(2025-03-27) |
| Well-log Lithology Classification Benchmark(2026) | [论文](https://doi.org/10.1007/s11004-026-10300-1) / [仓库](https://github.com/uai-ufmg/well-log-lithology-classification) | ✅ | 统一协议对比传统ML与深度模型，代码+环境+实验结构公开，更现代的复现标准 |

**建议**：先复现SEG2016与FORCE2020的按井划分树模型基线，再比较1D CNN/序列模型，指标用macro-F1+按井混淆矩阵。

## ⑤ 甜点/有利油气目标预测

⚠️ **本赛道未找到权威方法对应的作者公开端到端仓库，公开材料只到论文级别**——这不是搜索不够，是这个细分方向工业界论文多、开源少的行业现状。

| 源头 | 链接 | 方法要点 |
|---|---|---|
| Intelligent prediction of shale sweet spots | [论文](https://link.springer.com/article/10.1007/s12182-018-0261-y) | 井约束筛选地震属性→模糊评价/机器学习外推 |
| Prediction of sweet spots from well logging+3D seismic | [论文](https://doi.org/10.1177/0144598716679961) | 测井解释→岩石物理敏感参数→叠前反演→多指标甜点分级 |
| Shale-gas sweet-spot CNN | [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470566/) | 1D CNN在井上学PHI/TOC/GAS，逐地震道生成3D参数体 |
| BHO-LCE可解释甜点预测(2026) | [论文](https://academic.oup.com/jge/article/23/3/843/8466409) | 贝叶斯调参+局部级联集成+SHAP，少井条件下的甜点类型预测 |

**🔴 关键判断（与task_plan既有结论一致，本轮审查是加固证据不是新发现）**：赛道⑤的最大风险不是"缺一个网络架构"，而是**Volve没有现成甜点真值**。必须先由军伟明确定义地质甜点/工程甜点/生产甜点，以及代理标签怎么由孔隙度、渗透率、岩相、断层距离、产量、井位组合构造；不同论文的"甜点"定义不能直接套用比较。**不建议在标签定义冻结前开工。**

## ⑥ 三维模型重建

| 源头 | 链接 | 代码 | 对本赛道价值 |
|---|---|---|---|
| GeoINR 1.0 | [论文](https://gmd.copernicus.org/articles/16/6987/2023/) / [仓库](https://github.com/MichaelHillier/GeoINR) | ✅ | 隐式神经表示，从界面/法向/地层约束恢复连续3D地质体，"稀疏井/层位约束→连续模型"的直接神经基线 |
| GemPy v3 | [主页](https://www.gempy.org/) / [仓库](https://github.com/gempy-project/gempy) | ✅ | 基于界面点和产状的隐式3D构造建模，支持断层/不整合/概率建模，传统可解释强基线 |
| LoopStructural | [论文](https://gmd.copernicus.org/articles/14/3915/2021/) / [仓库](https://github.com/Loop3D/LoopStructural) | ✅ | 支持断层运动学/褶皱/多种隐式插值/多格式导出，适合融合Volve断层+层位+井分层点 |
| DeepSDF | [论文](https://arxiv.org/abs/1901.05103) / [仓库](https://github.com/facebookresearch/DeepSDF) | ✅ | 通用连续SDF重建/补全源头，非地质专用，需加入地层顺序/断层/井约束才有地学意义 |
| MDA-GAN(地震数据插值) | [论文](https://arxiv.org/pdf/2204.03197) / [仓库](https://github.com/douyimin/MDA_GAN) | ✅ | 三判别器3D GAN，复杂缺失模式下的地震数据插值重建 |
| SeisFusion(扩散模型插值) | [论文](https://arxiv.org/abs/2403.11482) / [仓库](https://github.com/WAL-l/SeisFusion) | ✅ | 约束扩散模型做3D地震数据插值重建，精度优于GAN类方法 |

**建议**：先用GemPy/LoopStructural建立可解释结构基线，再比较GeoINR；**注意区分任务目标**——如果赛题实际要的是地震速度体而非地层结构体，应改用OpenFWI/InversionNet3D方向，二者不能混称同一任务。

## 相关文档

- 审查过程：`_wiki-methodology/_top/_findings/P2.5_joint_data_and_algorithm_source_audit.md`
- Codex完整审查报告(含GitHub仓库可访问性`git ls-remote`实测结果)：`_wiki-methodology/_top/_external_reviews/codex_data_algorithm_audit_20260713.md`
- 数据档案：`_wiki-methodology/_wiki/_entities/volve-dataset.md`
