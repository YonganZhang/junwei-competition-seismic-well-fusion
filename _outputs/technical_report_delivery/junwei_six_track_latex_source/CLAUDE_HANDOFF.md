# 六赛道技术报告制作与 Claude 接手说明

## 一、交接结论

本轮工作的核心交付是一份 46 页中文 LaTeX 技术报告。报告围绕六个地学赛道展开，统一说明任务背景、方法原理、训练策略、实验设计、评估指标与实验结果，并把基础模型放入可比较、可回退的实验框架中。

当前成品如下：

- PDF 真源：`_paper/technical_report/build/junwei_six_track_technical_report.pdf`
- LaTeX 工程：`_paper/technical_report/`
- 稳定下载地址：`https://share.yongan.site/junwei-six-track-report/junwei_six_track_technical_report.pdf`
- 当前版本地址：`https://share.yongan.site/junwei-six-track-report/junwei_six_track_technical_report.20260731-032543.pdf`
- PDF 页数：46 页，A4
- PDF SHA-256：`78899a5bc0152bd210693a622ee1ef17e0fe489cd2587b36020f51a59799c4a4`
- 报告清单：`_paper/technical_report/report_manifest.yml`
- LaTeX 源码包：`https://share.yongan.site/junwei-six-track-latex-source/junwei_six_track_latex_source_latest.zip`

稳定地址后续应继续复用。重新编译并发布时，覆盖同名文件，不要另造新的稳定链接。

## 二、用户要求如何落成报告结构

用户先要求整理六赛道技术报告，随后把 PDF 设为最高优先级，并逐步固定了三项写作约束。

第一，六个赛道必须采用相同的学术目录。每个赛道严格保留以下六节，顺序不能改变：

1. 任务背景
2. 方法原理
3. 训练策略
4. 实验设计
5. 评估指标
6. 实验结果

第二，标题只使用学术概念，不把“门禁、状态、证据边界、流水线”等工程词写入章节标题。工程复现信息只放在附录、清单与交接材料中。

第三，正文遵循永安写作内核。面向低耐心的普通大学生，每段只推进一条主线；方法章节先解释为什么需要该模型，再给公式和信息流；实验章节先交代输入、输出、领域背景与动机，再报告指标与结果。项目级约束已写入根目录 `AGENTS.md`。

## 三、报告的写作方法

### 3.1 先固定科学问题，再写模型

报告没有从模型名称出发堆叠组件，而是先把每个赛道改写为一个可以被实验回答的问题。例如，断层赛道研究三维预训练表征能否在稀疏正样本条件下改善连续断层识别；三维重建赛道研究预训练体表征能否在不破坏克里金基线的前提下补回空间非均质性。

这一写法使大模型成为待检验的研究变量，而不是报告的宣传词。

### 3.2 基础模型采用统一归因合同

六赛道共用同一条归因原则：预训练分支必须同时比较原有强基线和同架构随机初始化。数据划分、输入处理、评价指标与训练预算应保持一致。只有预训练初始化在主要指标和领域结构诊断上同时改善，模型才可晋级。

完整合同位于：

`_pipelines/05_research_visualization_expansion/foundation_model_experiment_contract.json`

该合同解决了“大模型更复杂，所以结果更好”这一常见归因漏洞。随机初始化对照未完成时，正文只报告“分支已经连通”或“开发集系统更优”，不宣称收益来自预训练知识。

### 3.3 六赛道的方法主线

| 赛道 | 强科学基线 | 基础模型候选 | 报告中的研究角色 |
|---|---|---|---|
| 断层预测 | 多尺度不连续属性与 Logistic 概率头 | SAM-Med3D | 验证三维预训练表征能否改善稀疏断层识别；当前受合法三维开发折限制 |
| 地震相分割 | FPN-ResNet18 / DeepLabV3+ | SAM2.1 Hiera | 用 CNN 特征查询 Hiera 表征；开发集增益尚需随机初始化归因 |
| 储层物性 | PHIF/KLOGH 的 ExtraTrees 与 SW 的 XGBoost | TabICL | 作为相同特征表上的表格预训练对照；当前未晋级 |
| 岩相分类 | 固定九类 XGBoost | MOMENT-1-base | 比较真实预训练权重、随机初始化与树模型；预训练效应弱且未超过 XGBoost |
| 甜点评价 | 七目标独立路由；T3 含历史均值与 XGBoost 对照 | Chronos-2 | 只用于 T3 因果时间序列，不把七个目标合成伪甜点评分 |
| 三维重建 | PyKrige OK3D | OpenMind-MAE | 只学习受控残差；若空间折退化，融合门退回克里金 |

### 3.4 方法、训练与结果的分工

“方法原理”只解释模型与公式。“训练策略”说明损失、划分、优化、阈值和模型选择。“实验设计”说明输入、输出、对照和研究问题。“评估指标”给出公式与物理意义。“实验结果”只报告由现有证据支持的结论。

这种分工避免同一段同时介绍模型、训练命令和结果，降低正文的工程感。

## 四、架构图如何制作

六张 Framework 图使用 Nano Banana 2 生成。生成过程不是一次完成，而是经历了多轮提示词修订和人工筛选。

提示词位于：

- `_paper/technical_report/prompts/nano_banana_v1/`
- `_paper/technical_report/prompts/nano_banana_v2/`
- `_paper/technical_report/prompts/nano_banana_v3/`
- `_paper/technical_report/prompts/nano_banana_v4/`
- `_paper/technical_report/prompts/nano_banana_v5/`

早期版本的问题是信息量不足、图内出现拼写幻觉、工程状态词过多。Claude 审查后重新生成 v5，并对储层物性与岩相图单独做 v5r2 修订。当前 LaTeX 使用的六张图位于：

`_paper/technical_report/figures/architecture/`

最终选择为：

- 断层：v5
- 地震相：v5
- 储层物性：v5r2
- 岩相：v5r2
- 甜点：v5
- 三维重建：v5

架构图同时展示强基线路径、基础模型路径、融合位置、消融轨道与输出，但它们只承担方法解释功能，不能作为模型性能证据。

## 五、实验图与三维图如何制作

### 5.1 统一科研图流水线

新增的科研图入口是：

`_pipelines/05_research_visualization_expansion/render_research_figures.py`

该脚本不训练模型，也不选择模型。它只读取已归档的观测、预测和指标，生成可追溯的科研图。绘图公共层位于：

`_code/visualization/geo3d_viz.py`

输出位于：

`_outputs/research_visualization_expansion/v1/`

其中包含 12 组 PNG/PDF 静态图、3 个交互式 HTML、QA 接触表和逐文件 SHA-256 清单。输入来源、证据类型、尺寸和科学限制写入：

`_outputs/research_visualization_expansion/v1/artifact_manifest.json`

### 5.2 领域三维表达

断层图直接从约 1.1 GB 的真实 SEG-Y 数据中惰性读取 Inline 10243，并叠合官方断层解释控制点。图中没有把稀疏断层棒插值成未验证的连续概率体。

地震相图使用 F3 连续 inline 切片和同位置十类参考解释。参考类别是数据证据，不是 SAM2 的稠密预测。

三维重建使用 Volve MAPAXES 配准网格，展示参考孔隙度、条件留出真值、重建结果和残差，并增加三组正交切片和方向经验变差函数。这里的残差是预测减真值，不是不确定性。

储层物性与岩相没有伪造三维体。前者使用井深诊断和 PHIF–KLOGH 一致性；后者因缺少已验证 XYZ 井轨迹，只展示 TWT 岩相序列。

## 六、这次扩展得到的主要科学认识

1. 三维重建的点值误差不足以判断空间质量。当前条件重建在约 16.6 m 的 K 向距离处只保留真值约 `5.4e-5` 的变差，在约 612 m 的 I–J 向距离处只保留约 `0.0066`。主要失败模式是过度平滑，而不是均值偏移。
2. 岩相已知留出集有 120 条记录，其中 70 条误分类。错误样本和正确样本的平均最大类别概率约为 0.506 与 0.501，当前概率不能有效识别错误。
3. 甜点 T3 已知留出集的 `R²=-0.0636`。Chronos-2 的开发期收益不能替代冻结时间留出评价。
4. 储层物性中，PHIF 与 `log1p(KLOGH)` 的 Pearson 相关系数由观测约 0.704 变为预测约 0.790。模型保持了主趋势，但可能压缩真实散布，因此必须与逐井深残差和区间覆盖联合判断。
5. OpenMind-MAE 的真实预训练权重优于随机初始化，但仍明显弱于克里金。这说明预训练表征包含可迁移信息，却不足以直接替代地统计方法。

这些负面或有限结果被保留在报告中，因为它们直接定义了下一轮研究应解决的问题。

## 七、LaTeX 工程组织

主入口为 `_paper/technical_report/src/main.tex`。六赛道分别拆成独立章节：

- `sections/02_fault.tex`
- `sections/03_facies.tex`
- `sections/04_property.tex`
- `sections/05_lithofacies.tex`
- `sections/06_sweetspot.tex`
- `sections/07_reconstruction.tex`

技术基础、概述、结论和复现说明也各自独立。`main.tex` 统一管理字体、页眉、图片宏、双图宏与架构图宏。正文已有 17 处图表交叉引用，避免图片和叙述脱节。

参考文献位于 `src/references.bib`。报告通过 XeLaTeX 编译，再由 Ghostscript 压缩为下载版 PDF。构建命令是：

```bash
bash _paper/technical_report/build_report.sh
```

构建脚本同时生成打印版和压缩版。最终下载版固定为：

`_paper/technical_report/build/junwei_six_track_technical_report.pdf`

## 八、审查与验收

Claude 的前一轮完整审查位于：

`_reports/_foreign_aid/20260730T225812__claude__1027349/result.md`

Claude 当时提出三个主要问题：架构图存在文字幻觉，正文缺少图表交叉引用，地震相开发集收益的归因表述过强。当前版本已经分别通过 v5/v5r2 重绘、增加 17 处交叉引用和改写证据口径完成修复。

后续 Gemini 科研图审查位于：

`_reports/_foreign_aid/20260731T023009__gemini__1801638/result.md`

该审查推动了真实断层地震剖面、岩石物理交会图、T3 实测–预测与 Q–Q 图、方向变差函数的加入。

最终验收结果：

- XeLaTeX 编译成功，46 页；
- 无未定义引用、未定义文献或 Overfull hbox；
- 中文、英文字体均已嵌入；
- 科研图流水线 4 项测试通过；
- 本地 PDF 与公网稳定文件 SHA-256 一致；
- 稳定地址和版本地址均返回 HTTP 200。

科研图测试命令：

```bash
python3 -m unittest -v \
  _pipelines.05_research_visualization_expansion.tests.test_research_visualization_expansion
```

快速重绘命令：

```bash
python3 _pipelines/05_research_visualization_expansion/render_research_figures.py \
  --skip-interactive
```

## 九、Claude 接手时必须保留的边界

1. 不得把 F3、Penobscot 与 Volve 当作同一已配准空间体。
2. 没有 XYZ 的岩相结果不能堆叠成三维体。
3. 残差不等于不确定性；没有 ensemble 或 posterior 时不要写“不确定性体”。
4. 架构图是生成式解释图，不是数据证据。
5. 开发集收益不等于独立留出收益；随机初始化对照不完整时不要把收益归因于预训练。
6. 六个固定四字节标题不能改变，也不要把工程状态写进标题。
7. 正文继续遵循永安写作内核，一段一条主线，删除工具流水账和版本修补史。
8. 当前工作区包含大量既有修改与未跟踪资产。禁止 `git reset --hard`、`git clean`、`git add -A` 或覆盖用户文件。

## 十、建议 Claude 的读取顺序

Claude 接手后应按以下顺序读取，避免从图片或零散脚本反推全貌：

1. `AGENTS.md`
2. `_wiki-methodology/_top/_phases/P13_six_track_research_depth_expansion.md`
3. `_paper/technical_report/report_manifest.yml`
4. 本交接报告
5. `_paper/technical_report/src/main.tex`
6. 六个赛道章节
7. `foundation_model_experiment_contract.json`
8. `artifact_manifest.json`
9. 最终 PDF

若需要继续修改，建议一次只处理一个赛道，并在修改后完成四项检查：正文证据口径、图片路径、交叉引用、完整编译。改完 PDF 后重新计算 SHA-256，更新 `report_manifest.yml`，再使用相同发布主题覆盖稳定链接：

```bash
bash /mnt/data/yongan-admin-2/.codex/skills/share-docs/scripts/pubfile.sh \
  _paper/technical_report/build/junwei_six_track_technical_report.pdf \
  junwei-six-track-report
```

## 十一、可以直接交给 Claude 的任务说明

> 请先按本交接报告的读取顺序理解现有 46 页技术报告。不要从零重写，也不要修改六赛道固定目录。后续工作应以 `_paper/technical_report/report_manifest.yml` 为单一报告入口，以科研图 artifact manifest 为数据图血缘真源。所有大模型收益必须对照强基线和同架构随机初始化；没有合法证据时保留负面结果或待验证状态。修改完成后重新编译、检查引用、逐页抽检，并覆盖既有稳定 PDF 地址。
