# Foreign Aid Result

JOB_ID: 20260730T225812__claude__1027349
PROVIDER: claude
COMPLETION_STATUS: COMPLETE
ERROR_CLASS: NONE

## Summary
只读复核已完成：实际编译产物、LaTeX 真源与 PDF（含 6 张全页架构图 PNG）均已逐一阅读。报告的目录与六节固定模板高度规范、章节标题学术、无工程状态词进标题，编译零错误、无未定义引用、图件与引用一一存在，各赛道表格数字与正文内部自洽。没有 P0（无编译失败、无捏造头条数字、无秘密泄漏）。存在 3 个 P1 与若干 P2，最突出的是：六张架构图是 Nano Banana 生成、含幻觉乱码英文与工程状态词（直接违反"去 AI 味/标题不写工程状态"要求）；全文没有任何图/表交叉引用（违反"图文对应"）；概述对地震相赛道的证据边界与其正文自相矛盾。以上均可用现有材料修复，无需新增实验；唯一需回原始 JSON 复核的是三维重建里克里金出现两个不同数值。

## Work Performed
- 核对目录与六节模板：读取 `build/main.toc`，确认六赛道均严格使用 任务背景/方法原理/训练策略/实验设计/评估指标/实验结果，顺序一致，章节标题学术、无工程状态词。
- 编译健康检查：`build/main.log` 无 undefined reference / citation，Overfull hbox 计数为 0，`references.bib` 六个基础模型 key 全部解析，`figures/` 下被引用的 38 张结果图 + 6 张架构图全部存在。
- 交叉引用审计：`grep` 全部 `src/sections/` 未发现任何 `\ref/\cref/\autoref`；结果图/表未设 `\label`。
- PDF 视觉核对：读取 fault 结果页(物理 p9-11)、facies 页(物理 p12)，并直接读取全部 6 张 `figures/architecture/*_architecture.png`。
- 数字一致性核对：fault / facies(0.1327→0.2487, 增益0.1160) / property / sweetspot(T3 MAE 186.57 及±百分比) 表格与正文逐项复算，均自洽。
- 术语与证据边界核对：概述 vs 各赛道结论、manifest 状态码 vs 正文口径、TabICL 命名。

## Evidence
P1-1 架构图含 AI 幻觉乱码 + 工程状态词（违反"去 AI 味 / 不写工程状态词"）
- `figures/architecture/fault_architecture.png`：`Flarsk encoder`（应为 Mask/编码器，纯幻觉词）、`Local discontinuiuity`（拼写错误 discontinuity）、`Explicit three mask`（应为 three-class mask）、右侧两个不同输出体都标 `3D fault-probability volume`（重复/错标）；底部 `3D legality gate fails`、`Blocked foundation ranking`（工程状态词）。
- `figures/architecture/sweetspot_architecture.png`：T5/T6/T7 输出框三次出现 `Status status cards`（重复词乱码）；顶栏与路由框内 `Seven-target evidence router...prevents any synthetic composite sweetspot score` 整句重复；`not feasible / blocked / promotion gate` 工程状态词。
- `figures/architecture/reconstruction_architecture.png`：`embedde ccard`（乱码）、`Lottom training`（应为 Bottom）、`structured queries query dimensi`（截断乱码）、`gate-zero exact PyKrige visits the result result unchanged`（`result result` 重复）。
- `figures/architecture/lithofacies_architecture.png` / `facies` / `property`：无严重乱码，但同样全英文且嵌入 `no-verified-gain promotion gate`、`frozen test firewall`、`40-update strong CNN control` 等工程流水线术语。
- 根因：架构图由 `prompts/nano_banana_v1/` 经 Nano Banana 2 图像生成（见 `src/sections/A_provenance.tex` A.2），图像模型对图内文字产生幻觉/重复，且把 pipeline 状态词画进图里。正文标题层做得很干净，问题全集中在图内。
- 影响：面向"耐心有限的普通大学生"，全页英文 + 可见乱码词直接暴露 AI 味、损伤可信度与可读性；这是当前报告最刺眼的缺陷。

P1-2 全文零图/表交叉引用（违反"图文对应"）
- `grep -rE '\\(auto|c|C)?ref\{' src/sections/` 返回空；`\ResultFigure/\ResultPair` 宏未生成 `\label`，表格无 `\label`；架构图/训练流程图虽有 `\label`（main.tex 宏 `#3/#7`）但正文从不 `\cref`。
- 影响：图/表虽被自动编号（图3.1、表3.1…），但正文没有一处"如图3.1所示""见表5.1"。读者无法把某段文字对应到具体图/表，架构图又用 `[p]` 整页浮动、易漂离本赛道正文，横向阅读断裂。

P1-3 概述对地震相证据边界与正文自相矛盾
- `00_executive_summary.tex:13-16`：把 SAM2 交叉注意力列为明确成功（"明显提高了平均交并比"），随后仅对"物性预测和岩相分类"补"仍需补充同架构随机权重对照"的保留。
- 但 `03_facies.tex:133-135` 自陈 facies 同样"仍需谨慎区分收益来自预训练权重还是额外模型容量……使用相同结构、相同参数量和随机初始化的 SAM2 分支进行对照"。
- 即：facies 需要与 property/lithofacies 完全相同的对照，却在概述里被单独归入"已胜出"，与报告核心论点（dev≠holdout、可归因才算数、见 `report_manifest.yml` claim_policy 与 facies 状态 `DEVELOPMENT_GAIN_ATTRIBUTION_PENDING`）不一致。

P2-4 facies 训练策略与实验设计重复描述（"训练/设计混写"）
- 三臂 40/160-update 设计在 `03_facies.tex:47-49`（训练策略）与 `03_facies.tex:65-66`（实验设计）各写一遍，内容几乎重合。

P2-5 facies 头条用开发集增益，已知留出集数字缺席
- 结果表 caption 明写"开发集结果"(`03_facies.tex:94`)，正文头条"0.1363→0.2922""系统更好"(`:112,:135`) 均为 dev 数；同时展示了 F3/Penobscot 已知留出集诊断图(`:129-130`) 却从不给出留出 mIoU 数值。读者无法判断 dev 增益是否在留出集成立——与报告反复强调的"development gain is not holdout gain"张力最大处。

P2-6 TabICL / TabICLv2 命名不一致
- `01_technology.tex:15` 与 `references.bib`(qu2025tabicl 标题) 作 `TabICL`；`04_property.tex:26,46` 与 manifest 作 `TabICLv2`。同一模型两种名字，且被引文献并非"v2"。

P2-7 三维重建克里金出现两个未解释的数值
- 表内普通克里金 RMSE `0.02845` / MAE `0.02141`(`07_reconstruction.tex:90,93`)；正文 `:104` 又称"同一实验中的普通克里金 0.0212"。两值分别来自 strict 外层折与 direct 宏折实验，但正文未点明口径差异，且 0.0212 与表内 MAE 0.02141 数值接近、易被误读为 RMSE/MAE 混用。属需回原始 JSON 复核的一处（低置信度，可能只是缺一句口径说明）。

P2-8 少量公式符号未定义
- `05_lithofacies.tex` 融合式中的激活 $\phi$ 未定义；`02_fault.tex` 逻辑回归 $\bm{w},b$ 与各交叉注意力式的投影矩阵 $\bm{W}_Q/\bm{W}_K/\bm{W}_V$ 未显式说明（对"耐心有限普通学生"可各补半句）。TP/FP/FN 全文未一次性定义（通用，属可选补充）。

P2-9 图件全英文 + 页眉版本状态词
- 六张架构图为纯英文，面向中文普通学生读者存在语言门槛；页眉每页 `技术报告首版`、扉页 `状态：内部审阅版`(`main.tex` 与标题页) 属版本/状态标注，与"不写工程状态词"精神轻微抵触（较轻，可保留或改中性版本号）。

P2-10 fault "阶段比较"图信息量为零 + 英文工程注释
- `figures/results/fault/before_after_primary_metric.png`（正文 `02_fault.tex` 末）实际只有一根柱 `Before(audited_v2)=0.006933`，`After(SAM-Med3D gate)` 为 `data_blocked / no legal 3D fold` 空注释；caption 却称"只比较已具备相同评价口径的结果"。该图无对照信息，且 `audited_v2 / data_blocked / no legal 3D fold` 为英文工程词。建议删图或改为纯文字状态说明。

## Files Changed Or Reviewed
- 只读审查，未修改任何文件。
- 已读：`AGENTS.md`；`_paper/technical_report/report_manifest.yml`；`_paper/technical_report/src/main.tex`、`references.bib`、`sections/00`~`08`、`A_provenance.tex`；编译产物 `build/main.toc`、`build/main.log`；`build/junwei_six_track_technical_report.pdf`(多页)；`figures/architecture/*_architecture.png`(全 6 张)；`figures/results/*` 目录清单。

## Errors Or Blockers
- None（三维重建 0.0212 vs 0.02845 需主控回原始 JSON 复核，非阻塞）。

## Next Steps
优先级清单（主控执行）：
1. [P1] 修复架构图：重画为 TikZ（与现有训练流程图同栈、可控无幻觉）或用修正提示词重生成，务必消除 `Flarsk`/`discontinuiuity`/`embedde ccard`/`Lottom`/`Status status cards`/`result result` 等乱码与重复标签，去掉图内 `blocked/gate/firewall/promotion gate` 等状态词；建议图内文字改中文或中英对照。此为最高 ROI。
2. [P1] 补图/表交叉引用：给 `\ResultFigure/\ResultPair` 宏与各表加 `\label`，正文关键处补"如图X/表X所示"；架构图 `[p]` 改 `[htbp]` 或就近 `\cref`，恢复图文对应。
3. [P1] 统一 facies 证据口径：概述(`00:13-16`)把 facies 与 property/lithofacies 一样标注"分支已连通、随机权重对照未完成"，不单列为已胜出。
4. [P2] 删去 facies 训练策略/实验设计中三臂设计的重复段(`03:47-49` vs `65-66`)，只在实验设计保留一处。
5. [P2] facies 结果补一句已知留出集 mIoU 数值或明确"留出评价仅作诊断、未纳入头条"。
6. [P2] 全文统一 `TabICL` 命名（与 bib 一致，除非确有 v2 文献可引）。
7. [P2] 三维重建补一句克里金 0.02845(strict) 与 0.0212(direct 宏折) 的口径差异；先回 JSON 核实 0.0212 是否为 RMSE。
8. [P2] 补 $\phi$、$\bm{w},b$、投影矩阵等符号半句定义；可选统一 TP/FP/FN 定义。
9. [P2] 视需要中性化页眉/扉页版本状态词；处理 fault before/after 单柱图。
