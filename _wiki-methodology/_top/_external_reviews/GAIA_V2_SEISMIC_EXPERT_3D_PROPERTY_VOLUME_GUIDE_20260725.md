# 盖亚 V2 地震/储层专家咨询 · 完整三维属性体与 SCI 可视化实施指南

- 咨询日期：2026-07-25
- 目标项目：`师弟-军伟的比赛-2693e5`（Volve 数据集，六赛道）
- 咨询平台：盖亚 V2（`gaia.yongan.site` · 本机 6186）
- 调用专家：`seismic_interpretation_expert`（地震解释专家）+ `reservoir_modeler`（储层建模专家）
- 本文性质：外部专家复核 + 目标项目真实文件独立复核。**只写本文档，未修改任何模型/数据/split/训练代码/图片。**

---

## 结论摘要（先行）

1. **六赛道里只有 ⑥ 三维重构拥有真正的三维体素网格**（来自 Volve Eclipse 地质模型 PORO 关键字），其余五个赛道当前都不是"完整三维属性体"。
2. **⑥ 虽是真网格，但只有原生数组 K/J/I 索引、没有物理 XYZ**，且交付图是"下采样体素散点"，不是连续体渲染——严格说是"有网格、无物理坐标、渲染成点云"。
3. **① 断层、③ 物性 是 `spatial_context`**：① 是 3 个带真实 Inline/Crossline/TWT 坐标的二维概率平面，③ 是 344 个单井稀疏样点的点云；两者都**不是体**，但坐标真实、标注诚实。
4. **② 地震相、④ 岩相、⑤ 甜点 是 `not_feasible`**：当前产物分别是独立二维分割 patch、仅带 TWT 的逐样本类别预测、无 XYZ 的表格预测，都缺连续网格/物理坐标，**当前诚实不可做成三维体**。
5. **两位盖亚专家的独立判断与项目自身审计一致**：把稀疏井点直接连面/插值成体属于科研误导；真三维属性体 = 规则体素 + 物理坐标 + 可重采样 + 完整标注 + 不确定性披露。
6. **最小但正确的第一轮交付**：先把 ⑥ 的 Eclipse 体和井族做**物理坐标配准**；配准后先出 **2D Kriging 平面图**作 baseline；再以 **SGS 多实现 + SIS 岩相 + 经验变差函数先验（井点后验校正）**走条件模拟出带不确定性的三维体。**在拿到多井坐标/地震属性约束之前，不要装 3D 体，不要用克里金当最终物性场，不要给岩相用克里金。**

> 术语纪律（全文遵守）：**切片/平面 ≠ 点云 ≠ 表面 ≠ 等值面 ≠ 体素体**；**模型直接预测 ≠ 后处理插值**。凡插值/模拟得到的井间值一律标注为"模型假设，非观测"。

---

## 0. 咨询方法与专家调用证据

### 0.1 调用了哪两位专家、为什么选

| 专家 persona_id | 显示名 | 选择理由 |
|---|---|---|
| `seismic_interpretation_expert` | 地震解释专家（Tier 2，parent=geophysics_director） | 六赛道的坐标真源是**地震时间域 Inline/Crossline/TWT**，坐标/几何/时深/体定义与可视化必须由地球物理视角裁定 |
| `reservoir_modeler` | 储层建模专家（Tier 2，parent=reservoir_director） | 从稀疏井到三维孔渗饱属性体属于**储层建模/地质统计**问题（克里金/协同克里金/条件模拟/不确定性），需要该专家交叉复核 |

选择前先枚举了盖亚 V2 全部 79 个 live persona，候选还包括 `seismic_attribute_expert / velocity_modeling_expert / structural_model_software_expert / resqml_model_software_expert / geophysics_director / reservoir_director`。最终按"地震坐标裁定 + 储层地质统计裁定"这条主轴选定上述两位，符合任务"最多再调一位储层/地质统计专家交叉检查"的约束。

### 0.2 真实调用证据（可核实）

- 两位专家均通过 `POST /api/persona/set_active_for_session` 激活成功（`status=ok`），再经 webchat 发送实质咨询、收到实质领域回复。
- journald 硬证据（`gaia.service`，2026-07-25）：
  - `[active_persona] ... 7eb8260e-... → persona=seismic_interpretation_expert`
  - `[active_persona] ... 8b7fa74e-... → persona=reservoir_modeler`
- 会话/回复对应：
  - 地震解释专家：session `7eb8260e-b559-43cd-8837-ac9743440399`，回复约 5.3k 字，工具调用为空（纯领域推理）。
  - 储层建模专家：session `8b7fa74e-eabb-4f10-8a57-9ada90e92678`，回复约 4.3k 字，工具调用为空（纯领域推理）。

### 0.3 诚实边界：这两位专家能给什么、不能给什么

- 这两位是**领域 LLM persona**，给出的是**方法论判断**（概念、可行性、地质统计路线、可视化规范），**不是三维体工具的运行结果**。
- 它们各自绑定的真实工具其实很窄：地震解释专家的实装工具是 `seismic_horizon_simple`（**1D 单道**振幅峰值层位追踪）；储层建模专家是 `parse_well_log_file` + 测井→孔渗饱 XGBoost + `reservoir_kriging_2d`（**2D 平面**克里金，grid_size=50）。**盖亚 V2 当前没有"一键出三维属性体"的工具**——三维体产品仍需按第 C/E 节自建。
- 两位专家**没有直接读取军伟项目文件**（LLM persona 无该项目文件系统访问权）；它们回答的是我向其描述的数据情形。**本文对每个赛道明确区分：【专家意见】= 领域方法论判断；【文件复核】= 我对军伟真实文件的独立核验结论。**下方结论以【文件复核】为准，专家意见提供方法与边界。

---

## A. 概念纠正（点 / 线 / 面 / 切片 / 等值面 / 体素体）

【专家意见 · 地震解释专家】形态与"是不是三维属性体"的对照：

| 形态 | 数据结构 | 物理维度 | 是否"三维属性体" |
|---|---|---|---|
| 正交切片（inline/xline/timeslice） | 二维图像 × N | 2D × N | ❌ |
| 点云（sparse samples） | (x,y,z,value) 列表 | 0D 集合 | ❌ |
| 表面 / horizon / fault stick | 三角网/quad grid | 2D 流形（无厚度） | ❌（除非带属性仍非体） |
| 等值面（isosurface） | 隐函数阈值曲面 | 2D 流形 | ❌ |
| **体素属性体（voxel volume）** | 规则 3D 数组 V[I,J,K] | **真 3D** | ✅ |

【专家意见】**"完整三维属性体"的最低契约**（五条同时满足）：① 规则网格坐标（I×J×K，每轴起点/步长/单位）；② 从索引到物理坐标的单调映射（I/J→Inline/Crossline，K→TWT ms 或 Depth m）；③ 体素全覆盖或带掩膜/NaN（不能"哪里有值填哪里"）；④ 属性连续 + 量纲一致（每 voxel 一个标量，单位/采样/极性固定）；⑤ 可重采样（trilinear/nearest，可在任意切片恢复同一数据）。

【专家意见】**为什么"稀疏井点直接连面/插值成体"是科研误导**（三宗罪）：① 采样支持度不够——几口井在百 km² 工区是 10⁻⁴~10⁻⁶ 采样密度，井间任何插值都是伪信息不是数据，会虚增精度；② 几何/物理不一致——井轨迹是斜的、直接体素化把井上 1 m 样本"涂"满一个大 voxel，人为平滑真实非均质；③ 统计塌方——井点是硬数据、井间是模型假设，二者方差结构不同，基于"体"的统计（连通性/EUR）成为伪结论。**合规做法**：稀疏井 + 地震约束做条件化协同克里金/协同模拟，但必须写明"井间为模型假设，非观测"，并透明披露不确定性。

【文件复核 · 与本项目现状印证】军伟项目自身审计 `_wiki-methodology/_top/_phases/P5_three_dimensional_sci_visualization_contract.md` 已定义同一套判据（`native_volume / spatial_context / not_feasible`），并**明文禁止**"把重复二维切片、任意 z-offset、随机插值、示意网格或合成点云冒充三维结果"。即：项目审计与外部专家的概念判据完全一致，本次咨询在概念层无冲突。

---

## B. 六赛道逐项复核

> 说明：六赛道的训练/推理/产物**不在主仓 master**，而分散在 44 个未合并 git worktree 分支（`.claude/worktrees/`）；`_wiki-methodology/_top/_task_plan.md` 记有"主仓集成（merge）仍未执行"。下方路径为各赛道 worktree 内相对路径。

### ① 断层预测 — `spatial_context`
- 【文件复核】数据粒度/坐标：真实 **Inline/Crossline/TWT(ms)**（Volve ST0202 SEG-Y，网格由 `_pipelines/01_common_preprocess/step_01_load_seismic.py` 从真实道头读出：Inline 9985–10369、Crossline 1932–2536、TWT 0–4500ms@4ms）。
- 【文件复核】模型输出：**二维概率 patch**（patch≈33×65），非体。三维交付=**3 个带真实坐标的二维概率平面**。证据：`.claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/3d_sci_v1/three_d_feasibility.json`（`native_volume:false`，判定 `spatial_context`，明示"平面未插值、未当作连续地震体"）。
- 是否具备完整体：**否**。缺什么：相干体/倾角-方位体/曲率体等**连续属性体**作为模型输入 + 在整网格上推理。
- 能否补成体 / 方法：**能**，走【专家意见】"模型直接体预测"——在整个 SEG-Y 网格上以多地震属性体为输入做 fault likelihood 体推理。推荐最终产品：**断层概率体（fault probability volume）+ 等值面**，附 AUC/QC。断层本身是 2D 流形，交付叫"断层概率体"而非"断层体"。

### ② 地震相分类 — `not_feasible`
- 【文件复核】数据/坐标：训练数据来自 **F3(荷兰)/Penobscot(加拿大)** 公开体，非 Volve；输出为**独立二维分割 patch**（inline/crossline PNG 掩膜），无连续网格/体素点阵/有序遍历。证据：`.claude/worktrees/track-facies/.../facies/_outputs/3d_sci_v1/three_d_feasibility.json`（判定 `not_feasible`，无 3D 产物）。
- 是否具备完整体：**否**。缺什么：把二维分割组织进**规则体素网格**的有序遍历 + 目标层位区间界定。
- 能否补成体 / 方法：**能（需重跑体推理）**。【专家意见】需 3–5 个地震属性体作输入、井旁地震相标定或无监督波形聚类、明确 top/bottom 层位窗。推荐产品：**soft probability 体（每 voxel 每类一个概率）**，非硬标签体。

### ③ 储层物性（PHIF/KLOGH/SW）— `spatial_context`
- 【文件复核】数据/坐标：**344 个稀疏样点、单一井族 `15/9-F-15 D`**（我独立核验 `.claude/worktrees/track-property/_pipelines/02_task_datasets/reservoir/_outputs/test_predictions.csv`：344 数据行，唯一 well_id）；带 Inline/Crossline/TWT + depth_m。
- 【文件复核】模型输出：**逐样本点预测**（列 `PHIF_gt/pred, log1p_KLOGH_gt/pred, SW_gt/pred`），三维交付=**点云**（`volume_used:false, trajectory_used:false, interpolation_used:false`，判定 `spatial_context`）。**它既不是井轨迹，也不是属性体，更没有做插值——不得称其为体或轨迹。**
- 是否具备完整体：**否**。缺什么：**时深转换到深度域（米）**、≥3 口井盲测、地震+井协同的体框架。
- 能否补成体 / 方法：**当前不能裸跑**。【专家意见 · 储层建模专家】1 口井族做 XGBoost 端到端体预测=训练集 0 信息量，只会输出全局均值；单井族也无法估计变差函数。唯一可工程化路径是**地震属性约束的协同克里金/协同模拟**，但前置是"地震体物理 XYZ 重建 + 井轨迹时深配准"。推荐产品：**带不确定度的孔隙度体（P10/P50/P90 或 单值+σ 体）**；渗透率走 log(K) 协同。

### ④ 岩相预测 — `not_feasible`
- 【文件复核】数据/坐标：**120 条逐样本类别预测**，`center_md_m` 全为空、只有 `twt_ms`，无 XYZ/inline/crossline/轨迹。证据：`.claude/worktrees/track-lithofacies/.../lithofacies/_outputs/p5_stage4_confirmation/predictions.json`（判定 `not_feasible`）。
- 是否具备完整体：**否**。缺什么：可验证的第三维（物理坐标或有序空间遍历）。
- 能否补成体 / 方法：**属类别变量，禁用克里金/SGS**。【专家意见】走 **SIS（序贯指示模拟）** 或 **MPS/多点统计**；配齐坐标后输出**离散相带的多实现体**。

### ⑤ 甜点预测 — `not_feasible`
- 【文件复核】数据/坐标：**表格逐样本预测**（7 目标 T1–T7，列如 `sample_id,group,depth_m,actual,prediction`），无注册三维网格、无 X/Y/Z/inline/crossline/UTM/TVDSS/轨迹。证据：`.claude/worktrees/p5-r2-sweetspot/.../sweetspot/p5/_outputs/stage4_confirmation/targets/T*/predictions.csv.gz`（判定 `not_feasible`）。
- 是否具备完整体：**否**。缺什么：目标层位 interval + 输入属性体（阻抗/振幅/频率）+ 标定阈值。
- 能否补成体 / 方法：**需重建体流程**。【专家意见】甜点=f(至少阻抗/振幅/频率三体，最好叠加孔隙度/含油体)，必须落在层位 clip 内 + 井标定阈值；否则=假甜点。推荐产品：**甜点指数概率/得分体**。

### ⑥ 三维模型重构 — `native_volume`（唯一真体）
- 【文件复核】数据/坐标：**真三维地质网格**，来自 Volve **Eclipse 地质模型 PORO** 关键字（`reconstruction/build_dataset.py` 读 `VOLVE_2016.INIT/GRID`）。我独立核验 `predictions.npz`：含 `indices_kji`（N×3 整型）、`volume_shape_kji`、逐体素 `truth/prediction/residual`——**确为规则体素网格**。
- 【文件复核】关键局限：**坐标仅原生数组 K/J/I，未引入物理 XYZ**（feasibility `coordinate_units:"voxel index"`）；且交付图是**下采样体素散点**（`visualize_3d_sci.py`：`ax.scatter(projection="3d")`，`MAX_RENDER_POINTS≈12000`，按 K/J/I 排序均匀抽样）——**是稀疏体素点云，不是连续体渲染/等值面**。
- 是否具备完整体：**网格具备、物理坐标与体渲染不具备**。缺什么：K/J/I↔物理 XYZ 映射（Eclipse GRID/EGRID 的 `MAPAXES`/角点坐标）+ 真正的体渲染/等值面。
- 能否补 / 方法：**能（直接利用现有产物升级）**。恢复物理坐标后即可从"体素散点"升级为"连续体渲染 + 残差体 + 不确定性体"。

### 六赛道总表（文件复核）

| 赛道 | 坐标域（实测） | 输出粒度 | 是否 voxel 网格 | 是否有物理坐标 | 判定 | 补体方法 |
|---|---|---|---|---|---|---|
| ① 断层 | Inline/Crossline/TWT | 3 个 2D 概率平面 | 否 | 是 | spatial_context | 模型直接体预测（多地震属性体） |
| ② 地震相 | Inline/Crossline(F3/Penobscot) | 独立 2D 分割 patch | 否 | 部分 | not_feasible | 重跑体推理→soft 概率体 |
| ③ 物性 | IL/XL/TWT + depth_m | 344 稀疏点(1 井族) | 否(点云) | 是 | spatial_context | 地震约束协同克里金/协同模拟 |
| ④ 岩相 | 仅 TWT | 120 逐样本类别 | 否 | 否(MD 全空) | not_feasible | 配坐标后 SIS/MPS |
| ⑤ 甜点 | depth/cutoff/状态 | 表格逐样本(T1–T7) | 否 | 否 | not_feasible | 层位 clip + 属性体阈值 |
| ⑥ 重构 | 原生 K/J/I(Eclipse) | 稀疏体素散点 | **是** | **否(仅索引)** | native_volume | 恢复 XYZ + 体渲染/不确定性体 |

---

## C. 从当前项目走向完整属性体的技术路线

> 综合两位专家意见 + 军伟数据现实。**每一步先判"缺失是否可补"，不可补就明示停下，绝不跳到下游编造。**

### C.1 SEG-Y 几何与坐标恢复
- 军伟已有真实几何（`step_01_load_seismic.py` 从道头读出 Inline/Crossline 范围、4ms 采样、`CDP_X/CDP_Y` UTM31N、`scalco=-100`）。**这是全项目坐标真源，务必所有赛道统一引用它**，不要各赛道各自造坐标。
- ⑥ 的 Eclipse 体缺物理坐标：从 GRID/EGRID 读 `MAPAXES`/角点坐标建立 K/J/I↔X/Y/Z 映射；**若无真实坐标参数，不要编造坐标**（专家硬约束）。

### C.2 Inline/Crossline/TWT ↔ XY / 时间或深度
- IL/XL↔UTM：用 `step_01` 已拟合的仿射变换（`step_03_load_fault_horizon.py` 已用它把断层棒回投到最近 IL/XL）。
- 时间↔深度：**这是本项目最大短板**。【专家意见】标准做法是 checkshot v(z)→Dix→层速度→v(x,y,z) 体→`Z=∫v dz`；QC 要求井点时深误差 < 1/4 波长（典型 <5ms/<10m）。

### C.3 井轨迹 / checkshot / 时深关系（诚实边界）
- 【文件复核】军伟的 well-tie 是**弱配准**：`step_04_well_tie_weak.py` 用井 pick 点（`Well_picks_Volve_v1.dat` 的 MD/TWT/UTM）做分段线性 MD→TWT，再→最近 IL/XL，**明确不是速度模型/合成记录/checkshot 配准**；且比赛本身不提供 VSP/合成记录/时深表。
- 【专家意见】没有合成记录/井标定，只能用区域速度粗略时深转换，**不能进生产、不能发表**深度域孔渗饱结论。→ 因此③在深度域的严格属性体，在补充时深数据前诚实受限。

### C.4 地震-测井配准、重采样、网格定义
- 【专家意见】well tie：测井编辑(井斜校正到 TVD)→反射系数→子波褶积→合成记录→与井旁道拉伸/压缩使相关>0.7→输出每井 TWT↔MD/TVD 标定表。
- 体素网格三轴契约（强制显式声明）：I(Inline)、J(Crossline)、K(TWT ms 或 Depth m)，各轴起点/步长/单位 + bounding box + CRS/EPSG + 面元 size + 采样率 + Z 极性约定。

### C.5 模型在完整体素网格上推理
- ①②⑤：把模型从"patch/切片"改为在整网格逐 voxel（或分块 tile）推理，输出概率体。
- ⑥：已是整网格；恢复坐标后直接升级可视化。

### C.6 地质统计：克里金 / 协同克里金 / 条件模拟的适用边界
【专家意见 · 储层建模专家】：

| 方法 | 适用 | 禁忌 |
|---|---|---|
| 纯克里金 OK/UK | 只有井点、作 baseline / 风险下限 | 数据稀→变差函数不可靠、远井退化为全局均值（"牛眼"假富集）；**不作最终交付体** |
| 协同克里金/协同模拟 CoKriging/CoSGS | 有连续地震属性体 + 井点硬数据、相关 r>0.5、变差函数可拟合 | 低相关区产生"地震驱动的虚假细节"；协同位置配准误差>半网格会被平滑掩盖 |
| SGS（序贯高斯模拟） | 孔隙度 φ、log(K) 等连续正态变量 | 不能直接给岩相 |
| SIS（序贯指示模拟） | **岩相/沉积微相（类别变量）** | 单实现不光滑（是特征非缺陷） |

- **岩相类别变量绝不能用克里金/SGS**（克里金是连续高斯理论）→ 走 SIS/MPS。
- **为什么克里金过度平滑**：克里金是最优线性无偏估计，最小化估计方差→数学上必然抹平极值→储量偏低、连通性被夸大（隔夹层消失）；**SGS 重现直方图+变差函数**，用随机路径引入的高频对应真实非均质→P10/P50/P90 更可靠。过渡方案：克里金估值 + 残差随机模拟。
- **单井族硬约束**：`reservoir_modeler` 明确——单井族禁止用于估计变差函数（无方向多点对，主/次/垂向变程不可分辨）；可用 Force 2020 同类储层经验变差函数 + 井点直方图后验校正。

### C.7 概率体 / 残差体 / 不确定性体
| 体 | 形态 | 含义 | 验证 |
|---|---|---|---|
| 概率体 P(sand\|x) | 每 voxel [0,1] | 决策可用 | 逐井交叉验证 + 校准曲线 |
| 残差体 R=真值−估值 | 正负连续场 | 未捕获结构 | 检验 R 独立于估值、近似 N(0,σ²) |
| 不确定性体 σ² | 非负 | 距井越远越大 | 远井 σ² 却很小 = 过度平滑警报 |
| P10/P50/P90 多实现 | N 个等概率实现 | 反映真实非均质 | 每实现硬数据吻合 + 变差/直方图重现 |

### C.8 防数据泄漏与过度平滑
【专家意见】防泄漏三件套：① 空间——留一井（LOO），禁止把待预测井邻域当训练；② 时间——地震属性井点标定处不参加训练（Jackknife）；③ 维度——3D CNN 时井附近网格不得既训又验。防过度平滑三件套：① 每个 SGS 实现的实验变差函数与模型误差<10%；② 实现 max/min 与硬数据同分布；③ 高孔渗段体积分数对齐硬数据统计。

---

## D. SCI 级三维可视化方案

【专家意见 · 地震解释专家】不同成果的推荐组合：

| 元素 | 推荐 | 理由 |
|---|---|---|
| 体渲染 volume rendering | ✅ 半透明 + 感知均匀色表（如 viridis） | 看整体结构、避免假细节 |
| Fence diagram | ✅ 三向正交切片 + 测井柱 | 同时看 I/J/K |
| 等值面 isosurface | ✅ **仅在标定阈值后** | 突出甜点/断层连通 |
| 切片（沿 horizon flatten） | ✅ | 反映真实地层结构 |
| 多视图对比 | ✅ prediction vs ground truth vs residual vs uncertainty 同视角并排 | 公平可视化 |
| 分类体 vs 连续体 | 分类体用**离散色表**，连续属性体用**感知均匀色表** | 避免色觉误导 |

**必标注清单**：轴标签（Inline/Crossline/TWT 或 Depth + 单位）、CRS/EPSG、时间域 vs 深度域、极性、colormap 名与 vmin/vmax、scale bar、inline/xline 号、n_voxels、source data、training wells、QC 指标（R²/accuracy/AUC）。

**禁止的视觉误导手段**（评委常见扣分点）：🚫 沿任意方向切片后 rotate 制造"立体感"；🚫 拉伸 colormap 放大微小变化；🚫 隐藏坐标/CRS/单位；🚫 未标定直接给体；🚫 3 个正交切片冒充体；🚫 用切片亮度制造假高产区。

**静态 SCI 主图 vs 交互页职责**：静态主图（PNG/PDF，用于论文正文）承担"结论性对比 + 完整标注"；交互 WebGL/VTK/PyVista/Plotly HTML 承担"审稿人自查体内部结构、任意切片、旋转核对坐标"。

【文件复核 · 本项目现状】现有三维媒体只有 3/6 赛道产出，且都非连续体：⑥ `reconstruction/_outputs/3d_sci_v1/{strict,conditional}/*.png|*.html`（体素散点）、① `fault/.../3d_sci_v1/*spatial_context.png|.html`（3 平面）、③ `reservoir/.../3d_sci_v1/*spatial_context.png|.html`（点云）；②④⑤ 无三维媒体。全库无 `.vtk/.vti/.vtu/.nc` 体格式文件——即当前不存在任何真体渲染/等值面产物。图版规范上，项目 P5 契约已强制"禁止 `plt.title`/`ax.set_title`，面板只用粗体小写 a/b/c"，与本节 SCI 规范不冲突，可直接沿用。

---

## E. 下一轮实施清单（按优先级）

### 优先级 1 — 可直接利用现有产物升级
- **E1｜⑥ 重构：恢复物理坐标 + 升级为体渲染**
  - 输入：`predictions.npz`（`indices_kji/volume_shape_kji/truth/prediction/residual`）+ Eclipse GRID/EGRID 的 `MAPAXES`/角点坐标。
  - 输出：K/J/I↔XYZ 映射表 + 连续体渲染 + 残差体 + 不确定性体（若有多实现）。
  - 验收：任意切片可从体恢复；坐标轴标真实 XYZ/单位；残差体与 `residual` 一致。
  - 失败条件：Eclipse 无可用坐标参数 → **停在体素索引域，明示"无物理坐标"**，不得编造 XYZ。

### 优先级 2 — 需补坐标或重跑完整体推理
- **E2｜① 断层：整网格断层概率体**
  - 输入：Volve ST0202 SEG-Y + 相干/倾角/曲率等地震属性体。
  - 输出：整网格 fault probability 体 + 等值面 + AUC。
  - 验收：覆盖整 Inline/Crossline/TWT 网格、非 3 平面；含盲测 AUC。失败：只能出平面 → 维持 `spatial_context` 诚实标注。
- **E3｜② 地震相 / ⑤ 甜点：体推理**
  - 输入：②多地震属性体 + 层位窗；⑤阻抗/振幅/频率体 + top/bottom horizon clip + 井标定阈值。
  - 输出：②soft 概率体；⑤甜点指数概率体。
  - 验收：落在层位 interval 内、带标定；失败（无层位/无标定）→ 维持 `not_feasible`。

### 优先级 3 — 需重训或引入地质统计建模
- **E4｜③ 物性：地震约束协同模拟属性体**
  - 输入：多井（>5，各带时深关系）+ 地震属性体（AI/弹性阻抗）+ 时深转换。
  - 输出：φ 体（P10/P50/P90）+ log(K) 协同体 + 不确定性体 + 井点交叉验证表。
  - 验收：变差函数实测/模型误差<10%、远井 σ² 增大、LOO 校准合格。失败（仍单井/无地震约束）→ 只交付点云 + 2D Kriging baseline，**不装 3D 体**。
- **E5｜④ 岩相：SIS/MPS 多实现相体**
  - 输入：配齐 XYZ 的岩相硬数据 + 相比例/变差或训练图像。
  - 输出：离散相带多实现体 + 相比例校验。
  - 验收：SIS 单实现非光滑、相比例吻合。失败（MD/坐标缺失）→ 维持 `not_feasible`。

### 优先级 4 — 当前诚实不能完成
- 无配准坐标直接出 3D 体；单井族做变差函数精细分析；1 口井族 XGBoost 端到端体预测；岩相用克里金；仅给克里金估值体当最终物性场；把③点云/①平面/⑥体素散点称作"完整三维属性体"。**以上均按项目 P5 契约与专家判断维持 `not_feasible`/诚实标注，不生成假三维图。**

### 最小但正确的第一轮完整属性体交付建议
1. **物理坐标配准前置**：⑥ Eclipse 体 K/J/I→XYZ；井族 MD/Inc/Azi 最小曲率→TVD/N/E 与地震对齐（缺参数→明示停下）。
2. **2D Kriging 平面图 baseline**（真实可跑）：多井 X/Y + φ 出热力图，能力标注为"2D 储层属性 Kriging 平面插值"，**明确不是 3D 体**。
3. **⑥ 升级为带不确定性的连续体**：真网格恢复坐标后出体渲染 + 残差体 +（可行则）多实现不确定性体。
4. 其余赛道按 E2–E5 逐步补，**每步保留 `native_volume/spatial_context/not_feasible` 三态诚实标注**。

---

## 附录 · 专家意见 vs 文件复核 对照

| 主题 | 【Gaia 专家意见】 | 【文件复核结论】 |
|---|---|---|
| 概念判据 | 体=规则体素+物理坐标+可重采样+完整标注 | 与项目 P5 契约判据一致 |
| ③ 物性 | 单井族不能端到端体预测/不能估变差函数 | 实测 344 点 / 1 井族 / 点云 → 一致 |
| ④ 岩相 | 类别变量禁克里金/SGS，走 SIS | 实测仅 TWT、MD 全空 → `not_feasible` 一致 |
| ⑥ 重构 | 有网格才叫体 | 实测真 Eclipse 体素网格，但仅 K/J/I 无 XYZ、渲染成散点 |
| 时深 | 无合成记录/checkshot 不能出深度域属性体 | 实测 well-tie 为弱配准、比赛不含 VSP/时深表 → 一致受限 |

## 自检（对照任务要求）

- ✅ 文件存在且非空：本文件已写入指定路径 `_wiki-methodology/_top/_external_reviews/GAIA_V2_SEISMIC_EXPERT_3D_PROPERTY_VOLUME_GUIDE_20260725.md`。
- ✅ 六赛道均有独立结论（B 节逐项 + 总表）。
- ✅ 明确区分点/面/切片/体（A 节形态表 + 全文术语纪律）。
- ✅ 没有把插值结果冒充模型直接预测（③ 标注 `interpolation_used:false` 为点云；所有插值/模拟值标注"模型假设，非观测"）。
- ✅ 盖亚专家调用有真实证据（0.2 节 session id + journald `active_persona` 记录 + 实质回复）。
- ✅ 未修改交付文件之外的目标项目文件（本轮仅新建本文件；盘点与核验均为只读）。

## 引用的目标项目真实相对路径（证据）
- `_wiki-methodology/_top/_phases/P5_three_dimensional_sci_visualization_contract.md`
- `_wiki-methodology/_top/_external_reviews/codex_data_algorithm_audit_20260713.md`
- `_wiki-methodology/_top/_task_plan.md`
- `_meta/_data_registry.yml` · `_wiki-methodology/_wiki/_entities/volve-dataset.md`
- `_pipelines/01_common_preprocess/step_01_load_seismic.py` · `step_03_load_fault_horizon.py` · `step_04_well_tie_weak.py`
- `.claude/worktrees/track-fault/_pipelines/02_task_datasets/fault/_outputs/3d_sci_v1/three_d_feasibility.json`
- `.claude/worktrees/track-facies/.../facies/_outputs/3d_sci_v1/three_d_feasibility.json`
- `.claude/worktrees/track-property/_pipelines/02_task_datasets/reservoir/_outputs/test_predictions.csv`（独立核验：344 行 / 单井 15/9-F-15 D）
- `.claude/worktrees/track-lithofacies/.../lithofacies/_outputs/p5_stage4_confirmation/predictions.json`
- `.claude/worktrees/p5-r2-sweetspot/.../sweetspot/p5/_outputs/stage4_confirmation/targets/T*/predictions.csv.gz`
- `.claude/worktrees/p5-r2-reconstruction-v2/.../reconstruction/...predictions.npz`（独立核验 npz 结构：indices_kji / volume_shape_kji / truth / prediction / residual）+ `reconstruction/build_dataset.py`

---
*本文档由盖亚 V2 地震解释专家 + 储层建模专家咨询 + 目标项目真实文件独立复核生成；仅新增本文件，未改动任何模型/数据/split/训练代码/图片。*
