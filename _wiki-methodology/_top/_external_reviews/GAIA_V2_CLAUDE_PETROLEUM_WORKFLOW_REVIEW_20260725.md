---
phase_id: P5.2
status: accepted
owner_col: COL2
source: external
created_at: 2026-07-25
status_basis: 终审结论 ✅ PASS（附必须修改项）
---

# 第六赛道 · 三维属性体流水线独立终审(石油地质建模/油藏工程视角)

- 审查日期：2026-07-25
- 审查对象：`track-6 reconstruction` 新三维属性体流水线（worktree `p5-r2-reconstruction-v2`）
- 审查人角色：Gaia V2 石油地质建模与油藏工程专家（独立终审）
- 审查方式：实际读取代码+产物+图像并逐项核验；**未修改任何代码/数据/图片**，仅新增本报告。

---

## 终审结论：✅ PASS（附必须修改项）

**该三维属性体可视化流水线在物理语义、参考/预测分离、strict/conditional 边界、残差与不确定性区分、静态精确坐标 vs 浏览器规则化坐标的诚实表述、以及"无 scatter/点云的正常石油属性体展示流程"六个核验点上全部通过。** 工程与科学诚实度高。

⚠️ **但一个科学事实必须随所有交付物显式传递**：底层**模型预测几乎是常数**（strict `R²=-0.34`、conditional `R²=0.014`，prediction `std≈0.0015`），即**当前重建相对"常数均值基线"无有效预测技能**。这不是可视化流水线的缺陷（流水线已忠实渲染并在 `caption.md` 报了负 R²），但**交互 HTML 的图注未带该指标**，PNG 单看又易被误读为成功重建 —— 故列为**必须修改项**（披露传递，而非纠错）。

> 判定口径：PASS 针对"可视化流水线的物理正确性与诚实性"；模型预测技能问题是上游建模的独立结论，必须在展示层充分披露，不得让 panel b 被当作成功的孔隙度重建。

---

## 审查范围与实际已读文件

| 文件 | 已读 | 关键核验点 |
|---|---|---|
| `.../reconstruction/visualize_property_volume.py`（1079 行，SHA 锁于 manifest） | ✅ 全文 | 全流程实现 |
| `.../reconstruction/build_dataset.py`（`_parse_eclipse_grid` 等） | ✅ 相关段 | MAPAXES/PORO 解析 |
| `.../_tests/test_visualize_property_volume.py`（65 行） | ✅ 全文 | 不变量测试 |
| `.../3d_property_volume_v2/artifact_manifest.json` | ✅ 全文 | 声明契约 |
| `reference/ | strict/ | conditional/` 三份 `caption.md` | ✅ 全部 | 图注诚实性 |
| `reference/reference_property_volume.npz` | ✅ 数值核验 | 物理坐标真伪 |
| `reference/reference_volume.png` | ✅ 视觉核验 | 真体渲染 |
| `strict/heldout_volume_comparison.png` | ✅ 视觉核验 | truth/pred/residual 三联 |
| 三份 `*.html`（reference/strict/conditional） | ✅ trace 解析 | Plotly trace 类型/轴标 |

---

## 逐项核验（逐条证据）

### ① Eclipse GRID / INIT / MAPAXES / PORO 物理语义 — ✅ 通过
- **MAPAXES 变换正确**：`build_dataset.py:_parse_eclipse_grid`（L173–240）读 `VOLVE_2016.GRID` 的 `DIMENS/MAPAXES/COORDS/CORNERS`；cell 中心=8 角点均值（L216 `corners.reshape(8,3).mean(axis=0)`）；MAPAXES 按标准三点约定（L225–227 注释"point-on-Y-axis, origin, point-on-X-axis"）把局部坐标投影为 UTM（L231–237）。深度取角点均值 z、**不**被 MAPAXES 水平变换污染 —— 符合"MAPAXES 只定义水平地图投影"的石油惯例。
- **PORO 双源交叉校验**：`visualize_property_volume.py:180–181` 断言 `ascii_poro[active_flat] == init_poro`，不一致即 `raise` —— INIT 二进制 PORO 与 ASCII GRDECL 导出在活动格上必须逐格相等。
- **产物物理坐标为真**（我独立读 `reference_property_volume.npz`）：easting `432851.2–437175.2`、northing `6477478.5–6480275.0`、depth `2800.7–3543.8 m` = 真实 Volve UTM31N + Hugin 储层深度；poro `0.001–0.2918`（mean 0.2199）；inactive→NaN。**非索引、非合成**。
- **CRS 诚实**：manifest `coordinate_contract.crs_note`="GRID provides MAPAXES but no EPSG code; no unverified EPSG claim is made" —— 不编造 EPSG。
- 证据：`build_dataset.py:173-240` · `visualize_property_volume.py:159-221,180-181` · `artifact_manifest.json:eclipse_source.*` · npz 实测。

### ② 全场参考属性体 vs 区域模型预测体 是否分离 — ✅ 通过
- **物理分目录 + 语义分离**：`reference/` 为全场活动格 PHIF_NW/PORO 参考体；`strict/`、`conditional/` 为区域 truth/prediction/residual。
- **参考体明确不是预测**：`reference/caption.md`="This is a reference property body, not a model prediction and not uncertainty"；manifest `volume_contract.forbidden_mislabels` 首两条即禁止"regional model volume as full-field prediction / reference PORO as model prediction"。
- **区域体是真实子域**：`_regional_model_volume`（L249–253）按 archive 索引 min..max 裁出 bounding box，形状 strict `[63,56,41]`、conditional `[63,52,29]`，远小于全场 `[63,100,108]` —— 结构上不可能冒充全场。
- **truth 精确等于原生 Eclipse PORO**：L244–248 `archive["truth"] == volume.porosity[indices]` 不等即 `raise`；测试 `test_archived_model_cells_map_to_physical_regional_volumes` 复核。
- 证据：`visualize_property_volume.py:224-283` · 三份 caption · manifest `volume_contract`。

### ③ strict 与 conditional 边界 — ✅ 通过
- **边界表述与证据类一致**：strict="no test-region well constraints"，conditional="test-region constraints supplied and exact well cells excluded"（代码 L884–888、L903–907；caption 同步）。
- **诚实标注非盲测**：两模式 `evidence_class=previously_seen_reusable_holdout`、`prior_test_consumed=true`、`fresh_blind=false`（manifest `modes.*`；caption 明写）—— **不冒充新盲测**（forbidden_mislabels 末条）。
- **指标随模式落盘**：strict `RMSE=0.0356/MAE=0.0272/R²=-0.34`；conditional `RMSE=0.021/MAE=0.015/R²=0.0137`（`caption.md` 由代码 L913–914 从 archive metrics 注入）。
- 证据：`visualize_property_volume.py:884-917` · `artifact_manifest.json:modes.strict|conditional` · 两份 caption。

### ④ 残差 与 不确定性 — ✅ 通过
- **残差=预测−真实且被校验**：L262–270 断言 `residual == prediction - truth`（atol 1e-7），不满足即 `raise`；VTS 场名显式 `residual_prediction_minus_truth`（L385）。
- **不做不确定性冒充**：manifest `volume_contract.uncertainty_available=false` + `uncertainty_note`="residual is error, not uncertainty"；每份 caption 末句"No ensemble or posterior samples are archived, so no uncertainty body is claimed"；测试断言源码含"residual is error, not uncertainty"。
- 石油口径正确：残差是对已知 truth 的误差，不确定性需 ensemble/后验样本 —— 二者严格区分，**未编造 P10/P50/P90 或 σ 体**。
- 证据：`visualize_property_volume.py:262-270,385` · manifest `volume_contract` · 三份 caption · 测试第 4 例。

### ⑤ 静态 VTK 精确中心格 vs 浏览器规则化 I/J/K —— 表述是否诚实 — ✅ 通过（本次重点）
- **静态用精确 MAPAXES 坐标**：`_structured_grid`（L341–356）以精确 `easting/northing/-depth` 建 `pv.StructuredGrid`，静态 PNG（体渲染+正交切片）与 `.vts` 均基于此 —— 真曲线角点中心几何。
- **浏览器显式规则化 + 三处披露**：`_rectilinear_display_coordinates`（L632–664）docstring 明写"Plotly Volume does not support curvilinear/corner-point physical geometry. Along-grid I/J distances retain measured median horizontal spacing, while K remains the global stratigraphic layer index. Exact MAPAXES centres remain in the NPZ/VTS assets and in the VTK static render"；轴标为"Along-I distance (m)/Along-J distance (m)/Global K layer index"（L792–794、876–878）；hovertemplate 同口径；manifest `coordinate_contract.interactive_display_vertical` 与 caption HTML 三处一致披露。
- **结论**：浏览器视图**未**声称精确物理 XYZ，明确标注为规则化沿网距离+K 层号，且指明精确坐标存于 NPZ/VTS/静态 VTK —— **表述诚实、无误导**。
- 证据：`visualize_property_volume.py:341-356,632-664,792-805,876-894` · manifest `coordinate_contract` · reference PNG 轴显示真实 UTM（Easting 432179–437507 / Northing 6476498–6481426 / Elevation −depth）。

### ⑥ 是否符合石油领域正常三维属性建模展示流程 — ✅ 通过
- **无 scatter/点云**（关键修正 v1 缺陷）：精确解析三份 HTML 的 `Plotly.newPlot` 数据 —— 实际图 trace 为 reference `[volume, isosurface, volume]`、strict/conditional `[volume×3]`，**零 scatter/scatter3d**（HTML 中"scatter"字样属内嵌 plotly.js 库样板，非图 trace）；静态为 PyVista 体积光线投射+正交切片；源码无 `go.Scatter/Scatter3d`（测试断言）。
- **正常展示要素齐备**：全场参考体 + 区域 truth/prediction/residual 三联 + 正交切片 + 体渲染 + 等值面 + 交互；连续属性用感知渐变色表，残差用**对称发散色表**（`_scalar_limits(symmetric=True)`，L571）；面板 a/b/c、无标题（合项目 P5 图版规则）；`no_vertical_exaggeration=true`。
- **图像实测**：`reference_volume.png` 为真体渲染、真实 UTM 轴、colorbar "Reference porosity 0.001–0.292"；`strict/heldout_volume_comparison.png` 三联中 panel a(truth)非均质、panel b(prediction)近均匀、panel c(residual)对称发散 —— **忠实呈现、未美化掩盖模型缺陷**。
- 证据：newPlot trace 解析 · manifest `rendering` · 测试第 3/4 例 · 两张 PNG。

---

## 必须修改项（Must-fix）

1. **把模型技能指标与"无有效技能"结论绑定到每一个交付物**。
   - 现状：`R²=-0.34`（strict）仅在 `caption.md` 与 manifest；**交互 HTML 图注（代码 L889–894）缺 RMSE/R²**；PNG 单看 panel b 近均匀，易被当作成功孔隙度重建。
   - 要求：在 HTML 图注（`_render_regional_interactive` 的 caption）与 PNG 同级 caption 中显式写出 `RMSE/MAE/R²`，并加一句解释**"R²<0 表示预测不优于常数均值基线,当前区域重建无有效预测技能"**。
   - 验收：三处交付物（PNG 侧 caption、HTML note、manifest）指标一致且含"无有效技能"判读。

2. **在 strict/conditional 图注/manifest 显式点明"预测近乎常数"**（prediction `std≈0.0015`，接近全场均值 0.22）。
   - 现状：`prediction_statistics.std` 在 manifest 有值，但未转化为读者可懂的结论。
   - 要求：加一句"预测标准差≈0.0015,接近常数,空间非均质性基本未被重建",避免 panel b 被过度解读。

## 建议项（Suggestions）

1. **参考体渲染的 transfer function**：孔隙度集中在 ~0.22、`opacity="sigmoid"` 使全场偏均匀橙、内部非均质不易看清。建议按分位（如 P25–P75）设不透明度或直方图均衡,提升内部结构可读性（不改数据,仅显示）。
2. **深度语义标注**：manifest/caption 写明 Eclipse 深度是 TVDSS 还是模型深度(当前只写"grid depth, positive downward")；石油评审常追问深度基准。若无法确证则显式写"深度基准未标定"。
3. **浏览器规则化的可比性提示**：`Along-I/Along-J` 用中值网距,单元尺寸不均时会有几何轻微失真;建议在 note 里补一句"规则化用于显示,定量分析请用 NPZ/VTS 精确坐标"（现有披露已接近,补一句更稳）。
4. **reference 交互 stride (1,2,2)**：降采样已在 manifest 披露,但可在 HTML note 提示"浏览器为性能做了 J/K 抽稀,完整分辨率见 VTS"。

---

## 一句话结论

> **PASS。** 该三维属性体流水线在 Eclipse MAPAXES/PORO 物理语义、参考体/预测体分离、strict/conditional 证据边界、残差≠不确定性、静态精确坐标 vs 浏览器规则化坐标的诚实表述、以及无 scatter 的正常石油属性体展示流程上**全部通过且诚实度高**；唯一必须整改的是**把"模型预测近乎常数、R²<0、无有效重建技能"这一已知事实充分传递到 HTML 图注与 PNG caption**,防止 panel b 被误读为成功重建。模型技能本身属上游建模问题,不影响本可视化流水线的合规判定。

---
*本报告由 Gaia V2 石油专家视角对第六赛道三维属性体流水线独立终审生成；实际读取并核验了代码、测试、manifest、三份 caption、npz、PNG、HTML;仅新增本报告,未改动任何代码/数据/图片。*

---

## 复核闭环追加（2026-07-25 · Must-fix 闭环终审）

**闭环结论：✅ CLOSED。** 两条 must-fix 全部闭环；我先前的 3 条建议（深度基准 / 规则化仅供显示 / reference stride）亦一并落实。终审判定 **PASS 维持不变**。仅新增本追加节，未改动任何代码/数据/图片。

**实读文件（均于 21:25–21:26 重新生成，晚于原终审）**：`strict/caption.md`、`conditional/caption.md`、`reference/caption.md`、三份 `heldout_volume_comparison.html`/`reference_volume.html` 的 note、`artifact_manifest.json`、`_tests/test_visualize_property_volume.py`；并从原始 `heldout_reconstruction_volume.npz` 独立重算核对。

### 三处一致性（caption.md / HTML note / manifest.model_skill_interpretation）

| 指标 | strict | conditional | 三处一致 |
|---|---|---|---|
| RMSE | 0.035625 | 0.021013 | ✅ |
| MAE | 0.027173 | 0.015088 | ✅ |
| R² | -0.339679 | 0.013686 | ✅ |
| prediction_std | 0.001565 | 0.001414 | ✅ |
| truth_std | 0.030779 | 0.021158 | ✅ |
| pred/truth std 比值 | 5.1% | 6.7% | ✅ |
| practical_skill_verdict | no_practically_useful_spatial_reconstruction_skill | 同左 | ✅（manifest）|

### 独立重算（确认数字真实，非仅字符串一致）
- 从 npz 原始 `truth`/`prediction` 重算：strict `R²=-0.339679`、`pred_std=0.001565`、`truth_std=0.030779`；conditional `R²=0.013686`、`pred_std=0.001414`、`truth_std=0.021158`、比值 `6.7%` —— 与三处声明逐项吻合（diff<1e-4）。**声明的负 R²、近乎常数、6.7% 比值均为真实计算值。**

### Must-fix 1（指标 + 无技能结论传到所有交付物）—— ✅ CLOSED
- `caption.md`：新增 "Model-skill warning: prediction std=… versus truth std=…; the prediction is nearly constant. R²<0 means the prediction is worse than a constant-mean baseline; this regional reconstruction has no effective predictive skill."（strict）/ "…retains only 6.7% of the truth standard deviation; spatial heterogeneity is essentially not reconstructed."（conditional）。
- **HTML note**（原缺口）：现含 "模型技能警告：RMSE=…，MAE=…，R²=…；…R²<0，说明预测还不如常数均值基线；当前区域重建没有有效预测技能。" —— **原缺口已补齐**。
- `manifest`：新增 `modes.*.model_skill_interpretation`（rmse/mae/r2/prediction_std/truth_std/prediction_to_truth_std_ratio + `practical_skill_verdict` + 中英 `reader_warning`）。

### Must-fix 2（预测近乎常数）—— ✅ CLOSED
- 三处均显式对比 `prediction std` vs `truth std` 并写明"预测近乎常数 / 空间非均质性基本没有被重建"，且比值有数字（6.7%）。

### 先前建议闭环
- 深度基准：`reference/caption.md` + reference HTML 明写"Eclipse 深度基准未在 GRID 中独立标定，因此不冒充 TVDSS" ✅
- 规则化仅供显示：三份 HTML note 均加"规则化仅用于显示，定量分析请用 NPZ/VTS 精确坐标" ✅
- reference stride：reference HTML 加"浏览器为性能采用 K/J/I 步长 1/2/2，完整分辨率见 VTS" ✅

### 新增回归测试
- `_tests/test_visualize_property_volume.py::test_model_skill_warning_exposes_near_constant_predictions`（L68–82）锁定：`strict r2<0`、`"constant-mean baseline"` 在 warning、`conditional prediction_to_truth_std_ratio<0.1`、`"spatial heterogeneity is essentially not reconstructed"`、`practical_skill_verdict=="no_practically_useful_spatial_reconstruction_skill"`。
- **测试实跑绿证（2026-07-25 收尾补记）**：主控已在系统 Python 环境实跑完整目标测试 `pytest _tests/test_visualize_property_volume.py _tests/test_visualize_3d_sci.py -q` → **8 passed in 46.38s**；静态渲染另由 `fault-grid-techscan` 环境完成。此前复核时本机可用 env 缺 `pytest+pyvista`，我以从原始 npz 独立重算逐项确认断言值为真作为等价证据；现测试已实跑通过，该缺口闭合。

### 闭环判定
> **CLOSED。** 流水线现已把"模型无有效空间重建技能（R²<0 / 预测近乎常数 / 仅保留 5–7% truth 标准差）"这一已知事实，以**三处一致且数值真实**的方式传递到 caption.md、HTML note 与 manifest，并有新回归测试锁定；深度基准、规则化仅供显示、reference stride 三条建议均落实。原 **PASS** 维持。
