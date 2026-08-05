# P5 六赛道真实三维成像与 SCI Plot 合同

> 状态：2026-07-25 完成并独立验收。  
> 目标：把六赛道中**确实具有空间坐标、三维体或井轨迹**的数据做成可复现、可审计的三维领域可视化；不把二维结果伪装成三维体。

## 共同边界

1. 只读取各赛道已经存在的真实数据、真实模型预测和冻结证据；本轮不重训、不改模型、不改标签、不改 split。
2. 每赛道先输出 `three_d_feasibility.json`，明确：
   - `native_volume`：有真实三维网格/体素；
   - `spatial_context`：只有带真实 X/Y/Z 或井轨迹的点、曲线；
   - `not_feasible`：没有可验证的第三维或空间顺序。
3. 禁止把重复二维切片、任意 z-offset、随机插值、示意网格或合成点云冒充三维结果。
4. 若为了性能下采样，必须记录原始形状、显示形状、采样规则和是否影响定量结论。
5. 每赛道写域仅限自己的 `visualize_3d_sci.py`、对应测试及 `_outputs/3d_sci_v1/`；不得覆盖旧图和用户已有脏改动。
6. Worker 不负责公开发布；主控独立验收后统一走卡片渲染 Pipeline。

## 统一交付物

具备 `native_volume` 或 `spatial_context` 的赛道必须交付：

- `visualize_3d_sci.py`：可从登记的真实输入重复生成全部图；
- `three_d_feasibility.json`：维度、坐标、来源与判定；
- `provenance.json`：输入路径、输入 SHA-256、代码 HEAD、模型/数据边界、采样规则、限制；
- 至少一张 300 DPI PNG 和一份 PDF；复杂图优先拆成单图，不强拼大面板；
- 一个自包含或依赖最少的交互式 HTML，可旋转、缩放、查看坐标/数值；
- `caption.md`：只写图注与科学边界，不把解释性结论塞进图内；
- 至少一个自动测试，覆盖真实输入存在、维度合同、禁止 dummy/fallback 和输出完整性。

若为 `not_feasible`，必须交付可执行的缺口说明和缺少字段清单；不得生成假三维图。

## SCI Plot 硬规则

- 静态图必须使用 A–E 标准图幅之一；复杂三维单图默认 `D_full`（7.2 × 7.2 in）。
- 所有英文文字使用 Times New Roman；保存前调用 `normalize_fonts(fig)`。
- 禁止 `plt.title()`、`ax.set_title()`、`fig.suptitle()`；唯一允许的面板标记是粗体小写 `a/b/c`。
- 图内不写结论、指标解读或长说明；放入 `caption.md`。
- 默认使用阿昆浮世绘色板；连续量使用感知均匀顺序色图，残差使用以 0 为中心的发散色图。
- 禁止 `jet`、`rainbow`、红绿二元对比；颜色条必须有变量名和单位。
- 坐标轴必须使用真实含义与单位（如 Inline、Crossline、TWT/ms、UTM/m、TVDSS/m）。
- Plotly HTML 同样使用 Times New Roman、白底、无标题，并保持与静态图一致的变量颜色。
- 按 `share-sci-plot` 要求通过 `~/.codex/scripts/gemini-ask.sh` 做一次代码/视觉复核；若外援额度或服务不可用，原样记录，禁止伪造已复核。

## 六赛道预期（以实际审计为准）

1. **断层检测**：优先真实地震子体 + 真值断层体素/面 + 预测概率等值面；只能展示真实测试 patch 或可证明连续的体。
2. **地震相识别**：有连续切片索引时可组成真实相体；若只有独立二维样本，则判 `not_feasible`，不得堆片造体。
3. **储层物性**：优先真实井轨迹/空间点按 PHIF、KLOGH、SW 着色；没有经验证的空间插值合同，不生成连续属性体。
4. **岩相预测**：优先真实井轨迹上的九类岩相真值/预测三维条带；没有真实 X/Y/Z 时不得生成体。
5. **甜点区评价**：优先带真实空间/深度坐标的目标值或概率点云/井轨迹；没有网格预测时不插值成甜点体。
6. **三维重构**：优先真实 PORO 参考、重构、残差的三维体/等值面；conditional 与 strict 的科学边界必须分开。

## 独立验收

- `python -m py_compile` 与赛道测试通过；
- 绘图脚本无 title 调用、无 dummy/fallback、无未登记随机数造数据；
- 主控逐张打开 PNG 检查裁切、文字、颜色、遮挡和真实三维表达；
- 主控读取 HTML 验证有真实三轴、可交互且不引用本机绝对路径；
- 主控复算输入/输出 SHA，并核对 `provenance.json`；
- 只有通过以上检查的成果才能进入公共卡片渲染。

## 完成记录

| 赛道 | 判定 | 真实三维依据与边界 | 交付与验收 | 本地 commit |
|---|---|---|---|---|
| ① 断层 | `spatial_context` | 三个真实测试 patch 带 Inline、Crossline、TWT；不连续，未插值成体 | 3 张 2160×2160/300 DPI PNG+PDF、1 个交互 HTML；unittest 1/1 PASS | `track-fault@9077ab8` |
| ② 地震相 | `not_feasible` | 独立二维 patch 无连续网格、空间顺序或轨迹 | 审计三件套；unittest 1/1 PASS；无伪 3D 图 | `track-facies@beff5b6` |
| ③ 储层物性 | `spatial_context` | 真实 Inline、Crossline、TWT 采样点与 PHIF/KLOGH/SW 预测；不是井轨迹或属性体 | 3 张 2160×2160/300 DPI PNG+PDF、1 个交互 HTML；pytest 2/2 PASS | `track-property@6b5b7cf` |
| ④ 岩相 | `not_feasible` | 120 条记录的 `center_md_m` 均非有限值，仅有 TWT，无 XYZ/轨迹 | 审计三件套；unittest 2/2 PASS；无伪 3D 图 | `p5-3d-lithofacies-audit@b7c2a93` |
| ⑤ 甜点 | `not_feasible` | T1/T2 仅 depth，T3/T4 仅 cutoff_date，T5–T7 仅状态；无 XYZ/网格/轨迹 | 审计三件套；unittest 1/1 PASS；无伪 3D 图 | `p5-r2-sweetspot@d008a62` |
| ⑥ 三维重构 | `native_volume` | 原生 Eclipse GRID/INIT 提供 MAPAXES、角点和 PORO；全场参考体与 strict/conditional 区域 truth/prediction/residual 分离，精确 Easting/Northing/depth 坐标写入 NPZ/VTS/静态 VTK | 全场参考体、正交切片、2 组区域三联体及 3 个交互 HTML；新旧 pytest 8/8 PASS；Gaia V2 石油专家 PASS/CLOSED | `p5-r2-reconstruction-v2@7e35b45` |

共同 QA：

- 12 张静态图由主控逐张打开检查，并通过 Gemini SCI 视觉二审，证据：
  `_reports/_foreign_aid/20260725T175925__gemini__3102320/result.md`。
- 4 个交互 HTML 均在 Chromium 中加载真实 Plotly trace；拖拽后相机参数发生变化，控制台 0 error。
- 所有 16 个永久卡片渲染 URL 经发布脚本自验并再次返回 HTTP 200。
- 本机未安装 Times New Roman；静态图按合同显式记录限制并确定性降级为 Liberation Serif，
  没有静默替换或安装系统字体。

### 公共卡片渲染

① 断层：

- seismic: `https://share.yongan.site/junwei-fault-3d-sci/seismic_spatial_context.20260725-180458.png`
- truth: `https://share.yongan.site/junwei-fault-3d-sci/truth_spatial_context.20260725-180459.png`
- probability: `https://share.yongan.site/junwei-fault-3d-sci/probability_spatial_context.20260725-180459.png`
- interactive: `https://share.yongan.site/junwei-fault-3d-sci/spatial_context.20260725-180500.html`

③ 储层物性：

- PHIF: `https://share.yongan.site/junwei-property-3d-sci/phif_spatial_context.20260725-180458.png`
- KLOGH: `https://share.yongan.site/junwei-property-3d-sci/klogh_spatial_context.20260725-180458.png`
- SW: `https://share.yongan.site/junwei-property-3d-sci/sw_spatial_context.20260725-180459.png`
- interactive: `https://share.yongan.site/junwei-property-3d-sci/spatial_context.20260725-180500.html`

⑥ 三维重构：

- strict truth/prediction/residual:
  `https://share.yongan.site/junwei-reconstruction-3d-sci/truth.20260725-180458.png`,
  `https://share.yongan.site/junwei-reconstruction-3d-sci/reconstruction.20260725-180458.png`,
  `https://share.yongan.site/junwei-reconstruction-3d-sci/residual.20260725-180459.png`
- strict interactive:
  `https://share.yongan.site/junwei-reconstruction-3d-sci/prediction_comparison.20260725-180500.html`
- conditional truth/prediction/residual:
  `https://share.yongan.site/junwei-reconstruction-3d-sci/truth.20260725-180501.png`,
  `https://share.yongan.site/junwei-reconstruction-3d-sci/reconstruction.20260725-180502.png`,
  `https://share.yongan.site/junwei-reconstruction-3d-sci/residual.20260725-180503.png`
- conditional interactive:
  `https://share.yongan.site/junwei-reconstruction-3d-sci/prediction_comparison.20260725-180504.html`

⑥ 三维重构 v2 物理属性体（取代仅 K/J/I 点云式展示）：

- 全场 PORO 参考属性体：
  `https://share.yongan.site/volve-3d-reference/reference_volume.20260725-213059.html`
- 全场正交切片：
  `https://share.yongan.site/volve-3d-reference-slices/reference_orthogonal_slices.20260725-213127.png`
- strict 区域 truth/prediction/residual：
  `https://share.yongan.site/volve-3d-strict/heldout_volume_comparison.20260725-213059.html`
- conditional 区域 truth/prediction/residual：
  `https://share.yongan.site/volve-3d-conditional/heldout_volume_comparison.20260725-213059.html`

v2 科学边界：

- 浏览器视图因 Plotly 限制使用规则化 Along-I/Along-J/K；定量分析必须使用 NPZ/VTS 中的精确 MAPAXES 坐标。
- strict `R²=-0.339679`，conditional `R²=0.013686`；两者 prediction std 仅为 truth std 的约 5–7%，当前模型没有实用的空间非均质重建技能。
- residual 是 prediction − truth 的误差体；没有 ensemble/posterior，故不声明不确定性体。
- 独立石油流程终审：
  `_external_reviews/GAIA_V2_CLAUDE_PETROLEUM_WORKFLOW_REVIEW_20260725.md`。
