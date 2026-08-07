---
phase_id: P44
status: accepted
severity: critical
owner_col: COL4
source: audit
created_at: 2026-08-07
---

# 甜点七目标里有四个的标签是 CPI 解释产物，T6/T7 的 `is_proxy=False` 标注与证据不符

## Local Case

为评估"是否重做 T6/T7 的 split"（P43 遗留决策），本轮在 development 井族上用可重建的
井-深度样本做了一次只读基线探测：特征取 P5 现用的 16 条原始曲线 `RAW_LOG_FEATURES`，
标签取 `PHIF` 与 `log1p(KLOGH)`，按 well_family 做 LOGO 交叉验证（3 个井族，35810 样本）。

结果表面上很好，HistGradientBoosting 在两个目标上都大幅优于 P4 的 OOF：

| | P4 OOF（153 特征 / 4 井族 / 1216 样本）| 本次 LOGO-OOF（16 特征 / 3 井族 / 35810 样本）|
|---|---|---|
| T6 `PHIF` | MAE `0.017833`，R² `0.85997` | MAE `0.005550`，R² `0.970948` |
| T7 `KLOGH` (log1p) | MAE `0.835387`，R² `0.829915` | MAE `0.386481`，R² `0.931273` |
| T7 `KLOGH` (物理 mD) | MAE `197.127`，R² `0.565046` | MAE `111.580`，R² `0.702950` |

随后做的特征消融推翻了这个"提升"的意义：

| 特征子集 | LOGO-OOF MAE | R² |
|---|---|---|
| **仅 `RHOB`** | `0.009284` | **`0.969637`** |
| 仅 `NPHI` | `0.066999` | `-0.037223` |
| `RHOB`+`NPHI` | `0.005872` | `0.964646` |
| `RHOB`+`NPHI`+`GR` | `0.006135` | `0.964741` |
| 全部 16 条 | `0.005550` | `0.970948` |

**单独一条体积密度曲线就能达到 R² `0.9696`，全部 16 条曲线只把 R² 抬到 `0.9709`。**
`RHOB` 在样本中缺失率为 0。这与"CPI 的 `PHIF` 是由体积密度经确定性孔隙度关系导出"完全一致：
模型学到的主要不是地质规律，而是一个解析式。`KLOGH` 在测井解释中通常又是 `PHIF` 的函数，
其高分具有同源性质。

## Class Pattern

这是 P43 在 T2 上记录的同一类问题的扩大版：**标签是解释产物，特征是该解释产物的输入，
于是模型在逆向拟合一条确定性规则，指标越高越说明规则被复现得越彻底，而不是预测能力越强。**

按标签来源重排七个目标：

| 目标 | 标签来源 | 性质 | 现有 `is_proxy` 标注 |
|---|---|---|---|
| T1 `RQI` | `0.0314·sqrt(KLOGH/PHIF)` | CPI 解释产物的组合 | `True` ✅ 一致 |
| T2 `SAND_FLAG` | 解释人员截断规则 | CPI 解释产物 | `True` ✅ 一致 |
| T3 产能 | 生产日报 | **实测** | 不适用 |
| T4 见水 | 生产时序 | **实测** | 不适用 |
| T5 剩余油 | 数值模拟 | 仿真（本轮决策：接受为标签）| 不适用 |
| T6 `PHIF` | 体积密度的确定性函数 | CPI 解释产物 | **`False` ❌ 与证据不符** |
| T7 `KLOGH` | `PHIF` 的函数 | CPI 解释产物 | **`False` ❌ 与证据不符** |

T1/T2 老实标了 `is_proxy=True` 并带 warning；T6/T7 标了 `is_proxy=False`、
`p4_status=complete`，但四者同源。标注不一致本身比单个指标虚高更危险，因为下游按标注决定
一个数字能不能对外引用。

## Evidence

- 探测脚本与结果：`_sandbox/t6t7_probe/probe.py`、`_sandbox/t6t7_probe/result.json`
  （sandbox，未纳入版本控制；结论已抄录于本文件）
- 特征定义：`p5/sweetspot_p5_stage2_data.py:34-40`（`RAW_LOG_FEATURES` 含 `RHOB`/`NPHI`；
  `LABEL_ONLY_FIELDS` 含 `PHIF`/`KLOGH`）
- 标注来源：`p5/sweetspot_p5_label_mapping.v1.json`（`targets.T6.is_proxy=False`、
  `targets.T7.is_proxy=False`）
- P4 对照：`targets/porosity/_outputs/phif/oof/metrics.json`、
  `targets/permeability/_outputs/klogh/oof/metrics.json`
- 探测为 development-only，未打开任何 frozen test 产物。

## Impact

- **重做 T6/T7 的 split 已失去原本设想的价值。** P43 认为重做能把它们送进 P5 多模型与智能体
  优化流程；现在看，在这两个目标上比较模型强弱意义有限——任何模型只要拿到 `RHOB` 就能到
  R² 0.97，优化空间本身是伪的。
- **不应在 T6/T7 上投入智能体优化。** 这与 P28/P29 在 T3 上被拒是两种不同的"没有收益"：
  T3 是信号被噪声淹没，T6/T7 是任务本身没有待学的未知量。
- **七目标中具备真实预测意义的只有 T3、T4 和（本轮新接受的）T5。** 这三个的标签分别来自生产
  日报、生产时序和数值模拟，不是同批测井曲线的解释产物。甜点赛道后续的模型与智能体投入应集中在此。
- 任何写入对外材料的 T1/T2/T6/T7 指标，都必须同时说明标签是 CPI 解释产物。特别是 T6 的
  P4 frozen-test `R²=0.93411`，单独引用等价于宣称"孔隙度预测接近完美"。

## Prevention Rule (candidate)

新增目标进入排行榜前，应强制跑一次"最小特征子集消融"：若某个单一输入特征即可达到接近全特征的
指标，则该目标默认标为解释产物复现，`is_proxy` 置 `True`，并在排行榜渲染时带出。
`is_proxy` 不应由目标登记时人工填写而不加验证。

## Links

- task_plan: ../_task_plan.md
- 前序: P43_sweetspot_seven_target_gate_root_cause.md（T2 的同类问题、T6/T7 样本身份根因）
- 七目标合同: ../_phases/P4_sweetspot_seven_target_contract.md
