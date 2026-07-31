# 011: P23 Checkshot 目标向井震标定

status: L3_validated_keep_calibration | created: 2026-08-01 | updated: 2026-08-01

## Hypothesis

现有三维重建把 Eclipse 的 TVD-like 储层深度直接代入三口井的
MD–TWT 弱映射，在斜井中会把目标储层映射到错误地震时窗。Volve 本地
VSP 包实际包含五口井的 checkshot。若使用真实 TVD/TVDSS–TWT 曲线代替
MD 弱映射，应显著降低目标深度范围的时深误差。

## Frozen protocol

- 拟合井固定为当前流水线已用的 `15/9-19 A`、`15/9-19 BT2`、
  `15/9-19 SR`。
- `15/9-F-11 T2` 与 `15/9-F-15 A` 的 checkshot 完全不参与拟合，只作
  独立标定验证。
- 评价深度冻结为 Eclipse 活跃储层范围 `2800.7185–3543.7759 m`。
- 对照是当前三井 MD–TWT 弱标定的水平反距离加权；候选只把时深轴
  改为 checkshot TVDSS/TVD–TWT，不使用孔隙度标签。
- 主指标是两口独立井合并 TWT MAE，同时报告逐井 MAE/RMSE 与等效 4 ms
  地震样点误差。
- 该步不打开 reconstruction `train.h5`/`test.h5`，不评价孔隙度指标。

## Acceptance

候选必须同时降低 F11T2 和 F15A 的 TWT MAE，且合并 MAE 严格低于弱标定。
通过后才允许重采样地震属性并复测 P21 OOF；这两口井仍只评价标定，不作
孔隙度盲测声称。

## Implementation

正式可复跑入口：
`_pipelines/02_task_datasets/reconstruction/p23_checkshot_calibration.py`。
机读结果：
`_pipelines/02_task_datasets/reconstruction/_outputs/p23_checkshot_calibration/summary.json`。
回归测试：
`_pipelines/02_task_datasets/reconstruction/_tests/test_p23_checkshot_calibration.py`。

## Result

两口独立井共 80 个目标储层 checkshot 点上，当前三井 MD–TWT 弱标定的
合并 MAE/RMSE 为 `633.1867/633.5426 ms`，等效 MAE 为 158.30 个 4 ms
地震样点。三井 checkshot 候选的合并 MAE/RMSE 为 `8.7389/11.1878 ms`，
等效 MAE 为 2.18 个样点。

- F11T2：MAE 从 `635.4387 ms` 降至 `9.6871 ms`。
- F15A：MAE 从 `631.0446 ms` 降至 `7.8370 ms`。

两口井的 checkshot 均未参与候选拟合；候选仅使用 19A/19BT2/19SR。该步
未打开 reconstruction HDF5，未使用孔隙度标签或 holdout。

## Verdict + Reason

KEEP AS CALIBRATION。两口独立井均改善，合并 MAE 降低 624.45 ms。结果
直接支持“当前 MD 弱标定把 TVD-like 储层映射到了错误时窗”这一根因。
该结论允许触发 012 重采样地震属性，但不等于孔隙度模型已在新盲测上提升。

## Sub-explorations triggered

012：在不改变 P21 模型、五折、512 标签预算和指标的前提下，用 checkshot
对齐后的地震属性和真实预训练 GFM 特征复测 OOF。
