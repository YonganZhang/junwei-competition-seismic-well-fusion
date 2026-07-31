# 甜点目标6：孔隙度

主label version固定为官方解释曲线`PHIF`，物理单位为fraction。`PHIE`只允许作为独立case；不与PHIF拼接、不把`LFP_PHIE`改名为PHIE，也不共享split、指标或checkpoint。

## 当前状态

- `PHIF / target6-phif-cpi-v1`：`complete`。冻结test为母井家族`15/9-F-15`，development为其余4个家族；requested5诚实降级为effective4。
- 简单baseline：动态注册的`reservoir_ridge`，输入为真实ST0202 `3×3×9`地震patch与GR/RT/NPHI/RHOB九点原始测井序列和显式观测mask。
- 冻结test 344样本：MAE=0.0120557、RMSE=0.0172235、R²=0.934115。它是新增P4 ridge结果，不替换旧reservoir tiny_mlp三输出指标。
- `PHIE / target6-exact-phie-v1`：`not_feasible`。33个相关LAS实扫未发现精确PHIE；`LFP_PHIE`只覆盖`15/9-19`一个母井家族且明确排除。

`_outputs/phif/`含TaskSpec、label availability、独立split、fold checkpoint/OOF、冻结配置、refit checkpoint、物理指标、预测CSV、生命周期防火墙、artifact manifest和四张图。`_outputs/phie/`含独立TaskSpec、覆盖证据和`not_feasible`状态。

## 便携测试与真实复跑

从项目根运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  _pipelines/02_task_datasets/reservoir/tests

python3 _pipelines/02_task_datasets/sweetspot/targets/porosity/run.py --help
python3 _pipelines/02_task_datasets/sweetspot/targets/porosity/run.py \
  --mode baseline \
  --processed-dir /path/to/read-only/reservoir \
  --guard-path /path/to/read-only/guard.npz \
  --well-log-zip /path/to/read-only/Volve_Well_logs.zip
```

`--processed-dir`只需包含既有`train.h5`和`test.h5`。运行器不复制原始ZIP/SEG-Y、不重建HDF5，也不会把命令中的机器路径写进manifest。
