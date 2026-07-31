# 甜点目标7：水平渗透率

唯一label version为官方解释曲线`KLOGH`，物理单位为mD。训练目标严格为`log1p(KLOGH)`；冻结测试时先反标准化，再以`expm1`回到mD。`KLOGH_NEW`、KLOGV和其他派生解释曲线均不作为标签或输入。

## 当前状态

- `target7-klogh-cpi-v1`：`complete`。
- 冻结test：母井家族`15/9-F-15`，344样本；development为其余4个母井家族，requested5→effective4。
- 简单baseline：动态注册的`reservoir_ridge`；真实ST0202地震patch + GR/RT/NPHI/RHOB原始测井序列与显式mask。
- 物理mD指标：MAE=226.476、RMSE=544.599、R²=0.577088。
- log1p诊断：MAE=0.706676、RMSE=0.864602、R²=0.868730。
- HPO目标是development OOF物理mD MAE，方向`minimize`；本轮未运行Optuna或长HPO。

`_outputs/klogh/`包含独立TaskSpec、label availability、split manifest、4折checkpoint与OOF、冻结配置、全development refit checkpoint、单次test指标/预测、四张专属图、状态和artifact manifest。

## 便携测试与真实复跑

从项目根运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider \
  _pipelines/02_task_datasets/reservoir/tests

python3 _pipelines/02_task_datasets/sweetspot/targets/permeability/run.py --help
python3 _pipelines/02_task_datasets/sweetspot/targets/permeability/run.py \
  --mode baseline \
  --processed-dir /path/to/read-only/reservoir \
  --guard-path /path/to/read-only/guard.npz
```

运行器只读消费已有真实HDF5/guard，不会把命令中的绝对路径写进运行证据。
