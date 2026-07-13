# 目标6 P4复现SOP

1. 先按reservoir README完成真实数据build；目标6只读消费`train.h5`、`test.h5`和`guard.npz`。
2. 运行`run.py --mode audit`。先登记每个母井家族的label availability，再冻结F-15；若F-15不足，fallback只能按已登记标签数确定，不能查看特征、模型、HPO或指标。
3. 运行`--mode smoke`检查多模态batch、mask、单输出动态模型和test未消费状态。
4. 运行`--mode baseline`：每折只用fold-train拟合输入与目标统计；完成4折LOGO/OOF后冻结配置，用全部development refit一次，再通过生命周期防火墙读取一次test。
5. 审核`split_manifest.json`、`lifecycle.json`、`oof/metrics.json`、`refit/checkpoint_best.pkl`、`frozen_test/metrics.json`、四张专属图及`manifest.json`。
6. PHIE只审核`_outputs/phie/`。若状态为`not_feasible`，不得改用PHIF或LFP_PHIE补足；需等待新的精确PHIE标签版本。

HPO合同方向为`minimize development OOF physical_MAE`。本SOP的baseline步骤不运行Optuna；未来HPO仍不得接收test loader。
