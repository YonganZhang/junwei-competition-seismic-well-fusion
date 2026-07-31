# 目标7 P4复现SOP

1. 先按reservoir README完成真实数据build；本目标只读消费已有HDF5和guard。
2. `run.py --mode audit`只依据KLOGH有效标签登记独立split。优先F-15；fallback只能在建模前按label availability确定并记录。
3. `--mode smoke`验证真实多模态batch、显式输入mask、单输出模型和test未消费。
4. `--mode baseline`依次执行fold-train统计、4折LOGO/OOF、配置冻结、全development refit、checkpoint封存和一次性test。
5. KLOGH训练域是`log1p(mD)`；反演为`mD=expm1(clamp(log_prediction,0))`。主报告使用物理mD MAE/RMSE/R²，log1p指标只作诊断。
6. 检查`split_manifest.json`、`lifecycle.json`、`oof/`、`refit/checkpoint_best.pkl`、`frozen_test/`、四张图和`manifest.json`。

HPO方向固定为`minimize development OOF physical_MAE`，不得把test传入HPO。当前baseline不执行长HPO。
