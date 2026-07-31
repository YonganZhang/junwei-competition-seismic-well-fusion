---
phase_id: P21
status: accepted
severity: major
owner_col: COL2
source: runtime
created_at: 2026-08-01
---

# 赛道⑥ 改变 PEFT 学习信号仍未形成空间可迁移残差，固定基础模型核更稳

## Local Case

P21 在 P20 的真实预训练 GFM、原生 SEG-Y 窗口和 rank-4 LoRA 基础上，先用两种
地震属性遮挡视图做 16 步无标签一致性适配，再以 512 标签内部五折 P19 同构基准
的样本外残差为监督。神经头、单次校准 Ridge 和完整五折 Ridge 的 OOF RMSE 分别
为 `0.028278976997`、`0.029618418227` 和 `0.028080039761`，均未超过 P19 的
`0.027751397628`。

与此同时，去掉 P19 的逐折元选择、在所有外折统一平均三个固定 foundation 核，
得到 `0.027734374378`。该路线与 P19 四折等价，一折改善，零折退化；因此作为
`ACCEPTED_SIMPLICITY_WIN` 启用，但不把约 0.0613% 的相对改善解释为广泛统计效应。

## Class Pattern

真实梯度、参数移动和自监督损失下降只证明适配链路打通。小样本空间任务中，内层
残差均值和局部纹理可能不具有跨空间块迁移性；校准集越小，越容易把局部偏置当成
基础模型贡献。允许残差路线精确退化为零，以及把“更简单且不退化”纳入晋级规则，
比继续扩大 PEFT 容量更可靠。

## Evidence

- `_pipelines/02_task_datasets/reconstruction/_outputs/p21_fixed_foundation_ensemble/summary.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p21_fixed_foundation_ensemble/verification.json`
- `_pipelines/02_task_datasets/reconstruction/_outputs/p21_fixed_foundation_ensemble/predictions.npz`
- `_wiki-methodology/_tests/P21_reconstruction_fixed_ensemble_acceptance.md`

## Impact

赛道⑥当前默认候选从 P19 的逐折元选择改为 P21 固定 foundation 三核平均。大模型
仍通过真实预训练特征参与邻域度量，但 P21 不声称 LoRA 残差产生了正增益。后续若
增加大模型训练，应先增加独立地质约束或合法监督信号，再重新做严格外折验证。

## Prevention Rule (candidate)

空间小样本残差校准必须使用完整内层空间交叉拟合，并设置可精确退化为零的门；
若残差路线不能跨外折迁移，不得用非零梯度或训练损失下降替代泛化证据。

## Links

- method: ../../_wiki/_methods/explorations/006-p21-reconstruction-contrastive-residual-and-fixed-ensemble.md
