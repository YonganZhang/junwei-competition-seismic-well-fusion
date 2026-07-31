# P20 Reconstruction PEFT Acceptance Evidence

## Scope

P20 在赛道⑥固定五折/512 标签协议下复测非零初始化、LoRA、Adapter 与分阶段
解冻。没有打开 `test.h5` 或冻结 holdout，基础模型随机初始化消融继续按用户要求
后置。

## Recomputed result

- PyKrige RMSE: `0.028449728170183285`。
- P19 accepted RMSE: `0.027751397627827728`。
- nonzero-head: `0.027814382567190387`。
- LoRA r4: `0.027789635296497726`。
- staged Adapter: `0.027814441628425134`。
- staged LoRA r4: `0.027789615699745370`。
- 80-update staged LoRA: `0.027791517166415700`。
- P19/P20 staged-LoRA OOF error correlation: `0.9992036727355113`。
- Fixed P19/P20 blend grid optimum: P20 weight `0.0`。

## Optimization evidence

- prefix batch: `[8, 3, 161, 1200]`；query width: `7`。
- LoRA r4 trainable low-rank parameters: `76,800`。
- final GFM block base parameters: `17,295,600`；terminal norm: `2,400`。
- LoRA refit: 160/160 PEFT-gradient steps nonzero。
- staged Adapter refit: 120 PEFT and 80 terminal-norm steps nonzero。
- staged LoRA refit: 120 PEFT、80 terminal-norm、40 full-tail steps nonzero。
- 五折 LoRA/Adapter/terminal norm/full-tail parameter update L2 均按预期大于零。

## Verification

独立验证脚本重新读取各路线 NPZ，逐项重算 pooled/fold RMSE、MAE、bias、梯度计数、
P19 对齐、误差相关与固定融合网格。状态为 `PASSED`；决策为
`VERIFIED_NO_PROMOTION`，`default_enabled=false`。

复跑命令：

```bash
/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python3 \
  _pipelines/02_task_datasets/reconstruction/p20_peft_verification.py \
  --p19-predictions _tmp/p19_meta_purged_repro/meta_purge_predictions.npz \
  --extended-80-summary \
    _sandbox/p20_peft_staged_unfreeze/outputs_80/summary.json
```
