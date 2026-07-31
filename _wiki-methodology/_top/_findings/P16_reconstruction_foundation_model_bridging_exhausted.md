---
phase_id: P16
status: accepted
owner_col: COL2
---

# ⑥三维重建赛道：三种不同桥接方式全部验证大模型无增益，方法论层面已排除"桥接方式错误"这个解释

## Local Case

⑥三维模型重建赛道（Volve PHIF/孔隙度体重建，强baseline=PyKrige克里金）此前接入OpenMind-MAE（医学MRI预训练模型）门控残差融合被判定`VERIFIED_NO_PROMOTION`（域错配是主要怀疑根因，见P8/P9证据）。为排除"域错配"和"桥接方法本身有问题"这两个可能的解释，本轮（P14-P16）连续用**三种性质完全不同的桥接方式**接入`thinkonward/geophysical-foundation-model`（一个真正的地震领域专用ViT-MAE，450个Synthoseis仿真3D地震体上做trace-masking自监督预训练，Apache-2.0许可，已确认非医学模型），全部独立验证：

1. **P14 冻结特征提取**：GFM编码器完全冻结，抽embedding接Ridge回归头。结果：`pretrained_gated_rmse=0.028622` vs `random_init_gated_rmse=0.028601` vs `baseline=0.028450`——预训练权重和随机初始化几乎无法区分，两者都比baseline差。commit `fb8eea1`。
2. **P15 真实部分微调**：解冻GFM最后1-2个transformer block做真实微调（差异化学习率+early stopping+weight decay），对照组同样解冻但权重随机初始化。结果：`genuine_nonzero_gradient_signal=true`、`genuine_encoder_parameter_movement=true`（编码器确实产生了真实、非零、方向一致的梯度，训练动态与P14冻结版本完全不同）——但`pretrained_random_difference_ci_excludes_zero=false`，且`development_generalization_conclusion_different_from_p14=false`，最终仍是`VERIFIED_NO_PROMOTION`。commit `4a2e2b4`。
3. **P16 用GFM做本职工作（去噪/masked插值重建）而非直接预测物性**：把GFM原生的trace-masking去噪/重建能力用于增强地震特征，再喂给现有"no_foundation_structural"残差回归路线（而不是让GFM去预测它从未训练过的储层物性）。结果：`pretrained_gfm_reconstruction gated_rmse=0.028588`，`better_than_random_init=false`，`better_than_raw_structural=false`（既不比随机初始化更好，也不比完全不用GFM的原始地震特征更好），bootstrap置信区间跨零。commit `2998ddf`。

三轮结果的真实数字、随机初始化对照、gate=0退化检查均由主控独立复算/核实过（重新读取summary.json原始数字，非转述worker自报）。

## Class Pattern

当"接入大模型没有效果"这个结论出现时，容易被简化归因为单一原因（"域错配"或"桥接方法差"），但只有在系统性排除了同一模型的多种性质不同的接入方式（纯特征提取 / 真实梯度微调 / 用于其原生预训练任务而非目标任务）之后，"这个任务在当前数据规模下确实榨不出大模型的增量价值"这个结论才具有方法论上的说服力。三种桥接方式收敛到同一个否定结果，比任何单一方式的否定结果都更能排除"是我们方法用错了"这个反驳。

## Evidence

- P14/P15/P16均在worktree `.claude/worktrees/p10-results-reconstruction`（基于`p11-residual-reconstruction`分支），commit分别为`fb8eea1`、`4a2e2b4`、`2998ddf`，git diff --check均通过，工作树均clean，测试全部通过（P16汇报50 passed，含P11/P14/P15/P16全部定向测试）。
- 产物：`_pipelines/02_task_datasets/reconstruction/_outputs/{p14_geophysical_fm,p15_gfm_finetune,p16_gfm_denoise}/{summary.json,evidence.md}`。
- 模型来源：`~/.cache/huggingface/hub/models--thinkonward--geophysical-foundation-model`（1.4GB，Apache-2.0，已确认非gated滥用、真实域内预训练）；代码：`/mnt/data/yongan-admin-2/.cache/gfm_vendor_src/GFM/`。
- 三轮评测框架完全复用（强baseline=PyKrige OOF、gate=0精确退化断言、5个独立空间fold+3个种子伪重复的诚实区分、随机初始化架构对照），未被降低标准。
- **重要提醒**：此三轮工作全部发生在隔离worktree/branch里（未合并进主仓master，未push），本文件是唯一记录这一结论的地方；master的`_task_plan.md`当前phase编号（"P13科研深度扩展完成"）来自另一条不同步的分支线，与本finding的P11-P16编号不是同一套体系，交接时需注意这一点，不要混淆。

## Impact

- ⑥赛道"是否应该继续投入接入现成开源预训练大模型"这个问题，已经用三种独立方法给出了一致的否定答案，不建议再用同一批候选模型换第4种桥接方式反复尝试。
- 唯一还未验证、且有真实数据支撑的剩余选项：域内继续预训练（用项目已下载但仅解压1.1GB的ST0202地震体1.17TB原始数据，对GFM做无标签的continued self-supervised pretraining，再回到5个带标签patch微调）——这是一个按天/按GPU资源投入的决定，用户已知晓代价，尚未拍板是否投入。
- 若不投入继续预训练，建议的下一步是把资源转向传统地质统计学方法改进（协同克里金/带外部漂移克里金、变差函数各向异性重新拟合、加入物理约束），而不是继续在开源预训练模型的桥接方式上打转。

## Prevention Rule (candidate)

判定"大模型对某任务无增益"这类结论前，若第一次否定结果只测试了单一桥接方式（尤其是"冻结特征提取+浅层头"这种最省事的默认做法），应至少补测"真实微调"和"用于模型原生预训练任务而非改造成目标任务"这两种性质不同的桥接方式之一，再下方法论层面的最终判断；单一桥接方式的否定结果只能排除"这一种用法没用"，不能排除"这个模型/这类信息对这个任务没用"。

## Links

- task_plan: ../_task_plan.md（注意：本finding的P11-P16编号来自独立分支线，与task_plan当前编号体系暂不同步，见Evidence末尾提醒）
- 相关代码分支：`p11-residual-reconstruction`（worktree `.claude/worktrees/p10-results-reconstruction`），commits `fb8eea1`/`4a2e2b4`/`2998ddf`
