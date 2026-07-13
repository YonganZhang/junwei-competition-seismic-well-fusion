"""ml_framework — 六赛道共用的 P4 训练、验证和复现合同。

设计目标(军伟2026-07-11拍板)：以后换模型 = 换一个模型文件，不用碰pipeline其他部分。
每个阶段都是独立、可替换的模块，暂时没有策略的阶段先放骨架(no-op)，不是不写。

阶段划分：
    preprocess.py  — 去噪(denoise) / 归一化+反归一化(normalize/denormalize) / 深度对齐+时域转换
    model_registry.py — 模型注册与切换(--model unet_v2 这种命令行切换)
    train.py       — 训练循环:train/val loss监控、按最小val loss选最佳checkpoint、防欠拟合/过拟合
    visualize.py   — 训练曲线、混淆矩阵等可视化

公共层只定义外层合同与实验生命周期；数据语义、loss、metric、shape、
物理约束和赛道图仍由各赛道插件负责。
"""

from .contracts import ModelBatch, ModelOutput, TaskSpec
from .lifecycle import ExperimentLifecycle, ExperimentState
from .reduction import WeightedReducer
from .seeding import DEFAULT_ROOT_SEED, SeedTree, seed_everything
from .trainer import StepResult, TrainerConfig, TrainerState, train_with_validation

__all__ = [
    "DEFAULT_ROOT_SEED",
    "ExperimentLifecycle",
    "ExperimentState",
    "ModelBatch",
    "ModelOutput",
    "SeedTree",
    "StepResult",
    "TaskSpec",
    "TrainerConfig",
    "TrainerState",
    "WeightedReducer",
    "seed_everything",
    "train_with_validation",
]
