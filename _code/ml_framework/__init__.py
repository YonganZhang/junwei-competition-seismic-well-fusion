"""ml_framework — 6赛道共用的模块化训练骨架。

设计目标(军伟2026-07-11拍板)：以后换模型 = 换一个模型文件，不用碰pipeline其他部分。
每个阶段都是独立、可替换的模块，暂时没有策略的阶段先放骨架(no-op)，不是不写。

阶段划分：
    preprocess.py  — 去噪(denoise) / 归一化+反归一化(normalize/denormalize) / 深度对齐+时域转换
    model_registry.py — 模型注册与切换(--model unet_v2 这种命令行切换)
    train.py       — 训练循环:train/val loss监控、按最小val loss选最佳checkpoint、防欠拟合/过拟合
    visualize.py   — 训练曲线、混淆矩阵等可视化

各赛道的build_dataset.py/train_baseline.py应该改为调用这里的公共函数，
而不是各自重复实现预处理/训练循环。
"""
