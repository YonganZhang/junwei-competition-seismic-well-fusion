# ml_framework — 6赛道共用训练骨架

> 2026-07-11 军伟拍板：先把pipeline骨架打好，模型/策略后续替换只改一个文件，不碰其他部分。

## 设计原则

- **换模型 = 加一个文件**：新模型文件导出`build_model()`，用`@register_model("名字")`注册，命令行`--model 名字`切换，训练循环完全不用改。
- **没策略的阶段先放骨架(no-op)，不是不写**：比如目前不去噪，`denoise_identity`是显式的identity函数+注释说明为什么（尖锐特征可能是真实地质信号，通用平滑会抹掉），不是"这个阶段压根没有函数"。以后要加策略，只替换这一个函数。
- **归一化必须配套反归一化**，用同一个`NormStats`对象保证可逆，可视化/评估阶段把模型输出换回物理量纲时统一调用，不让每个赛道各写一份容易写错符号的反变换。
- **训练循环强制记录train/val loss + 按最小val loss存best checkpoint**：不能只存最后一个epoch的权重——训练前期欠拟合（train/val loss都在降），后期过拟合（val loss回升），中间val loss最低点才是该用的权重。epoch数必须设得够多，覆盖到能看出这条"先降后升"曲线，不能太少看不到过拟合拐点就停。

## 模块

| 文件 | 职责 |
|---|---|
| `preprocess.py` | 去噪(当前no-op) / 归一化+反归一化(zscore/minmax) / 深度对齐(指向Layer1的`well_tie_weak.npz`，不重复实现) |
| `model_registry.py` | 模型注册表，`register_model`/`get_model` |
| `train.py` | 通用训练循环，`train_loop()`接收任意框架的step函数，记录train/val loss、按val loss存best checkpoint |
| `visualize.py` | `plot_loss_curve()`画train/val loss曲线，标出best epoch分界点 |

## 各赛道怎么接入

各赛道的`build_dataset.py`/`train_baseline.py`应该：
1. 预处理阶段调用`ml_framework.preprocess`里的函数，不要自己重复实现归一化/去噪逻辑
2. 模型定义放单独文件（如`models/unet.py`），用`@register_model`注册，训练脚本读`--model`参数选择
3. 训练循环调用`ml_framework.train.train_loop()`，不要自己写train/val循环
4. 训练完调用`ml_framework.visualize.plot_loss_curve()`产出loss曲线图，存到自己赛道的`_outputs/`下

## TODO（明确列为后续，不是本轮要做的）

- 接入更先进的预训练/基础模型：去开源社区（HuggingFace/GitHub）找"输入输出结构类似"的已包装好的大模型/智能体方案，包一层适配到这套接口里（`build_model()`签名不变，内部换成调用预训练模型）。
- 去噪、插值等策略的具体实现——目前故意留白（骨架阶段用简单策略/no-op），需要时把`denoise_identity`换成真实策略函数即可，不用碰训练循环。
- 结合项目里已有的深度学习训练相关skill（`share-autoresearch-engine`等）进一步打通。
