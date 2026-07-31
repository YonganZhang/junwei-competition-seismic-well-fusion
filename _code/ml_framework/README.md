# ml_framework — 六赛道 P4 训练、验证与复现框架

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
| `train.py` | P3 简单基线兼容入口；已修复为按样本/有效标签加权，但 P4 新实现使用 `trainer.py` |
| `visualize.py` | `plot_loss_curve()`画train/val loss曲线，标出best epoch分界点 |

## 各赛道怎么接入

各赛道的`build_dataset.py`/`train_baseline.py`应该：
1. 预处理阶段调用`ml_framework.preprocess`里的函数，不要自己重复实现归一化/去噪逻辑
2. 模型定义放单独文件（如`models/unet.py`），用`@register_model`注册，训练脚本读`--model`参数选择
3. P4 训练调用 `ml_framework.trainer.train_with_validation()`；旧 baseline 可继续用已加权的 `train_loop()`
4. 训练完调用`ml_framework.visualize.plot_loss_curve()`产出loss曲线图，存到自己赛道的`_outputs/`下

## P4 公共合同

| 文件 | 职责 |
|---|---|
| `contracts.py` | `TaskSpec`、`ModelBatch`、`ModelOutput` 外层合同及输入泄漏检查 |
| `seeding.py` | 默认 `root_seed=2693`、稳定角色 seed 派生及 Python/NumPy/Torch 确定性报告 |
| `splits.py` | 先冻结 test，再按不可泄漏 group 建 folds；不足 5 折时显式降级 |
| `cv.py` | development-only CV 和 OOF 一次覆盖；API 不接收 test |
| `trainer.py` | 按样本/有效标签数加权的 train/validation、早停和 resume state |
| `checkpoint.py` | model/optimizer/scheduler/scaler/epoch/RNG/config/split 的完整 checkpoint |
| `lifecycle.py` | `DRAFT → ... → TEST_CONSUMED → VERIFIED` 单向状态机和一次性 test 防火墙 |
| `hpo.py` | 无 Optuna 的固定 baseline + 可选顺序 TPE，目标只读 development folds |
| `artifacts.py` | 原子 JSON、SHA-256 和可复核 artifact manifest |
| `model_discovery.py` | 动态发现 `_models/<track>/<model_id>.py`，禁止集中手工 import |

### 最短公共测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s _code/ml_framework/tests -p 'test_*.py' -v
```

赛道插件只能在各自 TaskSpec 中声明 loss、输出 transform、metric、CV buffer、HPO 搜索空间和图件。模型文件不负责 split、test、预处理拟合或画图。

## 后续扩展

- 接入更先进的预训练/基础模型：去开源社区（HuggingFace/GitHub）找"输入输出结构类似"的已包装好的大模型/智能体方案，包一层适配到这套接口里（`build_model()`签名不变，内部换成调用预训练模型）。
- 去噪、插值等策略的具体实现——目前故意留白（骨架阶段用简单策略/no-op），需要时把`denoise_identity`换成真实策略函数即可，不用碰训练循环。
- 结合项目里已有的深度学习训练相关skill（`share-autoresearch-engine`等）进一步打通。
