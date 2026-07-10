"""模型注册与切换 —— 目标：换模型 = 加一个模型文件 + 改一行注册表，不碰训练循环。

约定：每个模型文件(如 unet.py / unet_v2.py)必须导出一个 `build_model(**kwargs) -> torch.nn.Module`
函数。注册后用命令行 `--model unet_v2` 切换，train.py 不需要知道具体是哪个模型。

放哪里：赛道自己的模型文件放在各自 _pipelines/02_task_datasets/<track>/models/ 下，
在这里(或赛道自己的一个小型registry.py里复用MODEL_REGISTRY这个字典模式)注册。
"""
from __future__ import annotations

from typing import Callable

MODEL_REGISTRY: dict[str, Callable[..., object]] = {}


def register_model(name: str):
    """装饰器：给模型文件的build_model函数注册一个可在命令行切换的名字。

    用法(在赛道自己的模型文件里)：
        from ml_framework.model_registry import register_model

        @register_model("unet")
        def build_model(**kwargs):
            return SmallUNet(**kwargs)
    """
    def deco(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"模型 {name!r} 未注册。已注册模型: {list(MODEL_REGISTRY.keys())}。"
            f"新模型：写一个新文件导出 build_model()，用 @register_model('新名字') 注册，"
            f"import一下这个文件即可，不用改训练循环。"
        )
    return MODEL_REGISTRY[name](**kwargs)
