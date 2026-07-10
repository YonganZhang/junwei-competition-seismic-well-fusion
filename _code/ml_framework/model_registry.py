"""模型注册与切换 —— 目标：换模型 = 加一个模型文件，不碰训练循环、不用改任何集中import。

约定：每个模型文件(如 unet.py / unet_v2.py)放在赛道自己的 models/ 目录下，必须导出一个
`build_model(**kwargs) -> torch.nn.Module` 函数，用 `@register_model("名字")` 装饰。

🔴 2026-07-11 修复(Codex审查发现)：旧版本只有"手动import过的模型文件才会被注册"，
不满足"加一个文件就行"的承诺——写了新模型文件但没人在什么地方import它，字典里就是空的。
现在 get_model() 按名字去 models_package 里动态 import 对应模块，注册在 import 时自动触发，
调用方不需要在任何地方手写 import 语句。
"""
from __future__ import annotations

import importlib
from typing import Callable

MODEL_REGISTRY: dict[str, Callable[..., object]] = {}


def register_model(name: str):
    """装饰器：给模型文件的build_model函数注册一个可在命令行切换的名字。

    用法(在赛道自己的 models/<name>.py 里)：
        from ml_framework.model_registry import register_model

        @register_model("unet")
        def build_model(**kwargs):
            return SmallUNet(**kwargs)

    🔴 重复注册报错，不静默覆盖 —— 同名模型文件写重了必须立刻暴露，不能悄悄用后写的那个。
    """
    def deco(fn):
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"模型名 {name!r} 重复注册。已注册于 {MODEL_REGISTRY[name].__module__}，"
                f"现在又在 {fn.__module__} 里注册——重命名其中一个，不允许静默覆盖。"
            )
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


def get_model(name: str, models_package: str | None = None, **kwargs):
    """按名字取模型。

    如果 name 还没注册且提供了 models_package(比如 "fault.models")，
    会尝试动态 import `{models_package}.{name}` 触发该模块里的 @register_model 装饰器执行，
    这样新增一个模型文件（文件名=注册名）就能直接被发现，不需要手动改任何集中 import 列表。
    """
    if name not in MODEL_REGISTRY and models_package:
        target_module = f"{models_package}.{name}"
        try:
            importlib.import_module(target_module)
        except ModuleNotFoundError as e:
            # 🔴 2026-07-11 二轮审查修复(Codex实测复现)：只有"目标模块本身就不存在"才吞掉走
            # 统一报错；如果是目标模块存在、但它内部import了别的缺失依赖导致的ModuleNotFoundError，
            # e.name不等于target_module，必须原样抛出，不能被误判成"模型未注册"掩盖真实缺依赖故障。
            if e.name != target_module:
                raise
            # 走到下面统一报错，给出已注册列表

    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"模型 {name!r} 未注册。已注册模型: {list(MODEL_REGISTRY.keys())}。"
            f"新模型：在 models/{name}.py 里写 build_model() 并用 @register_model('{name}') 注册，"
            f"文件名要和注册名一致，这样 get_model(models_package=...) 才能自动发现它。"
        )
    return MODEL_REGISTRY[name](**kwargs)
