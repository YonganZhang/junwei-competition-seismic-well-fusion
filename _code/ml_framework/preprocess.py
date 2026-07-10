"""预处理骨架：去噪 / 归一化+反归一化 / 深度对齐+时域转换。

原则(军伟2026-07-11)：没有策略的阶段先放骨架(no-op)，不是不写这个阶段。
比如"暂时不去噪"要显式写成一个identity函数+注释说明为什么，
不能什么函数都没有——这样以后要加策略时，只改这一个函数，训练循环完全不用动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# 去噪 (denoise)
# ---------------------------------------------------------------------------

def denoise_identity(x: np.ndarray) -> np.ndarray:
    """当前策略：不去噪，原样返回。

    Why: 地震振幅/测井曲线里的"尖锐"特征(比如薄互层反射、异常读数)本身可能是
    真实地质信号，通用平滑/去噪很容易把这些真实异常一并抹掉。目前没有针对具体
    噪声类型的可靠策略，宁可不做，也不做通用平滑滤波糊弄。
    需要去噪策略时，实现一个新函数(如 denoise_bandpass)替换这里的调用，
    训练/评估流程不用改。
    """
    return x


DENOISE_FN = denoise_identity  # 换策略只改这一行指向的函数


# ---------------------------------------------------------------------------
# 归一化 / 反归一化 (normalize / denormalize) — 必须配对，可逆
# ---------------------------------------------------------------------------

@dataclass
class NormStats:
    """归一化参数，随样本/数据集一起持久化，保证反归一化时能精确还原。

    🔴 只能在训练集上fit（fit_zscore/fit_minmax传训练数据），不能在验证/测试集上重新fit，
    否则等于让模型提前看到测试集分布，是数据泄漏。"""
    method: str  # "zscore" | "minmax"
    mean: float | None = None
    std: float | None = None
    vmin: float | None = None
    vmax: float | None = None

    def __post_init__(self):
        if self.method == "zscore" and (self.mean is None or self.std is None):
            raise ValueError("method='zscore' 必须提供 mean 和 std，用 fit_zscore() 生成，不要手填")
        if self.method == "minmax" and (self.vmin is None or self.vmax is None):
            raise ValueError("method='minmax' 必须提供 vmin 和 vmax，用 fit_minmax() 生成，不要手填")
        if self.method not in ("zscore", "minmax"):
            raise ValueError(f"未知归一化方法: {self.method!r}，只支持 zscore/minmax")

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "mean": self.mean, "std": self.std,
                "vmin": self.vmin, "vmax": self.vmax}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "NormStats":
        return NormStats(**d)


def fit_zscore(x: np.ndarray) -> NormStats:
    return NormStats(method="zscore", mean=float(np.mean(x)), std=float(np.std(x) + 1e-8))


def fit_minmax(x: np.ndarray) -> NormStats:
    return NormStats(method="minmax", vmin=float(np.min(x)), vmax=float(np.max(x)))


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    if stats.method == "zscore":
        return (x - stats.mean) / stats.std
    if stats.method == "minmax":
        span = (stats.vmax - stats.vmin) or 1e-8
        return (x - stats.vmin) / span
    raise ValueError(f"未知归一化方法: {stats.method}")


def denormalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """反归一化 —— 可视化/评估阶段把模型输出/输入换回物理量纲时必须用这个，
    不能让每个赛道各写一份容易写错符号的反变换。"""
    if stats.method == "zscore":
        return x * stats.std + stats.mean
    if stats.method == "minmax":
        span = (stats.vmax - stats.vmin) or 1e-8
        return x * span + stats.vmin
    raise ValueError(f"未知归一化方法: {stats.method}")


# ---------------------------------------------------------------------------
# 深度对齐 / 时域转换 (depth alignment / time-depth conversion)
# ---------------------------------------------------------------------------
# 实际实现已经在 _pipelines/01_common_preprocess/step_04_well_tie_weak.py 里做了
# (弱井震标定：分段线性插值+外推，MD->TWT->最近inline/crossline)。
# 这里只放一个骨架级的接口占位，指向真实实现，避免每个赛道各写一份深度转换逻辑。

def depth_to_time_placeholder(*args, **kwargs):
    """深度->时间转换的真实实现见 _pipelines/01_common_preprocess/step_04_well_tie_weak.py。
    这里不重复实现，只作为文档指针；各赛道直接复用Layer1产出的 well_tie_weak.npz，
    不要各自重新写一份深度对齐逻辑。"""
    raise NotImplementedError(
        "请直接读取 _pipelines/01_common_preprocess/outputs/well_tie_weak.npz，"
        "不要在这里重新实现深度对齐"
    )
