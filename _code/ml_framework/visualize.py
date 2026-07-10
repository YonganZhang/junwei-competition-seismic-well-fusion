"""训练过程可视化 —— 每次训练都要产出这些图，不能只有最终数字。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import TrainHistory


def plot_loss_curve(history: TrainHistory, out_path: Path) -> Path:
    """画train/val loss曲线，标出best epoch(验证loss最低点，欠拟合/过拟合分界)。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history.train_loss) + 1)
    ax.plot(epochs, history.train_loss, label="train loss")
    ax.plot(epochs, history.val_loss, label="val loss")
    if history.best_epoch >= 0:
        ax.axvline(history.best_epoch + 1, color="gray", linestyle="--",
                   label=f"best epoch={history.best_epoch + 1}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
