"""通用训练循环 —— 6赛道共用，跑什么模型/什么数据都调这一个函数。

核心纪律(军伟2026-07-11口述要求)：
- 每个epoch都要看train loss和val loss。
- train loss应该持续下降；val loss应该先降后升——先降是欠拟合区，后升进入过拟合区，
  两者之间val loss最低的那个点就是最佳权重，不能只看最后一个epoch的权重。
- epoch数必须设得够多，覆盖到能看出"先降后升"这个完整曲线为止，不能epoch数太少
  看不到过拟合拐点就提前停。
- 训练曲线必须画出来存盘(visualize.py)，不能只有最终数字没有过程曲线。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }


def train_loop(
    model,
    train_step_fn: Callable[[Any], float],
    val_step_fn: Callable[[Any], float],
    train_batches: Iterable,
    val_batches: Iterable,
    epochs: int,
    save_checkpoint_fn: Callable[[Any, Path], None],
    checkpoint_dir: Path,
    min_epochs_before_early_check: int = 10,
) -> TrainHistory:
    """通用训练循环骨架。

    参数说明（故意做成"传函数"而不是"传具体框架对象"，这样PyTorch/其他框架都能用）：
        train_step_fn(batch) -> float：跑一个训练batch，返回loss（副作用里做反向传播）
        val_step_fn(batch) -> float：跑一个验证batch，返回loss（不能有梯度更新）
        save_checkpoint_fn(model, path)：保存权重的具体实现（不同框架不同，调用方传入）

    行为：
        - 每个epoch结束记录train/val平均loss
        - val loss创新低就存一份"best"checkpoint（不是只存最后一个epoch的权重）
        - epochs数量由调用方决定，此函数只负责如实记录，不擅自提前停
          （早停策略如果要加，后续在这里加，目前先如实跑完全部epoch让人自己看拐点，
          这是骨架阶段的保守选择——避免早停参数没调好反而错过真实最优点）
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = TrainHistory()

    for epoch in range(epochs):
        train_losses = [train_step_fn(b) for b in train_batches]
        val_losses = [val_step_fn(b) for b in val_batches]

        epoch_train_loss = sum(train_losses) / max(len(train_losses), 1)
        epoch_val_loss = sum(val_losses) / max(len(val_losses), 1)

        history.train_loss.append(epoch_train_loss)
        history.val_loss.append(epoch_val_loss)

        if epoch_val_loss < history.best_val_loss:
            history.best_val_loss = epoch_val_loss
            history.best_epoch = epoch
            save_checkpoint_fn(model, checkpoint_dir / "best.ckpt")

        save_checkpoint_fn(model, checkpoint_dir / "last.ckpt")

    (checkpoint_dir / "history.json").write_text(
        json.dumps(history.to_dict(), ensure_ascii=False, indent=2)
    )

    if epochs < min_epochs_before_early_check:
        # 只是提醒，不阻断：epoch数太少可能看不到val loss的拐点
        pass

    return history
