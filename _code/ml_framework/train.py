"""通用训练循环 —— 6赛道共用，跑什么模型/什么数据都调这一个函数。

核心纪律(军伟2026-07-11口述要求)：
- 每个epoch都要看train loss和val loss。
- train loss应该持续下降；val loss应该先降后升——先降是欠拟合区，后升进入过拟合区，
  两者之间val loss最低的那个点就是最佳权重，不能只看最后一个epoch的权重。
- epoch数必须设得够多，覆盖到能看出"先降后升"这个完整曲线为止，不能epoch数太少
  看不到过拟合拐点就提前停。
- 训练曲线必须画出来存盘(visualize.py)，不能只有最终数字没有过程曲线。

🔴 2026-07-11 修复(Codex审查实测复现的严重bug)：旧版本 train_batches/val_batches 接收
一次性可迭代对象(比如生成器)，第2个epoch起对象被耗尽，for循环立刻结束，
sum([])/max(0,1) 把空批次静默算成 0.0 ——不报错，还被误判成"最佳checkpoint"存下来，
足以污染六条赛道的实验结论。复现结果：train loss变成[3.0, 0.0, 0.0]，第2个epoch被误存为best。
现在改成：调用方传"每次调用返回新一批数据的工厂函数"(train_batches_fn/val_batches_fn)，
且任何一个epoch如果batch数为0或loss不是有限数字，立即fail-loud报错，不允许静默按0处理。
"""
from __future__ import annotations

import json
import math
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


def _run_epoch(step_fn: Callable[[Any], float], batches_fn: Callable[[], Iterable], split_name: str) -> float:
    """跑一个epoch的一个split(train或val)，fail-loud：0个batch或非有限loss都直接报错。"""
    losses = [step_fn(b) for b in batches_fn()]
    if not losses:
        raise RuntimeError(
            f"{split_name} 这个epoch拿到0个batch —— 这在旧版本会被静默算成loss=0.0并可能误判为最佳"
            f"checkpoint，现在直接报错。检查 {split_name}_batches_fn() 是不是传了一次性生成器"
            f"（生成器用完就空了，必须传能重复调用、每次都产出完整一轮数据的工厂函数）。"
        )
    epoch_loss = sum(losses) / len(losses)
    if not math.isfinite(epoch_loss):
        raise RuntimeError(f"{split_name} loss出现非有限值({epoch_loss})，train_step_fn/val_step_fn返回了NaN/Inf。")
    return epoch_loss


def train_loop(
    model,
    train_step_fn: Callable[[Any], float],
    val_step_fn: Callable[[Any], float],
    train_batches_fn: Callable[[], Iterable],
    val_batches_fn: Callable[[], Iterable],
    epochs: int,
    save_checkpoint_fn: Callable[[Any, Path], None],
    checkpoint_dir: Path,
    min_epochs_before_early_check: int = 10,
) -> TrainHistory:
    """通用训练循环骨架。

    参数说明（故意做成"传函数"而不是"传具体框架对象"，这样PyTorch/其他框架都能用）：
        train_step_fn(batch) -> float：跑一个训练batch，返回loss（副作用里做反向传播，
            必须返回脱离计算图的python float，PyTorch用户记得调用.item()，
            否则计算图整个epoch都不释放，在共享GPU紧张显存下会OOM）
        val_step_fn(batch) -> float：跑一个验证batch，返回loss（不能有梯度更新）
        train_batches_fn() -> Iterable：🔴每次调用必须返回一批"完整一轮"的新数据，
            不能传"一次性用完的对象本身"——每个epoch都会重新调用这个函数取数据。
            对DataLoader来说，直接传 `lambda: my_dataloader` 就对（DataLoader本身可重复iter）。
        val_batches_fn() -> Iterable：同上，验证集版本
        save_checkpoint_fn(model, path)：保存权重的具体实现（不同框架不同，调用方传入）

    行为：
        - 每个epoch结束记录train/val平均loss，任何一个epoch数据为空或loss非有限直接报错(fail-loud)
        - val loss创新低就存一份"best"checkpoint（不是只存最后一个epoch的权重）
        - history.json 每个epoch结束就增量写盘（不是等全部epoch跑完才写）——共享GPU服务器
          随时可能被抢占/OOM/kill，中途崩溃也要保留已完成epoch的loss曲线，不能全丢
        - epochs数量由调用方决定，此函数只负责如实记录，不擅自提前停
          （早停策略如果要加，后续在这里加，目前先如实跑完全部epoch让人自己看拐点，
          这是骨架阶段的保守选择——避免早停参数没调好反而错过真实最优点）
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = TrainHistory()
    history_path = checkpoint_dir / "history.json"

    if epochs < min_epochs_before_early_check:
        print(
            f"⚠️ epochs={epochs} 小于建议值{min_epochs_before_early_check}，"
            f"可能看不全val loss先降后升的完整曲线，找不到真正的过拟合拐点。"
        )

    try:
        for epoch in range(epochs):
            epoch_train_loss = _run_epoch(train_step_fn, train_batches_fn, "train")
            epoch_val_loss = _run_epoch(val_step_fn, val_batches_fn, "val")

            history.train_loss.append(epoch_train_loss)
            history.val_loss.append(epoch_val_loss)

            if epoch_val_loss < history.best_val_loss:
                history.best_val_loss = epoch_val_loss
                history.best_epoch = epoch
                save_checkpoint_fn(model, checkpoint_dir / "best.ckpt")

            save_checkpoint_fn(model, checkpoint_dir / "last.ckpt")
            history_path.write_text(json.dumps(history.to_dict(), ensure_ascii=False, indent=2))
    finally:
        # 即使中途抛异常(比如共享GPU被抢占)，已完成epoch的曲线也已经逐epoch落盘，不会全丢
        pass

    return history
