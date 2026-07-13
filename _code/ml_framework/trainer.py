"""Framework-neutral P4 trainer with valid-label weighted epochs and resume hooks."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .artifacts import atomic_write_json
from .reduction import WeightedReducer


@dataclass(frozen=True)
class StepResult:
    loss_sum: float
    valid_count: int
    metric_sums: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.loss_sum):
            raise ValueError("loss_sum must be finite")
        if self.valid_count <= 0:
            raise ValueError("valid_count must be >0")
        if not all(math.isfinite(value) for value in self.metric_sums.values()):
            raise ValueError("metric sums must be finite")


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int
    min_epochs: int = 1
    patience: int | None = None
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be >0")
        if not 1 <= self.min_epochs <= self.max_epochs:
            raise ValueError("min_epochs must be in [1, max_epochs]")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be >0 when enabled")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >=0")


@dataclass
class TrainerState:
    next_epoch: int = 0
    global_step: int = 0
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    stopped_early: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainerState":
        return cls(**dict(payload))


BatchFactory = Callable[[], Iterable[Any]]
StepFunction = Callable[[Any], StepResult]
CheckpointWriter = Callable[[TrainerState, Path], None]


def _run_epoch(step_fn: StepFunction, batches_fn: BatchFactory, split: str) -> tuple[float, int, dict[str, float], int]:
    loss = WeightedReducer()
    metric_sums: dict[str, float] = {}
    valid_count = 0
    batches = 0
    for batch in batches_fn():
        result = step_fn(batch)
        loss.update_sum(result.loss_sum, result.valid_count)
        valid_count += result.valid_count
        batches += 1
        for name, value_sum in result.metric_sums.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + float(value_sum)
    if batches == 0:
        raise RuntimeError(f"{split} batch factory returned zero batches")
    metrics = {name: value_sum / valid_count for name, value_sum in metric_sums.items()}
    return loss.mean, valid_count, metrics, batches


def train_with_validation(
    *,
    train_step: StepFunction,
    validation_step: StepFunction,
    train_batches_fn: BatchFactory,
    validation_batches_fn: BatchFactory,
    config: TrainerConfig,
    output_dir: Path,
    checkpoint_writer: CheckpointWriter,
    scheduler_step: Callable[[float], None] | None = None,
    resume_state: TrainerState | None = None,
) -> TrainerState:
    """Run development training; this API deliberately accepts no test loader."""
    output_dir.mkdir(parents=True, exist_ok=True)
    state = resume_state or TrainerState()
    if state.next_epoch > config.max_epochs:
        raise ValueError("resume state is beyond configured max_epochs")

    for epoch in range(state.next_epoch, config.max_epochs):
        train_loss, train_count, train_metrics, train_batches = _run_epoch(train_step, train_batches_fn, "train")
        val_loss, val_count, val_metrics, val_batches = _run_epoch(
            validation_step, validation_batches_fn, "validation"
        )
        if scheduler_step is not None:
            scheduler_step(val_loss)
        improved = val_loss < state.best_val_loss - config.min_delta
        if improved:
            state.best_val_loss = val_loss
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
        else:
            state.epochs_without_improvement += 1
        state.global_step += train_batches
        state.next_epoch = epoch + 1
        state.history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "train_valid_count": train_count,
                "validation_valid_count": val_count,
                "train_batches": train_batches,
                "validation_batches": val_batches,
                "train_metrics": train_metrics,
                "validation_metrics": val_metrics,
                "improved": improved,
            }
        )
        checkpoint_writer(state, output_dir / "checkpoint_last.pkl")
        if improved:
            checkpoint_writer(state, output_dir / "checkpoint_best.pkl")
        atomic_write_json(output_dir / "history.json", state.to_dict())

        can_stop = state.next_epoch >= config.min_epochs
        if config.patience is not None and can_stop and state.epochs_without_improvement >= config.patience:
            state.stopped_early = True
            checkpoint_writer(state, output_dir / "checkpoint_last.pkl")
            atomic_write_json(output_dir / "history.json", state.to_dict())
            break
    return state
