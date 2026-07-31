"""Reducers that weight by samples or valid labels, never by batch count."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class WeightedReducer:
    weighted_sum: float = 0.0
    weight_sum: float = 0.0
    updates: int = 0

    def update_mean(self, mean_value: float, weight: float) -> None:
        if not math.isfinite(mean_value):
            raise ValueError(f"mean_value must be finite, got {mean_value}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"weight must be finite and >0, got {weight}")
        self.weighted_sum += float(mean_value) * float(weight)
        self.weight_sum += float(weight)
        self.updates += 1

    def update_sum(self, value_sum: float, count: float) -> None:
        if not math.isfinite(value_sum):
            raise ValueError(f"value_sum must be finite, got {value_sum}")
        if not math.isfinite(count) or count <= 0:
            raise ValueError(f"count must be finite and >0, got {count}")
        self.weighted_sum += float(value_sum)
        self.weight_sum += float(count)
        self.updates += 1

    @property
    def mean(self) -> float:
        if self.weight_sum <= 0:
            raise RuntimeError("reducer has no valid samples/labels")
        return self.weighted_sum / self.weight_sum

    def to_dict(self) -> dict[str, float | int]:
        return {
            "weighted_sum": self.weighted_sum,
            "weight_sum": self.weight_sum,
            "updates": self.updates,
            "mean": self.mean,
            "averaging": "sample_or_valid_label_weighted",
        }


def weighted_mean(pairs: Iterable[tuple[float, float]]) -> float:
    reducer = WeightedReducer()
    for value, weight in pairs:
        reducer.update_mean(value, weight)
    return reducer.mean
