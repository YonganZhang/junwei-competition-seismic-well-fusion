"""Fault-track spatial split helpers shared by build and training code."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidationPlan:
    val_start_inline: int
    guard_start_inline: int
    guard_sampled_inlines: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "fit_rule": f"inline < {self.guard_start_inline}",
            "guard_range": [self.guard_start_inline, self.val_start_inline - 1],
            "guard_sampled_inlines": list(self.guard_sampled_inlines),
            "validation_rule": f"inline >= {self.val_start_inline}",
            "validation_start_inline": self.val_start_inline,
        }


def validation_masks(
    inlines: np.ndarray, val_fraction: float, guard_inlines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ValidationPlan]:
    values = np.asarray(inlines, dtype=np.int32)
    unique_inlines = np.unique(values)
    if len(unique_inlines) < 5:
        raise ValueError("not enough unique train inlines for a spatial validation split")
    if not 0.05 <= val_fraction <= 0.4:
        raise ValueError("val-fraction must be between 0.05 and 0.4")
    if guard_inlines < 1:
        raise ValueError("val-guard-inlines must be >= 1")
    val_start_index = max(1, int(np.floor(len(unique_inlines) * (1.0 - val_fraction))))
    val_start_index = min(val_start_index, len(unique_inlines) - 1)
    val_start_inline = int(unique_inlines[val_start_index])
    guard_start_inline = val_start_inline - guard_inlines
    fit_mask = values < guard_start_inline
    guard_mask = (values >= guard_start_inline) & (values < val_start_inline)
    validation_mask = values >= val_start_inline
    if not fit_mask.any() or not validation_mask.any():
        raise ValueError("validation split leaves an empty fit or validation subset")
    if np.any(fit_mask & guard_mask) or np.any(fit_mask & validation_mask) or np.any(guard_mask & validation_mask):
        raise AssertionError("fit/guard/validation masks overlap")
    plan = ValidationPlan(
        val_start_inline=val_start_inline,
        guard_start_inline=guard_start_inline,
        guard_sampled_inlines=tuple(sorted(set(int(value) for value in values[guard_mask]))),
    )
    return fit_mask, guard_mask, validation_mask, plan
