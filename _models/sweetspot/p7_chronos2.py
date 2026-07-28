"""Chronos-2 adapter for the sweetspot T3 production forecast.

The adapter deliberately exposes only causal history.  It does not accept
future covariates or labels, and it converts the 30-day oil forecast into the
scalar target defined by the existing T3 contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
MODEL_LICENSE = "Apache-2.0"
PREDICTION_LENGTH = 30
T3_HISTORY_LENGTH = 30
T4_HISTORY_LENGTH = 7
MEDIAN_QUANTILE = 0.5
INPUT_COLUMNS = (
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
)
PAST_COVARIATE_NAMES = tuple(column.lower() for column in INPUT_COLUMNS[1:])
WATER_TARGET_INDEX = 2
WATER_PAST_COVARIATE_INDICES = (0, 1, 3, 4, 5, 6)


def validate_sequences(sequences: np.ndarray) -> np.ndarray:
    array = np.asarray(sequences, dtype=np.float32)
    expected_tail = (len(INPUT_COLUMNS), T3_HISTORY_LENGTH)
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
        raise ValueError(
            f"T3 Chronos input must have shape (samples, {expected_tail[0]}, "
            f"{expected_tail[1]}), got {array.shape}"
        )
    if np.isinf(array).any():
        raise ValueError("T3 Chronos input contains an infinite history value")
    if np.isnan(array[:, 0, :]).all(axis=1).any():
        raise ValueError("T3 Chronos input contains an all-missing oil history")
    return array


def build_inputs(
    sequences: np.ndarray,
    *,
    mode: str = "past_covariates",
) -> list[np.ndarray | dict[str, Any]]:
    """Build Chronos inputs without any post-cutoff values."""
    array = validate_sequences(sequences)
    if mode == "univariate":
        return [row[0].copy() for row in array]
    if mode == "multivariate_target":
        return [row.copy() for row in array]
    if mode == "past_covariates":
        return [
            {
                "target": row[0].copy(),
                "past_covariates": {
                    name: row[index].copy()
                    for index, name in enumerate(PAST_COVARIATE_NAMES, start=1)
                },
            }
            for row in array
        ]
    raise ValueError(f"unsupported Chronos input mode: {mode}")


def build_water_risk_inputs(sequences: np.ndarray) -> list[dict[str, Any]]:
    """Build T4 inputs from its causal seven-day history only."""
    array = np.asarray(sequences, dtype=np.float32)
    expected_tail = (len(INPUT_COLUMNS), T4_HISTORY_LENGTH)
    if array.ndim != 3 or tuple(array.shape[1:]) != expected_tail:
        raise ValueError(
            f"T4 Chronos input must have shape (samples, {expected_tail[0]}, "
            f"{expected_tail[1]}), got {array.shape}"
        )
    if np.isinf(array).any():
        raise ValueError("T4 Chronos input contains an infinite history value")
    if np.isnan(array[:, WATER_TARGET_INDEX, :]).all(axis=1).any():
        raise ValueError("T4 Chronos input contains an all-missing water history")
    return [
        {
            "target": row[WATER_TARGET_INDEX].copy(),
            "past_covariates": {
                INPUT_COLUMNS[index].lower(): row[index].copy()
                for index in WATER_PAST_COVARIATE_INDICES
            },
        }
        for row in array
    ]


def load_pipeline(
    snapshot_path: Path,
    *,
    device: str = "cuda",
) -> Any:
    """Load the source-locked model lazily so contract tests stay CPU-only."""
    import torch
    from chronos import Chronos2Pipeline

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    return Chronos2Pipeline.from_pretrained(
        str(Path(snapshot_path).resolve()),
        device_map=device,
        torch_dtype=dtype,
    )


def _median_index(pipeline: Any) -> int:
    quantiles = np.asarray(pipeline.quantiles, dtype=float)
    matches = np.flatnonzero(np.isclose(quantiles, MEDIAN_QUANTILE))
    if matches.size != 1:
        raise ValueError("Chronos pipeline must expose exactly one median quantile")
    return int(matches[0])


def forecast_oil(
    pipeline: Any,
    sequences: np.ndarray,
    *,
    mode: str = "past_covariates",
    batch_size: int = 196,
) -> tuple[np.ndarray, np.ndarray]:
    """Return non-negative daily median forecasts and their 30-day means."""
    inputs = build_inputs(sequences, mode=mode)
    outputs: Sequence[Any] = pipeline.predict(
        inputs,
        prediction_length=PREDICTION_LENGTH,
        batch_size=int(batch_size),
        cross_learning=False,
    )
    if len(outputs) != len(inputs):
        raise ValueError("Chronos returned a different number of forecast items")
    median_index = _median_index(pipeline)
    daily: list[np.ndarray] = []
    for output in outputs:
        values = output.detach().float().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
        if values.ndim != 3 or values.shape[2] != PREDICTION_LENGTH:
            raise ValueError(f"unexpected Chronos forecast shape: {values.shape}")
        oil = np.asarray(values[0, median_index], dtype=np.float64)
        daily.append(np.maximum(oil, 0.0))
    daily_array = np.asarray(daily, dtype=np.float64)
    return daily_array, daily_array.mean(axis=1)


def forecast_water_risk_scores(
    pipeline: Any,
    sequences: np.ndarray,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Score T4 by the largest forecast seven-day mean at each quantile."""
    inputs = build_water_risk_inputs(sequences)
    outputs: Sequence[Any] = pipeline.predict(
        inputs,
        prediction_length=PREDICTION_LENGTH,
        batch_size=int(batch_size),
        cross_learning=False,
    )
    if len(outputs) != len(inputs):
        raise ValueError("Chronos returned a different number of T4 forecast items")
    all_scores: list[np.ndarray] = []
    for output in outputs:
        values = output.detach().float().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != PREDICTION_LENGTH:
            raise ValueError(f"unexpected Chronos T4 forecast shape: {values.shape}")
        non_negative = np.maximum(np.asarray(values[0], dtype=np.float64), 0.0)
        window_means = np.stack(
            [
                non_negative[:, start : start + T4_HISTORY_LENGTH].mean(axis=1)
                for start in range(PREDICTION_LENGTH - T4_HISTORY_LENGTH + 1)
            ],
            axis=1,
        )
        all_scores.append(window_means.max(axis=1))
    return np.asarray(all_scores, dtype=np.float64), np.asarray(pipeline.quantiles, dtype=np.float64)
