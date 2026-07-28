"""Chronos-2 adapter for the sweetspot T3 production forecast.

The adapter deliberately exposes only causal history.  It does not accept
future covariates or labels, and it converts the 30-day oil forecast into the
scalar target defined by the existing T3 contract.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from _code.ml_framework.contracts import TaskSpec
from _models.gaia_dagt.foundation_runtime import consume_config, verify_checkpoint


model_id = "p7_chronos2"
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


def capabilities() -> dict[str, Any]:
    return {
        "task_types": ["regression"],
        "input_modalities": ["regular_calendar_time_series"],
        "input_shape": "[B,7,30] plus [B,30] daily timestamps",
        "output_shape": "[B,30] daily forecast plus [B] mean",
        "foundation_model": MODEL_ID,
        "conditioning": "time_window",
        "supports_missing_mask": True,
        "supports_uncertainty": True,
        "requires_pretrained_weight": True,
        "auto_download": False,
    }


def build_model(task_spec: TaskSpec, **config: Any) -> Any:
    if task_spec.track_id != "sweetspot":
        raise ValueError("Chronos-2 adapter is restricted to the sweetspot track")
    values = consume_config(
        config,
        required=("snapshot_path",),
        optional=("device",),
    )
    snapshot = Path(values["snapshot_path"]).resolve()
    verify_checkpoint("sweetspot", snapshot / "model.safetensors")
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError("Chronos-2 local snapshot is missing config.json")
    return load_pipeline(snapshot, device=str(values.get("device", "cuda")))


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

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
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


def build_calendar_frame(
    sequences: np.ndarray,
    timestamps: np.ndarray,
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    """Build the regular long-format DataFrame required by Chronos-2.

    This is the only approved path for a claim expressed in calendar days.
    The legacy array API remains available for reproducing the archived
    observation-index experiment, but must not be described as daily.
    """
    array = validate_sequences(sequences)
    time = np.asarray(timestamps)
    if time.shape != (array.shape[0], T3_HISTORY_LENGTH):
        raise ValueError(
            "T3 calendar timestamps must have shape "
            f"({array.shape[0]}, {T3_HISTORY_LENGTH})"
        )
    if len(sample_ids) != array.shape[0] or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("T3 calendar sample_ids must be unique and match the batch")
    records: list[dict[str, Any]] = []
    for batch_index, sample_id in enumerate(sample_ids):
        item_time = pd.DatetimeIndex(pd.to_datetime(time[batch_index]))
        expected = pd.date_range(item_time[0], periods=T3_HISTORY_LENGTH, freq="D")
        if not item_time.equals(expected):
            raise ValueError("T3 calendar timestamps must be gap-free daily values")
        for step_index, timestamp in enumerate(item_time):
            record = {
                "item_id": str(sample_id),
                "timestamp": timestamp,
                "target": float(array[batch_index, 0, step_index]),
            }
            for channel_index, name in enumerate(PAST_COVARIATE_NAMES, start=1):
                record[name] = float(array[batch_index, channel_index, step_index])
            records.append(record)
    return pd.DataFrame.from_records(records).sort_values(
        ["item_id", "timestamp"], kind="stable"
    ).reset_index(drop=True)


def forecast_oil_calendar(
    pipeline: Any,
    sequences: np.ndarray,
    timestamps: np.ndarray,
    sample_ids: Sequence[str],
    *,
    batch_size: int = 196,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast exactly 30 calendar days with explicit timestamps/frequency."""
    frame = build_calendar_frame(sequences, timestamps, sample_ids)
    forecast = pipeline.predict_df(
        frame,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=PREDICTION_LENGTH,
        quantile_levels=[MEDIAN_QUANTILE],
        batch_size=int(batch_size),
        context_length=T3_HISTORY_LENGTH,
        cross_learning=False,
        validate_inputs=True,
    )
    required = {"item_id", "timestamp", "predictions"}
    if not required <= set(forecast.columns):
        raise ValueError(f"Chronos calendar forecast is missing columns: {sorted(required - set(forecast.columns))}")
    by_item = {str(item_id): group for item_id, group in forecast.groupby("item_id", sort=False)}
    daily: list[np.ndarray] = []
    for sample_id in sample_ids:
        if str(sample_id) not in by_item:
            raise ValueError(f"Chronos calendar forecast is missing item: {sample_id}")
        group = by_item[str(sample_id)].sort_values("timestamp")
        if len(group) != PREDICTION_LENGTH:
            raise ValueError("Chronos calendar forecast length mismatch")
        values = np.maximum(group["predictions"].to_numpy(dtype=np.float64), 0.0)
        if not np.isfinite(values).all():
            raise ValueError("Chronos calendar forecast contains non-finite values")
        daily.append(values)
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
