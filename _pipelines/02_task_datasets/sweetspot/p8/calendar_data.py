"""Exact daily-grid T3 data builder.

P5/P7 selected 30 observations and tolerated spans up to 45 days.  This
module deliberately re-freezes the development sample universe on a regular
daily grid; it never pretends that the archived observation windows were
calendar windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


data_module = importlib.import_module(
    "_pipelines.02_task_datasets.sweetspot.p5.sweetspot_p5_stage2_data"
)

ROOT_SEED = 2693
HISTORY_DAYS = 30
FORECAST_DAYS = 30
STEP_DAYS = 7
MIN_OBSERVED_FUTURE_DAYS = 21
VOLUME_COLUMNS = (
    "BORE_OIL_VOL",
    "BORE_GAS_VOL",
    "BORE_WAT_VOL",
    "ON_STREAM_HRS",
)
STATE_COLUMNS = (
    "AVG_DOWNHOLE_PRESSURE",
    "AVG_CHOKE_SIZE_P",
    "AVG_WHP_P",
)
SEQUENCE_COLUMNS = VOLUME_COLUMNS[:3] + (VOLUME_COLUMNS[3],) + STATE_COLUMNS


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CalendarPilotData:
    fold_id: int
    train_sample_ids: tuple[str, ...]
    validation_sample_ids: tuple[str, ...]
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_sequence: np.ndarray
    validation_sequence: np.ndarray
    train_timestamps: np.ndarray
    validation_timestamps: np.ndarray
    train_target: np.ndarray
    validation_target: np.ndarray
    split_sha256: str
    input_budget_sha256: str
    provenance: Mapping[str, Any]


def calendarize_well(well: pd.DataFrame) -> pd.DataFrame:
    required = {"WELL_BORE_CODE", "DATEPRD", *SEQUENCE_COLUMNS}
    if not required <= set(well.columns):
        raise ValueError(f"production frame is missing: {sorted(required - set(well.columns))}")
    frame = well.loc[:, sorted(required)].copy()
    frame["DATEPRD"] = pd.to_datetime(frame["DATEPRD"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["DATEPRD"])
    if frame.empty:
        raise ValueError("production well has no valid dates")
    groups: dict[str, str] = {column: "sum" for column in VOLUME_COLUMNS}
    groups.update({column: "mean" for column in STATE_COLUMNS})
    daily = frame.groupby("DATEPRD", sort=True).agg(groups)
    index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(index)
    daily.index.name = "DATEPRD"
    daily["WELL_BORE_CODE"] = str(well["WELL_BORE_CODE"].iloc[0])
    return daily.reset_index()


def _history_sequence(history: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [history[column].to_numpy(dtype=np.float64) for column in SEQUENCE_COLUMNS]
    )


def build_calendar_productivity(frame: pd.DataFrame) -> dict[str, Any]:
    sample_ids: list[str] = []
    groups: list[str] = []
    sequences: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    targets: list[float] = []
    for group, raw_well in frame.groupby("WELL_BORE_CODE", sort=True):
        well = calendarize_well(raw_well)
        for index in range(HISTORY_DAYS, len(well) - FORECAST_DAYS + 1, STEP_DAYS):
            history = well.iloc[index - HISTORY_DAYS : index]
            future = well.iloc[index : index + FORECAST_DAYS]
            expected_history = pd.date_range(
                history["DATEPRD"].iloc[0], periods=HISTORY_DAYS, freq="D"
            )
            expected_future = pd.date_range(
                future["DATEPRD"].iloc[0], periods=FORECAST_DAYS, freq="D"
            )
            if not pd.DatetimeIndex(history["DATEPRD"]).equals(expected_history):
                raise AssertionError("calendarized history unexpectedly contains a gap")
            if not pd.DatetimeIndex(future["DATEPRD"]).equals(expected_future):
                raise AssertionError("calendarized future unexpectedly contains a gap")
            oil_future = future["BORE_OIL_VOL"]
            if oil_future.notna().sum() < MIN_OBSERVED_FUTURE_DAYS:
                continue
            target = float(oil_future.mean(skipna=True))
            sequence = _history_sequence(history)
            if np.isnan(sequence[0]).all() or not np.isfinite(target):
                continue
            cutoff = pd.Timestamp(future["DATEPRD"].iloc[0])
            sample_ids.append(f"productivity-calendar:{group}:{cutoff.date().isoformat()}")
            groups.append(str(group))
            sequences.append(sequence)
            timestamps.append(history["DATEPRD"].to_numpy(dtype="datetime64[ns]"))
            targets.append(target)
    return {
        "sample_ids": tuple(sample_ids),
        "groups": tuple(groups),
        "sequences": np.asarray(sequences, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype="datetime64[ns]"),
        "targets": np.asarray(targets, dtype=np.float64),
    }


def _seeded_subset(
    sample_ids: Sequence[str],
    limit: int,
    *,
    fold_id: int,
    lane: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            sample_ids,
            key=lambda value: hashlib.sha256(
                f"{ROOT_SEED}|T3-calendar|{fold_id}|{lane}|{value}".encode("utf-8")
            ).hexdigest(),
        )[:limit]
    )


def _take(dataset: Mapping[str, Any], ids: Sequence[str]) -> tuple[Any, ...]:
    by_id = {value: index for index, value in enumerate(dataset["sample_ids"])}
    indices = np.asarray([by_id[value] for value in ids], dtype=int)
    return (
        np.asarray(dataset["sequences"])[indices],
        np.asarray(dataset["timestamps"])[indices],
        np.asarray(dataset["targets"])[indices],
        tuple(dataset["groups"][index] for index in indices),
    )


def load_calendar_pilot_data(
    audit: Any,
    *,
    source_root: Path,
    fold_id: int,
    train_limit: int = 196,
    validation_limit: int = 84,
) -> CalendarPilotData:
    target = audit.target("T3")
    split_path = data_module.PROJECT_ROOT / target["split_manifest"]["path"]
    split = json.loads(split_path.read_text(encoding="utf-8"))
    matching = [row for row in split["folds"] if int(row["fold_id"]) == int(fold_id)]
    if len(matching) != 1:
        raise ValueError(f"T3 split does not contain exactly one fold {fold_id}")
    fold = matching[0]
    development_groups = set(split["development_groups"])
    frame, source_provenance = data_module._load_development_production(
        Path(source_root), development_groups
    )
    dataset = build_calendar_productivity(frame)
    train_candidates = [
        sample_id
        for sample_id, group in zip(dataset["sample_ids"], dataset["groups"])
        if group in set(fold["train_groups"])
    ]
    validation_candidates = [
        sample_id
        for sample_id, group in zip(dataset["sample_ids"], dataset["groups"])
        if group in set(fold["validation_groups"])
    ]
    train_ids = _seeded_subset(
        train_candidates, train_limit, fold_id=fold_id, lane="train"
    )
    validation_ids = _seeded_subset(
        validation_candidates, validation_limit, fold_id=fold_id, lane="validation"
    )
    train_sequence, train_time, train_target, train_groups = _take(dataset, train_ids)
    validation_sequence, validation_time, validation_target, validation_groups = _take(
        dataset, validation_ids
    )
    if set(train_groups) & set(validation_groups):
        raise ValueError("calendar T3 train/validation groups overlap")
    split_hash = _canonical_sha256(
        {
            "parent_split_sha256": target["split_manifest"]["sha256"],
            "policy": "daily_grid_v1",
            "history_days": HISTORY_DAYS,
            "forecast_days": FORECAST_DAYS,
            "step_days": STEP_DAYS,
            "development_sample_ids": list(dataset["sample_ids"]),
        }
    )
    budget_hash = _canonical_sha256(
        {
            "split_sha256": split_hash,
            "train_sample_ids": list(train_ids),
            "validation_sample_ids": list(validation_ids),
            "sequence_columns": list(SEQUENCE_COLUMNS),
        }
    )
    return CalendarPilotData(
        fold_id=int(fold_id),
        train_sample_ids=train_ids,
        validation_sample_ids=validation_ids,
        train_groups=train_groups,
        validation_groups=validation_groups,
        train_sequence=train_sequence,
        validation_sequence=validation_sequence,
        train_timestamps=train_time,
        validation_timestamps=validation_time,
        train_target=train_target,
        validation_target=validation_target,
        split_sha256=split_hash,
        input_budget_sha256=budget_hash,
        provenance={
            **dict(source_provenance),
            "calendar_policy": "daily_grid_v1",
            "history_days": HISTORY_DAYS,
            "forecast_days": FORECAST_DAYS,
            "missing_dates_reindexed_not_zero_filled": True,
            "archived_p5_sample_ids_reused": False,
            "known_holdout_accessed": False,
            "frozen_test_accessed": False,
        },
    )
