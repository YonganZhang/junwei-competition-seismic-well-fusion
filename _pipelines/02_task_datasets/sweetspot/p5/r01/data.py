"""Development-only reconstruction and field audit for sweetspot P5.1 R0/R1.

Authorization is applied to ZIP members or workbook rows before numerical
values are opened.  This module has no API for physical test or known-holdout
artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..sweetspot_p5_stage2_data import (
    PRODUCTION_SEQUENCE_COLUMNS,
    RAW_LOG_FEATURES,
    DevelopmentDataUnavailable,
    _load_development_petrophysical_tables,
    _load_development_production,
    _resolve_source_root,
    _rqi,
    _sand_flag,
)
from ..sweetspot_p5_stage2_labels import PROJECT_ROOT, sha256_file, validate_label_mapping
from .contracts import CONTRACT_ORDER, canonical_sha256


MAX_SAMPLES_PER_GROUP = 768
HISTORY_CALENDAR_DAYS = 30
ORIGIN_STRIDE_DAYS = 7


@dataclass(frozen=True)
class R01Dataset:
    target_id: str
    task_type: str
    target_name: str
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    groups: tuple[str, ...]
    cutoffs: tuple[str | None, ...]
    features: np.ndarray
    targets: np.ndarray
    development_groups: tuple[str, ...]
    provenance: Mapping[str, Any]
    coverage: Mapping[str, Any]

    @property
    def sample_sha256(self) -> str:
        return canonical_sha256([
            {"sample_id": sample, "group": group, "cutoff": cutoff}
            for sample, group, cutoff in zip(self.sample_ids, self.groups, self.cutoffs)
        ])


def _stable_cap(indices: Sequence[int], sample_ids: Sequence[str], limit: int) -> list[int]:
    return sorted(indices, key=lambda index: canonical_sha256(sample_ids[index]))[:limit]


def _cap_by_group(payload: dict[str, Any], limit: int = MAX_SAMPLES_PER_GROUP) -> dict[str, Any]:
    selected: list[int] = []
    groups = np.asarray(payload["groups"], dtype=object)
    for group in sorted(set(groups.tolist())):
        indices = np.flatnonzero(groups == group).tolist()
        selected.extend(_stable_cap(indices, payload["sample_ids"], limit))
    selected.sort(key=lambda index: payload["sample_ids"][index])
    return {
        **payload,
        "sample_ids": [payload["sample_ids"][i] for i in selected],
        "groups": [payload["groups"][i] for i in selected],
        "cutoffs": [payload["cutoffs"][i] for i in selected],
        "features": np.asarray(payload["features"], dtype=float)[selected],
        "targets": np.asarray(payload["targets"], dtype=float)[selected],
        "pre_cap_count": len(payload["sample_ids"]),
    }


def _split_evidence(mapping: Any, target_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = mapping.target(target_id)["split_manifest"]
    path = PROJECT_ROOT / reference["path"]
    payload = mapping.split_manifest(target_id)
    groups = tuple(payload.get("development_groups", ()))
    if not groups:
        raise DevelopmentDataUnavailable("development_split_missing", f"{target_id}: no development groups")
    evidence = {
        "p4_split_manifest": reference["path"],
        "p4_split_manifest_sha256": sha256_file(path),
        "p4_split_reference_sha256": reference["sha256"],
        "development_groups": list(groups),
        "p4_sample_ids_reused": False,
        "reason": "R0 label semantics rebuilds samples but preserves P4 development groups",
    }
    return payload, evidence


def _petrophysical_payload(
    tables: Sequence[Mapping[str, Any]], target_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_ids: list[str] = []
    groups: list[str] = []
    cutoffs: list[None] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    total_rows = 0
    invalid_label = 0
    no_feature = 0
    target_name = {"T1": "t1_rqi_proxy", "T2": "t2_sand_flag_proxy", "T6": "t6_phif", "T7": "t7_klogh"}[target_id]
    for table in tables:
        labels = table["labels"]
        if target_id == "T1":
            if not {"PHIF", "KLOGH"} <= set(labels):
                continue
            label = _rqi(labels)
        elif target_id == "T2":
            if "SAND_FLAG" not in labels:
                continue
            label = _sand_flag(labels)
        elif target_id == "T6":
            if "PHIF" not in labels:
                continue
            label = np.asarray(labels["PHIF"], dtype=float)
        else:
            if "KLOGH" not in labels:
                continue
            raw = np.asarray(labels["KLOGH"], dtype=float)
            label = np.where(raw >= 0.0, raw, np.nan)
        depth = np.asarray(table["depth_m"], dtype=float)
        matrix = np.column_stack([
            np.asarray(table["features"][field], dtype=float) for field in RAW_LOG_FEATURES
        ])
        total_rows += len(label)
        label_valid = np.isfinite(label) & np.isfinite(depth)
        feature_valid = np.isfinite(matrix).any(axis=1)
        invalid_label += int((~label_valid).sum())
        no_feature += int((label_valid & ~feature_valid).sum())
        well_token = re.sub(r"[^A-Za-z0-9]+", "-", str(table["wellbore"])).strip("-")
        for index in np.flatnonzero(label_valid & feature_valid):
            sample_ids.append(f"{target_id.lower()}:{well_token}:{depth[index]:.4f}")
            groups.append(str(table["well_family"]))
            cutoffs.append(None)
            features.append(matrix[index])
            targets.append(float(label[index]))
    if not sample_ids:
        raise DevelopmentDataUnavailable("development_rebuild_empty", f"{target_id}: no valid development samples")
    payload = _cap_by_group({
        "sample_ids": sample_ids, "groups": groups, "cutoffs": cutoffs,
        "features": np.asarray(features), "targets": np.asarray(targets),
    })
    coverage = {
        "source_join_rows": total_rows,
        "invalid_or_missing_label_rows": invalid_label,
        "valid_label_but_no_raw_feature_rows": no_feature,
        "eligible_before_bounded_cap": payload["pre_cap_count"],
        "bounded_samples": len(payload["sample_ids"]),
        "per_group": {group: payload["groups"].count(group) for group in sorted(set(payload["groups"]))},
        "feature_nonmissing_fraction": {
            field: float(np.isfinite(payload["features"][:, index]).mean())
            for index, field in enumerate(RAW_LOG_FEATURES)
        },
        "time_coverage": "not_applicable_depth_domain",
        "spatial_scale": "wellbore_depth_point",
    }
    return payload, coverage


def _daily_frame(well: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    subset = well.copy()
    subset = subset[subset["DATEPRD"].notna()].sort_values("DATEPRD")
    duplicate_count = int(subset["DATEPRD"].duplicated().sum())
    if duplicate_count:
        raise DevelopmentDataUnavailable(
            "duplicate_production_dates", "daily production has duplicate well/date rows",
        )
    if subset.empty:
        return subset, duplicate_count
    index = pd.date_range(subset["DATEPRD"].min().normalize(), subset["DATEPRD"].max().normalize(), freq="D")
    subset = subset.set_index(subset["DATEPRD"].dt.normalize()).reindex(index)
    subset.index.name = "calendar_date"
    subset["_row_observed"] = subset["DATEPRD"].notna()
    return subset, duplicate_count


def _history_features(history: pd.DataFrame) -> tuple[tuple[str, ...], np.ndarray]:
    names: list[str] = []
    values: list[float] = []
    for column in PRODUCTION_SEQUENCE_COLUMNS:
        values_raw = pd.to_numeric(history[column], errors="coerce").to_numpy(dtype=float)
        finite = values_raw[np.isfinite(values_raw)]
        for suffix, value in (
            ("mean", np.mean(finite) if finite.size else np.nan),
            ("std", np.std(finite) if finite.size else np.nan),
            ("last", finite[-1] if finite.size else np.nan),
            ("observed_fraction", finite.size / HISTORY_CALENDAR_DAYS),
        ):
            names.append(f"history_{HISTORY_CALENDAR_DAYS}cd_{column.lower()}_{suffix}")
            values.append(float(value))
    return tuple(names), np.asarray(values, dtype=float)


def label_t3_window(future: pd.DataFrame, *, boundary_complete: bool) -> tuple[float | None, str]:
    if not boundary_complete or len(future) != 30:
        return None, "right_boundary_incomplete"
    values = pd.to_numeric(future["BORE_OIL_VOL"], errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(values)
    if int(observed.sum()) < 24:
        return None, "fewer_than_24_observed_days"
    return float(values[observed].mean()), "observed"


def label_t4_window(future: pd.DataFrame, *, boundary_complete: bool) -> tuple[float | None, str]:
    if not boundary_complete or len(future) != 30:
        return None, "right_boundary_incomplete"
    values = pd.to_numeric(future["BORE_WAT_VOL"], errors="coerce").to_numpy(dtype=float)
    observed = np.isfinite(values)
    positive = observed & (values > 0.0)
    for start in range(0, 24):
        if bool(np.all(positive[start:start + 7])):
            return 1.0, "event"
    if not bool(np.all(observed)):
        return None, "missing_makes_nonevent_indeterminate"
    return 0.0, "no_event"


def _history_has_water_onset(history: pd.DataFrame) -> bool:
    values = pd.to_numeric(history["BORE_WAT_VOL"], errors="coerce").to_numpy(dtype=float)
    positive = np.isfinite(values) & (values > 0.0)
    return any(bool(np.all(positive[start:start + 7])) for start in range(max(0, len(values) - 6)))


def _production_payload(frame: pd.DataFrame, target_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_ids: list[str] = []
    groups: list[str] = []
    cutoffs: list[str] = []
    features: list[np.ndarray] = []
    targets: list[float] = []
    censor_reasons: dict[str, int] = {}
    post_onset_excluded = 0
    origin_count = 0
    feature_names: tuple[str, ...] = ()
    date_coverage: dict[str, Any] = {}
    for group, well in frame.groupby("WELL_BORE_CODE", sort=True):
        daily, duplicates = _daily_frame(well)
        if daily.empty:
            continue
        date_coverage[str(group)] = {
            "start": daily.index.min().date().isoformat(),
            "end": daily.index.max().date().isoformat(),
            "calendar_days": len(daily),
            "observed_rows": int(daily["_row_observed"].sum()),
            "duplicate_dates": duplicates,
        }
        first_origin = daily.index.min() + pd.Timedelta(days=HISTORY_CALENDAR_DAYS - 1)
        last_origin = daily.index.max()
        for t0 in pd.date_range(first_origin, last_origin, freq=f"{ORIGIN_STRIDE_DAYS}D"):
            origin_count += 1
            history = daily.loc[t0 - pd.Timedelta(days=HISTORY_CALENDAR_DAYS - 1):t0]
            future_end = t0 + pd.Timedelta(days=30)
            future = daily.loc[t0 + pd.Timedelta(days=1):min(future_end, daily.index.max())]
            boundary_complete = daily.index.max() >= future_end
            if target_id == "T4" and _history_has_water_onset(history):
                post_onset_excluded += 1
                continue
            target, state = (
                label_t3_window(future, boundary_complete=boundary_complete)
                if target_id == "T3" else
                label_t4_window(future, boundary_complete=boundary_complete)
            )
            if target is None:
                censor_reasons[state] = censor_reasons.get(state, 0) + 1
                continue
            names, row = _history_features(history)
            if not np.isfinite(row).any():
                censor_reasons["no_finite_history_features"] = censor_reasons.get("no_finite_history_features", 0) + 1
                continue
            feature_names = names
            sample_ids.append(f"{target_id.lower()}:{group}:{t0.date().isoformat()}")
            groups.append(str(group))
            cutoffs.append(t0.date().isoformat())
            features.append(row)
            targets.append(float(target))
    if not sample_ids:
        raise DevelopmentDataUnavailable("development_rebuild_empty", f"{target_id}: no uncensored samples")
    payload = _cap_by_group({
        "sample_ids": sample_ids, "groups": groups, "cutoffs": cutoffs,
        "features": np.asarray(features), "targets": np.asarray(targets),
    })
    coverage = {
        "candidate_origins": origin_count,
        "uncensored_before_bounded_cap": payload["pre_cap_count"],
        "bounded_samples": len(payload["sample_ids"]),
        "censored_or_unusable": sum(censor_reasons.values()),
        "censor_reasons": censor_reasons,
        "post_onset_origins_excluded": post_onset_excluded,
        "per_group": {group: payload["groups"].count(group) for group in sorted(set(payload["groups"]))},
        "date_coverage": date_coverage,
        "origin_stride_days": ORIGIN_STRIDE_DAYS,
        "history_calendar_days": HISTORY_CALENDAR_DAYS,
        "target_horizon_calendar_days": 30,
        "spatial_scale": "production_wellbore",
    }
    payload["feature_names"] = feature_names
    return payload, coverage


def _field_audit(
    petrophysical_tables: Sequence[Mapping[str, Any]], production: pd.DataFrame,
) -> dict[str, Any]:
    petro_labels = sorted({field for table in petrophysical_tables for field in table["labels"]})
    return {
        "schema_version": "sweetspot-p5.1-r0-field-audit/v1",
        "petrophysical": {
            "table_count": len(petrophysical_tables),
            "development_groups": sorted({str(table["well_family"]) for table in petrophysical_tables}),
            "raw_input_fields": list(RAW_LOG_FEATURES),
            "interpreted_label_fields_observed": petro_labels,
            "field_roles": {field: "label_only_not_inference_input" for field in petro_labels},
        },
        "production": {
            "authorized_row_count": len(production),
            "development_groups": sorted(production["WELL_BORE_CODE"].dropna().astype(str).unique().tolist()),
            "fields": list(production.columns),
            "date_min": production["DATEPRD"].min().date().isoformat(),
            "date_max": production["DATEPRD"].max().date().isoformat(),
        },
        "test_firewall": {
            "physical_test_h5_accessed": False,
            "known_holdout_accessed": False,
            "frozen_test_metrics_accessed": False,
            "source_authorization": "development groups before value read",
        },
    }


def build_development_datasets(source_root: Path | None = None) -> tuple[dict[str, R01Dataset], dict[str, Any]]:
    root = _resolve_source_root(source_root)
    mapping = validate_label_mapping()
    split_payloads: dict[str, dict[str, Any]] = {}
    split_evidence: dict[str, dict[str, Any]] = {}
    for target_id in CONTRACT_ORDER:
        if target_id == "T5":
            continue
        split_payloads[target_id], split_evidence[target_id] = _split_evidence(mapping, target_id)

    petro_groups = set().union(*(
        set(split_payloads[target_id]["development_groups"])
        for target_id in ("T1", "T2", "T6", "T7")
    ))
    production_groups = set().union(*(
        set(split_payloads[target_id]["development_groups"])
        for target_id in ("T3", "T4")
    ))
    tables, petro_source = _load_development_petrophysical_tables(root, petro_groups)
    production, production_source = _load_development_production(root, production_groups)
    if petro_source.get("test_accessed") or production_source.get("test_accessed"):
        raise PermissionError("development source loader reported test access")

    datasets: dict[str, R01Dataset] = {}
    blocked_targets: dict[str, dict[str, str]] = {}
    for target_id in ("T1", "T2", "T6", "T7"):
        allowed = set(split_payloads[target_id]["development_groups"])
        target_tables = [table for table in tables if table["well_family"] in allowed]
        try:
            payload, coverage = _petrophysical_payload(target_tables, target_id)
        except DevelopmentDataUnavailable as exc:
            blocked_targets[target_id] = {"reason_code": exc.reason_code, "detail": exc.detail}
            continue
        source = {**petro_source, "authorized_development_groups": sorted(allowed)}
        datasets[target_id] = R01Dataset(
            target_id, "binary" if target_id == "T2" else "regression",
            {"T1": "t1_rqi_proxy", "T2": "t2_sand_flag_proxy", "T6": "t6_phif", "T7": "t7_klogh"}[target_id],
            tuple(RAW_LOG_FEATURES), tuple(payload["sample_ids"]), tuple(payload["groups"]),
            tuple(payload["cutoffs"]), payload["features"], payload["targets"], tuple(sorted(allowed)),
            {"source": source, "split": split_evidence[target_id]}, coverage,
        )
    for target_id in ("T3", "T4"):
        allowed = set(split_payloads[target_id]["development_groups"])
        target_frame = production[production["WELL_BORE_CODE"].isin(allowed)].copy()
        try:
            payload, coverage = _production_payload(target_frame, target_id)
        except DevelopmentDataUnavailable as exc:
            blocked_targets[target_id] = {"reason_code": exc.reason_code, "detail": exc.detail}
            continue
        source = {**production_source, "authorized_development_groups": sorted(allowed)}
        datasets[target_id] = R01Dataset(
            target_id, "regression" if target_id == "T3" else "binary",
            "t3_future_30d_mean_oil" if target_id == "T3" else "t4_water_onset_30d_proxy",
            tuple(payload["feature_names"]), tuple(payload["sample_ids"]), tuple(payload["groups"]),
            tuple(payload["cutoffs"]), payload["features"], payload["targets"], tuple(sorted(allowed)),
            {"source": source, "split": split_evidence[target_id]}, coverage,
        )

    audit = _field_audit(tables, production)
    audit["source_manifest_sha256"] = canonical_sha256({
        "petrophysical": petro_source, "production": production_source,
    })
    audit["dataset_sample_sha256"] = {
        target_id: dataset.sample_sha256 for target_id, dataset in datasets.items()
    }
    audit["dataset_split_sha256"] = {
        target_id: evidence["p4_split_manifest_sha256"]
        for target_id, evidence in split_evidence.items()
    }
    audit["blocked_targets"] = blocked_targets
    return datasets, audit
