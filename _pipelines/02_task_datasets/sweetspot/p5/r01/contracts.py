"""Versioned, fail-closed contracts for the seven independent sweetspot targets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec


HERE = Path(__file__).resolve().parent
CONTRACT_DIR = HERE / "contracts"
CONTRACT_ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
SCHEMA_VERSION = "sweetspot-p5.1-r0-target-contract/v1"
ROOT_SEED = 2693
RAW_LOG_FEATURES = (
    "GR", "RHOB", "NPHI", "RT", "DT", "DTS", "PEF", "DRHO", "CALI", "BS",
    "RD", "RM", "RACEHM", "RACELM", "RPCEHM", "RPCELM",
)
PRODUCTION_HISTORY_FEATURES = (
    "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL", "ON_STREAM_HRS",
    "AVG_DOWNHOLE_PRESSURE", "AVG_CHOKE_SIZE_P", "AVG_WHP_P",
)
FORBIDDEN_INPUTS = (
    "PHIF", "PHIE", "KLOGH", "KLOGV", "SW", "VSH", "SAND_FLAG", "PERF_FLAG",
    "FUTURE_30D_MEAN_OIL", "WATER_EVENT_WITHIN_30D", "test.h5", "known_holdout",
    "frozen_test",
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def validate_contract(target_id: str, payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "contract_version", "target_id", "task_id", "head_name",
        "status", "task_type", "semantic_name", "truth_class", "field_truth",
        "label", "units", "source_fields", "input_fields", "support", "censoring", "split", "metrics",
        "approval", "r1_lane", "warnings",
    }
    _require(required <= set(payload), f"{target_id}: missing contract keys {sorted(required-set(payload))}")
    _require(payload["schema_version"] == SCHEMA_VERSION, f"{target_id}: unsupported schema")
    _require(payload["target_id"] == target_id, f"{target_id}: target id mismatch")
    _require(payload["head_name"].startswith(f"{target_id.lower()}_"), f"{target_id}: non-isolated head")
    _require(payload["split"].get("root_seed") == ROOT_SEED, f"{target_id}: root seed changed")
    _require(payload["split"].get("test_access") == "forbidden", f"{target_id}: test firewall absent")
    overlap = set(payload["input_fields"]) & set(payload["source_fields"])
    if target_id in {"T3", "T4"}:
        cutoff = payload["split"].get("time_cutoff", {})
        _require(cutoff.get("feature_end") == "t0", f"{target_id}: history cutoff missing")
        _require(cutoff.get("target_start") == "t0+1d", f"{target_id}: future target boundary missing")
    else:
        _require(not overlap, f"{target_id}: label leakage")
    _require(not set(payload["input_fields"]) & set(FORBIDDEN_INPUTS), f"{target_id}: forbidden input")
    _require(payload["approval"].get("basis") == "explicit_user_goal_directive", f"{target_id}: approval basis missing")
    _require(str(payload["support"].get("spatial_scale", "")).strip(), f"{target_id}: spatial scale missing")
    _require(str(payload["support"].get("time_scale", "")).strip(), f"{target_id}: time scale missing")
    _require(str(payload["censoring"]).strip(), f"{target_id}: censoring rule missing")
    if target_id == "T1":
        _require(payload["label"]["formula"] == "0.0314*sqrt(KLOGH/PHIF)", "T1: RQI formula changed")
        _require(payload["label"].get("threshold") is None, "T1: threshold is forbidden")
        _require(payload["field_truth"] is False and payload["truth_class"] == "proxy", "T1: proxy warning changed")
    elif target_id == "T2":
        _require(payload["label"]["formula"] == "near_binary(SAND_FLAG)", "T2: label changed")
        _require(payload["semantic_name"] == "net_reservoir_sand_proxy", "T2: hydrocarbon-pay overclaim")
    elif target_id == "T3":
        _require(payload["label"]["horizon_calendar_days"] == 30, "T3: horizon changed")
        _require(payload["label"]["minimum_observed_days"] == 24, "T3: coverage changed")
        _require(payload["label"]["missing_is_zero"] is False, "T3: missing may not become zero")
        _require(payload["label"]["explicit_zero_retained"] is True, "T3: explicit zero must be retained")
    elif target_id == "T4":
        _require(payload["semantic_name"] == "water_onset_30d_proxy", "T4: only proxy lane is approved")
        _require(payload["label"]["consecutive_calendar_days"] == 7, "T4: event run changed")
        _require(payload["label"]["explicit_zero_interrupts"] is True, "T4: zero must interrupt event")
        _require(payload["label"]["missing_can_censor"] is True, "T4: missing censoring changed")
        _require(payload["blocked_lanes"]["formal_failure_survival"] == "unapproved", "T4: survival lane must remain blocked")
    elif target_id == "T5":
        _require(payload["status"] == "not_feasible", "T5: must remain not_feasible")
        _require(payload["approval"]["approved"] is False, "T5: may not be approved")
    elif target_id == "T6":
        _require(payload["label"]["formula"] == "PHIF", "T6: target changed")
        _require(payload["split"]["strategy"] == "mother_well_logo", "T6: split changed")
    elif target_id == "T7":
        _require(payload["label"]["formula"] == "KLOGH", "T7: target changed")
        _require(payload["label"]["model_transform"] == "log1p", "T7: transform changed")
        _require(payload["split"]["strategy"] == "mother_well_logo", "T7: split changed")


def load_contracts(contract_dir: Path = CONTRACT_DIR) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for target_id in CONTRACT_ORDER:
        path = Path(contract_dir) / f"{target_id}.v1.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_contract(target_id, payload)
        payload["contract_sha256"] = sha256_file(path)
        payload["contract_path"] = path.relative_to(HERE.parents[4]).as_posix()
        contracts[target_id] = payload
    _require(len({item["head_name"] for item in contracts.values()}) == 7, "heads must be independent")
    _require(contracts["T6"]["head_name"] != contracts["T7"]["head_name"], "T6/T7 heads collide")
    return contracts


def task_spec(contract: Mapping[str, Any]) -> TaskSpec:
    target = str(contract["head_name"])
    regression = contract["task_type"] == "regression"
    metrics = tuple(contract["metrics"]["primary"])
    directions = dict(contract["metrics"]["directions"])
    transform = contract["label"].get("model_transform", "identity")
    return TaskSpec(
        track_id="sweetspot",
        task_id=str(contract["task_id"]),
        task_type=str(contract["task_type"]),
        input_modalities=(str(contract["r1_lane"]["input_modality"]),),
        targets=(target,),
        units={target: str(contract["units"])},
        label_version=str(contract["label"]["version"]),
        target_masks={target: "finite_and_uncensored"},
        group_keys=(str(contract["split"]["group_key"]),),
        target_transform={target: {"name": transform}},
        inverse_transform={target: {"name": "expm1" if transform == "log1p" else "identity"}},
        train_loss={target: {"name": "huber" if regression else "log_loss"}},
        inference_transform={target: {"name": "identity" if regression else "sigmoid"}},
        threshold_policy={} if regression else {"name": "fixed", "value": 0.5, "fit_on": "never"},
        calibration_policy={"name": "none", "fit_on": "never"},
        primary_metrics=metrics,
        metric_directions=directions,
        visualizer_id=f"sweetspot_p51_{contract['target_id'].lower()}",
        required_figures=(f"{contract['target_id'].lower()}_legal_vs_random_diagnostic",),
        input_whitelist=tuple(contract["input_fields"]),
        forbidden_inputs=FORBIDDEN_INPUTS,
        time_cutoff=contract["split"].get("time_cutoff"),
        hpo={"enabled": False, "stage": "R1_protocol_mechanism_only"},
        metadata={
            "single_target_head": True,
            "truth_class": contract["truth_class"],
            "field_truth": contract["field_truth"],
            "r1_no_final_ranking": True,
            "root_seed": ROOT_SEED,
            "test_access": "forbidden",
        },
    )
