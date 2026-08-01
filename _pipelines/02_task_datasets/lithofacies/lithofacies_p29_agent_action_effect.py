#!/usr/bin/env python3
"""P29 lithofacies agent-action effect repair and bounded development pilot.

P29 preserves the P28 nested LOGO split and fixed-schema nine-class Macro-F1,
but repairs two protocol defects.  Policy observations now expose only bounded
development-only effect units, anonymous per-class train support, and inner-fold
uncertainty.  They never expose raw metrics, labels, sample/group identities,
paths, or any outer-promotion result.  The deterministic XGBoost executor is
explicitly collapsed to one seed; the three inner LOGO folds, not duplicate
seed hashes, are the only uncertainty units.

All policy calls finish before the post-policy inner-to-outer transfer matrix is
computed.  That matrix and the exhaustive action ceiling are diagnostic only
and cannot affect legal promotion selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from lithofacies_p28_agentic_optimization import (  # noqa: E402
    A0,
    ACTION_BY_ID,
    EXPECTED_A0_FOLDS,
    EXPECTED_A0_MEAN,
    EXPECTED_SPLIT_HASH,
    OUTER_FOLDS,
    REPEAT_SEEDS,
    _inner_arrays,
    _metric_payload,
    _outer_arrays,
    _train_action,
    class_weight_vector,
    load_nested_batch,
)
from p4_contract import CLASS_NAMES, DEVELOPMENT_FAMILIES  # noqa: E402


SCHEMA_VERSION = "lithofacies-p29-agent-action-effect/v1"
PROTOCOL_SCHEMA = "lithofacies-p29-protocol/v1"
RESULT_SCHEMA = "lithofacies-p29-result/v1"
MANIFEST_SCHEMA = "lithofacies-p29-artifact-manifest/v1"
OBSERVATION_SCHEMA = "p29_safe_normalized_v1"
CATEGORICAL_ABLATION_SCHEMA = "p29_categorical_ablation_v1"
PRIMARY_METRIC = "fixed_schema_macro_f1"
NUM_CLASSES = 9
TRIAL_BUDGET = 3
MODEL_SEED = int(REPEAT_SEEDS[0])
MODEL_SEEDS = (MODEL_SEED,)
MODEL_REPEAT_MODE = "single_deterministic_seed_no_pseudo_replicates"
UNCERTAINTY_UNIT = "three_disjoint_inner_logo_folds"
EFFECT_UNIT = 0.005
MAX_NORMALIZED_MAGNITUDE = 4.0
T_CRITICAL_DF2_95 = 4.302652729911275
POLICY_SEEDS = (2693, 2694, 2695)
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p29_agent_action_effect"
DEFAULT_BATCH = (
    TRACK_DIR
    / "_outputs"
    / "p28_agentic_optimization"
    / "runtime"
    / "nested_development.npz"
)
P28_SUMMARY = (
    TRACK_DIR / "_outputs" / "p28_agentic_optimization" / "summary.json"
)
P28_RUNNER = TRACK_DIR / "lithofacies_p28_agentic_optimization.py"
PILOT_ACTION_IDS = (
    "ACT_DEPTH4_ETA0075_ROUNDS80",
    "ACT_WEIGHT_EXP075_MEAN1",
    "ACT_WELL_MASK_ONLY_858",
    "ACT_PRIOR_SHRINK010",
)
PILOT_ACTIONS = tuple(ACTION_BY_ID[action_id] for action_id in PILOT_ACTION_IDS)
MAIN_ARMS = ("A0", "A1", "A2L", "A2D", "A3", "ORACLE")
FORBIDDEN_PATH_MARKERS = ("test", "holdout", "frozen")
FORBIDDEN_POLICY_KEYS = (
    "raw",
    "metric",
    "score",
    "label",
    "residual",
    "path",
    "family",
    "sample",
    "promotion",
    "validation",
)
REASON_CODES = (
    "capacity",
    "imbalance",
    "feature_noise",
    "calibration",
    "uncertainty",
    "diversify",
)


class CredentialUnavailable(RuntimeError):
    """Raised before a live policy call when the credential is absent."""


class ProviderUnavailable(RuntimeError):
    """Raised when the provider cannot return a valid strict-JSON action."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def ensure_development_only_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        lowered = str(Path(path)).lower()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise ValueError(f"P29 rejects frozen/test-like path before opening: {path}")


def _owned_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("P29 output must use its isolated owner directory")
    return resolved


def _config_payload(action: Any) -> dict[str, Any]:
    return {
        **asdict(action),
        "model_seed_mode": MODEL_REPEAT_MODE,
        "model_seeds": list(MODEL_SEEDS),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
    }


def _clip_units(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("effect unit must be finite")
    return float(
        np.clip(
            float(value) / EFFECT_UNIT,
            -MAX_NORMALIZED_MAGNITUDE,
            MAX_NORMALIZED_MAGNITUDE,
        )
    )


def _standard_error(values: Sequence[float]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.size < 2:
        return 0.0
    return float(np.std(vector, ddof=1) / math.sqrt(vector.size))


def _effect_category(delta: float) -> str:
    if delta >= EFFECT_UNIT:
        return "improved"
    if delta <= -EFFECT_UNIT:
        return "worse"
    return "flat"


def assert_policy_payload_safe(payload: Mapping[str, Any]) -> None:
    """Reject raw outcomes, identities, paths, and unbounded numeric values."""

    def visit(value: Any, parent_key: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in FORBIDDEN_POLICY_KEYS):
                    raise ValueError(f"forbidden policy field: {key}")
                visit(item, lowered)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, parent_key)
        elif isinstance(value, str):
            lowered = value.lower()
            if any(name.lower() in lowered for name in DEVELOPMENT_FAMILIES):
                raise ValueError("group identity leaked into policy payload")
            if "/" in value or "\\" in value:
                raise ValueError("path-like value leaked into policy payload")
            if any(name.lower() in lowered for name in CLASS_NAMES):
                raise ValueError("class name leaked into policy payload")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite policy value")
            if abs(value) > MAX_NORMALIZED_MAGNITUDE + 1e-12:
                raise ValueError("unbounded continuous policy value")
            if "support_share" in parent_key and not 0.0 <= value <= 1.0:
                raise ValueError("support shares must be normalized")
            if "uncertainty" in parent_key and value < 0.0:
                raise ValueError("uncertainty must be non-negative")
        elif isinstance(value, (int, bool)) or value is None:
            return
        else:
            raise ValueError(f"unsupported policy value type: {type(value).__name__}")

    visit(payload)


def _support_observable(arrays: Mapping[str, np.ndarray], outer_id: int) -> list[dict[str, Any]]:
    counts = np.asarray(
        [
            arrays[f"o{outer_id}_i{inner_id}_class_counts"]
            for inner_id in range(3)
        ],
        dtype=np.float64,
    )
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    rows = []
    for class_slot in range(NUM_CLASSES):
        values = shares[:, class_slot]
        rows.append(
            {
                "class_slot": class_slot,
                "support_share_mean": float(np.mean(values)),
                "support_share_min": float(np.min(values)),
                "support_share_max": float(np.max(values)),
                "support_present_inner_folds": int(np.sum(counts[:, class_slot] > 0)),
            }
        )
    return rows


def _fit_observable(baseline: Mapping[str, Any]) -> dict[str, Any]:
    gap = float(baseline["train_mean"] - baseline["selection_mean"])
    if gap >= 0.10:
        state = "overfit"
    elif gap <= 0.03:
        state = "underfit"
    else:
        state = "balanced"
    return {
        "state": state,
        "normalized_gap_units": _clip_units(gap),
    }


def _public_effect(effect: Mapping[str, Any]) -> dict[str, Any]:
    per_class = []
    for item in effect["per_class"]:
        per_class.append(
            {
                "class_slot": int(item["class_slot"]),
                "normalized_delta_units": _clip_units(item["mean_delta"]),
                "uncertainty_units": float(
                    min(
                        MAX_NORMALIZED_MAGNITUDE,
                        item["standard_error"] / EFFECT_UNIT,
                    )
                ),
            }
        )
    ci_low, ci_high = effect["confidence_interval_95"]
    return {
        "normalized_delta_units": _clip_units(effect["mean_delta"]),
        "uncertainty_units": float(
            min(
                MAX_NORMALIZED_MAGNITUDE,
                effect["standard_error"] / EFFECT_UNIT,
            )
        ),
        "confidence_interval_units": [
            _clip_units(ci_low),
            _clip_units(ci_high),
        ],
        "inner_fold_outcomes": dict(effect["inner_fold_outcomes"]),
        "per_class_effect": per_class,
    }


def build_enhanced_observation(
    *,
    arrays: Mapping[str, np.ndarray],
    outer_id: int,
    baseline: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    remaining_actions: Sequence[str],
    trial_index: int,
) -> dict[str, Any]:
    observation = {
        "observation_schema": OBSERVATION_SCHEMA,
        "trial_ordinal": int(trial_index + 1),
        "trial_budget": TRIAL_BUDGET,
        "remaining_budget": int(TRIAL_BUDGET - trial_index),
        "advance_threshold_units": 1.0,
        "uncertainty_unit": UNCERTAINTY_UNIT,
        "class_support": _support_observable(arrays, outer_id),
        "fit_diagnostic": _fit_observable(baseline),
        "history": [
            {
                "action_id": str(item["action_id"]),
                "effect": _public_effect(item["effect"]),
            }
            for item in history
        ],
        "available_actions": [
            {
                "action_id": action_id,
                "description": ACTION_BY_ID[action_id].description,
            }
            for action_id in remaining_actions
        ],
    }
    assert_policy_payload_safe(observation)
    return observation


def build_categorical_ablation_observation(
    *,
    arrays: Mapping[str, np.ndarray],
    outer_id: int,
    baseline: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    remaining_actions: Sequence[str],
    trial_index: int,
) -> dict[str, Any]:
    del arrays, outer_id
    observation = {
        "observation_schema": CATEGORICAL_ABLATION_SCHEMA,
        "trial_ordinal": int(trial_index + 1),
        "trial_budget": TRIAL_BUDGET,
        "fit_state": _fit_observable(baseline)["state"],
        "history": [
            {
                "action_id": str(item["action_id"]),
                "feedback": str(item["feedback"]),
            }
            for item in history
        ],
        "available_actions": [
            {
                "action_id": action_id,
                "description": ACTION_BY_ID[action_id].description,
            }
            for action_id in remaining_actions
        ],
    }
    assert_policy_payload_safe(observation)
    if any(isinstance(value, float) for value in _walk_values(observation)):
        raise RuntimeError("categorical prompt ablation unexpectedly contains floats")
    return observation


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _validate_decision(
    content: str,
    *,
    remaining_actions: Sequence[str],
) -> dict[str, Any]:
    try:
        decision = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("provider response is not strict JSON") from exc
    if not isinstance(decision, dict) or set(decision) != {
        "action_id",
        "reason_code",
        "stop",
    }:
        raise ValueError("provider response keys changed")
    if decision["action_id"] not in set(remaining_actions):
        raise ValueError("provider selected an unavailable action")
    if decision["reason_code"] not in REASON_CODES:
        raise ValueError("provider returned an unknown reason code")
    if decision["stop"] is not False:
        raise ValueError("the fixed trial budget cannot stop early")
    return decision


def call_deepseek_action(
    *,
    observation: Mapping[str, Any],
    api_key: str,
    timeout_seconds: float = 60.0,
    attempts: int = 2,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not api_key.strip():
        raise CredentialUnavailable("DEEPSEEK_KEY is missing; P29 A2L fails closed")
    assert_policy_payload_safe(observation)
    enhanced = observation["observation_schema"] == OBSERVATION_SCHEMA
    system_prompt = (
        "You are a bounded development experiment-search policy. Choose exactly one "
        "available action without replacement. The observation is anonymous and "
        "contains only train-derived support plus bounded inner-fold effect units. "
        "One effect unit equals the preregistered advancement threshold. Prefer robust "
        "positive effects over noisy effects. Never request or infer raw outcomes, "
        "class names, row-level targets, group identities, identifiers, filesystem "
        "locations, or outer-fold outcomes. Return one JSON object and no markdown "
        "with exactly action_id, reason_code, stop; stop must be false."
        if enhanced
        else
        "You are a bounded development experiment-search policy. Choose exactly one "
        "available action without replacement using only categorical fit state and "
        "improved, flat, or worse history. Never request hidden numeric outcomes, "
        "targets, identities, identifiers, filesystem locations, or outer-fold "
        "outcomes. Return one JSON object and no markdown with exactly action_id, "
        "reason_code, stop; stop must be false."
    )
    user_payload = {
        "protocol": (
            "P29_SAFE_EFFECT_THREE_TRIAL"
            if enhanced
            else "P29_CATEGORICAL_ABLATION_THREE_TRIAL"
        ),
        "allowed_reason_codes": list(REASON_CODES),
        "observation": dict(observation),
        "required_output": {
            "action_id": observation["available_actions"][0]["action_id"],
            "reason_code": "diversify",
            "stop": False,
        },
    }
    prompt_sha256 = _stable_hash({"system": system_prompt, "user": user_payload})
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        body = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        user_payload, ensure_ascii=False, sort_keys=True
                    ),
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request = urllib.request.Request(
            DEEPSEEK_ENDPOINT,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.load(response)
            content = str(response_payload["choices"][0]["message"]["content"])
            remaining = [
                item["action_id"] for item in observation["available_actions"]
            ]
            decision = _validate_decision(content, remaining_actions=remaining)
            metadata = {
                "request_model": DEEPSEEK_MODEL,
                "response_model": response_payload.get("model"),
                "response_id": response_payload.get("id"),
                "usage": response_payload.get("usage"),
                "prompt_sha256": prompt_sha256,
                "attempt": attempt,
                "valid": True,
                "credential_persisted": False,
            }
            return decision, metadata
        except Exception as exc:  # fail closed for transport and schema errors
            failures.append(f"{type(exc).__name__}:{str(exc)[:160]}")
    raise ProviderUnavailable(
        "DeepSeek did not return a valid live P29 action after "
        f"{attempts} attempts: {' | '.join(failures)}"
    )


def _executor_state_sha256(
    *,
    action: Any,
    train_well: np.ndarray,
    train_seismic: np.ndarray,
    class_counts: np.ndarray,
    seed: int,
) -> str:
    return _stable_hash(
        {
            "config_sha256": _stable_hash(_config_payload(action)),
            "seed": int(seed),
            "train_well_sha256": _array_hash(train_well),
            "train_seismic_sha256": _array_hash(train_seismic),
            "class_counts_sha256": _array_hash(class_counts),
            "executor": "p28_xgboost_train_action",
        }
    )


def _evaluate_inner_action(
    arrays: Mapping[str, np.ndarray], outer_id: int, action: Any
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for inner_id in range(3):
        fold = _inner_arrays(arrays, outer_id, inner_id)
        train_logits, selection_logits = _train_action(
            action=action,
            train_well=fold["train_well"],
            train_seismic=fold["train_seismic"],
            train_labels=fold["train_labels"],
            evaluation_well=fold["selection_well"],
            evaluation_seismic=fold["selection_seismic"],
            class_counts=fold["class_counts"],
            seed=MODEL_SEED,
        )
        cells.append(
            {
                "outer_rollout_id": outer_id,
                "inner_fold_id": inner_id,
                "model_seed": MODEL_SEED,
                "config_sha256": _stable_hash(_config_payload(action)),
                "executor_state_sha256": _executor_state_sha256(
                    action=action,
                    train_well=fold["train_well"],
                    train_seismic=fold["train_seismic"],
                    class_counts=fold["class_counts"],
                    seed=MODEL_SEED,
                ),
                "train_metrics": _metric_payload(fold["train_labels"], train_logits),
                "selection_metrics": _metric_payload(
                    fold["selection_labels"], selection_logits
                ),
                "prediction_sha256": _array_hash(
                    selection_logits,
                    np.argmax(selection_logits, axis=1).astype(np.int64),
                ),
            }
        )
    selection_values = [
        cell["selection_metrics"][PRIMARY_METRIC] for cell in cells
    ]
    train_values = [cell["train_metrics"][PRIMARY_METRIC] for cell in cells]
    return {
        "action_id": action.action_id,
        "config_sha256": _stable_hash(_config_payload(action)),
        "cells": cells,
        "selection_mean": float(np.mean(selection_values)),
        "train_mean": float(np.mean(train_values)),
        "prediction_sha256": _stable_hash(
            [cell["prediction_sha256"] for cell in cells]
        ),
        "executor_state_sha256": _stable_hash(
            [cell["executor_state_sha256"] for cell in cells]
        ),
    }


def _evaluate_outer_action(
    arrays: Mapping[str, np.ndarray], outer_id: int, action: Any
) -> dict[str, Any]:
    fold = _outer_arrays(arrays, outer_id)
    _, logits = _train_action(
        action=action,
        train_well=fold["train_well"],
        train_seismic=fold["train_seismic"],
        train_labels=fold["train_labels"],
        evaluation_well=fold["promotion_well"],
        evaluation_seismic=fold["promotion_seismic"],
        class_counts=fold["class_counts"],
        seed=MODEL_SEED,
    )
    return {
        "outer_rollout_id": outer_id,
        "model_seed": MODEL_SEED,
        "action_id": action.action_id,
        "config_sha256": _stable_hash(_config_payload(action)),
        "executor_state_sha256": _executor_state_sha256(
            action=action,
            train_well=fold["train_well"],
            train_seismic=fold["train_seismic"],
            class_counts=fold["class_counts"],
            seed=MODEL_SEED,
        ),
        "metrics": _metric_payload(fold["promotion_labels"], logits),
        "prediction_sha256": _array_hash(
            logits,
            np.argmax(logits, axis=1).astype(np.int64),
        ),
    }


class ActionEvaluator:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.inner_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self.outer_cache: dict[tuple[int, str], dict[str, Any]] = {}

    @staticmethod
    def action(action_id: str) -> Any:
        return A0 if action_id == A0.action_id else ACTION_BY_ID[action_id]

    def inner(self, outer_id: int, action_id: str) -> dict[str, Any]:
        key = (outer_id, action_id)
        if key not in self.inner_cache:
            self.inner_cache[key] = _evaluate_inner_action(
                self.arrays, outer_id, self.action(action_id)
            )
        return self.inner_cache[key]

    def outer(self, outer_id: int, action_id: str) -> dict[str, Any]:
        key = (outer_id, action_id)
        if key not in self.outer_cache:
            self.outer_cache[key] = _evaluate_outer_action(
                self.arrays, outer_id, self.action(action_id)
            )
        return self.outer_cache[key]


def summarize_inner_effect(
    action_result: Mapping[str, Any], baseline_result: Mapping[str, Any]
) -> dict[str, Any]:
    action_cells = {
        int(cell["inner_fold_id"]): cell for cell in action_result["cells"]
    }
    baseline_cells = {
        int(cell["inner_fold_id"]): cell for cell in baseline_result["cells"]
    }
    fold_deltas = []
    per_class_deltas: list[list[float]] = [[] for _ in range(NUM_CLASSES)]
    for inner_id in range(3):
        action_metrics = action_cells[inner_id]["selection_metrics"]
        baseline_metrics = baseline_cells[inner_id]["selection_metrics"]
        fold_deltas.append(
            float(action_metrics[PRIMARY_METRIC] - baseline_metrics[PRIMARY_METRIC])
        )
        for class_slot, (action_value, baseline_value) in enumerate(
            zip(action_metrics["per_class_f1"], baseline_metrics["per_class_f1"])
        ):
            per_class_deltas[class_slot].append(float(action_value - baseline_value))
    mean_delta = float(np.mean(fold_deltas))
    standard_error = _standard_error(fold_deltas)
    margin = T_CRITICAL_DF2_95 * standard_error
    outcomes = {"improved": 0, "flat": 0, "worse": 0}
    for value in fold_deltas:
        outcomes[_effect_category(value)] += 1
    return {
        "mean_delta": mean_delta,
        "standard_error": standard_error,
        "confidence_interval_95": [mean_delta - margin, mean_delta + margin],
        "inner_fold_deltas": fold_deltas,
        "inner_fold_outcomes": outcomes,
        "per_class": [
            {
                "class_slot": class_slot,
                "mean_delta": float(np.mean(values)),
                "standard_error": _standard_error(values),
            }
            for class_slot, values in enumerate(per_class_deltas)
        ],
    }


def deterministic_enhanced_action(
    observation: Mapping[str, Any], remaining_actions: Sequence[str]
) -> str:
    remaining = set(remaining_actions)
    history = list(observation["history"])
    support = list(observation["class_support"])
    rare_support = min(float(item["support_share_mean"]) for item in support)
    if not history:
        order = [
            "ACT_WEIGHT_EXP075_MEAN1",
            "ACT_DEPTH4_ETA0075_ROUNDS80",
            "ACT_WELL_MASK_ONLY_858",
            "ACT_PRIOR_SHRINK010",
        ] if rare_support < 0.02 else [
            "ACT_DEPTH4_ETA0075_ROUNDS80",
            "ACT_WEIGHT_EXP075_MEAN1",
            "ACT_WELL_MASK_ONLY_858",
            "ACT_PRIOR_SHRINK010",
        ]
    else:
        last = history[-1]["effect"]
        mean_units = float(last["normalized_delta_units"])
        uncertainty = float(last["uncertainty_units"])
        if mean_units >= 1.0 and uncertainty <= 1.0:
            order = [
                "ACT_PRIOR_SHRINK010",
                "ACT_WELL_MASK_ONLY_858",
                "ACT_DEPTH4_ETA0075_ROUNDS80",
                "ACT_WEIGHT_EXP075_MEAN1",
            ]
        elif uncertainty > 1.0:
            order = [
                "ACT_WELL_MASK_ONLY_858",
                "ACT_PRIOR_SHRINK010",
                "ACT_DEPTH4_ETA0075_ROUNDS80",
                "ACT_WEIGHT_EXP075_MEAN1",
            ]
        else:
            order = [
                "ACT_DEPTH4_ETA0075_ROUNDS80",
                "ACT_WELL_MASK_ONLY_858",
                "ACT_PRIOR_SHRINK010",
                "ACT_WEIGHT_EXP075_MEAN1",
            ]
    return next(action_id for action_id in order if action_id in remaining)


def normalized_auc(deltas: Sequence[float]) -> float:
    if len(deltas) != TRIAL_BUDGET:
        raise ValueError("P29 AUC requires the exact trial budget")
    best = 0.0
    curve = []
    for delta in deltas:
        best = max(best, float(delta))
        curve.append(best)
    return float(np.mean(curve) / EFFECT_UNIT)


def _rollout(
    *,
    arm: str,
    instance_id: str,
    outer_id: int,
    evaluator: ActionEvaluator,
    baseline: Mapping[str, Any],
    api_key: str,
    random_seed: int | None = None,
) -> dict[str, Any]:
    remaining = list(PILOT_ACTION_IDS)
    history: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    rng = random.Random(
        None if random_seed is None else int(random_seed) + outer_id * 1009
    )
    random_order = rng.sample(remaining, len(remaining)) if arm == "A3" else []
    for trial_index in range(TRIAL_BUDGET):
        if arm == "A2L_CATEGORICAL_ABLATION":
            observation = build_categorical_ablation_observation(
                arrays=evaluator.arrays,
                outer_id=outer_id,
                baseline=baseline,
                history=history,
                remaining_actions=remaining,
                trial_index=trial_index,
            )
        else:
            observation = build_enhanced_observation(
                arrays=evaluator.arrays,
                outer_id=outer_id,
                baseline=baseline,
                history=history,
                remaining_actions=remaining,
                trial_index=trial_index,
            )
        provider = None
        if arm in {"A2L", "A2L_CATEGORICAL_ABLATION"}:
            decision, provider = call_deepseek_action(
                observation=observation,
                api_key=api_key,
            )
            action_id = str(decision["action_id"])
            reason_code = str(decision["reason_code"])
        elif arm == "A2D":
            action_id = deterministic_enhanced_action(observation, remaining)
            reason_code = "deterministic_enhanced_diagnostic"
        elif arm == "A3":
            action_id = next(item for item in random_order if item in remaining)
            reason_code = "random_without_replacement"
        else:
            raise ValueError(f"unsupported P29 arm: {arm}")
        result = evaluator.inner(outer_id, action_id)
        effect = summarize_inner_effect(result, baseline)
        feedback = _effect_category(effect["mean_delta"])
        trial = {
            "trial_index": trial_index,
            "action_id": action_id,
            "reason_code": reason_code,
            "valid_action": action_id in remaining,
            "policy_observation": observation,
            "provider": provider,
            "inner_effect_local_evaluator_only": effect,
            "feedback": feedback,
            "config_sha256": result["config_sha256"],
            "prediction_sha256": result["prediction_sha256"],
            "executor_state_sha256": result["executor_state_sha256"],
            "outer_result_exposed_to_policy": False,
        }
        trials.append(trial)
        history.append(
            {
                "action_id": action_id,
                "effect": effect,
                "feedback": feedback,
            }
        )
        remaining.remove(action_id)
    eligible = [
        trial
        for trial in trials
        if trial["inner_effect_local_evaluator_only"]["mean_delta"] >= EFFECT_UNIT
        and trial["inner_effect_local_evaluator_only"]["inner_fold_outcomes"][
            "improved"
        ]
        >= 2
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: item["inner_effect_local_evaluator_only"]["mean_delta"],
        )
        selected_action_id = str(selected["action_id"])
        selected_delta = float(
            selected["inner_effect_local_evaluator_only"]["mean_delta"]
        )
    else:
        selected_action_id = A0.action_id
        selected_delta = 0.0
    return {
        "arm": arm,
        "instance_id": instance_id,
        "outer_rollout_id": outer_id,
        "trial_count": len(trials),
        "trials": trials,
        "normalized_auc_at_3": normalized_auc(
            [
                trial["inner_effect_local_evaluator_only"]["mean_delta"]
                for trial in trials
            ]
        ),
        "selected_for_promotion": selected_action_id,
        "selected_inner_delta": selected_delta,
        "promotion_rule": "mean_delta_ge_0.005_and_at_least_2_of_3_inner_folds_improved",
        "outer_result_exposed_to_policy": False,
    }


def _promotion_summary(
    rollouts: Sequence[Mapping[str, Any]],
    evaluator: ActionEvaluator,
    a0_outer: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    cells = []
    deltas = []
    for rollout in sorted(rollouts, key=lambda item: int(item["outer_rollout_id"])):
        outer_id = int(rollout["outer_rollout_id"])
        action_id = str(rollout["selected_for_promotion"])
        cell = evaluator.outer(outer_id, action_id)
        baseline = a0_outer[outer_id]
        delta = float(
            cell["metrics"][PRIMARY_METRIC]
            - baseline["metrics"][PRIMARY_METRIC]
        )
        deltas.append(delta)
        cells.append({**cell, "delta_from_a0": delta})
    per_class = np.asarray([cell["metrics"]["per_class_f1"] for cell in cells])
    return {
        "cells": cells,
        "fixed_schema_macro_f1_mean": float(
            np.mean([cell["metrics"][PRIMARY_METRIC] for cell in cells])
        ),
        "delta_from_a0": float(np.mean(deltas)),
        "outer_fold_deltas": deltas,
        "positive_outer_folds": int(sum(delta > 0.0 for delta in deltas)),
        "material_positive_outer_folds": int(
            sum(delta >= EFFECT_UNIT for delta in deltas)
        ),
        "per_class_f1_mean": per_class.mean(axis=0).tolist(),
        "all_class_metrics_finite": bool(np.isfinite(per_class).all()),
        "selection_used_outer_diagnostic": False,
    }


def _actual_a1_replay(
    arrays: Mapping[str, np.ndarray],
    a0_outer: Mapping[int, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    replay = [
        _evaluate_outer_action(arrays, outer_id, A0) for outer_id in OUTER_FOLDS
    ]
    comparisons = []
    for observed in replay:
        outer_id = int(observed["outer_rollout_id"])
        expected = a0_outer[outer_id]
        comparisons.append(
            {
                "outer_rollout_id": outer_id,
                "same_seed": observed["model_seed"] == expected["model_seed"],
                "same_config": observed["config_sha256"] == expected["config_sha256"],
                "same_prediction": observed["prediction_sha256"]
                == expected["prediction_sha256"],
                "same_primary": observed["metrics"][PRIMARY_METRIC]
                == expected["metrics"][PRIMARY_METRIC],
            }
        )
    if not all(
        item["same_seed"]
        and item["same_config"]
        and item["same_prediction"]
        and item["same_primary"]
        for item in comparisons
    ):
        raise RuntimeError("P29 actual A1 identity replay diverged from A0")
    return (
        {
            "actual_same_entrypoint_replay": True,
            "executor_entrypoint": "_evaluate_outer_action",
            "model_seeds": list(MODEL_SEEDS),
            "cell_count": len(replay),
            "comparisons": comparisons,
            "all_equal": True,
            "prediction_sha256": _stable_hash(
                [item["prediction_sha256"] for item in replay]
            ),
            "primary_sha256": _stable_hash(
                [item["metrics"][PRIMARY_METRIC] for item in replay]
            ),
        },
        replay,
    )


def mark_action_effect_flags(entries: Sequence[dict[str, Any]]) -> dict[str, bool]:
    """Mark config/prediction no-ops against the first, frozen A0 entry."""
    if not entries or entries[0].get("action_id") != A0.action_id:
        raise ValueError("action-effect entries must start with A0")
    baseline = entries[0]
    for entry in entries:
        entry["config_differs_from_a0"] = (
            entry["config_sha256"] != baseline["config_sha256"]
        )
        entry["inner_prediction_differs_from_a0"] = (
            entry["inner_prediction_sha256"]
            != baseline["inner_prediction_sha256"]
        )
        entry["outer_prediction_differs_from_a0"] = (
            entry["outer_prediction_sha256"]
            != baseline["outer_prediction_sha256"]
        )
        entry["effective_action_noop"] = (
            entry["action_id"] != A0.action_id
            and not entry["inner_prediction_differs_from_a0"]
            and not entry["outer_prediction_differs_from_a0"]
        )
    return {
        "all_nonbaseline_configs_differ": all(
            item["config_differs_from_a0"] for item in entries[1:]
        ),
        "all_nonbaseline_actions_change_prediction": all(
            item["inner_prediction_differs_from_a0"]
            or item["outer_prediction_differs_from_a0"]
            for item in entries[1:]
        ),
    }


def _action_effects_and_transfer(
    *,
    evaluator: ActionEvaluator,
    a0_inner: Mapping[int, Mapping[str, Any]],
    a0_outer: Mapping[int, Mapping[str, Any]],
    a2l_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    action_ids = (A0.action_id,) + PILOT_ACTION_IDS
    entries = []
    transfer_pairs = []
    for action_id in action_ids:
        action = ActionEvaluator.action(action_id)
        inner_results = {
            outer_id: evaluator.inner(outer_id, action_id)
            for outer_id in OUTER_FOLDS
        }
        outer_results = {
            outer_id: evaluator.outer(outer_id, action_id)
            for outer_id in OUTER_FOLDS
        }
        inner_effects = {
            outer_id: summarize_inner_effect(
                inner_results[outer_id], a0_inner[outer_id]
            )
            for outer_id in OUTER_FOLDS
        }
        outer_deltas = {
            outer_id: float(
                outer_results[outer_id]["metrics"][PRIMARY_METRIC]
                - a0_outer[outer_id]["metrics"][PRIMARY_METRIC]
            )
            for outer_id in OUTER_FOLDS
        }
        entries.append(
            {
                "action_id": action_id,
                "config": _config_payload(action),
                "config_sha256": _stable_hash(_config_payload(action)),
                "executor_state_sha256": _stable_hash(
                    [
                        inner_results[outer_id]["executor_state_sha256"]
                        for outer_id in OUTER_FOLDS
                    ]
                    + [
                        outer_results[outer_id]["executor_state_sha256"]
                        for outer_id in OUTER_FOLDS
                    ]
                ),
                "inner_prediction_sha256": _stable_hash(
                    [
                        inner_results[outer_id]["prediction_sha256"]
                        for outer_id in OUTER_FOLDS
                    ]
                ),
                "outer_prediction_sha256": _stable_hash(
                    [
                        outer_results[outer_id]["prediction_sha256"]
                        for outer_id in OUTER_FOLDS
                    ]
                ),
                "inner_primary_delta_mean": float(
                    np.mean(
                        [
                            inner_effects[outer_id]["mean_delta"]
                            for outer_id in OUTER_FOLDS
                        ]
                    )
                ),
                "outer_primary_delta_mean": float(
                    np.mean(list(outer_deltas.values()))
                ),
                "inner_by_outer": [
                    {
                        "outer_rollout_id": outer_id,
                        "effect": inner_effects[outer_id],
                    }
                    for outer_id in OUTER_FOLDS
                ],
                "outer_by_outer": [
                    {
                        "outer_rollout_id": outer_id,
                        "primary_delta": outer_deltas[outer_id],
                        "prediction_sha256": outer_results[outer_id][
                            "prediction_sha256"
                        ],
                    }
                    for outer_id in OUTER_FOLDS
                ],
                "policy_visibility": (
                    "a0_reference_only"
                    if action_id == A0.action_id
                    else "normalized_inner_effect_only_after_action_execution"
                ),
                "oracle_results_visible_to_policy": False,
                "outer_results_visible_to_policy": False,
            }
        )
        if action_id != A0.action_id:
            for outer_id in OUTER_FOLDS:
                inner_delta = float(inner_effects[outer_id]["mean_delta"])
                outer_delta = float(outer_deltas[outer_id])
                transfer_pairs.append(
                    {
                        "action_id": action_id,
                        "outer_rollout_id": outer_id,
                        "inner_delta": inner_delta,
                        "outer_delta": outer_delta,
                        "inner_category": _effect_category(inner_delta),
                        "outer_category": _effect_category(outer_delta),
                        "sign_agreement": (
                            (inner_delta > 0 and outer_delta > 0)
                            or (inner_delta < 0 and outer_delta < 0)
                            or (inner_delta == 0 and outer_delta == 0)
                        ),
                        "used_for_policy_feedback": False,
                        "used_for_legal_promotion_selection": False,
                        "computed_after_policy_calls": True,
                    }
                )
    effect_flags = mark_action_effect_flags(entries)
    inner_values = np.asarray(
        [item["inner_delta"] for item in transfer_pairs], dtype=np.float64
    )
    outer_values = np.asarray(
        [item["outer_delta"] for item in transfer_pairs], dtype=np.float64
    )
    correlation = (
        float(np.corrcoef(inner_values, outer_values)[0, 1])
        if np.std(inner_values) > 0 and np.std(outer_values) > 0
        else None
    )
    inner_positive = [item for item in transfer_pairs if item["inner_delta"] > 0]
    transferred_positive = [item for item in inner_positive if item["outer_delta"] > 0]
    selected_pairs = []
    for rollout in a2l_rollouts:
        outer_id = int(rollout["outer_rollout_id"])
        action_id = str(rollout["selected_for_promotion"])
        if action_id == A0.action_id:
            inner_delta = 0.0
            outer_delta = 0.0
        else:
            pair = next(
                item
                for item in transfer_pairs
                if item["outer_rollout_id"] == outer_id
                and item["action_id"] == action_id
            )
            inner_delta = float(pair["inner_delta"])
            outer_delta = float(pair["outer_delta"])
        selected_pairs.append(
            {
                "outer_rollout_id": outer_id,
                "action_id": action_id,
                "inner_delta": inner_delta,
                "outer_delta": outer_delta,
            }
        )
    transfer = {
        "diagnostic_only": True,
        "computed_after_all_policy_calls": True,
        "used_for_policy_feedback": False,
        "used_for_legal_promotion_selection": False,
        "pair_count": len(transfer_pairs),
        "pearson_inner_outer_delta": correlation,
        "sign_agreement_count": int(
            sum(bool(item["sign_agreement"]) for item in transfer_pairs)
        ),
        "inner_positive_count": len(inner_positive),
        "inner_positive_that_remain_outer_positive": len(transferred_positive),
        "positive_transfer_rate": (
            len(transferred_positive) / len(inner_positive)
            if inner_positive
            else None
        ),
        "pairs": transfer_pairs,
        "a2l_selected_pairs": selected_pairs,
        "mean_a2l_selected_inner_delta": float(
            np.mean([item["inner_delta"] for item in selected_pairs])
        ),
        "mean_a2l_selected_outer_delta": float(
            np.mean([item["outer_delta"] for item in selected_pairs])
        ),
    }
    oracle_outer = []
    ceiling_deltas = []
    a2l_regrets = []
    a2l_by_outer = {
        int(item["outer_rollout_id"]): item for item in a2l_rollouts
    }
    for outer_id in OUTER_FOLDS:
        candidates = []
        for action_id in PILOT_ACTION_IDS:
            result = evaluator.inner(outer_id, action_id)
            effect = summarize_inner_effect(result, a0_inner[outer_id])
            candidates.append(
                {
                    "action_id": action_id,
                    "inner_delta": effect["mean_delta"],
                    "prediction_sha256": result["prediction_sha256"],
                    "config_sha256": result["config_sha256"],
                }
            )
        best = max(candidates, key=lambda item: float(item["inner_delta"]))
        ceiling_delta = max(0.0, float(best["inner_delta"]))
        selected_delta = float(a2l_by_outer[outer_id]["selected_inner_delta"])
        ceiling_deltas.append(ceiling_delta)
        a2l_regrets.append(ceiling_delta - selected_delta)
        oracle_outer.append(
            {
                "outer_rollout_id": outer_id,
                "actions": candidates,
                "best_reachable_action_id": (
                    str(best["action_id"])
                    if float(best["inner_delta"]) > 0
                    else A0.action_id
                ),
                "ceiling_delta": ceiling_delta,
                "a2l_selected_action_id": a2l_by_outer[outer_id][
                    "selected_for_promotion"
                ],
                "a2l_regret": ceiling_delta - selected_delta,
            }
        )
    oracle = {
        "layer": "inner_selection_only",
        "action_ids": list(PILOT_ACTION_IDS),
        "outer_rollout_count": len(OUTER_FOLDS),
        "model_seed_count": 1,
        "used_for_policy_feedback": False,
        "used_for_legal_promotion_selection": False,
        "outer_diagnostic_is_separate": True,
        "mean_inner_ceiling_delta_from_a0": float(np.mean(ceiling_deltas)),
        "mean_a2l_regret_to_inner_ceiling": float(np.mean(a2l_regrets)),
        "outer_rollouts": oracle_outer,
    }
    effects = {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "split_hash": EXPECTED_SPLIT_HASH,
        "model_repeat_mode": MODEL_REPEAT_MODE,
        "model_seeds": list(MODEL_SEEDS),
        "actions": entries,
        **effect_flags,
    }
    return effects, transfer, oracle, transfer_pairs


def _protocol_payload(batch_sha256: str) -> dict[str, Any]:
    action_payload = [_config_payload(action) for action in PILOT_ACTIONS]
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "primary_metric": PRIMARY_METRIC,
        "split_hash": EXPECTED_SPLIT_HASH,
        "fixed_class_schema": list(CLASS_NAMES),
        "a0": {
            "config": _config_payload(A0),
            "config_sha256": _stable_hash(_config_payload(A0)),
            "expected_mean": EXPECTED_A0_MEAN,
        },
        "pilot_actions": action_payload,
        "action_table_sha256": _stable_hash(action_payload),
        "pilot_action_count": len(PILOT_ACTION_IDS),
        "trial_budget_per_outer_rollout": TRIAL_BUDGET,
        "model_repeat_mode": MODEL_REPEAT_MODE,
        "model_seeds": list(MODEL_SEEDS),
        "uncertainty_unit": UNCERTAINTY_UNIT,
        "effect_unit_primary_delta": EFFECT_UNIT,
        "observation": {
            "schema": OBSERVATION_SCHEMA,
            "allowed": [
                "bounded_normalized_inner_delta",
                "bounded_inner_fold_uncertainty",
                "anonymous_per_class_train_support_share",
                "inner_fold_win_flat_loss",
                "remaining_budget",
                "advance_threshold_in_normalized_units",
            ],
            "denied": list(FORBIDDEN_POLICY_KEYS),
            "outer_result_exposed": False,
        },
        "prompt_information_ablation": {
            "arm": "A2L_CATEGORICAL_ABLATION",
            "same_actions": True,
            "same_trial_budget": True,
            "same_model_seed": True,
            "difference": "remove_normalized_effect_support_and_uncertainty",
        },
        "promotion_rule": "mean_delta_ge_0.005_and_at_least_2_of_3_inner_folds_improved",
        "outer_transfer_diagnostic": {
            "computed_after_all_policy_calls": True,
            "used_for_policy_feedback": False,
            "used_for_legal_promotion_selection": False,
        },
        "deepseek": {
            "endpoint": DEEPSEEK_ENDPOINT,
            "model": DEEPSEEK_MODEL,
            "strict_json": True,
            "missing_or_invalid_provider": "fail_closed",
        },
        "nested_batch_sha256": batch_sha256,
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
    }


def _arm_summary(
    rollouts: Sequence[Mapping[str, Any]],
    promotion: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "rollouts": list(rollouts),
        "normalized_auc_at_3": float(
            np.mean([item["normalized_auc_at_3"] for item in rollouts])
        ),
        "promotion": dict(promotion),
    }


def _make_root_cause(summary: Mapping[str, Any]) -> str:
    effect = summary["action_effects"]
    transfer = summary["inner_to_outer_transfer"]
    return "\n".join(
        [
            "# P29 lithofacies root-cause audit",
            "",
            "## Conclusion",
            "",
            (
                "P28 connected actions to XGBoost predictions and the fixed-schema "
                "metric, but its categorical-only observation discarded effect "
                "magnitude and uncertainty. Its three model seeds were also "
                "deterministic duplicates. P29 repairs both defects without changing "
                "the split or primary metric."
            ),
            "",
            "## Causal-chain audit",
            "",
            "| stage | P28 finding | P29 evidence |",
            "|---|---|---|",
            "| observation | Only support buckets and categorical feedback were visible. | Bounded normalized deltas, anonymous class-support shares, and three-inner-fold uncertainty are visible. |",
            "| prompt | Numeric effect size and confidence were removed. | The live enhanced prompt receives effect units and an equal-budget categorical ablation is retained. |",
            "| selected action | Live strict-JSON actions were valid. | Every live decision remains strict JSON and without replacement. |",
            f"| executor | Actions changed XGBoost configuration. | All non-A0 config hashes differ: `{effect['all_nonbaseline_configs_differ']}`. |",
            f"| prediction | P28 actions changed predictions, but duplicated seed hashes were counted as repeats. | One seed is declared; every non-A0 action changes an inner or outer prediction hash: `{effect['all_nonbaseline_actions_change_prediction']}`. |",
            "| metric | Fixed-nine Macro-F1 was correctly computed. | The same primary metric is preserved. |",
            "| promotion | Inner LOGO3 selected actions for a disjoint outer fold. | The same nested split is preserved; the robust inner rule is frozen before evaluation. |",
            f"| endpoint | P28 outer improvement failed. | Post-policy transfer correlation is `{transfer['pearson_inner_outer_delta']}` and is diagnostic only. |",
            "",
            "## Leakage firewall",
            "",
            "The live policy sees no raw metric, row-level target, class name, group identity, sample identifier, path, residual, or outer result. All outer action diagnostics are computed after policy calls and are never used for legal selection.",
            "",
        ]
    )


def _make_evidence(summary: Mapping[str, Any]) -> str:
    a2l = summary["arms"]["A2L"]
    ablation = summary["arms"]["A2L_CATEGORICAL_ABLATION"]
    a2d = summary["arms"]["A2D"]
    a3 = summary["arms"]["A3"]
    transfer = summary["inner_to_outer_transfer"]
    oracle = summary["oracle_inner_ceiling"]
    lines = [
        "# P29 lithofacies agent-action effect evidence",
        "",
        "## Outcome",
        "",
        (
            f"The bounded development verdict is **{summary['verdict']}**. A0 "
            f"reproduced `{summary['a0']['observed_mean']:.10f}` fixed-schema "
            f"Macro-F1. Enhanced A2L produced "
            f"`{a2l['promotion']['fixed_schema_macro_f1_mean']:.10f}` "
            f"(`{a2l['promotion']['delta_from_a0']:+.10f}`)."
        ),
        "",
        "This is adaptive development evidence and does not use a frozen holdout.",
        "",
        "## Repaired observation and repeat contract",
        "",
        "- The policy sees clipped deltas in 0.005 effect units, anonymous per-class train-support shares, and uncertainty from three disjoint inner LOGO folds.",
        "- Raw metrics, labels, class names, group identities, sample identifiers, paths, residuals, and outer results remain hidden.",
        f"- The executor uses exactly one model seed `{MODEL_SEED}`. No duplicated deterministic seed is described as a replicate.",
        "- The primary metric and split hash are unchanged.",
        "",
        "## Arms",
        "",
        "| arm | observation | normalized AUC@3 | outer mean | delta vs A0 | positive folds |",
        "|---|---|---:|---:|---:|---:|",
        f"| A2L | safe normalized | {a2l['normalized_auc_at_3']:.10f} | {a2l['promotion']['fixed_schema_macro_f1_mean']:.10f} | {a2l['promotion']['delta_from_a0']:+.10f} | {a2l['promotion']['positive_outer_folds']}/4 |",
        f"| A2L categorical ablation | categorical only | {ablation['normalized_auc_at_3']:.10f} | {ablation['promotion']['fixed_schema_macro_f1_mean']:.10f} | {ablation['promotion']['delta_from_a0']:+.10f} | {ablation['promotion']['positive_outer_folds']}/4 |",
        f"| A2D | deterministic enhanced | {a2d['normalized_auc_at_3']:.10f} | {a2d['promotion']['fixed_schema_macro_f1_mean']:.10f} | {a2d['promotion']['delta_from_a0']:+.10f} | {a2d['promotion']['positive_outer_folds']}/4 |",
        f"| A3 | random median | {a3['normalized_auc_median']:.10f} | {a3['promotion_median_mean']:.10f} | {a3['promotion_median_delta']:+.10f} | n/a |",
        "",
        "## Action effects and transfer",
        "",
        f"All four non-baseline actions have different config hashes: **{summary['action_effects']['all_nonbaseline_configs_differ']}**.",
        "",
        f"All four change an inner or outer prediction hash: **{summary['action_effects']['all_nonbaseline_actions_change_prediction']}**.",
        "",
        f"The exhaustive inner ceiling is `{oracle['mean_inner_ceiling_delta_from_a0']:+.10f}`. A2L regret to that ceiling is `{oracle['mean_a2l_regret_to_inner_ceiling']:+.10f}`.",
        "",
        f"Across {transfer['pair_count']} action-by-outer pairs, inner-to-outer delta correlation is `{transfer['pearson_inner_outer_delta']}`. {transfer['inner_positive_that_remain_outer_positive']} of {transfer['inner_positive_count']} inner-positive pairs remain outer-positive.",
        "",
        "The transfer matrix was computed only after all live decisions. It was not shown to a policy and did not select any legal endpoint.",
        "",
        "## A2L outer results",
        "",
        "| outer fold | selected action | inner delta | outer delta |",
        "|---:|---|---:|---:|",
    ]
    for rollout, outer_delta in zip(
        a2l["rollouts"], a2l["promotion"]["outer_fold_deltas"]
    ):
        lines.append(
            f"| {rollout['outer_rollout_id']} | {rollout['selected_for_promotion']} | "
            f"{rollout['selected_inner_delta']:+.10f} | {outer_delta:+.10f} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
        ]
    )
    for gate, passed in summary["gates"].items():
        lines.append(f"- `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "No threshold, split, action, or result was changed after observing the pilot. The model remains disabled by default unless the preregistered retention gates pass.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    *,
    output_dir: Path,
    batch_file: Path,
    protocol: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    action_effects: Mapping[str, Any],
) -> None:
    protocol_path = output_dir / "protocol.json"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    effects_path = output_dir / "action_effects.json"
    evidence_path = output_dir / "evidence.md"
    root_cause_path = output_dir / "root_cause.md"
    _write_json(protocol_path, protocol)
    _write_jsonl(results_path, results)
    _write_json(summary_path, summary)
    _write_json(effects_path, action_effects)
    _write_text(evidence_path, _make_evidence(summary))
    _write_text(root_cause_path, _make_root_cause(summary))
    test_path = TRACK_DIR / "tests" / "test_p29_agent_action_effect.py"
    artifact_paths = (
        protocol_path,
        results_path,
        summary_path,
        effects_path,
        evidence_path,
        root_cause_path,
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
        "sources": {
            Path(__file__).name: _sha256(Path(__file__)),
            str(test_path.relative_to(TRACK_DIR)): _sha256(test_path),
            "lithofacies_p28_agentic_optimization.py": _sha256(P28_RUNNER),
            "p28_summary.json": _sha256(P28_SUMMARY),
            "nested_development.npz": _sha256(batch_file),
        },
        "primary_metric": PRIMARY_METRIC,
        "split_hash": EXPECTED_SPLIT_HASH,
        "verdict": summary["verdict"],
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "credential_persisted": False,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)


def run_pilot(
    *,
    batch_file: Path = DEFAULT_BATCH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    api_key: str,
) -> dict[str, Any]:
    ensure_development_only_paths((batch_file, output_dir))
    output_dir = _owned_output(output_dir)
    if not api_key.strip():
        raise CredentialUnavailable("DEEPSEEK_KEY is missing; P29 fails closed")
    arrays, batch_manifest = load_nested_batch(batch_file)
    started = time.perf_counter()
    protocol = _protocol_payload(_sha256(batch_file))
    evaluator = ActionEvaluator(arrays)
    a0_inner = {
        outer_id: evaluator.inner(outer_id, A0.action_id)
        for outer_id in OUTER_FOLDS
    }
    a0_outer = {
        outer_id: evaluator.outer(outer_id, A0.action_id)
        for outer_id in OUTER_FOLDS
    }
    a0_outer_values = [
        a0_outer[outer_id]["metrics"][PRIMARY_METRIC] for outer_id in OUTER_FOLDS
    ]
    observed_a0 = float(np.mean(a0_outer_values))
    if not math.isclose(observed_a0, EXPECTED_A0_MEAN, abs_tol=5e-10, rel_tol=0.0):
        raise RuntimeError("P29 A0 mean changed under the one-seed contract")
    if any(
        not math.isclose(value, expected, abs_tol=5e-10, rel_tol=0.0)
        for value, expected in zip(a0_outer_values, EXPECTED_A0_FOLDS)
    ):
        raise RuntimeError("P29 A0 outer fold metric changed")

    # All live decisions finish before any non-A0 outer diagnostic is computed.
    a2l_rollouts = [
        _rollout(
            arm="A2L",
            instance_id="deepseek_safe_normalized",
            outer_id=outer_id,
            evaluator=evaluator,
            baseline=a0_inner[outer_id],
            api_key=api_key,
        )
        for outer_id in OUTER_FOLDS
    ]
    categorical_rollouts = [
        _rollout(
            arm="A2L_CATEGORICAL_ABLATION",
            instance_id="deepseek_categorical_ablation",
            outer_id=outer_id,
            evaluator=evaluator,
            baseline=a0_inner[outer_id],
            api_key=api_key,
        )
        for outer_id in OUTER_FOLDS
    ]
    a2d_rollouts = [
        _rollout(
            arm="A2D",
            instance_id="deterministic_safe_normalized",
            outer_id=outer_id,
            evaluator=evaluator,
            baseline=a0_inner[outer_id],
            api_key="",
        )
        for outer_id in OUTER_FOLDS
    ]
    a3_instances: dict[str, list[dict[str, Any]]] = {}
    for policy_seed in POLICY_SEEDS:
        instance_id = f"random_seed_{policy_seed}"
        a3_instances[instance_id] = [
            _rollout(
                arm="A3",
                instance_id=instance_id,
                outer_id=outer_id,
                evaluator=evaluator,
                baseline=a0_inner[outer_id],
                api_key="",
                random_seed=policy_seed,
            )
            for outer_id in OUTER_FOLDS
        ]

    a1, a1_cells = _actual_a1_replay(arrays, a0_outer)
    a2l_promotion = _promotion_summary(a2l_rollouts, evaluator, a0_outer)
    categorical_promotion = _promotion_summary(
        categorical_rollouts, evaluator, a0_outer
    )
    a2d_promotion = _promotion_summary(a2d_rollouts, evaluator, a0_outer)
    a3_summaries = []
    for instance_id, rollouts in a3_instances.items():
        a3_summaries.append(
            {
                "instance_id": instance_id,
                "rollouts": rollouts,
                "normalized_auc_at_3": float(
                    np.mean([item["normalized_auc_at_3"] for item in rollouts])
                ),
                "promotion": _promotion_summary(rollouts, evaluator, a0_outer),
            }
        )

    action_effects, transfer, oracle, transfer_rows = _action_effects_and_transfer(
        evaluator=evaluator,
        a0_inner=a0_inner,
        a0_outer=a0_outer,
        a2l_rollouts=a2l_rollouts,
    )
    a2l_auc = float(
        np.mean([item["normalized_auc_at_3"] for item in a2l_rollouts])
    )
    categorical_auc = float(
        np.mean(
            [item["normalized_auc_at_3"] for item in categorical_rollouts]
        )
    )
    a2d_auc = float(
        np.mean([item["normalized_auc_at_3"] for item in a2d_rollouts])
    )
    a3_auc_values = [item["normalized_auc_at_3"] for item in a3_summaries]
    a3_promotion_means = [
        item["promotion"]["fixed_schema_macro_f1_mean"] for item in a3_summaries
    ]
    a3_promotion_deltas = [
        item["promotion"]["delta_from_a0"] for item in a3_summaries
    ]
    live_trials = [
        trial
        for rollout in a2l_rollouts + categorical_rollouts
        for trial in rollout["trials"]
    ]
    enhanced_observations = [
        trial["policy_observation"]
        for rollout in a2l_rollouts
        for trial in rollout["trials"]
    ]
    for observation in enhanced_observations:
        assert_policy_payload_safe(observation)
        if observation["observation_schema"] != OBSERVATION_SCHEMA:
            raise RuntimeError("enhanced live policy received the wrong observation")
    gates = {
        "a1_identity_replay": bool(a1["all_equal"]),
        "all_live_actions_valid": all(trial["valid_action"] for trial in live_trials),
        "safe_observation_firewall": True,
        "single_seed_no_pseudo_replicates": MODEL_SEEDS == (MODEL_SEED,),
        "all_actions_change_prediction": bool(
            action_effects["all_nonbaseline_actions_change_prediction"]
        ),
        "a2l_auc_above_categorical_ablation": a2l_auc > categorical_auc,
        "a2l_auc_above_a2d": a2l_auc > a2d_auc,
        "a2l_auc_above_a3_median": a2l_auc > float(np.median(a3_auc_values)),
        "a2l_outer_mean_delta_at_least_0_005": a2l_promotion["delta_from_a0"]
        >= EFFECT_UNIT,
        "a2l_outer_positive_on_at_least_3_folds": a2l_promotion[
            "positive_outer_folds"
        ]
        >= 3,
        "no_non_finite_class_metric": bool(
            a2l_promotion["all_class_metrics_finite"]
        ),
    }
    infrastructure_gates = all(
        gates[key]
        for key in (
            "a1_identity_replay",
            "all_live_actions_valid",
            "safe_observation_firewall",
            "single_seed_no_pseudo_replicates",
            "all_actions_change_prediction",
            "no_non_finite_class_metric",
        )
    )
    agent_gates = all(
        gates[key]
        for key in (
            "a2l_auc_above_categorical_ablation",
            "a2l_auc_above_a2d",
            "a2l_auc_above_a3_median",
            "a2l_outer_mean_delta_at_least_0_005",
            "a2l_outer_positive_on_at_least_3_folds",
        )
    )
    a2d_hybrid_gate = (
        a2d_promotion["delta_from_a0"] >= EFFECT_UNIT
        and a2d_promotion["positive_outer_folds"] >= 3
    )
    if infrastructure_gates and agent_gates:
        verdict = "RETAIN_AGENT"
    elif infrastructure_gates and a2d_hybrid_gate:
        verdict = "RETAIN_HYBRID"
    else:
        verdict = "REJECT_AGENT"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "split_hash": EXPECTED_SPLIT_HASH,
        "a0": {
            "observed_mean": observed_a0,
            "outer_fold_values": a0_outer_values,
            "config_sha256": _stable_hash(_config_payload(A0)),
            "prediction_sha256": _stable_hash(
                [a0_outer[outer_id]["prediction_sha256"] for outer_id in OUTER_FOLDS]
            ),
        },
        "arms": {
            "A1": a1,
            "A2L": _arm_summary(a2l_rollouts, a2l_promotion),
            "A2L_CATEGORICAL_ABLATION": _arm_summary(
                categorical_rollouts, categorical_promotion
            ),
            "A2D": _arm_summary(a2d_rollouts, a2d_promotion),
            "A3": {
                "instances": a3_summaries,
                "normalized_auc_values": a3_auc_values,
                "normalized_auc_median": float(np.median(a3_auc_values)),
                "promotion_median_mean": float(np.median(a3_promotion_means)),
                "promotion_median_delta": float(np.median(a3_promotion_deltas)),
            },
            "ORACLE": oracle,
        },
        "action_effects": {
            "all_nonbaseline_configs_differ": action_effects[
                "all_nonbaseline_configs_differ"
            ],
            "all_nonbaseline_actions_change_prediction": action_effects[
                "all_nonbaseline_actions_change_prediction"
            ],
            "action_effects_sha256": _stable_hash(action_effects),
        },
        "inner_to_outer_transfer": transfer,
        "oracle_inner_ceiling": oracle,
        "prompt_information_ablation": {
            "enhanced_normalized_auc_at_3": a2l_auc,
            "categorical_normalized_auc_at_3": categorical_auc,
            "delta": a2l_auc - categorical_auc,
            "same_budget": True,
            "same_action_table": True,
            "same_model_seed": True,
        },
        "model_repeat_contract": {
            "mode": MODEL_REPEAT_MODE,
            "seeds": list(MODEL_SEEDS),
            "seed_count": 1,
            "uncertainty_unit": UNCERTAINTY_UNIT,
            "pseudo_replicate_claimed": False,
        },
        "gates": gates,
        "verdict": verdict,
        "default_enabled": False,
        "provider": {
            "enhanced_live_call_count": len(a2l_rollouts) * TRIAL_BUDGET,
            "categorical_ablation_live_call_count": len(categorical_rollouts)
            * TRIAL_BUDGET,
            "valid_response_count": sum(
                trial["provider"] is not None and trial["provider"]["valid"]
                for trial in live_trials
            ),
            "credential_persisted": False,
        },
        "data": {
            "nested_batch_sha256": _sha256(batch_file),
            "development_hdf5_sha256": batch_manifest[
                "development_hdf5_sha256"
            ],
            "loaded_files_this_run": ["nested_development.npz"],
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
            "test_metrics_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    results: list[dict[str, Any]] = []
    for outer_id in OUTER_FOLDS:
        results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "a0_outer_cell",
                "arm": "A0",
                **a0_outer[outer_id],
            }
        )
    for cell in a1_cells:
        results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "a1_actual_replay_outer_cell",
                "arm": "A1",
                **cell,
            }
        )
    rollout_groups = [
        ("A2L", [a2l_rollouts]),
        ("A2L_CATEGORICAL_ABLATION", [categorical_rollouts]),
        ("A2D", [a2d_rollouts]),
        ("A3", list(a3_instances.values())),
    ]
    for arm, groups in rollout_groups:
        for group in groups:
            for rollout in group:
                for trial in rollout["trials"]:
                    results.append(
                        {
                            "schema_version": RESULT_SCHEMA,
                            "record_type": "selection_trial",
                            "arm": arm,
                            "instance_id": rollout["instance_id"],
                            "outer_rollout_id": rollout["outer_rollout_id"],
                            **trial,
                        }
                    )
    for row in transfer_rows:
        results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "post_policy_transfer_diagnostic",
                "arm": "TRANSFER_DIAGNOSTIC",
                **row,
            }
        )
    _write_artifacts(
        output_dir=output_dir,
        batch_file=batch_file,
        protocol=protocol,
        results=results,
        summary=summary,
        action_effects=action_effects,
    )
    return summary


def verify_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = _owned_output(output_dir)
    manifest = json.loads(
        (output_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unknown P29 artifact manifest schema")
    verified = {}
    for name, expected in manifest["artifacts"].items():
        path = output_dir / name
        digest = _sha256(path)
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"P29 artifact verification failed: {name}")
        verified[name] = digest
    test_path = TRACK_DIR / "tests" / "test_p29_agent_action_effect.py"
    source_paths = {
        Path(__file__).name: Path(__file__),
        str(test_path.relative_to(TRACK_DIR)): test_path,
        "lithofacies_p28_agentic_optimization.py": P28_RUNNER,
        "p28_summary.json": P28_SUMMARY,
        "nested_development.npz": DEFAULT_BATCH,
    }
    for name, path in source_paths.items():
        if _sha256(path) != manifest["sources"][name]:
            raise RuntimeError(f"P29 source verification failed: {name}")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    effects = json.loads(
        (output_dir / "action_effects.json").read_text(encoding="utf-8")
    )
    results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if summary["split_hash"] != EXPECTED_SPLIT_HASH or protocol[
        "split_hash"
    ] != EXPECTED_SPLIT_HASH:
        raise RuntimeError("P29 split hash changed")
    if summary["primary_metric"] != PRIMARY_METRIC:
        raise RuntimeError("P29 primary metric changed")
    repeat = summary["model_repeat_contract"]
    if (
        repeat["seed_count"] != 1
        or repeat["seeds"] != [MODEL_SEED]
        or repeat["pseudo_replicate_claimed"] is not False
    ):
        raise RuntimeError("P29 pseudo-replicate repair is not explicit")
    a1 = summary["arms"]["A1"]
    if not a1["actual_same_entrypoint_replay"] or not a1["all_equal"]:
        raise RuntimeError("P29 A1 identity replay failed")
    if len(effects["actions"]) != 1 + len(PILOT_ACTION_IDS):
        raise RuntimeError("P29 action-effect roster changed")
    if not effects["all_nonbaseline_configs_differ"]:
        raise RuntimeError("P29 contains a config no-op action")
    if not effects["all_nonbaseline_actions_change_prediction"]:
        raise RuntimeError("P29 contains an endpoint no-op action")
    live_trials = [
        row
        for row in results
        if row.get("record_type") == "selection_trial"
        and row.get("arm") in {"A2L", "A2L_CATEGORICAL_ABLATION"}
    ]
    expected_live = 2 * len(OUTER_FOLDS) * TRIAL_BUDGET
    if len(live_trials) != expected_live:
        raise RuntimeError("P29 live prompt ablation budget changed")
    for row in live_trials:
        assert_policy_payload_safe(row["policy_observation"])
        if row["provider"] is None or row["provider"]["valid"] is not True:
            raise RuntimeError("P29 persisted an invalid live decision")
        if row["outer_result_exposed_to_policy"] is not False:
            raise RuntimeError("P29 outer result leaked into policy")
    enhanced = [row for row in live_trials if row["arm"] == "A2L"]
    if any(
        row["policy_observation"]["observation_schema"] != OBSERVATION_SCHEMA
        or "class_support" not in row["policy_observation"]
        for row in enhanced
    ):
        raise RuntimeError("P29 enhanced observations are incomplete")
    transfer_rows = [
        row
        for row in results
        if row.get("record_type") == "post_policy_transfer_diagnostic"
    ]
    if len(transfer_rows) != len(PILOT_ACTION_IDS) * len(OUTER_FOLDS):
        raise RuntimeError("P29 transfer matrix is incomplete")
    if any(
        row["computed_after_policy_calls"] is not True
        or row["used_for_policy_feedback"] is not False
        or row["used_for_legal_promotion_selection"] is not False
        for row in transfer_rows
    ):
        raise RuntimeError("P29 transfer diagnostic crossed the selection firewall")
    if summary["data"]["frozen_test_accessed"] is not False:
        raise RuntimeError("P29 summary violates the frozen-test firewall")
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.glob("*")
        if path.is_file()
    )
    credential = os.environ.get("DEEPSEEK_KEY", "")
    if credential and credential in serialized:
        raise RuntimeError("DeepSeek credential leaked into P29 artifacts")
    return {
        "verified_artifacts": verified,
        "verdict": summary["verdict"],
        "primary_metric": summary["primary_metric"],
        "split_hash": summary["split_hash"],
        "model_seed_count": repeat["seed_count"],
        "live_valid_action_count": len(live_trials),
        "action_effect_count": len(effects["actions"]),
        "transfer_pair_count": len(transfer_rows),
        "frozen_test_accessed": False,
        "credential_persisted": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--batch-file", type=Path, default=DEFAULT_BATCH)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run":
        payload = run_pilot(
            batch_file=args.batch_file,
            output_dir=args.output_dir,
            api_key=os.environ.get("DEEPSEEK_KEY", ""),
        )
    else:
        payload = verify_artifacts(args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
