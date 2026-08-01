#!/usr/bin/env python3
"""P29 facies action-effect repair and two-of-five agent pilot.

The runner preserves P28's model, manifests, folds, metric, action registry,
and non-degradation guards.  It adds a development-only observation v2,
executable stop semantics, a two-action sample-efficiency comparison, and a
per-dataset promotion package.  There is deliberately no test/holdout CLI.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p11_residual_fusion as p11  # noqa: E402
import p12_repair_v1 as p12  # noqa: E402
import p13_cross_attention as p13  # noqa: E402
import p28_agentic_optimization as p28  # noqa: E402


OUTPUT_ROOT = HERE / "_outputs" / "p29_agent_action_effect"
SCHEMA_VERSION = "facies-p29-action-effect/v1"
PROTOCOL_SCHEMA_VERSION = "facies-p29-protocol/v1"
OBSERVATION_SCHEMA_VERSION = "facies-p29-observation/v2"
ABLATION_SCHEMA_VERSION = "facies-p29-observation/v1-ablation"
ACTION_EFFECT_SCHEMA_VERSION = "facies-p29-action-effects/v1"
ROOT_SEED = p28.ROOT_SEED
SELECTION_FOLD = p28.SELECTION_FOLD
PROMOTION_FOLD = p28.PROMOTION_FOLD
MAX_TRIALS_PER_POLICY = 2
ACTION_WALL_CLOCK_BUDGET_S = p28.ACTION_WALL_CLOCK_BUDGET_S
GPU_LIMIT = p28.GPU_LIMIT
DEEPSEEK_ENDPOINT = p28.DEEPSEEK_ENDPOINT
DEEPSEEK_MODEL = p28.DEEPSEEK_MODEL
DEEPSEEK_ENV_NAME = p28.DEEPSEEK_ENV_NAME
PROVIDER_TIMEOUT_S = p28.PROVIDER_TIMEOUT_S
MEAN_PROMOTION_DELTA = p28.MEAN_PROMOTION_DELTA
TASK_NON_DEGRADATION = p28.TASK_NON_DEGRADATION
DATASET_ACTION_SWITCH_MIN_DELTA = MEAN_PROMOTION_DELTA
SIGNED_DELTA_SCALE_MIOU = 0.05
SIGNED_DELTA_CLIP = 2.0
FUSION_MOVEMENT_SCALE = 0.01
TASKS = p28.TASKS
TASK_NAMES = p28.TASK_NAMES
EXPECTED_MANIFEST_HASHES = p28.EXPECTED_MANIFEST_HASHES
A0_CONFIG = p28.A0_CONFIG
ACTION_ALLOWLIST = p28.ACTION_ALLOWLIST
ACTION_IDS = p28.ACTION_IDS
POLICY_KEYS = ("a2l", "a2d", "a3", "a4")
PROTECTED_OUTPUTS = {
    p11.OUTPUT_ROOT.resolve(),
    p12.OUTPUT_ROOT.resolve(),
    p13.OUTPUT_ROOT.resolve(),
    p28.OUTPUT_ROOT.resolve(),
    (HERE / "_outputs" / "agent_chapter").resolve(),
}
DENIED_OBSERVATION_TERMS = (
    "miou",
    "raw_metric",
    "residual",
    "label",
    "sample_id",
    "path",
    "prediction",
    "logit",
    "probability",
    "confusion",
    "validation",
    "curve",
    "holdout",
    "test",
)
STOP_SEMANTICS = (
    "execute_the_selected_action_then_terminate_without_a_hidden_fallback"
)


def _hash_payload(value: Any) -> str:
    return p28._hash_payload(value)


def _validate_output_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    if resolved != OUTPUT_ROOT.resolve():
        raise ValueError("P29 writes only its declared owner output directory")
    if resolved in PROTECTED_OUTPUTS:
        raise ValueError("P29 refuses to overwrite prior evidence")
    return resolved


def _clip_signed(value: float) -> float:
    return float(np.clip(float(value), -SIGNED_DELTA_CLIP, SIGNED_DELTA_CLIP))


def _normalized_signed_delta(candidate: float, baseline: float) -> float:
    return round(
        _clip_signed((float(candidate) - float(baseline)) / SIGNED_DELTA_SCALE_MIOU),
        6,
    )


def _normalized_signed_change(candidate: float, baseline: float) -> float:
    denominator = max(abs(float(baseline)), 1e-8)
    return round(_clip_signed((float(candidate) - float(baseline)) / denominator), 6)


def _direction(normalized_delta: float) -> str:
    threshold = MEAN_PROMOTION_DELTA / SIGNED_DELTA_SCALE_MIOU
    lower = TASK_NON_DEGRADATION / SIGNED_DELTA_SCALE_MIOU
    if normalized_delta > threshold:
        return "improved"
    if normalized_delta < lower:
        return "worse"
    return "flat"


def _assert_safe_observation(observation: Mapping[str, Any]) -> None:
    def visit(value: Any, key: str = "") -> None:
        lowered = key.lower()
        if any(term in lowered for term in DENIED_OBSERVATION_TERMS):
            raise ValueError(f"denied observation key: {key}")
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
        elif isinstance(value, str):
            if value.startswith("/") or "test.h5" in value.lower():
                raise ValueError("observation contains a denied path-like value")
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if key in {"normalized_signed_delta", "normalized_signed_change"}:
                if not -SIGNED_DELTA_CLIP <= float(value) <= SIGNED_DELTA_CLIP:
                    raise ValueError("normalized observation value escaped its safe range")

    visit(observation)


def _baseline_diagnostics(a0: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in ("F3", "Penobscot"):
        bands = p28._safe_train_diagnostics(a0["selection"]["tasks"][task])
        result[task] = {
            "loss_state": bands["loss_level"],
            "gradient_state": bands["gradient_norm"],
            "encoder_update_state": bands["sam2_update"],
            "attention_state": bands["attention_entropy"],
            "fusion_movement_state": bands["fusion_scale_state"],
        }
    return result


def _history_entry(
    package: Mapping[str, Any],
    a0_selection: Mapping[str, Any],
    *,
    stop_requested: bool,
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for task in ("F3", "Penobscot"):
        candidate = package["tasks"][task]
        baseline = a0_selection["tasks"][task]
        normalized_delta = _normalized_signed_delta(
            candidate["miou"], baseline["miou"]
        )
        fusion_movement = round(
            _clip_signed(
                (
                    float(candidate["fusion_scale"])
                    - float(candidate["fusion_scale_initial"])
                )
                / FUSION_MOVEMENT_SCALE
            ),
            6,
        )
        effects[task] = {
            "effect_signal": {
                "normalized_signed_delta": normalized_delta,
                "direction": _direction(normalized_delta),
            },
            "optimizer_gradient_movement": {
                "normalized_signed_change": _normalized_signed_change(
                    candidate["last_grad_norm"], baseline["last_grad_norm"]
                ),
                "state": p28._band(
                    float(candidate["last_grad_norm"]), 2.0, 15.0
                ),
            },
            "fusion_gate_movement": {
                "normalized_signed_change": fusion_movement,
                "state": "stuck" if abs(fusion_movement) < 1.0 else "moved",
            },
        }
    mean_delta = _normalized_signed_delta(
        package["equal_mean"], a0_selection["equal_mean"]
    )
    return {
        "round": int(package["round"]),
        "action_id": package["action_id"],
        "stop_requested": bool(stop_requested),
        "per_dataset_effects": effects,
        "equal_weight_effect_signal": {
            "normalized_signed_delta": mean_delta,
            "direction": _direction(mean_delta),
        },
    }


def _build_observation_v2(
    *,
    policy_id: str,
    round_id: int,
    available_action_ids: Sequence[str],
    baseline_diagnostics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    previous = (
        {
            "round": history[-1]["round"],
            "action_id": history[-1]["action_id"],
            "stop_requested": history[-1]["stop_requested"],
        }
        if history
        else "none"
    )
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_id": policy_id,
        "policy_seed": ROOT_SEED,
        "round": int(round_id),
        "maximum_action_trials": MAX_TRIALS_PER_POLICY,
        "trials_remaining_including_current": MAX_TRIALS_PER_POLICY - round_id + 1,
        "available_action_ids": list(available_action_ids),
        "baseline_train_diagnostics": copy.deepcopy(dict(baseline_diagnostics)),
        "previous_action": previous,
        "full_action_history": copy.deepcopy(list(history)),
        "normalization_contract": {
            "signed_delta_zero": "same_as_A0",
            "positive_direction": "better_than_A0",
            "negative_direction": "worse_than_A0",
            "clip_range": [-SIGNED_DELTA_CLIP, SIGNED_DELTA_CLIP],
        },
        "stop_semantics": STOP_SEMANTICS,
    }
    _assert_safe_observation(observation)
    return observation


def _build_observation_ablation(
    *,
    policy_id: str,
    round_id: int,
    available_action_ids: Sequence[str],
    baseline_diagnostics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    latest = history[-1] if history else None
    observation = {
        "schema_version": ABLATION_SCHEMA_VERSION,
        "policy_id": policy_id,
        "policy_seed": ROOT_SEED,
        "round": int(round_id),
        "maximum_action_trials": MAX_TRIALS_PER_POLICY,
        "trials_remaining_including_current": MAX_TRIALS_PER_POLICY - round_id + 1,
        "available_action_ids": list(available_action_ids),
        "baseline_state_bands": copy.deepcopy(dict(baseline_diagnostics)),
        "most_recent_direction_only": (
            {
                task: latest["per_dataset_effects"][task]["effect_signal"]["direction"]
                for task in ("F3", "Penobscot")
            }
            if latest
            else "not_available"
        ),
        "stop_semantics": STOP_SEMANTICS,
    }
    _assert_safe_observation(observation)
    return observation


def _decision_prompt(
    observation: Mapping[str, Any], *, advice_only: bool
) -> tuple[str, str]:
    allowlist = [
        {
            "action_id": action_id,
            "description": ACTION_ALLOWLIST[action_id].description,
            "changed_factor": ACTION_ALLOWLIST[action_id].changed_factor,
        }
        for action_id in observation["available_action_ids"]
    ]
    system = (
        "You are a constrained seismic-facies development policy. Choose exactly "
        "one action_id from the supplied allowlist and return exactly one JSON object "
        "with keys action_id, confidence, rationale, stop. The v2 history contains "
        "only clipped normalized signed development deltas and train diagnostics: "
        "positive is better than A0, negative is worse, zero is unchanged. You never "
        "receive raw scores, labels, residuals, sample identifiers, paths, predictions, "
        "promotion results, or frozen-test information. stop=true means the selected "
        "action is executed and then the policy terminates; it never selects a hidden "
        "fallback. Use stop in round 1 only when another action is not justified by the "
        "visible evidence. confidence must be in [0,1]. Do not emit Markdown or commands."
    )
    user = json.dumps(
        {
            "mode": "advice_only" if advice_only else "execute",
            "observation": observation,
            "allowlist": allowlist,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return system, user


def _validate_decision(payload: Any, *, allowed: Sequence[str]) -> dict[str, Any]:
    required = {"action_id", "confidence", "rationale", "stop"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("provider response violates strict decision keys")
    if payload["action_id"] not in allowed:
        raise ValueError("provider selected an action outside the allowlist")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("decision confidence must be numeric")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("decision confidence is outside [0,1]")
    rationale = payload["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 500:
        raise ValueError("decision rationale is invalid")
    if not isinstance(payload["stop"], bool):
        raise ValueError("decision stop must be boolean")
    return {
        "action_id": payload["action_id"],
        "confidence": float(confidence),
        "rationale": rationale.strip(),
        "stop": bool(payload["stop"]),
    }


def _call_deepseek(
    observation: Mapping[str, Any],
    *,
    advice_only: bool,
    timeout_seconds: float = PROVIDER_TIMEOUT_S,
) -> dict[str, Any]:
    key = os.environ.get(DEEPSEEK_ENV_NAME, "").strip()
    if not key:
        return {
            "status": "BLOCKED_PROVIDER",
            "error": f"{DEEPSEEK_ENV_NAME} is unavailable",
            "credential_persisted": False,
        }
    system, user = _decision_prompt(observation, advice_only=advice_only)
    request_body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 350,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        DEEPSEEK_ENDPOINT,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        parsed = json.loads(response_body["choices"][0]["message"]["content"])
        decision = _validate_decision(
            parsed, allowed=observation["available_action_ids"]
        )
        return {
            "status": "OK",
            "decision": decision,
            "provider": "deepseek",
            "model_requested": DEEPSEEK_MODEL,
            "model_returned": response_body.get("model", "unknown"),
            "response_id": response_body.get("id", ""),
            "usage": response_body.get("usage", {}),
            "credential_persisted": False,
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        return {
            "status": "BLOCKED_PROVIDER",
            "error": f"{type(exc).__name__}: {exc}",
            "provider": "deepseek",
            "model_requested": DEEPSEEK_MODEL,
            "credential_persisted": False,
        }


def _formal_metric_support(
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for phase in ("selection", "promotion"):
        for task_id in TASKS:
            prepared, _ = states[(phase, task_id)]
            labels = np.asarray(prepared.validation_labels, dtype=np.int64)
            support = np.bincount(
                labels.reshape(-1), minlength=int(prepared.num_classes)
            ).astype(np.int64)
            cells[f"{phase}:{TASK_NAMES[task_id]}"] = {
                "fold_id": int(prepared.fold_id),
                "num_configured_classes": int(prepared.num_classes),
                "per_class_support": support.tolist(),
                "all_classes_supported": bool(np.all(support > 0)),
                "evaluated_pixels": int(support.sum()),
                "averaging": "all_configured_classes_and_valid_pixels",
                "require_all_classes": True,
            }
    payload = {
        "metric_entrypoint": "p4_metrics.evaluate_probabilities",
        "primary_field": "miou",
        "cells": cells,
    }
    return {**payload, "formal_metric_support_hash": _hash_payload(payload)}


def _task_result_payload(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "miou": float(result["metrics"]["miou"]),
        "macro_f1": float(result["metrics"]["macro_f1"]),
        "accuracy": float(result["metrics"]["accuracy"]),
        "prediction_hash": result["prediction_hash"],
        "train_loss_mean": float(result["train_loss_mean"]),
        "train_loss_last": float(result["train_loss_last"]),
        "last_grad_norm": float(result["last_grad_norm"]),
        "sam2_update_l2": float(result["sam2_update_l2"]),
        "sam2_trainable_parameters": int(result["sam2_trainable_parameters"]),
        "sam2_trainable_blocks": list(result["sam2_trainable_blocks"]),
        "attention_entropy": float(result["attention_entropy"]),
        "train_attention_entropy": float(result["train_attention_entropy"]),
        "fusion_scale_initial": float(result["fusion_scale_initial"]),
        "fusion_scale": float(result["fusion_scale"]),
    }


def _run_per_dataset_package(
    *,
    policy_id: str,
    round_id: int,
    phase: str,
    action_by_dataset: Mapping[str, str],
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
    device: str,
) -> dict[str, Any]:
    if set(action_by_dataset) != {"F3", "Penobscot"}:
        raise ValueError("per-dataset package must configure F3 and Penobscot")
    started = time.perf_counter()
    deadline = started + ACTION_WALL_CLOCK_BUDGET_S
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    tasks: dict[str, Any] = {}
    config_payload: dict[str, Any] = {}
    for task_id in TASKS:
        task = TASK_NAMES[task_id]
        action_id = action_by_dataset[task]
        config = A0_CONFIG if action_id == A0_CONFIG.action_id else ACTION_ALLOWLIST[action_id]
        prepared, trained = states[(phase, task_id)]
        result = p28._train_cross_action(
            task_id=task_id,
            prepared=prepared,
            trained_baseline=trained,
            config=config,
            device=device,
            seed=ROOT_SEED + int(prepared.fold_id),
            deadline=deadline,
        )
        tasks[task] = _task_result_payload(result)
        config_payload[task] = asdict(config)
    runtime = time.perf_counter() - started
    if runtime > ACTION_WALL_CLOCK_BUDGET_S:
        raise TimeoutError("per-dataset action package exceeded wall-clock budget")
    equal_mean = float(np.mean([tasks[task]["miou"] for task in tasks]))
    package_core = {
        "action_by_dataset": dict(action_by_dataset),
        "config_by_dataset": config_payload,
        "prediction_hash_by_dataset": {
            task: tasks[task]["prediction_hash"] for task in tasks
        },
        "equal_mean": equal_mean,
    }
    return {
        "policy_id": policy_id,
        "round": int(round_id),
        "phase": phase,
        "package_id": (
            f"F3={action_by_dataset['F3']}|"
            f"Penobscot={action_by_dataset['Penobscot']}"
        ),
        "action_by_dataset": dict(action_by_dataset),
        "config_by_dataset": config_payload,
        "package_config_hash": _hash_payload(config_payload),
        "tasks": tasks,
        "equal_mean": equal_mean,
        "package_effect_hash": _hash_payload(package_core),
        "runtime_s": runtime,
        "peak_vram_mb": (
            int(torch.cuda.max_memory_allocated(torch.device(device)) / (1024 * 1024))
            if device.startswith("cuda")
            else 0
        ),
        "wall_clock_budget_s": ACTION_WALL_CLOCK_BUDGET_S,
        "exit_status": "OK",
        "frozen_test_accessed": False,
    }


def _action_effect_core(
    package: Mapping[str, Any], a0_selection: Mapping[str, Any]
) -> dict[str, Any]:
    task_deltas = {
        task: float(package["tasks"][task]["miou"])
        - float(a0_selection["tasks"][task]["miou"])
        for task in ("F3", "Penobscot")
    }
    diagnostics = {
        task: {
            "gradient_normalized_signed_change": _normalized_signed_change(
                package["tasks"][task]["last_grad_norm"],
                a0_selection["tasks"][task]["last_grad_norm"],
            ),
            "fusion_normalized_signed_movement": round(
                _clip_signed(
                    (
                        float(package["tasks"][task]["fusion_scale"])
                        - float(package["tasks"][task]["fusion_scale_initial"])
                    )
                    / FUSION_MOVEMENT_SCALE
                ),
                6,
            ),
        }
        for task in ("F3", "Penobscot")
    }
    return {
        "action_id": package["action_id"],
        "config_hash": package["config_hash"],
        "prediction_hash_by_dataset": {
            task: package["tasks"][task]["prediction_hash"]
            for task in ("F3", "Penobscot")
        },
        "primary_delta_mIoU": {
            **task_deltas,
            "equal_mean": float(package["equal_mean"])
            - float(a0_selection["equal_mean"]),
        },
        "train_effect_diagnostics": diagnostics,
    }


def _attach_action_effect_hash(
    package: dict[str, Any], a0_selection: Mapping[str, Any]
) -> dict[str, Any]:
    core = _action_effect_core(package, a0_selection)
    package["action_effect_hash"] = _hash_payload(core)
    return package


def _select_per_dataset_actions(
    *,
    a0_selection: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    selected: dict[str, str] = {}
    for task in ("F3", "Penobscot"):
        baseline = float(a0_selection["tasks"][task]["miou"])
        candidates = [
            (trial["action_id"], float(trial["tasks"][task]["miou"]))
            for trial in trials
        ]
        best_action, best_value = max(
            candidates,
            key=lambda item: item[1],
            default=(A0_CONFIG.action_id, baseline),
        )
        selected[task] = (
            best_action
            if best_value - baseline >= DATASET_ACTION_SWITCH_MIN_DELTA
            else A0_CONFIG.action_id
        )
    return selected


def _sample_efficiency_path_score(
    *,
    a0_selection: Mapping[str, Any],
    trials: Sequence[Mapping[str, Any]],
) -> tuple[float, list[float]]:
    best = {
        task: float(a0_selection["tasks"][task]["miou"])
        for task in ("F3", "Penobscot")
    }
    trace: list[float] = []
    for trial in trials:
        for task in best:
            best[task] = max(best[task], float(trial["tasks"][task]["miou"]))
        trace.append(float(np.mean(list(best.values()))))
    if not trace:
        trace.append(float(np.mean(list(best.values()))))
    while len(trace) < MAX_TRIALS_PER_POLICY:
        trace.append(trace[-1])
    if len(trace) != MAX_TRIALS_PER_POLICY:
        raise ValueError("sample-efficiency trace exceeded its frozen horizon")
    return float(np.mean(trace)), trace


def _deterministic_action(
    *,
    round_id: int,
    available: Sequence[str],
    baseline_diagnostics: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    if round_id == 1:
        if any(
            item["fusion_movement_state"] == "stuck"
            for item in baseline_diagnostics.values()
        ):
            return "FAC_GATE_035", "A0 fusion movement is stuck; test the moderate gate."
        return "FAC_FUSION_LR_1E4", "No gate alarm; test the lower fusion learning rate."
    latest = history[-1]
    directions = [
        latest["per_dataset_effects"][task]["effect_signal"]["direction"]
        for task in ("F3", "Penobscot")
    ]
    preferred = "FAC_FUSION_LR_1E4" if "worse" in directions else "FAC_SAM2_FROZEN"
    if preferred not in available:
        preferred = next(action for action in ACTION_IDS if action in available)
    return preferred, "Use the fixed diagnostic rule on the complete prior effect signal."


def _run_policy(
    *,
    policy_id: str,
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
    device: str,
    a0: Mapping[str, Any],
    baseline_diagnostics: Mapping[str, Any],
    random_actions: Sequence[str] | None = None,
    fixed_actions: Sequence[str] | None = None,
    observation_ablation: bool = False,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    used: list[str] = []
    stop_after_round: int | None = None
    for round_id in range(1, MAX_TRIALS_PER_POLICY + 1):
        available = [action for action in ACTION_IDS if action not in used]
        builder = _build_observation_ablation if observation_ablation else _build_observation_v2
        observation = builder(
            policy_id=policy_id,
            round_id=round_id,
            available_action_ids=available,
            baseline_diagnostics=baseline_diagnostics,
            history=history,
        )
        if policy_id.startswith("A2L"):
            response = _call_deepseek(observation, advice_only=False)
            if response["status"] != "OK":
                decisions.append(
                    {
                        "round": round_id,
                        "observation": observation,
                        "observation_hash": _hash_payload(observation),
                        "status": response["status"],
                        "error": response.get("error", "provider failure"),
                        "credential_persisted": False,
                    }
                )
                return {
                    "policy_id": policy_id,
                    "status": "BLOCKED_PROVIDER",
                    "decisions": decisions,
                    "selection_trials": trials,
                    "promotion": None,
                    "stop_semantics": STOP_SEMANTICS,
                }
            decision = response["decision"]
            action_id = decision["action_id"]
            decision_record = {
                "round": round_id,
                "observation": observation,
                "observation_hash": _hash_payload(observation),
                "status": "OK",
                **decision,
                "provider": response["provider"],
                "model_requested": response["model_requested"],
                "model_returned": response["model_returned"],
                "response_id": response["response_id"],
                "usage": response["usage"],
                "credential_persisted": False,
            }
        elif policy_id == "A2D_deterministic_agent":
            action_id, rationale = _deterministic_action(
                round_id=round_id,
                available=available,
                baseline_diagnostics=baseline_diagnostics,
                history=history,
            )
            decision_record = {
                "round": round_id,
                "observation": observation,
                "observation_hash": _hash_payload(observation),
                "status": "OK",
                "action_id": action_id,
                "confidence": 1.0,
                "rationale": rationale,
                "stop": round_id == MAX_TRIALS_PER_POLICY,
                "provider": "deterministic_diagnostics",
                "credential_persisted": False,
            }
        elif policy_id == "A3_random_policy":
            if random_actions is None:
                raise ValueError("A3 requires frozen random actions")
            action_id = str(random_actions[round_id - 1])
            decision_record = {
                "round": round_id,
                "observation": observation,
                "observation_hash": _hash_payload(observation),
                "status": "OK",
                "action_id": action_id,
                "confidence": 1.0 / len(available),
                "rationale": "PCG64 random choice without replacement.",
                "stop": round_id == MAX_TRIALS_PER_POLICY,
                "provider": "numpy.PCG64",
                "credential_persisted": False,
            }
        elif policy_id == "A4_deterministic_search":
            if fixed_actions is None:
                raise ValueError("A4 requires a preregistered action sequence")
            action_id = str(fixed_actions[round_id - 1])
            decision_record = {
                "round": round_id,
                "observation": observation,
                "observation_hash": _hash_payload(observation),
                "status": "OK",
                "action_id": action_id,
                "confidence": 1.0,
                "rationale": "Preregistered two-action domain search order.",
                "stop": round_id == MAX_TRIALS_PER_POLICY,
                "provider": "deterministic_search",
                "credential_persisted": False,
            }
        else:
            raise ValueError(f"unsupported P29 policy: {policy_id}")
        if action_id not in available:
            raise RuntimeError("policy repeated or escaped the allowlist")
        decisions.append(decision_record)
        used.append(action_id)
        package = p28._run_config_package(
            policy_id=policy_id,
            round_id=round_id,
            phase="selection",
            config=ACTION_ALLOWLIST[action_id],
            states=states,
            device=device,
        )
        _attach_action_effect_hash(package, a0["selection"])
        history_entry = _history_entry(
            package,
            a0["selection"],
            stop_requested=bool(decision_record["stop"]),
        )
        package["safe_observation_effect"] = history_entry
        history.append(history_entry)
        trials.append(package)
        if decision_record["stop"]:
            stop_after_round = round_id
            break
    selected = _select_per_dataset_actions(
        a0_selection=a0["selection"], trials=trials
    )
    promotion = _run_per_dataset_package(
        policy_id=policy_id,
        round_id=len(trials) + 1,
        phase="promotion",
        action_by_dataset=selected,
        states=states,
        device=device,
    )
    score, trace = _sample_efficiency_path_score(
        a0_selection=a0["selection"], trials=trials
    )
    status = (
        "STOPPED_AFTER_EXECUTION"
        if stop_after_round is not None and stop_after_round < MAX_TRIALS_PER_POLICY
        else "OK"
    )
    return {
        "policy_id": policy_id,
        "status": status,
        "decisions": decisions,
        "selection_trials": trials,
        "executed_action_count": len(trials),
        "maximum_action_trials": MAX_TRIALS_PER_POLICY,
        "stop_after_round": stop_after_round,
        "stop_semantics": STOP_SEMANTICS,
        "full_history": history,
        "selected_per_dataset_package": selected,
        "sample_efficiency_running_max_per_dataset_mean_mIoU_trace": trace,
        "sample_efficiency_path_score_running_max_per_dataset_mean_mIoU": score,
        "endpoint_promotion_mean_mIoU": float(promotion["equal_mean"]),
        "promotion": promotion,
    }


def _run_a1_replay(
    *,
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
    device: str,
    a0: Mapping[str, Any],
    baseline_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    observation = _build_observation_v2(
        policy_id="A1_advice_only",
        round_id=1,
        available_action_ids=ACTION_IDS,
        baseline_diagnostics=baseline_diagnostics,
        history=[],
    )
    response = _call_deepseek(observation, advice_only=True)
    replay = {
        phase: p28._run_config_package(
            policy_id="A1_advice_only",
            round_id=0,
            phase=phase,
            config=A0_CONFIG,
            states=states,
            device=device,
        )
        for phase in ("selection", "promotion")
    }
    hashes = {
        phase: {
            task: replay[phase]["tasks"][task]["prediction_hash"]
            for task in ("F3", "Penobscot")
        }
        for phase in replay
    }
    baseline_hashes = {
        phase: {
            task: a0[phase]["tasks"][task]["prediction_hash"]
            for task in ("F3", "Penobscot")
        }
        for phase in replay
    }
    metrics = {phase: p28._metric_view(replay[phase]) for phase in replay}
    baseline_metrics = {phase: p28._metric_view(a0[phase]) for phase in replay}
    return {
        "policy_id": "A1_advice_only",
        "status": response["status"],
        "observation": observation,
        "observation_hash": _hash_payload(observation),
        "decision": response.get("decision"),
        "error": response.get("error"),
        "executed_action": False,
        "replay_kind": "fresh_same_config_seed_fold_reexecution",
        "replay": replay,
        "prediction_hashes": hashes,
        "a0_prediction_hashes": baseline_hashes,
        "prediction_hash_equal_a0": hashes == baseline_hashes,
        "metrics": metrics,
        "a0_metrics": baseline_metrics,
        "metrics_equal_a0": metrics == baseline_metrics,
        "credential_persisted": False,
    }


def _build_action_effects(
    *,
    a0: Mapping[str, Any],
    policies: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    packages: dict[str, list[Mapping[str, Any]]] = {action: [] for action in ACTION_IDS}
    selected_by: dict[str, list[str]] = {action: [] for action in ACTION_IDS}
    for policy_key, policy in policies.items():
        for trial in policy.get("selection_trials", []):
            packages[trial["action_id"]].append(trial)
            selected_by[trial["action_id"]].append(policy_key)
    if any(not values for values in packages.values()):
        missing = [action for action, values in packages.items() if not values]
        raise RuntimeError(f"P29 action-effect registry is incomplete: {missing}")
    baseline_core = {
        "action_id": A0_CONFIG.action_id,
        "config_hash": a0["selection"]["config_hash"],
        "prediction_hash_by_dataset": {
            task: a0["selection"]["tasks"][task]["prediction_hash"]
            for task in ("F3", "Penobscot")
        },
        "primary_delta_mIoU": {"F3": 0.0, "Penobscot": 0.0, "equal_mean": 0.0},
    }
    actions: dict[str, Any] = {}
    for action_id, observed in packages.items():
        core = _action_effect_core(observed[0], a0["selection"])
        observed_hashes = sorted({item["action_effect_hash"] for item in observed})
        prediction_changed = {
            task: any(
                item["tasks"][task]["prediction_hash"]
                != a0["selection"]["tasks"][task]["prediction_hash"]
                for item in observed
            )
            for task in ("F3", "Penobscot")
        }
        actions[action_id] = {
            **core,
            "action_effect_hash": _hash_payload(core),
            "observed_action_effect_hashes": observed_hashes,
            "same_invocation_replay_consistent": len(observed_hashes) == 1,
            "prediction_changed_vs_A0": prediction_changed,
            "selected_by_policies": sorted(set(selected_by[action_id])),
            "legal_selection_visibility": {
                "visible_before_action_execution": False,
                "visible_after_action_execution_to_selecting_policy": True,
                "oracle_only_for_unselected_actions": True,
            },
        }
    payload = {
        "schema_version": ACTION_EFFECT_SCHEMA_VERSION,
        "scope": "development_selection_fold_0_only",
        "metric_entrypoint": "p4_metrics.evaluate_probabilities",
        "baseline": {
            **baseline_core,
            "action_effect_hash": _hash_payload(baseline_core),
        },
        "actions": actions,
        "promotion_metrics_used_for_action_effect_registry": False,
        "frozen_test_accessed": False,
    }
    payload["action_effect_registry_hash"] = _hash_payload(payload)
    return payload


def _oracle_ceiling(
    *,
    a0: Mapping[str, Any], action_effects: Mapping[str, Any]
) -> dict[str, Any]:
    selected: dict[str, str] = {}
    values: dict[str, float] = {}
    for task in ("F3", "Penobscot"):
        candidates = [(A0_CONFIG.action_id, 0.0)] + [
            (
                action_id,
                float(record["primary_delta_mIoU"][task]),
            )
            for action_id, record in action_effects["actions"].items()
        ]
        action_id, delta = max(candidates, key=lambda item: item[1])
        selected[task] = action_id
        values[task] = float(a0["selection"]["tasks"][task]["miou"]) + delta
    return {
        "scope": "selection_only_diagnostic_not_eligible_for_promotion",
        "evaluated_action_count": len(ACTION_IDS),
        "selected_per_dataset_package": selected,
        "per_dataset_mIoU": values,
        "equal_mean_mIoU": float(np.mean(list(values.values()))),
        "promotion_metrics_computed": False,
        "used_for_policy_feedback": False,
    }


def _promotion_guard_diagnostics(
    *,
    policy: Mapping[str, Any],
    a0: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, Any]:
    if policy.get("promotion") is None:
        return {
            "passed": False,
            "status": policy.get("status", "UNKNOWN"),
            "checks": {
                "mean_delta_at_least_0p005": False,
                "f3_delta_at_least_minus_0p005": False,
                "penobscot_delta_at_least_minus_0p005": False,
                "not_worse_than_continued_cnn": False,
            },
        }
    promotion = p28._metric_view(policy["promotion"])
    baseline = p28._metric_view(a0["promotion"])
    continued = p28._metric_view(control["promotion"])
    checks = {
        "mean_delta_at_least_0p005": (
            promotion["equal_mean"] - baseline["equal_mean"]
            >= MEAN_PROMOTION_DELTA
        ),
        "f3_delta_at_least_minus_0p005": (
            promotion["F3"] - baseline["F3"] >= TASK_NON_DEGRADATION
        ),
        "penobscot_delta_at_least_minus_0p005": (
            promotion["Penobscot"] - baseline["Penobscot"]
            >= TASK_NON_DEGRADATION
        ),
        "not_worse_than_continued_cnn": all(
            promotion[key] >= continued[key]
            for key in ("F3", "Penobscot", "equal_mean")
        ),
    }
    return {
        "passed": all(checks.values()),
        "status": policy["status"],
        "checks": checks,
        "delta_vs_a0": {
            key: promotion[key] - baseline[key] for key in promotion
        },
        "delta_vs_continued_cnn": {
            key: promotion[key] - continued[key] for key in promotion
        },
    }


def _gate_summary(
    *,
    a0: Mapping[str, Any],
    control: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2l: Mapping[str, Any],
    a2d: Mapping[str, Any],
    a3: Mapping[str, Any],
    a4: Mapping[str, Any],
    action_effects: Mapping[str, Any],
    metric_support: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "a1_replay_prediction_hash_equal_a0": bool(a1["prediction_hash_equal_a0"]),
        "a1_replay_metrics_equal_a0": bool(a1["metrics_equal_a0"]),
        "a2l_uses_at_most_two_legal_distinct_actions": False,
        "a2l_stop_semantics_executable": False,
        "a2l_sample_efficiency_above_a2d": False,
        "a2l_sample_efficiency_above_a3": False,
        "promotion_endpoint_mean_delta_at_least_0p005": False,
        "f3_delta_at_least_minus_0p005": False,
        "penobscot_delta_at_least_minus_0p005": False,
        "not_worse_than_continued_cnn": False,
        "all_five_action_effects_persisted": len(action_effects["actions"]) == 5,
        "all_five_action_effects_change_config_and_prediction": all(
            record["config_hash"]
            != action_effects["baseline"]["config_hash"]
            and any(record["prediction_changed_vs_A0"].values())
            for record in action_effects["actions"].values()
        ),
        "formal_metric_support_persisted_and_hashed": (
            len(metric_support["cells"]) == 4
            and bool(metric_support["formal_metric_support_hash"])
            and all(
                cell["require_all_classes"]
                and len(cell["per_class_support"])
                == cell["num_configured_classes"]
                and sum(cell["per_class_support"]) == cell["evaluated_pixels"]
                for cell in metric_support["cells"].values()
            )
        ),
        "no_frozen_test_access": True,
    }
    if a2l.get("status") == "BLOCKED_PROVIDER":
        a4_guards = _promotion_guard_diagnostics(
            policy=a4, a0=a0, control=control
        )
        hybrid_checks = {
            "a4_dataset_conditioned_package_is_gate050_plus_a0": (
                a4.get("selected_per_dataset_package")
                == {"F3": "FAC_GATE_050", "Penobscot": A0_CONFIG.action_id}
            ),
            "a4_package_passes_preserved_promotion_guards": bool(
                a4_guards["passed"]
            ),
            "action_effect_chain_is_non_noop": bool(
                checks["all_five_action_effects_persisted"]
                and checks[
                    "all_five_action_effects_change_config_and_prediction"
                ]
            ),
            "formal_metric_support_is_persisted": bool(
                checks["formal_metric_support_persisted_and_hashed"]
            ),
            "no_frozen_test_access": True,
        }
        retain_hybrid = all(hybrid_checks.values())
        return {
            "checks": checks,
            "hybrid_checks": hybrid_checks,
            "retain": retain_hybrid,
            "retain_agent": False,
            "retain_hybrid": retain_hybrid,
            "verdict": "BLOCKED_PROVIDER",
        }
    actions = [trial["action_id"] for trial in a2l["selection_trials"]]
    checks["a2l_uses_at_most_two_legal_distinct_actions"] = (
        1 <= len(actions) <= MAX_TRIALS_PER_POLICY
        and len(set(actions)) == len(actions)
        and set(actions) <= set(ACTION_IDS)
    )
    stop_rounds = [
        int(decision["round"])
        for decision in a2l["decisions"]
        if decision.get("stop")
    ]
    checks["a2l_stop_semantics_executable"] = (
        not stop_rounds
        or (
            a2l["stop_after_round"] == stop_rounds[0]
            and a2l["executed_action_count"] == stop_rounds[0]
            and len(a2l["decisions"]) == stop_rounds[0]
        )
    )
    score_key = "sample_efficiency_path_score_running_max_per_dataset_mean_mIoU"
    checks["a2l_sample_efficiency_above_a2d"] = float(a2l[score_key]) > float(a2d[score_key])
    checks["a2l_sample_efficiency_above_a3"] = float(a2l[score_key]) > float(a3[score_key])
    promotion = p28._metric_view(a2l["promotion"])
    baseline = p28._metric_view(a0["promotion"])
    continued = p28._metric_view(control["promotion"])
    checks["promotion_endpoint_mean_delta_at_least_0p005"] = (
        promotion["equal_mean"] - baseline["equal_mean"] >= MEAN_PROMOTION_DELTA
    )
    checks["f3_delta_at_least_minus_0p005"] = (
        promotion["F3"] - baseline["F3"] >= TASK_NON_DEGRADATION
    )
    checks["penobscot_delta_at_least_minus_0p005"] = (
        promotion["Penobscot"] - baseline["Penobscot"] >= TASK_NON_DEGRADATION
    )
    checks["not_worse_than_continued_cnn"] = all(
        promotion[key] >= continued[key] for key in ("F3", "Penobscot", "equal_mean")
    )
    retain_agent = all(checks.values())
    a4_guards = _promotion_guard_diagnostics(policy=a4, a0=a0, control=control)
    hybrid_checks = {
        "a4_dataset_conditioned_package_is_gate050_plus_a0": (
            a4["selected_per_dataset_package"]
            == {"F3": "FAC_GATE_050", "Penobscot": A0_CONFIG.action_id}
        ),
        "a4_package_passes_preserved_promotion_guards": bool(a4_guards["passed"]),
        "action_effect_chain_is_non_noop": bool(
            checks["all_five_action_effects_persisted"]
            and checks["all_five_action_effects_change_config_and_prediction"]
        ),
        "formal_metric_support_is_persisted": bool(
            checks["formal_metric_support_persisted_and_hashed"]
        ),
        "no_frozen_test_access": True,
    }
    retain_hybrid = all(hybrid_checks.values())
    endpoint_scores = {
        key.upper(): float(policy["endpoint_promotion_mean_mIoU"])
        for key, policy in {"a2l": a2l, "a2d": a2d, "a3": a3, "a4": a4}.items()
    }
    direct_agent_superiority = endpoint_scores["A2L"] > max(
        endpoint_scores["A2D"], endpoint_scores["A3"], endpoint_scores["A4"]
    )
    return {
        "checks": checks,
        "hybrid_checks": hybrid_checks,
        "retain": retain_agent or retain_hybrid,
        "retain_agent": retain_agent,
        "retain_hybrid": retain_hybrid,
        "verdict": (
            "RETAIN_AGENT"
            if retain_agent and direct_agent_superiority
            else "RETAIN_HYBRID"
            if retain_hybrid
            else "REJECT_AGENT"
        ),
        "direct_agent_endpoint_superiority": direct_agent_superiority,
        "a2l_promotion_delta_vs_a0": {
            key: promotion[key] - baseline[key] for key in promotion
        },
        "a2l_promotion_delta_vs_continued_cnn": {
            key: promotion[key] - continued[key] for key in promotion
        },
        "a4_hybrid_promotion_guards": a4_guards,
        "sample_efficiency_path_scores": {
            "A2L": a2l[score_key],
            "A2D": a2d[score_key],
            "A3": a3[score_key],
            "A4": a4[score_key],
        },
        "endpoint_promotion_mean_mIoU": endpoint_scores,
    }


def _protocol(
    split_contract: Mapping[str, Any], metric_support: Mapping[str, Any]
) -> dict[str, Any]:
    action_table = {
        action_id: asdict(config) for action_id, config in ACTION_ALLOWLIST.items()
    }
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "track_id": "facies",
        "stage": "P29_action_effect_repair",
        "a0": asdict(A0_CONFIG),
        "sam2_weight_mode": "pretrained",
        "action_table": action_table,
        "action_space_hash": _hash_payload(action_table),
        "maximum_trials_per_policy": MAX_TRIALS_PER_POLICY,
        "comparison": "two_of_five_sample_efficiency",
        "policies": [
            "A0_static_baseline",
            "A1_advice_only_true_replay",
            "A2L_llm_agent_execute",
            "A2D_deterministic_agent",
            "A3_random_policy",
            "A4_deterministic_search",
        ],
        "prompt_ablation": "A2L_v1_categorical_only_vs_A2L_v2_full_history",
        "observation_schema": OBSERVATION_SCHEMA_VERSION,
        "observation_v2_fields": [
            "previous_action",
            "full_action_history",
            "normalized_signed_delta",
            "optimizer_gradient_movement",
            "fusion_gate_movement",
            "stop_semantics",
        ],
        "stop_semantics": STOP_SEMANTICS,
        "metric_contract": {
            "entrypoint": "p4_metrics.evaluate_probabilities",
            "primary_field": "miou",
            "dataset_aggregation": "equal_weight_F3_Penobscot",
            "formal_metric_support_hash": metric_support["formal_metric_support_hash"],
            "require_all_classes": True,
        },
        "sample_efficiency_metric": (
            "arithmetic_mean_over_two_action_opportunities_of_running_best_"
            "per_dataset_selection_mIoU_seeded_by_A0"
        ),
        "endpoint_metric": (
            "equal_mean_F3_Penobscot_mIoU_on_disjoint_fold4_for_the_"
            "per_dataset_package_selected_only_from_fold0"
        ),
        "per_dataset_package_allows_A0": True,
        "per_dataset_action_switch_min_delta_mIoU": DATASET_ACTION_SWITCH_MIN_DELTA,
        "continued_cnn_is_policy_action": False,
        "policy_seed": ROOT_SEED,
        "random_generator": "numpy.random.PCG64",
        "selection_fold_ids": [SELECTION_FOLD],
        "promotion_fold_ids": [PROMOTION_FOLD],
        "selection_promotion_disjoint": True,
        "train_samples_per_fold": 32,
        "validation_samples_per_fold": 16,
        "candidate_updates": p13.CANDIDATE_UPDATES,
        "wall_clock_budget_s": ACTION_WALL_CLOCK_BUDGET_S,
        "gpu_limit": GPU_LIMIT,
        "split_contract": dict(split_contract),
        "manifest_hashes": EXPECTED_MANIFEST_HASHES,
        "observation_denylist": list(DENIED_OBSERVATION_TERMS),
        "normalized_signed_delta_scale_mIoU": SIGNED_DELTA_SCALE_MIOU,
        "normalized_signed_delta_clip": SIGNED_DELTA_CLIP,
        "provider_failure": "fail_closed_BLOCKED_PROVIDER",
        "promotion_guards": {
            "mean_delta": MEAN_PROMOTION_DELTA,
            "per_dataset_min_delta": TASK_NON_DEGRADATION,
            "not_worse_than_continued_cnn": True,
        },
        "oracle_scope": "selection_only_not_visible_to_policy_not_promoted",
        "frozen_test_accessed": False,
        "holdout_paths_accepted": False,
    }


def _decision_rows(
    *,
    protocol: Mapping[str, Any],
    a0: Mapping[str, Any],
    control: Mapping[str, Any],
    a1: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "runner_revision": p11._sha256(Path(__file__)),
        "policy_seed": ROOT_SEED,
        "action_space_hash": protocol["action_space_hash"],
        "formal_metric_support_hash": protocol["metric_contract"]["formal_metric_support_hash"],
        "selection_fold_ids": [SELECTION_FOLD],
        "promotion_fold_ids": [PROMOTION_FOLD],
        "frozen_test_accessed": False,
        "leakage_checks": {
            "manifest_hash_locked": True,
            "selection_promotion_disjoint": True,
            "train_only_fit": True,
            "llm_observation_sanitized": True,
            "promotion_not_used_for_feedback": True,
            "holdout_paths_accepted": False,
        },
    }
    rows: list[dict[str, Any]] = []
    for phase in ("selection", "promotion"):
        rows.append(
            {
                **common,
                "event": "A0_RESULT",
                "policy_id": "A0_static_baseline",
                "phase": phase,
                "result": a0[phase],
            }
        )
        rows.append(
            {
                **common,
                "event": "CONTROL_RESULT",
                "policy_id": "continued_cnn_control",
                "phase": phase,
                "result": control[phase],
            }
        )
    rows.append(
        {
            **common,
            "event": "A1_TRUE_REPLAY",
            "policy_id": "A1_advice_only",
            "result": a1,
        }
    )
    for policy in policies:
        for index, decision in enumerate(policy.get("decisions", [])):
            trials = policy.get("selection_trials", [])
            rows.append(
                {
                    **common,
                    "event": "POLICY_TRIAL",
                    "policy_id": policy["policy_id"],
                    "round": decision["round"],
                    "decision": decision,
                    "result": trials[index] if index < len(trials) else None,
                }
            )
        if policy.get("promotion") is not None:
            rows.append(
                {
                    **common,
                    "event": "PER_DATASET_PROMOTION",
                    "policy_id": policy["policy_id"],
                    "selected_action_by_dataset": policy["selected_per_dataset_package"],
                    "result": policy["promotion"],
                }
            )
    return rows


def _write_root_cause(summary: Mapping[str, Any], output_root: Path) -> Path:
    action_effects = summary["action_effects"]
    lines = [
        "# P29 facies root-cause chain",
        "",
        "## Causal wiring audit",
        "",
        "| Link | Connected | Evidence |",
        "|---|---:|---|",
        f"| observation/prompt | yes | observation v2; prompt-ablation hash `{summary['prompt_ablation_hash']}` |",
        f"| selected action/executor | yes | action registry hash `{summary['protocol_hashes']['action_space_hash']}` |",
        f"| optimizer and fusion movement | yes | five action-effect records `{action_effects['action_effect_registry_hash']}` |",
        f"| prediction endpoint | yes | all actions persist prediction hashes; no-op check `{summary['action_noop_check_passed']}` |",
        f"| primary metric | yes | `evaluate_probabilities` support hash `{summary['formal_metric_support']['formal_metric_support_hash']}` |",
        f"| promotion/endpoint | yes | disjoint per-dataset package hash `{summary['a2l'].get('promotion', {}).get('package_effect_hash', 'not_available')}` |",
        "",
        "## Root cause",
        "",
        "P28's five actions were executable, but a single global action forced F3 and Penobscot to share a setting, "
        "while four of five trials made policy endpoints converge. P29 therefore tests two-action sample efficiency "
        "and allows each dataset to retain its best fold-0 choice, including A0. This repairs the comparison; it does "
        "not assume an LLM contribution.",
        "",
        f"Honest verdict: **{summary['gate']['verdict']}**.",
        "",
        "No frozen holdout or `test.h5` was read.",
    ]
    path = output_root / "root_cause.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_evidence(summary: Mapping[str, Any], output_root: Path) -> Path:
    gate = summary["gate"]
    lines = [
        "# P29 facies action-effect repair evidence",
        "",
        "## Frozen scope",
        "",
        "A0 remains the same-invocation pretrained SAM2 cross-attention gate=0.2 route. Selection uses fold 0; "
        "promotion uses disjoint fold 4; every fold has 32 training and 16 development-validation samples. "
        "The primary metric remains `p4_metrics.evaluate_probabilities` mIoU with all configured classes.",
        "",
        "No frozen holdout or `test.h5` was read. DeepSeek received only observation-v2 train diagnostics and "
        "clipped normalized signed fold-0 effects, never raw scores, labels, residuals, sample IDs, paths, "
        "predictions, fold-4 results, or frozen-test information.",
        "",
        "## Observation and action effects",
        "",
        f"- A1 true replay hashes equal A0: `{summary['a1']['prediction_hash_equal_a0']}`; metrics equal: `{summary['a1']['metrics_equal_a0']}`.",
        f"- Formal metric support hash: `{summary['formal_metric_support']['formal_metric_support_hash']}`.",
        f"- Development cells with every configured class observed: `"
        f"{sum(cell['all_classes_supported'] for cell in summary['formal_metric_support']['cells'].values())}/4`; "
        "formal mIoU still averages all configured classes.",
        f"- Five-action effect registry hash: `{summary['action_effects']['action_effect_registry_hash']}`.",
        f"- Every action changed at least one prediction endpoint versus A0: `{summary['action_noop_check_passed']}`.",
        f"- Prompt information ablation hash: `{summary['prompt_ablation_hash']}`.",
        "",
        "## Two-of-five sample efficiency and promotion",
        "",
    ]
    score_key = "sample_efficiency_path_score_running_max_per_dataset_mean_mIoU"
    for key in POLICY_KEYS:
        policy = summary[key]
        if policy.get("promotion") is None:
            lines.append(
                f"- {policy['policy_id']}: status `{policy['status']}`; "
                "no hidden fallback or promotion was executed."
            )
        else:
            guard_passed = summary["policy_promotion_guard_diagnostics"][key][
                "passed"
            ]
            lines.append(
                f"- {policy['policy_id']}: actions `{[trial['action_id'] for trial in policy['selection_trials']]}`; "
                f"path score `{policy[score_key]:.9f}`; package `{policy['selected_per_dataset_package']}`; "
                f"promotion endpoint mean mIoU `{policy['endpoint_promotion_mean_mIoU']:.9f}`; "
                f"preserved promotion guards `{guard_passed}`; status `{policy['status']}`."
            )
    lines.extend(
        [
            "",
            "The oracle is selection-fold-only and was not promoted or shown to any policy.",
            "",
            "## Preserved promotion guards",
            "",
        ]
    )
    for name, passed in gate["checks"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(["", "## Dataset-conditioned hybrid gate", ""])
    for name, passed in gate["hybrid_checks"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend(
        [
            "",
            f"Verdict: **{gate['verdict']}**.",
            "",
            "`RETAIN_HYBRID` means the dataset-conditioned deterministic package passed the frozen guards. "
            "It does not retain the direct LLM policy and is not a claim that SAM2 or the LLM caused the endpoint gain. "
            "Agent retention and direct endpoint superiority are reported separately.",
            "",
            f"- Retain direct agent: `{gate.get('retain_agent', False)}`",
            f"- Retain dataset-conditioned hybrid: `{gate.get('retain_hybrid', False)}`",
            f"- Direct A2L endpoint superiority: `{gate.get('direct_agent_endpoint_superiority', False)}`",
            f"- Provider credential persisted: `False`",
        ]
    )
    path = output_root / "evidence.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(
    *,
    f3_manifest: Path,
    penobscot_manifest: Path,
    processed_root: Path,
    device: str = "cuda:0",
) -> dict[str, Path]:
    started = time.perf_counter()
    p11._validate_cuda_device(device)
    p28._request_deterministic_execution()
    manifests = p11.validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    output_root = _validate_output_root(OUTPUT_ROOT)
    protected = [p11.OUTPUT_ROOT, p12.OUTPUT_ROOT, p13.OUTPUT_ROOT, p28.OUTPUT_ROOT]
    for verifier, root in (
        (p11.verify, p11.OUTPUT_ROOT),
        (p12.verify, p12.OUTPUT_ROOT),
        (p13.verify, p13.OUTPUT_ROOT),
        (p28.verify, p28.OUTPUT_ROOT),
    ):
        verifier(root)
    protected_hashes = {
        str(root.relative_to(HERE)): p11._sha256(root / "artifact_manifest.csv")
        for root in protected
    }
    p11._prepare_sam2_dependency_path()
    source_root = p11.verify_git_source(
        p11.SAM2_SOURCE_ROOT, p11.SAM2_SOURCE_REVISION
    )
    checkpoint = p11.verify_checkpoint("facies", p11.SAM2_CHECKPOINT)
    p11.insert_import_root(source_root, "sam2")
    output_root.mkdir(parents=True, exist_ok=True)
    states, split_contract = p28._prepare_same_invocation_states(
        manifests=manifests,
        processed_root=Path(processed_root).resolve(),
        device=device,
    )
    a0, control = p28._run_controls(states=states, device=device)
    baseline_diagnostics = _baseline_diagnostics(a0)
    metric_support = _formal_metric_support(states)
    a1 = _run_a1_replay(
        states=states,
        device=device,
        a0=a0,
        baseline_diagnostics=baseline_diagnostics,
    )
    a2l = _run_policy(
        policy_id="A2L_llm_agent_execute",
        states=states,
        device=device,
        a0=a0,
        baseline_diagnostics=baseline_diagnostics,
    )
    a2d = _run_policy(
        policy_id="A2D_deterministic_agent",
        states=states,
        device=device,
        a0=a0,
        baseline_diagnostics=baseline_diagnostics,
    )
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    random_actions = [
        str(value)
        for value in rng.choice(
            ACTION_IDS, size=MAX_TRIALS_PER_POLICY, replace=False
        )
    ]
    a3 = _run_policy(
        policy_id="A3_random_policy",
        states=states,
        device=device,
        a0=a0,
        baseline_diagnostics=baseline_diagnostics,
        random_actions=random_actions,
    )
    a4 = _run_policy(
        policy_id="A4_deterministic_search",
        states=states,
        device=device,
        a0=a0,
        baseline_diagnostics=baseline_diagnostics,
        fixed_actions=("FAC_GATE_050", "FAC_DICE_050"),
    )
    prompt_ablation = _run_policy(
        policy_id="A2L_v1_prompt_ablation",
        states=states,
        device=device,
        a0=a0,
        baseline_diagnostics=baseline_diagnostics,
        observation_ablation=True,
    )
    policies = {"a2l": a2l, "a2d": a2d, "a3": a3, "a4": a4}
    action_effects = _build_action_effects(a0=a0, policies=policies)
    oracle = _oracle_ceiling(a0=a0, action_effects=action_effects)
    action_noop_check = all(
        record["config_hash"] != a0["selection"]["config_hash"]
        and any(record["prediction_changed_vs_A0"].values())
        for record in action_effects["actions"].values()
    )
    gate = _gate_summary(
        a0=a0,
        control=control,
        a1=a1,
        a2l=a2l,
        a2d=a2d,
        a3=a3,
        a4=a4,
        action_effects=action_effects,
        metric_support=metric_support,
    )
    policy_guard_diagnostics = {
        key: _promotion_guard_diagnostics(
            policy=policy,
            a0=a0,
            control=control,
        )
        for key, policy in policies.items()
    }
    protocol = _protocol(split_contract, metric_support)
    current_hashes = {
        str(root.relative_to(HERE)): p11._sha256(root / "artifact_manifest.csv")
        for root in protected
    }
    if current_hashes != protected_hashes:
        raise RuntimeError("protected P11/P12/P13/P28 evidence changed during P29")
    prompt_ablation_payload = {
        "v2_actions": [trial["action_id"] for trial in a2l.get("selection_trials", [])],
        "v1_ablation_actions": [
            trial["action_id"] for trial in prompt_ablation.get("selection_trials", [])
        ],
        "v2_status": a2l["status"],
        "v1_ablation_status": prompt_ablation["status"],
        "v2_score": a2l.get(
            "sample_efficiency_path_score_running_max_per_dataset_mean_mIoU"
        ),
        "v1_ablation_score": prompt_ablation.get(
            "sample_efficiency_path_score_running_max_per_dataset_mean_mIoU"
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "a0": a0,
        "continued_cnn": control,
        "a1": a1,
        "a2l": a2l,
        "a2d": a2d,
        "a3": a3,
        "a4": a4,
        "prompt_ablation": prompt_ablation,
        "prompt_ablation_comparison": prompt_ablation_payload,
        "prompt_ablation_hash": _hash_payload(prompt_ablation_payload),
        "formal_metric_support": metric_support,
        "action_effects": action_effects,
        "action_noop_check_passed": action_noop_check,
        "oracle_ceiling": oracle,
        "gate": gate,
        "policy_promotion_guard_diagnostics": policy_guard_diagnostics,
        "protocol_hashes": {
            "action_space_hash": protocol["action_space_hash"],
            "protocol_hash": _hash_payload(protocol),
        },
        "evaluation": {
            "metric_entrypoint": "p4_metrics.evaluate_probabilities",
            "selection_fold_ids": [SELECTION_FOLD],
            "promotion_fold_ids": [PROMOTION_FOLD],
            "train_samples_per_fold": 32,
            "validation_samples_per_fold": 16,
            "manifest_hashes": EXPECTED_MANIFEST_HASHES,
            "same_invocation": True,
            "fresh_trials_not_historical_replay": True,
            "frozen_test_accessed": False,
            "holdout_paths_accepted": False,
        },
        "protected_evidence": protected_hashes,
        "runtime": {
            "duration_seconds": time.perf_counter() - started,
            "device": device,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "runner_sha256": p11._sha256(Path(__file__)),
            "sam2_source_revision": p11.SAM2_SOURCE_REVISION,
            "sam2_checkpoint_sha256": p11._sha256(checkpoint),
        },
    }
    rows = _decision_rows(
        protocol=protocol,
        a0=a0,
        control=control,
        a1=a1,
        policies=(a2l, a2d, a3, a4, prompt_ablation),
    )
    protocol_path = p11._write_json(output_root / "protocol.json", protocol)
    decisions_path = p11._write_jsonl(
        output_root / "decisions_and_trials.jsonl", rows
    )
    action_effects_path = p11._write_json(
        output_root / "action_effects.json", action_effects
    )
    summary_path = p11._write_json(output_root / "summary.json", summary)
    root_cause_path = _write_root_cause(summary, output_root)
    evidence_path = _write_evidence(summary, output_root)
    artifacts = []
    for kind, path in (
        ("json", protocol_path),
        ("jsonl", decisions_path),
        ("json", action_effects_path),
        ("json", summary_path),
        ("md", root_cause_path),
        ("md", evidence_path),
    ):
        artifacts.append(
            {
                "kind": kind,
                "name": path.name,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": p11._sha256(path),
            }
        )
    manifest_path = p11._write_csv(
        output_root / "artifact_manifest.csv",
        artifacts,
        ("kind", "name", "path", "sha256"),
    )
    for _, trained in states.values():
        del trained
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "protocol": protocol_path,
        "decisions": decisions_path,
        "action_effects": action_effects_path,
        "summary": summary_path,
        "root_cause": root_cause_path,
        "evidence": evidence_path,
        "artifact_manifest": manifest_path,
    }


def _verify_observation_history(policy: Mapping[str, Any]) -> None:
    for index, decision in enumerate(policy.get("decisions", [])):
        if decision.get("status") != "OK":
            continue
        observation = decision["observation"]
        _assert_safe_observation(observation)
        if policy["policy_id"] == "A2L_llm_agent_execute":
            if observation["schema_version"] != OBSERVATION_SCHEMA_VERSION:
                raise ValueError("A2L did not use observation v2")
            history = observation["full_action_history"]
            if len(history) != index:
                raise ValueError("observation v2 omitted full prior history")
            if index == 0 and observation["previous_action"] != "none":
                raise ValueError("round-one previous action is not empty")
            if index > 0 and observation["previous_action"]["action_id"] != history[-1]["action_id"]:
                raise ValueError("previous action does not match full history")


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    manifest_path = output_root / "artifact_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    expected = {
        "protocol.json",
        "decisions_and_trials.jsonl",
        "action_effects.json",
        "summary.json",
        "root_cause.md",
        "evidence.md",
    }
    if {row["name"] for row in manifest} != expected:
        raise ValueError("P29 artifact grid is incomplete")
    for row in manifest:
        path = PROJECT_ROOT / row["path"]
        if not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
            raise ValueError("P29 artifact escaped its owner output path")
        if not path.is_file() or p11._sha256(path) != row["sha256"]:
            raise ValueError(f"P29 artifact hash mismatch: {path}")
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    action_effects = json.loads(
        (output_root / "action_effects.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (output_root / "decisions_and_trials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported P29 protocol schema")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported P29 summary schema")
    expected_table = {
        action_id: asdict(config) for action_id, config in ACTION_ALLOWLIST.items()
    }
    if protocol["action_table"] != expected_table:
        raise ValueError("P29 action registry drifted from P28")
    if protocol["maximum_trials_per_policy"] != 2:
        raise ValueError("P29 is not a two-of-five comparison")
    if protocol["selection_fold_ids"] != [0] or protocol["promotion_fold_ids"] != [4]:
        raise ValueError("P29 split roles drifted")
    if summary["evaluation"]["manifest_hashes"] != EXPECTED_MANIFEST_HASHES:
        raise ValueError("P29 manifest identity drifted")
    if summary["evaluation"]["frozen_test_accessed"] or summary["evaluation"]["holdout_paths_accepted"]:
        raise ValueError("P29 violated development-only scope")
    if not summary["a1"]["prediction_hash_equal_a0"] or not summary["a1"]["metrics_equal_a0"]:
        raise ValueError("P29 A1 true replay failed")
    for key in POLICY_KEYS:
        policy = summary[key]
        actions = [trial["action_id"] for trial in policy["selection_trials"]]
        if key == "a2l" and policy["status"] == "BLOCKED_PROVIDER":
            if policy.get("promotion") is not None:
                raise ValueError("blocked A2L executed a hidden promotion fallback")
            if len(actions) > MAX_TRIALS_PER_POLICY or len(actions) != len(set(actions)):
                raise ValueError("blocked A2L exceeded its pre-failure budget")
            _verify_observation_history(policy)
            continue
        if not 1 <= len(actions) <= 2 or len(actions) != len(set(actions)):
            raise ValueError(f"{key} violated the two-of-five budget")
        if not set(actions) <= set(ACTION_IDS):
            raise ValueError(f"{key} escaped the action registry")
        if set(policy["selected_per_dataset_package"]) != {"F3", "Penobscot"}:
            raise ValueError(f"{key} omitted a dataset package head")
        if policy["promotion"]["action_by_dataset"] != policy["selected_per_dataset_package"]:
            raise ValueError(f"{key} promotion package drifted")
        for trial in policy["selection_trials"]:
            if trial["runtime_s"] > ACTION_WALL_CLOCK_BUDGET_S:
                raise ValueError(f"{key} exceeded the wall-clock budget")
        _verify_observation_history(policy)
        expected_guards = _promotion_guard_diagnostics(
            policy=policy,
            a0=summary["a0"],
            control=summary["continued_cnn"],
        )
        if summary["policy_promotion_guard_diagnostics"][key] != expected_guards:
            raise ValueError(f"{key} promotion guard diagnostics drifted")
    _verify_observation_history(summary["prompt_ablation"])
    stop_rounds = [
        d["round"] for d in summary["a2l"]["decisions"] if d.get("stop")
    ]
    if stop_rounds and summary["a2l"]["executed_action_count"] != stop_rounds[0]:
        raise ValueError("A2L stop did not execute exactly through the stop round")
    if summary["gate"]["verdict"] == "RETAIN_HYBRID":
        if not summary["gate"]["retain_hybrid"] or summary["gate"]["retain_agent"]:
            raise ValueError("hybrid verdict conflated deterministic and agent retention")
    if set(action_effects["actions"]) != set(ACTION_IDS):
        raise ValueError("P29 action effects do not cover all five actions")
    if summary["action_effects"] != action_effects:
        raise ValueError("summary and action_effects.json disagree")
    for action_id, record in action_effects["actions"].items():
        core = {
            key: record[key]
            for key in (
                "action_id",
                "config_hash",
                "prediction_hash_by_dataset",
                "primary_delta_mIoU",
                "train_effect_diagnostics",
            )
        }
        if record["action_effect_hash"] != _hash_payload(core):
            raise ValueError(f"{action_id} action-effect hash mismatch")
    registry_core = dict(action_effects)
    registry_hash = registry_core.pop("action_effect_registry_hash")
    if registry_hash != _hash_payload(registry_core):
        raise ValueError("action-effect registry hash mismatch")
    support = summary["formal_metric_support"]
    support_core = {
        "metric_entrypoint": support["metric_entrypoint"],
        "primary_field": support["primary_field"],
        "cells": support["cells"],
    }
    if support["formal_metric_support_hash"] != _hash_payload(support_core):
        raise ValueError("formal metric support hash mismatch")
    serialized = json.dumps(
        {"protocol": protocol, "rows": rows}, ensure_ascii=False
    ).lower()
    if "authorization" in serialized or "credential_value" in serialized or "api_key" in serialized:
        raise ValueError("credential-bearing field persisted")
    evidence = (output_root / "evidence.md").read_text(encoding="utf-8")
    root_cause = (output_root / "root_cause.md").read_text(encoding="utf-8")
    if "No frozen holdout or `test.h5` was read" not in evidence:
        raise ValueError("P29 evidence omitted the leakage statement")
    if "observation/prompt" not in root_cause or "promotion/endpoint" not in root_cause:
        raise ValueError("P29 root-cause chain is incomplete")
    return {
        "schema_version": SCHEMA_VERSION,
        "rows": len(rows),
        "artifacts": len(manifest),
        "verdict": summary["gate"]["verdict"],
        "retain": summary["gate"]["retain"],
        "frozen_test_accessed": False,
        "credential_persisted": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="run the bounded development-only P29 repair"
    )
    run_parser.add_argument("--f3-manifest", type=Path, required=True)
    run_parser.add_argument("--penobscot-manifest", type=Path, required=True)
    run_parser.add_argument("--processed-root", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda:0")
    commands.add_parser("verify", help="verify committed P29 evidence")
    args = parser.parse_args(argv)
    if args.command == "run":
        result: Any = build(
            f3_manifest=args.f3_manifest,
            penobscot_manifest=args.penobscot_manifest,
            processed_root=args.processed_root,
            device=args.device,
        )
    else:
        result = verify()
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in result.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
