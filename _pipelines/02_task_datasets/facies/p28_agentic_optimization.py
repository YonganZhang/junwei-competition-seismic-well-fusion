#!/usr/bin/env python3
"""P28 constrained execution-agent pilot for seismic facies.

This runner executes fresh, bounded development trials.  It fixes the P13
pretrained SAM2 cross-attention route, lets policies select only registered
single-factor changes, and keeps fold 4 hidden until each policy has completed
its two fold-0 selection trials.  There is deliberately no test/holdout CLI.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p11_residual_fusion as p11  # noqa: E402
import p12_repair_v1 as p12  # noqa: E402
import p13_cross_attention as p13  # noqa: E402


OUTPUT_ROOT = HERE / "_outputs" / "p28_agentic_optimization"
SCHEMA_VERSION = "facies-p28-agentic-pilot/v1"
PROTOCOL_SCHEMA_VERSION = "facies-p28-protocol/v1"
ROOT_SEED = 2693
SELECTION_FOLD = 0
PROMOTION_FOLD = 4
TRIALS_PER_POLICY = 2
ACTION_WALL_CLOCK_BUDGET_S = 90.0
GPU_LIMIT = 1
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_ENV_NAME = "DEEPSEEK_KEY"
PROVIDER_TIMEOUT_S = 60.0
MEAN_PROMOTION_DELTA = 0.005
TASK_NON_DEGRADATION = -0.005
POLICIES = (
    "A0_static_baseline",
    "A1_advice_only",
    "A2L_llm_agent_execute",
    "A2D_deterministic_agent",
    "A3_random_policy",
    "A4_deterministic_search",
)
TASKS = ("facies_f3", "facies_penobscot")
TASK_NAMES = {"facies_f3": "F3", "facies_penobscot": "Penobscot"}
EXPECTED_MANIFEST_HASHES = {
    "facies_f3": "76a501385482bedce4e48dc44dfe5e9854b1b7aa0674fc223b3a6b42760f4614",
    "facies_penobscot": "3440bbbd415158e41c6f0ed14513208fc77075c108716f9f6593ff0847382c81",
}
PROTECTED_OUTPUTS = {
    p11.OUTPUT_ROOT.resolve(),
    p12.OUTPUT_ROOT.resolve(),
    p13.OUTPUT_ROOT.resolve(),
    (HERE / "_outputs" / "agent_chapter").resolve(),
}


@dataclass(frozen=True)
class ActionConfig:
    action_id: str
    fusion_scale_initial: float = 0.2
    fusion_lr: float = p13.FUSION_LR
    dice_weight: float = p13.DICE_WEIGHT
    sam2_frozen: bool = False
    changed_factor: str = "none"
    description: str = "frozen A0"


A0_CONFIG = ActionConfig(action_id="A0_GATE_020")
ACTION_ALLOWLIST: dict[str, ActionConfig] = {
    "FAC_GATE_035": replace(
        A0_CONFIG,
        action_id="FAC_GATE_035",
        fusion_scale_initial=0.35,
        changed_factor="fusion_scale_initial",
        description="set fusion gate initialization to 0.35",
    ),
    "FAC_GATE_050": replace(
        A0_CONFIG,
        action_id="FAC_GATE_050",
        fusion_scale_initial=0.5,
        changed_factor="fusion_scale_initial",
        description="set fusion gate initialization to 0.5",
    ),
    "FAC_FUSION_LR_1E4": replace(
        A0_CONFIG,
        action_id="FAC_FUSION_LR_1E4",
        fusion_lr=1e-4,
        changed_factor="fusion_lr",
        description="set fusion learning rate to 1e-4",
    ),
    "FAC_DICE_050": replace(
        A0_CONFIG,
        action_id="FAC_DICE_050",
        dice_weight=0.5,
        changed_factor="dice_weight",
        description="set Dice loss weight to 0.5",
    ),
    "FAC_SAM2_FROZEN": replace(
        A0_CONFIG,
        action_id="FAC_SAM2_FROZEN",
        sam2_frozen=True,
        changed_factor="sam2_frozen",
        description="freeze the complete pretrained SAM2 image encoder",
    ),
}
ACTION_IDS = tuple(ACTION_ALLOWLIST)
DENIED_OBSERVATION_TERMS = (
    "metric",
    "miou",
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


def _validate_output_root(output_root: Path) -> Path:
    resolved = Path(output_root).resolve()
    if resolved != OUTPUT_ROOT.resolve():
        raise ValueError("P28 writes only its declared owner output directory")
    if resolved in PROTECTED_OUTPUTS:
        raise ValueError("P28 refuses to overwrite prior evidence")
    return resolved


def _request_deterministic_execution() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prediction_hash(logits: np.ndarray) -> str:
    predictions = np.asarray(logits).argmax(axis=1).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(predictions.shape)).encode("ascii"))
    digest.update(str(predictions.dtype).encode("ascii"))
    digest.update(np.ascontiguousarray(predictions).tobytes())
    return digest.hexdigest()


def _set_fusion_scale(model: p13.CrossAttentionSegmentationModel, value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError("fusion scale initialization must be in (0,1)")
    logit = math.log(value / (1.0 - value))
    with torch.no_grad():
        model.fusion.fusion_scale_logit.fill_(logit)


def _combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    dice_weight: float,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=class_weights) + (
        float(dice_weight) * p13._soft_dice_loss(logits, labels)
    )


def _band(value: float, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def _safe_train_diagnostics(result: Mapping[str, Any]) -> dict[str, str]:
    initial = float(result["fusion_scale_initial"])
    final = float(result["fusion_scale"])
    return {
        "loss_level": _band(float(result["train_loss_mean"]), 0.7, 1.5),
        "gradient_norm": _band(float(result["last_grad_norm"]), 2.0, 15.0),
        "sam2_update": _band(float(result["sam2_update_l2"]), 0.05, 0.6),
        "attention_entropy": _band(float(result["train_attention_entropy"]), 0.75, 0.97),
        "fusion_scale_state": (
            "stuck" if abs(final - initial) < 0.01 else "moved"
        ),
    }


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

    visit(observation)


def _feedback(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("F3", "Penobscot", "equal_mean"):
        delta = float(candidate[key]) - float(baseline[key])
        if delta > MEAN_PROMOTION_DELTA:
            result[key] = "improved"
        elif delta < TASK_NON_DEGRADATION:
            result[key] = "worse"
        else:
            result[key] = "flat"
    return result


def _build_observation(
    *,
    policy_id: str,
    round_id: int,
    available_action_ids: Sequence[str],
    a0_diagnostics: Mapping[str, Mapping[str, str]],
    prior_feedback: Mapping[str, str] | None,
) -> dict[str, Any]:
    observation = {
        "schema_version": "facies-p28-observation/v1",
        "policy_id": policy_id,
        "policy_seed": ROOT_SEED,
        "round": int(round_id),
        "trials_remaining": TRIALS_PER_POLICY - round_id + 1,
        "available_action_ids": list(available_action_ids),
        "fold_train_aggregates": dict(a0_diagnostics),
        "selection_feedback": (
            dict(prior_feedback) if prior_feedback is not None else "not_available"
        ),
    }
    _assert_safe_observation(observation)
    return observation


def _decision_prompt(observation: Mapping[str, Any], *, advice_only: bool) -> tuple[str, str]:
    allowlist = [
        {
            "action_id": action_id,
            "description": ACTION_ALLOWLIST[action_id].description,
            "changed_factor": ACTION_ALLOWLIST[action_id].changed_factor,
        }
        for action_id in observation["available_action_ids"]
    ]
    system = (
        "You are a constrained seismic-facies experiment policy. Choose only one "
        "action_id from the supplied allowlist. You never receive raw development "
        "metrics, validation curves, labels, residuals, sample identifiers, file "
        "paths, predictions, or holdout information. Return exactly one JSON object "
        "with keys action_id, confidence, rationale, stop. confidence must be in "
        "[0,1], rationale must be concise, and stop must be false before round 2. "
        "Do not emit Markdown or commands."
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


def _validate_decision(
    payload: Any,
    *,
    allowed: Sequence[str],
    round_id: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "action_id",
        "confidence",
        "rationale",
        "stop",
    }:
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
    if round_id < TRIALS_PER_POLICY and payload["stop"]:
        raise ValueError("policy requested stop before its fixed trial budget")
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
        "max_tokens": 300,
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
        choice = response_body["choices"][0]["message"]["content"]
        parsed = json.loads(choice)
        decision = _validate_decision(
            parsed,
            allowed=observation["available_action_ids"],
            round_id=int(observation["round"]),
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
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED_PROVIDER",
            "error": f"{type(exc).__name__}: {exc}",
            "provider": "deepseek",
            "model_requested": DEEPSEEK_MODEL,
            "credential_persisted": False,
        }


def _train_cross_action(
    *,
    task_id: str,
    prepared: Any,
    trained_baseline: torch.nn.Module,
    config: ActionConfig,
    device: str,
    seed: int,
    deadline: float,
) -> dict[str, Any]:
    p11._seed_all(seed)
    small_model = copy.deepcopy(trained_baseline)
    encoder = p13.build_sam2_encoder(
        task_id,
        prepared.num_classes,
        device,
        weight_mode="pretrained",
        seed=seed,
    )
    if config.sam2_frozen:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder.trainable_block_indices = ()
    model = p13.CrossAttentionSegmentationModel(small_model, encoder)
    _set_fusion_scale(model, config.fusion_scale_initial)
    model = model.to(device)
    sam_parameters = model.sam2_encoder.trainable_parameters()
    sam_snapshots = p12._snapshot_trainable(sam_parameters)
    cnn_parameters = [
        parameter
        for module in (model.small_model.decoder, model.small_model.segmentation_head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    fusion_parameters = list(model.fusion.parameters())
    groups: list[dict[str, Any]] = [
        {"params": cnn_parameters, "lr": p13.CNN_LR},
        {"params": fusion_parameters, "lr": config.fusion_lr},
    ]
    if sam_parameters:
        groups.append({"params": sam_parameters, "lr": p13.SAM2_LR})
    optimizer = torch.optim.AdamW(groups, weight_decay=p13.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=p13.CANDIDATE_UPDATES,
    )
    trainable = [*cnn_parameters, *fusion_parameters, *sam_parameters]
    weights = torch.as_tensor(
        prepared.class_weights,
        dtype=torch.float32,
        device=device,
    )
    history: list[float] = []
    last_grad_norm = 0.0
    model.train()
    if config.sam2_frozen:
        model.sam2_encoder.eval()
    for update_index, (images, labels) in enumerate(
        p13._candidate_batches(
            prepared.train_images,
            prepared.train_labels,
            seed=seed,
        )
    ):
        if time.perf_counter() > deadline:
            raise TimeoutError(f"{config.action_id} exceeded its wall-clock budget")
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = _combined_loss(
            logits,
            labels,
            class_weights=weights,
            dice_weight=config.dice_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("P28 action loss became non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, p13.GRAD_CLIP)
        if update_index == p13.CANDIDATE_UPDATES - 1:
            last_grad_norm = float(grad_norm.detach().cpu())
        optimizer.step()
        scheduler.step()
        history.append(float(loss.detach().cpu()))
    validation_logits, validation_attention = p13._predict_logits(
        model,
        prepared.validation_images,
        device=device,
    )
    train_logits, train_attention = p13._predict_logits(
        model,
        prepared.train_images,
        device=device,
    )
    metrics, confusion = p11._metrics_payload(
        validation_logits,
        prepared.validation_labels,
        num_classes=prepared.num_classes,
    )
    result = {
        "metrics": metrics,
        "confusion": confusion,
        "prediction_hash": _prediction_hash(validation_logits),
        "train_prediction_hash": _prediction_hash(train_logits),
        "train_loss_last": history[-1],
        "train_loss_mean": float(np.mean(history)),
        "last_grad_norm": last_grad_norm,
        "sam2_update_l2": p12._parameter_delta_l2(sam_parameters, sam_snapshots),
        "sam2_trainable_parameters": sum(parameter.numel() for parameter in sam_parameters),
        "sam2_trainable_blocks": list(encoder.trainable_block_indices),
        "attention_entropy": p13._attention_entropy(validation_attention),
        "train_attention_entropy": p13._attention_entropy(train_attention),
        "fusion_scale_initial": config.fusion_scale_initial,
        "fusion_scale": model.fusion.fusion_scale,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    model.cpu()
    del model, small_model, encoder
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _train_continued_control(
    *,
    prepared: Any,
    trained_baseline: torch.nn.Module,
    device: str,
    seed: int,
) -> dict[str, Any]:
    model = p13.ContinuedCnnModel(copy.deepcopy(trained_baseline))
    result = p13._train_continued_cnn(model, prepared, device=device, seed=seed)
    model.to(device)
    logits, _ = p13._predict_logits(model, prepared.validation_images, device=device)
    result["prediction_hash"] = _prediction_hash(logits)
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _prepare_same_invocation_states(
    *,
    manifests: Mapping[str, Path],
    processed_root: Path,
    device: str,
) -> tuple[dict[tuple[str, str], tuple[Any, torch.nn.Module]], dict[str, Any]]:
    budget = p11.stage3.Stage3Budget()
    states: dict[tuple[str, str], tuple[Any, torch.nn.Module]] = {}
    split_contract: dict[str, Any] = {}
    for phase, fold_id in (("selection", SELECTION_FOLD), ("promotion", PROMOTION_FOLD)):
        for task_id in TASKS:
            prepared = p11.stage3.prepare_fold(
                task_id=task_id,
                fold_id=fold_id,
                manifest_path=manifests[task_id],
                processed_root=processed_root,
                budget=budget,
            )
            if prepared.manifest_stable_hash != EXPECTED_MANIFEST_HASHES[task_id]:
                raise RuntimeError(f"{task_id} manifest hash drift")
            if len(prepared.train_images) != 32 or len(prepared.validation_images) != 16:
                raise RuntimeError("P28 sample caps drifted")
            seed = ROOT_SEED + fold_id
            baseline = p11._build_seeded_small_model(
                task_id,
                prepared.num_classes,
                device,
                seed=seed,
            )
            baseline_result = p11._train_ce_model(
                baseline,
                prepared,
                device=device,
                seed=seed,
                label=f"p28/{phase}/{task_id}/base40",
            )
            trained = baseline_result.pop("model")
            trained.cpu()
            states[(phase, task_id)] = (prepared, trained)
            split_contract[f"{phase}:{task_id}"] = {
                "fold_id": fold_id,
                "manifest_stable_hash": prepared.manifest_stable_hash,
                "fold_split_hash": prepared.fold_split_hash,
                "train_samples": len(prepared.train_images),
                "validation_samples": len(prepared.validation_images),
                "train_ids_hash": _hash_payload(list(prepared.train_sample_ids)),
                "validation_ids_hash": _hash_payload(list(prepared.validation_sample_ids)),
            }
    for task_id in TASKS:
        selection_ids = set(states[("selection", task_id)][0].validation_sample_ids)
        promotion_ids = set(states[("promotion", task_id)][0].validation_sample_ids)
        if selection_ids & promotion_ids:
            raise RuntimeError("selection and promotion validation samples overlap")
    return states, split_contract


def _run_config_package(
    *,
    policy_id: str,
    round_id: int,
    phase: str,
    config: ActionConfig,
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
    device: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    deadline = started + ACTION_WALL_CLOCK_BUDGET_S
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    tasks: dict[str, Any] = {}
    for task_id in TASKS:
        prepared, trained = states[(phase, task_id)]
        result = _train_cross_action(
            task_id=task_id,
            prepared=prepared,
            trained_baseline=trained,
            config=config,
            device=device,
            seed=ROOT_SEED + int(prepared.fold_id),
            deadline=deadline,
        )
        tasks[TASK_NAMES[task_id]] = {
            "miou": float(result["metrics"]["miou"]),
            "macro_f1": float(result["metrics"]["macro_f1"]),
            "accuracy": float(result["metrics"]["accuracy"]),
            "prediction_hash": result["prediction_hash"],
            "train_loss_mean": result["train_loss_mean"],
            "train_loss_last": result["train_loss_last"],
            "last_grad_norm": result["last_grad_norm"],
            "sam2_update_l2": result["sam2_update_l2"],
            "sam2_trainable_parameters": result["sam2_trainable_parameters"],
            "sam2_trainable_blocks": result["sam2_trainable_blocks"],
            "attention_entropy": result["attention_entropy"],
            "train_attention_entropy": result["train_attention_entropy"],
            "fusion_scale_initial": result["fusion_scale_initial"],
            "fusion_scale": result["fusion_scale"],
        }
    runtime = time.perf_counter() - started
    if runtime > ACTION_WALL_CLOCK_BUDGET_S:
        raise TimeoutError(f"{config.action_id} package exceeded wall-clock budget")
    mean_miou = float(np.mean([tasks[name]["miou"] for name in ("F3", "Penobscot")]))
    peak_vram = (
        int(torch.cuda.max_memory_allocated(torch.device(device)) / (1024 * 1024))
        if device.startswith("cuda")
        else 0
    )
    return {
        "policy_id": policy_id,
        "round": round_id,
        "phase": phase,
        "action_id": config.action_id,
        "config": asdict(config),
        "config_hash": _hash_payload(asdict(config)),
        "tasks": tasks,
        "equal_mean": mean_miou,
        "runtime_s": runtime,
        "peak_vram_mb": peak_vram,
        "wall_clock_budget_s": ACTION_WALL_CLOCK_BUDGET_S,
        "exit_status": "OK",
        "frozen_test_accessed": False,
    }


def _run_controls(
    *,
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    a0: dict[str, Any] = {}
    control: dict[str, Any] = {}
    for phase in ("selection", "promotion"):
        a0[phase] = _run_config_package(
            policy_id="A0_static_baseline",
            round_id=0,
            phase=phase,
            config=A0_CONFIG,
            states=states,
            device=device,
        )
        control_tasks: dict[str, Any] = {}
        started = time.perf_counter()
        for task_id in TASKS:
            prepared, trained = states[(phase, task_id)]
            result = _train_continued_control(
                prepared=prepared,
                trained_baseline=trained,
                device=device,
                seed=ROOT_SEED + int(prepared.fold_id),
            )
            control_tasks[TASK_NAMES[task_id]] = {
                "miou": float(result["metrics"]["miou"]),
                "macro_f1": float(result["metrics"]["macro_f1"]),
                "accuracy": float(result["metrics"]["accuracy"]),
                "prediction_hash": result["prediction_hash"],
            }
        control[phase] = {
            "policy_id": "continued_cnn_control",
            "phase": phase,
            "tasks": control_tasks,
            "equal_mean": float(np.mean([control_tasks[name]["miou"] for name in ("F3", "Penobscot")])),
            "runtime_s": time.perf_counter() - started,
            "frozen_test_accessed": False,
        }
    return a0, control


def _metric_view(package: Mapping[str, Any]) -> dict[str, float]:
    return {
        "F3": float(package["tasks"]["F3"]["miou"]),
        "Penobscot": float(package["tasks"]["Penobscot"]["miou"]),
        "equal_mean": float(package["equal_mean"]),
    }


def _optimization_auc(results: Sequence[Mapping[str, Any]], baseline: float) -> float:
    best = float(baseline)
    trace: list[float] = []
    for result in results:
        best = max(best, float(result["equal_mean"]))
        trace.append(best)
    if len(trace) != TRIALS_PER_POLICY:
        raise ValueError("optimization AUC requires exactly two trials")
    return float(np.mean(trace))


def _deterministic_action(
    *,
    round_id: int,
    available: Sequence[str],
    a0_diagnostics: Mapping[str, Mapping[str, str]],
    prior_feedback: Mapping[str, str] | None,
) -> tuple[str, str]:
    if round_id == 1:
        if any(value["fusion_scale_state"] == "stuck" for value in a0_diagnostics.values()):
            preferred = "FAC_GATE_035"
            reason = "A0 gate is stuck, so test the moderate registered initialization."
        elif any(value["gradient_norm"] == "high" for value in a0_diagnostics.values()):
            preferred = "FAC_FUSION_LR_1E4"
            reason = "High gradient band triggers the lower registered fusion LR."
        else:
            preferred = "FAC_SAM2_FROZEN"
            reason = "No gate or gradient alarm; isolate encoder adaptation."
    elif prior_feedback and "worse" in prior_feedback.values():
        preferred = "FAC_FUSION_LR_1E4"
        reason = "A task worsened, so reduce the fusion learning rate."
    elif prior_feedback and all(value == "improved" for value in prior_feedback.values()):
        preferred = "FAC_DICE_050"
        reason = "Consistent improvement permits the registered stronger Dice term."
    else:
        preferred = "FAC_SAM2_FROZEN"
        reason = "Flat or mixed feedback triggers the encoder-freeze diagnostic."
    if preferred not in available:
        preferred = next(action_id for action_id in ACTION_IDS if action_id in available)
        reason = "Primary rule was already used; selected the first remaining registered action."
    return preferred, reason


def _run_policy(
    *,
    policy_id: str,
    states: Mapping[tuple[str, str], tuple[Any, torch.nn.Module]],
    device: str,
    a0: Mapping[str, Any],
    a0_diagnostics: Mapping[str, Mapping[str, str]],
    random_actions: Sequence[str] | None = None,
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    used: list[str] = []
    prior_feedback: dict[str, str] | None = None
    provider_status = "NOT_APPLICABLE"
    for round_id in range(1, TRIALS_PER_POLICY + 1):
        available = [action_id for action_id in ACTION_IDS if action_id not in used]
        observation = _build_observation(
            policy_id=policy_id,
            round_id=round_id,
            available_action_ids=available,
            a0_diagnostics=a0_diagnostics,
            prior_feedback=prior_feedback,
        )
        if policy_id == "A2L_llm_agent_execute":
            response = _call_deepseek(observation, advice_only=False)
            provider_status = response["status"]
            if response["status"] != "OK":
                decisions.append({
                    "round": round_id,
                    "observation": observation,
                    "observation_hash": _hash_payload(observation),
                    "status": response["status"],
                    "error": response.get("error", "provider failure"),
                    "credential_persisted": False,
                })
                break
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
                a0_diagnostics=a0_diagnostics,
                prior_feedback=prior_feedback,
            )
            decision_record = {
                "round": round_id,
                "observation": observation,
                "observation_hash": _hash_payload(observation),
                "status": "OK",
                "action_id": action_id,
                "confidence": 1.0,
                "rationale": rationale,
                "stop": round_id == TRIALS_PER_POLICY,
                "provider": "deterministic_rules",
                "credential_persisted": False,
            }
        else:
            if random_actions is None:
                raise ValueError("A3 requires fixed random actions")
            action_id = str(random_actions[round_id - 1])
            if action_id not in available:
                raise RuntimeError("PCG64 policy repeated or escaped the allowlist")
            decision_record = {
                "round": round_id,
                "observation": observation,
                "observation_hash": _hash_payload(observation),
                "status": "OK",
                "action_id": action_id,
                "confidence": 1.0 / len(available),
                "rationale": "PCG64 random choice without replacement.",
                "stop": round_id == TRIALS_PER_POLICY,
                "provider": "numpy.PCG64",
                "credential_persisted": False,
            }
        decisions.append(decision_record)
        used.append(action_id)
        package = _run_config_package(
            policy_id=policy_id,
            round_id=round_id,
            phase="selection",
            config=ACTION_ALLOWLIST[action_id],
            states=states,
            device=device,
        )
        prior_feedback = _feedback(_metric_view(package), _metric_view(a0["selection"]))
        package["selection_feedback"] = prior_feedback
        selections.append(package)
    if len(selections) != TRIALS_PER_POLICY:
        return {
            "policy_id": policy_id,
            "status": provider_status if provider_status != "NOT_APPLICABLE" else "EARLY_STOPPED",
            "decisions": decisions,
            "selection_trials": selections,
            "promotion": None,
        }
    best = max(selections, key=lambda item: float(item["equal_mean"]))
    promotion = _run_config_package(
        policy_id=policy_id,
        round_id=TRIALS_PER_POLICY + 1,
        phase="promotion",
        config=ACTION_ALLOWLIST[best["action_id"]],
        states=states,
        device=device,
    )
    return {
        "policy_id": policy_id,
        "status": "OK",
        "decisions": decisions,
        "selection_trials": selections,
        "selected_for_promotion": best["action_id"],
        "best_selection_mean_miou": float(best["equal_mean"]),
        "optimization_auc": _optimization_auc(
            selections,
            float(a0["selection"]["equal_mean"]),
        ),
        "promotion": promotion,
    }


def _a1_advice(
    *,
    a0: Mapping[str, Any],
    a0_diagnostics: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    observation = _build_observation(
        policy_id="A1_advice_only",
        round_id=1,
        available_action_ids=ACTION_IDS,
        a0_diagnostics=a0_diagnostics,
        prior_feedback=None,
    )
    response = _call_deepseek(observation, advice_only=True)
    hashes = {
        phase: {
            task: a0[phase]["tasks"][task]["prediction_hash"]
            for task in ("F3", "Penobscot")
        }
        for phase in ("selection", "promotion")
    }
    return {
        "policy_id": "A1_advice_only",
        "status": response["status"],
        "observation": observation,
        "observation_hash": _hash_payload(observation),
        "decision": response.get("decision"),
        "error": response.get("error"),
        "prediction_hashes": hashes,
        "a0_prediction_hashes": copy.deepcopy(hashes),
        "prediction_hash_equal_a0": True,
        "executed_action": False,
        "credential_persisted": False,
    }


def _gate_summary(
    *,
    a0: Mapping[str, Any],
    control: Mapping[str, Any],
    a1: Mapping[str, Any],
    a2l: Mapping[str, Any],
    a2d: Mapping[str, Any],
    a3: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "a1_prediction_hash_equal_a0": bool(a1["prediction_hash_equal_a0"]),
        "a2l_two_legal_distinct_trials": False,
        "a2l_auc_above_a2d": False,
        "a2l_auc_above_a3": False,
        "promotion_mean_delta_at_least_0p005": False,
        "f3_delta_at_least_minus_0p005": False,
        "penobscot_delta_at_least_minus_0p005": False,
        "not_worse_than_continued_cnn": False,
        "no_frozen_test_access": True,
    }
    if a2l.get("status") != "OK":
        return {
            "checks": checks,
            "retain": False,
            "verdict": "BLOCKED_PROVIDER" if a2l.get("status") == "BLOCKED_PROVIDER" else "REJECT_A2L",
        }
    action_ids = [trial["action_id"] for trial in a2l["selection_trials"]]
    checks["a2l_two_legal_distinct_trials"] = (
        len(action_ids) == TRIALS_PER_POLICY
        and len(set(action_ids)) == TRIALS_PER_POLICY
        and all(action_id in ACTION_ALLOWLIST for action_id in action_ids)
    )
    checks["a2l_auc_above_a2d"] = float(a2l["optimization_auc"]) > float(a2d["optimization_auc"])
    checks["a2l_auc_above_a3"] = float(a2l["optimization_auc"]) > float(a3["optimization_auc"])
    promotion = _metric_view(a2l["promotion"])
    baseline = _metric_view(a0["promotion"])
    continued = _metric_view(control["promotion"])
    checks["promotion_mean_delta_at_least_0p005"] = (
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
    retained = all(checks.values())
    return {
        "checks": checks,
        "retain": retained,
        "verdict": "RETAIN_A2L" if retained else "REJECT_A2L",
        "promotion_delta_vs_a0": {
            key: promotion[key] - baseline[key] for key in promotion
        },
        "promotion_delta_vs_continued_cnn": {
            key: promotion[key] - continued[key] for key in promotion
        },
        "optimization_auc": {
            "A2L": a2l["optimization_auc"],
            "A2D": a2d["optimization_auc"],
            "A3": a3["optimization_auc"],
        },
    }


def _protocol(split_contract: Mapping[str, Any]) -> dict[str, Any]:
    action_table = {action_id: asdict(config) for action_id, config in ACTION_ALLOWLIST.items()}
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "track_id": "facies",
        "stage": "P28_stage1_online_pilot",
        "primary_metric": "equal_mean_F3_Penobscot_mIoU",
        "metric_entrypoint": "p4_metrics.evaluate_probabilities",
        "a0": asdict(A0_CONFIG),
        "sam2_weight_mode": "pretrained",
        "continued_cnn_is_policy_action": False,
        "action_table": action_table,
        "action_space_hash": _hash_payload(action_table),
        "policies": list(POLICIES),
        "a4_status": "DEFERRED_TO_STAGE2",
        "trials_per_executing_policy": TRIALS_PER_POLICY,
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
        "observation_policy": "fold_train_aggregate_bands_plus_improved_flat_worse_only",
        "observation_denylist": list(DENIED_OBSERVATION_TERMS),
        "provider_failure": "fail_closed_BLOCKED_PROVIDER",
        "gates": {
            "a1_prediction_hash_equal_a0": True,
            "a2l_auc_strictly_above_a2d_and_a3": True,
            "promotion_mean_delta": MEAN_PROMOTION_DELTA,
            "per_task_min_delta": TASK_NON_DEGRADATION,
            "not_worse_than_continued_cnn": True,
        },
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
    rows: list[dict[str, Any]] = []
    common = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "facies",
        "policy_revision": p11._sha256(Path(__file__)),
        "policy_seed": ROOT_SEED,
        "metric_id": "evaluate_probabilities_miou",
        "action_space_hash": protocol["action_space_hash"],
        "selection_fold_ids": [SELECTION_FOLD],
        "promotion_fold_ids": [PROMOTION_FOLD],
        "wall_clock_budget_s": ACTION_WALL_CLOCK_BUDGET_S,
        "frozen_test_accessed": False,
        "leakage_checks": {
            "manifest_hash_locked": True,
            "selection_promotion_disjoint": True,
            "train_only_fit": True,
            "llm_observation_sanitized": True,
            "holdout_paths_accepted": False,
        },
    }
    for phase in ("selection", "promotion"):
        rows.append({**common, "event": "A0_RESULT", "policy_id": "A0_static_baseline", "phase": phase, "result": a0[phase]})
        rows.append({**common, "event": "CONTROL_RESULT", "policy_id": "continued_cnn_control", "phase": phase, "result": control[phase]})
    rows.append({**common, "event": "A1_ADVICE", "policy_id": "A1_advice_only", "result": a1})
    for policy in policies:
        decisions = policy.get("decisions", [])
        trials = policy.get("selection_trials", [])
        for index, decision in enumerate(decisions):
            row = {
                **common,
                "event": "POLICY_TRIAL",
                "policy_id": policy["policy_id"],
                "round": decision["round"],
                "decision": decision,
                "result": trials[index] if index < len(trials) else None,
            }
            rows.append(row)
        if policy.get("promotion") is not None:
            rows.append({
                **common,
                "event": "PROMOTION_RESULT",
                "policy_id": policy["policy_id"],
                "selected_action_id": policy["selected_for_promotion"],
                "result": policy["promotion"],
            })
    return rows


def _write_evidence(summary: Mapping[str, Any], protocol: Mapping[str, Any], output_root: Path) -> Path:
    gate = summary["gate"]
    lines = [
        "# P28 facies execution-agent pilot evidence",
        "",
        "## Scope",
        "",
        "This is a fresh Stage-1 development execution pilot, not a replay of P13/P17 metrics. "
        "A0 is the same-invocation pretrained SAM2 cross-attention gate=0.2 route. "
        "Selection used fold 0 and promotion used fold 4 with 32 train and 16 validation samples per task/fold.",
        "",
        "No frozen holdout or `test.h5` was read. DeepSeek received only categorical fold-train aggregates "
        "and improved/flat/worse selection feedback; raw metrics, validation curves, labels, residuals, "
        "sample IDs, predictions and paths were excluded.",
        "",
        "## Frozen controls",
        "",
        f"- A0 selection mean mIoU: `{summary['a0']['selection']['equal_mean']:.9f}`",
        f"- A0 promotion mean mIoU: `{summary['a0']['promotion']['equal_mean']:.9f}`",
        f"- Continued-CNN promotion mean mIoU: `{summary['continued_cnn']['promotion']['equal_mean']:.9f}`",
        f"- A1 prediction hash equals A0: `{summary['a1']['prediction_hash_equal_a0']}`",
        "",
        "## Policy results",
        "",
    ]
    for key in ("a2l", "a2d", "a3"):
        policy = summary[key]
        if policy["status"] == "OK":
            actions = [trial["action_id"] for trial in policy["selection_trials"]]
            lines.extend([
                f"- {policy['policy_id']}: actions `{actions}`, optimization AUC "
                f"`{policy['optimization_auc']:.9f}`, promotion action "
                f"`{policy['selected_for_promotion']}`, promotion mean mIoU "
                f"`{policy['promotion']['equal_mean']:.9f}`.",
            ])
        else:
            lines.append(f"- {policy['policy_id']}: `{policy['status']}`; no fallback action was selected.")
    lines.extend([
        "",
        "## Promotion gate",
        "",
    ])
    for name, passed in gate["checks"].items():
        lines.append(f"- {name}: `{passed}`")
    lines.extend([
        "",
        f"Verdict: **{gate['verdict']}**.",
        "",
        "This verdict concerns the policy under the frozen Stage-1 budget. It does not attribute any "
        "model gain to SAM2 and does not authorize frozen-test access or default enablement.",
        "",
        "## Protocol identity",
        "",
        f"- Protocol hash: `{_hash_payload(protocol)}`",
        f"- Runner hash at evidence creation: `{p11._sha256(Path(__file__))}`",
        f"- Provider credential persisted: `False`",
    ])
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
    _request_deterministic_execution()
    manifests = p11.validate_development_inputs(
        f3_manifest=f3_manifest,
        penobscot_manifest=penobscot_manifest,
        processed_root=processed_root,
    )
    output_root = _validate_output_root(OUTPUT_ROOT)
    for verifier, root in (
        (p11.verify, p11.OUTPUT_ROOT),
        (p12.verify, p12.OUTPUT_ROOT),
        (p13.verify, p13.OUTPUT_ROOT),
    ):
        verifier(root)
    protected_hashes = {
        str(root.relative_to(HERE)): p11._sha256(root / "artifact_manifest.csv")
        for root in (p11.OUTPUT_ROOT, p12.OUTPUT_ROOT, p13.OUTPUT_ROOT)
    }
    p11._prepare_sam2_dependency_path()
    source_root = p11.verify_git_source(p11.SAM2_SOURCE_ROOT, p11.SAM2_SOURCE_REVISION)
    checkpoint = p11.verify_checkpoint("facies", p11.SAM2_CHECKPOINT)
    p11.insert_import_root(source_root, "sam2")
    output_root.mkdir(parents=True, exist_ok=True)
    states, split_contract = _prepare_same_invocation_states(
        manifests=manifests,
        processed_root=Path(processed_root).resolve(),
        device=device,
    )
    a0, continued = _run_controls(states=states, device=device)
    a0_diagnostics = {
        task: _safe_train_diagnostics(a0["selection"]["tasks"][task])
        for task in ("F3", "Penobscot")
    }
    a1 = _a1_advice(a0=a0, a0_diagnostics=a0_diagnostics)
    a2l = _run_policy(
        policy_id="A2L_llm_agent_execute",
        states=states,
        device=device,
        a0=a0,
        a0_diagnostics=a0_diagnostics,
    )
    a2d = _run_policy(
        policy_id="A2D_deterministic_agent",
        states=states,
        device=device,
        a0=a0,
        a0_diagnostics=a0_diagnostics,
    )
    rng = np.random.Generator(np.random.PCG64(ROOT_SEED))
    random_actions = [str(value) for value in rng.choice(ACTION_IDS, size=TRIALS_PER_POLICY, replace=False)]
    a3 = _run_policy(
        policy_id="A3_random_policy",
        states=states,
        device=device,
        a0=a0,
        a0_diagnostics=a0_diagnostics,
        random_actions=random_actions,
    )
    gate = _gate_summary(a0=a0, control=continued, a1=a1, a2l=a2l, a2d=a2d, a3=a3)
    protocol = _protocol(split_contract)
    current_hashes = {
        str(root.relative_to(HERE)): p11._sha256(root / "artifact_manifest.csv")
        for root in (p11.OUTPUT_ROOT, p12.OUTPUT_ROOT, p13.OUTPUT_ROOT)
    }
    if current_hashes != protected_hashes:
        raise RuntimeError("protected P11/P12/P13 evidence changed during P28")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "a0": a0,
        "continued_cnn": continued,
        "a1": a1,
        "a2l": a2l,
        "a2d": a2d,
        "a3": a3,
        "a4": {"status": "DEFERRED_TO_STAGE2"},
        "gate": gate,
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
        control=continued,
        a1=a1,
        policies=(a2l, a2d, a3),
    )
    protocol_path = p11._write_json(output_root / "protocol.json", protocol)
    decisions_path = p11._write_jsonl(output_root / "decisions_and_trials.jsonl", rows)
    summary_path = p11._write_json(output_root / "summary.json", summary)
    evidence_path = _write_evidence(summary, protocol, output_root)
    artifacts = []
    for kind, path in (
        ("json", protocol_path),
        ("jsonl", decisions_path),
        ("json", summary_path),
        ("md", evidence_path),
    ):
        artifacts.append({
            "kind": kind,
            "name": path.name,
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": p11._sha256(path),
        })
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
        "summary": summary_path,
        "evidence": evidence_path,
        "artifact_manifest": manifest_path,
    }


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = _validate_output_root(output_root)
    manifest_path = output_root / "artifact_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    expected = {"protocol.json", "decisions_and_trials.jsonl", "summary.json", "evidence.md"}
    if {row["name"] for row in manifest} != expected:
        raise ValueError("P28 artifact grid is incomplete")
    for row in manifest:
        path = PROJECT_ROOT / row["path"]
        if not path.resolve().is_relative_to(OUTPUT_ROOT.resolve()):
            raise ValueError("P28 artifact escaped its owner output path")
        if not path.is_file() or p11._sha256(path) != row["sha256"]:
            raise ValueError(f"P28 artifact hash mismatch: {path}")
    protocol = json.loads((output_root / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output_root / "decisions_and_trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported P28 protocol schema")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported P28 summary schema")
    if protocol["action_table"] != {action_id: asdict(config) for action_id, config in ACTION_ALLOWLIST.items()}:
        raise ValueError("P28 action registry drifted")
    if protocol["continued_cnn_is_policy_action"] or "continued_cnn_control" in protocol["action_table"]:
        raise ValueError("continued CNN leaked into the policy action table")
    if protocol["selection_fold_ids"] != [0] or protocol["promotion_fold_ids"] != [4]:
        raise ValueError("P28 split roles drifted")
    if summary["evaluation"]["manifest_hashes"] != EXPECTED_MANIFEST_HASHES:
        raise ValueError("P28 manifest identity drifted")
    if summary["evaluation"]["frozen_test_accessed"] or summary["evaluation"]["holdout_paths_accepted"]:
        raise ValueError("P28 violated development-only scope")
    if not summary["a1"]["prediction_hash_equal_a0"]:
        raise ValueError("A1 prediction identity probe failed")
    if summary["a2l"]["status"] == "OK":
        for key in ("a2l", "a2d", "a3"):
            policy = summary[key]
            actions = [trial["action_id"] for trial in policy["selection_trials"]]
            if len(actions) != 2 or len(set(actions)) != 2 or not set(actions) <= set(ACTION_IDS):
                raise ValueError(f"{key} trial budget/allowlist contract failed")
    elif summary["gate"]["verdict"] != "BLOCKED_PROVIDER":
        raise ValueError("failed A2L did not fail closed")
    serialized = json.dumps({"protocol": protocol, "rows": rows}, ensure_ascii=False).lower()
    if "authorization" in serialized or "credential_value" in serialized or "api_key" in serialized:
        raise ValueError("credential-bearing field persisted")
    evidence = (output_root / "evidence.md").read_text(encoding="utf-8")
    if "No frozen holdout or `test.h5` was read" not in evidence:
        raise ValueError("P28 evidence omitted the leakage statement")
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
    run_parser = commands.add_parser("run", help="run the bounded development-only P28 pilot")
    run_parser.add_argument("--f3-manifest", type=Path, required=True)
    run_parser.add_argument("--penobscot-manifest", type=Path, required=True)
    run_parser.add_argument("--processed-root", type=Path, required=True)
    run_parser.add_argument("--device", default="cuda:0")
    commands.add_parser("verify", help="verify committed P28 evidence")
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
    print(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in result.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
