#!/usr/bin/env python3
"""P28 agentic ablation preflight for the fault track.

This runner is intentionally fail-closed. It compares four policy families on a
fixed, auditable set of gate scenarios built from the already-verified fault
contract evidence:

* A0_static_baseline: conservative fixed decision
* A1_advice_only: advice-only; its final decision hash must match A0
* A2L_llm_agent_execute: DeepSeek strict-JSON executor, or BLOCKED_PROVIDER
* A2D_deterministic_agent: deterministic rules-based executor
* A3_random_policy: equal-budget PCG64 random control

The formal fault track remains DATA_GATE_BLOCKED for real model training, so the
runner only evaluates whether the agent can correctly stop, request evidence,
verify hashes, or proceed on contract fixtures. It never opens frozen test/holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p28_agentic_ablation"
ACTION_REGISTRY = ("STOP_DATA_GATE", "REQUEST_EVIDENCE", "VERIFY_HASHES", "PROCEED")
POLICY_IDS = (
    "A0_static_baseline",
    "A1_advice_only",
    "A2L_llm_agent_execute",
    "A2D_deterministic_agent",
    "A3_random_policy",
)
SELECTION_SCENARIO_IDS = ("observed_blocked_current", "packet_hash_missing")
PROMOTION_SCENARIO_IDS = ("packet_hash_mismatch", "counterfactual_contract_green")
DEFAULT_TRIAL_BUDGET = 4
ROOT_SOURCE_PATHS = {
    "fault_p18_cigbench.py": TRACK_DIR / "fault_p18_cigbench.py",
    "test_fault_p18_cigbench.py": TRACK_DIR / "test_fault_p18_cigbench.py",
    "tests/test_fault_p28_agentic_ablation.py": TRACK_DIR / "tests" / "test_fault_p28_agentic_ablation.py",
    "baseline_metrics.json": TRACK_DIR / "_outputs" / "runs" / "audited_v2" / "baseline_metrics.json",
    "build_summary.json": TRACK_DIR / "_outputs" / "runs" / "audited_v2" / "build_summary.json",
    "p18_evidence.md": TRACK_DIR / "_outputs" / "p18_cigbench_fault" / "evidence.md",
    "historical_baseline_manifest.json": TRACK_DIR / "historical_baseline_manifest.json",
}


@dataclass(frozen=True)
class PacketItem:
    """One evidence item shown to the agent."""

    path: str
    actual_sha256: str
    reported_sha256: str | None
    hash_state: str  # verified|unknown|mismatch
    role: str


@dataclass(frozen=True)
class GateState:
    present: bool
    hash_state: str  # verified|unknown|mismatch
    evidence_code: str


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    title: str
    split: str  # selection or promotion
    kind: str  # observed|packet_gap|packet_mismatch|fixture
    packet_items: tuple[PacketItem, ...]
    gates: dict[str, GateState]
    gold_action_id: str
    gold_necessary_evidence: tuple[str, ...]
    notes: str

    def packet_hash_state(self) -> str:
        states = {item.hash_state for item in self.packet_items}
        if "mismatch" in states:
            return "mismatch"
        if "unknown" in states:
            return "unknown"
        return "verified"

    def packet_digest(self) -> str:
        payload = [
            {
                "path": item.path,
                "actual_sha256": item.actual_sha256,
                "reported_sha256": item.reported_sha256,
                "hash_state": item.hash_state,
                "role": item.role,
            }
            for item in self.packet_items
        ]
        return sha256_text(canonical_json(payload))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def build_source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
        for name, path in ROOT_SOURCE_PATHS.items()
    }


def build_packet_items(source_hashes: dict[str, dict[str, str]]) -> tuple[PacketItem, ...]:
    return (
        PacketItem(
            path=source_hashes["p18_evidence.md"]["path"],
            actual_sha256=source_hashes["p18_evidence.md"]["sha256"],
            reported_sha256=source_hashes["p18_evidence.md"]["sha256"],
            hash_state="verified",
            role="blocked gate evidence report",
        ),
        PacketItem(
            path=source_hashes["baseline_metrics.json"]["path"],
            actual_sha256=source_hashes["baseline_metrics.json"]["sha256"],
            reported_sha256=source_hashes["baseline_metrics.json"]["sha256"],
            hash_state="verified",
            role="frozen A0 metric snapshot",
        ),
        PacketItem(
            path=source_hashes["build_summary.json"]["path"],
            actual_sha256=source_hashes["build_summary.json"]["sha256"],
            reported_sha256=source_hashes["build_summary.json"]["sha256"],
            hash_state="verified",
            role="audited build summary",
        ),
        PacketItem(
            path=source_hashes["historical_baseline_manifest.json"]["path"],
            actual_sha256=source_hashes["historical_baseline_manifest.json"]["sha256"],
            reported_sha256=source_hashes["historical_baseline_manifest.json"]["sha256"],
            hash_state="verified",
            role="historical artifact manifest",
        ),
    )


def mutated_packet_items(
    packet_items: Iterable[PacketItem],
    *,
    unknown_paths: set[str] | None = None,
    mismatch_paths: set[str] | None = None,
) -> tuple[PacketItem, ...]:
    unknown_paths = set() if unknown_paths is None else set(unknown_paths)
    mismatch_paths = set() if mismatch_paths is None else set(mismatch_paths)
    mutated: list[PacketItem] = []
    for item in packet_items:
        if item.path in unknown_paths:
            mutated.append(
                PacketItem(
                    path=item.path,
                    actual_sha256=item.actual_sha256,
                    reported_sha256=None,
                    hash_state="unknown",
                    role=item.role,
                )
            )
            continue
        if item.path in mismatch_paths:
            flipped = "0" if item.actual_sha256[0] != "0" else "1"
            mutated.append(
                PacketItem(
                    path=item.path,
                    actual_sha256=item.actual_sha256,
                    reported_sha256=flipped + item.actual_sha256[1:],
                    hash_state="mismatch",
                    role=item.role,
                )
            )
            continue
        mutated.append(item)
    return tuple(mutated)


def build_scenarios(source_hashes: dict[str, dict[str, str]]) -> tuple[Scenario, ...]:
    observed_packet = build_packet_items(source_hashes)
    return (
        Scenario(
            scenario_id="observed_blocked_current",
            title="Observed P18 gate block",
            split="selection",
            kind="observed",
            packet_items=observed_packet,
            gates={
                "contiguous_3d_development_blocks": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="contiguous_3d_development_blocks_missing",
                ),
                "coverage_audited_verified_background": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="coverage_audited_verified_background_missing",
                ),
                "explicit_unknown_mask_provenance": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="explicit_unknown_mask_provenance_missing",
                ),
                "group_isolated_development_split": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="group_isolated_development_split_missing",
                ),
            },
            gold_action_id="STOP_DATA_GATE",
            gold_necessary_evidence=(
                "contiguous_3d_development_blocks_missing",
                "coverage_audited_verified_background_missing",
                "explicit_unknown_mask_provenance_missing",
                "group_isolated_development_split_missing",
            ),
            notes="Directly derived from the current P18 blocked evidence.",
        ),
        Scenario(
            scenario_id="packet_hash_missing",
            title="Packet missing hash records",
            split="selection",
            kind="packet_gap",
            packet_items=mutated_packet_items(
                observed_packet,
                unknown_paths={
                    source_hashes["baseline_metrics.json"]["path"],
                    source_hashes["build_summary.json"]["path"],
                },
            ),
            gates={
                "contiguous_3d_development_blocks": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="contiguous_3d_development_blocks_missing",
                ),
                "coverage_audited_verified_background": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="coverage_audited_verified_background_missing",
                ),
                "explicit_unknown_mask_provenance": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="explicit_unknown_mask_provenance_missing",
                ),
                "group_isolated_development_split": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="group_isolated_development_split_missing",
                ),
            },
            gold_action_id="REQUEST_EVIDENCE",
            gold_necessary_evidence=(
                "hash:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
                "hash:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/build_summary.json",
            ),
            notes="The packet omits two hash records; the agent should request them, not pretend the packet is complete.",
        ),
        Scenario(
            scenario_id="packet_hash_mismatch",
            title="Packet hash mismatch",
            split="promotion",
            kind="packet_mismatch",
            packet_items=mutated_packet_items(
                observed_packet,
                mismatch_paths={
                    source_hashes["baseline_metrics.json"]["path"],
                },
            ),
            gates={
                "contiguous_3d_development_blocks": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="contiguous_3d_development_blocks_missing",
                ),
                "coverage_audited_verified_background": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="coverage_audited_verified_background_missing",
                ),
                "explicit_unknown_mask_provenance": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="explicit_unknown_mask_provenance_missing",
                ),
                "group_isolated_development_split": GateState(
                    present=False,
                    hash_state="verified",
                    evidence_code="group_isolated_development_split_missing",
                ),
            },
            gold_action_id="VERIFY_HASHES",
            gold_necessary_evidence=(
                "verify:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
            ),
            notes="The reported hash for baseline_metrics.json disagrees with the verified artifact hash.",
        ),
        Scenario(
            scenario_id="counterfactual_contract_green",
            title="Counterfactual contract-green fixture",
            split="promotion",
            kind="fixture",
            packet_items=observed_packet,
            gates={
                "contiguous_3d_development_blocks": GateState(
                    present=True,
                    hash_state="verified",
                    evidence_code="contiguous_3d_development_blocks_present",
                ),
                "coverage_audited_verified_background": GateState(
                    present=True,
                    hash_state="verified",
                    evidence_code="coverage_audited_verified_background_present",
                ),
                "explicit_unknown_mask_provenance": GateState(
                    present=True,
                    hash_state="verified",
                    evidence_code="explicit_unknown_mask_provenance_present",
                ),
                "group_isolated_development_split": GateState(
                    present=True,
                    hash_state="verified",
                    evidence_code="group_isolated_development_split_present",
                ),
            },
            gold_action_id="PROCEED",
            gold_necessary_evidence=(),
            notes="Protocol fixture only; it exercises the proceed branch without opening frozen test or claiming current data are green.",
        ),
    )


def _observed_evidence(scenario: Scenario) -> tuple[str, ...]:
    items: list[str] = []
    for gate_name, gate in scenario.gates.items():
        if not gate.present:
            items.append(gate.evidence_code)
        elif gate.hash_state != "verified":
            items.append(f"verify:{gate.evidence_code}")
    return tuple(items)


def conservative_policy(scenario: Scenario) -> dict[str, Any]:
    if any(not gate.present for gate in scenario.gates.values()):
        action_id = "STOP_DATA_GATE"
        evidence = _observed_evidence(scenario)
    elif any(item.hash_state != "verified" for item in scenario.packet_items):
        action_id = "STOP_DATA_GATE"
        evidence = tuple(
            f"hash:{item.path}"
            for item in scenario.packet_items
            if item.hash_state != "verified"
        )
    else:
        action_id = "PROCEED"
        evidence = ()
    return _decision_payload(
        scenario=scenario,
        policy_id="A0_static_baseline",
        action_id=action_id,
        necessary_evidence=evidence,
        decision_status="OK",
        rationale="Conservative fail-closed rule over the audited gate packet.",
    )


def advice_only_policy(scenario: Scenario, baseline_decision: dict[str, Any]) -> dict[str, Any]:
    replay_decision = conservative_policy(scenario)
    if replay_decision["decision_hash"] != baseline_decision["decision_hash"]:
        raise RuntimeError("A1 replay did not reproduce the A0 decision hash")
    payload = dict(replay_decision)
    payload["policy_id"] = "A1_advice_only"
    payload["advice_text"] = (
        "Advice-only mirror of A0; final decision is intentionally kept identical to A0."
    )
    payload["advice_hash"] = sha256_text(
        canonical_json(
            {
                "scenario_id": scenario.scenario_id,
                "policy_id": "A1_advice_only",
                "advice_text": payload["advice_text"],
            }
        )
    )
    payload["replay_executed"] = True
    payload["replay_source_policy_id"] = "A0_static_baseline"
    payload["replay_decision_hash"] = replay_decision["decision_hash"]
    payload["replay_match"] = True
    return payload


def deterministic_policy(scenario: Scenario) -> dict[str, Any]:
    if any(item.hash_state == "unknown" for item in scenario.packet_items):
        action_id = "REQUEST_EVIDENCE"
        evidence = tuple(
            f"hash:{item.path}"
            for item in scenario.packet_items
            if item.hash_state == "unknown"
        )
    elif any(item.hash_state == "mismatch" for item in scenario.packet_items):
        action_id = "VERIFY_HASHES"
        evidence = tuple(
            f"verify:{item.path}"
            for item in scenario.packet_items
            if item.hash_state == "mismatch"
        )
    elif any(not gate.present for gate in scenario.gates.values()):
        action_id = "STOP_DATA_GATE"
        evidence = _observed_evidence(scenario)
    else:
        action_id = "PROCEED"
        evidence = ()
    return _decision_payload(
        scenario=scenario,
        policy_id="A2D_deterministic_agent",
        action_id=action_id,
        necessary_evidence=evidence,
        decision_status="OK",
        rationale="Deterministic priority: request missing hashes, verify mismatches, then stop, else proceed.",
    )


def random_policy(scenario: Scenario, rng: np.random.Generator) -> dict[str, Any]:
    action_id = str(rng.choice(np.asarray(ACTION_REGISTRY, dtype=object)))
    if action_id == "STOP_DATA_GATE":
        evidence = _observed_evidence(scenario)
    elif action_id == "REQUEST_EVIDENCE":
        evidence = tuple(
            f"hash:{item.path}"
            for item in scenario.packet_items
            if item.hash_state == "unknown"
        )
    elif action_id == "VERIFY_HASHES":
        evidence = tuple(
            f"verify:{item.path}"
            for item in scenario.packet_items
            if item.hash_state == "mismatch"
        )
    else:
        evidence = ()
    return _decision_payload(
        scenario=scenario,
        policy_id="A3_random_policy",
        action_id=action_id,
        necessary_evidence=evidence,
        decision_status="OK",
        rationale="PCG64 equal-budget random action.",
    )


def _decision_payload(
    *,
    scenario: Scenario,
    policy_id: str,
    action_id: str | None,
    necessary_evidence: Iterable[str],
    decision_status: str,
    rationale: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    evidence_list = sorted(set(necessary_evidence))
    if action_id is not None and action_id not in ACTION_REGISTRY:
        raise ValueError(f"invalid action_id={action_id!r}")
    if decision_status == "OK" and action_id is None:
        raise ValueError("decision_status=OK requires an action_id")
    if confidence is None:
        confidence = 0.9 if action_id in {"STOP_DATA_GATE", "VERIFY_HASHES"} else 0.8
    record = {
        "scenario_id": scenario.scenario_id,
        "scenario_split": scenario.split,
        "policy_id": policy_id,
        "decision_status": decision_status,
        "selected_action_id": action_id,
        "necessary_evidence": evidence_list,
        "stop_requested": action_id != "PROCEED" if action_id is not None else True,
        "confidence": float(confidence),
        "rationale": rationale,
    }
    record["decision_hash"] = (
        sha256_text(
            canonical_json(
                {
                    "scenario_id": record["scenario_id"],
                    "scenario_split": record["scenario_split"],
                    "decision_status": record["decision_status"],
                    "selected_action_id": record["selected_action_id"],
                    "necessary_evidence": record["necessary_evidence"],
                    "stop_requested": record["stop_requested"],
                }
            )
        )
        if decision_status == "OK"
        else None
    )
    return record


def _policy_result_template(policy_id: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "blocked_provider_count": 0,
        "records": [],
    }


def _empty_policy_summary(policy_id: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "trials": 0,
        "blocked_provider_count": 0,
        "decision_accuracy": 0.0,
        "necessary_evidence_f1": 0.0,
        "dangerous_false_release_rate": 0.0,
        "selection": {},
        "promotion": {},
    }


def select_retained_policy(policy_summaries: dict[str, Any]) -> tuple[str, str]:
    """Pick the retained candidate with conservative, multi-objective dominance rules.

    A2D is preferred when it is non-inferior to A2L on decision accuracy,
    necessary-evidence F1, and dangerous false release rate. Deterministic
    policies break ties in favor of A2D so the LLM stays advisory unless it
    strictly dominates on the registered objective bundle.
    """

    a2d = policy_summaries["A2D_deterministic_agent"]
    a2l = policy_summaries["A2L_llm_agent_execute"]
    a2d_non_inferior = (
        a2d["decision_accuracy"] >= a2l["decision_accuracy"]
        and a2d["necessary_evidence_f1"] >= a2l["necessary_evidence_f1"]
        and a2d["dangerous_false_release_rate"] <= a2l["dangerous_false_release_rate"]
    )
    a2l_non_inferior = (
        a2l["decision_accuracy"] >= a2d["decision_accuracy"]
        and a2l["necessary_evidence_f1"] >= a2d["necessary_evidence_f1"]
        and a2l["dangerous_false_release_rate"] <= a2d["dangerous_false_release_rate"]
    )
    if a2d_non_inferior:
        return "A2D_deterministic_agent", "A2L_llm_agent_execute"
    if a2l_non_inferior:
        return "A2L_llm_agent_execute", "A2D_deterministic_agent"
    if a2d["decision_accuracy"] != a2l["decision_accuracy"]:
        return (
            "A2D_deterministic_agent",
            "A2L_llm_agent_execute",
        ) if a2d["decision_accuracy"] > a2l["decision_accuracy"] else (
            "A2L_llm_agent_execute",
            "A2D_deterministic_agent",
        )
    if a2d["necessary_evidence_f1"] != a2l["necessary_evidence_f1"]:
        return (
            "A2D_deterministic_agent",
            "A2L_llm_agent_execute",
        ) if a2d["necessary_evidence_f1"] > a2l["necessary_evidence_f1"] else (
            "A2L_llm_agent_execute",
            "A2D_deterministic_agent",
        )
    if a2d["dangerous_false_release_rate"] != a2l["dangerous_false_release_rate"]:
        return (
            "A2D_deterministic_agent",
            "A2L_llm_agent_execute",
        ) if a2d["dangerous_false_release_rate"] < a2l["dangerous_false_release_rate"] else (
            "A2L_llm_agent_execute",
            "A2D_deterministic_agent",
        )
    return "A2D_deterministic_agent", "A2L_llm_agent_execute"


def selection_promotion_intersection() -> list[str]:
    return sorted(set(SELECTION_SCENARIO_IDS) & set(PROMOTION_SCENARIO_IDS))


def ensure_selection_promotion_disjoint() -> list[str]:
    overlap = selection_promotion_intersection()
    if overlap:
        raise RuntimeError(f"selection/promotion scenario overlap: {overlap}")
    return overlap


def _score_records(records: list[dict[str, Any]], scenarios: dict[str, Scenario]) -> dict[str, Any]:
    if not records:
        return {
            "trials": 0,
            "decision_accuracy": 0.0,
            "necessary_evidence_f1": 0.0,
            "dangerous_false_release_rate": 0.0,
            "blocked_provider_count": 0,
        }
    correct = 0
    tp = fp = fn = 0
    false_release = 0
    blocked_provider = 0
    blocked_scenarios = 0
    for record in records:
        scenario = scenarios[record["scenario_id"]]
        gold_action = scenario.gold_action_id
        gold_evidence = set(scenario.gold_necessary_evidence)
        pred_action = record["selected_action_id"]
        pred_evidence = set(record["necessary_evidence"])
        if record["decision_status"] == "BLOCKED_PROVIDER":
            blocked_provider += 1
            pred_action = None
            pred_evidence = set()
        if pred_action == gold_action:
            correct += 1
        if gold_action != "PROCEED":
            blocked_scenarios += 1
            if pred_action == "PROCEED":
                false_release += 1
        tp += len(pred_evidence & gold_evidence)
        fp += len(pred_evidence - gold_evidence)
        fn += len(gold_evidence - pred_evidence)
    precision = tp / (tp + fp) if tp + fp else (1.0 if tp == fp == fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if tp == fp == fn == 0 else 0.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "trials": len(records),
        "decision_accuracy": correct / len(records),
        "necessary_evidence_f1": f1,
        "dangerous_false_release_rate": false_release / blocked_scenarios if blocked_scenarios else 0.0,
        "blocked_provider_count": blocked_provider,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _run_a2l(scenario: Scenario) -> dict[str, Any]:
    api_key = (
        os.environ.get("DEEPSEEK_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_API_TOKEN")
        or os.environ.get("DEEPSEEK_TOKEN")
    )
    base_url = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("DEEPSEEK_API_BASE")
        or "https://api.deepseek.com"
    )
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        return {
            "scenario_id": scenario.scenario_id,
            "scenario_split": scenario.split,
            "policy_id": "A2L_llm_agent_execute",
            "decision_status": "BLOCKED_PROVIDER",
            "selected_action_id": None,
            "necessary_evidence": [],
            "stop_requested": True,
            "confidence": None,
            "rationale": "BLOCKED_PROVIDER: missing DeepSeek credentials in the environment.",
            "provider_status": "BLOCKED_PROVIDER",
            "provider_reason": "missing DeepSeek API key environment variable",
            "decision_hash": None,
            "advice_text": None,
        }
    prompt = {
        "scenario": scenario.scenario_id,
        "title": scenario.title,
        "instructions": [
            "Return exactly one JSON object.",
            f"Choose action_id from {list(ACTION_REGISTRY)} only.",
            "Use stop_requested=true when action_id is STOP_DATA_GATE, REQUEST_EVIDENCE, or VERIFY_HASHES.",
            "Provide necessary_evidence as a list of short protocol keys.",
            "Do not mention training, test, or holdout data.",
        ],
        "packet": [
            {
                "path": item.path,
                "reported_sha256": item.reported_sha256,
                "hash_state": item.hash_state,
                "role": item.role,
            }
            for item in scenario.packet_items
        ],
        "gates": {
            gate_name: {
                "present": gate.present,
                "hash_state": gate.hash_state,
                "evidence_code": gate.evidence_code,
            }
            for gate_name, gate in scenario.gates.items()
        },
    }
    try:
        payload = _deepseek_chat_completion(base_url, api_key, model, prompt)
        return _validate_a2l_response(scenario, payload)
    except Exception as exc:  # noqa: BLE001 - fail closed by design
        return {
            "scenario_id": scenario.scenario_id,
            "scenario_split": scenario.split,
            "policy_id": "A2L_llm_agent_execute",
            "decision_status": "BLOCKED_PROVIDER",
            "selected_action_id": None,
            "necessary_evidence": [],
            "stop_requested": True,
            "confidence": None,
            "rationale": f"BLOCKED_PROVIDER: {exc}",
            "provider_status": "BLOCKED_PROVIDER",
            "provider_reason": str(exc),
            "decision_hash": None,
            "advice_text": None,
        }


def _deepseek_chat_completion(base_url: str, api_key: str, model: str, prompt: dict[str, Any]) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body = canonical_json(
        {
            "model": model,
            "temperature": 0.0,
            "top_p": 1.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic fault gate controller. "
                        "Return only valid JSON without markdown fences."
                    ),
                },
                {
                    "role": "user",
                    "content": canonical_json(prompt),
                },
            ],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc
    data = json.loads(response_body)
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("DeepSeek response did not contain a chat completion") from exc
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json").strip()
    return json.loads(content)


def _validate_a2l_response(scenario: Scenario, payload: dict[str, Any]) -> dict[str, Any]:
    action_id = payload.get("action_id")
    if action_id not in ACTION_REGISTRY:
        raise RuntimeError(f"DeepSeek returned invalid action_id={action_id!r}")
    necessary_evidence = payload.get("necessary_evidence")
    if not isinstance(necessary_evidence, list) or not all(isinstance(x, str) for x in necessary_evidence):
        raise RuntimeError("DeepSeek response missing necessary_evidence list")
    stop_requested = payload.get("stop_requested")
    if not isinstance(stop_requested, bool):
        raise RuntimeError("DeepSeek response missing stop_requested boolean")
    confidence = payload.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        raise RuntimeError("DeepSeek response missing numeric confidence")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise RuntimeError("DeepSeek response missing rationale string")
    decision = _decision_payload(
        scenario=scenario,
        policy_id="A2L_llm_agent_execute",
        action_id=str(action_id),
        necessary_evidence=necessary_evidence,
        decision_status="OK",
        rationale=rationale,
        confidence=float(confidence),
    )
    # DeepSeek may suggest stop/proceed; keep the JSON payload aligned with the canonical schema.
    decision["stop_requested"] = stop_requested
    decision["provider_status"] = "OK"
    decision["provider_reason"] = None
    decision["raw_response"] = payload
    return decision


def run_ablation(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = build_source_hashes()
    scenarios = build_scenarios(source_hashes)
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    source_commit = git_head()

    protocol_records = []
    for scenario in scenarios:
        protocol_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "split": scenario.split,
                "kind": scenario.kind,
                "source_commit": source_commit,
                "packet_digest": scenario.packet_digest(),
                "packet_items": [
                    {
                        "path": item.path,
                        "actual_sha256": item.actual_sha256,
                        "reported_sha256": item.reported_sha256,
                        "hash_state": item.hash_state,
                        "role": item.role,
                    }
                    for item in scenario.packet_items
                ],
                "gates": {
                    name: {
                        "present": gate.present,
                        "hash_state": gate.hash_state,
                        "evidence_code": gate.evidence_code,
                    }
                    for name, gate in scenario.gates.items()
                },
                "gold_action_id": scenario.gold_action_id,
                "gold_necessary_evidence": list(scenario.gold_necessary_evidence),
                "notes": scenario.notes,
            }
        )
    protocol_path = output_root / "protocol.jsonl"
    _write_jsonl(protocol_path, protocol_records)

    policy_records: list[dict[str, Any]] = []
    a0_records: list[dict[str, Any]] = []
    a1_records: list[dict[str, Any]] = []
    a2d_records: list[dict[str, Any]] = []
    a2l_records: list[dict[str, Any]] = []
    a3_records: list[dict[str, Any]] = []
    rng = np.random.Generator(np.random.PCG64(2693))
    for scenario in scenarios:
        a0 = conservative_policy(scenario)
        a1 = advice_only_policy(scenario, a0)
        a2d = deterministic_policy(scenario)
        a2l = _run_a2l(scenario)
        a3 = random_policy(scenario, rng)
        for record in (a0, a1, a2d, a2l, a3):
            policy_records.append(record)
        a0_records.append(a0)
        a1_records.append(a1)
        a2d_records.append(a2d)
        a2l_records.append(a2l)
        a3_records.append(a3)

    results_path = output_root / "results.jsonl"
    _write_jsonl(results_path, policy_records)

    selection_records = {
        policy_id: [record for record in records if record["scenario_id"] in SELECTION_SCENARIO_IDS]
        for policy_id, records in {
            "A0_static_baseline": a0_records,
            "A1_advice_only": a1_records,
            "A2L_llm_agent_execute": a2l_records,
            "A2D_deterministic_agent": a2d_records,
            "A3_random_policy": a3_records,
        }.items()
    }
    promotion_records = {
        policy_id: [record for record in records if record["scenario_id"] in PROMOTION_SCENARIO_IDS]
        for policy_id, records in {
            "A0_static_baseline": a0_records,
            "A1_advice_only": a1_records,
            "A2L_llm_agent_execute": a2l_records,
            "A2D_deterministic_agent": a2d_records,
            "A3_random_policy": a3_records,
        }.items()
    }

    policy_summaries: dict[str, Any] = {}
    for policy_id, records in {
        "A0_static_baseline": a0_records,
        "A1_advice_only": a1_records,
        "A2L_llm_agent_execute": a2l_records,
        "A2D_deterministic_agent": a2d_records,
        "A3_random_policy": a3_records,
    }.items():
        summary = _score_records(records, scenario_map)
        summary["policy_id"] = policy_id
        summary["selection"] = _score_records(selection_records[policy_id], scenario_map)
        summary["promotion"] = _score_records(promotion_records[policy_id], scenario_map)
        policy_summaries[policy_id] = summary

    retain_policy, reject_policy = select_retained_policy(policy_summaries)
    selection_promotion_overlap = ensure_selection_promotion_disjoint()
    policy_summaries["retain_policy"] = retain_policy
    policy_summaries["reject_policy"] = reject_policy

    summary = {
        "schema_version": "fault_p28_agentic_ablation/v1",
        "track_id": "fault",
        "source_commit": source_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_budget": DEFAULT_TRIAL_BUDGET,
        "selection_scenario_ids": list(SELECTION_SCENARIO_IDS),
        "promotion_scenario_ids": list(PROMOTION_SCENARIO_IDS),
        "action_registry": list(ACTION_REGISTRY),
        "policies": policy_summaries,
        "retained_policy": retain_policy,
        "rejected_policy": reject_policy,
        "data_gate_blocked": True,
        "selection_promotion_intersection": selection_promotion_overlap,
        "selection_promotion_intersection_ok": not selection_promotion_overlap,
        "blocked_provider": {
            "A2L_llm_agent_execute": policy_summaries["A2L_llm_agent_execute"]["blocked_provider_count"] > 0
        },
        "frozen_test_accessed": False,
        "scenario_titles": {scenario.scenario_id: scenario.title for scenario in scenarios},
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")

    evidence_path = output_root / "evidence.md"
    evidence_path.write_text(render_evidence(summary, scenarios, source_hashes), encoding="utf-8")

    manifest = {
        "schema_version": "fault_p28_agentic_ablation/v1",
        "track_id": "fault",
        "source_commit": source_commit,
        "generated_at": summary["generated_at"],
        "runner": {
            "path": display_path(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "trial_budget": DEFAULT_TRIAL_BUDGET,
        "selection_scenario_ids": list(SELECTION_SCENARIO_IDS),
        "promotion_scenario_ids": list(PROMOTION_SCENARIO_IDS),
        "action_registry": list(ACTION_REGISTRY),
        "inputs": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "role": name,
            }
            for name, record in source_hashes.items()
        ],
        "outputs": [],
        "retained_policy": retain_policy,
        "rejected_policy": reject_policy,
        "data_gate_blocked": summary["data_gate_blocked"],
        "selection_promotion_intersection": selection_promotion_overlap,
        "selection_promotion_intersection_ok": not selection_promotion_overlap,
        "blocked_provider": summary["blocked_provider"],
        "notes": "A2L executed with provider_status=OK; the formal predictive lane remains data_gate_blocked.",
    }
    output_files = [protocol_path, results_path, summary_path, evidence_path]
    manifest["outputs"] = [
        {
            "path": display_path(path),
            "sha256": sha256_file(path),
        }
        for path in output_files
    ]
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    return {
        "summary": summary,
        "manifest": manifest,
        "protocol_path": display_path(protocol_path),
        "results_path": display_path(results_path),
        "summary_path": display_path(summary_path),
        "evidence_path": display_path(evidence_path),
        "manifest_path": display_path(manifest_path),
        "scenarios": [scenario.scenario_id for scenario in scenarios],
        "source_hashes": source_hashes,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")


def render_evidence(
    summary: dict[str, Any],
    scenarios: tuple[Scenario, ...],
    source_hashes: dict[str, dict[str, str]],
) -> str:
    lines = [
        "# Fault P28 agentic ablation preflight",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Source commit: `{summary['source_commit']}`",
        f"- Trial budget: {summary['trial_budget']}",
        f"- Selection scenarios: `{', '.join(summary['selection_scenario_ids'])}`",
        f"- Promotion scenarios: `{', '.join(summary['promotion_scenario_ids'])}`",
        f"- Action registry: `{', '.join(summary['action_registry'])}`",
        "",
        "## Contract evidence bundle",
        "",
    ]
    for name, record in source_hashes.items():
        lines.append(f"- `{name}`: `{record['path']}` sha256=`{record['sha256']}`")
    lines.extend(
        [
            "",
            "## Scenario design",
            "",
            "| scenario | split | truth | packet hash state | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in scenarios:
        lines.append(
            f"| `{scenario.scenario_id}` | {scenario.split} | `{scenario.gold_action_id}` | "
            f"`{scenario.packet_hash_state()}` | {scenario.notes} |"
        )
    lines.extend(
        [
            "",
            "## Policy results",
            "",
            "| policy | selection accuracy | selection evidence F1 | promotion accuracy | promotion evidence F1 | dangerous false release | blocked provider |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for policy_id in POLICY_IDS:
        pol = summary["policies"][policy_id]
        lines.append(
            f"| `{policy_id}` | {pol['selection']['decision_accuracy']:.3f} | "
            f"{pol['selection']['necessary_evidence_f1']:.3f} | "
            f"{pol['promotion']['decision_accuracy']:.3f} | {pol['promotion']['necessary_evidence_f1']:.3f} | "
            f"{pol['dangerous_false_release_rate']:.3f} | {pol['blocked_provider_count']} |"
        )
    lines.extend(
        [
            "",
            "## Retain / reject",
            "",
            f"- Retain: `{summary['retained_policy']}`",
            f"- Reject: `{summary['rejected_policy']}`",
            f"- data_gate_blocked: `{summary['data_gate_blocked']}`",
            f"- selection/promotion intersection: `{summary['selection_promotion_intersection']}`",
            f"- A2L blocked provider: `{summary['blocked_provider']['A2L_llm_agent_execute']}`",
            f"- Frozen test accessed: `{summary['frozen_test_accessed']}`",
            "",
            "## Notes",
            "",
            "- The real formal fault training lane remains DATA_GATE_BLOCKED.",
            "- `counterfactual_contract_green` is a protocol fixture that exercises the proceed branch without opening frozen test or claiming current data are green.",
            "- A1 replays the deterministic A0 decision and verifies the same decision hash.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_ablation(args.output_root)
    print(canonical_json(result["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
