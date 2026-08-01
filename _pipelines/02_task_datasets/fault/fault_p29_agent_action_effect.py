#!/usr/bin/env python3
"""P29 action-effect repair for the fault gate controller.

This runner treats P28 as a gate controller, not a prediction optimizer.
It keeps the formal fault lane DATA_GATE_BLOCKED and only verifies that
gate decisions, evidence-token validation, canonical decision hashing, and
executor side effects behave exactly as specified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from fault_p28_agentic_ablation import (  # reuse audited P28 contract fixtures
    ACTION_REGISTRY,
    DEFAULT_TRIAL_BUDGET,
    PROJECT_ROOT,
    Scenario,
    build_scenarios,
    canonical_json,
    conservative_policy,
    deterministic_policy,
    display_path,
    git_head,
    random_policy,
    sha256_file,
    sha256_text,
)

TRACK_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p29_agent_action_effect"
PREDICTOR_REGISTRY = ("fault_gate_predictor_v1",)

ROOT_SOURCE_PATHS = {
    "fault_p28_agentic_ablation.py": TRACK_DIR / "fault_p28_agentic_ablation.py",
    "tests/test_fault_p29_agent_action_effect.py": TRACK_DIR / "tests" / "test_fault_p29_agent_action_effect.py",
    "p28_summary.json": TRACK_DIR / "_outputs" / "p28_agentic_ablation" / "summary.json",
    "p28_manifest.json": TRACK_DIR / "_outputs" / "p28_agentic_ablation" / "manifest.json",
    "p18_evidence.md": TRACK_DIR / "_outputs" / "p18_cigbench_fault" / "evidence.md",
    "baseline_metrics.json": TRACK_DIR / "_outputs" / "runs" / "audited_v2" / "baseline_metrics.json",
    "build_summary.json": TRACK_DIR / "_outputs" / "runs" / "audited_v2" / "build_summary.json",
    "historical_baseline_manifest.json": TRACK_DIR / "historical_baseline_manifest.json",
}

EVIDENCE_TOKEN_ALLOWLIST: dict[str, tuple[str, ...]] = {
    "observed_blocked_current": (
        "contiguous_3d_development_blocks_missing",
        "coverage_audited_verified_background_missing",
        "explicit_unknown_mask_provenance_missing",
        "group_isolated_development_split_missing",
    ),
    "packet_hash_missing": (
        "hash:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
        "hash:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/build_summary.json",
    ),
    "packet_hash_mismatch": (
        "verify:_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_metrics.json",
    ),
    "counterfactual_contract_green": (),
}


@dataclass(frozen=True)
class SpyDispatch:
    action_id: str
    dispatched_predictors: tuple[str, ...]

    @property
    def dispatch_count(self) -> int:
        return len(self.dispatched_predictors)


class SpyPredictorExecutor:
    """A minimal spy executor for gate actions."""

    def __init__(self, registered_predictors: tuple[str, ...] = PREDICTOR_REGISTRY) -> None:
        self.registered_predictors = registered_predictors

    def dispatch(self, action_id: str) -> SpyDispatch:
        if action_id == "PROCEED":
            return SpyDispatch(action_id=action_id, dispatched_predictors=(self.registered_predictors[0],))
        return SpyDispatch(action_id=action_id, dispatched_predictors=())


def build_source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)}
        for name, path in ROOT_SOURCE_PATHS.items()
    }


def canonical_decision_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": record["scenario_id"],
        "scenario_split": record["scenario_split"],
        "selected_action_id": record["selected_action_id"],
        "necessary_evidence": list(record["necessary_evidence"]),
        "stop_requested": record["stop_requested"],
    }


def canonical_decision_hash(record: dict[str, Any]) -> str:
    return sha256_text(canonical_json(canonical_decision_payload(record)))


def canonical_decision_hash_from_fields(
    *,
    scenario_id: str,
    scenario_split: str,
    policy_id: str,
    selected_action_id: str | None,
    necessary_evidence: Iterable[str],
    stop_requested: bool,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "scenario_id": scenario_id,
                "scenario_split": scenario_split,
                "selected_action_id": selected_action_id,
                "necessary_evidence": sorted(set(necessary_evidence)),
                "stop_requested": stop_requested,
            }
        )
    )


def validate_action_stop_contract(action_id: str | None, stop_requested: bool) -> None:
    if action_id == "PROCEED" and stop_requested:
        raise RuntimeError("PROCEED must not request stop")
    if action_id in {"STOP_DATA_GATE", "REQUEST_EVIDENCE", "VERIFY_HASHES"} and not stop_requested:
        raise RuntimeError(f"{action_id} must request stop")


def validate_evidence_tokens(scenario: Scenario, evidence_tokens: Iterable[str]) -> None:
    allowed = set(EVIDENCE_TOKEN_ALLOWLIST[scenario.scenario_id])
    invalid = sorted(set(evidence_tokens) - allowed)
    if invalid:
        raise RuntimeError(f"invalid evidence tokens for {scenario.scenario_id}: {invalid}")


def validate_a2l_response(scenario: Scenario, payload: dict[str, Any]) -> dict[str, Any]:
    action_id = payload.get("action_id")
    if action_id not in ACTION_REGISTRY:
        raise RuntimeError(f"DeepSeek returned invalid action_id={action_id!r}")
    necessary_evidence = payload.get("necessary_evidence")
    if not isinstance(necessary_evidence, list) or not all(isinstance(x, str) for x in necessary_evidence):
        raise RuntimeError("DeepSeek response missing necessary_evidence list")
    stop_requested = payload.get("stop_requested")
    if not isinstance(stop_requested, bool):
        raise RuntimeError("DeepSeek response missing stop_requested boolean")
    validate_action_stop_contract(action_id, stop_requested)
    validate_evidence_tokens(scenario, necessary_evidence)
    confidence = payload.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        raise RuntimeError("DeepSeek response missing numeric confidence")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise RuntimeError("DeepSeek response missing rationale string")
    record = {
        "scenario_id": scenario.scenario_id,
        "scenario_split": scenario.split,
        "policy_id": "A2L_llm_agent_execute",
        "decision_status": "OK",
        "selected_action_id": action_id,
        "necessary_evidence": sorted(set(necessary_evidence)),
        "stop_requested": stop_requested,
        "confidence": float(confidence),
        "rationale": rationale,
        "provider_status": "OK",
        "provider_reason": None,
        "advice_text": None,
        "replay_executed": False,
    }
    record["decision_hash"] = canonical_decision_hash(record)
    record["raw_response"] = payload
    return record


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
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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


def _run_a2l(scenario: Scenario) -> dict[str, Any]:
    api_key = (
        os.environ.get("DEEPSEEK_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_API_TOKEN")
        or os.environ.get("DEEPSEEK_TOKEN")
    )
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("DEEPSEEK_API_BASE") or "https://api.deepseek.com"
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
            "replay_executed": False,
        }
    prompt = {
        "scenario": scenario.scenario_id,
        "title": scenario.title,
        "instructions": [
            "Return exactly one JSON object.",
            f"Choose action_id from {list(ACTION_REGISTRY)} only.",
            "Use stop_requested=true when action_id is STOP_DATA_GATE, REQUEST_EVIDENCE, or VERIFY_HASHES.",
            "Provide necessary_evidence as a list of short protocol keys from the evidence-token allowlist only.",
            "Do not mention training, test, or holdout data.",
        ],
        "evidence_token_allowlist": list(EVIDENCE_TOKEN_ALLOWLIST[scenario.scenario_id]),
        "gates": {
            name: {"present": gate.present, "hash_state": gate.hash_state, "evidence_code": gate.evidence_code}
            for name, gate in scenario.gates.items()
        },
    }
    try:
        payload = _deepseek_chat_completion(base_url, api_key, model, prompt)
        return validate_a2l_response(scenario, payload)
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
            "replay_executed": False,
        }


def action_effect(spy: SpyPredictorExecutor, action_id: str | None) -> SpyDispatch:
    if action_id is None:
        return SpyDispatch(action_id="BLOCKED_PROVIDER", dispatched_predictors=())
    return spy.dispatch(action_id)


def execution_record(
    *,
    scenario: Scenario,
    policy_id: str,
    decision: dict[str, Any],
    spy: SpyPredictorExecutor,
) -> dict[str, Any]:
    dispatch = action_effect(spy, decision["selected_action_id"])
    record = dict(decision)
    record["policy_id"] = policy_id
    record["dispatch_count"] = dispatch.dispatch_count
    record["dispatched_predictors"] = list(dispatch.dispatched_predictors)
    record["dispatch_effect_hash"] = sha256_text(
        canonical_json(
            {
                "scenario_id": scenario.scenario_id,
                "policy_id": policy_id,
                "action_id": decision["selected_action_id"],
                "dispatch_count": dispatch.dispatch_count,
                "dispatched_predictors": list(dispatch.dispatched_predictors),
            }
        )
    )
    return record


def build_records(
    scenarios: tuple[Scenario, ...],
    *,
    api_enabled: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy_records: list[dict[str, Any]] = []
    spy = SpyPredictorExecutor()
    rng = np.random.Generator(np.random.PCG64(2693))
    source_commit = git_head()
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    a0_records: list[dict[str, Any]] = []
    a1_records: list[dict[str, Any]] = []
    a2d_records: list[dict[str, Any]] = []
    a2l_records: list[dict[str, Any]] = []
    a3_records: list[dict[str, Any]] = []
    for scenario in scenarios:
        a0 = conservative_policy(scenario)
        a1 = _a1_replay(scenario, a0)
        a2d = deterministic_policy(scenario)
        a2l = _run_a2l(scenario)
        a3 = random_policy(scenario, rng)
        a0_records.append(execution_record(scenario=scenario, policy_id="A0_static_baseline", decision=a0, spy=spy))
        a1_records.append(execution_record(scenario=scenario, policy_id="A1_advice_only", decision=a1, spy=spy))
        a2d_records.append(execution_record(scenario=scenario, policy_id="A2D_deterministic_agent", decision=a2d, spy=spy))
        a2l_records.append(execution_record(scenario=scenario, policy_id="A2L_llm_agent_execute", decision=a2l, spy=spy))
        a3_records.append(execution_record(scenario=scenario, policy_id="A3_random_policy", decision=a3, spy=spy))
        policy_records.extend([a0_records[-1], a1_records[-1], a2d_records[-1], a2l_records[-1], a3_records[-1]])

    policy_summaries = {
        policy_id: _score_records(records, scenario_map)
        for policy_id, records in {
            "A0_static_baseline": a0_records,
            "A1_advice_only": a1_records,
            "A2L_llm_agent_execute": a2l_records,
            "A2D_deterministic_agent": a2d_records,
            "A3_random_policy": a3_records,
        }.items()
    }
    for policy_id, summary in policy_summaries.items():
        summary["policy_id"] = policy_id
    retain_policy, reject_policy = _select_retained_policy(policy_summaries)
    policy_summaries["retain_policy"] = retain_policy
    policy_summaries["reject_policy"] = reject_policy
    return policy_records, {
        "source_commit": source_commit,
        "policy_summaries": policy_summaries,
        "retain_policy": retain_policy,
        "reject_policy": reject_policy,
        "spy_registered_predictors": list(PREDICTOR_REGISTRY),
        "selection_promotion_intersection": [],
        "data_gate_blocked": True,
        "api_enabled": api_enabled,
    }


def _a1_replay(scenario: Scenario, baseline_decision: dict[str, Any]) -> dict[str, Any]:
    replay = conservative_policy(scenario)
    if replay["decision_hash"] != baseline_decision["decision_hash"]:
        raise RuntimeError("A1 replay failed to reproduce the A0 decision hash")
    payload = dict(replay)
    payload["policy_id"] = "A1_advice_only"
    payload["replay_executed"] = True
    payload["replay_source_policy_id"] = "A0_static_baseline"
    payload["replay_decision_hash"] = replay["decision_hash"]
    payload["replay_match"] = True
    payload["advice_text"] = "Advice-only mirror of A0; final decision is intentionally kept identical to A0."
    payload["advice_hash"] = sha256_text(
        canonical_json(
            {"scenario_id": scenario.scenario_id, "policy_id": "A1_advice_only", "advice_text": payload["advice_text"]}
        )
    )
    return payload


def _select_retained_policy(policy_summaries: dict[str, Any]) -> tuple[str, str]:
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
    return ("A2D_deterministic_agent", "A2L_llm_agent_execute")


def _score_records(records: list[dict[str, Any]], scenarios: dict[str, Scenario]) -> dict[str, Any]:
    if not records:
        return {
            "trials": 0,
            "decision_accuracy": 0.0,
            "necessary_evidence_f1": 0.0,
            "dangerous_false_release_rate": 0.0,
            "blocked_provider_count": 0,
            "dispatch_count": 0,
        }
    correct = 0
    tp = fp = fn = 0
    false_release = 0
    blocked_provider = 0
    dispatch_count = 0
    blocked_scenarios = 0
    for record in records:
        scenario = scenarios[record["scenario_id"]]
        gold_action = scenario.gold_action_id
        gold_evidence = set(scenario.gold_necessary_evidence)
        pred_action = record["selected_action_id"]
        pred_evidence = set(record["necessary_evidence"])
        dispatch_count += int(record["dispatch_count"])
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
        "dispatch_count": dispatch_count,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _render_root_cause(summary: dict[str, Any], source_hashes: dict[str, dict[str, str]]) -> str:
    lines = [
        "# Fault P29 action-effect repair",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Source commit: `{summary['source_commit']}`",
        f"- data_gate_blocked: `{summary['data_gate_blocked']}`",
        f"- selection/promotion intersection: `{summary['selection_promotion_intersection']}`",
        "",
        "## Root cause",
        "",
        "P28 proved the gate controller could stop or proceed, but it did not prove that PROCEED "
        "actually invoked a registered predictor exactly once or that all other gate actions were no-ops.",
        "",
        "## Repairs",
        "",
        "- Evidence tokens are now validated against a per-scenario allowlist.",
        "- stop_requested is validated against the selected action.",
        "- decision hashes use a canonical decision payload and are replay-checked for A1.",
        "- PROCEED dispatches exactly one registered predictor; all other gate actions dispatch none.",
        "",
        "## Sources",
        "",
    ]
    for name, record in source_hashes.items():
        lines.append(f"- `{name}`: `{record['path']}` sha256=`{record['sha256']}`")
    return "\n".join(lines) + "\n"


def run_p29(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_hashes = build_source_hashes()
    scenarios = build_scenarios(source_hashes)
    protocol_records = []
    source_commit = git_head()
    for scenario in scenarios:
        protocol_records.append(
            {
                "scenario_id": scenario.scenario_id,
                "title": scenario.title,
                "split": scenario.split,
                "kind": scenario.kind,
                "source_commit": source_commit,
                "evidence_token_allowlist": list(EVIDENCE_TOKEN_ALLOWLIST[scenario.scenario_id]),
                "gold_action_id": scenario.gold_action_id,
                "gold_necessary_evidence": list(scenario.gold_necessary_evidence),
                "notes": scenario.notes,
            }
        )
    protocol_path = output_root / "protocol.jsonl"
    with protocol_path.open("w", encoding="utf-8") as handle:
        for row in protocol_records:
            handle.write(canonical_json(row) + "\n")
    policy_records, audit = build_records(scenarios, api_enabled=bool(
        os.environ.get("DEEPSEEK_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("DEEPSEEK_API_TOKEN")
        or os.environ.get("DEEPSEEK_TOKEN")
    ))
    results_path = output_root / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in policy_records:
            handle.write(canonical_json(row) + "\n")
    summary = {
        "schema_version": "fault_p29_agent_action_effect/v1",
        "track_id": "fault",
        "source_commit": source_commit,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trial_budget": DEFAULT_TRIAL_BUDGET,
        "action_registry": list(ACTION_REGISTRY),
        "predictor_registry": list(PREDICTOR_REGISTRY),
        "data_gate_blocked": True,
        "selection_promotion_intersection": audit["selection_promotion_intersection"],
        "selection_promotion_intersection_ok": True,
        "blocked_provider": {
            "A2L_llm_agent_execute": audit["policy_summaries"]["A2L_llm_agent_execute"]["blocked_provider_count"] > 0
        },
        "frozen_test_accessed": False,
        "retained_policy": audit["retain_policy"],
        "rejected_policy": audit["reject_policy"],
        "policies": audit["policy_summaries"],
        "scenario_titles": {scenario.scenario_id: scenario.title for scenario in scenarios},
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    action_effects = {
        "schema_version": "fault_p29_action_effect/v1",
        "track_id": "fault",
        "source_commit": source_commit,
        "generated_at": summary["generated_at"],
        "data_gate_blocked": True,
        "selection_promotion_intersection": audit["selection_promotion_intersection"],
        "retained_policy": audit["retain_policy"],
        "rejected_policy": audit["reject_policy"],
        "registered_predictors": list(PREDICTOR_REGISTRY),
        "records": policy_records,
    }
    action_effects_path = output_root / "action_effects.json"
    action_effects_path.write_text(canonical_json(action_effects) + "\n", encoding="utf-8")
    root_cause_path = output_root / "root_cause.md"
    root_cause_path.write_text(_render_root_cause(summary, source_hashes), encoding="utf-8")
    manifest = {
        "schema_version": "fault_p29_agent_action_effect/v1",
        "track_id": "fault",
        "source_commit": source_commit,
        "generated_at": summary["generated_at"],
        "runner": {"path": display_path(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "inputs": [
            {"path": record["path"], "sha256": record["sha256"], "role": name}
            for name, record in source_hashes.items()
        ],
        "outputs": [
            {"path": display_path(path), "sha256": sha256_file(path)}
            for path in (protocol_path, results_path, summary_path, action_effects_path, root_cause_path)
        ],
        "data_gate_blocked": True,
        "selection_promotion_intersection": audit["selection_promotion_intersection"],
        "retained_policy": audit["retain_policy"],
        "rejected_policy": audit["reject_policy"],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    return {
        "protocol_path": display_path(protocol_path),
        "results_path": display_path(results_path),
        "summary_path": display_path(summary_path),
        "action_effects_path": display_path(action_effects_path),
        "root_cause_path": display_path(root_cause_path),
        "manifest_path": display_path(manifest_path),
        "summary": summary,
        "action_effects": action_effects,
        "manifest": manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_p29(args.output_root)
    print(canonical_json({"summary": result["summary"], "action_effects_path": result["action_effects_path"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
