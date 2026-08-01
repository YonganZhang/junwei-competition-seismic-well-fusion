"""Sweetspot P28 T3-only hybrid execution-agent pilot.

This implementation is fail-closed by design. It can archive a real A2L
DeepSeek decision and deterministic/random policy choices, but if no portable
executor is present in the private P28 code, the outcome is
``BLOCKED_EXECUTOR`` and no online evidence is replayed as new science.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_SEED = 2693
TRACK_ID = "sweetspot"
P28_ID = "p28_agentic_optimization"
SCHEMA_VERSION = "sweetspot-p28-agentic-optimization/v1"
ACTION_IDS = (
    "T3_ROUTE_XGB64_BASELINE",
    "T3_ROUTE_CHRONOS2_FIXED",
    "T3_BLEND_WEIGHT_0_63",
    "T3_BLEND_WEIGHT_0_65",
    "T3_XGB_DEPTH_4_ETA_0_05",
    "T3_XGB_DEPTH_6_ETA_0_05",
)
XGB_ACTION_IDS = ("T3_XGB_DEPTH_4_ETA_0_05", "T3_XGB_DEPTH_6_ETA_0_05")
CHRONOS_ACTION_IDS = ("T3_BLEND_WEIGHT_0_63", "T3_BLEND_WEIGHT_0_65")
SELECTION_FOLDS = [0, 1, 2]
PROMOTION_FOLDS = [3]

HERE = Path(__file__).resolve().parent
TRACK_DIR = HERE.parent
WORKTREE_ROOT = TRACK_DIR.parents[2]
WORKTREES_DIR = WORKTREE_ROOT.parent
PROJECT_ROOT = WORKTREE_ROOT.parent.parent.parent
REFERENCE_ROOT = WORKTREES_DIR / "p10-results-sweetspot"
REFERENCE_SWEETSPOT = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot"
OUTPUT_DIR = TRACK_DIR / "_outputs" / P28_ID

STAGE3_T3_PATH = REFERENCE_SWEETSPOT / "p5/_outputs/stage3_cv/leaderboards/T3.json"
STAGE3_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p5/_outputs/stage3_cv/p5_stage3_summary.json"
STAGE4_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p5/_outputs/stage4_confirmation/p5_stage4_summary.json"
P7_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p7/_outputs/t3_chronos2_cv/summary.json"
P8_SUMMARY_PATH = REFERENCE_SWEETSPOT / "p8/_outputs/t3_chronos2_calendar_cv/summary.json"
P17_EVIDENCE_PATH = PROJECT_ROOT / "_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _find_ported_executor() -> Path | None:
    candidate = TRACK_DIR / "p28" / "executor.py"
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class DeepSeekDecision:
    selected_route: str
    selected_action_id: str
    stop_requested: bool
    confidence: float
    rationale: str
    raw: dict[str, Any]


def _reference_inputs() -> list[dict[str, Any]]:
    def display_path(path: Path) -> str:
        if path.is_relative_to(REFERENCE_ROOT):
            return str(path.relative_to(REFERENCE_ROOT))
        if path.is_relative_to(PROJECT_ROOT):
            return str(path.relative_to(PROJECT_ROOT))
        return path.name

    items = [
        ("p5 stage-3 T3 leaderboard", STAGE3_T3_PATH),
        ("p5 stage-3 summary", STAGE3_SUMMARY_PATH),
        ("p5 stage-4 confirmation summary", STAGE4_SUMMARY_PATH),
        ("p7 Chronos summary", P7_SUMMARY_PATH),
        ("p8 calendar Chronos summary", P8_SUMMARY_PATH),
        ("P17 acceptance evidence", P17_EVIDENCE_PATH),
    ]
    return [
        {
            "role": role,
            "path": display_path(path),
            "sha256": _sha256_file(path),
            "shape_or_row_count": path.stat().st_size,
            "scientific_role": role,
            "split_scope": "immutable reference",
        }
        for role, path in items
    ]


def _observation() -> dict[str, Any]:
    return {
        "track_id": TRACK_ID,
        "task_id": "T3",
        "root_seed": ROOT_SEED,
        "a0_baseline": {
            "route": "xgboost",
            "fold_train_state": "flat",
            "selection_dev_feedback": "flat",
            "promotion_dev_feedback": "flat",
            "selection_fold_ids": SELECTION_FOLDS,
            "promotion_fold_ids": PROMOTION_FOLDS,
        },
        "chronos_stratum": {
            "route": "chronos",
            "fold_train_state": "improved",
            "selection_dev_feedback": "improved",
            "promotion_dev_feedback": "improved",
            "selection_fold_ids": SELECTION_FOLDS,
            "promotion_fold_ids": PROMOTION_FOLDS,
        },
        "gate_states": {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"},
        "allowlist": {
            "xgboost": list(XGB_ACTION_IDS),
            "chronos": list(CHRONOS_ACTION_IDS),
        },
        "budget": {"trials": 4, "root_seed": ROOT_SEED},
        "selection_dev_feedback_contract": "LLM sees only improved|flat|worse summaries, never raw metrics, labels, residuals, or paths",
        "promotion_dev_contract": "selection-dev and promotion-dev are nested disjoint folds",
        "foundation_route_fixed_or_separate_factor": True,
    }


def _build_prompt(observation: dict[str, Any]) -> list[dict[str, str]]:
    user = {
        "task": "Choose the T3 pilot route stratum and stop/continue flag.",
        "constraints": {
            "no_raw_metrics": True,
            "no_labels": True,
            "no_residuals": True,
            "no_paths": True,
            "no_holdout_or_test": True,
            "selection_dev_feedback_only": True,
            "folds_disjoint": True,
            "t5_status": observation["gate_states"]["T5"],
            "t6_status": observation["gate_states"]["T6"],
            "t7_status": observation["gate_states"]["T7"],
        },
        "observation": {
            "a0_baseline": observation["a0_baseline"],
            "chronos_stratum": observation["chronos_stratum"],
            "allowlist": observation["allowlist"],
            "budget": observation["budget"],
        },
        "response_schema": {
            "selected_route": "xgboost|chronos|stop",
            "selected_action_id": "one of the allowlisted action ids or A0_static_baseline",
            "stop_requested": "boolean",
            "confidence": "number from 0 to 1",
            "rationale": "short practical rationale",
        },
        "selection_dev_feedback_vocab": ["improved", "flat", "worse"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict decision policy for the sweetspot P28 T3-only pilot. "
                "Return JSON only. Do not invent metrics, labels, paths, or holdout access. "
                "Choose between the XGBoost stratum and the Chronos stratum, or stop."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
        },
    ]


def _call_deepseek(messages: list[dict[str, str]]) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_KEY")
    if not key:
        return {"status": "UNAVAILABLE", "reason": "DEEPSEEK_KEY missing"}
    body = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {"raw_content": content}
    parsed["_provider"] = "deepseek"
    parsed["_model"] = payload.get("model")
    return parsed


def _deterministic_choice() -> dict[str, Any]:
    return {
        "selected_route": "chronos",
        "selected_action_id": CHRONOS_ACTION_IDS[0],
        "stop_requested": False,
        "confidence": 0.71,
        "rationale": "Chronos is the only route with existing development evidence of gain; keep route fixed and stop if no ported executor exists.",
    }


def _random_choice() -> dict[str, Any]:
    import random

    rng = random.Random(ROOT_SEED)
    route = rng.choice(["xgboost", "chronos"])
    if route == "xgboost":
        action = rng.choice(list(XGB_ACTION_IDS))
    else:
        action = rng.choice(list(CHRONOS_ACTION_IDS))
    return {
        "selected_route": route,
        "selected_action_id": action,
        "stop_requested": False,
        "confidence": 0.5,
        "rationale": "Random baseline with the same allowlist and seed roster.",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(encoded, encoding="utf-8")
    return _sha256_bytes(encoded.encode("utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    encoded = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(encoded, encoding="utf-8")
    return _sha256_bytes(encoded.encode("utf-8"))


def _write_text(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return _sha256_bytes(text.encode("utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(TRACK_DIR))
    except ValueError:
        return str(path)


def generate_report(output_dir: Path = OUTPUT_DIR, *, call_deepseek: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    protocol_jsonl_path = output_dir / "protocol.jsonl"
    summary_path = output_dir / "summary.json"
    evidence_path = output_dir / "evidence.md"
    manifest_path = output_dir / "manifest.json"
    refs = _reference_inputs()
    observation = _observation()
    prompt_messages = _build_prompt(observation)
    deepseek = _call_deepseek(prompt_messages) if call_deepseek else {"status": "SKIPPED"}
    a2d = _deterministic_choice()
    a3 = _random_choice()
    executor = _find_ported_executor()
    blocked_reason = (
        "BLOCKED_EXECUTOR: no portable P28 executor exists in the current sweetspot worktree; "
        "immutable sibling results are only references and cannot be replayed as new online evidence."
    )
    verdict = "BLOCKED_EXECUTOR" if executor is None else "READY"
    a0_reference = {
        "route": "xgboost",
        "baseline_commit": "16bebd18a0bc722afcbc4b841610bf76ce9503e4",
        "split_hash": "c44277ffc1f6fb6b5dd740952e921c732d45193c7cb3b5c3dfc061e79025c62a",
        "stage3_leaderboard_sha256": _sha256_file(STAGE3_T3_PATH),
        "stage3_summary_sha256": _sha256_file(STAGE3_SUMMARY_PATH),
        "stage4_summary_sha256": _sha256_file(STAGE4_SUMMARY_PATH),
        "selection_dev_feedback": "flat",
        "promotion_dev_feedback": "flat",
    }
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "task_id": "T3",
        "root_seed": ROOT_SEED,
        "frozen_baseline": a0_reference,
        "observation_schema": {
            "visible_to_llm": [
                "fold_train_aggregates",
                "selection_dev_feedback_improved_flat_worse",
                "promotion_dev_feedback_improved_flat_worse",
                "allowlist",
                "stop_requested",
            ],
            "hidden_from_llm": [
                "raw_metric_values",
                "validation_curves",
                "per_sample_residuals",
                "labels",
                "paths",
            ],
            "selection_fold_ids": SELECTION_FOLDS,
            "promotion_fold_ids": PROMOTION_FOLDS,
            "foundation_route_fixed_or_separate_factor": True,
        },
        "action_allowlist": {
            "xgboost": list(XGB_ACTION_IDS),
            "chronos": list(CHRONOS_ACTION_IDS),
        },
        "leakage_firewall": {
            "no_holdout_or_test": True,
            "no_raw_metrics": True,
            "no_labels": True,
            "nested_disjoint_selection_and_promotion": True,
            "fold_train_only_fit": True,
            "foundation_route_fixed_or_factorized": True,
        },
        "promotion_gate": {
            "stage1_min_trials": 2,
            "stage1_max_trials": 4,
            "require_a2l_helpful_vs_a3": True,
            "require_non_overlapping_selection_and_promotion": True,
            "require_no_leakage": True,
        },
        "gates": {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"},
        "verdict": verdict,
        "blocked_reason": blocked_reason,
        "inputs": refs,
    }
    a2l = {
        "status": "UNAVAILABLE",
        "selected_route": None,
        "selected_action_id": None,
        "stop_requested": True,
        "confidence": None,
        "rationale": None,
        "raw": deepseek,
    }
    if deepseek.get("status") not in {"UNAVAILABLE", "SKIPPED"}:
        a2l.update(
            {
                "status": "OK",
                "selected_route": deepseek.get("selected_route"),
                "selected_action_id": deepseek.get("selected_action_id"),
                "stop_requested": bool(deepseek.get("stop_requested", False)),
                "confidence": deepseek.get("confidence"),
                "rationale": deepseek.get("rationale"),
            }
        )
    rows = [
        {
            "strategy": "A0_static_baseline",
            "status": "RETAINED_REFERENCE",
            "selected_action_id": "T3_ROUTE_XGB64_BASELINE",
            "prediction_hash": _sha256_file(STAGE4_SUMMARY_PATH),
            "note": "Archived baseline reference only; not executed in current worktree.",
        },
        {
            "strategy": "A1_advice_only",
            "status": "NOT_EXECUTED",
            "selected_action_id": "A1_advice_only",
            "prediction_hash": "same_as_A0_by_contract_not_replayed",
            "note": "Prediction equality is contractual only; no new run was replayed.",
        },
        {
            "strategy": "A2L_llm_agent_execute",
            "status": verdict,
            "selected_route": a2l["selected_route"],
            "selected_action_id": a2l["selected_action_id"],
            "decision_source": "deepseek" if a2l["status"] == "OK" else a2l["raw"].get("status", "UNAVAILABLE"),
            "note": "LLM decision archived; executor unavailable in private code, so no new online evidence.",
        },
        {
            "strategy": "A2D_deterministic_agent",
            "status": verdict,
            "selected_route": a2d["selected_route"],
            "selected_action_id": a2d["selected_action_id"],
            "note": "Deterministic route choice recorded; blocked before execution.",
        },
        {
            "strategy": "A3_random_policy",
            "status": verdict,
            "selected_route": a3["selected_route"],
            "selected_action_id": a3["selected_action_id"],
            "note": "Random policy recorded; blocked before execution.",
        },
        {
            "strategy": "A4_deterministic_search",
            "status": verdict,
            "selected_action_id": None,
            "note": "No portable executor; deterministic search not replayed as new evidence.",
        },
        {
            "strategy": "FINAL",
            "status": verdict,
            "selected_route": a2l["selected_route"] or a2d["selected_route"] or a3["selected_route"] or "stop",
            "selected_action_id": a2l["selected_action_id"] or a2d["selected_action_id"] or a3["selected_action_id"],
            "note": blocked_reason,
        },
    ]
    protocol_sha = _write_json(protocol_path, protocol)
    jsonl_sha = _write_jsonl(protocol_jsonl_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "task_id": "T3",
        "verdict": verdict,
        "retention": "retain A0 reference only",
        "hybrid": "A2L decision archived, but no executor to execute",
        "reject": "A2D/A3/A4 execution blocked",
        "blocked_reason": blocked_reason,
        "root_seed": ROOT_SEED,
        "a0_reference": a0_reference,
        "a1_prediction_hash_contract": "same_as_a0_not_replayed",
        "a2l": a2l,
        "a2d": a2d,
        "a3": a3,
        "a4": {"status": verdict, "reason": blocked_reason},
        "gates": protocol["gates"],
        "selection_fold_ids": SELECTION_FOLDS,
        "promotion_fold_ids": PROMOTION_FOLDS,
        "llm_prompt_contract": {
            "no_raw_metrics": True,
            "no_labels": True,
            "no_residuals": True,
            "no_paths": True,
            "only_improved_flat_worse_feedback": True,
        },
        "executor_available": executor is not None,
        "reference_commits": {
            "p10_results_sweetspot": _git_commit(REFERENCE_ROOT),
            "current_worktree": _git_commit(WORKTREE_ROOT),
        },
        "protocol_sha256": protocol_sha,
        "protocol_jsonl_sha256": jsonl_sha,
    }
    summary_sha = _write_json(summary_path, summary)
    evidence = textwrap.dedent(
        f"""
        # P28 T3-only hybrid execution-agent pilot

        Verdict: `{verdict}`

        ## What was actually done

        - Archived A0 reference for T3 XGBoost was frozen from immutable stage-3 / stage-4 evidence.
        - A2L DeepSeek selection was archived under a strict no-raw-metric, no-label, no-path prompt contract when available.
        - A2D deterministic route choice and A3 random-policy choice were computed from the same allowlist.
        - The private P28 worktree does not contain a portable executor entrypoint, so no new online evidence was replayed.

        ## Why it is blocked

        {blocked_reason}

        ## Honest retain / hybrid / reject verdict

        - Retain: A0 archived baseline reference only.
        - Hybrid: A2L decision record is usable as a policy artifact, but not as executed science.
        - Reject: A2D/A3/A4 execution until a legal portable executor is ported into P28 private code.

        ## T5–T7 gate states

        - T5: `not_feasible`
        - T6: `blocked`
        - T7: `blocked`
        """
    ).strip() + "\n"
    evidence_sha = _write_text(evidence_path, evidence)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "source_commit": _git_commit(WORKTREE_ROOT),
        "renderer": {
            "path": "p28/sweetspot_p28_agentic_optimization.py",
            "sha256": _sha256_file(HERE / "sweetspot_p28_agentic_optimization.py"),
        },
        "generated_at": "2026-08-01T00:00:00Z",
        "manual_review": {
            "reviewed": False,
            "reviewer": None,
            "reviewed_at": None,
            "reviewed_sha256": None,
            "colors_consistent": None,
            "labels_legible": None,
            "no_clipping": None,
            "no_overlap": None,
            "scientific_boundary_preserved": None,
            "notes": "pending",
        },
        "inputs": refs,
        "outputs": [
            {
                "role": "protocol",
                "path": _display_path(protocol_path),
                "sha256": protocol_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "protocol_jsonl",
                "path": _display_path(protocol_jsonl_path),
                "sha256": jsonl_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "summary",
                "path": _display_path(summary_path),
                "sha256": summary_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "evidence",
                "path": _display_path(evidence_path),
                "sha256": evidence_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
        ],
        "artifact_count": 4,
    }
    _write_json(manifest_path, manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-deepseek", action="store_true", help="skip the DeepSeek policy call")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = generate_report(args.output_dir, call_deepseek=not args.no_deepseek)
    sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0 if summary["verdict"] != "BLOCKED_EXECUTOR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
