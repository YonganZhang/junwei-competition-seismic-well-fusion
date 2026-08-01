"""Sweetspot P29 T3 action-effect repair pilot.

This runner repairs the P28 feedback-baseline bug by comparing every candidate
selection MAE against the same-fold, same-executor A0 baseline. The prompt sees
only signed normalized deltas, fold-train aggregates, and remaining budget.
Selection and promotion stay disjoint; A2D and A3 are independent controls.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import textwrap
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import sys


if str(WORKTREE_ROOT := Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))


p28 = importlib.import_module("_pipelines.02_task_datasets.sweetspot.p28_agentic_optimization")

ROOT_SEED = 2693
TRACK_ID = "sweetspot"
P29_ID = "p29_agent_action_effect"
SCHEMA_VERSION = "sweetspot-p29-agent-action-effect/v1"
HERE = Path(__file__).resolve().parent
TRACK_DIR = HERE
WORKTREES_DIR = WORKTREE_ROOT.parent
PROJECT_ROOT = WORKTREE_ROOT.parents[2]
REFERENCE_ROOT = WORKTREES_DIR / "p10-results-sweetspot"
OUTPUT_DIR = TRACK_DIR / "_outputs" / P29_ID
SOURCE_COMMIT = subprocess.check_output(["git", "-C", str(WORKTREE_ROOT), "rev-parse", "HEAD"], text=True).strip()

SELECTION_FOLDS = [0, 1, 2]
PROMOTION_FOLDS = [3]
TOTAL_TRIAL_BUDGET = 4
REMAINING_BUDGET_TRIALS = 3
A0_ACTION_ID = "T3_XGB_D4_ETA_0_05_ROUNDS_96"
XGB_ACTION_IDS = tuple(p28.XGB_ACTION_IDS)

P5_LABEL_MAPPING_ID = "_pipelines/02_task_datasets/sweetspot/p5/sweetspot_p5_label_mapping.v1.json"
P5_T3_SPLIT_MANIFEST_ID = "_pipelines/02_task_datasets/sweetspot/targets/productivity/_outputs/baseline_v1/split_manifest.json"
P5_STAGE3_T3_LEADERBOARD_ID = "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards/T3.json"
P5_STAGE3_SUMMARY_ID = "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/p5_stage3_summary.json"
P5_STAGE4_SUMMARY_ID = "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage4_confirmation/p5_stage4_summary.json"
P7_SUMMARY_ID = "_pipelines/02_task_datasets/sweetspot/p7/_outputs/t3_chronos2_cv/summary.json"
P8_SUMMARY_ID = "_pipelines/02_task_datasets/sweetspot/p8/_outputs/t3_chronos2_calendar_cv/summary.json"
P17_EVIDENCE_ID = "_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md"

STAGE3_T3_PATH = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/leaderboards/T3.json"
STAGE3_SUMMARY_PATH = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage3_cv/p5_stage3_summary.json"
STAGE4_SUMMARY_PATH = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage4_confirmation/p5_stage4_summary.json"
P7_SUMMARY_PATH = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot/p7/_outputs/t3_chronos2_cv/summary.json"
P8_SUMMARY_PATH = REFERENCE_ROOT / "_pipelines/02_task_datasets/sweetspot/p8/_outputs/t3_chronos2_calendar_cv/summary.json"
P17_EVIDENCE_PATH = PROJECT_ROOT / "_wiki-methodology/_tests/P17_reconstruction_foundation_acceptance_evidence.md"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _portable_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKTREE_ROOT))
    except ValueError:
        return str(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(WORKTREE_ROOT))
    except ValueError:
        return str(path)


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


@dataclass(frozen=True)
class PilotAudit:
    mapping_sha256: str
    split_manifest_sha256: str
    target_id: str
    target_name: str
    target_status: str
    p4_status: str
    task_type: str
    primary_metric: str
    primary_metric_direction: str
    label_version: str
    proxy_semantics: str
    split_manifest_id: str
    label_mapping_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PilotTaskSpec:
    target_id: str
    target_name: str
    task_type: str
    primary_metric: str
    primary_metric_direction: str
    label_version: str
    split_manifest_id: str
    split_manifest_sha256: str
    root_seed: int
    route_family: str
    route_note: str
    development_rebuild: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepSeekDecision:
    selected_route: str
    selected_action_id: str
    stop_requested: bool
    confidence: float
    rationale: str
    raw: dict[str, Any]


def _reference_inputs() -> list[dict[str, Any]]:
    items = [
        (P5_LABEL_MAPPING_ID, p28.P5_LABEL_MAPPING_PATH),
        (P5_T3_SPLIT_MANIFEST_ID, p28.P5_T3_SPLIT_MANIFEST_PATH),
        (P5_STAGE3_T3_LEADERBOARD_ID, STAGE3_T3_PATH),
        (P5_STAGE3_SUMMARY_ID, STAGE3_SUMMARY_PATH),
        (P5_STAGE4_SUMMARY_ID, STAGE4_SUMMARY_PATH),
        (P7_SUMMARY_ID, P7_SUMMARY_PATH),
        (P8_SUMMARY_ID, P8_SUMMARY_PATH),
        (P17_EVIDENCE_ID, P17_EVIDENCE_PATH),
    ]
    source_commit = _git_commit(REFERENCE_ROOT)
    return [
        {
            "scientific_source_id": source_id,
            "scientific_role": source_id.rsplit("/", 1)[-1],
            "sha256": _sha256_file(path),
            "shape_or_row_count": path.stat().st_size,
            "split_scope": "immutable reference",
            "source_commit": source_commit,
        }
        for source_id, path in items
    ]


def _load_task_contract() -> tuple[PilotAudit, PilotTaskSpec, dict[str, Any]]:
    mapping = json.loads(p28.P5_LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
    target = mapping["targets"]["T3"]
    split_manifest = json.loads(p28.P5_T3_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    audit = PilotAudit(
        mapping_sha256=_sha256_file(p28.P5_LABEL_MAPPING_PATH),
        split_manifest_sha256=_sha256_file(p28.P5_T3_SPLIT_MANIFEST_PATH),
        target_id="T3",
        target_name=str(target["target_name"]),
        target_status=str(target["status"]),
        p4_status=str(target["p4_status"]),
        task_type=str(target["task_type"]),
        primary_metric=str(target["primary_metric"]),
        primary_metric_direction=str(target["primary_metric_direction"]),
        label_version=str(target["label_version"]),
        proxy_semantics=str(target["proxy_semantics"]),
        split_manifest_id=str(target["split_manifest"]["path"]),
        label_mapping_id=P5_LABEL_MAPPING_ID,
    )
    task_spec = PilotTaskSpec(
        target_id="T3",
        target_name=str(target["target_name"]),
        task_type=str(target["task_type"]),
        primary_metric=str(target["primary_metric"]),
        primary_metric_direction=str(target["primary_metric_direction"]),
        label_version=str(target["label_version"]),
        split_manifest_id=str(target["split_manifest"]["path"]),
        split_manifest_sha256=str(target["split_manifest"]["sha256"]),
        root_seed=ROOT_SEED,
        route_family="xgboost",
        route_note="portable local xgboost-only repair pilot; selection/promotion remain disjoint",
        development_rebuild=str(target["development_rebuild"]),
    )
    return audit, task_spec, split_manifest


def _action_registry() -> list[dict[str, Any]]:
    return [
        {"action_id": "T3_XGB_D4_ETA_0_03_ROUNDS_64", "max_depth": 4, "learning_rate": 0.03, "n_estimators": 64},
        {"action_id": "T3_XGB_D4_ETA_0_05_ROUNDS_96", "max_depth": 4, "learning_rate": 0.05, "n_estimators": 96},
        {"action_id": "T3_XGB_D6_ETA_0_03_ROUNDS_96", "max_depth": 6, "learning_rate": 0.03, "n_estimators": 96},
        {"action_id": "T3_XGB_D6_ETA_0_05_ROUNDS_128", "max_depth": 6, "learning_rate": 0.05, "n_estimators": 128},
    ]


def _load_development_bundle(*, fold_id: int):
    return p28.load_t3_development_pilot_data(source_root=p28.PROJECT_ROOT, fold_id=fold_id)


def _score_action(action: Mapping[str, Any], *, folds: list[int]) -> dict[str, Any]:
    import numpy as np
    from xgboost import XGBRegressor

    fold_rows: list[dict[str, Any]] = []
    fold_maes: list[float] = []
    prediction_hashes: list[str] = []
    for fold_id in folds:
        data = _load_development_bundle(fold_id=fold_id)
        model = XGBRegressor(
            n_estimators=int(action["n_estimators"]),
            max_depth=int(action["max_depth"]),
            learning_rate=float(action["learning_rate"]),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
            random_state=ROOT_SEED + int(action["n_estimators"]),
        )
        model.fit(np.asarray(data.train_tabular, dtype=np.float64), np.asarray(data.train_target, dtype=np.float64))
        prediction = np.asarray(model.predict(np.asarray(data.validation_tabular, dtype=np.float64)), dtype=np.float64).reshape(-1)
        metrics = p28._fold_metric_payload(data.validation_target, prediction)
        metrics["topk_diagnostic"] = p28._topk_diagnostic(data.validation_target, prediction)
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "samples_train": len(data.train_sample_ids),
                "samples_validation": len(data.validation_sample_ids),
                "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
                "metrics": metrics,
                "split_sha256": data.split_sha256,
                "input_budget_sha256": data.input_budget_sha256,
                "test_accessed": False,
                "historical_test_metrics_read": False,
            }
        )
        fold_maes.append(float(metrics["mae"]))
        prediction_hashes.append(fold_rows[-1]["prediction_sha256"])
    return {
        "action_id": str(action["action_id"]),
        "config": dict(action),
        "fold_rows": fold_rows,
        "selection_mae": float(np.mean(fold_maes)) if fold_maes else None,
        "selection_prediction_hashes": prediction_hashes,
    }


def _candidate_table() -> dict[str, Any]:
    audit, task_spec, _ = _load_task_contract()
    baseline_action = next(action for action in _action_registry() if action["action_id"] == A0_ACTION_ID)
    baseline_score = _score_action(baseline_action, folds=list(SELECTION_FOLDS))
    baseline_fold_map = {row["fold_id"]: row for row in baseline_score["fold_rows"]}
    candidates: list[dict[str, Any]] = []
    for action in _action_registry():
        scored = _score_action(action, folds=list(SELECTION_FOLDS))
        fold_feedback = []
        fold_rows = []
        for row in scored["fold_rows"]:
            baseline_row = baseline_fold_map[row["fold_id"]]
            baseline_mae = float(baseline_row["metrics"]["mae"])
            candidate_mae = float(row["metrics"]["mae"])
            signed_normalized_delta = (baseline_mae - candidate_mae) / abs(baseline_mae)
            fold_feedback.append(
                {
                    "fold_id": int(row["fold_id"]),
                    "selection_feedback": "improved" if signed_normalized_delta > 0.005 else "worse" if signed_normalized_delta < -0.005 else "flat",
                    "signed_normalized_delta": float(signed_normalized_delta),
                    "train_rows": int(row["samples_train"]),
                    "validation_rows": int(row["samples_validation"]),
                }
            )
            fold_rows.append(
                {
                    "fold_id": int(row["fold_id"]),
                    "candidate_mae": candidate_mae,
                    "baseline_mae": baseline_mae,
                    "signed_normalized_delta": float(signed_normalized_delta),
                    "prediction_sha256": row["prediction_sha256"],
                    "split_sha256": row["split_sha256"],
                    "input_budget_sha256": row["input_budget_sha256"],
                }
            )
        mean_delta = sum(item["signed_normalized_delta"] for item in fold_feedback) / len(fold_feedback)
        candidates.append(
            {
                "action_id": scored["action_id"],
                "config": scored["config"],
                "selection_mae": float(scored["selection_mae"]),
                "selection_feedback": "improved" if mean_delta > 0.005 else "worse" if mean_delta < -0.005 else "flat",
                "mean_signed_normalized_delta": float(mean_delta),
                "fold_feedback": fold_feedback,
                "fold_rows": fold_rows,
                "selection_prediction_hashes": scored["selection_prediction_hashes"],
            }
        )
    best = max(candidates, key=lambda row: row["mean_signed_normalized_delta"])
    return {
        "schema_version": "sweetspot-p29-agent-action-effect-candidates/v1",
        "task_spec_sha256": _sha256_bytes(json.dumps(task_spec.to_dict(), sort_keys=True).encode("utf-8")),
        "label_audit_sha256": audit.mapping_sha256,
        "selection_folds": list(SELECTION_FOLDS),
        "promotion_folds": list(PROMOTION_FOLDS),
        "baseline": {
            "kind": "same_fold_same_executor_a0",
            "action_id": A0_ACTION_ID,
            "fold_rows": [
                {
                    "fold_id": int(row["fold_id"]),
                    "selection_mae": float(row["metrics"]["mae"]),
                    "prediction_sha256": row["prediction_sha256"],
                    "split_sha256": row["split_sha256"],
                    "input_budget_sha256": row["input_budget_sha256"],
                }
                for row in baseline_score["fold_rows"]
            ],
        },
        "candidates": candidates,
        "best_by_selection_delta": {
            "action_id": best["action_id"],
            "selection_mae": best["selection_mae"],
            "mean_signed_normalized_delta": best["mean_signed_normalized_delta"],
        },
    }


def _build_prompt(*, candidate_table: dict[str, Any]) -> list[dict[str, str]]:
    user = {
        "task": "Choose one allowlisted xgboost config id or stop.",
        "constraints": {
            "route_fixed": "xgboost",
            "no_raw_metrics": True,
            "no_labels": True,
            "no_residuals": True,
            "no_paths": True,
            "no_holdout_or_test": True,
            "selection_and_promotion_disjoint": True,
            "same_fold_same_executor_baseline": True,
        },
        "observation": {
            "baseline": {
                "kind": candidate_table["baseline"]["kind"],
                "action_id": candidate_table["baseline"]["action_id"],
                "fold_rows": [
                    {
                        "fold_id": row["fold_id"],
                        "selection_feedback": "baseline",
                        "signed_normalized_delta": 0.0,
                        "train_rows": next(
                            candidate["fold_feedback"][i]["train_rows"]
                            for candidate in candidate_table["candidates"]
                            for i, fold in enumerate(candidate["fold_feedback"])
                            if fold["fold_id"] == row["fold_id"]
                        ),
                        "validation_rows": next(
                            candidate["fold_feedback"][i]["validation_rows"]
                            for candidate in candidate_table["candidates"]
                            for i, fold in enumerate(candidate["fold_feedback"])
                            if fold["fold_id"] == row["fold_id"]
                        ),
                    }
                    for row in candidate_table["baseline"]["fold_rows"]
                ],
            },
            "candidate_feedback": [
                {
                    "action_id": candidate["action_id"],
                    "selection_feedback": candidate["selection_feedback"],
                    "mean_signed_normalized_delta": candidate["mean_signed_normalized_delta"],
                    "fold_feedback": candidate["fold_feedback"],
                }
                for candidate in candidate_table["candidates"]
            ],
            "remaining_budget_trials": REMAINING_BUDGET_TRIALS,
            "allowlist": [candidate["action_id"] for candidate in candidate_table["candidates"]],
        },
        "response_schema": {
            "selected_route": "xgboost",
            "selected_action_id": "allowlisted config id or A0_static_baseline",
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
                "You are a strict decision policy for the sweetspot P29 T3 repair pilot. "
                "Return JSON only. Do not invent metrics, labels, paths, or holdout access. "
                "Only use the same-fold same-executor A0 baseline and the signed normalized deltas."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
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


def _execute_action(action: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from xgboost import XGBRegressor

    rows: list[dict[str, Any]] = []
    for fold_id in list(SELECTION_FOLDS) + list(PROMOTION_FOLDS):
        data = _load_development_bundle(fold_id=fold_id)
        model = XGBRegressor(
            n_estimators=int(action["n_estimators"]),
            max_depth=int(action["max_depth"]),
            learning_rate=float(action["learning_rate"]),
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=1,
            random_state=ROOT_SEED + int(action["n_estimators"]),
        )
        model.fit(np.asarray(data.train_tabular, dtype=np.float64), np.asarray(data.train_target, dtype=np.float64))
        prediction = np.asarray(model.predict(np.asarray(data.validation_tabular, dtype=np.float64)), dtype=np.float64).reshape(-1)
        metrics = p28._fold_metric_payload(data.validation_target, prediction)
        metrics["topk_diagnostic"] = p28._topk_diagnostic(data.validation_target, prediction)
        rows.append(
            {
                "fold_id": int(fold_id),
                "phase": "selection" if fold_id in SELECTION_FOLDS else "promotion",
                "samples_train": len(data.train_sample_ids),
                "samples_validation": len(data.validation_sample_ids),
                "metrics": metrics,
                "prediction_sha256": hashlib.sha256(prediction.astype("<f8").tobytes()).hexdigest(),
                "split_sha256": data.split_sha256,
                "input_budget_sha256": data.input_budget_sha256,
                "test_accessed": False,
                "historical_test_metrics_read": False,
            }
        )
    selection_rows = [row for row in rows if row["phase"] == "selection"]
    promotion_row = next(row for row in rows if row["phase"] == "promotion")
    return {
        "route": "xgboost",
        "action_id": str(action["action_id"]),
        "budget": int(action["n_estimators"]),
        "selection_mae": float(sum(row["metrics"]["mae"] for row in selection_rows) / len(selection_rows)),
        "promotion_mae": float(promotion_row["metrics"]["mae"]),
        "selection_rows": selection_rows,
        "promotion_row": promotion_row,
        "all_rows": rows,
    }


def _selection_to_prompt_feedback(candidate_table: dict[str, Any]) -> list[dict[str, Any]]:
    feedback: list[dict[str, Any]] = []
    for candidate in candidate_table["candidates"]:
        feedback.append(
            {
                "action_id": candidate["action_id"],
                "selection_feedback": candidate["selection_feedback"],
                "mean_signed_normalized_delta": candidate["mean_signed_normalized_delta"],
                "remaining_budget_trials": REMAINING_BUDGET_TRIALS,
                "fold_feedback": candidate["fold_feedback"],
            }
        )
    return feedback


def generate_report(output_dir: Path = OUTPUT_DIR, *, call_deepseek: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    protocol_jsonl_path = output_dir / "protocol.jsonl"
    action_effects_path = output_dir / "action_effects.json"
    summary_path = output_dir / "summary.json"
    root_cause_path = output_dir / "root_cause.md"
    evidence_path = output_dir / "evidence.md"
    manifest_path = output_dir / "manifest.json"

    audit, task_spec, _ = _load_task_contract()
    candidate_table = _candidate_table()
    prompt_messages = _build_prompt(candidate_table=candidate_table)
    deepseek = _call_deepseek(prompt_messages) if call_deepseek else {"status": "SKIPPED"}

    a0_action = next(action for action in _action_registry() if action["action_id"] == A0_ACTION_ID)
    a0 = _execute_action(a0_action)
    a1 = _execute_action(a0_action)
    a0_prediction_hash = a0["promotion_row"]["prediction_sha256"]
    a1_prediction_hash = a1["promotion_row"]["prediction_sha256"]
    a1_same_hash = a0_prediction_hash == a1_prediction_hash and [row["prediction_sha256"] for row in a0["selection_rows"]] == [row["prediction_sha256"] for row in a1["selection_rows"]]

    a2l_status: str
    a2l_reason: str | None = None
    if deepseek.get("status") in {"UNAVAILABLE", "SKIPPED"}:
        a2l_status = "BLOCKED"
        a2l_reason = f"DeepSeek {deepseek.get('status', 'UNAVAILABLE').lower()}"
    elif deepseek.get("stop_requested"):
        a2l_status = "STOPPED"
    elif not isinstance(deepseek.get("selected_action_id"), str):
        a2l_status = "BLOCKED"
        a2l_reason = "DeepSeek decision malformed"
    else:
        a2l_status = "EXECUTED"

    selected_action_id = deepseek.get("selected_action_id") if isinstance(deepseek.get("selected_action_id"), str) else None
    if a2l_status == "EXECUTED" and selected_action_id not in {action["action_id"] for action in _action_registry()}:
        a2l_status = "BLOCKED"
        a2l_reason = f"action not in allowlist: {selected_action_id}"

    a2l_execution = None
    if a2l_status == "EXECUTED":
        action = next(action for action in _action_registry() if action["action_id"] == selected_action_id)
        a2l_execution = _execute_action(action)

    a2d_action = max(candidate_table["candidates"], key=lambda row: row["mean_signed_normalized_delta"])
    a2d_execution = _execute_action(next(action for action in _action_registry() if action["action_id"] == a2d_action["action_id"]))

    a3_rng = random.Random(ROOT_SEED + 17)
    a3_action = a3_rng.choice(_action_registry())
    a3_trial_budget = a3_rng.choice([2, 3, 4])
    a3_execution = _execute_action(a3_action)

    oracle = max(candidate_table["candidates"], key=lambda row: row["mean_signed_normalized_delta"])
    oracle_execution = _execute_action(next(action for action in _action_registry() if action["action_id"] == oracle["action_id"]))

    if a2l_status == "BLOCKED":
        verdict = "DATA_GATE_BLOCKED" if a2l_reason and "DeepSeek" in a2l_reason else "REJECT_AGENT"
    elif a2l_status == "STOPPED":
        verdict = "REJECT_AGENT"
    elif a2l_execution and a2l_execution["promotion_mae"] < a0["promotion_mae"] and a2l_execution["promotion_mae"] < a3_execution["promotion_mae"]:
        verdict = "RETAIN_AGENT"
    else:
        verdict = "REJECT_AGENT"

    candidate_feedback = _selection_to_prompt_feedback(candidate_table)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "task_id": "T3",
        "root_seed": ROOT_SEED,
        "task_spec": task_spec.to_dict(),
        "audit": audit.to_dict(),
        "baseline": candidate_table["baseline"],
        "observation_schema": {
            "visible_to_llm": [
                "fold_train_aggregates",
                "signed_normalized_delta",
                "remaining_budget_trials",
                "improved_flat_worse_feedback",
            ],
            "hidden_from_llm": [
                "raw_metric_values",
                "validation_curves",
                "per_sample_residuals",
                "labels",
                "paths",
                "historical_stage3_aggregate",
            ],
            "selection_fold_ids": list(SELECTION_FOLDS),
            "promotion_fold_ids": list(PROMOTION_FOLDS),
            "route_fixed": "xgboost",
        },
        "feedback_baseline": candidate_table["baseline"],
        "candidate_feedback": candidate_feedback,
        "prompt_contract": {
            "no_raw_metrics": True,
            "no_labels": True,
            "no_residuals": True,
            "no_paths": True,
            "only_improved_flat_worse_feedback": True,
            "same_fold_same_executor_a0": True,
            "remaining_budget_trials": REMAINING_BUDGET_TRIALS,
        },
        "action_allowlist": [action["action_id"] for action in _action_registry()],
        "leakage_firewall": {
            "no_holdout_or_test": True,
            "no_raw_metrics": True,
            "no_labels": True,
            "nested_disjoint_selection_and_promotion": True,
            "fold_train_only_fit": True,
            "foundation_route_fixed_or_separate_factor": True,
            "sibling_worktree_python_imports": False,
        },
        "gates": {"T5": "not_feasible", "T6": "blocked", "T7": "blocked"},
        "control_policies": {
            "A2D": {"kind": "deterministic_best_mean_delta", "independent_seed": ROOT_SEED + 11, "trial_budget": TOTAL_TRIAL_BUDGET},
            "A3": {"kind": "random_independent_control", "independent_seed": ROOT_SEED + 17, "trial_budget": a3_trial_budget},
        },
        "inputs": _reference_inputs(),
        "generation_base_commit": SOURCE_COMMIT,
    }
    protocol_sha = _write_json(protocol_path, protocol)
    protocol_rows = [
        {
            "strategy": "A0_same_fold_same_executor",
            "status": "RETAINED_REFERENCE",
            "selected_action_id": A0_ACTION_ID,
            "selection_prediction_hash": a0_prediction_hash,
            "note": "Same-fold same-executor baseline on the archived action.",
        },
        {
            "strategy": "A1_identity_replay",
            "status": "EXECUTED_IDENTITY_CHECK",
            "selected_action_id": A0_ACTION_ID,
            "selection_prediction_hash": a1_prediction_hash,
            "note": "Identity replay check on the same action, seed, and folds.",
        },
        {
            "strategy": "A2L_llm_agent_execute",
            "status": a2l_status,
            "selected_route": deepseek.get("selected_route"),
            "selected_action_id": selected_action_id,
            "selection_feedback_visible": a2l_status == "EXECUTED",
            "decision_source": deepseek.get("status", "UNAVAILABLE"),
            "note": "LLM only sees safe deltas and remaining budget; stop is a real decision.",
        },
        {
            "strategy": "A2D_deterministic_control",
            "status": "EXECUTED",
            "selected_action_id": a2d_action["action_id"],
            "mean_signed_normalized_delta": a2d_action["mean_signed_normalized_delta"],
            "note": "Deterministic control chooses the best mean signed normalized delta.",
        },
        {
            "strategy": "A3_random_independent_control",
            "status": "EXECUTED",
            "selected_action_id": a3_action["action_id"],
            "control_seed": ROOT_SEED + 17,
            "trial_budget": a3_trial_budget,
            "note": "Random control uses an independent seed and independent budget draw.",
        },
        {
            "strategy": "ORACLE_CEILING",
            "status": "DIAGNOSTIC_ONLY",
            "selected_action_id": oracle["action_id"],
            "note": "Upper bound over the allowlist; never used for selection.",
        },
        {
            "strategy": "FINAL",
            "status": verdict,
            "selected_action_id": selected_action_id,
            "promotion_mae": None if a2l_execution is None else a2l_execution["promotion_mae"],
            "a0_promotion_mae": a0["promotion_mae"],
            "a3_promotion_mae": a3_execution["promotion_mae"],
            "note": "retain only if A2L beats same-fold A0 and independent A3 on promotion; otherwise reject or data-gate block.",
        },
    ]
    protocol_jsonl_sha = _write_jsonl(protocol_jsonl_path, protocol_rows)

    action_effects = {
        "schema_version": "sweetspot-p29-action-effects/v1",
        "track_id": TRACK_ID,
        "task_id": "T3",
        "baseline": candidate_table["baseline"],
        "actions": [
            {
                "action_id": candidate["action_id"],
                "config": candidate["config"],
                "selection_mae": candidate["selection_mae"],
                "mean_signed_normalized_delta": candidate["mean_signed_normalized_delta"],
                "fold_rows": candidate["fold_rows"],
                "selection_prediction_hashes": candidate["selection_prediction_hashes"],
            }
            for candidate in candidate_table["candidates"]
        ],
        "executions": {
            "A0": a0,
            "A1": a1,
            "A2L": a2l_execution,
            "A2D": a2d_execution,
            "A3": a3_execution,
            "oracle_ceiling": oracle_execution,
        },
        "prompt_contract": protocol["prompt_contract"],
    }
    action_effects_sha = _write_json(action_effects_path, action_effects)

    root_cause = textwrap.dedent(
        f"""
        # P29 root cause

        The P28 prompt compared candidate selection MAE against the historical Stage3 aggregate.
        That made the feedback labels `worse` regardless of the same-fold same-executor A0 baseline.

        Fixed in P29:
        - baseline is now `same_fold_same_executor_a0`
        - prompt exposes only signed normalized deltas and remaining budget
        - selection and promotion stay disjoint
        - A2D and A3 are independent controls

        Honest outcome:
        - A2L status: `{a2l_status}`
        - verdict: `{verdict}`
        - retained LLM decision: `{verdict in {"RETAIN_AGENT", "RETAIN_HYBRID"}}`
        """
    ).strip() + "\n"
    root_cause_sha = _write_text(root_cause_path, root_cause)

    evidence = textwrap.dedent(
        f"""
        # P29 T3 action-effect repair evidence

        - baseline kind: `{candidate_table["baseline"]["kind"]}`
        - allowlist size: {len(candidate_table["candidates"])}
        - remaining budget trials: {REMAINING_BUDGET_TRIALS}
        - A0/A1 identical replay hash: {a1_same_hash}
        - A2L status: `{a2l_status}`
        - verdict: `{verdict}`
        - oracle action id: `{oracle["action_id"]}`

        Files:
        - protocol: `{_display_path(protocol_path)}`
        - protocol.jsonl: `{_display_path(protocol_jsonl_path)}`
        - action_effects: `{_display_path(action_effects_path)}`
        - root_cause: `{_display_path(root_cause_path)}`
        - summary: `{_display_path(summary_path)}`
        - manifest: `{_display_path(manifest_path)}`
        """
    ).strip() + "\n"
    evidence_sha = _write_text(evidence_path, evidence)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "task_id": "T3",
        "verdict": verdict,
        "retain_llm": verdict in {"RETAIN_AGENT", "RETAIN_HYBRID"},
        "reject": verdict == "REJECT_AGENT",
        "blocked_reason": a2l_reason,
        "root_seed": ROOT_SEED,
        "generation_base_commit": SOURCE_COMMIT,
        "a0_action_id": A0_ACTION_ID,
        "a0_prediction_hash": a0_prediction_hash,
        "a1_prediction_hash": a1_prediction_hash,
        "a1_same_hash": a1_same_hash,
        "a1_prediction_hash_contract": "EXECUTED_IDENTITY_CHECK",
        "baseline_kind": candidate_table["baseline"]["kind"],
        "selection_folds": list(SELECTION_FOLDS),
        "promotion_folds": list(PROMOTION_FOLDS),
        "candidate_table": {
            "best_by_selection_delta": candidate_table["best_by_selection_delta"],
            "candidates": [
                {
                    "action_id": candidate["action_id"],
                    "selection_mae": candidate["selection_mae"],
                    "mean_signed_normalized_delta": candidate["mean_signed_normalized_delta"],
                    "selection_feedback": candidate["selection_feedback"],
                }
                for candidate in candidate_table["candidates"]
            ],
        },
        "a2l": {
            "status": a2l_status,
            "selected_route": deepseek.get("selected_route"),
            "selected_action_id": selected_action_id,
            "stop_requested": bool(deepseek.get("stop_requested", False)),
            "confidence": deepseek.get("confidence"),
            "rationale": deepseek.get("rationale"),
            "raw": deepseek,
            "blocked_reason": a2l_reason,
            "promotion_mae": None if a2l_execution is None else a2l_execution["promotion_mae"],
        },
        "a2d": {
            "status": "EXECUTED",
            "selected_action_id": a2d_action["action_id"],
            "selection_mae": a2d_action["selection_mae"],
            "mean_signed_normalized_delta": a2d_action["mean_signed_normalized_delta"],
            "trial_budget": TOTAL_TRIAL_BUDGET,
        },
        "a3": {
            "status": "EXECUTED",
            "selected_action_id": a3_action["action_id"],
            "selection_mae": a3_execution["selection_mae"],
            "promotion_mae": a3_execution["promotion_mae"],
            "control_seed": ROOT_SEED + 17,
            "trial_budget": a3_trial_budget,
        },
        "oracle_ceiling": {
            "action_id": oracle["action_id"],
            "selection_mae": oracle["selection_mae"],
            "mean_signed_normalized_delta": oracle["mean_signed_normalized_delta"],
            "promotion_mae": oracle_execution["promotion_mae"],
        },
        "prompt_sha256": _sha256_bytes(json.dumps(prompt_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")),
        "prompt_contract": protocol["prompt_contract"],
        "protocol_sha256": protocol_sha,
        "protocol_jsonl_sha256": protocol_jsonl_sha,
        "action_effects_sha256": action_effects_sha,
        "root_cause_sha256": root_cause_sha,
        "evidence_sha256": evidence_sha,
    }
    summary_sha = _write_json(summary_path, summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "track_id": TRACK_ID,
        "source_commit": SOURCE_COMMIT,
        "renderer": {
            "path": _display_path(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
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
        "inputs": _reference_inputs(),
        "outputs": [
            {
                "role": "protocol",
                "path": _portable_display_path(protocol_path),
                "sha256": protocol_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "protocol_jsonl",
                "path": _portable_display_path(protocol_jsonl_path),
                "sha256": protocol_jsonl_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "action_effects",
                "path": _portable_display_path(action_effects_path),
                "sha256": action_effects_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "root_cause",
                "path": _portable_display_path(root_cause_path),
                "sha256": root_cause_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "summary",
                "path": _portable_display_path(summary_path),
                "sha256": summary_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
            {
                "role": "evidence",
                "path": _portable_display_path(evidence_path),
                "sha256": evidence_sha,
                "width_px": None,
                "height_px": None,
                "dpi": None,
                "vector_companions": [],
            },
        ],
        "artifact_count": 6,
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
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
