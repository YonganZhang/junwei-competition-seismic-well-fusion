#!/usr/bin/env python3
"""Matched-budget hybrid agent search for the lithofacies XGBoost baseline.

The LLM proposes four bounded configurations.  A deterministic scheduler ranks
them on LOGO folds 0--2.  Fold 3 is opened only after each strategy has chosen
one endpoint.  The frozen test is never accepted as an input.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
OUTPUT_ROOT = HERE / "_outputs" / "p33_hybrid_agent_optimizer"
SCHEMA_VERSION = "lithofacies-p33-hybrid-agent-optimizer/v1"
VERIFY_SCHEMA = "lithofacies-p33-independent-replay/v1"
EXPECTED_SPLIT_HASH = (
    "a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555"
)
SELECTION_FOLDS = (0, 1, 2)
PROMOTION_FOLD = 3
ROUNDS = 60
CANDIDATE_COUNT = 4
PROVIDER_ENDPOINT = "https://api.deepseek.com/chat/completions"
PROVIDER_MODEL = "deepseek-chat"
PROVIDER_TIMEOUT_S = 60.0
PRIMARY_METRIC = "fixed_schema_macro_f1"

for root in (str(PROJECT_ROOT), str(HERE)):
    if root not in sys.path:
        sys.path.insert(0, root)

import lithofacies_default_baseline as baseline  # noqa: E402
from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    NUM_CLASSES,
    multimodal_numpy_features,
    require_dependency,
)
from lithofacies_p5_stage3 import REPEAT_SEEDS, _fold_arrays, load_stage3_batch  # noqa: E402
from p4_contract import classification_metrics_from_logits  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    max_depth: int
    eta: float
    subsample: float
    colsample_bytree: float
    rationale: str
    source: str


DETERMINISTIC_POOL = (
    Candidate("det_a0", 3, 0.10, 1.00, 1.00, "current default", "frozen_grid"),
    Candidate("det_shallow", 2, 0.14, 0.90, 1.00, "shallow regularized", "frozen_grid"),
    Candidate("det_deep", 4, 0.07, 0.85, 0.90, "deeper lower-rate", "frozen_grid"),
    Candidate("det_diverse", 5, 0.05, 0.75, 0.80, "strong stochastic regularization", "frozen_grid"),
)
INCUMBENT = DETERMINISTIC_POOL[0]


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_float(value: Any, name: str, low: float, high: float) -> float:
    number = float(value)
    if not np.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} outside frozen bounds [{low}, {high}]")
    return number


def validate_candidate(raw: Mapping[str, Any], index: int, source: str) -> Candidate:
    expected = {
        "max_depth",
        "eta",
        "subsample",
        "colsample_bytree",
        "rationale",
    }
    if set(raw) != expected:
        raise ValueError("candidate schema drifted")
    depth = int(raw["max_depth"])
    if depth not in {2, 3, 4, 5, 6}:
        raise ValueError("max_depth outside frozen set")
    rationale = str(raw["rationale"]).strip()
    if not rationale or len(rationale) > 240:
        raise ValueError("rationale must contain 1..240 characters")
    return Candidate(
        candidate_id=f"agent_{index + 1}",
        max_depth=depth,
        eta=_bounded_float(raw["eta"], "eta", 0.03, 0.20),
        subsample=_bounded_float(raw["subsample"], "subsample", 0.75, 1.0),
        colsample_bytree=_bounded_float(
            raw["colsample_bytree"], "colsample_bytree", 0.75, 1.0
        ),
        rationale=rationale,
        source=source,
    )


def executable_config(candidate: Candidate | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(candidate, Candidate):
        return {
            "max_depth": candidate.max_depth,
            "eta": candidate.eta,
            "rounds": ROUNDS,
            "subsample": candidate.subsample,
            "colsample_bytree": candidate.colsample_bytree,
        }
    return {
        key: candidate[key]
        for key in ("max_depth", "eta", "rounds", "subsample", "colsample_bytree")
    }


def candidate_signature(candidate: Candidate | Mapping[str, Any]) -> str:
    return canonical_hash(executable_config(candidate))


def validate_agent_candidates(payload: Mapping[str, Any]) -> tuple[Candidate, ...]:
    if set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("provider response must contain candidates only")
    if len(payload["candidates"]) != CANDIDATE_COUNT:
        raise ValueError("provider must return exactly four candidates")
    candidates = tuple(
        validate_candidate(raw, index, "deepseek_candidate_generator")
        for index, raw in enumerate(payload["candidates"])
    )
    if len({candidate_signature(item) for item in candidates}) != CANDIDATE_COUNT:
        raise ValueError("provider returned duplicate executable candidates")
    return candidates


def build_candidate_prompt() -> dict[str, Any]:
    return {
        "task": "Propose four XGBoost candidates for nine-class well-log/seismic lithofacies classification",
        "known_incumbent": {
            "max_depth": 3,
            "eta": 0.1,
            "rounds": ROUNDS,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
        "objective": "maximize fixed-schema Macro-F1 under family-isolated LOGO validation",
        "data_boundary": {
            "raw_rows_visible": False,
            "selection_metrics_visible": False,
            "promotion_metrics_visible": False,
            "test_visible": False,
        },
        "budget": {
            "candidate_count": CANDIDATE_COUNT,
            "rounds_each": ROUNDS,
            "selection_folds": list(SELECTION_FOLDS),
            "promotion_fold_hidden": True,
        },
        "allowlist": {
            "max_depth": [2, 3, 4, 5, 6],
            "eta": [0.03, 0.20],
            "subsample": [0.75, 1.0],
            "colsample_bytree": [0.75, 1.0],
            "rounds": ROUNDS,
        },
        "rules": [
            "Return strict JSON with the single top-level key candidates.",
            "Return exactly four unique candidates.",
            "Each candidate contains max_depth, eta, subsample, colsample_bytree, rationale only.",
            "Do not request data, paths, labels, metrics, a model family change, or promotion access.",
        ],
    }


def _credential() -> str:
    key = os.environ.get("DEEPSEEK_KEY", "").strip()
    if key:
        return key
    helper = Path.home() / ".claude/skills/share-docs/scripts/get-credential.sh"
    if not helper.is_file():
        raise RuntimeError("DeepSeek credential helper is unavailable")
    result = subprocess.run(
        [str(helper), "DEEPSEEK_API_KEY"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    key = result.stdout.strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is unavailable")
    return key


def call_provider(prompt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = json.dumps(
        {
            "model": PROVIDER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Return strict JSON only. Stay inside the supplied hyperparameter allowlist.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        PROVIDER_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {_credential()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_S) as response:
            raw = json.loads(response.read().decode("utf-8"))
        payload = json.loads(raw["choices"][0]["message"]["content"])
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"provider request failed: {type(exc).__name__}: {exc}") from exc
    return payload, {
        "provider": "deepseek",
        "model_requested": PROVIDER_MODEL,
        "model_returned": raw.get("model", "unknown"),
        "response_id": raw.get("id", ""),
        "usage": raw.get("usage", {}),
        "credential_persisted": False,
    }


def _fit_predict(candidate: Candidate, fold: Mapping[str, np.ndarray], seed: int) -> np.ndarray:
    train_features = multimodal_numpy_features(
        np.asarray(fold["p_train_well"], dtype=np.float32),
        np.asarray(fold["p_train_seismic"], dtype=np.float32),
    )
    labels = np.asarray(fold["p_train_labels"], dtype=np.int64)
    counts = np.asarray(fold["class_counts"], dtype=np.float64)
    weights_by_class = np.zeros(NUM_CLASSES, dtype=np.float64)
    supported = counts > 0
    weights_by_class[supported] = 1.0 / np.sqrt(counts[supported])
    xgboost = require_dependency("xgboost_multisoftprob_window", "xgboost")
    train = xgboost.DMatrix(train_features, label=labels, weight=weights_by_class[labels])
    booster = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": NUM_CLASSES,
            "max_depth": candidate.max_depth,
            "eta": candidate.eta,
            "subsample": candidate.subsample,
            "colsample_bytree": candidate.colsample_bytree,
            "tree_method": "hist",
            "seed": int(seed),
            "nthread": 1,
            "verbosity": 0,
        },
        train,
        num_boost_round=ROUNDS,
    )
    validation_features = multimodal_numpy_features(
        np.asarray(fold["p_validation_well"], dtype=np.float32),
        np.asarray(fold["p_validation_seismic"], dtype=np.float32),
    )
    probabilities = np.asarray(
        booster.predict(xgboost.DMatrix(validation_features)), dtype=np.float64
    )
    return np.log(np.clip(probabilities, 1e-12, 1.0)).astype(np.float32)


def evaluate_candidate(
    candidate: Candidate,
    arrays: Mapping[str, np.ndarray],
    folds: Sequence[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold_id in folds:
        fold = _fold_arrays(arrays, int(fold_id))
        target = np.asarray(fold["p_validation_labels"], dtype=np.int64)
        for repeat_id, seed in enumerate(REPEAT_SEEDS):
            logits = _fit_predict(candidate, fold, int(seed))
            metrics = classification_metrics_from_logits(target.tolist(), logits)
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "repeat_id": int(repeat_id),
                    "seed": int(seed),
                    "metrics": metrics,
                    "prediction_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                }
            )
    values = [float(row["metrics"][PRIMARY_METRIC]) for row in rows]
    return {
        "candidate": dataclasses.asdict(candidate),
        "executable_config": executable_config(candidate),
        "candidate_signature": candidate_signature(candidate),
        "folds": list(folds),
        "cells": len(rows),
        "boosting_rounds": len(rows) * ROUNDS,
        "mean_fixed_schema_macro_f1": float(np.mean(values)),
        "rows": rows,
        "runtime_s": time.perf_counter() - started,
    }


def run_strategy(
    strategy_id: str,
    candidates: Sequence[Candidate],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if len(candidates) != CANDIDATE_COUNT:
        raise ValueError("strategy candidate budget drifted")
    selection = [evaluate_candidate(item, arrays, SELECTION_FOLDS) for item in candidates]
    selected = sorted(
        selection,
        key=lambda item: (-item["mean_fixed_schema_macro_f1"], item["candidate_signature"]),
    )[0]
    selected_candidate = next(
        item for item in candidates if candidate_signature(item) == selected["candidate_signature"]
    )
    promotion = evaluate_candidate(selected_candidate, arrays, (PROMOTION_FOLD,))
    return {
        "strategy_id": strategy_id,
        "selection": selection,
        "selected_candidate_signature": selected["candidate_signature"],
        "selected_executable_config": selected["executable_config"],
        "selection_primary": selected["mean_fixed_schema_macro_f1"],
        "promotion": promotion,
        "selection_boosting_rounds": sum(item["boosting_rounds"] for item in selection),
        "promotion_boosting_rounds": promotion["boosting_rounds"],
    }


def promotion_gate(
    agent: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    incumbent: Mapping[str, Any],
) -> dict[str, Any]:
    agent_values = [
        float(row["metrics"][PRIMARY_METRIC]) for row in agent["promotion"]["rows"]
    ]
    deterministic_values = [
        float(row["metrics"][PRIMARY_METRIC])
        for row in deterministic["promotion"]["rows"]
    ]
    incumbent_values = [
        float(row["metrics"][PRIMARY_METRIC]) for row in incumbent["rows"]
    ]
    delta_deterministic = float(np.mean(agent_values) - np.mean(deterministic_values))
    delta_incumbent = float(np.mean(agent_values) - np.mean(incumbent_values))
    wins_deterministic = sum(
        a > b for a, b in zip(agent_values, deterministic_values, strict=True)
    )
    wins_incumbent = sum(
        a > b for a, b in zip(agent_values, incumbent_values, strict=True)
    )
    retain = (
        delta_deterministic >= 0.005
        and delta_incumbent >= 0.005
        and wins_deterministic >= 2
        and wins_incumbent >= 2
    )
    return {
        "decision": "RETAIN_HYBRID" if retain else "KEEP_CURRENT_DEFAULT",
        "retain_hybrid": retain,
        "agent_minus_deterministic_promotion_macro_f1": delta_deterministic,
        "agent_minus_incumbent_promotion_macro_f1": delta_incumbent,
        "minimum_absolute_improvement": 0.005,
        "paired_seed_wins_vs_deterministic": wins_deterministic,
        "paired_seed_wins_vs_incumbent": wins_incumbent,
        "minimum_paired_seed_wins": 2,
    }


def _owned_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(HERE.resolve())
    except ValueError as exc:
        raise ValueError("output directory must remain under the lithofacies track") from exc
    return resolved


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evidence(summary: Mapping[str, Any]) -> str:
    gate = summary["promotion_gate"]
    agent = summary["agent"]
    deterministic = summary["deterministic"]
    return "\n".join(
        [
            "# Lithofacies P33 hybrid-agent evidence",
            "",
            f"Decision: **{gate['decision']}**.",
            "",
            f"- Agent selected: `{agent['selected_executable_config']}`.",
            f"- Deterministic selected: `{deterministic['selected_executable_config']}`.",
            f"- Promotion Macro-F1 delta: `{gate['agent_minus_deterministic_promotion_macro_f1']:+.12f}`.",
            f"- Promotion delta versus current A0: `{gate['agent_minus_incumbent_promotion_macro_f1']:+.12f}`.",
            f"- Paired seed wins versus deterministic/A0: `{gate['paired_seed_wins_vs_deterministic']}/3` and `{gate['paired_seed_wins_vs_incumbent']}/3`.",
            f"- Matched selection budget: `{summary['matched_budget']['agent_selection_boosting_rounds']}` boosting rounds per strategy.",
            "- Selection uses LOGO folds 0--2; promotion uses fold 3 only.",
            "- Frozen test and known holdout were not read.",
            "- Attribution is hybrid: LLM candidate proposal plus deterministic scheduling and promotion gate.",
            "",
        ]
    )


def execute(development_batch: Path, *, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    development_batch = development_batch.resolve()
    baseline.ensure_development_only_paths((development_batch,))
    output_root = _owned_output(output_root)
    # Fail before the external provider call when the registered executor is absent.
    require_dependency("xgboost_multisoftprob_window", "xgboost")
    arrays, manifest = load_stage3_batch(development_batch)
    if (
        manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("test_metrics_used") is not False
    ):
        raise RuntimeError("development batch violates the frozen LOGO4 contract")
    prompt = build_candidate_prompt()
    raw, provider = call_provider(prompt)
    agent_candidates = validate_agent_candidates(raw)
    agent = run_strategy("A2H_llm_candidates_deterministic_scheduler", agent_candidates, arrays)
    deterministic = run_strategy("A2D_frozen_grid_deterministic_scheduler", DETERMINISTIC_POOL, arrays)
    incumbent_promotion = evaluate_candidate(INCUMBENT, arrays, (PROMOTION_FOLD,))
    gate = promotion_gate(agent, deterministic, incumbent_promotion)
    matched = {
        "agent_selection_boosting_rounds": agent["selection_boosting_rounds"],
        "deterministic_selection_boosting_rounds": deterministic["selection_boosting_rounds"],
        "agent_promotion_boosting_rounds": agent["promotion_boosting_rounds"],
        "deterministic_promotion_boosting_rounds": deterministic["promotion_boosting_rounds"],
    }
    matched["equal"] = (
        matched["agent_selection_boosting_rounds"]
        == matched["deterministic_selection_boosting_rounds"]
        and matched["agent_promotion_boosting_rounds"]
        == matched["deterministic_promotion_boosting_rounds"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "metric_direction": "higher_is_better",
        "data": {
            "development_batch_sha256": file_hash(development_batch),
            "split_hash": EXPECTED_SPLIT_HASH,
            "selection_folds": list(SELECTION_FOLDS),
            "promotion_folds": [PROMOTION_FOLD],
            "selection_promotion_overlap": False,
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
        "provider": provider,
        "prompt_sha256": canonical_hash(prompt),
        "agent_candidate_set_sha256": canonical_hash(
            [executable_config(item) for item in agent_candidates]
        ),
        "deterministic_candidate_set_sha256": canonical_hash(
            [executable_config(item) for item in DETERMINISTIC_POOL]
        ),
        "matched_budget": matched,
        "agent": agent,
        "deterministic": deterministic,
        "incumbent_promotion": incumbent_promotion,
        "promotion_gate": gate,
    }
    summary["summary_core_sha256"] = canonical_hash(summary)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "candidate_prompt.json", prompt)
    _write_json(output_root / "candidate_response.json", {"payload": raw, "provenance": provider})
    _write_json(output_root / "summary.json", summary)
    (output_root / "evidence.md").write_text(_evidence(summary), encoding="utf-8")
    verify(output_root)
    return summary


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root = _owned_output(output_root)
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("schema version drifted")
    data = summary["data"]
    if (
        data.get("selection_promotion_overlap") is not False
        or data.get("frozen_test_accessed") is not False
        or data.get("known_holdout_accessed") is not False
    ):
        raise ValueError("data firewall failed")
    if summary["matched_budget"].get("equal") is not True:
        raise ValueError("strategy budgets are not matched")
    if canonical_hash({k: v for k, v in summary.items() if k != "summary_core_sha256"}) != summary.get("summary_core_sha256"):
        raise ValueError("summary hash mismatch")
    return {
        "status": "ok",
        "decision": summary["promotion_gate"]["decision"],
        "summary_core_sha256": summary["summary_core_sha256"],
    }


def verify_independent_replay(
    primary_root: Path = OUTPUT_ROOT,
    replay_root: Path = OUTPUT_ROOT / "independent_replay",
) -> dict[str, Any]:
    primary = json.loads((primary_root / "summary.json").read_text(encoding="utf-8"))
    replay = json.loads((replay_root / "summary.json").read_text(encoding="utf-8"))
    primary_ids = primary["provider"]["response_id"]
    replay_ids = replay["provider"]["response_id"]
    selected_stable = (
        primary["agent"]["selected_executable_config"]
        == replay["agent"]["selected_executable_config"]
    )
    def endpoint_payload(strategy: Mapping[str, Any]) -> dict[str, Any]:
        promotion = strategy["promotion"]
        return {
            "executable_config": promotion["executable_config"],
            "mean_fixed_schema_macro_f1": promotion[
                "mean_fixed_schema_macro_f1"
            ],
            "rows": [
                {
                    "fold_id": row["fold_id"],
                    "repeat_id": row["repeat_id"],
                    "seed": row["seed"],
                    "metrics": row["metrics"],
                    "prediction_sha256": row["prediction_sha256"],
                }
                for row in promotion["rows"]
            ],
        }

    metrics_stable = endpoint_payload(primary["agent"]) == endpoint_payload(
        replay["agent"]
    )
    decision_stable = primary["promotion_gate"] == replay["promotion_gate"]
    primary_pool = {
        row["candidate_signature"] for row in primary["agent"]["selection"]
    }
    replay_pool = {
        row["candidate_signature"] for row in replay["agent"]["selection"]
    }
    result = {
        "schema_version": VERIFY_SCHEMA,
        "independent_provider_calls": bool(primary_ids and replay_ids and primary_ids != replay_ids),
        "primary_response_id": primary_ids,
        "replay_response_id": replay_ids,
        "candidate_pool_exact_match": primary_pool == replay_pool,
        "candidate_pool_overlap_count": len(primary_pool & replay_pool),
        "selected_decision_stable": selected_stable,
        "endpoint_metrics_stable": metrics_stable,
        "promotion_decision_stable": decision_stable,
        "selected_executable_config": primary["agent"]["selected_executable_config"],
    }
    result["verified"] = all(
        result[key]
        for key in (
            "independent_provider_calls",
            "selected_decision_stable",
            "endpoint_metrics_stable",
            "promotion_decision_stable",
        )
    )
    result["verification_sha256"] = canonical_hash(result)
    _write_json(primary_root / "independent_verification.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--development-batch", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    sub.add_parser("verify")
    sub.add_parser("verify-replay")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "run":
        payload = execute(args.development_batch, output_root=args.output_root)
        result = {
            "decision": payload["promotion_gate"]["decision"],
            "agent_config": payload["agent"]["selected_executable_config"],
            "deterministic_config": payload["deterministic"]["selected_executable_config"],
            "promotion_delta": payload["promotion_gate"]["agent_minus_deterministic_promotion_macro_f1"],
            "promotion_delta_vs_incumbent": payload["promotion_gate"]["agent_minus_incumbent_promotion_macro_f1"],
        }
    elif args.command == "verify":
        result = verify()
    else:
        result = verify_independent_replay()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
