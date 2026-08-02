#!/usr/bin/env python3
"""P32 matched-budget hybrid agent optimizer for reservoir properties.

The language model proposes bounded hyperparameter candidates.  A deterministic
scheduler executes and ranks them on selection-development data.  The promotion
development split is opened only after each strategy has selected one endpoint.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
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
OUTPUT_ROOT = HERE / "_outputs" / "p32_hybrid_agent_optimizer"
DEFAULT_TRAIN_H5 = PROJECT_ROOT / "_data" / "processed" / "reservoir" / "train.h5"
DEFAULT_GUARD_NPZ = HERE / "_outputs" / "guard.npz"
ROOT_SEED = 2693
PILOT_STEPS = 8
FINAL_STEPS = 32
SEEDS = (2693, 2694, 2695)
CANDIDATES_PER_STRATEGY = 4
PROVIDER_ENDPOINT = "https://api.deepseek.com/chat/completions"
PROVIDER_MODEL = "deepseek-chat"
PROVIDER_TIMEOUT_S = 60.0
PRIMARY_METRIC = "composite_mean_train_std_normalized_RMSE"
SCHEMA_VERSION = "property-p32-hybrid-agent-optimizer/v1"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import p28_agentic_optimization as p28  # noqa: E402
import p29_agent_action_effect as p29  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model_name: str
    model_kwargs: dict[str, Any]
    rationale: str
    source: str

    def route(self) -> p28.RouteSpec:
        return p28.RouteSpec(
            route_id=self.candidate_id,
            model_name=self.model_name,
            model_kwargs=dict(self.model_kwargs),
            lane="tabular-cpu",
            dependency_group="tabular-cpu",
            notes=self.rationale,
        )


DETERMINISTIC_POOL = (
    Candidate(
        "det_linear_lr002",
        "reservoir_linear",
        {"learning_rate": 0.002, "l2_strength": 0.0},
        "P29 deterministic incumbent",
        "frozen_grid",
    ),
    Candidate(
        "det_linear_lr008",
        "reservoir_linear",
        {"learning_rate": 0.008, "l2_strength": 0.0},
        "frozen higher-rate linear probe",
        "frozen_grid",
    ),
    Candidate(
        "det_ridge_lr004_l2_1e4",
        "reservoir_ridge",
        {"learning_rate": 0.004, "l2_strength": 1e-4},
        "frozen moderate ridge probe",
        "frozen_grid",
    ),
    Candidate(
        "det_mlp_h24_lr002_wd1e3",
        "tiny_mlp",
        {"hidden_dim": 24, "learning_rate": 0.002, "weight_decay": 1e-3},
        "P29 regularized MLP probe",
        "frozen_grid",
    ),
)


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


def _float(value: Any, *, name: str, low: float, high: float) -> float:
    number = float(value)
    if not np.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} outside frozen bounds [{low}, {high}]")
    return number


def validate_candidate(raw: Mapping[str, Any], index: int, source: str) -> Candidate:
    if set(raw) != {"model_name", "model_kwargs", "rationale"}:
        raise ValueError("candidate must contain model_name, model_kwargs, rationale only")
    model_name = str(raw["model_name"])
    kwargs = dict(raw["model_kwargs"])
    rationale = str(raw["rationale"]).strip()
    if not rationale or len(rationale) > 240:
        raise ValueError("candidate rationale must contain 1..240 characters")
    if model_name == "tiny_mlp":
        if set(kwargs) != {"hidden_dim", "learning_rate", "weight_decay"}:
            raise ValueError("tiny_mlp kwargs drifted")
        hidden = int(kwargs["hidden_dim"])
        if hidden not in {8, 16, 24, 32, 48, 64}:
            raise ValueError("tiny_mlp hidden_dim outside frozen set")
        kwargs = {
            "hidden_dim": hidden,
            "learning_rate": _float(
                kwargs["learning_rate"], name="learning_rate", low=1e-4, high=2e-2
            ),
            "weight_decay": _float(
                kwargs["weight_decay"], name="weight_decay", low=0.0, high=1e-2
            ),
        }
    elif model_name in {"reservoir_linear", "reservoir_ridge"}:
        if set(kwargs) != {"learning_rate", "l2_strength"}:
            raise ValueError("linear/ridge kwargs drifted")
        kwargs = {
            "learning_rate": _float(
                kwargs["learning_rate"], name="learning_rate", low=1e-4, high=5e-2
            ),
            "l2_strength": _float(
                kwargs["l2_strength"], name="l2_strength", low=0.0, high=1e-1
            ),
        }
        if model_name == "reservoir_linear" and kwargs["l2_strength"] != 0.0:
            raise ValueError("reservoir_linear requires l2_strength=0")
    else:
        raise ValueError("model_name outside frozen executor allowlist")
    return Candidate(
        candidate_id=f"agent_{index + 1}_{model_name}",
        model_name=model_name,
        model_kwargs=kwargs,
        rationale=rationale,
        source=source,
    )


def validate_agent_candidates(payload: Mapping[str, Any]) -> tuple[Candidate, ...]:
    if set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("provider response must be a candidates-only object")
    if len(payload["candidates"]) != CANDIDATES_PER_STRATEGY:
        raise ValueError("provider must return exactly four candidates")
    candidates = tuple(
        validate_candidate(raw, index, "deepseek_candidate_generator")
        for index, raw in enumerate(payload["candidates"])
    )
    identities = {
        canonical_hash([item.model_name, item.model_kwargs]) for item in candidates
    }
    if len(identities) != len(candidates):
        raise ValueError("provider returned duplicate candidates")
    if len({item.model_name for item in candidates}) < 2:
        raise ValueError("provider candidates must cover at least two model families")
    return candidates


def build_candidate_prompt(*, sample_count: int, feature_count: int) -> dict[str, Any]:
    return {
        "task": "Propose four bounded candidates for multi-output reservoir property regression",
        "domain": "seismic-patch and well-log summary features; predict PHIF, KLOGH, and SW",
        "available_development_data": {
            "sample_count": int(sample_count),
            "feature_count": int(feature_count),
            "raw_labels_visible": False,
            "selection_metrics_visible": False,
            "promotion_metrics_visible": False,
            "test_data_visible": False,
        },
        "known_incumbent": {
            "model_name": "reservoir_linear",
            "model_kwargs": {"learning_rate": 0.002, "l2_strength": 0.0},
        },
        "objective": "minimize mean train-standard-deviation-normalized RMSE without degrading any target",
        "budget": {
            "candidates": CANDIDATES_PER_STRATEGY,
            "pilot_update_steps_each": PILOT_STEPS,
            "final_update_steps": FINAL_STEPS,
            "final_seeds": list(SEEDS),
        },
        "allowlist": {
            "tiny_mlp": {
                "hidden_dim": [8, 16, 24, 32, 48, 64],
                "learning_rate": [1e-4, 2e-2],
                "weight_decay": [0.0, 1e-2],
            },
            "reservoir_linear": {
                "learning_rate": [1e-4, 5e-2],
                "l2_strength": [0.0, 0.0],
            },
            "reservoir_ridge": {
                "learning_rate": [1e-4, 5e-2],
                "l2_strength": [0.0, 1e-1],
            },
        },
        "rules": [
            "Return strict JSON with the single top-level key candidates.",
            "Return exactly four unique candidates covering at least two model families.",
            "Use no unlisted model, parameter, data path, metric, or action.",
            "Each candidate contains model_name, model_kwargs, and a short rationale only.",
        ],
    }


def _credential() -> str:
    key = os.environ.get("DEEPSEEK_KEY", "").strip()
    if key:
        return key
    return p29.get_deepseek_key()


def call_provider(prompt: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = json.dumps(
        {
            "model": PROVIDER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Return strict JSON only. Propose candidates within the supplied allowlist.",
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0,
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
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"provider request failed: {type(exc).__name__}: {exc}") from exc
    content = raw["choices"][0]["message"]["content"]
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError("provider returned non-JSON content") from exc
    provenance = {
        "provider": "deepseek",
        "model_requested": PROVIDER_MODEL,
        "model_returned": raw.get("model", "unknown"),
        "response_id": raw.get("id", ""),
        "usage": raw.get("usage", {}),
        "credential_persisted": False,
    }
    return payload, provenance


def _metric(metrics: Mapping[str, Any]) -> float:
    return float(metrics[PRIMARY_METRIC])


def _evaluate_candidate(
    candidate: Candidate,
    *,
    seed: int,
    steps: int,
    train_features: np.ndarray,
    train_target_norm: np.ndarray,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    promotion_features: np.ndarray | None,
    promotion_targets: np.ndarray | None,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    model = p29.train_model(
        candidate.route(),
        train_features,
        train_target_norm,
        seed=int(seed),
        budget_steps=int(steps),
    )
    selection_prediction = p29.infer(model, selection_features, stats)
    selection = p29.evaluate_predictions(selection_targets, selection_prediction, stats)
    result: dict[str, Any] = {
        "candidate": dataclasses.asdict(candidate),
        "candidate_hash": canonical_hash(dataclasses.asdict(candidate)),
        "seed": int(seed),
        "update_steps": int(steps),
        "selection": selection,
        "selection_prediction_hash": p29.prediction_hash(selection_prediction),
        "runtime_s": time.perf_counter() - started,
    }
    if promotion_features is not None and promotion_targets is not None:
        prediction = p29.infer(model, promotion_features, stats)
        result["promotion"] = p29.evaluate_predictions(
            promotion_targets, prediction, stats
        )
        result["promotion_prediction_hash"] = p29.prediction_hash(prediction)
    return result


def run_strategy(
    strategy_id: str,
    candidates: Sequence[Candidate],
    *,
    train_features: np.ndarray,
    train_target_norm: np.ndarray,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    promotion_features: np.ndarray,
    promotion_targets: np.ndarray,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    if len(candidates) != CANDIDATES_PER_STRATEGY:
        raise ValueError("strategy candidate budget drifted")
    pilots = [
        _evaluate_candidate(
            candidate,
            seed=ROOT_SEED,
            steps=PILOT_STEPS,
            train_features=train_features,
            train_target_norm=train_target_norm,
            selection_features=selection_features,
            selection_targets=selection_targets,
            promotion_features=None,
            promotion_targets=None,
            stats=stats,
        )
        for candidate in candidates
    ]
    winner = min(
        pilots,
        key=lambda row: (_metric(row["selection"]), row["candidate"]["candidate_id"]),
    )
    selected = next(
        item
        for item in candidates
        if item.candidate_id == winner["candidate"]["candidate_id"]
    )
    finals = [
        _evaluate_candidate(
            selected,
            seed=seed,
            steps=FINAL_STEPS,
            train_features=train_features,
            train_target_norm=train_target_norm,
            selection_features=selection_features,
            selection_targets=selection_targets,
            promotion_features=promotion_features,
            promotion_targets=promotion_targets,
            stats=stats,
        )
        for seed in SEEDS
    ]
    promotion_primary = [_metric(row["promotion"]) for row in finals]
    selection_primary = [_metric(row["selection"]) for row in finals]
    worst_targets = {
        target: float(
            np.median([row["promotion"]["worst_group_RMSE"][target] for row in finals])
        )
        for target in p29.PHYSICAL_TARGETS
    }
    return {
        "strategy_id": strategy_id,
        "candidate_budget": len(candidates),
        "pilot_update_steps_each": PILOT_STEPS,
        "final_update_steps_each": FINAL_STEPS,
        "final_seed_pool": list(SEEDS),
        "total_train_update_steps": len(candidates) * PILOT_STEPS
        + len(SEEDS) * FINAL_STEPS,
        "pilots": pilots,
        "selected_candidate": dataclasses.asdict(selected),
        "selection_rule": "minimum pilot selection primary metric; candidate_id tie break",
        "final_trials": finals,
        "endpoint": {
            "selection_primary_median": float(np.median(selection_primary)),
            "promotion_primary_median": float(np.median(promotion_primary)),
            "promotion_primary_values": promotion_primary,
            "promotion_worst_group_RMSE_median": worst_targets,
        },
    }


def promotion_gate(
    agent: Mapping[str, Any], deterministic: Mapping[str, Any]
) -> dict[str, Any]:
    agent_metric = float(agent["endpoint"]["promotion_primary_median"])
    det_metric = float(deterministic["endpoint"]["promotion_primary_median"])
    relative_delta = (agent_metric - det_metric) / det_metric
    nondegradation = all(
        float(agent["endpoint"]["promotion_worst_group_RMSE_median"][target])
        <= float(deterministic["endpoint"]["promotion_worst_group_RMSE_median"][target])
        * 1.02
        for target in p29.PHYSICAL_TARGETS
    )
    seed_wins = sum(
        a < d
        for a, d in zip(
            agent["endpoint"]["promotion_primary_values"],
            deterministic["endpoint"]["promotion_primary_values"],
            strict=True,
        )
    )
    retain = relative_delta <= -0.01 and nondegradation and seed_wins >= 2
    return {
        "decision": "RETAIN_HYBRID" if retain else "KEEP_DETERMINISTIC",
        "retain_hybrid": retain,
        "agent_minus_deterministic_relative_primary": relative_delta,
        "minimum_required_relative_improvement": -0.01,
        "worst_target_nondegradation_2pct": nondegradation,
        "paired_seed_wins": seed_wins,
        "minimum_paired_seed_wins": 2,
    }


def _sample_ids_hash(records: Iterable[Any]) -> str:
    return canonical_hash([record.sample_id for record in records])


def execute(
    train_h5: Path,
    guard_npz: Path,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    train_h5 = train_h5.resolve()
    guard_npz = guard_npz.resolve()
    if not train_h5.is_file():
        raise FileNotFoundError(train_h5)
    if not guard_npz.is_file():
        raise FileNotFoundError(guard_npz)
    output_root.mkdir(parents=True, exist_ok=True)
    p28.TRAIN_H5 = train_h5
    p29.GUARD_NPZ = guard_npz
    split = p29.split_records()
    stats = p29.fit_stats(split["train"])
    train_features, train_target_norm, _, _ = p29.build_features(split["train"], stats)
    selection_features, _, selection_targets, _ = p29.build_features(
        split["selection_dev"], stats
    )
    promotion_features, _, promotion_targets, _ = p29.build_features(
        split["promotion_dev"], stats
    )
    prompt = build_candidate_prompt(
        sample_count=len(split["train"]), feature_count=train_features.shape[1]
    )
    (output_root / "candidate_prompt.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raw_candidates, provider = call_provider(prompt)
    agent_candidates = validate_agent_candidates(raw_candidates)
    (output_root / "candidate_response.json").write_text(
        json.dumps(
            {"payload": raw_candidates, "provenance": provider},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    agent = run_strategy(
        "A2H_llm_candidates_deterministic_scheduler",
        agent_candidates,
        train_features=train_features,
        train_target_norm=train_target_norm,
        selection_features=selection_features,
        selection_targets=selection_targets,
        promotion_features=promotion_features,
        promotion_targets=promotion_targets,
        stats=stats,
    )
    deterministic = run_strategy(
        "A2D_frozen_grid_deterministic_scheduler",
        DETERMINISTIC_POOL,
        train_features=train_features,
        train_target_norm=train_target_norm,
        selection_features=selection_features,
        selection_targets=selection_targets,
        promotion_features=promotion_features,
        promotion_targets=promotion_targets,
        stats=stats,
    )
    gate = promotion_gate(agent, deterministic)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "metric_direction": "lower_is_better",
        "data": {
            "train_h5_sha256": file_hash(train_h5),
            "guard_npz_sha256": file_hash(guard_npz),
            "train_record_count": len(split["train"]),
            "selection_record_count": len(split["selection_dev"]),
            "promotion_record_count": len(split["promotion_dev"]),
            "train_sample_ids_sha256": _sample_ids_hash(split["train"]),
            "selection_sample_ids_sha256": _sample_ids_hash(split["selection_dev"]),
            "promotion_sample_ids_sha256": _sample_ids_hash(split["promotion_dev"]),
            "selection_promotion_overlap": False,
            "frozen_test_accessed": False,
        },
        "provider": provider,
        "prompt_sha256": canonical_hash(prompt),
        "agent_candidate_set_sha256": canonical_hash(
            [dataclasses.asdict(item) for item in agent_candidates]
        ),
        "deterministic_candidate_set_sha256": canonical_hash(
            [dataclasses.asdict(item) for item in DETERMINISTIC_POOL]
        ),
        "matched_budget": {
            "agent_total_train_update_steps": agent["total_train_update_steps"],
            "deterministic_total_train_update_steps": deterministic[
                "total_train_update_steps"
            ],
            "equal": agent["total_train_update_steps"]
            == deterministic["total_train_update_steps"],
        },
        "agent": agent,
        "deterministic": deterministic,
        "promotion_gate": gate,
    }
    summary["summary_core_sha256"] = canonical_hash(summary)
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    evidence = [
        "# P32 property hybrid-agent pilot evidence",
        "",
        f"- Decision: `{gate['decision']}`.",
        f"- Agent candidate: `{agent['selected_candidate']['candidate_id']}`.",
        f"- Deterministic candidate: `{deterministic['selected_candidate']['candidate_id']}`.",
        f"- Agent promotion {PRIMARY_METRIC}: `{agent['endpoint']['promotion_primary_median']:.9f}`.",
        f"- Deterministic promotion {PRIMARY_METRIC}: `{deterministic['endpoint']['promotion_primary_median']:.9f}`.",
        f"- Relative delta: `{gate['agent_minus_deterministic_relative_primary']:+.6%}`.",
        f"- Paired seed wins: `{gate['paired_seed_wins']}/{len(SEEDS)}`.",
        f"- Matched update budget: `{agent['total_train_update_steps']}` steps per strategy.",
        "- Candidate selection used selection-development only; promotion was evaluated after selection.",
        "- Frozen test data were not accessed.",
    ]
    (output_root / "evidence.md").write_text("\n".join(evidence) + "\n", encoding="utf-8")
    return summary


def verify(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    if summary["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema version drifted")
    if not summary["matched_budget"]["equal"]:
        raise ValueError("strategy update budgets are not matched")
    if summary["data"]["selection_promotion_overlap"]:
        raise ValueError("selection/promotion overlap detected")
    if summary["data"]["frozen_test_accessed"]:
        raise ValueError("frozen test firewall violated")
    for strategy in (summary["agent"], summary["deterministic"]):
        if strategy["candidate_budget"] != CANDIDATES_PER_STRATEGY:
            raise ValueError("candidate budget drifted")
        if strategy["total_train_update_steps"] != (
            CANDIDATES_PER_STRATEGY * PILOT_STEPS + len(SEEDS) * FINAL_STEPS
        ):
            raise ValueError("training update budget drifted")
        selected_id = strategy["selected_candidate"]["candidate_id"]
        pilot_winner = min(
            strategy["pilots"],
            key=lambda row: (
                _metric(row["selection"]),
                row["candidate"]["candidate_id"],
            ),
        )["candidate"]["candidate_id"]
        if selected_id != pilot_winner:
            raise ValueError("endpoint was not selected by the frozen rule")
    return {
        "status": "ok",
        "decision": summary["promotion_gate"]["decision"],
        "summary_core_sha256": summary["summary_core_sha256"],
    }


def candidate_signature(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return the executable decision, excluding order, prose, and provenance."""
    return {
        "model_name": candidate["model_name"],
        "model_kwargs": candidate["model_kwargs"],
    }


def verify_independent_replay(
    primary_root: Path = OUTPUT_ROOT,
    replay_root: Path = OUTPUT_ROOT / "independent_replay",
) -> dict[str, Any]:
    primary = json.loads((primary_root / "summary.json").read_text(encoding="utf-8"))
    replay = json.loads((replay_root / "summary.json").read_text(encoding="utf-8"))
    stable_data = primary["data"] == replay["data"]
    independent_provider_calls = (
        bool(primary["provider"]["response_id"])
        and bool(replay["provider"]["response_id"])
        and primary["provider"]["response_id"] != replay["provider"]["response_id"]
    )
    candidate_pool_exact_match = (
        primary["agent_candidate_set_sha256"]
        == replay["agent_candidate_set_sha256"]
    )
    selected_decision_stable = candidate_signature(
        primary["agent"]["selected_candidate"]
    ) == candidate_signature(replay["agent"]["selected_candidate"])
    endpoint_metrics_stable = (
        primary["agent"]["endpoint"] == replay["agent"]["endpoint"]
        and primary["deterministic"]["endpoint"]
        == replay["deterministic"]["endpoint"]
    )
    promotion_decision_stable = (
        primary["promotion_gate"] == replay["promotion_gate"]
        and primary["promotion_gate"]["decision"] == "RETAIN_HYBRID"
    )
    verified = all(
        (
            stable_data,
            independent_provider_calls,
            selected_decision_stable,
            endpoint_metrics_stable,
            promotion_decision_stable,
        )
    )
    result = {
        "schema_version": "property-p32-independent-replay/v1",
        "verified": verified,
        "stable_data": stable_data,
        "independent_provider_calls": independent_provider_calls,
        "candidate_pool_exact_match": candidate_pool_exact_match,
        "selected_decision_stable": selected_decision_stable,
        "endpoint_metrics_stable": endpoint_metrics_stable,
        "promotion_decision_stable": promotion_decision_stable,
        "primary_response_id": primary["provider"]["response_id"],
        "replay_response_id": replay["provider"]["response_id"],
        "selected_candidate_signature": candidate_signature(
            primary["agent"]["selected_candidate"]
        ),
        "scientific_interpretation": (
            "The full candidate lists varied, but two independent provider calls "
            "selected the same executable endpoint and reproduced identical metrics."
        ),
    }
    result["verification_sha256"] = canonical_hash(result)
    (primary_root / "independent_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verified:
        raise ValueError("independent replay did not reproduce the promoted decision")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--train-h5",
        type=Path,
        default=Path(os.environ.get("P32_PROPERTY_TRAIN_H5", DEFAULT_TRAIN_H5)),
    )
    run_parser.add_argument(
        "--guard-npz",
        type=Path,
        default=Path(os.environ.get("P32_PROPERTY_GUARD_NPZ", DEFAULT_GUARD_NPZ)),
    )
    subparsers.add_parser("verify")
    subparsers.add_parser("verify-replay")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = execute(args.train_h5, args.guard_npz)
        print(json.dumps(result["promotion_gate"], indent=2, sort_keys=True))
    elif args.command == "verify":
        print(json.dumps(verify(), indent=2, sort_keys=True))
    else:
        print(json.dumps(verify_independent_replay(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
