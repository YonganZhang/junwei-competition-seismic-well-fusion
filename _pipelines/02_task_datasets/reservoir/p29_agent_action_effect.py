from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.random import Generator, PCG64


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESERVOIR_DIR = HERE
OUTPUT_DIR = RESERVOIR_DIR / "_outputs" / "p29_agent_action_effect"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAIN_H5 = PROJECT_ROOT / "_data" / "processed" / "reservoir" / "train.h5"
GUARD_NPZ = RESERVOIR_DIR / "_outputs" / "guard.npz"
BASELINE_RUN_MANIFEST = RESERVOIR_DIR / "_outputs" / "run_manifest.json"
BASELINE_METRICS = RESERVOIR_DIR / "_outputs" / "metrics.json"
BASELINE_CHECKPOINT = RESERVOIR_DIR / "_outputs" / "checkpoints" / "best.ckpt"
P18_EVIDENCE = RESERVOIR_DIR / "_outputs" / "p18_cigbench_property" / "evidence.md"

sys.path.insert(0, str(RESERVOIR_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

import p28_agentic_optimization as p28  # noqa: E402
from train_baseline import apply_statistics, fit_train_statistics, inverse_normalized_targets  # noqa: E402


Record = p28.Record
RouteSpec = p28.RouteSpec
ROUTES = p28.ROUTES
PHYSICAL_TARGETS = p28.PHYSICAL_TARGETS
ROOT_SEED = 2693
PILOT_STEPS = 8
DEFAULT_BUDGET_STEPS = 8
SAME_BUDGET_SEEDS = (2693, 2694, 2695)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def json_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train_records() -> list[Record]:
    records = p28.load_records()
    manifest = read_json(BASELINE_RUN_MANIFEST)
    train_manifest_families = set(manifest["families"]["train"])
    train_families = {record.family_id for record in records}
    if train_families != train_manifest_families:
        raise RuntimeError(f"train family mismatch: {sorted(train_families)} != {sorted(train_manifest_families)}")
    return records


def load_guard_records() -> list[Record]:
    if not GUARD_NPZ.is_file():
        raise FileNotFoundError(GUARD_NPZ)
    obj = np.load(GUARD_NPZ, allow_pickle=True)
    records: list[Record] = []
    for idx in range(len(obj["label"])):
        meta = json.loads(str(obj["meta_json"][idx]))
        position = json.loads(str(obj["position_json"][idx]))
        records.append(
            Record(
                sample_id=f"guard_{idx:05d}",
                family_id=str(meta["family_id"]),
                well_id=str(meta["well_id"]),
                depth_m=float(meta["depth_m"]),
                seismic_patch=np.asarray(obj["seismic_patch"][idx], dtype=np.float64),
                well_log_seq=np.asarray(obj["well_log_seq"][idx], dtype=np.float64),
                label=np.asarray(obj["label"][idx], dtype=np.float64).reshape(-1),
                position=position,
            )
        )
    if not records:
        raise RuntimeError("guard.npz is empty")
    return records


def group_records(records: Iterable[Record], key: str) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = {}
    for record in records:
        groups.setdefault(getattr(record, key), []).append(record)
    return {group: sorted(items, key=lambda r: (r.well_id, r.depth_m, r.sample_id)) for group, items in groups.items()}


def split_records() -> dict[str, list[Record]]:
    train = load_train_records()
    guard = sorted(load_guard_records(), key=lambda r: (r.depth_m, r.sample_id))
    if {r.family_id for r in guard} != {"15/9-F-12"}:
        raise RuntimeError("guard family mismatch")
    if {r.family_id for r in train} & {r.family_id for r in guard}:
        raise RuntimeError("train/guard family overlap")
    selection_dev = guard[::2]
    promotion_dev = guard[1::2]
    if not selection_dev or not promotion_dev:
        raise RuntimeError("guard split failed")
    if {r.sample_id for r in selection_dev} & {r.sample_id for r in promotion_dev}:
        raise RuntimeError("selection-dev and promotion-dev overlap")
    return {
        "train": sorted(train, key=lambda r: (r.well_id, r.depth_m, r.sample_id)),
        "selection_dev": selection_dev,
        "promotion_dev": promotion_dev,
    }


def stack(records: list[Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seismic = np.stack([record.seismic_patch for record in records]).astype(np.float64)
    logs = np.stack([record.well_log_seq for record in records]).astype(np.float64)
    labels = np.stack([record.label for record in records]).astype(np.float64)
    meta = [
        {
            "sample_id": record.sample_id,
            "family_id": record.family_id,
            "well_id": record.well_id,
            "depth_m": record.depth_m,
            "position": record.position,
        }
        for record in records
    ]
    return seismic, logs, labels, meta


def build_features(records: list[Record], stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    seismic, logs, labels, meta = stack(records)
    features, target_norm = apply_statistics(seismic, logs, labels, stats)
    if not np.isfinite(features).all() or not np.isfinite(target_norm).all():
        raise RuntimeError("feature construction produced non-finite values")
    return features, target_norm, labels, meta


def fit_stats(train_records: list[Record]) -> dict[str, Any]:
    seismic, logs, labels, _ = stack(train_records)
    return fit_train_statistics(seismic, logs, labels)


def load_a0_checkpoint(n_features: int):
    return p28.load_a0_checkpoint(n_features)


def infer(model: Any, features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    normalized = model.predict(features)
    return inverse_normalized_targets(normalized, stats)


def compare_primary_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    delta = candidate["composite_mean_train_std_normalized_RMSE"] - baseline["composite_mean_train_std_normalized_RMSE"]
    denom = baseline["composite_mean_train_std_normalized_RMSE"]
    rel = 0.0 if denom == 0 else delta / denom
    if rel <= -0.01:
        return "improved"
    if rel >= 0.01:
        return "worse"
    return "flat"


def route_semantics(route: RouteSpec) -> str:
    return f"{route.model_name}: {route.notes}"


def route_config_hash(route: RouteSpec, *, seed: int, budget_steps: int) -> str:
    return json_hash(
        {
            "route_id": route.route_id,
            "model_name": route.model_name,
            "model_kwargs": route.model_kwargs,
            "seed": seed,
            "budget_steps": budget_steps,
        }
    )


def prediction_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest()


def evaluate_predictions(actual_physical: np.ndarray, predicted_physical: np.ndarray, stats: dict[str, Any]) -> dict[str, Any]:
    metrics = p28.evaluate_predictions(actual_physical, predicted_physical, stats)
    return metrics


def make_batches(features: np.ndarray, targets: np.ndarray, batch_size: int, seed: int) -> callable:
    return p28.make_batches(features, targets, batch_size, seed)


def train_model(route: RouteSpec, features: np.ndarray, target_norm: np.ndarray, seed: int, budget_steps: int) -> Any:
    return p28.train_model(route, features, target_norm, seed=seed, budget_steps=budget_steps)


def compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    delta = candidate["composite_mean_train_std_normalized_RMSE"] - baseline["composite_mean_train_std_normalized_RMSE"]
    denom = baseline["composite_mean_train_std_normalized_RMSE"]
    rel = 0.0 if denom == 0 else delta / denom
    if rel <= -0.01:
        return "improved"
    if rel >= 0.01:
        return "worse"
    return "flat"


def small_normalized_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    denom = baseline["composite_mean_train_std_normalized_RMSE"]
    if denom == 0:
        return 0.0
    return round((candidate["composite_mean_train_std_normalized_RMSE"] - denom) / denom, 6)


def build_prompt(
    route_pilots: list[dict[str, Any]],
    selection_baseline: dict[str, Any],
    promotion_baseline: dict[str, Any],
    a0_hash: str,
) -> dict[str, Any]:
    routes = []
    for index, pilot in enumerate(route_pilots, start=1):
        routes.append(
            {
                "route_id": f"route_{index}",
                "semantics": pilot["semantics"],
                "lane": pilot["lane"],
                "blocked": pilot["blocked"],
                "feedback": pilot["feedback"],
                "selection_primary_delta_rel": small_normalized_delta(pilot["selection"], selection_baseline),
                "promotion_primary_delta_rel": None,
                "promotion_primary_delta_note": "unavailable_before_promotion",
            }
        )
    return {
        "task": "P29 action effect repair route selection",
        "rules": [
            "Use only route semantics, safe normalized deltas, and improved|flat|worse feedback.",
            "Do not use raw labels, test data, or frozen holdout signals.",
            "Return strict JSON only with action select|stop and route_id route_1..route_N or null.",
        ],
        "primary_metric": "composite_mean_train_std_normalized_RMSE",
        "a0_prediction_hash": a0_hash,
        "routes": routes,
        "schema": {
            "action": "select|stop",
            "route_id": "route_1|route_2|route_3|route_4|null",
            "reason": "string",
        },
    }


def get_deepseek_key() -> str:
    key = os.environ.get("DEEPSEEK_KEY")
    if key:
        return key.strip()
    script = Path.home() / ".claude" / "skills" / "share-docs" / "scripts" / "get-credential.sh"
    if not script.is_file():
        raise RuntimeError("DEEPSEEK credential helper missing")
    result = subprocess.run([str(script), "DEEPSEEK_API_KEY"], check=True, capture_output=True, text=True)
    key = result.stdout.strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY missing")
    return key


def call_deepseek(prompt: dict[str, Any]) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "Return strict JSON only. Choose one route_id or stop."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True)},
            ],
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {get_deepseek_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"DeepSeek HTTP error: {exc}") from exc
    content = raw["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek returned non-JSON content: {content!r}") from exc


def build_deepseek_decision(
    route_pilots: list[dict[str, Any]],
    selection_baseline: dict[str, Any],
    promotion_baseline: dict[str, Any],
    a0_hash: str,
    prompt_log: Path,
) -> dict[str, Any]:
    prompt = build_prompt(route_pilots, selection_baseline, promotion_baseline, a0_hash)
    prompt_log.write_text(json.dumps(prompt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    response = call_deepseek(prompt)
    if response.get("action") not in {"select", "stop"}:
        raise RuntimeError(f"DeepSeek returned invalid action: {response}")
    if response.get("action") == "select" and response.get("route_id") not in {f"route_{i}" for i in range(1, len(route_pilots) + 1)}:
        raise RuntimeError(f"DeepSeek returned invalid route_id: {response}")
    return response


def candidate_route_names() -> tuple[str, ...]:
    return ("A2L", "A2D", "A3")


def pilot_route(
    route: RouteSpec,
    seed: int,
    budget_steps: int,
    train_features: np.ndarray,
    train_target_norm: np.ndarray,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    stats: dict[str, Any],
    baseline_selection_metrics: dict[str, Any],
) -> dict[str, Any]:
    model = train_model(route, train_features, train_target_norm, seed=seed, budget_steps=budget_steps)
    selection_pred = infer(model, selection_features, stats)
    selection_metrics = evaluate_predictions(selection_targets, selection_pred, stats)
    feedback = compare_primary_to_baseline(selection_metrics, baseline_selection_metrics)
    return {
        "route_id": route.route_id,
        "model_name": route.model_name,
        "seed": seed,
        "budget_steps": budget_steps,
        "blocked": False,
        "selection": selection_metrics,
        "feedback": feedback,
        "prediction_hash": prediction_hash(selection_pred),
        "config_hash": route_config_hash(route, seed=seed, budget_steps=budget_steps),
        "semantics": route_semantics(route),
        "lane": route.lane,
    }


def identity_replay(
    *,
    train_features: np.ndarray,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    promotion_features: np.ndarray,
    promotion_targets: np.ndarray,
    stats: dict[str, Any],
    a0_selection_hash: str,
    a0_promotion_hash: str,
    a0_config_hash: str,
) -> dict[str, Any]:
    model = load_a0_checkpoint(train_features.shape[1])
    selection_pred = infer(model, selection_features, stats)
    promotion_pred = infer(model, promotion_features, stats)
    selection_metrics = evaluate_predictions(selection_targets, selection_pred, stats)
    promotion_metrics = evaluate_predictions(promotion_targets, promotion_pred, stats)
    selection_hash = prediction_hash(selection_pred)
    promotion_hash = prediction_hash(promotion_pred)
    return {
        "kind": "identity_replay",
        "strategy": "A1",
        "route_id": "A1_identity",
        "model_name": "tiny_mlp",
        "seed": ROOT_SEED,
        "budget_steps": 0,
        "status": "ok",
        "selection": selection_metrics,
        "promotion": promotion_metrics,
        "prediction_hash": {
            "selection": selection_hash,
            "promotion": promotion_hash,
        },
        "selection_prediction_hash": selection_hash,
        "promotion_prediction_hash": promotion_hash,
        "selection_hash_matches_a0": selection_hash == a0_selection_hash,
        "promotion_hash_matches_a0": promotion_hash == a0_promotion_hash,
        "config_hash": a0_config_hash,
        "executor": "a0_frozen_tiny_mlp",
        "action": "identity_replay",
        "visible_to_llm": False,
    }


def summarize(records: dict[str, Any]) -> dict[str, Any]:
    return {
        "a0": records["a0"],
        "a1": records["a1"],
        "split": records["split"],
        "routes": records["routes"],
        "gate": records["gate"],
        "pilots": records["pilots"],
        "strategies": records["strategies"],
        "promotion_gate": records["promotion_gate"],
        "oracle_ceiling": records["oracle_ceiling"],
        "commands": records["commands"],
    }


def markdown_root_cause(summary: dict[str, Any]) -> str:
    lines = ["# P29 root cause", ""]
    lines.append("| stage | connected | evidence |")
    lines.append("|---|---|---|")
    lines.append(f"| prompt | yes | route semantics + safe normalized deltas only |")
    lines.append(f"| action | yes | A2L/A2D/A3 route selected from pilot evidence; A1 is identity no-op |")
    lines.append(f"| executor | yes | shared NumPy/MLP training loop on train-only statistics |")
    lines.append(f"| prediction | yes | every trial records prediction hash |")
    lines.append(f"| metric | yes | documented composite primary metric used for selection/promotion |")
    lines.append(f"| promotion | yes | candidate-only gate excludes A0/A1 |")
    lines.append(f"| endpoint | yes | outputs written to `_outputs/p29_agent_action_effect/` |")
    lines.append("")
    lines.append(f"- A0 selection hash: `{summary['a0']['selection_prediction_hash']}`")
    lines.append(f"- A1 replay matches A0: `{summary['a1']['selection_hash_matches_a0']}`")
    lines.append(f"- oracle ceiling route: `{summary['oracle_ceiling']['route_id']}`")
    lines.append(f"- promotion gate passed: `{summary['promotion_gate']['passed']}`")
    return "\n".join(lines) + "\n"


def build_action_effects(summary: dict[str, Any]) -> dict[str, Any]:
    rows = []
    rows.append(
        {
            "kind": "A0",
            "route_id": "A0",
            "config_hash": summary["a0"]["config_hash"],
            "prediction_hash": summary["a0"]["selection_prediction_hash"],
            "primary_delta_rel": 0.0,
            "visible_to_llm": False,
        }
    )
    rows.append(
        {
            "kind": "A1",
            "route_id": "A1_identity",
            "config_hash": summary["a1"]["config_hash"],
            "prediction_hash": summary["a1"]["selection_prediction_hash"],
            "primary_delta_rel": 0.0,
            "visible_to_llm": False,
            "no_op": True,
        }
    )
    for strategy in candidate_route_names():
        result = summary["strategies"][strategy]
        rows.append(
            {
                "kind": strategy,
                "route_id": result["chosen_route_id"],
                "config_hash": result.get("config_hash"),
                "prediction_hash": result.get("selection_prediction_hash"),
                "primary_delta_rel": result.get("selection_primary_delta_rel"),
                "promotion_primary_delta_rel": result.get("promotion_primary_delta_rel"),
                "visible_to_llm": result.get("visible_to_llm", True),
                "semantics": result.get("semantics"),
                "lane": result.get("lane"),
                "status": result.get("status"),
            }
        )
    rows.append(
        {
            "kind": "oracle_ceiling",
            "route_id": summary["oracle_ceiling"]["route_id"],
            "config_hash": summary["oracle_ceiling"]["config_hash"],
            "prediction_hash": summary["oracle_ceiling"]["prediction_hash"],
            "primary_delta_rel": summary["oracle_ceiling"]["selection_primary_delta_rel"],
            "promotion_primary_delta_rel": summary["oracle_ceiling"]["promotion_primary_delta_rel"],
            "visible_to_llm": False,
        }
    )
    return {
        "schema_version": "p29_agent_action_effect/v1",
        "rows": rows,
        "gate_excludes": ["A0", "A1"],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def execute(budget_steps: int = DEFAULT_BUDGET_STEPS, pilot_steps: int = PILOT_STEPS) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_records()
    stats = fit_stats(split["train"])
    train_features, train_target_norm, _, _ = build_features(split["train"], stats)
    selection_features, _, selection_targets, selection_meta = build_features(split["selection_dev"], stats)
    promotion_features, _, promotion_targets, promotion_meta = build_features(split["promotion_dev"], stats)

    a0_model = load_a0_checkpoint(train_features.shape[1])
    a0_selection_pred = infer(a0_model, selection_features, stats)
    a0_promotion_pred = infer(a0_model, promotion_features, stats)
    a0_selection_metrics = evaluate_predictions(selection_targets, a0_selection_pred, stats)
    a0_promotion_metrics = evaluate_predictions(promotion_targets, a0_promotion_pred, stats)
    a0_selection_hash = prediction_hash(a0_selection_pred)
    a0_promotion_hash = prediction_hash(a0_promotion_pred)
    a0 = {
        "model": "tiny_mlp",
        "checkpoint_path": str(BASELINE_CHECKPOINT.relative_to(RESERVOIR_DIR)),
        "checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
        "run_manifest_path": str(BASELINE_RUN_MANIFEST.relative_to(RESERVOIR_DIR)),
        "run_manifest_sha256": sha256_file(BASELINE_RUN_MANIFEST),
        "selection_prediction_hash": a0_selection_hash,
        "promotion_prediction_hash": a0_promotion_hash,
        "selection_composite_mean_train_std_normalized_RMSE": a0_selection_metrics["composite_mean_train_std_normalized_RMSE"],
        "promotion_composite_mean_train_std_normalized_RMSE": a0_promotion_metrics["composite_mean_train_std_normalized_RMSE"],
        "config_hash": json_hash({"checkpoint": sha256_file(BASELINE_CHECKPOINT), "seed": ROOT_SEED, "kind": "A0"}),
    }
    a0_config_hash = a0["config_hash"]

    a1 = identity_replay(
        train_features=train_features,
        selection_features=selection_features,
        selection_targets=selection_targets,
        promotion_features=promotion_features,
        promotion_targets=promotion_targets,
        stats=stats,
        a0_selection_hash=a0_selection_hash,
        a0_promotion_hash=a0_promotion_hash,
        a0_config_hash=a0_config_hash,
    )

    route_pilots: list[dict[str, Any]] = []
    pilot_rows: list[dict[str, Any]] = [a1]
    commands = [
        "python3 -m py_compile _pipelines/02_task_datasets/reservoir/p29_agent_action_effect.py _pipelines/02_task_datasets/reservoir/tests/test_p29_agent_action_effect.py",
        "pytest -q _pipelines/02_task_datasets/reservoir/tests/test_p29_agent_action_effect.py",
        "python3 _pipelines/02_task_datasets/reservoir/p29_agent_action_effect.py --budget-steps 8 --pilot-steps 8",
    ]

    for route in ROUTES:
        pilot = pilot_route(
            route=route,
            seed=ROOT_SEED,
            budget_steps=pilot_steps,
            train_features=train_features,
            train_target_norm=train_target_norm,
            selection_features=selection_features,
            selection_targets=selection_targets,
            stats=stats,
            baseline_selection_metrics=a0_selection_metrics,
        )
        route_pilots.append(pilot)
        pilot_rows.append(
            {
                "kind": "pilot",
                "strategy": "pilot",
                "route_id": route.route_id,
                "model_name": route.model_name,
                "seed": ROOT_SEED,
                "budget_steps": pilot_steps,
                "selection": pilot["selection"],
                "feedback": pilot["feedback"],
                "prediction_hash": pilot["prediction_hash"],
                "config_hash": pilot["config_hash"],
                "semantics": pilot["semantics"],
                "lane": pilot["lane"],
                "blocked": False,
            }
        )

    deepseek_log = OUTPUT_DIR / "deepseek_prompt.json"
    try:
        deepseek_decision = build_deepseek_decision(route_pilots, a0_selection_metrics, a0_promotion_metrics, a0_selection_hash, deepseek_log)
    except Exception as exc:
        deepseek_decision = {"action": "stop", "route_id": None, "reason": f"provider_missing_or_invalid: {exc}"}

    candidate_names = candidate_route_names()
    if deepseek_decision.get("action") == "select":
        route_idx = int(str(deepseek_decision["route_id"]).split("_")[-1]) - 1
        a2l_route = ROUTES[route_idx]
        a2l_selected_by = "deepseek"
    else:
        a2l_route = None
        a2l_selected_by = "blocked"

    a2d_route = min(
        (pilot for pilot in route_pilots if not pilot["blocked"]),
        key=lambda pilot: (pilot["selection"]["composite_mean_train_std_normalized_RMSE"], pilot["route_id"]),
    )
    rng = Generator(PCG64(ROOT_SEED))
    eligible = [pilot for pilot in route_pilots if not pilot["blocked"]]
    a3_route = eligible[int(rng.permutation(len(eligible))[0])]

    strategy_choices = {
        "A1": {"selected_by": "identity_replay", "chosen_route_id": "A1_identity"},
        "A2L": {"selected_by": a2l_selected_by, "chosen_route_id": a2l_route.route_id if a2l_route else None, "deepseek": deepseek_decision},
        "A2D": {"selected_by": "deterministic_primary_metric_rank", "chosen_route_id": a2d_route["route_id"]},
        "A3": {"selected_by": "pcg64_no_replacement", "chosen_route_id": a3_route["route_id"]},
    }

    strategy_results: dict[str, dict[str, Any]] = {}
    for strategy, route in {"A2L": a2l_route, "A2D": next(r for r in ROUTES if r.route_id == a2d_route["route_id"]), "A3": next(r for r in ROUTES if r.route_id == a3_route["route_id"]) }.items():
        if route is None:
            strategy_results[strategy] = {
                "chosen_route_id": None,
                "selected_by": strategy_choices[strategy]["selected_by"],
                "status": "blocked",
                "reason": "DeepSeek stop action",
                "selection_trials": [],
                "promotion_trials": [],
                "promotion_gate": False,
                "selection_hash_matches_a0": False,
                "promotion_hash_matches_a0": False,
                "config_hash": None,
                "selection_prediction_hash": None,
                "promotion_prediction_hash": None,
                "selection_primary_delta_rel": None,
                "promotion_primary_delta_rel": None,
                "semantics": None,
                "lane": None,
                "visible_to_llm": True,
            }
            continue
        selection_trials = []
        promotion_trials = []
        trial_rows = []
        for seed in SAME_BUDGET_SEEDS:
            model = train_model(route, train_features, train_target_norm, seed=int(seed), budget_steps=budget_steps)
            selection_pred = infer(model, selection_features, stats)
            promotion_pred = infer(model, promotion_features, stats)
            selection_metrics = evaluate_predictions(selection_targets, selection_pred, stats)
            promotion_metrics = evaluate_predictions(promotion_targets, promotion_pred, stats)
            checkpoint_path = CHECKPOINT_DIR / strategy / route.route_id / f"seed_{seed}.npz"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(checkpoint_path)
            trial = {
                "kind": "trial",
                "strategy": strategy,
                "route_id": route.route_id,
                "model_name": route.model_name,
                "seed": seed,
                "budget_steps": budget_steps,
                "status": "ok",
                "selection": selection_metrics,
                "promotion": promotion_metrics,
                "prediction_hash": {
                    "selection": prediction_hash(selection_pred),
                    "promotion": prediction_hash(promotion_pred),
                },
                "checkpoint_path": str(checkpoint_path.relative_to(OUTPUT_DIR)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "config_hash": route_config_hash(route, seed=seed, budget_steps=budget_steps),
                "semantics": route_semantics(route),
                "lane": route.lane,
            }
            trial_rows.append(trial)
            selection_trials.append(selection_metrics)
            promotion_trials.append(promotion_metrics)
            pilot_rows.append({"kind": "trial", **trial})

        selection_median = {
            "composite_mean_train_std_normalized_RMSE": float(np.median([trial["composite_mean_train_std_normalized_RMSE"] for trial in selection_trials])),
            "physical_MAE_macro": float(np.median([trial["physical_MAE_macro"] for trial in selection_trials])),
            "worst_group_RMSE": {target: float(np.median([trial["worst_group_RMSE"][target] for trial in selection_trials])) for target in PHYSICAL_TARGETS},
        }
        promotion_median = {
            "composite_mean_train_std_normalized_RMSE": float(np.median([trial["composite_mean_train_std_normalized_RMSE"] for trial in promotion_trials])),
            "physical_MAE_macro": float(np.median([trial["physical_MAE_macro"] for trial in promotion_trials])),
            "worst_group_RMSE": {target: float(np.median([trial["worst_group_RMSE"][target] for trial in promotion_trials])) for target in PHYSICAL_TARGETS},
        }
        route_p = next(r for r in ROUTES if r.route_id == route.route_id)
        selection_primary_delta_rel = compare_primary_to_baseline(selection_median, a0_selection_metrics)
        promotion_primary_delta_rel = compare_primary_to_baseline(promotion_median, a0_promotion_metrics)
        strategy_results[strategy] = {
            "chosen_route_id": route.route_id,
            "selected_by": strategy_choices[strategy]["selected_by"],
            "selection_trials": selection_trials,
            "promotion_trials": promotion_trials,
            "selection_median": selection_median,
            "promotion_median": promotion_median,
            "route_model_name": route.model_name,
            "status": "ok",
            "promotion_gate": promotion_primary_delta_rel == "improved"
            and all(promotion_median["worst_group_RMSE"][target] <= a0_promotion_metrics["worst_group_RMSE"][target] * 1.02 for target in PHYSICAL_TARGETS),
            "config_hash": route_config_hash(route_p, seed=ROOT_SEED, budget_steps=budget_steps),
            "selection_prediction_hash": trial_rows[0]["prediction_hash"]["selection"],
            "promotion_prediction_hash": trial_rows[0]["prediction_hash"]["promotion"],
            "selection_primary_delta_rel": small_normalized_delta(selection_median, a0_selection_metrics),
            "promotion_primary_delta_rel": small_normalized_delta(promotion_median, a0_promotion_metrics),
            "semantics": route_semantics(route),
            "lane": route.lane,
        }

    strategy_results["A1"] = {
        "chosen_route_id": "A1_identity",
        "selected_by": "identity_replay",
        "selection_trials": [a1["selection"]],
        "promotion_trials": [a1["promotion"]],
        "selection_median": {
            "composite_mean_train_std_normalized_RMSE": float(a1["selection"]["composite_mean_train_std_normalized_RMSE"]),
            "physical_MAE_macro": float(a1["selection"]["physical_MAE_macro"]),
            "worst_group_RMSE": dict(a1["selection"]["worst_group_RMSE"]),
        },
        "promotion_median": {
            "composite_mean_train_std_normalized_RMSE": float(a1["promotion"]["composite_mean_train_std_normalized_RMSE"]),
            "physical_MAE_macro": float(a1["promotion"]["physical_MAE_macro"]),
            "worst_group_RMSE": dict(a1["promotion"]["worst_group_RMSE"]),
        },
        "route_model_name": "tiny_mlp",
        "status": "ok",
        "promotion_gate": False,
        "selection_hash_matches_a0": a1["selection_hash_matches_a0"],
        "promotion_hash_matches_a0": a1["promotion_hash_matches_a0"],
        "config_hash": a1["config_hash"],
        "selection_prediction_hash": a1["prediction_hash"]["selection"],
        "promotion_prediction_hash": a1["prediction_hash"]["promotion"],
        "selection_primary_delta_rel": 0.0,
        "promotion_primary_delta_rel": 0.0,
        "executor": a1["executor"],
        "action": a1["action"],
        "visible_to_llm": False,
    }

    candidate_gate_names = candidate_route_names()
    promotion_gate_passed = all(
        strategy_results[name]["promotion_median"]["composite_mean_train_std_normalized_RMSE"] <= a0_promotion_metrics["composite_mean_train_std_normalized_RMSE"] * 0.99
        and all(strategy_results[name]["promotion_median"]["worst_group_RMSE"][target] <= a0_promotion_metrics["worst_group_RMSE"][target] * 1.02 for target in PHYSICAL_TARGETS)
        for name in candidate_gate_names
        if strategy_results[name]["status"] == "ok"
    )
    candidate_ok = [name for name in candidate_gate_names if strategy_results[name]["status"] == "ok"]
    oracle_name = min(candidate_ok, key=lambda name: strategy_results[name]["promotion_median"]["composite_mean_train_std_normalized_RMSE"], default=None)
    oracle_ceiling = {
        "route_id": strategy_results[oracle_name]["chosen_route_id"] if oracle_name else None,
        "strategy": oracle_name,
        "config_hash": strategy_results[oracle_name]["config_hash"] if oracle_name else None,
        "prediction_hash": strategy_results[oracle_name]["promotion_prediction_hash"] if oracle_name else None,
        "selection_primary_delta_rel": strategy_results[oracle_name]["selection_primary_delta_rel"] if oracle_name else None,
        "promotion_primary_delta_rel": strategy_results[oracle_name]["promotion_primary_delta_rel"] if oracle_name else None,
    }
    keep_llm = strategy_results["A2L"]["status"] == "ok" and oracle_name is not None and strategy_results["A2L"]["selection_median"]["composite_mean_train_std_normalized_RMSE"] <= min(strategy_results[name]["selection_median"]["composite_mean_train_std_normalized_RMSE"] for name in candidate_ok)

    primary_gate_status = "aligned"
    primary_gate_note = "documented composite primary metric used for selection and promotion"
    prompt = build_prompt(route_pilots, a0_selection_metrics, a0_promotion_metrics, a0_selection_hash)
    root_cause = {
        "prompt": {
            "connected": True,
            "evidence": "route semantics and safe normalized deltas only",
        },
        "action": {
            "connected": True,
            "evidence": "A1 no-op; A2L/A2D/A3 candidate actions trained/evaluated on guard split",
        },
        "executor": {
            "connected": True,
            "evidence": "equal budget training on train-only statistics",
        },
        "prediction": {
            "connected": True,
            "evidence": "prediction hashes recorded per trial",
        },
        "metric": {
            "connected": True,
            "evidence": "composite primary metric used in ranking and gate",
        },
        "promotion": {
            "connected": True,
            "evidence": "A0 and A1 excluded from candidate gate",
        },
        "endpoint": {
            "connected": True,
            "evidence": "development-only outputs written under p29_agent_action_effect",
        },
        "verdict": "RETAIN_HYBRID" if promotion_gate_passed else "REJECT_AGENT",
        "note": primary_gate_note,
    }
    action_effects = build_action_effects({
        "a0": a0,
        "a1": a1,
        "strategies": strategy_results,
        "oracle_ceiling": oracle_ceiling,
    })

    results_rows = [{"kind": "A0", "strategy": "A0", "status": "reference"}]
    results_rows.append({"kind": "A1", "strategy": "A1", "status": "no_op", "selection": a1["selection"], "promotion": a1["promotion"]})
    for name in candidate_gate_names:
        result = strategy_results[name]
        for trial in result["selection_trials"]:
            results_rows.append({"kind": "trial", "strategy": name, "phase": "selection", "metrics": trial})
        for trial in result["promotion_trials"]:
            results_rows.append({"kind": "trial", "strategy": name, "phase": "promotion", "metrics": trial})
    results_rows.append({"kind": "oracle_ceiling", **oracle_ceiling})

    summary = summarize(
        {
            "a0": a0,
            "a1": a1,
            "split": {
                "train": {
                    "count": len(split["train"]),
                    "families": sorted({r.family_id for r in split["train"]}),
                    "wells": sorted({r.well_id for r in split["train"]}),
                },
                "selection_dev": {
                    "count": len(split["selection_dev"]),
                    "families": sorted({r.family_id for r in split["selection_dev"]}),
                    "wells": sorted({r.well_id for r in split["selection_dev"]}),
                },
                "promotion_dev": {
                    "count": len(split["promotion_dev"]),
                    "families": sorted({r.family_id for r in split["promotion_dev"]}),
                    "wells": sorted({r.well_id for r in split["promotion_dev"]}),
                },
            },
            "routes": [dataclasses.asdict(route) for route in ROUTES],
            "gate": {
                "status": "blocked",
                "reason": "CIG-Bench PropertyPredictor 404 and input contract mismatch",
                "evidence_path": str(P18_EVIDENCE.relative_to(RESERVOIR_DIR)),
            },
            "pilots": route_pilots,
            "strategies": strategy_results,
            "promotion_gate": {
                "passed": promotion_gate_passed,
                "primary_metric": "composite_mean_train_std_normalized_RMSE",
                "candidate_strategies": list(candidate_gate_names),
                "keep_llm": keep_llm,
                "best_deterministic": oracle_name,
            },
            "oracle_ceiling": oracle_ceiling,
            "commands": commands,
        }
    )

    protocol = {
        "schema_version": "p29_agent_action_effect/v1",
        "track_id": "property",
        "root_seed": ROOT_SEED,
        "primary_metric": "composite_mean_train_std_normalized_RMSE",
        "a0": a0,
        "a1": {
            "kind": "identity_replay",
            "executor": a1["executor"],
            "selection_hash_matches_a0": a1["selection_hash_matches_a0"],
            "promotion_hash_matches_a0": a1["promotion_hash_matches_a0"],
            "config_hash": a1["config_hash"],
        },
        "split": summary["split"],
        "route_catalog": [dataclasses.asdict(route) for route in ROUTES],
        "route_pilots": route_pilots,
        "strategy_choice": strategy_choices,
        "candidate_gate_names": list(candidate_gate_names),
        "promotion_gate": {
            "primary_metric": "composite_mean_train_std_normalized_RMSE",
            "candidate_strategies": list(candidate_gate_names),
            "a0_a1_excluded": True,
            "worst_group_rmse_max_worsening": 0.02,
            "primary_threshold_rel_improvement": 0.01,
        },
        "oracle_ceiling": oracle_ceiling,
        "train_h5_sha256": sha256_file(TRAIN_H5),
        "guard_npz_sha256": sha256_file(GUARD_NPZ),
        "baseline_run_manifest_sha256": sha256_file(BASELINE_RUN_MANIFEST),
        "baseline_metrics_sha256": sha256_file(BASELINE_METRICS),
    }

    write_json(OUTPUT_DIR / "root_cause.md", root_cause)
    # overwrite with actual markdown after JSON sanity? keep human-readable below
    (OUTPUT_DIR / "root_cause.md").write_text(markdown_root_cause(summary), encoding="utf-8")
    write_json(OUTPUT_DIR / "action_effects.json", action_effects)
    write_json(OUTPUT_DIR / "protocol.json", protocol)
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_jsonl(OUTPUT_DIR / "results.jsonl", results_rows)
    write_json(OUTPUT_DIR / "deepseek_prompt.json", prompt)
    manifest = {
        "schema_version": "p29_agent_action_effect/manifest/v1",
        "inputs": {
            "train_h5": {"path": str(TRAIN_H5.relative_to(PROJECT_ROOT)), "sha256": sha256_file(TRAIN_H5)},
            "guard_npz": {"path": str(GUARD_NPZ.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(GUARD_NPZ)},
            "baseline_metrics": {"path": str(BASELINE_METRICS.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(BASELINE_METRICS)},
            "baseline_run_manifest": {"path": str(BASELINE_RUN_MANIFEST.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(BASELINE_RUN_MANIFEST)},
            "baseline_checkpoint": {"path": str(BASELINE_CHECKPOINT.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(BASELINE_CHECKPOINT)},
            "p18_evidence": {"path": str(P18_EVIDENCE.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(P18_EVIDENCE)},
        },
        "outputs": {
            "root_cause_md": {"path": "root_cause.md", "sha256": sha256_file(OUTPUT_DIR / "root_cause.md")},
            "action_effects_json": {"path": "action_effects.json", "sha256": sha256_file(OUTPUT_DIR / "action_effects.json")},
            "protocol_json": {"path": "protocol.json", "sha256": sha256_file(OUTPUT_DIR / "protocol.json")},
            "summary_json": {"path": "summary.json", "sha256": sha256_file(OUTPUT_DIR / "summary.json")},
            "results_jsonl": {"path": "results.jsonl", "sha256": sha256_file(OUTPUT_DIR / "results.jsonl")},
            "deepseek_prompt_json": {"path": "deepseek_prompt.json", "sha256": sha256_file(OUTPUT_DIR / "deepseek_prompt.json")},
        },
        "split": summary["split"],
        "gate": summary["gate"],
        "route_catalog": summary["routes"],
        "command": commands,
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    return {
        "summary": summary,
        "protocol": protocol,
        "action_effects": action_effects,
        "prompt": prompt,
        "deepseek_decision": deepseek_decision,
        "root_cause": root_cause,
        "oracle_ceiling": oracle_ceiling,
        "promotion_gate_passed": promotion_gate_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P29 agent action-effect repair.")
    parser.add_argument("--budget-steps", type=int, default=DEFAULT_BUDGET_STEPS)
    parser.add_argument("--pilot-steps", type=int, default=PILOT_STEPS)
    args = parser.parse_args()
    if args.budget_steps <= 0 or args.pilot_steps <= 0:
        raise SystemExit("budget-steps and pilot-steps must be positive")
    result = execute(budget_steps=args.budget_steps, pilot_steps=args.pilot_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
