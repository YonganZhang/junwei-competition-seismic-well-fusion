from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from numpy.random import Generator, PCG64


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RESERVOIR_DIR = HERE
OUTPUT_DIR = RESERVOIR_DIR / "_outputs" / "p28_agentic_optimization"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
TRAIN_H5 = PROJECT_ROOT / "_data" / "processed" / "reservoir" / "train.h5"
BASELINE_RUN_MANIFEST = RESERVOIR_DIR / "_outputs" / "run_manifest.json"
BASELINE_METRICS = RESERVOIR_DIR / "_outputs" / "metrics.json"
BASELINE_CHECKPOINT = RESERVOIR_DIR / "_outputs" / "checkpoints" / "best.ckpt"
BASELINE_HISTORY = RESERVOIR_DIR / "_outputs" / "checkpoints" / "history.json"
P18_EVIDENCE = RESERVOIR_DIR / "_outputs" / "p18_cigbench_property" / "evidence.md"

sys.path.insert(0, str(RESERVOIR_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from train_baseline import (  # noqa: E402
    apply_statistics,
    compute_metrics,
    fit_train_statistics,
    inverse_normalized_targets,
    make_batches_factory,
)
from data_pipeline import inverse_targets, TARGET_NAMES  # noqa: E402
from ml_framework.model_registry import get_model  # noqa: E402


TARGETS = TARGET_NAMES
PHYSICAL_TARGETS = ("PHIF", "KLOGH", "SW")
ROOT_SEED = 2693
PILOT_STEPS = 8
DEFAULT_BUDGET_STEPS = 32
BATCH_SIZE = 64
SAME_BUDGET_SEEDS = (2693, 2694, 2695)


@dataclass(frozen=True)
class Record:
    sample_id: str
    family_id: str
    well_id: str
    depth_m: float
    seismic_patch: np.ndarray
    well_log_seq: np.ndarray
    label: np.ndarray
    position: dict[str, Any]


@dataclass(frozen=True)
class RouteSpec:
    route_id: str
    model_name: str
    model_kwargs: dict[str, Any]
    lane: str
    dependency_group: str
    notes: str


ROUTES = (
    RouteSpec(
        route_id="tiny_mlp_default",
        model_name="tiny_mlp",
        model_kwargs={"hidden_dim": 24, "learning_rate": 0.002, "weight_decay": 1e-5},
        lane="tabular-cpu",
        dependency_group="tabular-cpu",
        notes="A0-compatible tiny MLP with frozen default regularization",
    ),
    RouteSpec(
        route_id="tiny_mlp_l2",
        model_name="tiny_mlp",
        model_kwargs={"hidden_dim": 24, "learning_rate": 0.002, "weight_decay": 1e-3},
        lane="tabular-cpu",
        dependency_group="tabular-cpu",
        notes="Same architecture with stronger L2",
    ),
    RouteSpec(
        route_id="reservoir_ridge",
        model_name="reservoir_ridge",
        model_kwargs={"learning_rate": 0.002, "l2_strength": 1e-3},
        lane="tabular-cpu",
        dependency_group="tabular-cpu",
        notes="Closed-form family not used; SGD ridge baseline",
    ),
    RouteSpec(
        route_id="reservoir_linear",
        model_name="reservoir_linear",
        model_kwargs={"learning_rate": 0.002, "l2_strength": 0.0},
        lane="tabular-cpu",
        dependency_group="tabular-cpu",
        notes="Plain multi-output linear SGD baseline",
    ),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def json_sha256(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records() -> list[Record]:
    if not TRAIN_H5.is_file():
        raise FileNotFoundError(TRAIN_H5)
    records: list[Record] = []
    with h5py.File(TRAIN_H5, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            meta = json.loads(group.attrs["meta"])
            position = json.loads(group.attrs["position"])
            records.append(
                Record(
                    sample_id=key,
                    family_id=str(meta["family_id"]),
                    well_id=str(meta["well_id"]),
                    depth_m=float(meta["depth_m"]),
                    seismic_patch=np.asarray(group["seismic_patch"][()], dtype=np.float64),
                    well_log_seq=np.asarray(group["well_log_seq"][()], dtype=np.float64),
                    label=np.asarray(group["label"][()], dtype=np.float64).reshape(-1),
                    position=position,
                )
            )
    if not records:
        raise RuntimeError("train.h5 is empty")
    return records


def group_records(records: Iterable[Record], key: str) -> dict[str, list[Record]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        groups[getattr(record, key)].append(record)
    return {group: sorted(items, key=lambda r: (r.well_id, r.depth_m, r.sample_id)) for group, items in groups.items()}


def split_records(records: list[Record]) -> dict[str, list[Record]]:
    families = group_records(records, "family_id")
    family_counts = {family: len(items) for family, items in families.items()}
    train_families = tuple(sorted(family_counts, key=lambda family: (-family_counts[family], family))[:2])
    dev_family = next(family for family in sorted(family_counts, key=lambda family: (family_counts[family], family)))
    train = [record for record in records if record.family_id in train_families]
    dev_pool = [record for record in records if record.family_id == dev_family]
    dev_groups = group_records(dev_pool, "well_id")
    # Nested / disjoint dev: one selection well, two promotion wells, all from the same held-out family.
    selection_well = "15/9-19 A"
    promotion_wells = {"15/9-19 BT2", "15/9-19 SR"}
    selection_dev = [record for record in dev_pool if record.well_id == selection_well]
    promotion_dev = [record for record in dev_pool if record.well_id in promotion_wells]
    if not train or not selection_dev or not promotion_dev:
        raise RuntimeError("nested development split construction failed")
    if {r.well_id for r in selection_dev} & {r.well_id for r in promotion_dev}:
        raise RuntimeError("selection-dev and promotion-dev overlap")
    if {r.family_id for r in train} & {r.family_id for r in selection_dev + promotion_dev}:
        raise RuntimeError("train/dev family overlap")
    return {
        "train": train,
        "selection_dev": sorted(selection_dev, key=lambda r: (r.well_id, r.depth_m, r.sample_id)),
        "promotion_dev": sorted(promotion_dev, key=lambda r: (r.well_id, r.depth_m, r.sample_id)),
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


def fit_baseline_stats(train_records: list[Record]) -> dict[str, Any]:
    seismic, logs, labels, _ = stack(train_records)
    return fit_train_statistics(seismic, logs, labels)


def load_a0_checkpoint(n_features: int):
    model = get_model(
        "tiny_mlp",
        models_package="models",
        n_features=n_features,
        n_outputs=3,
        hidden_dim=24,
        learning_rate=0.002,
        weight_decay=1e-5,
        seed=ROOT_SEED,
    )
    model.load_checkpoint(BASELINE_CHECKPOINT)
    return model


def prediction_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest()


def evaluate_predictions(actual_physical: np.ndarray, predicted_physical: np.ndarray, stats: dict[str, Any]) -> dict[str, Any]:
    metrics = compute_metrics(actual_physical, predicted_physical, stats)
    per_target = metrics["per_target"]
    actual_physical_values = inverse_targets(actual_physical)
    predicted_physical_values = inverse_targets(predicted_physical)
    physical_mae_macro = float(
        np.mean(
            [
                float(np.mean(np.abs(predicted_physical_values[:, i] - actual_physical_values[:, i])))
                for i in range(len(PHYSICAL_TARGETS))
            ]
        )
    )
    worst_group_rmse = {
        target: float(np.sqrt(np.mean((predicted_physical_values[:, i] - actual_physical_values[:, i]) ** 2)))
        for i, target in enumerate(PHYSICAL_TARGETS)
    }
    metrics["physical_MAE_macro"] = physical_mae_macro
    metrics["worst_group_RMSE"] = worst_group_rmse
    return metrics


def make_batches(features: np.ndarray, targets: np.ndarray, batch_size: int, seed: int) -> callable:
    call_count = 0

    def factory() -> list[tuple[np.ndarray, np.ndarray]]:
        nonlocal call_count
        rng = np.random.default_rng(seed + call_count)
        indices = np.arange(len(features))
        rng.shuffle(indices)
        call_count += 1
        batches = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batches.append((features[batch_indices], targets[batch_indices]))
        return batches

    return factory


def train_model(route: RouteSpec, features: np.ndarray, target_norm: np.ndarray, seed: int, budget_steps: int) -> Any:
    model = get_model(
        route.model_name,
        models_package="models",
        n_features=features.shape[1],
        n_outputs=3,
        seed=seed,
        **route.model_kwargs,
    )
    batches_fn = make_batches(features, target_norm, BATCH_SIZE, seed)
    step = 0
    while step < budget_steps:
        batches = batches_fn()
        if not batches:
            raise RuntimeError("empty batch factory")
        for batch in batches:
            model.train_batch(batch)
            step += 1
            if step >= budget_steps:
                break
    return model


def infer(model: Any, features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    normalized = model.predict(features)
    return inverse_normalized_targets(normalized, stats)


def compare_to_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> str:
    delta = candidate["physical_MAE_macro"] - baseline["physical_MAE_macro"]
    denom = baseline["physical_MAE_macro"]
    rel = 0.0 if denom == 0 else delta / denom
    if rel <= -0.01:
        return "improved"
    if rel >= 0.01:
        return "worse"
    return "flat"


def route_catalog() -> list[dict[str, Any]]:
    return [dataclasses.asdict(route) for route in ROUTES]


def gather_cig_gate() -> dict[str, Any]:
    return {
        "route_id": "cig_bench_propertypredictor",
        "status": "blocked",
        "reason": "ModelScope default checkpoint 404 and input contract mismatch",
        "evidence_path": str(P18_EVIDENCE.relative_to(RESERVOIR_DIR)),
        "blocked_checkpoints": ["douyimin/CIG-Bench/CIG-Bench-Property.pth"],
        "input_contract": "PropertyPredictor expects seismic volume + sparse property volume, while the reservoir development data are sample-level seismic_patch + well_log_seq features only.",
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


def strict_anonymous_prompt(route_feedback: list[dict[str, Any]], route_pilots: list[dict[str, Any]], a0_hash: str) -> dict[str, Any]:
    anonymized_routes = []
    for index, route in enumerate(route_catalog(), start=1):
        anonymized_routes.append(
            {
                "route_id": f"route_{index}",
                "lane": route["lane"],
                "dependency_group": route["dependency_group"],
                "pilot_feedback": route_feedback[index - 1]["feedback"],
                "blocked": route_feedback[index - 1]["blocked"],
            }
        )
    return {
        "task": "A2L route selection or stop",
        "rules": [
            "Only use anonymous route metadata and improved|flat|worse feedback.",
            "Return strict JSON only.",
            "Choose exactly one route_id or stop.",
        ],
        "a0_prediction_hash": a0_hash,
        "routes": anonymized_routes,
        "pilot_summaries": [
            {
                "route_id": f"route_{index}",
                "feedback": route_pilots[index - 1]["feedback"],
            }
            for index in range(1, len(route_pilots) + 1)
        ],
        "schema": {
            "action": "select|stop",
            "route_id": "route_1|route_2|route_3|route_4|null",
            "reason": "string",
        },
    }


def choose_a2d_route(route_pilots: list[dict[str, Any]]) -> str:
    ranked = sorted(
        (
            (pilot["selection"]["physical_MAE_macro"], pilot["route_id"])
            for pilot in route_pilots
            if not pilot["blocked"]
        ),
        key=lambda item: (item[0], item[1]),
    )
    if not ranked:
        raise RuntimeError("no runnable routes for A2D")
    return ranked[0][1]


def choose_a3_route(route_pilots: list[dict[str, Any]], seed: int = ROOT_SEED) -> str:
    eligible = [pilot["route_id"] for pilot in route_pilots if not pilot["blocked"]]
    if not eligible:
        raise RuntimeError("no runnable routes for A3")
    rng = Generator(PCG64(seed))
    order = rng.permutation(len(eligible))
    return eligible[int(order[0])]


def path_for_trial(strategy: str, route_id: str, seed: int) -> Path:
    return CHECKPOINT_DIR / strategy / route_id / f"seed_{seed}.npz"


def run_trial(
    *,
    strategy: str,
    route: RouteSpec,
    seed: int,
    budget_steps: int,
    train_features: np.ndarray,
    train_target_norm: np.ndarray,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    promotion_features: np.ndarray,
    promotion_targets: np.ndarray,
    stats: dict[str, Any],
) -> dict[str, Any]:
    model = train_model(route, train_features, train_target_norm, seed=seed, budget_steps=budget_steps)
    selection_pred = infer(model, selection_features, stats)
    promotion_pred = infer(model, promotion_features, stats)
    selection_metrics = evaluate_predictions(selection_targets, selection_pred, stats)
    promotion_metrics = evaluate_predictions(promotion_targets, promotion_pred, stats)
    checkpoint_path = path_for_trial(strategy, route.route_id, seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
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
        "checkpoint_sha256": checkpoint_sha256,
    }
    return trial


def pilot_route(
    route: RouteSpec,
    seed: int,
    budget_steps: int,
    train_features: np.ndarray,
    train_target_norm: np.ndarray,
    selection_features: np.ndarray,
    selection_targets: np.ndarray,
    stats: dict[str, Any],
    a0_selection_metrics: dict[str, Any],
) -> dict[str, Any]:
    model = train_model(route, train_features, train_target_norm, seed=seed, budget_steps=budget_steps)
    selection_pred = infer(model, selection_features, stats)
    selection_metrics = evaluate_predictions(selection_targets, selection_pred, stats)
    feedback = compare_to_baseline(selection_metrics, a0_selection_metrics)
    return {
        "route_id": route.route_id,
        "model_name": route.model_name,
        "seed": seed,
        "budget_steps": budget_steps,
        "blocked": False,
        "selection": selection_metrics,
        "feedback": feedback,
        "prediction_hash": prediction_hash(selection_pred),
    }


def blocked_route(route_id: str, reason: str) -> dict[str, Any]:
    return {"route_id": route_id, "blocked": True, "reason": reason, "feedback": "blocked"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(records: dict[str, Any]) -> dict[str, Any]:
    return {
        "a0": records["a0"],
        "split": records["split"],
        "routes": records["routes"],
        "gate": records["gate"],
        "pilots": records["pilots"],
        "strategies": records["strategies"],
        "promotion_gate": records["promotion_gate"],
        "trial_budget_steps": records["trial_budget_steps"],
        "seed_pool": list(records["seed_pool"]),
        "commands": records["commands"],
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = ["# P28 agentic optimization", ""]
    lines.append("## A0 freeze")
    lines.append(f"- baseline model: `{summary['a0']['model']}`")
    lines.append(f"- baseline metrics path: `{summary['a0']['metrics_path']}`")
    lines.append(f"- baseline run_manifest hash: `{summary['a0']['run_manifest_sha256']}`")
    lines.append(f"- A1 prediction hash (selection-dev): `{summary['a0']['selection_prediction_hash']}`")
    lines.append("")
    lines.append("## Nested split")
    for split_name, split in summary["split"].items():
        lines.append(f"- {split_name}: families={split['families']} / wells={split['wells']} / n={split['count']}")
    lines.append("")
    lines.append("## CIG route gate")
    lines.append(f"- status: `{summary['gate']['status']}`")
    lines.append(f"- reason: {summary['gate']['reason']}")
    lines.append(f"- evidence: `{summary['gate']['evidence_path']}`")
    lines.append("")
    lines.append("## Pilot feedback")
    for pilot in summary["pilots"]:
        lines.append(f"- {pilot['route_id']}: feedback={pilot['feedback']} selection_MAE_macro={pilot.get('selection', {}).get('physical_MAE_macro', 'blocked')}")
    lines.append("")
    lines.append("## Strategy outcomes")
    for name, result in summary["strategies"].items():
        lines.append(f"- {name}: chosen_route={result['chosen_route_id']} selected_by={result['selected_by']} promotion_gate={result['promotion_gate']}")
        selection_median = result.get("selection_median")
        promotion_median = result.get("promotion_median")
        if selection_median is not None:
            lines.append(f"  - selection_median_MAE_macro={selection_median['physical_MAE_macro']:.6f}")
        else:
            lines.append("  - selection_median_MAE_macro=blocked")
        if promotion_median is not None:
            lines.append(f"  - promotion_median_MAE_macro={promotion_median['physical_MAE_macro']:.6f}")
        else:
            lines.append("  - promotion_median_MAE_macro=blocked")
    lines.append("")
    lines.append("## Promotion gate")
    lines.append(f"- passed: `{summary['promotion_gate']['passed']}`")
    lines.append(f"- primary threshold: {summary['promotion_gate']['primary_threshold']}")
    lines.append(f"- worst-group threshold: {summary['promotion_gate']['worst_group_threshold']}")
    lines.append("")
    lines.append("## Commands")
    for cmd in summary["commands"]:
        lines.append(f"- `{cmd}`")
    return "\n".join(lines) + "\n"


def build_protocol(
    *,
    split: dict[str, list[Record]],
    a0_selection_hash: str,
    a0_promotion_hash: str,
    gate: dict[str, Any],
    route_pilots: list[dict[str, Any]],
    strategy_choice: dict[str, Any],
    budget_steps: int,
    seed_pool: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "schema_version": "p28_agentic_optimization/v1",
        "track_id": "property",
        "root_seed": ROOT_SEED,
        "train_h5_sha256": sha256_file(TRAIN_H5),
        "a0": {
            "model": "tiny_mlp",
            "metrics_path": str(BASELINE_METRICS.relative_to(RESERVOIR_DIR)),
            "run_manifest_path": str(BASELINE_RUN_MANIFEST.relative_to(RESERVOIR_DIR)),
            "run_manifest_sha256": sha256_file(BASELINE_RUN_MANIFEST),
            "metrics_sha256": sha256_file(BASELINE_METRICS),
            "checkpoint_path": str(BASELINE_CHECKPOINT.relative_to(RESERVOIR_DIR)),
            "checkpoint_sha256": sha256_file(BASELINE_CHECKPOINT),
            "selection_prediction_hash": a0_selection_hash,
            "promotion_prediction_hash": a0_promotion_hash,
        },
        "split": {
            name: {
                "count": len(records),
                "families": sorted({record.family_id for record in records}),
                "wells": sorted({record.well_id for record in records}),
                "samples": [record.sample_id for record in records[:5]],
            }
            for name, records in split.items()
        },
        "gate": gate,
        "route_catalog": route_catalog(),
        "pilot_budget_steps": PILOT_STEPS,
        "trial_budget_steps": budget_steps,
        "seed_pool": list(seed_pool),
        "route_pilots": route_pilots,
        "strategy_choice": strategy_choice,
        "promotion_gate": {
            "primary_metric": "physical_MAE_macro",
            "primary_threshold_rel_improvement": 0.01,
            "worst_group_rmse_max_worsening": 0.02,
            "selection_vs_promotion_separation": "selection-dev only used for route choice; promotion-dev only used for promotion gate",
        },
        "allowlist": [route.route_id for route in ROUTES],
        "cig_rejection": gate,
    }


def write_outputs(
    *,
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    evidence_path: Path,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "results.jsonl", rows)
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    evidence_path.write_text(markdown_summary(summary), encoding="utf-8")
    manifest = {
        "schema_version": "p28_agentic_optimization/manifest/v1",
        "inputs": {
            "train_h5": {"path": str(TRAIN_H5.relative_to(PROJECT_ROOT)), "sha256": sha256_file(TRAIN_H5)},
            "baseline_metrics": {"path": str(BASELINE_METRICS.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(BASELINE_METRICS)},
            "baseline_run_manifest": {"path": str(BASELINE_RUN_MANIFEST.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(BASELINE_RUN_MANIFEST)},
            "baseline_checkpoint": {"path": str(BASELINE_CHECKPOINT.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(BASELINE_CHECKPOINT)},
            "p18_evidence": {"path": str(P18_EVIDENCE.relative_to(RESERVOIR_DIR)), "sha256": sha256_file(P18_EVIDENCE)},
        },
        "outputs": {
            "results_jsonl": {"path": "results.jsonl", "sha256": sha256_file(OUTPUT_DIR / "results.jsonl")},
            "protocol_json": {"path": "protocol.json", "sha256": sha256_file(OUTPUT_DIR / "protocol.json")},
            "summary_json": {"path": "summary.json", "sha256": sha256_file(OUTPUT_DIR / "summary.json")},
            "evidence_md": {"path": "evidence.md", "sha256": sha256_file(evidence_path)},
        },
        "routes": route_catalog(),
        "split": protocol["split"],
        "gate": protocol["gate"],
        "command": summary["commands"],
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def build_deepseek_decision(route_pilots: list[dict[str, Any]], a0_selection_hash: str, prompt_log: Path) -> dict[str, Any]:
    prompt = strict_anonymous_prompt(route_pilots, route_pilots, a0_selection_hash)
    prompt_log.write_text(json.dumps(prompt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    response = call_deepseek(prompt)
    if response.get("action") not in {"select", "stop"}:
        raise RuntimeError(f"DeepSeek returned invalid action: {response}")
    if response.get("action") == "select" and response.get("route_id") not in {f"route_{i}" for i in range(1, 5)}:
        raise RuntimeError(f"DeepSeek returned invalid route_id: {response}")
    return response


def execute(budget_steps: int = DEFAULT_BUDGET_STEPS, pilot_steps: int = PILOT_STEPS) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split = split_records(load_records())
    stats = fit_baseline_stats(split["train"])
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

    gate = gather_cig_gate()
    route_pilots: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    commands = [
        "python3 -m py_compile _pipelines/02_task_datasets/reservoir/p28_agentic_optimization.py _pipelines/02_task_datasets/reservoir/tests/test_p28_agentic_optimization.py",
        "pytest -q _pipelines/02_task_datasets/reservoir/tests/test_p28_agentic_optimization.py",
        "python3 _pipelines/02_task_datasets/reservoir/p28_agentic_optimization.py --budget-steps 32 --pilot-steps 8",
    ]

    for route in ROUTES:
        pilot = pilot_route(
            route,
            seed=ROOT_SEED,
            budget_steps=pilot_steps,
            train_features=train_features,
            train_target_norm=train_target_norm,
            selection_features=selection_features,
            selection_targets=selection_targets,
            stats=stats,
            a0_selection_metrics=a0_selection_metrics,
        )
        route_pilots.append(pilot)
        rows.append(
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
                "blocked": False,
            }
        )

    deepseek_log = OUTPUT_DIR / "deepseek_prompt.json"
    try:
        deepseek_decision = build_deepseek_decision(route_pilots, a0_selection_hash, deepseek_log)
    except Exception as exc:
        deepseek_decision = {"action": "stop", "route_id": None, "reason": f"provider_missing_or_invalid: {exc}"}
        rows.append({"kind": "strategy", "strategy": "A2L", "status": "blocked", "reason": str(exc)})

    if deepseek_decision.get("action") == "select":
        selected_route_index = int(str(deepseek_decision["route_id"]).split("_")[-1]) - 1
        a2l_route = ROUTES[selected_route_index]
        a2l_selected_by = "deepseek"
    else:
        a2l_route = None
        a2l_selected_by = "blocked"

    a2d_route_id = choose_a2d_route(route_pilots)
    a2d_route = next(route for route in ROUTES if route.route_id == a2d_route_id)
    a3_route_id = choose_a3_route(route_pilots, seed=ROOT_SEED)
    a3_route = next(route for route in ROUTES if route.route_id == a3_route_id)

    strategy_choices = {
        "A2L": {"selected_by": a2l_selected_by, "chosen_route_id": a2l_route.route_id if a2l_route else None, "deepseek": deepseek_decision},
        "A2D": {"selected_by": "deterministic_pilot_rank", "chosen_route_id": a2d_route.route_id},
        "A3": {"selected_by": "pcg64_no_replacement", "chosen_route_id": a3_route.route_id},
    }

    strategy_routes = {"A2L": a2l_route, "A2D": a2d_route, "A3": a3_route}
    strategy_results: dict[str, dict[str, Any]] = {}
    for strategy, route in strategy_routes.items():
        if route is None:
            strategy_results[strategy] = {
                "chosen_route_id": None,
                "selected_by": strategy_choices[strategy]["selected_by"],
                "status": "blocked",
                "reason": "DeepSeek stop action",
                "selection_trials": [],
                "promotion_trials": [],
                "promotion_gate": False,
            }
            continue
        trial_rows = []
        selection_trials = []
        promotion_trials = []
        for seed in SAME_BUDGET_SEEDS:
            trial = run_trial(
                strategy=strategy,
                route=route,
                seed=int(seed),
                budget_steps=budget_steps,
                train_features=train_features,
                train_target_norm=train_target_norm,
                selection_features=selection_features,
                selection_targets=selection_targets,
                promotion_features=promotion_features,
                promotion_targets=promotion_targets,
                stats=stats,
            )
            rows.append({"kind": "trial", **trial})
            trial_rows.append(trial)
            selection_trials.append(trial["selection"])
            promotion_trials.append(trial["promotion"])
        selection_median = {
            "physical_MAE_macro": float(np.median([trial["physical_MAE_macro"] for trial in selection_trials])),
            "worst_group_RMSE": {
                target: float(np.median([trial["worst_group_RMSE"][target] for trial in selection_trials]))
                for target in PHYSICAL_TARGETS
            },
        }
        promotion_median = {
            "physical_MAE_macro": float(np.median([trial["physical_MAE_macro"] for trial in promotion_trials])),
            "worst_group_RMSE": {
                target: float(np.median([trial["worst_group_RMSE"][target] for trial in promotion_trials]))
                for target in PHYSICAL_TARGETS
            },
        }
        baseline_macro = a0_promotion_metrics["physical_MAE_macro"]
        strategy_results[strategy] = {
            "chosen_route_id": route.route_id,
            "selected_by": strategy_choices[strategy]["selected_by"],
            "selection_trials": selection_trials,
            "promotion_trials": promotion_trials,
            "selection_median": selection_median,
            "promotion_median": promotion_median,
            "route_model_name": route.model_name,
            "status": "ok",
            "promotion_gate": (
                promotion_median["physical_MAE_macro"] <= baseline_macro * 0.99
                and all(
                    promotion_median["worst_group_RMSE"][target]
                    <= a0_promotion_metrics["worst_group_RMSE"][target] * 1.02
                    for target in PHYSICAL_TARGETS
                )
            ),
        }

    promotion_gate_passed = all(
        strategy_results[name]["promotion_median"]["physical_MAE_macro"] <= baseline_macro * 0.99
        and all(
            strategy_results[name]["promotion_median"]["worst_group_RMSE"][target]
            <= a0_promotion_metrics["worst_group_RMSE"][target] * 1.02
            for target in PHYSICAL_TARGETS
        )
        for name in strategy_results
        if strategy_results[name]["status"] == "ok"
    )
    best_deterministic = min(
        (
            (strategy_results[name]["promotion_median"]["physical_MAE_macro"], name)
            for name in ("A2D", "A3")
            if strategy_results[name]["status"] == "ok"
        ),
        default=(float("inf"), None),
    )[1]
    keep_llm = (
        strategy_results["A2L"]["status"] == "ok"
        and strategy_results["A2L"]["selection_median"]["physical_MAE_macro"]
        <= min(strategy_results[name]["selection_median"]["physical_MAE_macro"] for name in ("A2D", "A3") if strategy_results[name]["status"] == "ok")
    )
    if not promotion_gate_passed and best_deterministic is not None:
        keep_llm = False

    summary = summarize(
        {
            "a0": {
                "model": "tiny_mlp",
                "metrics_path": str(BASELINE_METRICS.relative_to(RESERVOIR_DIR)),
                "run_manifest_sha256": sha256_file(BASELINE_RUN_MANIFEST),
                "selection_prediction_hash": a0_selection_hash,
            },
            "routes": route_catalog(),
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
            "gate": gate,
            "pilots": route_pilots,
            "strategies": strategy_results,
            "promotion_gate": {
                "passed": promotion_gate_passed,
                "primary_threshold": "seed-median physical_MAE_macro improvement >= 1% vs A0 on promotion-dev",
                "worst_group_threshold": "per-target worst_group_RMSE worsening < 2% vs A0 on promotion-dev",
                "keep_llm": keep_llm,
                "best_deterministic": best_deterministic,
            },
            "trial_budget_steps": budget_steps,
            "seed_pool": SAME_BUDGET_SEEDS,
            "commands": commands,
        }
    )
    protocol = build_protocol(
        split=split,
        a0_selection_hash=a0_selection_hash,
        a0_promotion_hash=a0_promotion_hash,
        gate=gate,
        route_pilots=route_pilots,
        strategy_choice=strategy_choices,
        budget_steps=budget_steps,
        seed_pool=SAME_BUDGET_SEEDS,
    )
    write_outputs(protocol=protocol, rows=rows, summary=summary, evidence_path=OUTPUT_DIR / "evidence.md")
    return {
        "protocol": protocol,
        "summary": summary,
        "strategy_choice": strategy_choices,
        "a0_promotion_metrics": a0_promotion_metrics,
        "a0_selection_metrics": a0_selection_metrics,
        "a0_selection_hash": a0_selection_hash,
        "a0_promotion_hash": a0_promotion_hash,
        "route_pilots": route_pilots,
        "promotion_gate_passed": promotion_gate_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P28 agentic optimization preflight/ablation.")
    parser.add_argument("--budget-steps", type=int, default=DEFAULT_BUDGET_STEPS)
    parser.add_argument("--pilot-steps", type=int, default=PILOT_STEPS)
    args = parser.parse_args()
    if args.budget_steps <= 0 or args.pilot_steps <= 0:
        raise SystemExit("budget-steps and pilot-steps must be positive")
    result = execute(budget_steps=args.budget_steps, pilot_steps=args.pilot_steps)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
