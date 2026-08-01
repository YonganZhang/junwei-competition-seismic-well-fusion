#!/usr/bin/env python3
"""P28 bounded execution-agent ablation for reconstruction geostatistics.

The runner keeps the committed P21 pretrained-foundation kernel ensemble as
the immutable A0/A1 reference.  Three policies receive the same four-trial
budget over a frozen, single-factor kernel registry.  Candidate selection is
nested by outer spatial fold and reuses P19's coordinate purge, so held-fold
labels and duplicate coordinates never enter policy feedback.

A2L is a real DeepSeek JSON policy.  A2D is a deterministic four-action
search, while A3 is PCG64 random search over the same registry.  A3 never
changes or randomises the foundation encoder.  There is no test/holdout CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request

import numpy as np
from sklearn.neighbors import NearestNeighbors


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(PROJECT_ROOT)]

import p11_residual_fusion as base  # noqa: E402
import p17_foundation_geostatistics as p17  # noqa: E402
import p18_anisotropic_foundation_geostatistics as p18  # noqa: E402
import p19_meta_purged_geostatistics as p19  # noqa: E402
import p21_fixed_foundation_ensemble as p21  # noqa: E402


SCHEMA_VERSION = "reconstruction-p28-agentic-optimization/v1"
PROTOCOL_SCHEMA_VERSION = "reconstruction-p28-protocol/v1"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p28_agentic_optimization"
DEFAULT_DATA_DIR = (
    PROJECT_ROOT.parent
    / "track-reconstruction"
    / "_data"
    / "processed"
    / "reconstruction"
)
DEFAULT_STAGE3_ROOT = (
    PROJECT_ROOT.parent
    / "p5-stage3-reconstruction"
    / "_tmp"
    / "p5_stage3_reconstruction"
)
P21_OUTPUT_DIR = HERE / "_outputs" / "p21_fixed_foundation_ensemble"
P21_SUMMARY_PATH = P21_OUTPUT_DIR / "summary.json"
P21_PREDICTIONS_PATH = P21_OUTPUT_DIR / "predictions.npz"
P18_CIG_SUMMARY_PATH = HERE / "_outputs" / "p18_cigbench_ked" / "summary.json"
P15_EVIDENCE_PATH = HERE / "_outputs" / "p15_gfm_finetune" / "evidence.md"

EXPECTED_A0_RMSE = 0.027734374378067677
EXPECTED_A0_ARRAY_SHA256 = (
    "f0151248f30587465b54154b2eee6a8d724525a8640094838675feb56cb02e9f"
)
EXPECTED_P21_SUMMARY_SHA256 = (
    "25a92577ca954ac1b7ca214848d24a25d0cac8a9b2ba7af5d584e4360da882b8"
)
EXPECTED_P21_PREDICTIONS_SHA256 = (
    "7eb149021973362e8a9a788f2919a7775bffc702061f2e2ce44031f9ee835afb"
)
EXPECTED_FEATURE_CACHE_SHA256 = (
    "0e6233be85c8bc9f013d29b9ef07038132233befcaff23e84a5418935cfc82d8"
)
EXPECTED_CIG_SUMMARY_SHA256 = (
    "b3ab1d6090791373a65a98ff937acd4d78faef7e802c85afb916cf90fa024107"
)
EXPECTED_RGT_RELATIVE_CHANGE = 0.006421439169313357

ROOT_SEED = 2693
TRIALS_PER_STRATEGY = 4
FLAT_RELATIVE_TOLERANCE = 0.0005
MIN_LLM_RELATIVE_GAIN = 0.005
MAX_FOLD_RELATIVE_REGRESSION = 0.01
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
PROVIDER_TIMEOUT_SECONDS = 120.0
DEFAULT_CREDENTIAL_HELPER = (
    Path.home() / ".claude" / "skills" / "share-docs" / "scripts" / "get-credential.sh"
)

A0_PARAMETERS: dict[str, Any] = {
    "kernel": "inverse_distance",
    "neighbours": 64,
    "distance_power": 1.5,
    "blend_weight": 0.75,
    "vertical_weight": 4.0,
    "foundation_weight": 0.1,
    "seismic_weights": [0.0, 0.1, 0.2],
    "pca_components": 16,
}


def _action(
    action_id: str,
    changed_factor: str,
    value: Any,
    description: str,
) -> dict[str, Any]:
    parameters = dict(A0_PARAMETERS)
    parameters[changed_factor] = value
    return {
        "action_id": action_id,
        "changed_factor": changed_factor,
        "value": value,
        "description": description,
        "parameters": parameters,
    }


ACTION_REGISTRY: tuple[dict[str, Any], ...] = (
    _action("kernel_matern32", "kernel", "matern32", "Matérn-3/2 weights"),
    _action("kernel_gaussian", "kernel", "gaussian", "Gaussian weights"),
    _action(
        "kernel_rational_quadratic",
        "kernel",
        "rational_quadratic",
        "rational-quadratic weights with alpha=1",
    ),
    _action("neighbours_48", "neighbours", 48, "48 nearest neighbours"),
    _action("neighbours_80", "neighbours", 80, "80 nearest neighbours"),
    _action(
        "distance_power_1.25",
        "distance_power",
        1.25,
        "inverse-distance power 1.25",
    ),
    _action(
        "distance_power_1.75",
        "distance_power",
        1.75,
        "inverse-distance power 1.75",
    ),
    _action("blend_0.70", "blend_weight", 0.70, "kernel blend weight 0.70"),
    _action("blend_0.80", "blend_weight", 0.80, "kernel blend weight 0.80"),
)
ACTION_BY_ID = {row["action_id"]: row for row in ACTION_REGISTRY}
DETERMINISTIC_SEQUENCE = (
    "kernel_matern32",
    "neighbours_48",
    "distance_power_1.25",
    "blend_0.70",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return p21._metrics(target, prediction)  # noqa: SLF001


def _validate_registry() -> None:
    if len(ACTION_REGISTRY) != 9 or len(ACTION_BY_ID) != len(ACTION_REGISTRY):
        raise RuntimeError("P28 action registry identity drift")
    for action in ACTION_REGISTRY:
        changed = [
            key
            for key, value in action["parameters"].items()
            if value != A0_PARAMETERS[key]
        ]
        if changed != [action["changed_factor"]]:
            raise RuntimeError(
                f"{action['action_id']} is not a legal single-factor action"
            )
    if tuple(DETERMINISTIC_SEQUENCE) != tuple(dict.fromkeys(DETERMINISTIC_SEQUENCE)):
        raise RuntimeError("deterministic sequence contains duplicates")


def _validate_output_dir(output_dir: Path) -> Path:
    resolved = Path(output_dir).expanduser().resolve()
    if resolved != DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError("P28 may write only its declared reconstruction output")
    return resolved


def _kernel_weights(
    distance: np.ndarray,
    *,
    family: str,
    power: float,
    bandwidth: float,
) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    safe = np.maximum(distance, 1e-8)
    scale = max(float(bandwidth), 1e-8)
    if family == "inverse_distance":
        weights = safe ** (-float(power))
    elif family == "matern32":
        radius = np.sqrt(3.0) * safe / scale
        weights = (1.0 + radius) * np.exp(-radius)
    elif family == "gaussian":
        weights = np.exp(-0.5 * (safe / scale) ** 2)
    elif family == "rational_quadratic":
        weights = 1.0 / (1.0 + 0.5 * (safe / scale) ** 2)
    else:
        raise ValueError(f"unknown kernel family: {family}")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise FloatingPointError(f"invalid {family} kernel weights")
    return weights


def _build_action_bank(
    *,
    oof: base.OOFDevelopment,
    prepared: Sequence[Mapping[str, Any]],
    action_ids: Sequence[str] | None = None,
    seed: int = ROOT_SEED,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Vectorise all registered actions over fixed P21 metric features."""

    if int(seed) != ROOT_SEED:
        raise ValueError("P28 action execution seed is frozen")
    requested_actions = ("A0", *ACTION_BY_ID) if action_ids is None else tuple(action_ids)
    if not requested_actions or any(
        action_id != "A0" and action_id not in ACTION_BY_ID
        for action_id in requested_actions
    ):
        raise ValueError("unknown P28 action execution id")
    action_predictions: dict[str, list[np.ndarray]] = {
        action_id: [] for action_id in requested_actions
    }
    bandwidth_audit: list[dict[str, Any]] = []
    max_neighbours = max(
        int(action["parameters"]["neighbours"]) for action in ACTION_REGISTRY
    )
    for seismic_weight in A0_PARAMETERS["seismic_weights"]:
        fold_neighbours: list[tuple[Any, np.ndarray, np.ndarray, float]] = []
        for row in prepared:
            fold = row["fold"]
            train_coordinate = np.array(row["train_coordinate"], copy=True)
            validation_coordinate = np.array(
                row["validation_coordinate"], copy=True
            )
            train_coordinate[:, 2] *= A0_PARAMETERS["vertical_weight"]
            validation_coordinate[:, 2] *= A0_PARAMETERS["vertical_weight"]
            train_metric = np.column_stack(
                [
                    train_coordinate,
                    seismic_weight * row["train_seismic"],
                    A0_PARAMETERS["foundation_weight"] * row["train_latent"],
                ]
            )
            validation_metric = np.column_stack(
                [
                    validation_coordinate,
                    seismic_weight * row["validation_seismic"],
                    A0_PARAMETERS["foundation_weight"]
                    * row["validation_latent"],
                ]
            )
            model = NearestNeighbors(n_neighbors=max_neighbours, n_jobs=1).fit(
                train_metric
            )
            distance, neighbour_rows = model.kneighbors(validation_metric)
            train_query_count = min(65, len(train_metric))
            train_distance = NearestNeighbors(
                n_neighbors=train_query_count, n_jobs=1
            ).fit(train_metric).kneighbors(train_metric, return_distance=True)[0]
            bandwidth = float(np.median(train_distance[:, -1]))
            fold_neighbours.append((fold, distance, neighbour_rows, bandwidth))
            bandwidth_audit.append(
                {
                    "fold_id": int(fold.fold_id),
                    "seismic_weight": float(seismic_weight),
                    "bandwidth": bandwidth,
                    "bandwidth_fit_rows": int(len(train_metric)),
                    "target_used_for_bandwidth": False,
                }
            )

        action_specs = [("A0", A0_PARAMETERS)] + [
            (action["action_id"], action["parameters"])
            for action in ACTION_REGISTRY
            if action["action_id"] in requested_actions
        ]
        for action_id, parameters in action_specs:
            if action_id not in requested_actions:
                continue
            kernel_prediction = np.full(len(oof.target), np.nan, dtype=np.float64)
            for fold, distance, neighbour_rows, bandwidth in fold_neighbours:
                neighbours = int(parameters["neighbours"])
                selected_distance = distance[:, :neighbours]
                selected_rows = neighbour_rows[:, :neighbours]
                weights = _kernel_weights(
                    selected_distance,
                    family=str(parameters["kernel"]),
                    power=float(parameters["distance_power"]),
                    bandwidth=bandwidth,
                )
                prediction = np.sum(
                    weights * fold.train_target[selected_rows], axis=1
                ) / np.sum(weights, axis=1)
                kernel_prediction[oof.fold_ids == fold.fold_id] = prediction
            if not np.all(np.isfinite(kernel_prediction)):
                raise RuntimeError(f"incomplete OOF prediction for {action_id}")
            blend = float(parameters["blend_weight"])
            action_predictions[action_id].append(
                (1.0 - blend) * oof.baseline + blend * kernel_prediction
            )

    bank = {
        action_id: np.mean(np.stack(rows), axis=0)
        for action_id, rows in action_predictions.items()
    }
    return bank, bandwidth_audit


def _load_a0(oof: base.OOFDevelopment) -> tuple[np.ndarray, dict[str, Any]]:
    if _sha256(P21_SUMMARY_PATH) != EXPECTED_P21_SUMMARY_SHA256:
        raise RuntimeError("P21 summary hash drift")
    if _sha256(P21_PREDICTIONS_PATH) != EXPECTED_P21_PREDICTIONS_SHA256:
        raise RuntimeError("P21 prediction artifact hash drift")
    with np.load(P21_PREDICTIONS_PATH, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["indices_kji"], oof.indices_kji)
        np.testing.assert_array_equal(payload["fold_ids"], oof.fold_ids)
        np.testing.assert_allclose(payload["target"], oof.target, rtol=0.0, atol=0.0)
        prediction = np.asarray(payload["candidate_prediction"], dtype=np.float64)
    metrics = _metrics(oof.target, prediction)
    if not np.isclose(metrics["rmse"], EXPECTED_A0_RMSE, rtol=0.0, atol=1e-15):
        raise RuntimeError("P21 A0 RMSE drift")
    array_hash = p17._array_sha256(prediction)  # noqa: SLF001
    if array_hash != EXPECTED_A0_ARRAY_SHA256:
        raise RuntimeError("P21 A0 array hash drift")
    return prediction, {
        "summary_sha256": EXPECTED_P21_SUMMARY_SHA256,
        "predictions_sha256": EXPECTED_P21_PREDICTIONS_SHA256,
        "prediction_array_sha256": array_hash,
        "metrics": metrics,
    }


def _load_deepseek_key(helper: Path) -> str:
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    helper = Path(helper).expanduser().resolve()
    if not helper.is_file():
        raise RuntimeError("DeepSeek credential helper is unavailable")
    result = subprocess.run(
        [str(helper), "DEEPSEEK_API_KEY"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    key = result.stdout.strip()
    if result.returncode != 0 or not key:
        raise RuntimeError("DeepSeek credential is unavailable (fail-closed)")
    return key


def _deepseek_json(
    *,
    key: str,
    system: str,
    observation: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    request_body = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    observation, ensure_ascii=False, sort_keys=True
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        DEEPSEEK_ENDPOINT,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "reconstruction-p28/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=PROVIDER_TIMEOUT_SECONDS
        ) as response:
            envelope = json.loads(response.read().decode("utf-8"))
        raw_content = envelope["choices"][0]["message"]["content"]
        parsed = json.loads(raw_content)
    except (
        urllib.error.URLError,
        TimeoutError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"DeepSeek provider failed closed: {type(exc).__name__}: {exc}"
        ) from exc
    return parsed, {
        "provider": "deepseek",
        "endpoint": DEEPSEEK_ENDPOINT,
        "model_requested": DEEPSEEK_MODEL,
        "model_returned": envelope.get("model", "unknown"),
        "response_id": envelope.get("id", ""),
        "usage": envelope.get("usage", {}),
        "request_sha256": _hash_payload(request_body),
        "response_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        "credential_persisted": False,
    }


def _validate_route_gate(value: Any) -> dict[str, Any]:
    expected = {"schema_version", "route_id", "decision", "reason_code", "stop"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("route gate violates strict JSON keys")
    if value["schema_version"] != "p28-route-gate/v1":
        raise ValueError("route gate schema mismatch")
    if value["route_id"] != "rgt_ked":
        raise ValueError("route gate route mismatch")
    if value["decision"] not in {"accept", "reject"}:
        raise ValueError("route gate decision is invalid")
    if not isinstance(value["reason_code"], str) or not value["reason_code"]:
        raise ValueError("route gate reason_code is invalid")
    if not isinstance(value["stop"], bool):
        raise ValueError("route gate stop is not boolean")
    return dict(value)


def _run_route_gate(key: str) -> dict[str, Any]:
    if _sha256(P18_CIG_SUMMARY_PATH) != EXPECTED_CIG_SUMMARY_SHA256:
        raise RuntimeError("P18 CIG-Bench summary hash drift")
    summary = json.loads(P18_CIG_SUMMARY_PATH.read_text(encoding="utf-8"))
    experiment = summary["experiment"]
    relative_change = float(experiment["relative_rmse_change"])
    if not np.isclose(
        relative_change, EXPECTED_RGT_RELATIVE_CHANGE, rtol=0.0, atol=1e-15
    ):
        raise RuntimeError("P18 RGT-KED evidence drift")
    observation = {
        "schema_version": "p28-route-gate-observation/v1",
        "gate_mode": "confirmatory_preconstrained",
        "autonomous_route_discovery": False,
        "route": {
            "route_id": "rgt_ked",
            "baseline_rmse": experiment["baseline"]["pykrige_ok3d_repeat_0"][
                "rmse"
            ],
            "candidate_rmse": experiment["ked_metrics"]["rmse"],
            "relative_rmse_change_candidate_minus_baseline": relative_change,
            "fold_outcomes": experiment["independent_fold_outcome_counts"],
        },
        "decision_rule": (
            "reject a route when pooled RMSE is worse and losses exceed wins; "
            "return the exact response schema"
        ),
        "response_schema": {
            "schema_version": "p28-route-gate/v1",
            "route_id": "rgt_ked",
            "decision": "accept|reject",
            "reason_code": "short_snake_case",
            "stop": "boolean",
        },
    }
    parsed, provider = _deepseek_json(
        key=key,
        system=(
            "You are a fail-closed geostatistical route gate. Apply the supplied "
            "numeric rule exactly. Return only one JSON object with exactly the "
            "specified keys and no Markdown."
        ),
        observation=observation,
    )
    decision = _validate_route_gate(parsed)
    if decision["decision"] != "reject" or not decision["stop"]:
        raise RuntimeError("A2L failed to reject proven-negative RGT-KED")
    return {
        "observation": observation,
        "decision": decision,
        "provider_audit": provider,
        "source_summary_sha256": EXPECTED_CIG_SUMMARY_SHA256,
        "interpretation": (
            "The route was preconstrained to the already-evaluated RGT-KED path; "
            "DeepSeek confirms the registered negative-transfer rule but does not "
            "discover routes autonomously."
        ),
        "passed": True,
    }


def _feedback(
    *,
    target: np.ndarray,
    fold_ids: np.ndarray,
    held_fold: int,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    selection_folds = [fold for fold in base.FOLD_IDS if fold != held_fold]
    mask = fold_ids != held_fold
    baseline_rmse = float(_metrics(target[mask], baseline[mask])["rmse"])
    candidate_rmse = float(_metrics(target[mask], candidate[mask])["rmse"])
    relative = (candidate_rmse - baseline_rmse) / baseline_rmse
    if relative < -FLAT_RELATIVE_TOLERANCE:
        label = "improved"
    elif relative > FLAT_RELATIVE_TOLERANCE:
        label = "worse"
    else:
        label = "flat"
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    for fold in selection_folds:
        fold_mask = fold_ids == fold
        delta = float(_metrics(target[fold_mask], candidate[fold_mask])["rmse"]) - float(
            _metrics(target[fold_mask], baseline[fold_mask])["rmse"]
        )
        outcome = "win" if delta < -1e-12 else "loss" if delta > 1e-12 else "tie"
        outcomes[outcome] += 1
    return {
        "selection_fold_count": 4,
        "selection_folds": [int(value) for value in selection_folds],
        "relative_rmse_change": float(relative),
        "relative_gain": float(-relative),
        "outcomes": outcomes,
        "classification": label,
        "held_fold_metric_visible": False,
    }


def _validate_selection(
    value: Any,
    *,
    round_id: int,
    available_by_fold: Mapping[int, Sequence[str]],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"decisions"}:
        keys = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"A2L selection violates strict JSON keys: {keys}")
    decisions = value["decisions"]
    if not isinstance(decisions, list) or len(decisions) != len(base.FOLD_IDS):
        raise ValueError("A2L must decide once for every held fold")
    validated: list[dict[str, Any]] = []
    seen: set[int] = set()
    expected_keys = {"held_fold", "action_id", "rationale"}
    for row in decisions:
        if not isinstance(row, dict) or set(row) != expected_keys:
            keys = sorted(row) if isinstance(row, dict) else type(row).__name__
            raise ValueError(f"A2L fold decision violates strict keys: {keys}")
        held_fold = row["held_fold"]
        if isinstance(held_fold, bool) or not isinstance(held_fold, int):
            raise ValueError("A2L held_fold must be integer")
        if held_fold in seen or held_fold not in base.FOLD_IDS:
            raise ValueError("A2L held_fold set is invalid")
        seen.add(held_fold)
        if row["action_id"] not in available_by_fold[held_fold]:
            raise ValueError("A2L selected unavailable or repeated action")
        expected_stop = round_id == TRIALS_PER_STRATEGY
        if not isinstance(row["rationale"], str) or not row["rationale"].strip():
            raise ValueError("A2L rationale is invalid")
        validated.append(
            {
                "held_fold": held_fold,
                "action_id": row["action_id"],
                "stop_after_trial": expected_stop,
                "stop_source": "frozen_fixed_budget_protocol",
                "rationale": row["rationale"].strip()[:500],
            }
        )
    return sorted(validated, key=lambda row: row["held_fold"])


def _llm_round(
    *,
    key: str,
    round_id: int,
    histories: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    observation = _build_llm_observation(round_id=round_id, histories=histories)
    available_by_fold = {
        int(fold): [
            action_id
            for action_id in ACTION_BY_ID
            if action_id not in {row["action_id"] for row in histories[fold]}
        ]
        for fold in base.FOLD_IDS
    }
    parsed, provider = _deepseek_json(
        key=key,
        system=(
            "You are the constrained A2L search policy for reconstruction. "
            "Choose exactly one unique untried allowlisted action for each held "
            "fold. Use only the supplied categorical improved/flat/worse feedback. "
            "Never request or infer held-fold/promotion/test data. Return exactly "
            "the specified JSON object and no Markdown."
        ),
        observation=observation,
    )
    decisions = _validate_selection(
        parsed,
        round_id=round_id,
        available_by_fold=available_by_fold,
    )
    return decisions, observation, provider


def _build_llm_observation(
    *,
    round_id: int,
    histories: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the only observation allowed to reach the LLM."""

    available_by_fold = {
        int(fold): [
            action_id
            for action_id in ACTION_BY_ID
            if action_id not in {row["action_id"] for row in histories[fold]}
        ]
        for fold in base.FOLD_IDS
    }
    contexts = []
    for fold in base.FOLD_IDS:
        safe_history = [
            {
                "round": row["round"],
                "action_id": row["action_id"],
                "feedback": row["feedback"]["classification"],
            }
            for row in histories[fold]
        ]
        contexts.append(
            {
                "held_fold": int(fold),
                "available_action_ids": available_by_fold[fold],
                "prior_trials": safe_history,
            }
        )
    observation = {
        "schema_version": "p28-a2l-observation/v1",
        "round": round_id,
        "fixed_budget": TRIALS_PER_STRATEGY,
        "action_registry": [
            {
                "action_id": action["action_id"],
                "changed_factor": action["changed_factor"],
                "value": action["value"],
                "description": action["description"],
            }
            for action in ACTION_REGISTRY
        ],
        "fold_contexts": contexts,
        "visibility": {
            "only_categorical_feedback": True,
            "feedback_vocabulary": ["improved", "flat", "worse"],
            "held_fold_metrics_visible": False,
            "promotion_feedback_visible": False,
            "raw_predictions_visible": False,
        },
        "response_schema": {
            "decisions": [
                {
                    "held_fold": "integer 0..4",
                    "action_id": "one available action for that fold",
                    "rationale": "brief text",
                }
            ],
        },
    }
    return observation


def _random_sequences() -> dict[int, list[str]]:
    sequences: dict[int, list[str]] = {}
    action_ids = np.asarray(list(ACTION_BY_ID), dtype=object)
    for fold in base.FOLD_IDS:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([ROOT_SEED, fold])))
        sequences[int(fold)] = [
            str(value)
            for value in rng.choice(
                action_ids, size=TRIALS_PER_STRATEGY, replace=False
            ).tolist()
        ]
    return sequences


def _strategy_auc(histories: Mapping[int, Sequence[Mapping[str, Any]]]) -> float:
    values: list[float] = []
    for fold in base.FOLD_IDS:
        best = 0.0
        for row in histories[fold]:
            best = max(best, float(row["feedback"]["relative_gain"]))
            values.append(best)
    if len(values) != len(base.FOLD_IDS) * TRIALS_PER_STRATEGY:
        raise RuntimeError("strategy AUC does not contain the fixed trial budget")
    return float(np.mean(values))


def _select_promotions(
    *,
    histories: Mapping[int, Sequence[Mapping[str, Any]]],
    original_bank: Mapping[str, np.ndarray],
    a0_prediction: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.full(len(a0_prediction), np.nan, dtype=np.float64)
    selections: list[dict[str, Any]] = []
    for fold in base.FOLD_IDS:
        ranked = sorted(
            histories[fold],
            key=lambda row: (
                -float(row["feedback"]["relative_gain"]),
                int(row["round"]),
                str(row["action_id"]),
            ),
        )
        best = ranked[0]
        action_id = (
            best["action_id"]
            if float(best["feedback"]["relative_gain"]) > 0.0
            else "A0"
        )
        mask = fold_ids == fold
        source = a0_prediction if action_id == "A0" else original_bank[action_id]
        prediction[mask] = source[mask]
        selections.append(
            {
                "held_fold": int(fold),
                "selected_action_id": action_id,
                "selection_based_only_on_purged_other_four_folds": True,
                "selected_fold_train_relative_gain": max(
                    0.0, float(best["feedback"]["relative_gain"])
                ),
                "promotion_metric_fed_back": False,
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("strategy promotion prediction is incomplete")
    return prediction, selections


def _promotion_audit(
    *,
    target: np.ndarray,
    fold_ids: np.ndarray,
    a0: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    a0_metrics = _metrics(target, a0)
    candidate_metrics = _metrics(target, candidate)
    relative_gain = (
        float(a0_metrics["rmse"]) - float(candidate_metrics["rmse"])
    ) / float(a0_metrics["rmse"])
    rows: list[dict[str, Any]] = []
    counts = {"win": 0, "loss": 0, "tie": 0}
    max_regression = -np.inf
    for fold in base.FOLD_IDS:
        mask = fold_ids == fold
        base_rmse = float(_metrics(target[mask], a0[mask])["rmse"])
        candidate_rmse = float(_metrics(target[mask], candidate[mask])["rmse"])
        relative = (candidate_rmse - base_rmse) / base_rmse
        max_regression = max(max_regression, relative)
        outcome = (
            "win"
            if candidate_rmse < base_rmse - 1e-12
            else "loss"
            if candidate_rmse > base_rmse + 1e-12
            else "tie"
        )
        counts[outcome] += 1
        rows.append(
            {
                "fold_id": int(fold),
                "a0_rmse": base_rmse,
                "candidate_rmse": candidate_rmse,
                "relative_rmse_change_candidate_minus_a0": float(relative),
                "outcome": outcome,
            }
        )
    return {
        "metrics": candidate_metrics,
        "relative_gain_vs_a0": float(relative_gain),
        "maximum_fold_relative_regression": float(max_regression),
        "fold_outcomes": counts,
        "per_fold": rows,
    }


def _protocol() -> dict[str, Any]:
    _validate_registry()
    registry = [dict(action) for action in ACTION_REGISTRY]
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "root_seed": ROOT_SEED,
        "primary_metric": "pooled development nested-OOF RMSE",
        "outer_spatial_folds": [int(value) for value in base.FOLD_IDS],
        "independent_spatial_units": 5,
        "trials_per_strategy_per_held_fold": TRIALS_PER_STRATEGY,
        "strategies": {
            "A2L": "real DeepSeek constrained search",
            "A2D": "deterministic registered search",
            "A3": "PCG64 random search; not random-init foundation",
        },
        "route_gate": {
            "mode": "confirmatory_preconstrained",
            "autonomous_route_discovery": False,
            "registered_route": "rgt_ked",
        },
        "a0_parameters": A0_PARAMETERS,
        "action_registry": registry,
        "action_registry_sha256": _hash_payload(registry),
        "deterministic_sequence": list(DETERMINISTIC_SEQUENCE),
        "feedback": {
            "vocabulary": ["improved", "flat", "worse"],
            "flat_relative_tolerance": FLAT_RELATIVE_TOLERANCE,
            "only_categorical_classification_visible_to_a2l": True,
        },
        "search_auc": "mean cumulative-best relative fold-train RMSE gain over 5x4 cells",
        "promotion_gate": {
            "minimum_a2l_relative_gain": MIN_LLM_RELATIVE_GAIN,
            "maximum_any_fold_relative_regression": MAX_FOLD_RELATIVE_REGRESSION,
            "a2l_auc_must_strictly_exceed_a2d_and_a3": True,
        },
        "firewall": {
            "p19_coordinate_purge_reused": True,
            "promotion_not_fed_back_to_strategies": True,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "random_init_foundation_in_main_registry": False,
        },
    }


def run(
    *,
    data_dir: Path,
    stage3_root: Path,
    output_dir: Path,
    credential_helper: Path,
    recompute: bool = False,
) -> dict[str, Any]:
    output_dir = _validate_output_dir(output_dir)
    base.ensure_no_holdout_paths((data_dir, stage3_root, output_dir))
    if (output_dir / "summary.json").exists() and not recompute:
        raise RuntimeError("P28 completed evidence already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    _write_json(output_dir / "protocol.json", protocol)

    key = _load_deepseek_key(credential_helper)
    route_gate = _run_route_gate(key)
    _write_json(output_dir / "route_gate.json", route_gate)

    inputs = base.resolve_dev_inputs(Path(data_dir).expanduser().resolve())
    oof = base.load_oof_development(Path(stage3_root).expanduser().resolve())
    a0_prediction, a0_audit = _load_a0(oof)
    folds, fold_loading_audit = p17.load_fold_samples(
        stage3_root=stage3_root,
        train_h5=inputs.train_h5,
        oof=oof,
    )
    requested = p17._unique_indices(folds)  # noqa: SLF001
    cache_indices, foundation, feature_audit = p18._load_feature_cache(  # noqa: SLF001
        p18.DEFAULT_FEATURE_CACHE,
        train_h5=inputs.train_h5,
        expected_indices=requested,
    )
    if feature_audit["feature_cache_sha256"] != EXPECTED_FEATURE_CACHE_SHA256:
        raise RuntimeError("pretrained GFM feature cache identity drift")

    prepared, transform_audits = p18._prepare_fold_metrics(  # noqa: SLF001
        folds=folds,
        requested_indices=cache_indices,
        foundation_features=foundation,
    )
    original_bank, original_bandwidth_audit = _build_action_bank(
        oof=oof, prepared=prepared
    )
    a1_replay_bank, _ = _build_action_bank(
        oof=oof,
        prepared=prepared,
        action_ids=("A0",),
        seed=ROOT_SEED,
    )
    a1_prediction = np.asarray(a1_replay_bank["A0"], dtype=np.float64)
    if p17._array_sha256(a1_prediction) != p17._array_sha256(  # noqa: SLF001
        original_bank["A0"]
    ):
        raise RuntimeError("A1 same-entrypoint/action/seed replay diverged")
    a1_reference_prediction = np.asarray(original_bank["A0"], dtype=np.float64)
    a0_reproduction_max_abs = float(
        np.max(np.abs(original_bank["A0"] - a0_prediction))
    )
    if a0_reproduction_max_abs > 1e-12:
        raise RuntimeError(
            f"P28 failed to reproduce P21 A0: max abs {a0_reproduction_max_abs}"
        )
    original_bank["A0"] = a0_prediction

    purged_banks: dict[int, dict[str, np.ndarray]] = {}
    purge_audits: list[dict[str, Any]] = []
    for held_fold in base.FOLD_IDS:
        held = next(fold for fold in folds if fold.fold_id == held_fold)
        forbidden = p19._rows(held.validation_indices_kji)  # noqa: SLF001
        purged_folds: list[p17.FoldSamples] = []
        removed_by_fold: dict[str, int] = {}
        for fold in folds:
            if fold.fold_id == held_fold:
                purged_folds.append(fold)
                removed_by_fold[str(fold.fold_id)] = 0
            else:
                purged, removed = p19._without_coordinates(  # noqa: SLF001
                    fold, forbidden
                )
                purged_folds.append(purged)
                removed_by_fold[str(fold.fold_id)] = removed
        purged_prepared, purged_transform = p18._prepare_fold_metrics(  # noqa: SLF001
            folds=tuple(purged_folds),
            requested_indices=cache_indices,
            foundation_features=foundation,
        )
        purged_bank, _ = _build_action_bank(oof=oof, prepared=purged_prepared)
        purged_banks[int(held_fold)] = purged_bank
        purge_audits.append(
            {
                "held_fold": int(held_fold),
                "forbidden_unique_coordinates": int(len(forbidden)),
                "removed_train_labels_by_fold": removed_by_fold,
                "removed_occurrences": int(sum(removed_by_fold.values())),
                "selection_folds": [
                    int(value) for value in base.FOLD_IDS if value != held_fold
                ],
                "held_fold_labels_used_for_selection": False,
                "transform_audits": purged_transform,
            }
        )

    histories: dict[str, dict[int, list[dict[str, Any]]]] = {
        strategy: {int(fold): [] for fold in base.FOLD_IDS}
        for strategy in ("A2L", "A2D", "A3")
    }
    trial_rows: list[dict[str, Any]] = []
    provider_rounds: list[dict[str, Any]] = []
    random_sequences = _random_sequences()

    for round_id in range(1, TRIALS_PER_STRATEGY + 1):
        llm_decisions, observation, provider = _llm_round(
            key=key,
            round_id=round_id,
            histories=histories["A2L"],
        )
        provider_rounds.append(
            {
                "round": round_id,
                "observation_sha256": _hash_payload(observation),
                "decisions": llm_decisions,
                "provider_audit": provider,
            }
        )
        llm_by_fold = {row["held_fold"]: row for row in llm_decisions}
        for strategy in ("A2L", "A2D", "A3"):
            for held_fold in base.FOLD_IDS:
                if strategy == "A2L":
                    decision = llm_by_fold[held_fold]
                    action_id = decision["action_id"]
                    decision_audit = {
                        "source": "deepseek",
                        "stop_after_trial": decision["stop_after_trial"],
                        "rationale": decision["rationale"],
                    }
                elif strategy == "A2D":
                    action_id = DETERMINISTIC_SEQUENCE[round_id - 1]
                    decision_audit = {
                        "source": "deterministic_registered_sequence",
                        "stop_after_trial": round_id == TRIALS_PER_STRATEGY,
                    }
                else:
                    action_id = random_sequences[held_fold][round_id - 1]
                    decision_audit = {
                        "source": "PCG64_random_search",
                        "stop_after_trial": round_id == TRIALS_PER_STRATEGY,
                        "foundation_weight_mode": "pretrained_fixed",
                    }
                feedback = _feedback(
                    target=oof.target,
                    fold_ids=oof.fold_ids,
                    held_fold=int(held_fold),
                    baseline=purged_banks[int(held_fold)]["A0"],
                    candidate=purged_banks[int(held_fold)][action_id],
                )
                row = {
                    "schema_version": "reconstruction-p28-trial/v1",
                    "strategy": strategy,
                    "held_fold": int(held_fold),
                    "round": round_id,
                    "action_id": action_id,
                    "changed_factor": ACTION_BY_ID[action_id]["changed_factor"],
                    "feedback": feedback,
                    "decision_audit": decision_audit,
                    "promotion_metric_fed_back": False,
                }
                histories[strategy][int(held_fold)].append(row)
                trial_rows.append(row)

    trial_path = output_dir / "trials.jsonl"
    trial_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in trial_rows
        ),
        encoding="utf-8",
    )

    strategy_predictions: dict[str, np.ndarray] = {}
    strategy_results: dict[str, Any] = {}
    for strategy in ("A2L", "A2D", "A3"):
        prediction, selections = _select_promotions(
            histories=histories[strategy],
            original_bank=original_bank,
            a0_prediction=a0_prediction,
            fold_ids=oof.fold_ids,
        )
        strategy_predictions[strategy] = prediction
        strategy_results[strategy] = {
            "search_auc": _strategy_auc(histories[strategy]),
            "promotion_selections": selections,
            "promotion": _promotion_audit(
                target=oof.target,
                fold_ids=oof.fold_ids,
                a0=a0_prediction,
                candidate=prediction,
            ),
            "trials": int(
                sum(len(rows) for rows in histories[strategy].values())
            ),
            "trials_per_held_fold": TRIALS_PER_STRATEGY,
        }

    a2l = strategy_results["A2L"]
    a2l_gate = {
        "pooled_gain_pass": bool(
            a2l["promotion"]["relative_gain_vs_a0"] >= MIN_LLM_RELATIVE_GAIN
        ),
        "fold_non_degradation_pass": bool(
            a2l["promotion"]["maximum_fold_relative_regression"]
            <= MAX_FOLD_RELATIVE_REGRESSION
        ),
        "auc_superiority_pass": bool(
            a2l["search_auc"] > strategy_results["A2D"]["search_auc"]
            and a2l["search_auc"] > strategy_results["A3"]["search_auc"]
        ),
    }
    a2l_gate["passed"] = bool(all(a2l_gate.values()))
    if a2l_gate["passed"]:
        retained_strategy = "A2L"
        decision_state = "PROMOTE_LLM_AGENT"
    else:
        safe_rules = []
        for strategy in ("A2D", "A3"):
            promotion = strategy_results[strategy]["promotion"]
            if (
                promotion["relative_gain_vs_a0"] > 0.0
                and promotion["maximum_fold_relative_regression"]
                <= MAX_FOLD_RELATIVE_REGRESSION
            ):
                safe_rules.append(strategy)
        if safe_rules:
            retained_strategy = min(
                safe_rules,
                key=lambda strategy: strategy_results[strategy]["promotion"][
                    "metrics"
                ]["rmse"],
            )
            decision_state = "RETAIN_BETTER_NON_LLM_RULE"
        else:
            retained_strategy = "A0"
            decision_state = "RETAIN_FROZEN_BASELINE"

    prediction_path = output_dir / "predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            indices_kji=oof.indices_kji,
            fold_ids=oof.fold_ids,
            target=oof.target,
            a0_prediction=a0_prediction,
            a1_prediction=a1_prediction,
            a2l_prediction=strategy_predictions["A2L"],
            a2d_prediction=strategy_predictions["A2D"],
            a3_prediction=strategy_predictions["A3"],
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "objective": "bounded agentic kernel-parameter search over frozen P21 features",
        "protocol_sha256": _sha256(output_dir / "protocol.json"),
        "a0": a0_audit,
        "a1": {
            "replay": {
                "entrypoint": "_build_action_bank",
                "action_id": "A0",
                "seed": ROOT_SEED,
                "same_fold_inputs": True,
                "replayed_independently": True,
                "reference_array_sha256": p17._array_sha256(  # noqa: SLF001
                    a1_reference_prediction
                ),
                "replay_array_sha256": p17._array_sha256(  # noqa: SLF001
                    a1_prediction
                ),
                "max_abs_difference_vs_p21_canonical": float(
                    np.max(np.abs(a1_prediction - a0_prediction))
                ),
            },
            "prediction_array_sha256": p17._array_sha256(  # noqa: SLF001
                a1_prediction
            ),
            "equal_to_replayed_reference": bool(
                p17._array_sha256(a1_prediction)
                == p17._array_sha256(a1_reference_prediction)
            ),
            "within_p21_canonical_tolerance": bool(
                np.max(np.abs(a1_prediction - a0_prediction)) <= 1e-12
            ),
        },
        "independent_route_gate": route_gate,
        "strategies": strategy_results,
        "a2l_provider_rounds": provider_rounds,
        "a2l_promotion_gate": a2l_gate,
        "decision": {
            "state": decision_state,
            "retained_strategy": retained_strategy,
            "pretrained_foundation_contribution_claimed": False,
            "random_init_foundation_used_as_a3": False,
            "attribution_caveat": (
                "All P28 arms reuse the same pretrained GFM feature cache; any "
                "overall change is a kernel-search result, not a causal claim "
                "about foundation pretraining. Existing matched random-init "
                "evidence remains separate."
            ),
        },
        "purge_audits": purge_audits,
        "fold_loading_audit": fold_loading_audit,
        "feature_audit": feature_audit,
        "transform_audits": transform_audits,
        "bandwidth_audit": original_bandwidth_audit,
        "a0_reproduction_max_abs": a0_reproduction_max_abs,
        "inputs": {
            "train_h5_sha256": _sha256(inputs.train_h5),
            "feature_cache_sha256": _sha256(p18.DEFAULT_FEATURE_CACHE),
            "p21_summary_sha256": _sha256(P21_SUMMARY_PATH),
            "p21_predictions_sha256": _sha256(P21_PREDICTIONS_PATH),
            "p18_cig_summary_sha256": _sha256(P18_CIG_SUMMARY_PATH),
            "existing_random_init_attribution_evidence": {
                "path": str(P15_EVIDENCE_PATH.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(P15_EVIDENCE_PATH),
                "used_as_a3": False,
            },
        },
        "prediction_artifact": {
            "path": str(prediction_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(prediction_path),
            "rows": int(len(oof.target)),
        },
        "trial_artifact": {
            "path": str(trial_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(trial_path),
            "rows": len(trial_rows),
        },
        "firewall": {
            "development_only": True,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "held_fold_metrics_visible_to_strategy": False,
            "promotion_feedback_visible_to_strategy": False,
            "llm_prompt_feedback_is_classification_only": True,
            "p19_coordinate_purge_reused": True,
        },
    }
    _write_json(output_dir / "summary.json", result)
    _write_evidence(output_dir, result)
    return result


def _write_evidence(output_dir: Path, result: Mapping[str, Any]) -> None:
    a0_rmse = float(result["a0"]["metrics"]["rmse"])
    lines = [
        "# P28 agentic reconstruction optimization",
        "",
        "## Development promotion result",
        "",
        f"- Frozen P21 A0 RMSE: `{a0_rmse:.15f}`.",
    ]
    for strategy in ("A2L", "A2D", "A3"):
        row = result["strategies"][strategy]
        promotion = row["promotion"]
        lines.append(
            f"- {strategy} promotion RMSE: `{promotion['metrics']['rmse']:.15f}`; "
            f"relative gain vs A0 `{promotion['relative_gain_vs_a0']:+.6%}`; "
            f"fold outcomes `{promotion['fold_outcomes']}`; search AUC "
            f"`{row['search_auc']:.9f}`."
        )
    lines.extend(
        [
            f"- Decision: `{result['decision']['state']}`; retained strategy "
            f"`{result['decision']['retained_strategy']}`.",
            f"- A2L gate: `{result['a2l_promotion_gate']}`.",
            "",
            "## Agent and route controls",
            "",
            "- A2L used the real DeepSeek provider with strict JSON validation.",
            "- The independent A2L route gate rejected P18 RGT-KED, whose pooled "
            "RMSE was 0.6421439169% worse with 2/5 spatial-fold wins.",
            "- This RGT-KED gate is confirmatory and preconstrained to a registered "
            "route; it is not autonomous route discovery.",
            "- A2D and A3 each used the same four-trial budget per held fold. A3 "
            "is PCG64 random kernel search; it is not a random-init foundation arm.",
            "- A1 replays the same action entrypoint, fold inputs and frozen seed, "
            "then compares the independently recomputed prediction hash to A0.",
            "",
            "## Selection firewall",
            "",
            "P19 coordinate purge was applied independently for every held fold. "
            "The strategies saw only improved/flat/worse classifications from the "
            "other four purged folds. "
            "Held-fold and final promotion metrics were calculated only after all "
            "four actions were fixed and were never fed back to a strategy.",
            "",
            "The evidence contains five genuinely independent spatial units. The "
            "four policy trials are search evaluations, not additional independent "
            "geological samples. No frozen test or holdout file was opened.",
            "",
            "## Attribution boundary",
            "",
            result["decision"]["attribution_caveat"],
            "The existing P15 matched pretrained/random-init audit is cited only "
            "as separate attribution evidence and is not reused as A3.",
            "",
            "## Per-fold promotion outcomes",
            "",
            "| strategy | fold | A0 RMSE | candidate RMSE | relative change | outcome |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in ("A2L", "A2D", "A3"):
        for row in result["strategies"][strategy]["promotion"]["per_fold"]:
            lines.append(
                f"| {strategy} | {row['fold_id']} | {row['a0_rmse']:.12f} | "
                f"{row['candidate_rmse']:.12f} | "
                f"{row['relative_rmse_change_candidate_minus_a0']:+.6%} | "
                f"{row['outcome']} |"
            )
    (output_dir / "evidence.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def verify_evidence(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = _validate_output_dir(output_dir)
    protocol_path = output_dir / "protocol.json"
    summary_path = output_dir / "summary.json"
    trial_path = output_dir / "trials.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if summary["protocol_sha256"] != _sha256(protocol_path):
        raise RuntimeError("P28 protocol hash mismatch")
    if protocol["action_registry_sha256"] != _hash_payload(
        protocol["action_registry"]
    ):
        raise RuntimeError("P28 action registry hash mismatch")
    prediction_path = PROJECT_ROOT / summary["prediction_artifact"]["path"]
    if _sha256(prediction_path) != summary["prediction_artifact"]["sha256"]:
        raise RuntimeError("P28 prediction artifact hash mismatch")
    if _sha256(trial_path) != summary["trial_artifact"]["sha256"]:
        raise RuntimeError("P28 trial artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        target = np.asarray(payload["target"], dtype=np.float64)
        fold_ids = np.asarray(payload["fold_ids"], dtype=np.int64)
        a0 = np.asarray(payload["a0_prediction"], dtype=np.float64)
        a1 = np.asarray(payload["a1_prediction"], dtype=np.float64)
        predictions = {
            strategy: np.asarray(payload[f"{strategy.lower()}_prediction"], dtype=np.float64)
            for strategy in ("A2L", "A2D", "A3")
        }
    if p17._array_sha256(a0) != EXPECTED_A0_ARRAY_SHA256:  # noqa: SLF001
        raise RuntimeError("P28 A0/A1 identity drift")
    replay = summary["a1"]["replay"]
    if p17._array_sha256(a1) != replay["replay_array_sha256"]:
        raise RuntimeError("P28 A1 replay hash drift")
    if replay["replay_array_sha256"] != replay["reference_array_sha256"]:
        raise RuntimeError("P28 A1 replay is not equal to its independent reference")
    if float(replay["max_abs_difference_vs_p21_canonical"]) > 1e-12:
        raise RuntimeError("P28 A1 replay diverged beyond P21 tolerance")
    np.testing.assert_allclose(
        _metrics(target, a0)["rmse"], EXPECTED_A0_RMSE, rtol=0.0, atol=1e-15
    )
    checks: dict[str, Any] = {}
    for strategy, prediction in predictions.items():
        recomputed = _promotion_audit(
            target=target,
            fold_ids=fold_ids,
            a0=a0,
            candidate=prediction,
        )
        reported = summary["strategies"][strategy]["promotion"]
        np.testing.assert_allclose(
            recomputed["metrics"]["rmse"],
            reported["metrics"]["rmse"],
            rtol=0.0,
            atol=1e-15,
        )
        checks[strategy] = recomputed
    trial_rows = [
        json.loads(line)
        for line in trial_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(trial_rows) != 3 * len(base.FOLD_IDS) * TRIALS_PER_STRATEGY:
        raise RuntimeError("P28 trial count is not 3x5x4")
    for strategy in ("A2L", "A2D", "A3"):
        for fold in base.FOLD_IDS:
            selected = [
                row
                for row in trial_rows
                if row["strategy"] == strategy and row["held_fold"] == fold
            ]
            if len(selected) != TRIALS_PER_STRATEGY:
                raise RuntimeError("P28 per-strategy/fold budget drift")
            if len({row["action_id"] for row in selected}) != TRIALS_PER_STRATEGY:
                raise RuntimeError("P28 strategy repeated an action within a fold")
    if any(
        "random_init" in json.dumps(row).lower()
        for row in trial_rows
        if row["strategy"] == "A3"
    ):
        raise RuntimeError("A3 was contaminated by random-init foundation")
    verification = {
        "schema_version": "reconstruction-p28-verification/v1",
        "status": "PASSED",
        "summary_sha256": _sha256(summary_path),
        "protocol_sha256": _sha256(protocol_path),
        "prediction_artifact_sha256": _sha256(prediction_path),
        "trial_artifact_sha256": _sha256(trial_path),
        "a0_a1_replay_identity": True,
        "strategy_metrics_recomputed": checks,
        "trial_rows": len(trial_rows),
        "deepseek_route_gate_passed": bool(
            summary["independent_route_gate"]["passed"]
        ),
        "deepseek_selection_rounds": len(summary["a2l_provider_rounds"]),
        "firewall": summary["firewall"],
    }
    _write_json(output_dir / "verification.json", verification)
    return verification


def write_artifact_manifest(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = _validate_output_dir(output_dir)
    test_path = HERE / "_tests" / "test_p28_agentic_optimization.py"
    paths = [
        Path(__file__).resolve(),
        test_path,
        output_dir / "protocol.json",
        output_dir / "route_gate.json",
        output_dir / "trials.jsonl",
        output_dir / "predictions.npz",
        output_dir / "summary.json",
        output_dir / "evidence.md",
        output_dir / "verification.json",
    ]
    manifest = {
        "schema_version": "reconstruction-p28-artifact-manifest/v1",
        "artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    execute.add_argument("--stage3-root", type=Path, default=DEFAULT_STAGE3_ROOT)
    execute.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    execute.add_argument(
        "--credential-helper", type=Path, default=DEFAULT_CREDENTIAL_HELPER
    )
    execute.add_argument(
        "--recompute",
        action="store_true",
        help="overwrite only this P28 canonical output after an explicit recomputation",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        verification = verify_evidence(args.output_dir)
        write_artifact_manifest(args.output_dir)
        print(json.dumps(verification, sort_keys=True))
        return 0
    result = run(
        data_dir=args.data_dir,
        stage3_root=args.stage3_root,
        output_dir=args.output_dir,
        credential_helper=args.credential_helper,
        recompute=args.recompute,
    )
    verification = verify_evidence(args.output_dir)
    write_artifact_manifest(args.output_dir)
    print(
        json.dumps(
            {
                "a0_rmse": result["a0"]["metrics"]["rmse"],
                "a2l_rmse": result["strategies"]["A2L"]["promotion"]["metrics"]["rmse"],
                "a2d_rmse": result["strategies"]["A2D"]["promotion"]["metrics"]["rmse"],
                "a3_rmse": result["strategies"]["A3"]["promotion"]["metrics"]["rmse"],
                "retained_strategy": result["decision"]["retained_strategy"],
                "state": result["decision"]["state"],
                "verification": verification["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
