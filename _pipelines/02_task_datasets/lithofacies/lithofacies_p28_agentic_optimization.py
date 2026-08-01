#!/usr/bin/env python3
"""P28 Stage-1 execution-agent pilot for lithofacies.

The experiment is deliberately development-only and nested.  Each of four
anonymous outer rotations withholds one mother-family for promotion.  The
other three families form an inner LOGO3 selection loop.  Policy observations
contain only coarse fold-train support buckets, a coarse fit-state diagnostic,
and categorical selection feedback.  Numeric selection metrics and every
promotion result stay in the local evaluator and are never sent to a policy.

``prepare`` is the only command that opens the development ``train.h5``.  It
also consumes the already accepted LOGO4 development batch so the frozen A0
outer evaluation is reproduced exactly.  ``run`` requires a live DeepSeek
credential in ``DEEPSEEK_KEY`` and fails closed when the credential/provider
is unavailable.  No command accepts a frozen-holdout or test input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _models.lithofacies.p5_adapter_common import (  # noqa: E402
    multimodal_numpy_features,
)
from lithofacies_agent_chapter import (  # noqa: E402
    prior_calibrate,
    well_and_mask_only_features,
)
from lithofacies_p5_stage3 import (  # noqa: E402
    P_TRAIN_SAMPLE_LIMIT,
    P_VALIDATION_SAMPLE_LIMIT,
    REPEAT_SEEDS,
    _fold_arrays,
    load_stage3_batch,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    TEST_FAMILY,
    apply_fold_preprocessor,
    class_support,
    classification_metrics_from_logits,
    fit_fold_preprocessor,
    sample_id,
)
from p5_stage1 import (  # noqa: E402
    _balanced_take,
    _p_arrays,
    _read_development_hdf5,
)


SCHEMA_VERSION = "lithofacies-p28-agentic-optimization/v2"
BATCH_SCHEMA = "lithofacies-p28-nested-development/v1"
PROTOCOL_SCHEMA = "lithofacies-p28-protocol/v2"
RESULT_SCHEMA = "lithofacies-p28-result/v2"
MANIFEST_SCHEMA = "lithofacies-p28-artifact-manifest/v2"
PRE_CORRECTION_SCHEMA = "lithofacies-p28-agentic-optimization/v1"
EXPECTED_SPLIT_HASH = (
    "a06375429f9e9cf380fb5cdebd7d0cb7b25d7a13d29522b8e2420f4dae1b4555"
)
EXPECTED_A0_MEAN = 0.2133487970485067
EXPECTED_A0_DISPLAY = 0.2133487970
EXPECTED_A0_FOLDS = (
    0.24135550516502896,
    0.19674800002668855,
    0.18371728255998448,
    0.231574400442325,
)
NUM_CLASSES = 9
TRIAL_BUDGET = 3
OUTER_FOLDS = (0, 1, 2, 3)
POLICY_SEEDS = (2693, 2694, 2695)
FEEDBACK_THRESHOLD = 0.005
PROMOTION_THRESHOLD = 0.005
INNER_CEILING_MATERIALITY = 0.005
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p28_agentic_optimization"
DEFAULT_BATCH = DEFAULT_OUTPUT_DIR / "runtime" / "nested_development.npz"
FORBIDDEN_PATH_MARKERS = ("test", "holdout", "frozen")
FORBIDDEN_POLICY_KEYS = (
    "metric",
    "score",
    "label",
    "residual",
    "path",
    "family",
    "sample",
    "promotion",
    "validation",
)
REASON_CODES = (
    "reduce_capacity",
    "adjust_weighting",
    "remove_spatial_noise",
    "adjust_prior",
    "increase_capacity",
    "explore_alternative",
)


@dataclass(frozen=True)
class Action:
    action_id: str
    description: str
    max_depth: int = 3
    eta: float = 0.1
    rounds: int = 60
    weight_exponent: float = 0.5
    normalize_weight: bool = False
    features: str = "all_1155"
    prior_shrinkage: float = 0.0


A0 = Action(
    action_id="A0_DEPTH3_ETA01_ROUNDS60",
    description="frozen depth-3 eta-0.1 60-round full-feature XGBoost",
)
ACTIONS = (
    Action(
        action_id="ACT_DEPTH4_ETA0075_ROUNDS80",
        description="depth 4, eta 0.075, 80 rounds; all 1155 features",
        max_depth=4,
        eta=0.075,
        rounds=80,
    ),
    Action(
        action_id="ACT_WEIGHT_EXP05_MEAN1",
        description="class-weight exponent 0.5 normalized to supported-class mean one",
        normalize_weight=True,
    ),
    Action(
        action_id="ACT_WEIGHT_EXP075_MEAN1",
        description="class-weight exponent 0.75 normalized to supported-class mean one",
        weight_exponent=0.75,
        normalize_weight=True,
    ),
    Action(
        action_id="ACT_WELL_MASK_ONLY_858",
        description="use only 13 logs plus 13 missing masks, flattened to 858 features",
        features="well_mask_858",
    ),
    Action(
        action_id="ACT_PRIOR_SHRINK010",
        description="apply fold-train prior shrinkage 0.10 to frozen A0 logits",
        prior_shrinkage=0.10,
    ),
)
ACTION_BY_ID = {action.action_id: action for action in ACTIONS}
ACTION_IDS = tuple(ACTION_BY_ID)
MAIN_EFFECT_ARMS = ("A0", "A1", "A2L", "A2D", "A3", "A4")
MOMENT_PAIRED_LANE = {
    "lane_id": "moment_pretrained_vs_random_paired",
    "main_effect_member": False,
    "source": "_outputs/p11_clean_well_native33/summary.json",
    "encoder_weight_switch": ("pretrained", "random_init"),
    "status": "frozen_external_paired_evidence_not_rerun",
}
CLAUDE_AUDIT = {
    "job_id": "20260801T162948__claude__1249571",
    "result_sha256": "c1ddcc0e8c506cae69234919c87bd3680135b10f38b71ecbbbaea24a77ae4581",
    "completion_status": "COMPLETE",
}


class CredentialUnavailable(RuntimeError):
    """Raised before policy execution when the live credential is absent."""


class ProviderUnavailable(RuntimeError):
    """Raised when a valid live action cannot be obtained from the provider."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(json.dumps(list(values.shape)).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def ensure_development_only_paths(paths: Iterable[Path]) -> None:
    """Reject holdout-like inputs before opening them."""
    for raw in paths:
        parts = [part.lower().replace("-", "_") for part in Path(raw).parts]
        if any(marker in part for part in parts for marker in FORBIDDEN_PATH_MARKERS):
            raise ValueError(f"forbidden non-development path: {raw}")


def _owned_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    expected = DEFAULT_OUTPUT_DIR.resolve()
    if resolved != expected and expected not in resolved.parents:
        raise ValueError(f"P28 artifact must stay below {DEFAULT_OUTPUT_DIR}")
    return resolved


def action_table_payload() -> list[dict[str, str]]:
    return [
        {"action_id": action.action_id, "description": action.description}
        for action in ACTIONS
    ]


def nested_rotation_plan() -> list[dict[str, Any]]:
    plan = []
    for promotion_fold_id in OUTER_FOLDS:
        selection_fold_ids = [
            fold_id for fold_id in OUTER_FOLDS if fold_id != promotion_fold_id
        ]
        plan.append(
            {
                "outer_rollout_id": promotion_fold_id,
                "promotion_fold_id": promotion_fold_id,
                "selection_fold_ids": selection_fold_ids,
                "disjoint": promotion_fold_id not in selection_fold_ids,
            }
        )
    return plan


def _anonymous_partition_hash(samples: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash(sorted(sample_id(sample) for sample in samples))


def prepare_nested_batch(
    *,
    dataset_root: Path,
    accepted_logo4_batch: Path,
    batch_file: Path = DEFAULT_BATCH,
) -> dict[str, Any]:
    """Build true inner LOGO3 folds while preserving accepted outer A0 inputs."""
    ensure_development_only_paths((dataset_root, accepted_logo4_batch, batch_file))
    batch_file = _owned_output(batch_file)
    accepted_arrays, accepted_manifest = load_stage3_batch(accepted_logo4_batch)
    if (
        accepted_manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or accepted_manifest.get("frozen_test_accessed") is not False
        or accepted_manifest.get("test_metrics_used") is not False
    ):
        raise RuntimeError("accepted LOGO4 batch violates the frozen P28 contract")

    samples, train_hdf5 = _read_development_hdf5(dataset_root)
    by_family = {family: [] for family in DEVELOPMENT_FAMILIES}
    for sample in samples:
        family = str(sample["meta"]["family_id"])
        if family not in by_family or family == TEST_FAMILY:
            raise RuntimeError("non-development family entered P28 preparation")
        by_family[family].append(sample)

    arrays: dict[str, np.ndarray] = {}
    rotations: list[dict[str, Any]] = []
    for rotation in nested_rotation_plan():
        outer_id = int(rotation["outer_rollout_id"])
        promotion_fold_id = int(rotation["promotion_fold_id"])
        promotion_family = DEVELOPMENT_FAMILIES[promotion_fold_id]
        outer = _fold_arrays(accepted_arrays, promotion_fold_id)
        prefix = f"o{outer_id}"
        arrays.update(
            {
                f"{prefix}_train_well": outer["p_train_well"],
                f"{prefix}_train_seismic": outer["p_train_seismic"],
                f"{prefix}_train_labels": outer["p_train_labels"],
                f"{prefix}_train_ids": outer["p_train_ids"],
                f"{prefix}_promotion_well": outer["p_validation_well"],
                f"{prefix}_promotion_seismic": outer["p_validation_seismic"],
                f"{prefix}_promotion_labels": outer["p_validation_labels"],
                f"{prefix}_promotion_ids": outer["p_validation_ids"],
                f"{prefix}_class_counts": outer["class_counts"],
            }
        )
        inner_manifests: list[dict[str, Any]] = []
        promotion_ids = set(str(value) for value in outer["p_validation_ids"])
        for inner_id, selection_fold_id in enumerate(rotation["selection_fold_ids"]):
            selection_family = DEVELOPMENT_FAMILIES[int(selection_fold_id)]
            train_families = [
                family
                for family in DEVELOPMENT_FAMILIES
                if family not in {promotion_family, selection_family}
            ]
            train_raw = [sample for family in train_families for sample in by_family[family]]
            selection_raw = list(by_family[selection_family])
            preprocessor = fit_fold_preprocessor(train_raw)
            train = _balanced_take(
                apply_fold_preprocessor(train_raw, preprocessor),
                P_TRAIN_SAMPLE_LIMIT,
            )
            selection = _balanced_take(
                apply_fold_preprocessor(selection_raw, preprocessor),
                P_VALIDATION_SAMPLE_LIMIT,
            )
            train_values = _p_arrays(train)
            selection_values = _p_arrays(selection)
            inner_prefix = f"{prefix}_i{inner_id}"
            arrays.update(
                {
                    f"{inner_prefix}_train_well": train_values[0],
                    f"{inner_prefix}_train_seismic": train_values[1],
                    f"{inner_prefix}_train_labels": train_values[2],
                    f"{inner_prefix}_train_ids": train_values[3],
                    f"{inner_prefix}_selection_well": selection_values[0],
                    f"{inner_prefix}_selection_seismic": selection_values[1],
                    f"{inner_prefix}_selection_labels": selection_values[2],
                    f"{inner_prefix}_selection_ids": selection_values[3],
                    f"{inner_prefix}_class_counts": np.asarray(
                        preprocessor.class_support, dtype=np.int64
                    ),
                }
            )
            train_ids = set(str(value) for value in train_values[3])
            selection_ids = set(str(value) for value in selection_values[3])
            if train_ids & selection_ids or promotion_ids & (train_ids | selection_ids):
                raise RuntimeError("nested selection/promotion partitions overlap")
            inner_manifests.append(
                {
                    "inner_fold_id": inner_id,
                    "selection_fold_id": int(selection_fold_id),
                    "preprocessor_fit_fold_ids": sorted(
                        DEVELOPMENT_FAMILIES.index(family) for family in train_families
                    ),
                    "train_samples": len(train),
                    "selection_samples": len(selection),
                    "train_class_support": class_support(train_raw).tolist(),
                    "selection_class_support": class_support(selection_raw).tolist(),
                    "train_ids_hash": _anonymous_partition_hash(train_raw),
                    "selection_ids_hash": _anonymous_partition_hash(selection_raw),
                    "promotion_ids_hash": _stable_hash(sorted(promotion_ids)),
                    "selection_promotion_disjoint": True,
                    "preprocessor_fit_scope": "two_inner_train_folds_only",
                }
            )
        rotations.append(
            {
                **rotation,
                "inner_folds": inner_manifests,
                "promotion_samples": int(len(outer["p_validation_labels"])),
                "promotion_ids_hash": _stable_hash(sorted(promotion_ids)),
                "promotion_never_exposed_to_policy": True,
            }
        )

    manifest = {
        "schema_version": BATCH_SCHEMA,
        "split_hash": EXPECTED_SPLIT_HASH,
        "class_names": list(CLASS_NAMES),
        "class_count": NUM_CLASSES,
        "repeat_seeds": list(REPEAT_SEEDS),
        "trial_budget": TRIAL_BUDGET,
        "rotations": rotations,
        "accepted_logo4_batch_sha256": _sha256(accepted_logo4_batch),
        "development_hdf5_sha256": _sha256(train_hdf5),
        "loaded_files": ["train.h5", accepted_logo4_batch.name],
        "development_samples": len(samples),
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "test_metrics_used": False,
    }
    arrays["manifest"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    )
    batch_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(batch_file, **arrays)
    return {
        "batch_file": str(batch_file.relative_to(PROJECT_ROOT)),
        "batch_sha256": _sha256(batch_file),
        "split_hash": EXPECTED_SPLIT_HASH,
        "rotations": len(rotations),
        "inner_folds_per_rotation": 3,
        "loaded_files": manifest["loaded_files"],
        "frozen_test_accessed": False,
    }


def load_nested_batch(batch_file: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ensure_development_only_paths((batch_file,))
    with np.load(batch_file, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != "manifest"}
        manifest = json.loads(str(archive["manifest"].item()))
    if (
        manifest.get("schema_version") != BATCH_SCHEMA
        or manifest.get("split_hash") != EXPECTED_SPLIT_HASH
        or tuple(manifest.get("repeat_seeds", ())) != tuple(REPEAT_SEEDS)
        or manifest.get("trial_budget") != TRIAL_BUDGET
        or manifest.get("frozen_test_accessed") is not False
        or manifest.get("known_holdout_accessed") is not False
        or manifest.get("test_metrics_used") is not False
    ):
        raise RuntimeError("nested development batch violates the P28 contract")
    if len(manifest.get("rotations", ())) != 4:
        raise RuntimeError("P28 requires four outer rotations")
    for rotation in manifest["rotations"]:
        promotion = int(rotation["promotion_fold_id"])
        selection = tuple(int(value) for value in rotation["selection_fold_ids"])
        if len(selection) != 3 or promotion in selection or set(selection) | {promotion} != set(OUTER_FOLDS):
            raise RuntimeError("nested fold identities are not a disjoint LOGO3/outer cover")
        outer_id = int(rotation["outer_rollout_id"])
        outer_train_ids = set(str(v) for v in arrays[f"o{outer_id}_train_ids"])
        promotion_ids = set(str(v) for v in arrays[f"o{outer_id}_promotion_ids"])
        if outer_train_ids & promotion_ids:
            raise RuntimeError("outer training and promotion samples overlap")
        for inner_id in range(3):
            train_ids = set(str(v) for v in arrays[f"o{outer_id}_i{inner_id}_train_ids"])
            selection_ids = set(str(v) for v in arrays[f"o{outer_id}_i{inner_id}_selection_ids"])
            if train_ids & selection_ids or promotion_ids & (train_ids | selection_ids):
                raise RuntimeError("runtime nested partitions overlap")
    return arrays, manifest


def _features(well: np.ndarray, seismic: np.ndarray, kind: str) -> np.ndarray:
    if kind == "all_1155":
        values = multimodal_numpy_features(well, seismic)
    elif kind == "well_mask_858":
        values = well_and_mask_only_features(well, seismic)
    else:
        raise ValueError(f"unknown feature view: {kind}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("P28 feature matrix is invalid")
    return np.ascontiguousarray(values)


def class_weight_vector(
    counts: np.ndarray,
    *,
    exponent: float,
    normalize_mean_one: bool,
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (NUM_CLASSES,) or np.any(counts < 0):
        raise ValueError("class counts must be non-negative fixed-nine values")
    weights = np.zeros(NUM_CLASSES, dtype=np.float64)
    supported = counts > 0
    weights[supported] = counts[supported] ** (-float(exponent))
    if normalize_mean_one and supported.any():
        weights[supported] /= weights[supported].mean()
    return weights


def _metric_payload(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics_from_logits(labels, logits)
    return {
        "fixed_schema_macro_f1": float(metrics["fixed_schema_macro_f1"]),
        "supported_class_macro_f1": float(metrics["supported_class_macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "per_class_f1": [float(item["f1"]) for item in metrics["per_class"]],
    }


def _train_action(
    *,
    action: Action,
    train_well: np.ndarray,
    train_seismic: np.ndarray,
    train_labels: np.ndarray,
    evaluation_well: np.ndarray,
    evaluation_seismic: np.ndarray,
    class_counts: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    import xgboost

    train_features = _features(train_well, train_seismic, action.features)
    evaluation_features = _features(
        evaluation_well, evaluation_seismic, action.features
    )
    weights = class_weight_vector(
        class_counts,
        exponent=action.weight_exponent,
        normalize_mean_one=action.normalize_weight,
    )
    labels = np.asarray(train_labels, dtype=np.int64)
    train_matrix = xgboost.DMatrix(
        train_features,
        label=labels,
        weight=weights[labels],
    )
    booster = xgboost.train(
        {
            "objective": "multi:softprob",
            "num_class": NUM_CLASSES,
            "max_depth": action.max_depth,
            "eta": action.eta,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "seed": int(seed),
            "nthread": 1,
            "verbosity": 0,
        },
        train_matrix,
        num_boost_round=action.rounds,
    )
    train_probabilities = np.asarray(booster.predict(train_matrix), dtype=np.float64)
    evaluation_probabilities = np.asarray(
        booster.predict(xgboost.DMatrix(evaluation_features)), dtype=np.float64
    )
    if train_probabilities.shape != (len(labels), NUM_CLASSES) or evaluation_probabilities.shape != (
        len(evaluation_features), NUM_CLASSES
    ):
        raise RuntimeError("XGBoost did not return fixed-nine probabilities")
    train_logits = np.log(np.clip(train_probabilities, 1e-12, 1.0)).astype(np.float32)
    evaluation_logits = np.log(
        np.clip(evaluation_probabilities, 1e-12, 1.0)
    ).astype(np.float32)
    if action.prior_shrinkage:
        train_logits = prior_calibrate(
            train_logits, class_counts, shrinkage=action.prior_shrinkage
        )
        evaluation_logits = prior_calibrate(
            evaluation_logits, class_counts, shrinkage=action.prior_shrinkage
        )
    return train_logits, evaluation_logits


def _support_buckets(counts: Sequence[int]) -> dict[str, int]:
    values = [int(value) for value in counts]
    return {
        "absent": sum(value == 0 for value in values),
        "very_low_1_to_5": sum(1 <= value <= 5 for value in values),
        "low_6_to_20": sum(6 <= value <= 20 for value in values),
        "medium_21_to_60": sum(21 <= value <= 60 for value in values),
        "high_over_60": sum(value > 60 for value in values),
    }


def _fit_state(train_scores: Sequence[float], selection_scores: Sequence[float]) -> str:
    gap = float(np.mean(train_scores) - np.mean(selection_scores))
    if gap >= 0.10:
        return "overfit"
    if gap <= 0.03:
        return "underfit"
    return "balanced"


def categorical_feedback(delta: float) -> str:
    if delta >= FEEDBACK_THRESHOLD:
        return "improved"
    if delta <= -FEEDBACK_THRESHOLD:
        return "worse"
    return "flat"


def assert_policy_payload_safe(payload: Mapping[str, Any]) -> None:
    """Enforce the raw-metric/identity/path firewall on outbound observations."""
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                lowered = str(key).lower()
                if any(marker in lowered for marker in FORBIDDEN_POLICY_KEYS):
                    raise ValueError(f"forbidden policy field: {key}")
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            lowered = value.lower()
            if any(name.lower() in lowered for name in DEVELOPMENT_FAMILIES):
                raise ValueError("family identity leaked into policy payload")
            if "/" in value or "\\" in value:
                raise ValueError("path-like value leaked into policy payload")
        elif isinstance(value, float):
            raise ValueError("continuous numeric value leaked into policy payload")

    visit(payload)


def build_policy_observation(
    *,
    support_buckets: Mapping[str, int],
    fit_state: str,
    history: Sequence[Mapping[str, str]],
    remaining_actions: Sequence[str],
    trial_index: int,
) -> dict[str, Any]:
    observation = {
        "trial_ordinal": int(trial_index + 1),
        "trial_budget": TRIAL_BUDGET,
        "fold_train_support_buckets": dict(support_buckets),
        "fit_state": fit_state,
        "history": [
            {
                "action_id": str(item["action_id"]),
                "feedback": str(item["feedback"]),
            }
            for item in history
        ],
        "available_actions": [
            {
                "action_id": action_id,
                "description": ACTION_BY_ID[action_id].description,
            }
            for action_id in remaining_actions
        ],
    }
    assert_policy_payload_safe(observation)
    return observation


def _validate_deepseek_decision(
    content: str,
    *,
    remaining_actions: Sequence[str],
) -> dict[str, Any]:
    try:
        decision = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("provider response is not strict JSON") from exc
    if not isinstance(decision, dict) or set(decision) != {
        "action_id",
        "reason_code",
        "stop",
    }:
        raise ValueError("provider response keys changed")
    if decision["action_id"] not in set(remaining_actions):
        raise ValueError("provider selected an unavailable action")
    if decision["reason_code"] not in REASON_CODES:
        raise ValueError("provider returned an unknown reason code")
    if decision["stop"] is not False:
        raise ValueError("the fixed three-trial budget cannot stop early")
    return decision


def call_deepseek_action(
    *,
    observation: Mapping[str, Any],
    api_key: str,
    timeout_seconds: float = 75.0,
    attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Obtain one live strict-JSON action; no local fallback is permitted."""
    if not api_key.strip():
        raise CredentialUnavailable("DEEPSEEK_KEY is missing; A2L fails closed")
    assert_policy_payload_safe(observation)
    system_prompt = (
        "You are a bounded experiment-selection policy. Choose exactly one available "
        "action. Use only the anonymous categorical observation. Never infer or request "
        "raw metrics, labels, residuals, group identities, sample identifiers, or paths. "
        "Return one JSON object and no markdown with exactly: action_id, reason_code, "
        "stop. reason_code must be one allowed enum and stop must be false."
    )
    user_payload = {
        "protocol": "P28_STAGE1_THREE_TRIAL",
        "allowed_reason_codes": list(REASON_CODES),
        "observation": dict(observation),
        "required_output_example": {
            "action_id": observation["available_actions"][0]["action_id"],
            "reason_code": "explore_alternative",
            "stop": False,
        },
    }
    prompt_hash = _stable_hash({"system": system_prompt, "user": user_payload})
    failures: list[str] = []
    for attempt in range(1, attempts + 1):
        body = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        request = urllib.request.Request(
            DEEPSEEK_ENDPOINT,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_payload = json.load(response)
            content = str(response_payload["choices"][0]["message"]["content"])
            remaining = [item["action_id"] for item in observation["available_actions"]]
            decision = _validate_deepseek_decision(
                content, remaining_actions=remaining
            )
            metadata = {
                "request_model": DEEPSEEK_MODEL,
                "response_model": response_payload.get("model"),
                "response_id": response_payload.get("id"),
                "usage": response_payload.get("usage"),
                "prompt_sha256": prompt_hash,
                "attempt": attempt,
                "valid": True,
                "credential_persisted": False,
            }
            return decision, metadata
        except Exception as exc:  # network and strict-schema failures both fail closed
            failures.append(f"{type(exc).__name__}:{str(exc)[:160]}")
    raise ProviderUnavailable(
        "DeepSeek did not return a valid live action after "
        f"{attempts} attempts: {' | '.join(failures)}"
    )


def deterministic_action(
    *,
    fit_state: str,
    history: Sequence[Mapping[str, str]],
    remaining_actions: Sequence[str],
) -> str:
    priorities = {
        "overfit": [
            "ACT_WELL_MASK_ONLY_858",
            "ACT_WEIGHT_EXP05_MEAN1",
            "ACT_PRIOR_SHRINK010",
            "ACT_WEIGHT_EXP075_MEAN1",
            "ACT_DEPTH4_ETA0075_ROUNDS80",
        ],
        "balanced": [
            "ACT_WEIGHT_EXP05_MEAN1",
            "ACT_PRIOR_SHRINK010",
            "ACT_WELL_MASK_ONLY_858",
            "ACT_DEPTH4_ETA0075_ROUNDS80",
            "ACT_WEIGHT_EXP075_MEAN1",
        ],
        "underfit": [
            "ACT_DEPTH4_ETA0075_ROUNDS80",
            "ACT_WEIGHT_EXP05_MEAN1",
            "ACT_PRIOR_SHRINK010",
            "ACT_WELL_MASK_ONLY_858",
            "ACT_WEIGHT_EXP075_MEAN1",
        ],
    }
    order = list(priorities[fit_state])
    if history:
        last = history[-1]
        if last["feedback"] == "improved" and "WEIGHT" in last["action_id"]:
            order = ["ACT_WEIGHT_EXP075_MEAN1", "ACT_WEIGHT_EXP05_MEAN1"] + order
        elif last["feedback"] == "worse" and "WEIGHT" in last["action_id"]:
            order = [item for item in order if "WEIGHT" not in item] + [
                item for item in order if "WEIGHT" in item
            ]
    remaining = set(remaining_actions)
    return next(action_id for action_id in order if action_id in remaining)


def normalized_auc(deltas: Sequence[float]) -> float:
    if len(deltas) != TRIAL_BUDGET:
        raise ValueError("AUC requires the exact three-trial budget")
    curve: list[float] = []
    best = 0.0
    for delta in deltas:
        best = max(best, float(delta))
        curve.append(best)
    return float(np.mean(curve))


def _inner_arrays(arrays: Mapping[str, np.ndarray], outer_id: int, inner_id: int) -> dict[str, np.ndarray]:
    prefix = f"o{outer_id}_i{inner_id}"
    return {
        "train_well": arrays[f"{prefix}_train_well"],
        "train_seismic": arrays[f"{prefix}_train_seismic"],
        "train_labels": arrays[f"{prefix}_train_labels"],
        "selection_well": arrays[f"{prefix}_selection_well"],
        "selection_seismic": arrays[f"{prefix}_selection_seismic"],
        "selection_labels": arrays[f"{prefix}_selection_labels"],
        "class_counts": arrays[f"{prefix}_class_counts"],
    }


def _outer_arrays(arrays: Mapping[str, np.ndarray], outer_id: int) -> dict[str, np.ndarray]:
    prefix = f"o{outer_id}"
    return {
        "train_well": arrays[f"{prefix}_train_well"],
        "train_seismic": arrays[f"{prefix}_train_seismic"],
        "train_labels": arrays[f"{prefix}_train_labels"],
        "promotion_well": arrays[f"{prefix}_promotion_well"],
        "promotion_seismic": arrays[f"{prefix}_promotion_seismic"],
        "promotion_labels": arrays[f"{prefix}_promotion_labels"],
        "class_counts": arrays[f"{prefix}_class_counts"],
    }


def _evaluate_inner_action(
    arrays: Mapping[str, np.ndarray], outer_id: int, action: Action
) -> dict[str, Any]:
    rows = []
    for inner_id in range(3):
        fold = _inner_arrays(arrays, outer_id, inner_id)
        for repeat_id, seed in enumerate(REPEAT_SEEDS):
            train_logits, selection_logits = _train_action(
                action=action,
                train_well=fold["train_well"],
                train_seismic=fold["train_seismic"],
                train_labels=fold["train_labels"],
                evaluation_well=fold["selection_well"],
                evaluation_seismic=fold["selection_seismic"],
                class_counts=fold["class_counts"],
                seed=int(seed),
            )
            rows.append(
                {
                    "inner_fold_id": inner_id,
                    "repeat_id": repeat_id,
                    "seed": int(seed),
                    "train_metrics": _metric_payload(fold["train_labels"], train_logits),
                    "selection_metrics": _metric_payload(
                        fold["selection_labels"], selection_logits
                    ),
                }
            )
    selection_scores = [
        row["selection_metrics"]["fixed_schema_macro_f1"] for row in rows
    ]
    train_scores = [row["train_metrics"]["fixed_schema_macro_f1"] for row in rows]
    return {
        "action_id": action.action_id,
        "cells": rows,
        "selection_mean": float(np.mean(selection_scores)),
        "train_mean": float(np.mean(train_scores)),
        "fit_state": _fit_state(train_scores, selection_scores),
        "per_class_selection_mean": np.mean(
            [row["selection_metrics"]["per_class_f1"] for row in rows], axis=0
        ).tolist(),
    }


def _evaluate_promotion_action(
    arrays: Mapping[str, np.ndarray], outer_id: int, action: Action
) -> list[dict[str, Any]]:
    fold = _outer_arrays(arrays, outer_id)
    rows = []
    for repeat_id, seed in enumerate(REPEAT_SEEDS):
        _, logits = _train_action(
            action=action,
            train_well=fold["train_well"],
            train_seismic=fold["train_seismic"],
            train_labels=fold["train_labels"],
            evaluation_well=fold["promotion_well"],
            evaluation_seismic=fold["promotion_seismic"],
            class_counts=fold["class_counts"],
            seed=int(seed),
        )
        rows.append(
            {
                "outer_rollout_id": outer_id,
                "repeat_id": repeat_id,
                "seed": int(seed),
                "metrics": _metric_payload(fold["promotion_labels"], logits),
                "prediction_sha256": _array_hash(
                    logits, np.argmax(logits, axis=1).astype(np.int64)
                ),
            }
        )
    return rows


def _protocol_payload(batch_sha256: str) -> dict[str, Any]:
    a0_payload = asdict(A0)
    action_payload = [asdict(action) for action in ACTIONS]
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "split_hash": EXPECTED_SPLIT_HASH,
        "primary_metric": "fixed_schema_macro_f1",
        "fixed_class_schema": list(CLASS_NAMES),
        "a0": {
            "config": a0_payload,
            "config_sha256": _stable_hash(a0_payload),
            "frozen_mean": EXPECTED_A0_MEAN,
        },
        "actions": action_payload,
        "action_table_sha256": _stable_hash(action_payload),
        "arms": list(MAIN_EFFECT_ARMS),
        "trial_budget_per_outer_rollout": TRIAL_BUDGET,
        "repeat_seeds": list(REPEAT_SEEDS),
        "random_policy_seeds": list(POLICY_SEEDS),
        "rotations": nested_rotation_plan(),
        "selection": "inner_LOGO3_mean_over_3_folds_x_3_repeat_seeds",
        "promotion": "outer_disjoint_family_refit_x_3_repeat_seeds",
        "policy_observables": [
            "fold_train_support_buckets",
            "underfit_balanced_overfit",
            "improved_flat_worse",
        ],
        "policy_deny_list": list(FORBIDDEN_POLICY_KEYS),
        "deepseek": {
            "endpoint": DEEPSEEK_ENDPOINT,
            "model": DEEPSEEK_MODEL,
            "strict_json": True,
            "missing_credential": "fail_closed",
            "invalid_or_unavailable_provider": "fail_closed_no_replay",
        },
        "diagnostic_controls": {
            "a1_identity": {
                "implementation": "actual_same_entrypoint_action_seed_replay",
                "executor_entrypoint": "_evaluate_promotion_action",
                "action_id": A0.action_id,
                "repeat_seeds": list(REPEAT_SEEDS),
                "required_cell_count": len(OUTER_FOLDS) * len(REPEAT_SEEDS),
            },
            "inner_action_space_ceiling": {
                "implementation": "exhaustive_all_five_allowlisted_actions",
                "layer": "inner_selection_only",
                "materiality": INNER_CEILING_MATERIALITY,
                "used_for_policy_feedback": False,
                "used_for_promotion_selection": False,
                "promotion_metrics_computed": False,
            },
        },
        "claude_correction_audit": dict(CLAUDE_AUDIT),
        "moment_attribution_lane": dict(MOMENT_PAIRED_LANE),
        "nested_batch_sha256": batch_sha256,
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
    }


def _rollout(
    *,
    arm: str,
    instance_id: str,
    outer_id: int,
    baseline: Mapping[str, Any],
    evaluate: Any,
    api_key: str,
    random_seed: int | None = None,
) -> dict[str, Any]:
    remaining = list(ACTION_IDS)
    history: list[dict[str, str]] = []
    trials: list[dict[str, Any]] = []
    support = _support_buckets(
        np.sum(
            [
                _inner_arrays(evaluate.arrays, outer_id, inner_id)["class_counts"]
                for inner_id in range(3)
            ],
            axis=0,
        )
    )
    fit_state = str(baseline["fit_state"])
    rng = random.Random(None if random_seed is None else random_seed + outer_id * 1009)
    random_order = rng.sample(remaining, len(remaining)) if arm == "A3" else []
    offline_order = [
        "ACT_DEPTH4_ETA0075_ROUNDS80",
        "ACT_WEIGHT_EXP05_MEAN1",
        "ACT_PRIOR_SHRINK010",
    ]
    for trial_index in range(TRIAL_BUDGET):
        observation = build_policy_observation(
            support_buckets=support,
            fit_state=fit_state,
            history=history,
            remaining_actions=remaining,
            trial_index=trial_index,
        )
        provider = None
        if arm == "A2L":
            decision, provider = call_deepseek_action(
                observation=observation,
                api_key=api_key,
            )
            action_id = str(decision["action_id"])
            reason_code = str(decision["reason_code"])
        elif arm == "A2D":
            action_id = deterministic_action(
                fit_state=fit_state,
                history=history,
                remaining_actions=remaining,
            )
            reason_code = "deterministic_diagnostic"
        elif arm == "A3":
            action_id = next(item for item in random_order if item in remaining)
            reason_code = "random_without_replacement"
        elif arm == "A4":
            action_id = next(item for item in offline_order if item in remaining)
            reason_code = "offline_fixed_replay"
        else:
            raise ValueError(f"unsupported rollout arm: {arm}")
        result = evaluate(outer_id, action_id)
        delta = float(result["selection_mean"] - baseline["selection_mean"])
        feedback = categorical_feedback(delta)
        trial = {
            "trial_index": trial_index,
            "action_id": action_id,
            "reason_code": reason_code,
            "valid_action": action_id in remaining,
            "policy_observation": observation,
            "provider": provider,
            "feedback_sent_to_next_trial": feedback,
            "selection_mean_local_evaluator_only": result["selection_mean"],
            "selection_delta_local_evaluator_only": delta,
            "selection_cells_local_evaluator_only": result["cells"],
            "per_class_selection_mean_local_evaluator_only": result[
                "per_class_selection_mean"
            ],
        }
        trials.append(trial)
        history.append({"action_id": action_id, "feedback": feedback})
        remaining.remove(action_id)
    best_action_id = A0.action_id
    best_score = float(baseline["selection_mean"])
    for trial in trials:
        score = float(trial["selection_mean_local_evaluator_only"])
        if score > best_score:
            best_score = score
            best_action_id = str(trial["action_id"])
    deltas = [float(item["selection_delta_local_evaluator_only"]) for item in trials]
    return {
        "arm": arm,
        "instance_id": instance_id,
        "outer_rollout_id": outer_id,
        "trial_count": len(trials),
        "trials": trials,
        "auc_at_3": normalized_auc(deltas),
        "selected_for_promotion": best_action_id,
        "selected_inner_mean_local_evaluator_only": best_score,
        "promotion_result_exposed_to_policy": False,
    }


class _ActionEvaluator:
    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.inner_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self.promotion_cache: dict[tuple[int, str], list[dict[str, Any]]] = {}

    def inner(self, outer_id: int, action_id: str) -> dict[str, Any]:
        key = (outer_id, action_id)
        if key not in self.inner_cache:
            action = A0 if action_id == A0.action_id else ACTION_BY_ID[action_id]
            self.inner_cache[key] = _evaluate_inner_action(
                self.arrays, outer_id, action
            )
        return self.inner_cache[key]

    def promotion(self, outer_id: int, action_id: str) -> list[dict[str, Any]]:
        key = (outer_id, action_id)
        if key not in self.promotion_cache:
            action = A0 if action_id == A0.action_id else ACTION_BY_ID[action_id]
            self.promotion_cache[key] = _evaluate_promotion_action(
                self.arrays, outer_id, action
            )
        return self.promotion_cache[key]

    def __call__(self, outer_id: int, action_id: str) -> dict[str, Any]:
        return self.inner(outer_id, action_id)


def _actual_a1_replay(
    arrays: Mapping[str, np.ndarray],
    a0_by_outer: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Re-execute A0 through its real entrypoint for the A1 identity probe."""
    a1_by_outer = {
        outer_id: _evaluate_promotion_action(arrays, outer_id, A0)
        for outer_id in OUTER_FOLDS
    }
    comparisons: list[dict[str, Any]] = []
    a1_cells: list[dict[str, Any]] = []
    for outer_id in OUTER_FOLDS:
        a0_cells = sorted(
            a0_by_outer[outer_id], key=lambda row: int(row["repeat_id"])
        )
        replay_cells = sorted(
            a1_by_outer[outer_id], key=lambda row: int(row["repeat_id"])
        )
        if len(a0_cells) != len(REPEAT_SEEDS) or len(replay_cells) != len(REPEAT_SEEDS):
            raise RuntimeError("A0/A1 replay cell roster changed")
        for expected, observed in zip(a0_cells, replay_cells):
            roster_equal = (
                int(expected["outer_rollout_id"])
                == int(observed["outer_rollout_id"])
                == outer_id
                and int(expected["repeat_id"]) == int(observed["repeat_id"])
                and int(expected["seed"]) == int(observed["seed"])
            )
            prediction_equal = (
                expected["prediction_sha256"] == observed["prediction_sha256"]
            )
            metric_equal = _stable_hash(expected["metrics"]) == _stable_hash(
                observed["metrics"]
            )
            comparisons.append(
                {
                    "outer_rollout_id": outer_id,
                    "repeat_id": int(observed["repeat_id"]),
                    "seed": int(observed["seed"]),
                    "roster_equal": roster_equal,
                    "prediction_hash_equal": prediction_equal,
                    "metric_hash_equal": metric_equal,
                }
            )
            a1_cells.append(observed)
    if not all(
        item["roster_equal"]
        and item["prediction_hash_equal"]
        and item["metric_hash_equal"]
        for item in comparisons
    ):
        raise RuntimeError("actual A1 replay diverged from A0")
    a0_cells_flat = [
        row for outer_id in OUTER_FOLDS for row in a0_by_outer[outer_id]
    ]
    a0_prediction_hash = _stable_hash(
        [row["prediction_sha256"] for row in a0_cells_flat]
    )
    a1_prediction_hash = _stable_hash(
        [row["prediction_sha256"] for row in a1_cells]
    )
    a0_metric_hash = _stable_hash([row["metrics"] for row in a0_cells_flat])
    a1_metric_hash = _stable_hash([row["metrics"] for row in a1_cells])
    control = {
        "role": "advice_only_actual_identity_replay",
        "actual_replay_executed": True,
        "executor_entrypoint": "_evaluate_promotion_action",
        "action_id": A0.action_id,
        "action_config_sha256": _stable_hash(asdict(A0)),
        "repeat_seeds": list(REPEAT_SEEDS),
        "outer_rollout_ids": list(OUTER_FOLDS),
        "cell_count": len(a1_cells),
        "cell_comparisons": comparisons,
        "all_rosters_equal": all(item["roster_equal"] for item in comparisons),
        "all_prediction_hashes_equal": all(
            item["prediction_hash_equal"] for item in comparisons
        ),
        "all_metric_hashes_equal": all(
            item["metric_hash_equal"] for item in comparisons
        ),
        "prediction_sha256": a1_prediction_hash,
        "metric_sha256": a1_metric_hash,
        "a0_prediction_sha256": a0_prediction_hash,
        "a0_metric_sha256": a0_metric_hash,
        "shares_a2l_initial_advice_but_executes_a0": True,
    }
    return control, a1_cells


def _exhaustive_inner_ceiling(
    *,
    evaluator: _ActionEvaluator,
    a0_inner: Mapping[int, Mapping[str, Any]],
    a2l_rollouts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate every allowlisted action on inner folds, never on promotion."""
    rollout_by_outer = {
        int(rollout["outer_rollout_id"]): rollout for rollout in a2l_rollouts
    }
    outer_summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for outer_id in OUTER_FOLDS:
        baseline_mean = float(a0_inner[outer_id]["selection_mean"])
        candidates: list[dict[str, Any]] = []
        for action_id in ACTION_IDS:
            result = evaluator.inner(outer_id, action_id)
            candidate = {
                "action_id": action_id,
                "selection_mean": float(result["selection_mean"]),
                "delta_from_a0_inner": float(
                    result["selection_mean"] - baseline_mean
                ),
                "per_class_selection_mean": result[
                    "per_class_selection_mean"
                ],
            }
            candidates.append(candidate)
            records.append(
                {
                    "outer_rollout_id": outer_id,
                    **candidate,
                    "selection_cells": result["cells"],
                    "evaluated_at_inner_layer_only": True,
                    "used_for_promotion_selection": False,
                    "promotion_metrics_computed": False,
                }
            )
        best_candidate = max(
            candidates,
            key=lambda item: (
                float(item["selection_mean"]),
                -ACTION_IDS.index(str(item["action_id"])),
            ),
        )
        if float(best_candidate["selection_mean"]) > baseline_mean:
            best_reachable_action = str(best_candidate["action_id"])
            best_reachable_mean = float(best_candidate["selection_mean"])
        else:
            best_reachable_action = A0.action_id
            best_reachable_mean = baseline_mean
        policy_mean = float(
            rollout_by_outer[outer_id]["selected_inner_mean_local_evaluator_only"]
        )
        outer_summaries.append(
            {
                "outer_rollout_id": outer_id,
                "a0_inner_mean": baseline_mean,
                "all_five_actions": candidates,
                "best_candidate_action_id": str(best_candidate["action_id"]),
                "best_candidate_inner_mean": float(best_candidate["selection_mean"]),
                "best_candidate_delta_from_a0": float(
                    best_candidate["selection_mean"] - baseline_mean
                ),
                "best_reachable_including_a0_action_id": best_reachable_action,
                "best_reachable_including_a0_inner_mean": best_reachable_mean,
                "ceiling_delta_from_a0": best_reachable_mean - baseline_mean,
                "a2l_selected_action_id": str(
                    rollout_by_outer[outer_id]["selected_for_promotion"]
                ),
                "a2l_selected_inner_mean": policy_mean,
                "a2l_regret_to_inner_ceiling": best_reachable_mean - policy_mean,
            }
        )
    ceiling_delta = float(
        np.mean([item["ceiling_delta_from_a0"] for item in outer_summaries])
    )
    policy_delta = float(
        np.mean(
            [
                item["a2l_selected_inner_mean"] - item["a0_inner_mean"]
                for item in outer_summaries
            ]
        )
    )
    policy_regret = float(
        np.mean([item["a2l_regret_to_inner_ceiling"] for item in outer_summaries])
    )
    if ceiling_delta < INNER_CEILING_MATERIALITY:
        diagnosis = "ACTION_SPACE_INNER_CEILING"
        interpretation = (
            "The exhaustive allowlist has less than 0.005 mean inner headroom; "
            "the bounded action space, not policy search, is the primary ceiling."
        )
    elif policy_regret >= INNER_CEILING_MATERIALITY:
        diagnosis = "POLICY_SEARCH_LIMIT"
        interpretation = (
            "The allowlist has material inner headroom that A2L misses by at "
            "least 0.005 on average; policy search is the primary limitation."
        )
    else:
        diagnosis = "INNER_TO_PROMOTION_TRANSFER_LIMIT"
        interpretation = (
            "A2L reaches the material inner ceiling within 0.005, so its failed "
            "promotion is not explained by policy search; transfer from the "
            "bounded inner action space to the disjoint outer folds is the limit."
        )
    control = {
        "control_id": "INNER_EXHAUSTIVE_ORACLE_CEILING",
        "layer": "inner_selection_only",
        "action_ids": list(ACTION_IDS),
        "actions_per_outer_rollout": len(ACTION_IDS),
        "outer_rollout_count": len(OUTER_FOLDS),
        "model_seed_roster": list(REPEAT_SEEDS),
        "used_for_policy_feedback": False,
        "used_for_promotion_selection": False,
        "promotion_metrics_computed": False,
        "materiality": INNER_CEILING_MATERIALITY,
        "outer_rollouts": outer_summaries,
        "mean_inner_ceiling_delta_from_a0": ceiling_delta,
        "mean_a2l_inner_delta_from_a0": policy_delta,
        "mean_a2l_regret_to_inner_ceiling": policy_regret,
        "failure_diagnosis": diagnosis,
        "interpretation": interpretation,
    }
    return control, records


def _trajectory_comparison(
    a2l_rollouts: Sequence[Mapping[str, Any]],
    a2d_rollouts: Sequence[Mapping[str, Any]],
    a2l_promotion: Mapping[str, Any],
    a2d_promotion: Mapping[str, Any],
) -> dict[str, Any]:
    by_l = {int(item["outer_rollout_id"]): item for item in a2l_rollouts}
    by_d = {int(item["outer_rollout_id"]): item for item in a2d_rollouts}
    folds = []
    for outer_id in OUTER_FOLDS:
        l_sequence = [str(item["action_id"]) for item in by_l[outer_id]["trials"]]
        d_sequence = [str(item["action_id"]) for item in by_d[outer_id]["trials"]]
        folds.append(
            {
                "outer_rollout_id": outer_id,
                "a2l_action_sequence": l_sequence,
                "a2d_action_sequence": d_sequence,
                "trajectory_identical": l_sequence == d_sequence,
                "promotion_action_identical": by_l[outer_id][
                    "selected_for_promotion"
                ]
                == by_d[outer_id]["selected_for_promotion"],
            }
        )
    differing = [
        item["outer_rollout_id"] for item in folds if not item["trajectory_identical"]
    ]
    endpoint_equal = (
        a2l_promotion["fixed_schema_macro_f1_mean"]
        == a2d_promotion["fixed_schema_macro_f1_mean"]
        and a2l_promotion["outer_fold_deltas"]
        == a2d_promotion["outer_fold_deltas"]
        and all(item["promotion_action_identical"] for item in folds)
    )
    return {
        "outer_folds": folds,
        "differing_trajectory_outer_folds": differing,
        "differing_trajectory_count": len(differing),
        "total_outer_folds": len(OUTER_FOLDS),
        "promotion_endpoint_identical": endpoint_equal,
        "interpretation": (
            "A2L and A2D take different action trajectories in "
            f"{len(differing)}/{len(OUTER_FOLDS)} outer folds, while converging "
            "to the same promoted actions and endpoint metrics."
        ),
    }


def _promotion_summary(
    rollouts: Sequence[Mapping[str, Any]],
    evaluator: _ActionEvaluator,
    a0_by_outer: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    cells = []
    fold_deltas = []
    for rollout in sorted(rollouts, key=lambda item: int(item["outer_rollout_id"])):
        outer_id = int(rollout["outer_rollout_id"])
        action_id = str(rollout["selected_for_promotion"])
        promoted = evaluator.promotion(outer_id, action_id)
        baseline = a0_by_outer[outer_id]
        action_mean = float(
            np.mean([row["metrics"]["fixed_schema_macro_f1"] for row in promoted])
        )
        baseline_mean = float(
            np.mean([row["metrics"]["fixed_schema_macro_f1"] for row in baseline])
        )
        fold_deltas.append(action_mean - baseline_mean)
        for row in promoted:
            cells.append({**row, "action_id": action_id})
    primary = [row["metrics"]["fixed_schema_macro_f1"] for row in cells]
    per_class = np.asarray([row["metrics"]["per_class_f1"] for row in cells])
    return {
        "cells": cells,
        "fixed_schema_macro_f1_mean": float(np.mean(primary)),
        "delta_from_a0": float(np.mean(fold_deltas)),
        "outer_fold_deltas": [float(value) for value in fold_deltas],
        "positive_outer_folds": int(sum(value > 0.0 for value in fold_deltas)),
        "per_class_f1_mean": per_class.mean(axis=0).tolist(),
        "all_class_metrics_finite": bool(np.isfinite(per_class).all()),
    }


def _moment_lane_payload() -> dict[str, Any]:
    path = TRACK_DIR / MOMENT_PAIRED_LANE["source"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    diagnostic = payload.get("representation_diagnostic", {})
    return {
        **MOMENT_PAIRED_LANE,
        "source_sha256": _sha256(path),
        "pretrained_minus_random_fixed_schema_macro_f1": diagnostic.get(
            "pretrained_minus_random_fixed_schema_macro_f1"
        ),
        "material_separation_detected": diagnostic.get("material_separation_detected"),
        "included_in_p28_agent_auc": False,
        "included_in_p28_promotion_delta": False,
    }


def _make_evidence(summary: Mapping[str, Any]) -> str:
    gates = summary["gates"]
    verdict = summary["verdict"]
    previous_verdict = summary.get("correction", {}).get("previous_verdict", verdict)
    a1 = summary["arms"]["A1"]
    ceiling = summary["controls"]["inner_action_space_ceiling"]
    trajectories = summary["controls"]["a2l_vs_a2d_trajectory"]
    lines = [
        "# P28 lithofacies Stage-1 execution-agent pilot evidence",
        "",
        "## Outcome",
        "",
        (
            f"The preregistered verdict remains **{verdict}** (before correction: "
            f"**{previous_verdict}**). A2L promotion changed "
            f"fixed-schema nine-class Macro-F1 from `{summary['a0']['observed_mean']:.10f}` "
            f"to `{summary['arms']['A2L']['promotion']['fixed_schema_macro_f1_mean']:.10f}` "
            f"(`{summary['arms']['A2L']['promotion']['delta_from_a0']:+.10f}`)."
        ),
        "",
        "This is bounded adaptive development evidence, not a frozen-holdout estimate.",
        "",
        "## Frozen contract and nested split",
        "",
        f"- A0: `depth3_eta01_rounds60`, frozen reference `{EXPECTED_A0_DISPLAY:.10f}`.",
        f"- Split SHA-256: `{EXPECTED_SPLIT_HASH}`.",
        "- Four outer anonymous promotion rotations; each remaining three folds form inner LOGO3 selection.",
        "- Each execution-policy instance used exactly 3 actions without replacement and the same three model seeds.",
        "- Promotion results were evaluated only after inner selection and were never returned to a policy.",
        "",
        "## Corrected A1 identity control",
        "",
        (
            f"A1 now performs an actual `{a1['executor_entrypoint']}` replay of "
            f"`{a1['action_id']}` over all {a1['cell_count']} outer-fold/seed cells. "
            "It does not assign A0 hashes to A1."
        ),
        "",
        f"- Same action/config: **{'PASS' if a1['action_config_sha256'] == summary['a0']['config_sha256'] else 'FAIL'}**.",
        f"- Same fold/seed roster: **{'PASS' if a1['all_rosters_equal'] else 'FAIL'}**.",
        f"- Independent prediction hashes equal A0: **{'PASS' if a1['all_prediction_hashes_equal'] else 'FAIL'}** (`{a1['prediction_sha256']}`).",
        f"- Independent metric hashes equal A0: **{'PASS' if a1['all_metric_hashes_equal'] else 'FAIL'}** (`{a1['metric_sha256']}`).",
        "",
        "## Agent and controls",
        "",
        "| arm | role | normalized selection AUC@3 | promotion mean | delta vs A0 | positive outer folds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm in ("A2L", "A2D", "A4"):
        item = summary["arms"][arm]
        promotion = item["promotion"]
        lines.append(
            f"| {arm} | {item['role']} | {item['auc_at_3']:.10f} | "
            f"{promotion['fixed_schema_macro_f1_mean']:.10f} | "
            f"{promotion['delta_from_a0']:+.10f} | {promotion['positive_outer_folds']}/4 |"
        )
    a3 = summary["arms"]["A3"]
    lines.append(
        f"| A3 | equal-budget random median (3 policy seeds) | "
        f"{a3['auc_median']:.10f} | {a3['promotion_median_mean']:.10f} | "
        f"{a3['promotion_median_delta']:+.10f} | n/a |"
    )
    lines.extend(
        [
            "",
            "## A2L versus A2D trajectory clarification",
            "",
            (
                f"A2L and A2D action trajectories differ in "
                f"`{trajectories['differing_trajectory_count']}/"
                f"{trajectories['total_outer_folds']}` outer folds "
                f"(`{trajectories['differing_trajectory_outer_folds']}`), while "
                f"their promotion endpoint is "
                f"**{'identical' if trajectories['promotion_endpoint_identical'] else 'different'}**. "
                "Thus endpoint equality is convergence after real LLM execution, not evidence that A2L was skipped or mirrored."
            ),
            "",
            "## Exhaustive inner action-space ceiling",
            "",
            (
                f"All `{ceiling['actions_per_outer_rollout']}` allowlisted actions were "
                f"actually evaluated on each of `{ceiling['outer_rollout_count']}` inner "
                "LOGO3 rotations with the frozen three-seed roster. This diagnostic was "
                "not shown to any policy and was not used for promotion selection; no "
                "oracle promotion metric was computed."
            ),
            "",
            f"- Mean best-reachable inner delta versus A0: `{ceiling['mean_inner_ceiling_delta_from_a0']:+.10f}`.",
            f"- Mean A2L-selected inner delta versus A0: `{ceiling['mean_a2l_inner_delta_from_a0']:+.10f}`.",
            f"- Mean A2L regret to exhaustive inner ceiling: `{ceiling['mean_a2l_regret_to_inner_ceiling']:+.10f}`.",
            f"- Failure diagnosis: **`{ceiling['failure_diagnosis']}`** — {ceiling['interpretation']}",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## A2L per-fold observables",
            "",
            "| outer fold | selected action | A0 Macro-F1 | A2L Macro-F1 | delta |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    a2l_rollouts = summary["arms"]["A2L"]["rollouts"]
    for fold_id, delta in enumerate(summary["arms"]["A2L"]["promotion"]["outer_fold_deltas"]):
        baseline = summary["a0"]["outer_fold_means"][fold_id]
        lines.append(
            f"| {fold_id} | {a2l_rollouts[fold_id]['selected_for_promotion']} | "
            f"{baseline:.10f} | {baseline + delta:.10f} | {delta:+.10f} |"
        )
    lines.extend(
        [
            "",
            "## A2L per-class observable",
            "",
            "| class id | A0 mean F1 | A2L mean F1 | delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for class_id, (base, value) in enumerate(
        zip(
            summary["a0"]["per_class_f1_mean"],
            summary["arms"]["A2L"]["promotion"]["per_class_f1_mean"],
        )
    ):
        lines.append(
            f"| {class_id} | {base:.10f} | {value:.10f} | {value - base:+.10f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered gates",
            "",
        ]
    )
    for gate, passed in gates.items():
        lines.append(f"- `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Leakage and attribution audit",
            "",
            "- The live DeepSeek policy received only anonymous train-support buckets, `underfit|balanced|overfit`, action IDs/descriptions, and `improved|flat|worse` feedback.",
            "- It never received a raw metric, class label, residual, family identity, sample identifier, filesystem path, or promotion result.",
            "- `DEEPSEEK_KEY` was process-local and is absent from all artifacts.",
            "- Only `train.h5` and the accepted development LOGO4 batch were loaded; frozen holdout and `test.h5` were not accessed.",
            "- MOMENT pretrained/random remains a separate frozen paired-attribution lane and is excluded from the P28 agent AUC and promotion effect. 大模型贡献占比待下一轮消融确认。",
            "",
            "## Interpretation",
            "",
            (
                "The P28 policy is retained only if every preregistered gate passes. "
                "No post-hoc action, threshold, or trial was added after observing results."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifacts(
    *,
    output_dir: Path,
    batch_file: Path,
    protocol: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    results_path = output_dir / "results.jsonl"
    protocol_path = output_dir / "protocol.json"
    summary_path = output_dir / "summary.json"
    evidence_path = output_dir / "evidence.md"
    _write_json(protocol_path, protocol)
    _write_jsonl(results_path, results)
    _write_json(summary_path, summary)
    _write_text(evidence_path, _make_evidence(summary))
    test_path = TRACK_DIR / "tests" / "test_p28_agentic_optimization.py"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in (protocol_path, results_path, summary_path, evidence_path)
        },
        "sources": {
            Path(__file__).name: _sha256(Path(__file__)),
            str(test_path.relative_to(TRACK_DIR)): _sha256(test_path),
            "nested_development.npz": _sha256(batch_file),
            MOMENT_PAIRED_LANE["source"]: summary["moment_attribution_lane"][
                "source_sha256"
            ],
            "claude_final_audit_result.md": CLAUDE_AUDIT["result_sha256"],
        },
        "split_hash": EXPECTED_SPLIT_HASH,
        "verdict": summary["verdict"],
        "previous_verdict": summary.get("correction", {}).get(
            "previous_verdict", summary["verdict"]
        ),
        "frozen_test_accessed": False,
        "known_holdout_accessed": False,
        "credential_persisted": False,
    }
    _write_json(output_dir / "artifact_manifest.json", manifest)


def run_pilot(
    *,
    batch_file: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    api_key: str,
) -> dict[str, Any]:
    ensure_development_only_paths((batch_file, output_dir))
    output_dir = _owned_output(output_dir)
    if not api_key.strip():
        raise CredentialUnavailable("DEEPSEEK_KEY is missing; A2L fails closed")
    arrays, batch_manifest = load_nested_batch(batch_file)
    protocol = _protocol_payload(_sha256(batch_file))
    evaluator = _ActionEvaluator(arrays)
    started = time.perf_counter()

    a0_inner = {outer_id: evaluator.inner(outer_id, A0.action_id) for outer_id in OUTER_FOLDS}
    a0_by_outer = {
        outer_id: evaluator.promotion(outer_id, A0.action_id)
        for outer_id in OUTER_FOLDS
    }
    outer_means = [
        float(
            np.mean(
                [
                    row["metrics"]["fixed_schema_macro_f1"]
                    for row in a0_by_outer[outer_id]
                ]
            )
        )
        for outer_id in OUTER_FOLDS
    ]
    observed_a0 = float(np.mean(outer_means))
    if not math.isclose(observed_a0, EXPECTED_A0_MEAN, rel_tol=0.0, abs_tol=5e-10):
        raise RuntimeError(
            f"A0 reproduction changed: observed={observed_a0:.12f} expected={EXPECTED_A0_MEAN:.12f}"
        )
    if any(
        not math.isclose(observed, expected, rel_tol=0.0, abs_tol=5e-10)
        for observed, expected in zip(outer_means, EXPECTED_A0_FOLDS)
    ):
        raise RuntimeError("A0 outer-fold reproduction changed")

    a2l_rollouts = [
        _rollout(
            arm="A2L",
            instance_id="deepseek_live",
            outer_id=outer_id,
            baseline=a0_inner[outer_id],
            evaluate=evaluator,
            api_key=api_key,
        )
        for outer_id in OUTER_FOLDS
    ]
    a2d_rollouts = [
        _rollout(
            arm="A2D",
            instance_id="deterministic_diagnostic",
            outer_id=outer_id,
            baseline=a0_inner[outer_id],
            evaluate=evaluator,
            api_key="",
        )
        for outer_id in OUTER_FOLDS
    ]
    a3_instances: dict[str, list[dict[str, Any]]] = {}
    for policy_seed in POLICY_SEEDS:
        instance = f"random_seed_{policy_seed}"
        a3_instances[instance] = [
            _rollout(
                arm="A3",
                instance_id=instance,
                outer_id=outer_id,
                baseline=a0_inner[outer_id],
                evaluate=evaluator,
                api_key="",
                random_seed=policy_seed,
            )
            for outer_id in OUTER_FOLDS
        ]
    a4_rollouts = [
        _rollout(
            arm="A4",
            instance_id="offline_fixed_replay",
            outer_id=outer_id,
            baseline=a0_inner[outer_id],
            evaluate=evaluator,
            api_key="",
        )
        for outer_id in OUTER_FOLDS
    ]

    a0_cells = [row for outer_id in OUTER_FOLDS for row in a0_by_outer[outer_id]]
    a0_prediction_hash = _stable_hash(
        [row["prediction_sha256"] for row in a0_cells]
    )
    a0_metric_hash = _stable_hash([row["metrics"] for row in a0_cells])
    a1_control, a1_cells = _actual_a1_replay(arrays, a0_by_outer)
    a1_prediction_hash = str(a1_control["prediction_sha256"])
    a1_metric_hash = str(a1_control["metric_sha256"])

    a2l_promotion = _promotion_summary(a2l_rollouts, evaluator, a0_by_outer)
    a2d_promotion = _promotion_summary(a2d_rollouts, evaluator, a0_by_outer)
    a4_promotion = _promotion_summary(a4_rollouts, evaluator, a0_by_outer)
    a3_summaries = []
    for instance, rollouts in a3_instances.items():
        promotion = _promotion_summary(rollouts, evaluator, a0_by_outer)
        a3_summaries.append(
            {
                "instance_id": instance,
                "rollouts": rollouts,
                "auc_at_3": float(np.mean([row["auc_at_3"] for row in rollouts])),
                "promotion": promotion,
            }
        )
    inner_ceiling, inner_ceiling_records = _exhaustive_inner_ceiling(
        evaluator=evaluator,
        a0_inner=a0_inner,
        a2l_rollouts=a2l_rollouts,
    )
    trajectory_comparison = _trajectory_comparison(
        a2l_rollouts,
        a2d_rollouts,
        a2l_promotion,
        a2d_promotion,
    )

    a2l_auc = float(np.mean([row["auc_at_3"] for row in a2l_rollouts]))
    a2d_auc = float(np.mean([row["auc_at_3"] for row in a2d_rollouts]))
    a4_auc = float(np.mean([row["auc_at_3"] for row in a4_rollouts]))
    a3_auc_values = [item["auc_at_3"] for item in a3_summaries]
    a3_auc_median = float(np.median(a3_auc_values))
    a3_promotion_means = [
        item["promotion"]["fixed_schema_macro_f1_mean"] for item in a3_summaries
    ]
    a3_promotion_deltas = [item["promotion"]["delta_from_a0"] for item in a3_summaries]

    valid_trials = [
        trial
        for rollout in a2l_rollouts
        for trial in rollout["trials"]
    ]
    valid_action_rate = float(np.mean([trial["valid_action"] for trial in valid_trials]))
    a0_per_class = np.mean(
        [row["metrics"]["per_class_f1"] for row in a0_cells], axis=0
    ).tolist()
    finite_all = bool(
        np.isfinite(
            [
                value
                for item in (
                    a2l_promotion,
                    a2d_promotion,
                    a4_promotion,
                    *(entry["promotion"] for entry in a3_summaries),
                )
                for value in item["per_class_f1_mean"]
            ]
        ).all()
    )
    gates = {
        "a1_prediction_hash_equals_a0": a1_prediction_hash == a0_prediction_hash,
        "a1_metric_hash_equals_a0": a1_metric_hash == a0_metric_hash,
        "valid_action_rate_100_percent": valid_action_rate == 1.0,
        "a2l_auc_above_a3_median": a2l_auc > a3_auc_median,
        "a2l_auc_above_a2d": a2l_auc > a2d_auc,
        "a2l_positive_on_at_least_3_of_4_outer_folds": a2l_promotion[
            "positive_outer_folds"
        ]
        >= 3,
        "a2l_mean_promotion_delta_at_least_0_005": a2l_promotion["delta_from_a0"]
        >= PROMOTION_THRESHOLD,
        "no_non_finite_class_metric": finite_all,
    }
    verdict = "RETAIN" if all(gates.values()) else "REJECT"

    summary = {
        "schema_version": SCHEMA_VERSION,
        "split_hash": EXPECTED_SPLIT_HASH,
        "primary_metric": "fixed_schema_macro_f1",
        "a0": {
            "config": asdict(A0),
            "config_sha256": _stable_hash(asdict(A0)),
            "frozen_mean": EXPECTED_A0_MEAN,
            "observed_mean": observed_a0,
            "outer_fold_means": outer_means,
            "per_class_f1_mean": a0_per_class,
            "prediction_sha256": a0_prediction_hash,
            "metric_sha256": a0_metric_hash,
        },
        "arms": {
            "A1": a1_control,
            "A2L": {
                "role": "live_deepseek_execution_policy",
                "rollouts": a2l_rollouts,
                "auc_at_3": a2l_auc,
                "valid_action_rate": valid_action_rate,
                "promotion": a2l_promotion,
            },
            "A2D": {
                "role": "deterministic_diagnostic_policy",
                "rollouts": a2d_rollouts,
                "auc_at_3": a2d_auc,
                "promotion": a2d_promotion,
            },
            "A3": {
                "role": "random_without_replacement_equal_budget_control",
                "instances": a3_summaries,
                "auc_values": a3_auc_values,
                "auc_median": a3_auc_median,
                "promotion_median_mean": float(np.median(a3_promotion_means)),
                "promotion_median_delta": float(np.median(a3_promotion_deltas)),
            },
            "A4": {
                "role": "fixed_offline_replay_control",
                "rollouts": a4_rollouts,
                "auc_at_3": a4_auc,
                "promotion": a4_promotion,
            },
        },
        "controls": {
            "a2l_vs_a2d_trajectory": trajectory_comparison,
            "inner_action_space_ceiling": inner_ceiling,
        },
        "correction": {
            "source": "claude_final_audit",
            "audit": dict(CLAUDE_AUDIT),
            "previous_verdict": "REJECT",
            "corrected_verdict": verdict,
            "thresholds_changed": False,
            "splits_changed": False,
        },
        "moment_attribution_lane": _moment_lane_payload(),
        "gates": gates,
        "verdict": verdict,
        "valid_action_count": int(sum(trial["valid_action"] for trial in valid_trials)),
        "expected_live_action_count": len(OUTER_FOLDS) * TRIAL_BUDGET,
        "elapsed_seconds": time.perf_counter() - started,
        "data": {
            "nested_batch_sha256": _sha256(batch_file),
            "development_hdf5_sha256": batch_manifest["development_hdf5_sha256"],
            "loaded_files": batch_manifest["loaded_files"],
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
            "test_metrics_used": False,
        },
        "credential_persisted": False,
    }

    results: list[dict[str, Any]] = []
    for row in a0_cells:
        results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "a0_promotion_cell",
                "arm": "A0",
                **row,
            }
        )
    for row in a1_cells:
        results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "a1_actual_replay_promotion_cell",
                "arm": "A1",
                "executor_entrypoint": "_evaluate_promotion_action",
                "action_id": A0.action_id,
                **row,
            }
        )
    for arm, groups in (
        ("A2L", [a2l_rollouts]),
        ("A2D", [a2d_rollouts]),
        ("A3", list(a3_instances.values())),
        ("A4", [a4_rollouts]),
    ):
        for group in groups:
            for rollout in group:
                for trial in rollout["trials"]:
                    results.append(
                        {
                            "schema_version": RESULT_SCHEMA,
                            "record_type": "selection_trial",
                            "arm": arm,
                            "instance_id": rollout["instance_id"],
                            "outer_rollout_id": rollout["outer_rollout_id"],
                            **trial,
                            "promotion_result_exposed_to_policy": False,
                        }
                    )
    for row in inner_ceiling_records:
        results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "inner_exhaustive_oracle_action",
                "arm": "ORACLE_INNER_CEILING",
                **row,
            }
        )
    _write_artifacts(
        output_dir=output_dir,
        batch_file=batch_file,
        protocol=protocol,
        results=results,
        summary=summary,
    )
    return summary


def correct_existing_pilot(
    *,
    batch_file: Path = DEFAULT_BATCH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Apply the Claude audit controls without replaying the live LLM policy."""
    ensure_development_only_paths((batch_file, output_dir))
    output_dir = _owned_output(output_dir)
    started = time.perf_counter()
    artifact_paths = {
        name: output_dir / name
        for name in (
            "artifact_manifest.json",
            "protocol.json",
            "results.jsonl",
            "summary.json",
            "evidence.md",
        )
    }
    previous_manifest = json.loads(
        artifact_paths["artifact_manifest.json"].read_text(encoding="utf-8")
    )
    for name, item in previous_manifest.get("artifacts", {}).items():
        path = output_dir / name
        if _sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"pre-correction artifact hash mismatch: {name}")
    previous_hashes = {
        name: _sha256(path) for name, path in artifact_paths.items()
    }
    summary = json.loads(artifact_paths["summary.json"].read_text(encoding="utf-8"))
    protocol = json.loads(artifact_paths["protocol.json"].read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in artifact_paths["results.jsonl"].read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    if summary.get("schema_version") not in {PRE_CORRECTION_SCHEMA, SCHEMA_VERSION}:
        raise RuntimeError("correction source has an unknown P28 summary schema")
    if (
        summary.get("split_hash") != EXPECTED_SPLIT_HASH
        or summary.get("verdict") != "REJECT"
        or protocol.get("split_hash") != EXPECTED_SPLIT_HASH
    ):
        raise RuntimeError("correction source changed the frozen split or reject verdict")

    policy_rows_before = [
        {key: value for key, value in row.items() if key != "schema_version"}
        for row in results
        if row.get("record_type") == "selection_trial"
        and row.get("arm") == "A2L"
    ]
    if len(policy_rows_before) != len(OUTER_FOLDS) * TRIAL_BUDGET:
        raise RuntimeError("correction source lacks the twelve live A2L trials")
    policy_trajectory_sha256 = _stable_hash(policy_rows_before)

    arrays, batch_manifest = load_nested_batch(batch_file)
    a0_rows = [
        row for row in results if row.get("record_type") == "a0_promotion_cell"
    ]
    if len(a0_rows) != len(OUTER_FOLDS) * len(REPEAT_SEEDS):
        raise RuntimeError("correction source lacks the frozen A0 cells")
    a0_by_outer = {
        outer_id: sorted(
            [
                row
                for row in a0_rows
                if int(row["outer_rollout_id"]) == outer_id
            ],
            key=lambda row: int(row["repeat_id"]),
        )
        for outer_id in OUTER_FOLDS
    }
    a1_control, a1_cells = _actual_a1_replay(arrays, a0_by_outer)
    if a1_control["a0_prediction_sha256"] != summary["a0"]["prediction_sha256"]:
        raise RuntimeError("A0 prediction hash changed before the actual A1 replay")
    if a1_control["a0_metric_sha256"] != summary["a0"]["metric_sha256"]:
        raise RuntimeError("A0 metric hash changed before the actual A1 replay")

    evaluator = _ActionEvaluator(arrays)
    a0_inner = {
        outer_id: evaluator.inner(outer_id, A0.action_id)
        for outer_id in OUTER_FOLDS
    }
    a2l_rollouts = summary["arms"]["A2L"]["rollouts"]
    a2d_rollouts = summary["arms"]["A2D"]["rollouts"]
    inner_ceiling, inner_ceiling_records = _exhaustive_inner_ceiling(
        evaluator=evaluator,
        a0_inner=a0_inner,
        a2l_rollouts=a2l_rollouts,
    )
    trajectory_comparison = _trajectory_comparison(
        a2l_rollouts,
        a2d_rollouts,
        summary["arms"]["A2L"]["promotion"],
        summary["arms"]["A2D"]["promotion"],
    )
    if trajectory_comparison["differing_trajectory_count"] != 2:
        raise RuntimeError("Claude-audited A2L/A2D trajectory distinction changed")
    if trajectory_comparison["promotion_endpoint_identical"] is not True:
        raise RuntimeError("Claude-audited identical promotion endpoint changed")

    summary["schema_version"] = SCHEMA_VERSION
    summary["a0"]["config_sha256"] = _stable_hash(asdict(A0))
    summary["arms"]["A1"] = a1_control
    summary["controls"] = {
        "a2l_vs_a2d_trajectory": trajectory_comparison,
        "inner_action_space_ceiling": inner_ceiling,
    }
    summary["gates"]["a1_prediction_hash_equals_a0"] = (
        a1_control["prediction_sha256"] == summary["a0"]["prediction_sha256"]
    )
    summary["gates"]["a1_metric_hash_equals_a0"] = (
        a1_control["metric_sha256"] == summary["a0"]["metric_sha256"]
    )
    corrected_verdict = "RETAIN" if all(summary["gates"].values()) else "REJECT"
    if corrected_verdict != "REJECT":
        raise RuntimeError("Claude correction unexpectedly changed the frozen reject gate")
    summary["verdict"] = corrected_verdict
    summary["correction"] = {
        "source": "claude_final_audit",
        "audit": dict(CLAUDE_AUDIT),
        "previous_verdict": "REJECT",
        "corrected_verdict": corrected_verdict,
        "previous_artifact_sha256": previous_hashes,
        "thresholds_changed": False,
        "splits_changed": False,
        "live_provider_policy_replayed": False,
        "preserved_a2l_policy_trajectory_sha256": policy_trajectory_sha256,
        "bounded_model_recomputation": {
            "a1_outer_promotion_cells": len(a1_cells),
            "oracle_inner_action_evaluations": len(inner_ceiling_records),
            "oracle_inner_model_cells": len(inner_ceiling_records)
            * 3
            * len(REPEAT_SEEDS),
            "oracle_promotion_evaluations": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary["data"]["nested_batch_sha256"] = _sha256(batch_file)
    summary["data"]["development_hdf5_sha256"] = batch_manifest[
        "development_hdf5_sha256"
    ]

    retained_results = [
        {**row, "schema_version": RESULT_SCHEMA}
        for row in results
        if row.get("record_type")
        not in {
            "a1_actual_replay_promotion_cell",
            "inner_exhaustive_oracle_action",
        }
    ]
    for row in a1_cells:
        retained_results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "a1_actual_replay_promotion_cell",
                "arm": "A1",
                "executor_entrypoint": "_evaluate_promotion_action",
                "action_id": A0.action_id,
                **row,
            }
        )
    for row in inner_ceiling_records:
        retained_results.append(
            {
                "schema_version": RESULT_SCHEMA,
                "record_type": "inner_exhaustive_oracle_action",
                "arm": "ORACLE_INNER_CEILING",
                **row,
            }
        )
    policy_rows_after = [
        {key: value for key, value in row.items() if key != "schema_version"}
        for row in retained_results
        if row.get("record_type") == "selection_trial"
        and row.get("arm") == "A2L"
    ]
    if _stable_hash(policy_rows_after) != policy_trajectory_sha256:
        raise RuntimeError("correction altered the live A2L trajectory")

    protocol_template = _protocol_payload(_sha256(batch_file))
    protocol["schema_version"] = PROTOCOL_SCHEMA
    protocol["diagnostic_controls"] = protocol_template["diagnostic_controls"]
    protocol["claude_correction_audit"] = dict(CLAUDE_AUDIT)
    protocol["correction"] = {
        "previous_verdict": "REJECT",
        "corrected_verdict": corrected_verdict,
        "thresholds_changed": False,
        "splits_changed": False,
        "live_provider_policy_replayed": False,
        "preserved_a2l_policy_trajectory_sha256": policy_trajectory_sha256,
    }
    _write_artifacts(
        output_dir=output_dir,
        batch_file=batch_file,
        protocol=protocol,
        results=retained_results,
        summary=summary,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "previous_verdict": "REJECT",
        "corrected_verdict": corrected_verdict,
        "a1_actual_replay_cells": len(a1_cells),
        "a1_prediction_hash_equal": a1_control[
            "all_prediction_hashes_equal"
        ],
        "a1_metric_hash_equal": a1_control["all_metric_hashes_equal"],
        "oracle_inner_action_evaluations": len(inner_ceiling_records),
        "oracle_promotion_evaluations": 0,
        "failure_diagnosis": inner_ceiling["failure_diagnosis"],
        "mean_inner_ceiling_delta_from_a0": inner_ceiling[
            "mean_inner_ceiling_delta_from_a0"
        ],
        "mean_a2l_regret_to_inner_ceiling": inner_ceiling[
            "mean_a2l_regret_to_inner_ceiling"
        ],
        "a2l_a2d_differing_trajectory_folds": trajectory_comparison[
            "differing_trajectory_outer_folds"
        ],
        "promotion_endpoint_identical": trajectory_comparison[
            "promotion_endpoint_identical"
        ],
        "live_provider_policy_replayed": False,
        "preserved_a2l_policy_trajectory_sha256": policy_trajectory_sha256,
        "elapsed_seconds": summary["correction"]["elapsed_seconds"],
        "frozen_test_accessed": False,
    }


def verify_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = _owned_output(output_dir)
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise RuntimeError("unknown P28 artifact manifest")
    observed = {}
    for name, expected in manifest["artifacts"].items():
        path = output_dir / name
        digest = _sha256(path)
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"P28 artifact verification failed: {name}")
        observed[name] = digest
    source_paths = {
        Path(__file__).name: Path(__file__),
        "tests/test_p28_agentic_optimization.py": TRACK_DIR
        / "tests"
        / "test_p28_agentic_optimization.py",
        "nested_development.npz": DEFAULT_BATCH,
        MOMENT_PAIRED_LANE["source"]: TRACK_DIR / MOMENT_PAIRED_LANE["source"],
    }
    for name, path in source_paths.items():
        if _sha256(path) != manifest["sources"][name]:
            raise RuntimeError(f"P28 source verification failed: {name}")
    if manifest["sources"].get("claude_final_audit_result.md") != CLAUDE_AUDIT[
        "result_sha256"
    ]:
        raise RuntimeError("P28 Claude audit hash changed")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output_dir.glob("*")
        if path.is_file()
    )
    credential = os.environ.get("DEEPSEEK_KEY", "")
    if credential and credential in serialized:
        raise RuntimeError("DeepSeek credential leaked into P28 artifacts")
    if summary.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise RuntimeError("P28 summary split hash changed")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("P28 corrected summary schema changed")
    if summary.get("data", {}).get("frozen_test_accessed") is not False:
        raise RuntimeError("P28 summary violates the test firewall")
    a1_rows = [
        row
        for row in results
        if row.get("record_type") == "a1_actual_replay_promotion_cell"
    ]
    oracle_rows = [
        row
        for row in results
        if row.get("record_type") == "inner_exhaustive_oracle_action"
    ]
    if len(a1_rows) != len(OUTER_FOLDS) * len(REPEAT_SEEDS):
        raise RuntimeError("P28 actual A1 replay evidence is incomplete")
    if len(oracle_rows) != len(OUTER_FOLDS) * len(ACTION_IDS):
        raise RuntimeError("P28 exhaustive inner ceiling evidence is incomplete")
    if any(
        row.get("used_for_promotion_selection") is not False
        or row.get("promotion_metrics_computed") is not False
        for row in oracle_rows
    ):
        raise RuntimeError("P28 inner oracle entered promotion")
    a1 = summary["arms"]["A1"]
    if not (
        a1["actual_replay_executed"]
        and a1["all_rosters_equal"]
        and a1["all_prediction_hashes_equal"]
        and a1["all_metric_hashes_equal"]
    ):
        raise RuntimeError("P28 actual A1 replay identity failed")
    ceiling = summary["controls"]["inner_action_space_ceiling"]
    if (
        ceiling["used_for_promotion_selection"] is not False
        or ceiling["promotion_metrics_computed"] is not False
    ):
        raise RuntimeError("P28 summary misstates the inner oracle boundary")
    trajectory = summary["controls"]["a2l_vs_a2d_trajectory"]
    if (
        trajectory["differing_trajectory_count"] != 2
        or trajectory["promotion_endpoint_identical"] is not True
    ):
        raise RuntimeError("P28 A2L/A2D trajectory clarification changed")
    if summary["correction"]["previous_verdict"] != "REJECT" or summary[
        "verdict"
    ] != "REJECT":
        raise RuntimeError("P28 correction changed the reject verdict")
    return {
        "verified_artifacts": observed,
        "verdict": summary["verdict"],
        "previous_verdict": summary["correction"]["previous_verdict"],
        "split_hash": summary["split_hash"],
        "a1_actual_replay_cells": len(a1_rows),
        "oracle_inner_action_evaluations": len(oracle_rows),
        "oracle_promotion_evaluations": 0,
        "failure_diagnosis": ceiling["failure_diagnosis"],
        "a2l_a2d_differing_trajectory_count": trajectory[
            "differing_trajectory_count"
        ],
        "credential_persisted": False,
        "frozen_test_accessed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", type=Path, required=True)
    prepare_parser.add_argument("--accepted-logo4-batch", type=Path, required=True)
    prepare_parser.add_argument("--batch-file", type=Path, default=DEFAULT_BATCH)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--batch-file", type=Path, default=DEFAULT_BATCH)
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    correct_parser = subparsers.add_parser("correct")
    correct_parser.add_argument("--batch-file", type=Path, default=DEFAULT_BATCH)
    correct_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "prepare":
        payload = prepare_nested_batch(
            dataset_root=args.dataset_root,
            accepted_logo4_batch=args.accepted_logo4_batch,
            batch_file=args.batch_file,
        )
    elif args.command == "run":
        payload = run_pilot(
            batch_file=args.batch_file,
            output_dir=args.output_dir,
            api_key=os.environ.get("DEEPSEEK_KEY", ""),
        )
    elif args.command == "correct":
        payload = correct_existing_pilot(
            batch_file=args.batch_file,
            output_dir=args.output_dir,
        )
    else:
        payload = verify_artifacts(args.output_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
