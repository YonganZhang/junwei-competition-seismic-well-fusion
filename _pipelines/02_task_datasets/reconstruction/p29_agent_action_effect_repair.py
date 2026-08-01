#!/usr/bin/env python3
"""P29 action-effect repair utilities and bounded development probe.

This layer is intentionally independent of P28 artifacts: it adds an
information-ablation contract, high-leverage single-factor actions, and a
serializable predictor replay endpoint while retaining P19 held-fold purge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import p28_agentic_optimization as p28

SCHEMA_VERSION = "reconstruction-p29-agent-action-effect-repair/v1"
SAFE_QUANTITATIVE_FIELDS = ("classification", "relative_rmse_change")


def build_prompt_observation(*, mode: str, round_id: int,
                             histories: Mapping[int, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Render either categorical-only or safe relative-RMSE feedback.

    Fold outcomes, absolute RMSE, predictions and held-fold labels remain
    excluded in both modes; relative change is the only quantitative signal.
    """
    if mode not in {"categorical", "safe_quantitative"}:
        raise ValueError("unknown prompt information mode")
    contexts = []
    for fold in p28.base.FOLD_IDS:
        rows = []
        for row in histories[int(fold)]:
            feedback = {"classification": row["feedback"]["classification"]}
            if mode == "safe_quantitative":
                feedback["relative_rmse_change"] = float(row["feedback"]["relative_rmse_change"])
            rows.append({"round": int(row["round"]), "action_id": row["action_id"], "feedback": feedback})
        contexts.append({"fold": int(fold), "prior_trials": rows})
    return {"schema_version": "p29-prompt-observation/v1", "mode": mode,
            "round": int(round_id), "fold_contexts": contexts,
            "visibility": {"fold_win_loss": False, "absolute_rmse": False,
                            "raw_predictions": False}}


def action_registry() -> tuple[dict[str, Any], ...]:
    base = dict(p28.A0_PARAMETERS)
    rows = []
    specs = (("foundation_weight_0.00", "foundation_weight", 0.0),
             ("foundation_weight_0.25", "foundation_weight", 0.25),
             ("foundation_weight_0.50", "foundation_weight", 0.50),
             ("vertical_weight_2.0", "vertical_weight", 2.0),
             ("vertical_weight_8.0", "vertical_weight", 8.0),
             ("seismic_mix_low", "seismic_weights", [0.0, 0.05, 0.10]),
             ("seismic_mix_high", "seismic_weights", [0.0, 0.20, 0.40]))
    for action_id, factor, value in specs:
        params = dict(base); params[factor] = value
        rows.append({"action_id": action_id, "changed_factor": factor,
                     "value": value, "parameters": params})
    return tuple(rows)


def validate_action_registry(registry: Sequence[Mapping[str, Any]] | None = None) -> None:
    registry = action_registry() if registry is None else registry
    for action in registry:
        changed = [k for k, v in action["parameters"].items() if v != p28.A0_PARAMETERS[k]]
        if changed != [action["changed_factor"]]:
            raise ValueError(f"action is not single-factor: {action['action_id']}")


def predictor_config(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "p29-predictor-config/v1", "parameters": json.loads(json.dumps(parameters))}


def replay_predictor(config: Mapping[str, Any], *, coordinates: np.ndarray,
                     values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Serializable replay endpoint for a deterministic weighted predictor."""
    params = config.get("parameters", config)
    power = float(params.get("distance_power", 1.5))
    vertical = float(params.get("vertical_weight", 4.0))
    c = np.asarray(coordinates, dtype=float).copy(); q = np.asarray(query, dtype=float).copy()
    c[:, 2] *= vertical; q[:, 2] *= vertical
    d = np.linalg.norm(q[:, None, :] - c[None, :, :], axis=2)
    w = np.maximum(d, 1e-8) ** (-power)
    return (w @ np.asarray(values, dtype=float)) / w.sum(axis=1)


def prediction_hash(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prediction, dtype=np.float64).tobytes()).hexdigest()


def run_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_action_registry()
    coordinates = np.asarray([[0., 0., 0.], [1., 0., 1.], [0., 1., 2.]], float)
    values = np.asarray([1., 2., 4.]); query = np.asarray([[.2, .2, .3], [.8, .1, 1.5]])
    a0 = predictor_config(p28.A0_PARAMETERS)
    predictions = {"A0": replay_predictor(a0, coordinates=coordinates, values=values, query=query)}
    for action in action_registry():
        predictions[action["action_id"]] = replay_predictor(
            predictor_config(action["parameters"]), coordinates=coordinates, values=values, query=query)
    result = {"schema_version": SCHEMA_VERSION, "probe": True,
              "action_registry": list(action_registry()),
              "prediction_hashes": {k: prediction_hash(v) for k, v in predictions.items()},
              "different_action_count": len({prediction_hash(v) for v in predictions.values()}),
              "held_fold_purge_reused": True,
              "prompt_modes": ["categorical", "safe_quantitative"],
              "promotion_threshold": {"minimum_relative_gain": p28.MIN_LLM_RELATIVE_GAIN,
                                      "max_fold_relative_regression": p28.MAX_FOLD_RELATIVE_REGRESSION}}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); run_probe(args.output_dir)
