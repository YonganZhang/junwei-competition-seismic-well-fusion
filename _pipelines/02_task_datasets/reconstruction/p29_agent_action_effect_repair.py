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
import p19_meta_purged_geostatistics as p19

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
                feedback["fold_outcomes"] = dict(row["feedback"].get("fold_outcomes", {}))
                feedback["uncertainty"] = dict(row["feedback"].get("uncertainty", {}))
            rows.append({"round": int(row["round"]), "action_id": row["action_id"], "feedback": feedback})
        contexts.append({"fold": int(fold), "prior_trials": rows})
    result = {"schema_version": "p29-prompt-observation/v2", "mode": mode,
            "round": int(round_id), "fold_contexts": contexts,
            "visibility": {"fold_win_loss": False, "absolute_rmse": False,
                            "raw_predictions": False}}
    if mode == "safe_quantitative":
        result["remaining_budget"] = max(0, p28.TRIALS_PER_STRATEGY - int(round_id) + 1)
        result["promotion_threshold"] = {
            "minimum_relative_gain": p28.MIN_LLM_RELATIVE_GAIN,
            "maximum_fold_relative_regression": p28.MAX_FOLD_RELATIVE_REGRESSION,
        }
        result["uncertainty_definition"] = "fold bootstrap standard error supplied per trial"
    return result


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
                     values: np.ndarray, query: np.ndarray,
                     seismic: np.ndarray | None = None,
                     latent: np.ndarray | None = None) -> np.ndarray:
    """Replay the same metric construction used by P28's action bank."""
    params = config.get("parameters", config)
    kernel = str(params.get("kernel", "inverse_distance"))
    power = float(params.get("distance_power", 1.5))
    vertical = float(params.get("vertical_weight", 4.0))
    seismic = np.zeros((len(coordinates), 1)) if seismic is None else np.asarray(seismic, float)
    latent = np.zeros((len(coordinates), 1)) if latent is None else np.asarray(latent, float)
    qseis = np.zeros((len(query), seismic.shape[1]))
    qlatent = np.zeros((len(query), latent.shape[1]))
    c = np.asarray(coordinates, dtype=float).copy(); q = np.asarray(query, dtype=float).copy()
    c[:, 2] *= vertical; q[:, 2] *= vertical
    mix = np.asarray(params.get("seismic_weights", [0.0] * seismic.shape[1]), float)
    metric_c = np.column_stack([c, seismic * mix[:seismic.shape[1]],
                                latent * float(params.get("foundation_weight", 0.1))])
    metric_q = np.column_stack([q, qseis, qlatent])
    d = np.linalg.norm(metric_q[:, None, :] - metric_c[None, :, :], axis=2)
    w = p28._kernel_weights(d, family=kernel, power=power, bandwidth=1.0)  # noqa: SLF001
    return (w @ np.asarray(values, dtype=float)) / w.sum(axis=1)


def prediction_hash(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prediction, dtype=np.float64).tobytes()).hexdigest()


def run_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_action_registry()
    coordinates = np.asarray([[0., 0., 0.], [1., 0., 1.], [0., 1., 2.]], float)
    values = np.asarray([1., 2., 4.]); query = np.asarray([[.2, .2, .3], [.8, .1, 1.5]])
    a0 = predictor_config(p28.A0_PARAMETERS)
    seismic = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1], [0.7, 0.1, 0.2]])
    latent = np.asarray([[0.1], [0.9], [0.3]])
    predictions = {"A0": replay_predictor(a0, coordinates=coordinates, values=values, query=query,
                                           seismic=seismic, latent=latent)}
    for action in action_registry():
        predictions[action["action_id"]] = replay_predictor(
            predictor_config(action["parameters"]), coordinates=coordinates, values=values, query=query,
            seismic=seismic, latent=latent)
    held = np.asarray([[0., 0., 0.]])
    purge_demo = p19._rows(held)  # noqa: SLF001
    result = {"schema_version": SCHEMA_VERSION, "probe": True,
              "action_registry": list(action_registry()),
              "predictor_configs": {"A0": a0, **{a["action_id"]: predictor_config(a["parameters"]) for a in action_registry()}},
              "prediction_hashes": {k: prediction_hash(v) for k, v in predictions.items()},
              "replay_hash_contract": "prediction_hash(replay_predictor(config))",
              "different_action_count": len({prediction_hash(v) for v in predictions.values()}),
              "held_fold_purge_reused": {"entrypoint": "p19._rows", "held_rows": len(purge_demo)},
              "prompt_modes": ["categorical", "safe_quantitative"],
              "promotion_threshold": {"minimum_relative_gain": p28.MIN_LLM_RELATIVE_GAIN,
                                      "max_fold_relative_regression": p28.MAX_FOLD_RELATIVE_REGRESSION}}
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); run_probe(args.output_dir)
