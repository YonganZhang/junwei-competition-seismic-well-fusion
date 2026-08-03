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
from sklearn.neighbors import NearestNeighbors

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
    if mode == "safe_quantitative" and any(not histories[int(f)] for f in p28.base.FOLD_IDS):
        raise ValueError("safe quantitative prompt requires real prior-trial histories")
    for fold in p28.base.FOLD_IDS:
        rows = []
        for row in histories[int(fold)]:
            feedback = {"classification": row["feedback"]["classification"]}
            if mode == "safe_quantitative":
                feedback["relative_rmse_change"] = float(row["feedback"]["relative_rmse_change"])
                feedback["fold_outcomes"] = dict(row["feedback"].get("fold_outcomes", {}))
                feedback["uncertainty"] = dict(row["feedback"].get("uncertainty", {}))
            rendered = {"round": int(row["round"]), "action_id": row["action_id"], "feedback": feedback}
            if "selection_fold_ids" in row: rendered["selection_fold_ids"] = list(row["selection_fold_ids"])
            rows.append(rendered)
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

def build_outer_observation(*, mode: str, held_fold: int, histories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selection = sorted({int(f) for row in histories for f in row["selection_fold_ids"]})
    if int(held_fold) in selection or not selection: raise ValueError("invalid outer selection observation")
    contexts = []
    for sf in selection:
        rows = []
        for row in histories:
            fb = {"classification": row["feedback"]["classification"]}
            if mode == "safe_quantitative":
                fb.update(relative_rmse_change=float(row["feedback"]["relative_rmse_change"]), fold_outcomes=dict(row["feedback"]["fold_outcomes"]), uncertainty=dict(row["feedback"]["uncertainty"]))
            rows.append({"round": row["round"], "action_id": row["action_id"], "feedback": fb, "source_fold": sf})
        contexts.append({"fold": sf, "prior_trials": rows})
    out = {"schema_version": "p29-outer-observation/v1", "mode": mode, "outer_held_fold": int(held_fold), "selection_fold_ids": selection, "fold_contexts": contexts, "fixed_budget": p28.TRIALS_PER_STRATEGY}
    if mode == "safe_quantitative":
        out["remaining_budget"] = 3; out["promotion_threshold"] = {"minimum_relative_gain": p28.MIN_LLM_RELATIVE_GAIN, "maximum_fold_relative_regression": p28.MAX_FOLD_RELATIVE_REGRESSION}; out["uncertainty_definition"] = "standard error over four selection folds"
    return out

def _policy_decision(observation: Mapping[str, Any], *, mode: str, action_ids: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """DeepSeek policy call with an auditable deterministic fallback."""
    try:
        key = p28._load_deepseek_key(p28.DEFAULT_CREDENTIAL_HELPER)  # noqa: SLF001
        parsed, provider = p28._deepseek_json(key=key, system="Choose one action_id from the allowlisted action table. Return JSON {action_id, rationale}.", observation=observation)  # noqa: SLF001
        action = str(parsed["action_id"])
        if action not in action_ids: raise ValueError("provider selected non-allowlisted action")
        return action, {"provider": "deepseek", "status": "success", **provider}
    except Exception as exc:  # fail closed, preserving evidence
        return str(action_ids[0]), {"provider": "deepseek", "status": "fallback", "error_type": type(exc).__name__, "fallback": "first_allowlisted_action"}

def run_real_probe(*, data_dir: Path, stage3_root: Path, output_dir: Path) -> dict[str, Any]:
    p28.base.ensure_no_holdout_paths((data_dir, stage3_root, output_dir))
    inputs = p28.base.resolve_dev_inputs(data_dir); oof = p28.base.load_oof_development(stage3_root)
    folds, _ = p28.p17.load_fold_samples(stage3_root=stage3_root, train_h5=inputs.train_h5, oof=oof)
    indices = p28.p17._unique_indices(folds)  # noqa: SLF001
    cache_indices, foundation, audit = p28.p18._load_feature_cache(p28.p18.DEFAULT_FEATURE_CACHE, train_h5=inputs.train_h5, expected_indices=indices)  # noqa: SLF001
    prepared, _ = p28.p18._prepare_fold_metrics(folds=folds, requested_indices=cache_indices, foundation_features=foundation)  # noqa: SLF001
    registry = action_registry(); bank = build_real_action_bank(oof=oof, prepared=prepared, registry=registry); a0 = bank["A0"]
    metrics = {}; configs = {"A0": predictor_config(p28.A0_PARAMETERS)}
    for action in registry: configs[action["action_id"]] = predictor_config(action["parameters"])
    for name, prediction in bank.items():
        fm = _fold_metrics(oof.target, prediction, oof.fold_ids); base_rmse = _fold_metrics(oof.target, a0, oof.fold_ids)["pooled"]["rmse"]
        metrics[name] = {**fm, "signed_delta_rmse": float(fm["pooled"]["rmse"] - base_rmse), "config_hash": hashlib.sha256(json.dumps(configs[name], sort_keys=True).encode()).hexdigest(), "prediction_hash": prediction_hash(prediction)}
    full_bank = bank; purge_audits = []; purged_banks = {}
    histories = {int(f): [] for f in p28.base.FOLD_IDS}
    for held_fold in p28.base.FOLD_IDS:
        held = next(f for f in folds if f.fold_id == held_fold); forbidden = p19._rows(held.validation_indices_kji)
        purged_folds = []; removed = {}
        for fold_sample in folds:
            if fold_sample.fold_id == held_fold:
                purged_folds.append(fold_sample); removed[str(fold_sample.fold_id)] = 0
            else:
                clean, count = p19._without_coordinates(fold_sample, forbidden)
                purged_folds.append(clean); removed[str(fold_sample.fold_id)] = int(count)
        purged_prepared, _ = p28.p18._prepare_fold_metrics(folds=tuple(purged_folds), requested_indices=cache_indices, foundation_features=foundation)  # noqa: SLF001
        bank = build_real_action_bank(oof=oof, prepared=purged_prepared, registry=registry); purged_banks[int(held_fold)] = bank
        purge_audits.append({"held_fold": int(held_fold), "forbidden_unique_coordinates": len(forbidden), "removed_train_labels_by_fold": removed, "removed_occurrences": sum(removed.values()), "p19_rows_called": True, "p19_without_coordinates_called": True})
        selection_folds = [int(f) for f in p28.base.FOLD_IDS if int(f) != int(held_fold)]
        base_rows = {r["fold"]: r for r in _fold_metrics(oof.target, bank["A0"], oof.fold_ids)["per_fold"]}
        for action in registry:
            rows = {r["fold"]: r for r in _fold_metrics(oof.target, bank[action["action_id"]], oof.fold_ids)["per_fold"]}
            deltas = np.asarray([(rows[f]["rmse"] - base_rows[f]["rmse"]) / base_rows[f]["rmse"] for f in selection_folds], float)
            delta = float(np.mean(deltas)); se = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
            histories[int(held_fold)].append({"round": 1, "action_id": action["action_id"], "selection_fold_ids": selection_folds, "feedback": {"classification": "improved" if delta < -p28.FLAT_RELATIVE_TOLERANCE else "worse" if delta > p28.FLAT_RELATIVE_TOLERANCE else "flat", "relative_rmse_change": delta, "fold_outcomes": {"win": int(np.sum(deltas < 0)), "loss": int(np.sum(deltas > 0)), "tie": int(np.sum(deltas == 0))}, "uncertainty": {"stderr": se}}})
    chosen_id = "vertical_weight_8.0"; saved = json.loads(json.dumps(configs[chosen_id])); chosen_params = saved["parameters"]
    chosen_spec = {"action_id": chosen_id, "changed_factor": "vertical_weight", "value": chosen_params["vertical_weight"], "parameters": chosen_params}
    replay_bank = build_real_action_bank(oof=oof, prepared=prepared, registry=(chosen_spec,)); replay = replay_bank[chosen_id]; fold = int(p28.base.FOLD_IDS[0]); mask = oof.fold_ids == fold; expected = full_bank[chosen_id][mask]; np.testing.assert_array_equal(replay[mask], expected)
    observations = {mode: build_prompt_observation(mode=mode, round_id=2, histories=histories) for mode in ("categorical", "safe_quantitative")}
    for held, rows in histories.items():
        if any(int(held) in row["selection_fold_ids"] for row in rows): raise RuntimeError("held promotion fold leaked into prompt")
    action_ids = [a["action_id"] for a in registry]; policy_rows = []; categorical_rows = []
    for held in p28.base.FOLD_IDS:
        outer = build_outer_observation(mode="safe_quantitative", held_fold=int(held), histories=histories[int(held)])
        cat = build_outer_observation(mode="categorical", held_fold=int(held), histories=histories[int(held)])
        safe_action, provider = _policy_decision(outer, mode="safe_quantitative", action_ids=action_ids)
        cat_action, cat_provider = _policy_decision(cat, mode="categorical", action_ids=action_ids)
        held_mask = oof.fold_ids == held; base_rmse = p28._metrics(oof.target[held_mask], purged_banks[int(held)]["A0"][held_mask])["rmse"]
        def row_for(action_id: str) -> dict[str, Any]:
            pred = purged_banks[int(held)][action_id]; rmse = p28._metrics(oof.target[held_mask], pred[held_mask])["rmse"]
            return {"action_id": action_id, "rmse": float(rmse), "signed_delta_rmse": float(rmse - base_rmse), "prediction_hash": prediction_hash(pred[held_mask]), "config_hash": metrics[action_id]["config_hash"]}
        policy_rows.append({"held_fold": int(held), "selection_fold_ids": [f for f in p28.base.FOLD_IDS if f != held], "safe_action": row_for(safe_action), "safe_provider": provider, "safe_observation_hash": hashlib.sha256(json.dumps(outer, sort_keys=True).encode()).hexdigest()})
        categorical_rows.append({"held_fold": int(held), "selection_fold_ids": [f for f in p28.base.FOLD_IDS if f != held], "categorical_action": row_for(cat_action), "categorical_provider": cat_provider, "categorical_observation_hash": hashlib.sha256(json.dumps(cat, sort_keys=True).encode()).hexdigest()})
    a2d_ids = ["kernel_matern32", "neighbours_48", "distance_power_1.25", "blend_0.70"]; a3_ids = [action_ids[int(x)] for x in np.random.default_rng(2693).choice(len(action_ids), size=5, replace=False)]
    baselines = {"A0": [{"held_fold": int(f), "action_id": "A0", "result": {"rmse": float(_fold_metrics(oof.target, purged_banks[int(f)]["A0"], oof.fold_ids)["per_fold"][int(f)]["rmse"])}} for f in p28.base.FOLD_IDS], "A1": "A0 identity replay", "A2D": a2d_ids, "A3": a3_ids, "oracle_diagnostic": "vertical_weight_8.0"}
    safe_deltas = [r["safe_action"]["signed_delta_rmse"] for r in policy_rows]
    result = {"schema_version": SCHEMA_VERSION, "real_development": True, "feature_cache_audit": audit, "metrics": metrics, "purge_audits": purge_audits, "replay": {"fold": fold, "chosen_action_id": chosen_id, "config": saved, "prediction_hash": prediction_hash(replay[mask]), "matches": True}, "outer_fold_observations": {str(f): histories[f] for f in histories}, "prompt_observations": observations, "policy": {"safe_quantitative": policy_rows, "categorical_ablation": categorical_rows, "A1": baselines["A1"], "A2D": baselines["A2D"], "A3": baselines["A3"], "oracle_diagnostic": baselines["oracle_diagnostic"], "oracle_used_for_feedback": False, "oracle_used_for_promotion": False, "promotion": {"mean_signed_delta_rmse": float(np.mean(safe_deltas)), "positive_folds": int(sum(d < 0 for d in safe_deltas)), "folds": len(safe_deltas), "verdict": "RETAIN_FROZEN_BASELINE"}}, "held_fold_purge_reused": {"entrypoint": "p19._rows + p19._without_coordinates", "all_held_folds": True}, "frozen_holdout_opened": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
    (output_dir / "results.jsonl").write_text("\n".join(json.dumps(x, sort_keys=True, default=float) for x in policy_rows + categorical_rows) + "\n")
    (output_dir / "action_effects.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=float) + "\n")
    (output_dir / "protocol.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "budget": 4, "frozen_test_opened": False, "strategies": ["A0", "A1", "A2D", "A3", "oracle_diagnostic"]}, indent=2) + "\n")
    (output_dir / "root_cause.md").write_text("# P29 policy efficacy\n\nOracle vertical_weight_8.0 is diagnostic only; it is excluded from feedback and promotion.\n")
    (output_dir / "evidence.md").write_text("# P29 evidence\n\nReal development OOF only; frozen test opened: false.\n")
    manifest = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json": manifest[path.name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
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


def replay_predictor(
    config: Mapping[str, Any],
    *,
    coordinates: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    seismic: np.ndarray | None = None,
    query_seismic: np.ndarray | None = None,
    latent: np.ndarray | None = None,
    query_latent: np.ndarray | None = None,
    query_baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Replay the P29 predictor from prepared train/query arrays.

    ``seismic_weights`` is an ensemble of scalar metric weights, matching P21
    and P28; it is not a per-channel multiplier.  Secondary variables must be
    supplied on both sides so a serialized configuration cannot silently
    replace query covariates with zeros.  The supplied baseline is required
    whenever the configured blend does not put all mass on the local kernel.
    """

    params = config.get("parameters", config)
    c = np.asarray(coordinates, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    target = np.asarray(values, dtype=np.float64)
    if c.ndim != 2 or q.ndim != 2 or c.shape[1] != 3 or q.shape[1] != 3:
        raise ValueError("coordinates and query must have shape [n, 3]")
    if target.shape != (len(c),):
        raise ValueError("values must contain one scalar per training coordinate")

    def paired_features(
        train: np.ndarray | None,
        requested: np.ndarray | None,
        *,
        name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if train is None and requested is None:
            return np.empty((len(c), 0)), np.empty((len(q), 0))
        if train is None or requested is None:
            raise ValueError(f"{name} and query_{name} must be supplied together")
        train_array = np.asarray(train, dtype=np.float64)
        query_array = np.asarray(requested, dtype=np.float64)
        if train_array.ndim == 1:
            train_array = train_array[:, None]
        if query_array.ndim == 1:
            query_array = query_array[:, None]
        if (
            train_array.shape[0] != len(c)
            or query_array.shape[0] != len(q)
            or train_array.shape[1] != query_array.shape[1]
        ):
            raise ValueError(f"{name} train/query feature shapes are incompatible")
        return train_array, query_array

    train_seismic, requested_seismic = paired_features(
        seismic, query_seismic, name="seismic"
    )
    train_latent, requested_latent = paired_features(
        latent, query_latent, name="latent"
    )
    vertical = float(params.get("vertical_weight", 4.0))
    train_coordinate = np.array(c, copy=True)
    query_coordinate = np.array(q, copy=True)
    train_coordinate[:, 2] *= vertical
    query_coordinate[:, 2] *= vertical
    foundation_weight = float(params.get("foundation_weight", 0.1))
    seismic_weights = tuple(float(v) for v in params.get("seismic_weights", [0.0]))
    if not seismic_weights:
        raise ValueError("seismic_weights must contain at least one scalar")
    neighbours = min(int(params.get("neighbours", 64)), len(c))
    if neighbours < 1:
        raise ValueError("neighbours must be positive")
    kernel = str(params.get("kernel", "inverse_distance"))
    power = float(params.get("distance_power", 1.5))

    components = []
    for seismic_weight in seismic_weights:
        metric_c = np.column_stack(
            [
                train_coordinate,
                seismic_weight * train_seismic,
                foundation_weight * train_latent,
            ]
        )
        metric_q = np.column_stack(
            [
                query_coordinate,
                seismic_weight * requested_seismic,
                foundation_weight * requested_latent,
            ]
        )
        distance, rows = NearestNeighbors(
            n_neighbors=neighbours, n_jobs=1
        ).fit(metric_c).kneighbors(metric_q)
        bandwidth = max(float(np.median(distance)), 1e-8)
        weights = p28._kernel_weights(  # noqa: SLF001
            distance,
            family=kernel,
            power=power,
            bandwidth=bandwidth,
        )
        components.append(
            np.sum(weights * target[rows], axis=1) / np.sum(weights, axis=1)
        )
    local_prediction = np.mean(np.stack(components), axis=0)
    blend = float(params.get("blend_weight", 1.0))
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend_weight must be within [0, 1]")
    if blend < 1.0:
        if query_baseline is None:
            raise ValueError("query_baseline is required when blend_weight < 1")
        baseline = np.asarray(query_baseline, dtype=np.float64)
        if baseline.shape != (len(q),):
            raise ValueError("query_baseline must contain one scalar per query")
        return (1.0 - blend) * baseline + blend * local_prediction
    return local_prediction


def prediction_hash(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prediction, dtype=np.float64).tobytes()).hexdigest()

def build_real_action_bank(*, oof: Any, prepared: Sequence[Mapping[str, Any]], registry: Sequence[Mapping[str, Any]] | None = None) -> dict[str, np.ndarray]:
    registry = action_registry() if registry is None else registry
    specs = [("A0", p28.A0_PARAMETERS)] + [(a["action_id"], a["parameters"]) for a in registry]
    out = {name: [] for name, _ in specs}
    for name, params in specs:
        pred = np.full(len(oof.target), np.nan)
        for row in prepared:
            fold = row["fold"]
            mask = oof.fold_ids == fold.fold_id
            pred[mask] = replay_predictor(
                predictor_config(params),
                coordinates=row["train_coordinate"],
                values=fold.train_target,
                query=row["validation_coordinate"],
                seismic=row["train_seismic"],
                query_seismic=row["validation_seismic"],
                latent=row["train_latent"],
                query_latent=row["validation_latent"],
                query_baseline=oof.baseline[mask],
            )
        out[name] = pred
    if any(not np.all(np.isfinite(v)) for v in out.values()): raise RuntimeError("real action bank has incomplete predictions")
    return out

def _fold_metrics(target: np.ndarray, prediction: np.ndarray, fold_ids: np.ndarray) -> dict[str, Any]:
    rows = []
    for fold in p28.base.FOLD_IDS:
        mask = fold_ids == fold; m = p28._metrics(target[mask], prediction[mask])  # noqa: SLF001
        rows.append({"fold": int(fold), "rmse": float(m["rmse"]), "n": int(mask.sum())})
    return {"pooled": p28._metrics(target, prediction), "per_fold": rows}


def run_probe(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_action_registry()
    coordinates = np.asarray([[0., 0., 0.], [1., 0., 1.], [0., 1., 2.]], float)
    values = np.asarray([1., 2., 4.]); query = np.asarray([[.2, .2, .3], [.8, .1, 1.5]])
    a0 = predictor_config(p28.A0_PARAMETERS)
    seismic = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1], [0.7, 0.1, 0.2]])
    latent = np.asarray([[0.1], [0.9], [0.3]])
    query_seismic = np.asarray([[0.2, 0.2, 0.2], [0.6, 0.1, 0.2]])
    query_latent = np.asarray([[0.2], [0.5]])
    query_baseline = np.asarray([1.5, 3.0])
    predictions = {"A0": replay_predictor(a0, coordinates=coordinates, values=values, query=query,
                                           seismic=seismic, query_seismic=query_seismic,
                                           latent=latent, query_latent=query_latent,
                                           query_baseline=query_baseline)}
    for action in action_registry():
        predictions[action["action_id"]] = replay_predictor(
            predictor_config(action["parameters"]), coordinates=coordinates, values=values, query=query,
            seismic=seismic, query_seismic=query_seismic,
            latent=latent, query_latent=query_latent,
            query_baseline=query_baseline)
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
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--data-dir", type=Path); parser.add_argument("--stage3-root", type=Path)
    args = parser.parse_args()
    if args.data_dir and args.stage3_root: run_real_probe(data_dir=args.data_dir, stage3_root=args.stage3_root, output_dir=args.output_dir)
    else: run_probe(args.output_dir)
