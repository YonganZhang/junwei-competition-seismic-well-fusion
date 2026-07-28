"""Development-only TabICLv2 effect check on the frozen property LOGO4 folds."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _code.ml_framework.contracts import ModelBatch
from _models.property.tabiclv2_regressor import build_model


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TARGETS = ("PHIF", "KLOGH", "SW")
FOLDS = (0, 1, 2, 3)
DEFAULT_OUTPUT = HERE / "_outputs" / "p9_tabicl_effect" / "summary.json"

stage3 = importlib.import_module(
    "_pipelines.02_task_datasets.reservoir.reservoir_p5_stage3"
)
contract = importlib.import_module(
    "_pipelines.02_task_datasets.reservoir.p5_contract"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shuffled_targets(batch: ModelBatch, *, seed: int) -> ModelBatch:
    if batch.targets is None:
        raise ValueError("shuffle control requires targets")
    rng = np.random.default_rng(seed)
    shuffled: dict[str, np.ndarray] = {}
    for target in TARGETS:
        values = np.asarray(batch.targets[target], dtype=np.float64).copy()
        mask = np.asarray(batch.target_masks[target], dtype=bool)
        values[mask] = rng.permutation(values[mask])
        shuffled[target] = values
    return ModelBatch(
        inputs=batch.inputs,
        targets=shuffled,
        input_masks=batch.input_masks,
        target_masks=batch.target_masks,
        sample_ids=batch.sample_ids,
        groups=batch.groups,
        coordinates=batch.coordinates,
        metadata={**dict(batch.metadata), "negative_control": "target_shuffle"},
    )


def _metrics(batch: ModelBatch, prediction: Any) -> dict[str, Mapping[str, float]]:
    if batch.targets is None:
        raise ValueError("metrics require labels")
    result: dict[str, Mapping[str, float]] = {}
    for target in TARGETS:
        mask = np.asarray(batch.target_masks[target], dtype=bool)
        truth_model = np.asarray(batch.targets[target], dtype=np.float64)[mask]
        predicted_model = np.asarray(prediction.raw[target], dtype=np.float64)[mask]
        truth = contract.model_to_physical(target, truth_model, prediction=False)
        predicted = contract.model_to_physical(
            target, predicted_model, prediction=True
        )
        error = predicted - truth
        result[target] = {
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "valid_count": int(mask.sum()),
        }
    return result


def _load_baselines(root: Path, split_hash: str) -> dict[str, Mapping[str, Any]]:
    values: dict[str, Mapping[str, Any]] = {}
    for target in TARGETS:
        path = root / f"leaderboard_{target.lower()}_tabular_cpu.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("split_hash") != split_hash or payload.get("test_access") is not False:
            raise RuntimeError(f"{target} strong baseline is not on the frozen development split")
        winner = payload["entries"][0]
        values[target] = {
            "model_id": winner["model_id"],
            "mean_physical_rmse": winner["mean_physical_RMSE"],
            "worst_fold_physical_rmse": winner["worst_fold_physical_RMSE"],
            "leaderboard_sha256": _sha256(path),
        }
    return values


def _aggregate(rows: list[Mapping[str, Any]], key: str) -> dict[str, float]:
    return {
        target: float(np.mean([row[key][target]["rmse"] for row in rows]))
        for target in TARGETS
    }


def run(
    *,
    development_batch: Path,
    baseline_root: Path,
    checkpoint: Path,
    output: Path = DEFAULT_OUTPUT,
    device: str = "cuda",
    n_estimators: int = 2,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_spec = contract.build_task_spec()
    fold_rows: list[dict[str, Any]] = []
    split_hash: str | None = None
    for fold_id in FOLDS:
        train, validation, evidence = stage3.load_fold(development_batch, fold_id)
        if split_hash is None:
            split_hash = evidence["split_hash"]
        elif evidence["split_hash"] != split_hash:
            raise RuntimeError("property folds do not share the frozen split hash")
        model = build_model(
            task_spec,
            checkpoint_path=checkpoint,
            seed=2693 + fold_id,
            n_estimators=n_estimators,
            batch_size=1,
            device=device,
            offload_mode="auto",
        )
        model.fit(train)
        actual = _metrics(validation, model.predict(validation))
        control = build_model(
            task_spec,
            checkpoint_path=checkpoint,
            seed=12693 + fold_id,
            n_estimators=n_estimators,
            batch_size=1,
            device=device,
            offload_mode="auto",
        )
        control.fit(_shuffled_targets(train, seed=12693 + fold_id))
        shuffled = _metrics(validation, control.predict(validation))
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_samples": len(train.sample_ids),
                "validation_samples": len(validation.sample_ids),
                "train_groups": evidence["train_groups"],
                "validation_groups": evidence["validation_groups"],
                "actual": actual,
                "target_shuffle_control": shuffled,
            }
        )
    assert split_hash is not None
    baselines = _load_baselines(Path(baseline_root), split_hash)
    actual_mean = _aggregate(fold_rows, "actual")
    control_mean = _aggregate(fold_rows, "target_shuffle_control")
    comparisons = {
        target: {
            "tabicl_mean_physical_rmse": actual_mean[target],
            "strong_baseline_model_id": baselines[target]["model_id"],
            "strong_baseline_mean_physical_rmse": baselines[target][
                "mean_physical_rmse"
            ],
            "relative_rmse_change_vs_strong_baseline": (
                actual_mean[target] / baselines[target]["mean_physical_rmse"] - 1.0
            ),
            "target_shuffle_control_rmse": control_mean[target],
            "pretrained_better_than_target_shuffle": (
                actual_mean[target] < control_mean[target]
            ),
            "pretrained_better_than_strong_baseline": (
                actual_mean[target] < baselines[target]["mean_physical_rmse"]
            ),
        }
        for target in TARGETS
    }
    all_baselines_beaten = all(
        row["pretrained_better_than_strong_baseline"]
        for row in comparisons.values()
    )
    summary = {
        "schema_version": "property-p9-tabicl-effect/v1",
        "model": {
            "model_id": "jingang/TabICL",
            "checkpoint_sha256": _sha256(checkpoint),
            "real_pretrained_weights_loaded": True,
            "n_estimators": n_estimators,
        },
        "evaluation": {
            "split_hash": split_hash,
            "folds": list(FOLDS),
            "same_fold_as_strong_baselines": True,
            "frozen_test_accessed": False,
            "known_holdout_accessed": False,
        },
        "folds": fold_rows,
        "strong_baselines": baselines,
        "comparisons": comparisons,
        "decision": {
            "state": (
                "EFFECT_SUPPORTED_NOT_PROMOTED"
                if all_baselines_beaten
                else "CONNECTED_NO_STRONG_BASELINE_WIN"
            ),
            "default_enabled": False,
            "reason": (
                "random-init same-architecture evidence is still required for promotion"
                if all_baselines_beaten
                else "TabICLv2 did not beat every target-specific strong tree baseline"
            ),
        },
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "raw_predictions_persisted": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-batch", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=2)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
