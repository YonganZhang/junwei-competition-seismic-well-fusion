"""Frozen-fold MOMENT effect check for the nine-class lithofacies task."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from _models.lithofacies.moment_depth import build_model


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
FOLDS = (0, 1, 2, 3)
UPDATES = 40
BATCH_SIZE = 32
DEFAULT_OUTPUT = HERE / "_outputs" / "p9_moment_effect" / "summary.json"

stage3 = importlib.import_module(
    "_pipelines.02_task_datasets.lithofacies.lithofacies_p5_stage3"
)
contract = importlib.import_module(
    "_pipelines.02_task_datasets.lithofacies.p4_contract"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(well: np.ndarray, seismic: np.ndarray) -> np.ndarray:
    well_values = np.asarray(well, dtype=np.float32)
    seismic_values = np.asarray(seismic, dtype=np.float32)
    flat_seismic = seismic_values.reshape(len(seismic_values), 9, 33)
    result = np.concatenate((well_values, flat_seismic), axis=1)
    if result.shape[1:] != (35, 33) or not np.isfinite(result).all():
        raise ValueError(f"invalid MOMENT fold input: {result.shape}")
    return result


def _randomize_frozen_backbone(model: Any, *, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("pipeline.head."):
                continue
            if parameter.ndim >= 2:
                values = torch.empty(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
                values.normal_(mean=0.0, std=0.02, generator=generator)
            elif "layer_norm" in name or "final_layer_norm" in name:
                values = torch.ones(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
            else:
                values = torch.zeros(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
            parameter.copy_(values.to(parameter.device))


def _train_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    *,
    snapshot: Path,
    device: str,
    seed: int,
    random_init: bool,
    class_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(
        contract.lithofacies_task_spec(),
        snapshot_path=snapshot,
        device=device,
        freeze_encoder=True,
        freeze_embedder=True,
    )
    if random_init:
        _randomize_frozen_backbone(model, seed=seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    model.train()
    last_loss = float("nan")
    for _ in range(UPDATES):
        indices = rng.choice(
            len(train_x), size=min(BATCH_SIZE, len(train_x)), replace=False
        )
        x = torch.as_tensor(train_x[indices], dtype=torch.float32, device=device)
        y = torch.as_tensor(train_y[indices], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("MOMENT training loss is non-finite")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(validation_x), BATCH_SIZE):
            x = torch.as_tensor(
                validation_x[start : start + BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(x).detach().cpu().numpy())
    return np.concatenate(outputs), last_loss


def run(
    *,
    development_batch: Path,
    leaderboard: Path,
    snapshot: Path,
    output: Path = DEFAULT_OUTPUT,
    device: str = "cuda:0",
) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, manifest = stage3.load_stage3_batch(development_batch)
    baseline = json.loads(leaderboard.read_text(encoding="utf-8"))
    if (
        baseline.get("split_hash") != manifest["split_hash"]
        or baseline.get("frozen_test_accessed") is not False
    ):
        raise RuntimeError("lithofacies baseline is not the frozen development split")
    winner = baseline["entries"][0]
    folds: list[dict[str, Any]] = []
    for fold_id in FOLDS:
        fold = stage3._fold_arrays(arrays, fold_id)
        train_x = _inputs(fold["p_train_well"], fold["p_train_seismic"])
        validation_x = _inputs(
            fold["p_validation_well"], fold["p_validation_seismic"]
        )
        train_y = np.asarray(fold["p_train_labels"], dtype=np.int64)
        validation_y = np.asarray(
            fold["p_validation_labels"], dtype=np.int64
        )
        seed = 2693 + fold_id
        pretrained_logits, pretrained_loss = _train_predict(
            train_x,
            train_y,
            validation_x,
            snapshot=snapshot,
            device=device,
            seed=seed,
            random_init=False,
            class_weights=fold["class_weights"],
        )
        random_logits, random_loss = _train_predict(
            train_x,
            train_y,
            validation_x,
            snapshot=snapshot,
            device=device,
            seed=seed,
            random_init=True,
            class_weights=fold["class_weights"],
        )
        pretrained_metrics = contract.classification_metrics_from_logits(
            validation_y, pretrained_logits
        )
        random_metrics = contract.classification_metrics_from_logits(
            validation_y, random_logits
        )
        folds.append(
            {
                "fold_id": fold_id,
                "train_samples": len(train_y),
                "validation_samples": len(validation_y),
                "pretrained_fixed_schema_macro_f1": pretrained_metrics[
                    "fixed_schema_macro_f1"
                ],
                "random_init_fixed_schema_macro_f1": random_metrics[
                    "fixed_schema_macro_f1"
                ],
                "pretrained_last_train_loss": pretrained_loss,
                "random_init_last_train_loss": random_loss,
            }
        )
    pretrained_mean = float(
        np.mean([row["pretrained_fixed_schema_macro_f1"] for row in folds])
    )
    random_mean = float(
        np.mean([row["random_init_fixed_schema_macro_f1"] for row in folds])
    )
    strong_baseline = float(winner["fixed_schema_macro_f1_mean"])
    summary = {
        "schema_version": "lithofacies-p9-moment-effect/v1",
        "model": {
            "model_id": "AutonLab/MOMENT-1-base",
            "weights_sha256": _sha256(snapshot / "model.safetensors"),
            "real_pretrained_weights_loaded": True,
            "trainable_scope": "classification_head_only",
        },
        "evaluation": {
            "split_hash": manifest["split_hash"],
            "folds": list(FOLDS),
            "updates_per_fold": UPDATES,
            "batch_size": BATCH_SIZE,
            "frozen_test_family": manifest["frozen_test_family"],
            "frozen_test_accessed": False,
        },
        "fold_results": folds,
        "comparison": {
            "pretrained_macro_fold_f1": pretrained_mean,
            "same_architecture_random_init_macro_fold_f1": random_mean,
            "pretrained_minus_random_init": pretrained_mean - random_mean,
            "strong_baseline_model_id": winner["model_id"],
            "strong_baseline_macro_f1": strong_baseline,
            "pretrained_minus_strong_baseline": pretrained_mean - strong_baseline,
        },
        "decision": {
            "state": (
                "EFFECT_AND_BASELINE_WIN"
                if pretrained_mean > random_mean
                and pretrained_mean > strong_baseline
                else "CONNECTED_NO_PROMOTION"
            ),
            "default_enabled": (
                pretrained_mean > random_mean and pretrained_mean > strong_baseline
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
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
