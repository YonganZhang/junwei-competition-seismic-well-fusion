"""Development-only Gaia/Time-LLM pilot for reservoir properties.

The command deliberately accepts a prepared P5 development fold and a local
backbone snapshot, but no test path.  It compares a pretrained frozen GPT-2
against an architecture-matched random frozen GPT-2 and a same-input Ridge
baseline.  All preprocessing and target scaling are fit on fold-train only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
RESERVOIR_DIR = PROJECT_ROOT / "_pipelines/02_task_datasets/reservoir"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(RESERVOIR_DIR))

from _code.foundation import parameter_report  # noqa: E402
from _code.ml_framework.contracts import ModelBatch  # noqa: E402
from _models.property.gaia_timellm_gpt2 import (  # noqa: E402
    GaiaTimeLLMPropertyAdapter,
    _sequence,
)
from p5_contract import TARGETS, build_task_spec, model_to_physical  # noqa: E402
from reservoir_p5_stage2 import load_fixed_fold  # noqa: E402


SCHEMA_VERSION = 1
FROZEN_TEST_FAMILY = "15/9-F-15"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode())
    temporary.replace(path)


def _subset(batch: ModelBatch, indices: np.ndarray) -> ModelBatch:
    chosen = np.asarray(indices, dtype=np.int64)
    return ModelBatch(
        inputs={key: np.asarray(values)[chosen] for key, values in batch.inputs.items()},
        targets=(
            None
            if batch.targets is None
            else {key: np.asarray(values)[chosen] for key, values in batch.targets.items()}
        ),
        input_masks={key: np.asarray(values)[chosen] for key, values in batch.input_masks.items()},
        target_masks={key: np.asarray(values)[chosen] for key, values in batch.target_masks.items()},
        sample_ids=[batch.sample_ids[index] for index in chosen],
        groups={
            key: [values[index] for index in chosen]
            for key, values in batch.groups.items()
        },
        coordinates={key: np.asarray(values)[chosen] for key, values in batch.coordinates.items()},
        metadata=dict(batch.metadata),
    )


def _targets(batch: ModelBatch) -> tuple[np.ndarray, np.ndarray]:
    assert batch.targets is not None
    values = np.column_stack([np.asarray(batch.targets[target], dtype=float) for target in TARGETS])
    masks = np.column_stack([np.asarray(batch.target_masks[target], dtype=bool) for target in TARGETS])
    return values, masks


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0:
        return None
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"targets": {}}
    standardized_errors: list[np.ndarray] = []
    for index, target in enumerate(TARGETS):
        valid = masks[:, index]
        true_model = y_true[valid, index]
        pred_model = y_pred[valid, index]
        true_physical = model_to_physical(target, true_model, prediction=False)
        pred_physical = model_to_physical(target, pred_model, prediction=True)
        error = pred_physical - true_physical
        model_error = pred_model - true_model
        scale = float(np.std(true_model)) + 1e-8
        standardized_errors.append(model_error / scale)
        result["targets"][target] = {
            "valid_count": int(valid.sum()),
            "model_domain": {
                "MAE": float(np.mean(np.abs(model_error))),
                "RMSE": float(np.sqrt(np.mean(model_error**2))),
                "R2": _r2(true_model, pred_model),
            },
            "physical": {
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(error**2))),
                "R2": _r2(true_physical, pred_physical),
            },
        }
    result["macro_standardized_RMSE"] = float(
        np.mean([np.sqrt(np.mean(error**2)) for error in standardized_errors])
    )
    return result


def ridge_baseline(train: ModelBatch, validation: ModelBatch) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    x_train = _sequence(train).reshape(len(train.sample_ids), -1)
    x_validation = _sequence(validation).reshape(len(validation.sample_ids), -1)
    y_train, train_masks = _targets(train)
    y_validation, validation_masks = _targets(validation)
    prediction = np.zeros_like(y_validation)
    started = time.monotonic()
    for index in range(len(TARGETS)):
        estimator = Ridge(alpha=1.0)
        estimator.fit(x_train[train_masks[:, index]], y_train[train_masks[:, index], index])
        prediction[:, index] = estimator.predict(x_validation)
    return {
        "variant": "ridge_same_sequence",
        "seed": None,
        "status": "completed",
        "metrics": metrics(y_validation, prediction, validation_masks),
        "resources": {"wall_seconds": time.monotonic() - started, "peak_cuda_bytes": 0},
        "parameterization": {"pretrained_backbone": False, "foundation_model": False},
    }


def _adapter_state(adapter: GaiaTimeLLMPropertyAdapter) -> dict[str, Any]:
    return {
        key: value.detach().cpu()
        for key, value in adapter.module.state_dict().items()
        if not key.startswith("backbone.") or ".lora_" in key
    }


def run_foundation_variant(
    train: ModelBatch,
    validation: ModelBatch,
    *,
    backbone_path: Path,
    output_dir: Path,
    seed: int,
    variant: str,
    device: str,
    update_steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    if variant not in {"pretrained", "random", "pretrained_lora", "random_lora"}:
        raise ValueError(f"unknown foundation variant {variant!r}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_values, target_masks = _targets(train)
    target_mean = np.asarray(
        [target_values[target_masks[:, i], i].mean() for i in range(len(TARGETS))]
    )
    target_std = np.asarray(
        [target_values[target_masks[:, i], i].std() + 1e-8 for i in range(len(TARGETS))]
    )
    adapter = GaiaTimeLLMPropertyAdapter(
        build_task_spec(),
        backbone_path=str(backbone_path),
        target_mean=target_mean.tolist(),
        target_std=target_std.tolist(),
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        seed=seed,
        random_backbone=variant.startswith("random"),
        lora_rank=4 if variant.endswith("_lora") else 0,
        lora_last_blocks=2,
        numerical_width=64,
        prototype_subset_size=256,
        prototype_count=64,
        heads=4,
        key_width=16,
        dropout=0.05,
    )
    report = parameter_report(adapter.module)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(adapter.device)
    generator = np.random.default_rng(seed)
    order = np.arange(len(train.sample_ids))
    losses: list[float] = []
    started = time.monotonic()
    cursor = len(order)
    for _ in range(update_steps):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        losses.append(float(adapter.fit(_subset(train, indices))["loss"]))
    prediction = adapter.predict_array(validation)
    y_validation, validation_masks = _targets(validation)
    checkpoint = output_dir / "checkpoints" / f"{variant}_seed{seed}_adapter.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "variant": variant,
            "seed": seed,
            "adapter_state": _adapter_state(adapter),
            "target_mean": target_mean,
            "target_std": target_std,
            "backbone_path_persisted": False,
        },
        checkpoint,
    )
    peak = (
        int(torch.cuda.max_memory_allocated(adapter.device))
        if device.startswith("cuda")
        else 0
    )
    value = {
        "variant": f"timellm_{variant}_gpt2",
        "seed": seed,
        "status": "completed",
        "metrics": metrics(y_validation, prediction, validation_masks),
        "training": {
            "update_steps": update_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_min": min(losses),
            "validation_used_for_early_stopping": False,
        },
        "parameterization": {
            **report,
            "pretrained_backbone": variant.startswith("pretrained"),
            "backbone_frozen": True,
            "lora_enabled": variant.endswith("_lora"),
            "lora_modules": adapter.lora_modules,
            "full_parameter_finetuning": False,
        },
        "checkpoint": {
            "adapter_only": True,
            "path_persisted": False,
            "sha256": _sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_cuda_bytes": peak,
        },
    }
    del adapter
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return value


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({value["variant"] for value in results})
    summary: dict[str, Any] = {}
    for variant in variants:
        cells = [value for value in results if value["variant"] == variant]
        values = [cell["metrics"]["macro_standardized_RMSE"] for cell in cells]
        summary[variant] = {
            "cell_count": len(cells),
            "macro_standardized_RMSE_mean": float(np.mean(values)),
            "macro_standardized_RMSE_std": float(np.std(values)),
            "lower_is_better": True,
        }
    for suffix, label in (("", "frozen_pretraining_ablation"), ("_lora", "lora_pretraining_ablation")):
        pretrained = summary.get(f"timellm_pretrained{suffix}_gpt2")
        random_ablation = summary.get(f"timellm_random{suffix}_gpt2")
        if pretrained and random_ablation:
            delta = (
                random_ablation["macro_standardized_RMSE_mean"]
                - pretrained["macro_standardized_RMSE_mean"]
            )
            summary[label] = {
                "absolute_RMSE_gain": float(delta),
                "relative_gain_fraction": float(
                    delta / random_ablation["macro_standardized_RMSE_mean"]
                ),
                "pretraining_helped": bool(delta > 0),
            }
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.fold.is_file():
        raise FileNotFoundError(args.fold)
    if not args.backbone.is_dir():
        raise FileNotFoundError(args.backbone)
    train, validation, split = load_fixed_fold(args.fold)
    train_families = set(train.groups["mother_well_family"])
    validation_families = set(validation.groups["mother_well_family"])
    if train_families & validation_families:
        raise RuntimeError("train/validation family leakage")
    if FROZEN_TEST_FAMILY in train_families | validation_families:
        raise RuntimeError("frozen test family reached the P6 pilot")
    model_file = args.backbone / "model.safetensors"
    results: list[dict[str, Any]] = [ridge_baseline(train, validation)]
    for variant in args.variants.split(","):
        clean = variant.strip()
        if not clean:
            continue
        for seed in args.seeds:
            value = run_foundation_variant(
                train,
                validation,
                backbone_path=args.backbone,
                output_dir=args.output_dir,
                seed=seed,
                variant=clean,
                device=args.device,
                update_steps=args.update_steps,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            results.append(value)
            _atomic_json(args.output_dir / "partial_results.json", results)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "property",
        "stage": "P6_foundation_development_pilot",
        "evidence_state": "development_only_not_blind",
        "split_hash": split["split_hash"],
        "development_batch_sha256": _sha256(args.fold),
        "backbone": {
            "family": "GPT-2 small",
            "local_snapshot_revision": args.backbone.name,
            "model_safetensors_sha256": _sha256(model_file),
            "path_persisted": False,
            "license": "MIT",
        },
        "domain_control": {
            "system": "Gaia V2 petroleum expert/control plane",
            "numeric_checkpoint_claimed": False,
            "role": "input whitelist, units, physical bounds and task prompt",
        },
        "protocol": {
            "preprocessing_fit": "fold_train_only",
            "test_access": False,
            "test_path_argument_exists": False,
            "validation_used_for_early_stopping": False,
            "architecture_matched_random_backbone_ablation": True,
            "full_parameter_llm_finetuning": False,
        },
        "results": results,
        "aggregate": aggregate(results),
    }
    payload["evidence_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _atomic_json(args.output_dir / "property_timellm_pilot.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="pretrained,random")
    parser.add_argument("--seeds", type=lambda text: [int(v) for v in text.split(",")], default=[2693])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--update-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
