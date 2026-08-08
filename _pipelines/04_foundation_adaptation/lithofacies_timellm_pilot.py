"""Development-only Time-LLM/LoRA pilot for GM09 lithofacies classification."""
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
LITHO_DIR = PROJECT_ROOT / "_pipelines/02_task_datasets/lithofacies"
for root in (PROJECT_ROOT, LITHO_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from _code.foundation import parameter_report  # noqa: E402
from _models.lithofacies.gaia_timellm_gpt2 import build_model  # noqa: E402
from p4_contract import lithofacies_task_spec  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    probabilities = _softmax(logits)
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((9, 9), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    f1_values: list[float] = []
    iou_values: list[float] = []
    for class_id in range(9):
        tp = float(confusion[class_id, class_id])
        fp = float(confusion[:, class_id].sum() - tp)
        fn = float(confusion[class_id, :].sum() - tp)
        f1_values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
        iou_values.append(0.0 if tp + fp + fn == 0 else tp / (tp + fp + fn))
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return {
        "accuracy": float(correct.mean()),
        "macro_F1_fixed9": float(np.mean(f1_values)),
        "mIoU_fixed9": float(np.mean(iou_values)),
        "NLL": float(
            -np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
        ),
        "ECE_10bin": float(ece),
        "confusion_matrix": confusion.tolist(),
        "validation_samples": int(len(labels)),
    }


def _features(well: np.ndarray, seismic: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            well.transpose(0, 2, 1),
            seismic.transpose(0, 3, 1, 2).reshape(len(seismic), 33, 9),
        ),
        axis=-1,
    )


def logistic_baseline(arrays: dict[str, np.ndarray], seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    train = _features(arrays["p_train_well"], arrays["p_train_seismic"]).reshape(
        len(arrays["p_train_labels"]), -1
    )
    validation = _features(
        arrays["p_validation_well"], arrays["p_validation_seismic"]
    ).reshape(len(arrays["p_validation_labels"]), -1)
    started = time.monotonic()
    estimator = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    estimator.fit(train, arrays["p_train_labels"])
    partial = estimator.decision_function(validation)
    logits = np.full((len(validation), 9), float(partial.min() - 20.0))
    logits[:, estimator.classes_.astype(int)] = partial
    return {
        "variant": "logistic_same_sequence",
        "seed": seed,
        "status": "completed",
        "metrics": metrics(arrays["p_validation_labels"], logits),
        "resources": {"wall_seconds": time.monotonic() - started, "peak_cuda_bytes": 0},
        "parameterization": {"foundation_model": False, "pretrained_backbone": False},
    }


def _adapter_state(model: Any) -> dict[str, Any]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".backbone." not in key or ".lora_" in key
    }


def run_variant(
    arrays: dict[str, np.ndarray],
    *,
    backbone: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device_name: str,
    update_steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    if variant not in {"pretrained", "random", "pretrained_lora", "random_lora"}:
        raise ValueError(variant)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device_name)
    model = build_model(
        lithofacies_task_spec(),
        num_classes=9,
        well_log_shape=(26, 33),
        seismic_shape=(3, 3, 33),
        backbone_path=str(backbone),
        random_backbone=variant.startswith("random"),
        lora_rank=4 if variant.endswith("_lora") else 0,
        lora_last_blocks=2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    class_weights = torch.as_tensor(arrays["class_weights"], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    if device_name.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    generator = np.random.default_rng(seed)
    order = np.arange(len(arrays["p_train_labels"]))
    cursor = len(order)
    losses: list[float] = []
    started = time.monotonic()
    model.train()
    for _ in range(update_steps):
        if cursor + batch_size > len(order):
            generator.shuffle(order)
            cursor = 0
        selected = order[cursor : cursor + batch_size]
        cursor += batch_size
        well = torch.as_tensor(
            arrays["p_train_well"][selected], dtype=torch.float32, device=device
        )
        seismic = torch.as_tensor(
            arrays["p_train_seismic"][selected], dtype=torch.float32, device=device
        )
        labels = torch.as_tensor(
            arrays["p_train_labels"][selected], dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(well, seismic), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite lithofacies loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arrays["p_validation_labels"]), batch_size):
            stop = start + batch_size
            well = torch.as_tensor(
                arrays["p_validation_well"][start:stop], dtype=torch.float32, device=device
            )
            seismic = torch.as_tensor(
                arrays["p_validation_seismic"][start:stop], dtype=torch.float32, device=device
            )
            logits.append(model(well, seismic).detach().cpu().numpy())
    prediction = np.concatenate(logits)
    checkpoint = output_dir / "checkpoints" / f"{variant}_seed{seed}_adapter.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "variant": variant,
            "seed": seed,
            "adapter_state": _adapter_state(model),
            "backbone_path_persisted": False,
        },
        checkpoint,
    )
    report = parameter_report(model)
    value = {
        "variant": f"timellm_{variant}_gpt2",
        "seed": seed,
        "status": "completed",
        "metrics": metrics(arrays["p_validation_labels"], prediction),
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
            "lora_modules": model.lora_modules,
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
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device_name.startswith("cuda")
                else 0
            ),
        },
    }
    del model
    if device_name.startswith("cuda"):
        torch.cuda.empty_cache()
    return value


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in sorted({result["variant"] for result in results}):
        cells = [result for result in results if result["variant"] == variant]
        output[variant] = {
            "cell_count": len(cells),
            "accuracy_mean": float(np.mean([cell["metrics"]["accuracy"] for cell in cells])),
            "accuracy_std": float(np.std([cell["metrics"]["accuracy"] for cell in cells])),
            "macro_F1_fixed9_mean": float(
                np.mean([cell["metrics"]["macro_F1_fixed9"] for cell in cells])
            ),
        }
    for suffix, label in (("", "frozen_pretraining_ablation"), ("_lora", "lora_pretraining_ablation")):
        pretrained = output.get(f"timellm_pretrained{suffix}_gpt2")
        random_cell = output.get(f"timellm_random{suffix}_gpt2")
        if pretrained and random_cell:
            output[label] = {
                "accuracy_gain": pretrained["accuracy_mean"] - random_cell["accuracy_mean"],
                "macro_F1_gain": (
                    pretrained["macro_F1_fixed9_mean"] - random_cell["macro_F1_fixed9_mean"]
                ),
                "pretraining_helped_accuracy": pretrained["accuracy_mean"] > random_cell["accuracy_mean"],
            }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.batch, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "manifest"}
        manifest = json.loads(str(archive["manifest"]))
    if manifest.get("frozen_test_accessed"):
        raise RuntimeError("frozen test accessed in development batch")
    partial_path = args.output_dir / "partial_results.json"
    if args.resume and partial_path.is_file():
        results = json.loads(partial_path.read_text(encoding="utf-8"))
    else:
        results = [logistic_baseline(arrays, args.seeds[0])]
    completed = {(value["variant"], value.get("seed")) for value in results}
    for variant in [value.strip() for value in args.variants.split(",") if value.strip()]:
        for seed in args.seeds:
            result_id = (f"timellm_{variant}_gpt2", seed)
            if result_id in completed:
                continue
            results.append(
                run_variant(
                    arrays,
                    backbone=args.backbone,
                    output_dir=args.output_dir,
                    variant=variant,
                    seed=seed,
                    device_name=args.device,
                    update_steps=args.update_steps,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                )
            )
            _atomic_json(args.output_dir / "partial_results.json", results)
            completed.add(result_id)
    split_payload = {
        "stage1_fold_id": manifest["stage1_fold_id"],
        "stage1_train_groups": manifest["stage1_train_groups"],
        "stage1_validation_groups": manifest["stage1_validation_groups"],
        "all_folds": manifest["all_folds"],
    }
    split_hash = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "track_id": "lithofacies",
        "stage": "P6_foundation_development_pilot",
        "evidence_state": "development_only_not_blind",
        "split_hash": split_hash,
        "development_batch_sha256": _sha256(args.batch),
        "backbone": {
            "family": "GPT-2 small",
            "local_snapshot_revision": args.backbone.name,
            "model_safetensors_sha256": _sha256(args.backbone / "model.safetensors"),
            "path_persisted": False,
            "license": "MIT",
        },
        "domain_control": {
            "system": "Gaia V2 petroleum expert/control plane",
            "role": "GM09 ontology, fixed output schema and leakage constraints",
            "numeric_checkpoint_claimed": False,
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
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(args.output_dir / "lithofacies_timellm_pilot.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", default="pretrained_lora,random_lora")
    parser.add_argument("--seeds", type=lambda value: [int(v) for v in value.split(",")], default=[2693])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--update-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
