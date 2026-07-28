"""Frozen spatial-fold SAM 2.1 effect check for one facies task."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _models.facies.sam2_semantic import build_model


HERE = Path(__file__).resolve().parent
FOLDS = (0, 1, 2, 3, 4)
DEFAULT_OUTPUT_ROOT = HERE / "_outputs" / "p9_sam2_effect"

stage3 = importlib.import_module(
    "_pipelines.02_task_datasets.facies.facies_p5_stage3"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _randomize_encoder(model: Any, *, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.backbone.image_encoder.named_parameters():
            if parameter.ndim >= 2:
                values = torch.empty(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
                values.normal_(0.0, 0.02, generator=generator)
            elif "norm" in name.lower():
                values = torch.ones(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
            else:
                values = torch.zeros(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
            parameter.copy_(values.to(parameter.device))


def _miou(labels: np.ndarray, logits: np.ndarray, classes: int) -> float:
    prediction = np.asarray(logits).argmax(axis=1)
    truth = np.asarray(labels, dtype=np.int64)
    values: list[float] = []
    for label_id in range(classes):
        target = truth == label_id
        guess = prediction == label_id
        union = np.logical_or(target, guess).sum()
        values.append(
            float(np.logical_and(target, guess).sum() / union) if union else 0.0
        )
    return float(np.mean(values))


def _train_predict(
    prepared: Any,
    *,
    source_root: Path,
    checkpoint: Path,
    device: str,
    seed: int,
    random_init: bool,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(
        stage3.get_task_spec(prepared.task_id),
        source_root=source_root,
        checkpoint_path=checkpoint,
        num_classes=prepared.num_classes,
        device=device,
        freeze_encoder=True,
    )
    if random_init:
        _randomize_encoder(model, seed=seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.0,
    )
    weights = torch.as_tensor(
        prepared.class_weights, dtype=torch.float32, device=device
    )
    rng = np.random.default_rng(seed)
    model.train()
    last_loss = float("nan")
    for _ in range(40):
        indices = rng.choice(
            len(prepared.train_images),
            size=min(2, len(prepared.train_images)),
            replace=False,
        )
        image = torch.as_tensor(
            prepared.train_images[indices], dtype=torch.float32, device=device
        )
        label = torch.as_tensor(
            prepared.train_labels[indices], dtype=torch.long, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = torch.nn.functional.cross_entropy(logits, label, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("SAM2 semantic loss is non-finite")
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared.validation_images), 2):
            image = torch.as_tensor(
                prepared.validation_images[start : start + 2],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(image).detach().cpu().numpy())
    return np.concatenate(outputs), last_loss


def run(
    *,
    task_id: str,
    manifest: Path,
    processed_root: Path,
    baseline_leaderboard: Path,
    source_root: Path,
    checkpoint: Path,
    output: Path | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    started = time.perf_counter()
    baseline = json.loads(baseline_leaderboard.read_text(encoding="utf-8"))
    winner = baseline["entries"][0]
    folds: list[dict[str, Any]] = []
    stable_hash: str | None = None
    budget = stage3.Stage3Budget()
    for fold_id in FOLDS:
        prepared = stage3.prepare_fold(
            task_id=task_id,
            fold_id=fold_id,
            manifest_path=manifest,
            processed_root=processed_root,
            budget=budget,
        )
        if stable_hash is None:
            stable_hash = prepared.manifest_stable_hash
        elif stable_hash != prepared.manifest_stable_hash:
            raise RuntimeError("facies folds changed manifest identity")
        pretrained_logits, pretrained_loss = _train_predict(
            prepared,
            source_root=source_root,
            checkpoint=checkpoint,
            device=device,
            seed=2693 + fold_id,
            random_init=False,
        )
        random_logits, random_loss = _train_predict(
            prepared,
            source_root=source_root,
            checkpoint=checkpoint,
            device=device,
            seed=2693 + fold_id,
            random_init=True,
        )
        folds.append(
            {
                "fold_id": fold_id,
                "train_samples": len(prepared.train_images),
                "validation_samples": len(prepared.validation_images),
                "pretrained_miou": _miou(
                    prepared.validation_labels,
                    pretrained_logits,
                    prepared.num_classes,
                ),
                "random_init_miou": _miou(
                    prepared.validation_labels,
                    random_logits,
                    prepared.num_classes,
                ),
                "pretrained_last_train_loss": pretrained_loss,
                "random_init_last_train_loss": random_loss,
            }
        )
    pretrained_mean = float(np.mean([row["pretrained_miou"] for row in folds]))
    random_mean = float(np.mean([row["random_init_miou"] for row in folds]))
    strong_baseline = float(winner["mean_miou"])
    wins = pretrained_mean > random_mean and pretrained_mean > strong_baseline
    result = {
        "schema_version": "facies-p9-sam2-effect/v1",
        "task_id": task_id,
        "model": {
            "model_id": "facebook/sam2.1-hiera-base-plus",
            "checkpoint_sha256": _sha256(checkpoint),
            "real_pretrained_weights_loaded": True,
            "trainable_scope": "fpn_projections_and_semantic_head",
        },
        "evaluation": {
            "manifest_stable_hash": stable_hash,
            "folds": list(FOLDS),
            "max_updates": 40,
            "max_train_samples": 32,
            "max_validation_samples": 16,
            "frozen_test_accessed": False,
        },
        "fold_results": folds,
        "comparison": {
            "pretrained_macro_fold_miou": pretrained_mean,
            "same_architecture_random_init_macro_fold_miou": random_mean,
            "pretrained_minus_random_init": pretrained_mean - random_mean,
            "strong_baseline_model_id": winner["model_id"],
            "strong_baseline_macro_fold_miou": strong_baseline,
            "pretrained_minus_strong_baseline": pretrained_mean - strong_baseline,
        },
        "decision": {
            "state": "EFFECT_AND_BASELINE_WIN" if wins else "CONNECTED_NO_PROMOTION",
            "default_enabled": wins,
        },
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "raw_predictions_persisted": False,
        },
    }
    output = output or DEFAULT_OUTPUT_ROOT / task_id / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--baseline-leaderboard", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
