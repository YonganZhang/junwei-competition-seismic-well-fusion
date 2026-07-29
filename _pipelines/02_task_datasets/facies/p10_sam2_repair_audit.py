#!/usr/bin/env python3
"""Development-only SAM2 repair audit for facies.

This script keeps the original pretrained SAM2 adapter as the before state and
adds a gated residual correction as the candidate repair.  The residual branch
is deliberately unable to erase the frozen base adapter: the final logits are

    base_logits + sigmoid(gate) * residual_logits

where the base adapter remains frozen during the repair run.

The run is constrained to locked development folds only.  No test or holdout
artifacts are opened, and the output is a compact JSON summary for downstream
reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
for import_root in (str(PROJECT_ROOT), str(HERE)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _models.facies.sam2_semantic import build_model as build_sam2_adapter
from _models.gaia_dagt.foundation_runtime import (
    insert_import_root,
    verify_checkpoint,
    verify_git_source,
)
DEFAULT_OUTPUT_ROOT = HERE / "_outputs" / "p10_sam2_repair_audit"
SOURCE_ROOT = Path("/mnt/data/yongan-admin-2/.cache/upstream/sam2")
CHECKPOINT = Path(
    "/mnt/data/yongan-admin-2/.cache/huggingface/hub/models--facebook--sam2.1-hiera-base-plus/"
    "blobs/a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)
FOLDS = (0, 4)
ROOT_SEED = 2693
VARIANTS = ("pretrained_adapter", "gated_residual_repair", "random_init_control")

stage3 = importlib.import_module("_pipelines.02_task_datasets.facies.facies_p5_stage3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _randomize_encoder(model: Any, *, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.backbone.image_encoder.named_parameters():
            if parameter.ndim >= 2:
                values = torch.empty(parameter.shape, dtype=parameter.dtype, device="cpu")
                values.normal_(0.0, 0.02, generator=generator)
            elif "norm" in name.lower():
                values = torch.ones(parameter.shape, dtype=parameter.dtype, device="cpu")
            else:
                values = torch.zeros(parameter.shape, dtype=parameter.dtype, device="cpu")
            parameter.copy_(values.to(parameter.device))


def _miou(labels: np.ndarray, logits: np.ndarray, classes: int) -> float:
    prediction = np.asarray(logits).argmax(axis=1)
    truth = np.asarray(labels, dtype=np.int64)
    values: list[float] = []
    for label_id in range(classes):
        target = truth == label_id
        guess = prediction == label_id
        union = np.logical_or(target, guess).sum()
        values.append(float(np.logical_and(target, guess).sum() / union) if union else 0.0)
    return float(np.mean(values))


class _ResidualProbe(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(num_classes, max(8, num_classes), kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(max(8, num_classes), num_classes, kernel_size=1),
        )
        self.logit_gate = torch.nn.Parameter(torch.tensor(-1.5))

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        residual = self.head(base_logits)
        gate = torch.sigmoid(self.logit_gate)
        return base_logits + gate * residual


class _FrozenResidualRepair(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, num_classes: int) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.probe = _ResidualProbe(num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_logits = self.base_model(inputs)
        return self.probe(base_logits)


@dataclass(frozen=True)
class RunResult:
    fold_id: int
    seed: int
    train_samples: int
    validation_samples: int
    pretrained_miou: float
    repair_miou: float
    random_init_miou: float
    pretrained_last_train_loss: float
    repair_last_train_loss: float
    random_init_last_train_loss: float


def _build_base_model(
    *,
    task_id: str,
    num_classes: int,
    device: str,
    freeze_encoder: bool = True,
) -> torch.nn.Module:
    task_spec = stage3.get_task_spec(task_id)
    model = build_sam2_adapter(
        task_spec,
        source_root=SOURCE_ROOT,
        checkpoint_path=CHECKPOINT,
        num_classes=num_classes,
        device=device,
        freeze_encoder=freeze_encoder,
    )
    return model


def _train_and_score(
    model: torch.nn.Module,
    prepared: Any,
    *,
    device: str,
    seed: int,
    updates: int = 40,
) -> tuple[np.ndarray, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.0,
    )
    weights = torch.as_tensor(prepared.class_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    model.train()
    last_loss = float("nan")
    for _ in range(updates):
        indices = rng.choice(len(prepared.train_images), size=min(2, len(prepared.train_images)), replace=False)
        image = torch.as_tensor(prepared.train_images[indices], dtype=torch.float32, device=device)
        label = torch.as_tensor(prepared.train_labels[indices], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = torch.nn.functional.cross_entropy(logits, label, weight=weights)
        if not torch.isfinite(loss):
            raise FloatingPointError("SAM2 repair loss is non-finite")
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


def _strong_baseline(task_id: str, fold_id: int) -> float:
    if task_id == "facies_f3":
        path = HERE / "_outputs" / "p5_stage3" / "facies_f3_scratch_leaderboard.json"
    else:
        path = HERE / "_outputs" / "p5_stage3" / "facies_penobscot_scratch_leaderboard.json"
    leaderboard = json.loads(path.read_text(encoding="utf-8"))
    return float(leaderboard["entries"][0]["fold_mean_miou"][str(fold_id)])


def run(
    *,
    task_id: str,
    manifest: Path,
    processed_root: Path,
    output: Path | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    started = time.perf_counter()
    source_root = verify_git_source(SOURCE_ROOT, "2b90b9f5ceec907a1c18123530e92e794ad901a4")
    checkpoint = verify_checkpoint("facies", CHECKPOINT)
    insert_import_root(source_root, "sam2")
    folds: list[dict[str, Any]] = []
    budget = stage3.Stage3Budget()
    stable_hash: str | None = None
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
        seed = ROOT_SEED + fold_id

        pretrained_adapter = _build_base_model(
            task_id=task_id, num_classes=prepared.num_classes, device=device, freeze_encoder=True
        )
        pretrained_logits, pretrained_loss = _train_and_score(
            pretrained_adapter, prepared, device=device, seed=seed
        )

        repair_base = _build_base_model(
            task_id=task_id, num_classes=prepared.num_classes, device=device, freeze_encoder=True
        )
        repair_model = _FrozenResidualRepair(repair_base, prepared.num_classes).to(device)
        repair_logits, repair_loss = _train_and_score(
            repair_model, prepared, device=device, seed=seed
        )

        random_base = _build_base_model(
            task_id=task_id, num_classes=prepared.num_classes, device=device, freeze_encoder=True
        )
        _randomize_encoder(random_base, seed=seed)
        random_model = _FrozenResidualRepair(random_base, prepared.num_classes).to(device)
        random_logits, random_loss = _train_and_score(
            random_model, prepared, device=device, seed=seed
        )

        strong_baseline = _strong_baseline(task_id, fold_id)
        folds.append(
            {
                "fold_id": fold_id,
                "seed": seed,
                "train_samples": len(prepared.train_images),
                "validation_samples": len(prepared.validation_images),
                "pretrained_adapter_miou": _miou(
                    prepared.validation_labels, pretrained_logits, prepared.num_classes
                ),
                "gated_residual_repair_miou": _miou(
                    prepared.validation_labels, repair_logits, prepared.num_classes
                ),
                "random_init_control_miou": _miou(
                    prepared.validation_labels, random_logits, prepared.num_classes
                ),
                "pretrained_adapter_last_train_loss": pretrained_loss,
                "gated_residual_repair_last_train_loss": repair_loss,
                "random_init_control_last_train_loss": random_loss,
                "strong_baseline_miou": strong_baseline,
            }
        )

    pretrained_mean = float(np.mean([row["pretrained_adapter_miou"] for row in folds]))
    repair_mean = float(np.mean([row["gated_residual_repair_miou"] for row in folds]))
    random_mean = float(np.mean([row["random_init_control_miou"] for row in folds]))
    strong_mean = float(np.mean([row["strong_baseline_miou"] for row in folds]))
    result = {
        "schema_version": "facies-p10-sam2-repair-audit/v1",
        "task_id": task_id,
        "model": {
            "model_id": "facebook/sam2.1-hiera-base-plus",
            "checkpoint_sha256": _sha256(checkpoint),
            "source_root": str(source_root),
            "real_pretrained_weights_loaded": True,
            "trainable_scope_before": "fpn_projections_and_semantic_head",
            "trainable_scope_after": "frozen_base_plus_gated_residual_probe",
            "repair_gate_bounds": [0.0, 1.0],
            "residual_cannot_erase_base": True,
        },
        "audit": {
            "input_channels": "[B,1,H,W]",
            "channel_scaling": "clamp[-5,5]->rescale_to_[0,1]->repeat_to_3_channels->imagenet_normalize",
            "native_preprocessing": "sam2.1_hiera_b+ official image encoder normalization",
            "prompt_generation": "none_no_validation_truth_path_exists",
            "label_mapping": "task-specific independent class IDs from locked schemas",
            "decoder": "frozen base adapter logits plus gated residual probe",
            "peft": "base adapter frozen; only residual head and scalar gate train",
            "fusion": "additive residual fusion with sigmoid gate",
            "loss": "weighted_cross_entropy_on_raw_logits",
            "postprocess": "argmax only at evaluation; no threshold tuning",
            "eval_parity": "same locked development folds and metrics as p9 evidence",
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
            "pretrained_adapter_macro_fold_miou": pretrained_mean,
            "gated_residual_repair_macro_fold_miou": repair_mean,
            "random_init_control_macro_fold_miou": random_mean,
            "strong_baseline_macro_fold_miou": strong_mean,
            "repair_minus_pretrained": repair_mean - pretrained_mean,
            "repair_minus_strong_baseline": repair_mean - strong_mean,
            "pretrained_minus_strong_baseline": pretrained_mean - strong_mean,
            "random_minus_pretrained": random_mean - pretrained_mean,
        },
        "decision": {
            "state": "REPAIR_PROMOTED" if repair_mean > max(pretrained_mean, strong_mean) else "NON_BENEFICIAL",
            "default_enabled": repair_mean > max(pretrained_mean, strong_mean),
        },
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "raw_predictions_persisted": False,
        },
    }
    output = output or DEFAULT_OUTPUT_ROOT / task_id / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
