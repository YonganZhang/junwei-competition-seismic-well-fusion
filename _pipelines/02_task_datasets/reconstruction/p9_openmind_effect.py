#!/usr/bin/env python3
"""Locked-development OpenMind MAE effect check for strict 3-D reconstruction.

The pretrained and random-initialized encoders receive the same four volume
patches, 512 target voxels, 20 updates, and 2,048 validation voxels per fold.
Only P4 strict development blocks are loaded. Frozen test and guard blocks are
not accepted as runner inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from _models.reconstruction.openmind_mae import build_model


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "_outputs" / "p9_openmind_effect" / "summary.json"
FOLDS = (0, 1, 2, 3, 4)
TRAIN_PATCHES = 4
TARGETS_PER_PATCH = 128
VALIDATION_VOXELS = 2048
UPDATES = 20

p4 = importlib.import_module(
    "_pipelines.02_task_datasets.reconstruction.p4_reconstruction"
)
stage1 = importlib.import_module(
    "_pipelines.02_task_datasets.reconstruction.p5_stage1"
)
stage2 = importlib.import_module(
    "_pipelines.02_task_datasets.reconstruction.reconstruction_p5_stage2"
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
        for name, parameter in model.network.encoder.named_parameters():
            if parameter.ndim >= 2:
                values = torch.empty(parameter.shape, dtype=parameter.dtype)
                values.normal_(0.0, 0.02, generator=generator)
            elif "norm" in name.lower():
                values = torch.ones(parameter.shape, dtype=parameter.dtype)
            else:
                values = torch.zeros(parameter.shape, dtype=parameter.dtype)
            parameter.copy_(values.to(parameter.device))


def _volume_batches(
    fold: Any,
    development: Sequence[Any],
) -> tuple[list[Any], Any]:
    spec = p4.task_spec("strict")
    prepared = p4.prepare_fold("strict", fold, development)
    if prepared.constraint_audit["constraints_supplied_to_model"] != 0:
        raise RuntimeError("strict OpenMind fold received target-derived constraints")
    by_id = {record.sample_id: record for record in development}
    train_records = tuple(
        by_id[sample_id] for sample_id in fold.purge["effective_train_sample_ids"]
    )
    validation_records = tuple(
        by_id[sample_id] for sample_id in fold.validation_sample_ids
    )
    bundle = stage1.DevelopmentBundle(
        "strict", fold, prepared, train_records, validation_records
    )
    ranked_train = sorted(
        train_records,
        key=lambda item: (-int(np.sum(item.seismic_patch[8] > 0.5)), item.sample_id),
    )[:TRAIN_PATCHES]
    train_batches: list[Any] = []
    for record in ranked_train:
        batch = stage1._record_feature_volume(  # noqa: SLF001
            spec, bundle, record, validation=False
        )
        active = np.flatnonzero(
            np.asarray(batch.target_masks[spec.targets[0]], dtype=bool).reshape(-1)
        )
        selected = active[stage2._sample_indices(active.size, TARGETS_PER_PATCH)]
        train_batches.append(
            stage2._replace_batch_mask(  # noqa: SLF001
                batch,
                spec,
                selected,
                metadata={"fixed_train_target_voxels": len(selected)},
            )
        )
    if len(train_batches) != TRAIN_PATCHES:
        raise RuntimeError("OpenMind fold lacks four legal volume train patches")

    validation_record = max(
        validation_records,
        key=lambda item: int(
            np.sum((item.seismic_patch[8] > 0.5) & ~(item.seismic_patch[7] > 0.5))
        ),
    )
    same_patch = prepared.validation_cells.sample_ids == validation_record.sample_id
    eligible = np.flatnonzero(same_patch & prepared.validation_metric_mask)
    selected = eligible[stage2._sample_indices(eligible.size, VALIDATION_VOXELS)]
    metric_indices = prepared.validation_cells.indices_kji[selected]
    validation_base = stage1._record_feature_volume(  # noqa: SLF001
        spec, bundle, validation_record, validation=True
    )
    validation_flat = stage2._record_metric_flat_indices(  # noqa: SLF001
        validation_record, metric_indices
    )
    validation_batch = stage2._replace_batch_mask(  # noqa: SLF001
        validation_base,
        spec,
        validation_flat,
        coordinates={"metric_indices_kji": metric_indices},
        metadata={"shared_validation_voxel_count": len(metric_indices)},
    )
    expected = prepared.validation_target[selected]
    actual = np.asarray(validation_batch.targets[spec.targets[0]])[
        np.asarray(validation_batch.target_masks[spec.targets[0]], dtype=bool)
    ]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    return train_batches, validation_batch


def _fit_and_score(
    train_batches: Sequence[Any],
    validation_batch: Any,
    *,
    source_root: Path,
    checkpoint: Path,
    device: str,
    seed: int,
    random_init: bool,
) -> dict[str, float]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    spec = p4.task_spec("strict")
    model = build_model(
        spec,
        source_root=source_root,
        checkpoint_path=checkpoint,
        freeze_encoder=True,
        device=device,
    )
    if random_init:
        _randomize_encoder(model, seed=seed + 100_000)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.0)
    target_name = spec.targets[0]
    model.train()
    last_loss = float("nan")
    for update in range(UPDATES):
        batch = train_batches[update % len(train_batches)]
        inputs = torch.as_tensor(
            np.asarray(batch.inputs["volume"])[:, :3],
            dtype=torch.float32,
            device=device,
        )
        target = torch.as_tensor(
            np.asarray(batch.targets[target_name]),
            dtype=torch.float32,
            device=device,
        )
        mask = torch.as_tensor(
            np.asarray(batch.target_masks[target_name]),
            dtype=torch.bool,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=device.startswith("cuda"),
        ):
            prediction = model(inputs)
            loss = torch.mean((prediction[mask] - target[mask]) ** 2)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("OpenMind training loss is non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    model.eval()
    inputs = torch.as_tensor(
        np.asarray(validation_batch.inputs["volume"])[:, :3],
        dtype=torch.float32,
        device=device,
    )
    mask = np.asarray(validation_batch.target_masks[target_name], dtype=bool)
    truth = np.asarray(validation_batch.targets[target_name], dtype=np.float64)[mask]
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.startswith("cuda"),
    ):
        prediction = model(inputs).float().cpu().numpy()[mask]
    error = prediction.astype(np.float64) - truth
    return {
        "rmse": float(math.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "last_train_loss": last_loss,
    }


def run(
    *,
    data_dir: Path,
    source_root: Path,
    checkpoint: Path,
    baseline_leaderboard: Path,
    expected_split_hash: str,
    output: Path = DEFAULT_OUTPUT,
    device: str = "cuda:0",
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog = p4.scan_patch_catalog(data_dir)
    manifest = p4.build_spatial_manifest("strict", catalog)
    if manifest.stable_hash() != expected_split_hash:
        raise RuntimeError("OpenMind strict split differs from frozen P5 split")
    active = p4.protocol("strict")
    development = p4.load_patch_records(active.development_i_blocks, data_dir)
    loaded_blocks = sorted({record.i_block for record in development})
    if loaded_blocks != list(active.development_i_blocks):
        raise RuntimeError("OpenMind did not load the complete strict development scope")
    if set(loaded_blocks) & (set(active.guard_i_blocks) | set(active.test_i_blocks)):
        raise RuntimeError("OpenMind crossed the strict test/guard firewall")
    baseline = json.loads(baseline_leaderboard.read_text(encoding="utf-8"))
    winner = baseline["entries"][0]
    fold_results: list[dict[str, Any]] = []
    for fold in manifest.folds:
        train, validation = _volume_batches(fold, development)
        pretrained = _fit_and_score(
            train,
            validation,
            source_root=source_root,
            checkpoint=checkpoint,
            device=device,
            seed=2693 + fold.fold_id,
            random_init=False,
        )
        random_init = _fit_and_score(
            train,
            validation,
            source_root=source_root,
            checkpoint=checkpoint,
            device=device,
            seed=2693 + fold.fold_id,
            random_init=True,
        )
        fold_results.append(
            {
                "fold_id": fold.fold_id,
                "effective_train_patches": len(
                    fold.purge["effective_train_sample_ids"]
                ),
                "validation_patch_count": len(fold.validation_sample_ids),
                "pretrained": pretrained,
                "same_architecture_random_init": random_init,
                "strong_baseline_rmse": float(
                    winner["fold_mean_rmse"][str(fold.fold_id)]
                ),
            }
        )
    pretrained_rmse = float(
        np.mean([row["pretrained"]["rmse"] for row in fold_results])
    )
    random_rmse = float(
        np.mean(
            [row["same_architecture_random_init"]["rmse"] for row in fold_results]
        )
    )
    baseline_rmse = float(winner["metrics"]["rmse"]["mean"])
    wins = pretrained_rmse < random_rmse and pretrained_rmse < baseline_rmse
    result = {
        "schema_version": "reconstruction-p9-openmind-effect/v1",
        "task_id": active.task_id,
        "model": {
            "model_id": "MIC-DKFZ/ResEncL-OpenMind-MAE",
            "checkpoint_sha256": _sha256(checkpoint),
            "real_pretrained_weights_loaded": True,
            "trainable_scope": "attribute_projection_and_decoder",
        },
        "evaluation": {
            "mode": "strict",
            "folds": list(FOLDS),
            "split_hash": manifest.stable_hash(),
            "train_volume_patches_per_fold": TRAIN_PATCHES,
            "train_target_voxels_per_fold": TRAIN_PATCHES * TARGETS_PER_PATCH,
            "validation_voxels_per_fold": VALIDATION_VOXELS,
            "updates_per_fold": UPDATES,
            "same_validation_sample_universe_as_strong_baseline": True,
            "frozen_test_accessed": False,
            "guard_accessed": False,
        },
        "fold_results": fold_results,
        "comparison": {
            "pretrained_macro_fold_rmse": pretrained_rmse,
            "same_architecture_random_init_macro_fold_rmse": random_rmse,
            "strong_baseline_model_id": winner["model_id"],
            "strong_baseline_macro_fold_rmse": baseline_rmse,
            "pretrained_minus_random_init_rmse": pretrained_rmse - random_rmse,
            "pretrained_minus_strong_baseline_rmse": pretrained_rmse - baseline_rmse,
        },
        "decision": {
            "state": "EFFECT_AND_BASELINE_WIN" if wins else "CONNECTED_NO_PROMOTION",
            "default_enabled": wins,
        },
        "runtime": {
            "device": device,
            "duration_seconds": time.perf_counter() - started,
            "raw_predictions_persisted": False,
            "checkpoint_written": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-leaderboard", type=Path, required=True)
    parser.add_argument("--expected-split-hash", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    result = run(**vars(_parser().parse_args()))
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
