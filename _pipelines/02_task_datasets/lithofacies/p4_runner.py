#!/usr/bin/env python3
"""P4 lifecycle runner for the real nine-class GM09 lithofacies task.

The commands are deliberately separated.  ``cv`` and ``refit`` can only load
development data; the frozen F-5 HDF5 path occurs only in the ``test`` entry.
No command launches a long HPO study: this track archives the agreed direction
and bounded plan for the integration owner to schedule later.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.cv import run_development_cv  # noqa: E402
from _code.ml_framework.lifecycle import (  # noqa: E402
    ExperimentLifecycle,
    ExperimentState,
)
from _code.ml_framework.model_registry import get_model  # noqa: E402
from _code.ml_framework.run_layout import create_run_layout  # noqa: E402
from _code.ml_framework.seeding import SeedTree, derive_seed, seed_everything  # noqa: E402
from _code.ml_framework.splits import Fold, SplitManifest, validate_manifest  # noqa: E402
from _code.ml_framework.trainer import (  # noqa: E402
    StepResult,
    TrainerConfig,
    TrainerState,
    train_with_validation,
)
from p4_contract import (  # noqa: E402
    CLASS_NAMES,
    DEVELOPMENT_FAMILIES,
    EFFECTIVE_N_SPLITS,
    FoldPreprocessor,
    TARGET_NAME,
    TEST_FAMILY,
    apply_fold_preprocessor,
    build_lithofacies_split_manifest,
    class_support,
    classification_metrics_from_logits,
    cross_entropy_loss,
    fit_fold_preprocessor,
    fit_temperature,
    lithofacies_hpo_plan,
    lithofacies_task_spec,
    model_output_from_logits,
    prediction_records,
    sample_id,
    samples_to_model_batch,
    softmax_probabilities,
    validate_p4_sample,
)


DEFAULT_CONFIG = {
    "model_id": "multimodal_mlp",
    "loss": "cross_entropy_sqrt_inverse_frequency",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "hidden_size": 64,
    "max_epochs": 40,
    "min_epochs": 4,
    "patience": 8,
    "min_delta": 1e-4,
    "root_seed": 2693,
    "device": "cpu",
}


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for lithofacies smoke/CV/refit/test; use the documented "
            "project environment. TaskSpec/split/HPO-plan tests remain torch-free."
        ) from exc
    return torch


def _dataset_path(split: str, dataset_root: Path | None = None) -> Path:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    root = dataset_root or (PROJECT_ROOT / "_data" / "processed" / "lithofacies")
    return root / f"{split}.h5"


def _read_hdf5(path: Path) -> list[dict[str, Any]]:
    """Read an existing HDF5 without calling dataset_io.task_dir().mkdir()."""
    if not path.is_file():
        raise FileNotFoundError(path)
    samples: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            samples.append(
                {
                    "seismic_patch": group["seismic_patch"][()],
                    "well_log_seq": group["well_log_seq"][()],
                    "label": group["label"][()],
                    "position": json.loads(group.attrs["position"]),
                    "meta": json.loads(group.attrs["meta"]),
                }
            )
    if not samples:
        raise ValueError(f"dataset is empty: {path}")
    return samples


def load_development_samples(dataset_root: Path | None = None) -> list[dict[str, Any]]:
    samples = _read_hdf5(_dataset_path("train", dataset_root))
    development = [
        sample for sample in samples if sample.get("meta", {}).get("family_id") in DEVELOPMENT_FAMILIES
    ]
    if len(development) != len(samples):
        unexpected = sorted(
            {
                str(sample.get("meta", {}).get("family_id"))
                for sample in samples
                if sample.get("meta", {}).get("family_id") not in DEVELOPMENT_FAMILIES
            }
        )
        raise ValueError(f"development HDF5 contains non-development families: {unexpected}")
    return development


def load_frozen_test_samples(dataset_root: Path | None = None) -> list[dict[str, Any]]:
    """The only track-private function allowed to open the F-5 test HDF5."""
    samples = _read_hdf5(_dataset_path("test", dataset_root))
    families = {str(sample.get("meta", {}).get("family_id")) for sample in samples}
    if families != {TEST_FAMILY}:
        raise ValueError(f"frozen test must contain only {TEST_FAMILY}, got {sorted(families)}")
    return samples


def _split_manifest_from_dict(payload: Mapping[str, Any]) -> SplitManifest:
    folds = tuple(
        Fold(
            fold_id=int(fold["fold_id"]),
            train_groups=tuple(fold["train_groups"]),
            validation_groups=tuple(fold["validation_groups"]),
            train_sample_ids=tuple(fold["train_sample_ids"]),
            validation_sample_ids=tuple(fold["validation_sample_ids"]),
            purge=dict(fold["purge"]),
            support=dict(fold["support"]),
        )
        for fold in payload["folds"]
    )
    manifest = SplitManifest(
        manifest_version=str(payload["manifest_version"]),
        group_key=str(payload["group_key"]),
        requested_n_splits=int(payload["requested_n_splits"]),
        effective_n_splits=int(payload["effective_n_splits"]),
        downgrade_reason=payload.get("downgrade_reason"),
        test_groups=tuple(payload["test_groups"]),
        test_sample_ids=tuple(payload["test_sample_ids"]),
        development_groups=tuple(payload["development_groups"]),
        development_sample_ids=tuple(payload["development_sample_ids"]),
        folds=folds,
        metadata=dict(payload["metadata"]),
    )
    validate_manifest(manifest)
    return manifest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_lifecycle(run_root: Path) -> ExperimentLifecycle:
    payload = _read_json(run_root / "lifecycle.json")
    return ExperimentLifecycle(
        experiment_id=str(payload["experiment_id"]),
        state=ExperimentState(payload["state"]),
        evidence={key: dict(value) for key, value in payload.get("evidence", {}).items()},
        test_consumed_at=payload.get("test_consumed_at"),
    )


def _write_lifecycle(run_root: Path, lifecycle: ExperimentLifecycle) -> None:
    atomic_write_json(run_root / "lifecycle.json", lifecycle.to_dict())


def _artifact_role(relative: str) -> str:
    if "checkpoint" in relative:
        return "checkpoint"
    if relative.endswith(".png"):
        return "visualization"
    if "prediction" in relative:
        return "prediction"
    if "metric" in relative or "summary" in relative:
        return "metric"
    if "split" in relative:
        return "split"
    if "preprocess" in relative or "normalization" in relative:
        return "preprocessing"
    if "hpo" in relative or "trial" in relative:
        return "hpo"
    return "run_evidence"


def refresh_artifact_manifest(run_root: Path) -> Path:
    manifest = ArtifactManifest(run_root.name, run_root)
    for path in sorted(run_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(run_root).as_posix()
        manifest.register(relative, role=_artifact_role(relative))
    return manifest.write()


def _environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }
    if importlib.util.find_spec("torch") is not None:
        torch = _require_torch()
        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
    else:
        payload["torch"] = None
    return payload


def _portable_data_path(path: Path) -> dict[str, Any]:
    """Never serialize a host/worktree prefix into a run manifest."""
    resolved = path.resolve()
    try:
        return {"path": str(resolved.relative_to(PROJECT_ROOT)), "external_dataset_root": False}
    except ValueError:
        return {"path": resolved.name, "external_dataset_root": True}


def _ensure_track_owned_run_root(path: Path) -> Path:
    resolved = path.resolve()
    root = TRACK_DIR.resolve()
    if root not in resolved.parents:
        raise ValueError(f"run root must stay under {TRACK_DIR}")
    return resolved


def _write_not_feasible(run_root: Path, stage: str, exc: Exception) -> Path:
    return atomic_write_json(
        run_root / f"not_feasible_{stage}.json",
        {
            "status": "not_feasible",
            "stage": stage,
            "reason": f"{type(exc).__name__}: {exc}",
            "required_action": (
                "provide the existing real lithofacies HDF5 inside this integration worktree or rerun the "
                "approved real-data builder; do not synthesize labels or split mother families"
            ),
        },
    )


def prepare_run(run_root: Path, *, config: Mapping[str, Any], dataset_root: Path | None = None) -> dict[str, Any]:
    if (run_root / "lifecycle.json").exists():
        raise RuntimeError(
            "run root already has a lifecycle; never reset or overwrite a consumed experiment, "
            "use a new run_id"
        )
    run_root = create_run_layout(run_root)
    train_path = _dataset_path("train", dataset_root)
    test_path = _dataset_path("test", dataset_root)
    development = load_development_samples(dataset_root)
    frozen_test = load_frozen_test_samples(dataset_root)
    combined = [*development, *frozen_test]
    split_manifest = build_lithofacies_split_manifest(combined)
    if split_manifest.test_groups != (TEST_FAMILY,) or split_manifest.effective_n_splits != EFFECTIVE_N_SPLITS:
        raise RuntimeError("frozen F-5 / LOGO-4 contract changed unexpectedly")
    task_spec = lithofacies_task_spec()
    seed_report = seed_everything(
        int(config["root_seed"]), strict=True, include_torch=importlib.util.find_spec("torch") is not None
    )
    environment = _environment()
    atomic_write_json(run_root / "task_spec.json", task_spec.to_dict())
    atomic_write_json(run_root / "run_config.json", dict(config))
    atomic_write_json(run_root / "hpo" / "plan.json", lithofacies_hpo_plan())
    atomic_write_json(run_root / "seed_report.json", seed_report.to_dict())
    atomic_write_json(run_root / "environment.json", environment)
    atomic_write_json(run_root / "split_manifest.json", split_manifest.to_dict())
    atomic_write_json(
        run_root / "data_manifest.json",
        {
            "train_hdf5": {**_portable_data_path(train_path), "sha256": hash_file(train_path)},
            "test_hdf5": {**_portable_data_path(test_path), "sha256": hash_file(test_path)},
            "development_samples": len(development),
            "frozen_test_samples": len(frozen_test),
            "development_class_support": class_support(development).tolist(),
            "test_class_support": class_support(frozen_test).tolist(),
        },
    )
    lifecycle = ExperimentLifecycle(run_root.name)
    lifecycle.advance(
        ExperimentState.SPLIT_LOCKED,
        {
            "split_hash": split_manifest.stable_hash(),
            "requested_n_splits": split_manifest.requested_n_splits,
            "effective_n_splits": split_manifest.effective_n_splits,
            "test_family": TEST_FAMILY,
        },
    )
    _write_lifecycle(run_root, lifecycle)
    refresh_artifact_manifest(run_root)
    return {
        "status": "SPLIT_LOCKED",
        "split_hash": split_manifest.stable_hash(),
        "requested_n_splits": split_manifest.requested_n_splits,
        "effective_n_splits": split_manifest.effective_n_splits,
        "downgrade_reason": split_manifest.downgrade_reason,
    }


def _batches(
    samples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: str,
    shuffle_seed: int | None = None,
) -> Iterable[Any]:
    indices = np.arange(len(samples))
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield samples_to_model_batch([samples[int(index)] for index in indices[start : start + batch_size]], device=device)


def _build_model(config: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> Any:
    _require_torch()
    first = samples[0]
    return get_model(
        str(config["model_id"]),
        models_package="models",
        num_classes=len(CLASS_NAMES),
        well_log_shape=tuple(int(value) for value in np.asarray(first["well_log_seq"]).shape),
        seismic_shape=tuple(int(value) for value in np.asarray(first["seismic_patch"]).shape),
        hidden_size=int(config["hidden_size"]),
    )


def _forward(model: Any, batch: Any) -> Any:
    logits = model(batch.inputs["well_log_seq"], batch.inputs["seismic_patch"])
    return model_output_from_logits(logits)


def _predict(
    model: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch = _require_torch()
    model.eval()
    arrays: list[np.ndarray] = []
    with torch.no_grad():
        for batch in _batches(samples, batch_size=batch_size, device=device):
            arrays.append(_forward(model, batch).raw[TARGET_NAME].detach().cpu().numpy())
    return np.concatenate(arrays, axis=0)


def tiny_overfit_probe(
    samples: Sequence[Mapping[str, Any]],
    *,
    steps: int = 30,
    root_seed: int = 2693,
    device: str = "cpu",
) -> dict[str, Any]:
    """Small gradient/label-map proof; never a generalization metric."""
    torch = _require_torch()
    if len(samples) < 2 or len({int(sample["label"]) for sample in samples}) < 2:
        raise ValueError("tiny-overfit requires at least two samples and two labels")
    seed_everything(root_seed, strict=True, include_torch=True)
    config = {**DEFAULT_CONFIG, "model_id": "lithofacies_concat_linear", "hidden_size": 8}
    model = _build_model(config, samples).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    batch = samples_to_model_batch(samples, device=device)

    def loss_value() -> Any:
        return cross_entropy_loss(_forward(model, batch), batch) / len(samples)

    model.eval()
    initial = float(loss_value().detach())
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value()
        loss.backward()
        optimizer.step()
    model.eval()
    final = float(loss_value().detach())
    if not math.isfinite(initial) or not math.isfinite(final) or final >= initial:
        raise RuntimeError(f"tiny-overfit did not reduce loss: {initial} -> {final}")
    return {"steps": steps, "initial_loss": initial, "final_loss": final, "status": "PASS"}


def run_real_smoke(run_root: Path, *, dataset_root: Path | None = None) -> dict[str, Any]:
    lifecycle = _load_lifecycle(run_root)
    if lifecycle.state != ExperimentState.SPLIT_LOCKED:
        raise RuntimeError("real smoke requires SPLIT_LOCKED")
    config = _read_json(run_root / "run_config.json")
    development = load_development_samples(dataset_root)
    manifest = _split_manifest_from_dict(_read_json(run_root / "split_manifest.json"))
    fold = manifest.folds[0]
    lookup = {sample_id(sample): sample for sample in development}
    train_raw = [lookup[sid] for sid in fold.train_sample_ids]
    validation_raw = [lookup[sid] for sid in fold.validation_sample_ids]
    preprocessor = fit_fold_preprocessor(train_raw)
    train = apply_fold_preprocessor(train_raw[: min(12, len(train_raw))], preprocessor)
    validation = apply_fold_preprocessor(validation_raw[: min(8, len(validation_raw))], preprocessor)
    probe_samples = train[: min(8, len(train))]
    if len({int(sample["label"]) for sample in probe_samples}) < 2:
        probe_samples = train
    tiny = tiny_overfit_probe(
        probe_samples,
        steps=10,
        root_seed=int(config["root_seed"]),
        device=str(config["device"]),
    )
    torch = _require_torch()
    model = _build_model(config, train).to(str(config["device"]))
    weights = torch.as_tensor(preprocessor.class_weights, dtype=torch.float32, device=str(config["device"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    batch = next(iter(_batches(train, batch_size=len(train), device=str(config["device"]))))
    optimizer.zero_grad(set_to_none=True)
    output = _forward(model, batch)
    loss = cross_entropy_loss(output, batch, weights) / len(train)
    loss.backward()
    optimizer.step()
    validation_logits = _predict(
        model,
        validation,
        batch_size=len(validation),
        device=str(config["device"]),
    )
    metrics = classification_metrics_from_logits(
        [int(sample["label"]) for sample in validation], validation_logits
    )
    report = {
        "status": "PASS",
        "mode": "real_data_one_fold_smoke",
        "formal_metric": False,
        "fold_id": fold.fold_id,
        "train_groups": fold.train_groups,
        "validation_groups": fold.validation_groups,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "optimizer_step_loss": float(loss.detach()),
        "finite_logits": bool(np.isfinite(validation_logits).all()),
        "tiny_overfit": tiny,
        "validation_supported_class_macro_f1": metrics["supported_class_macro_f1"],
    }
    atomic_write_json(run_root / "smoke" / "real_data_smoke.json", report)
    lifecycle.advance(ExperimentState.SMOKE_PASSED, {"report": "smoke/real_data_smoke.json"})
    _write_lifecycle(run_root, lifecycle)
    refresh_artifact_manifest(run_root)
    return report


def _train_one_fold(
    fold: Fold,
    *,
    samples_by_id: Mapping[str, Mapping[str, Any]],
    manifest: SplitManifest,
    config: Mapping[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    torch = _require_torch()
    root_seed = int(config["root_seed"])
    fold_seed = derive_seed(root_seed, "cv", fold.fold_id)
    report = seed_everything(fold_seed, strict=True, include_torch=True)
    train_raw = [samples_by_id[sid] for sid in fold.train_sample_ids]
    validation_raw = [samples_by_id[sid] for sid in fold.validation_sample_ids]
    preprocessor = fit_fold_preprocessor(train_raw)
    train = apply_fold_preprocessor(train_raw, preprocessor)
    validation = apply_fold_preprocessor(validation_raw, preprocessor)
    device = str(config["device"])
    model = _build_model(config, train).to(device)
    weights = torch.as_tensor(preprocessor.class_weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    fold_dir = run_root / "folds" / f"fold_{fold.fold_id}"
    config_hash = hash_payload(config)
    split_hash = manifest.stable_hash()
    train_epoch = {"value": 0}

    def train_batches() -> Iterable[Any]:
        current_epoch = train_epoch["value"]
        train_epoch["value"] += 1
        return _batches(
            train,
            batch_size=int(config["batch_size"]),
            device=device,
            shuffle_seed=derive_seed(root_seed, "loader", fold.fold_id, current_epoch),
        )

    def validation_batches() -> Iterable[Any]:
        return _batches(validation, batch_size=int(config["batch_size"]), device=device)

    def train_step(batch: Any) -> StepResult:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = _forward(model, batch)
        loss_sum = cross_entropy_loss(output, batch, weights)
        loss_sum.backward()
        optimizer.step()
        return StepResult(loss_sum=float(loss_sum.detach()), valid_count=len(batch.sample_ids))

    def validation_step(batch: Any) -> StepResult:
        model.eval()
        with torch.no_grad():
            loss_sum = cross_entropy_loss(_forward(model, batch), batch, weights)
        return StepResult(loss_sum=float(loss_sum.detach()), valid_count=len(batch.sample_ids))

    def checkpoint_writer(state: TrainerState, path: Path) -> None:
        save_checkpoint(
            path,
            epoch=max(0, state.next_epoch - 1),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None,
            scaler_state=None,
            config_hash=config_hash,
            split_hash=split_hash,
            trainer_state=state.to_dict(),
            seed_report=report.to_dict(),
            environment=_environment(),
            extra={
                "track_id": "lithofacies",
                "fold_id": fold.fold_id,
                "class_names": CLASS_NAMES,
                "preprocessor": preprocessor.to_dict(),
                "well_log_shape": tuple(int(value) for value in train[0]["well_log_seq"].shape),
                "seismic_shape": tuple(int(value) for value in train[0]["seismic_patch"].shape),
            },
        )

    state = train_with_validation(
        train_step=train_step,
        validation_step=validation_step,
        train_batches_fn=train_batches,
        validation_batches_fn=validation_batches,
        config=TrainerConfig(
            max_epochs=int(config["max_epochs"]),
            min_epochs=int(config["min_epochs"]),
            patience=int(config["patience"]),
            min_delta=float(config["min_delta"]),
        ),
        output_dir=fold_dir,
        checkpoint_writer=checkpoint_writer,
    )
    best = load_checkpoint(fold_dir / "checkpoint_best.pkl")
    if best["config_hash"] != config_hash or best["split_hash"] != split_hash:
        raise RuntimeError("fold checkpoint does not match run config/split hashes")
    model.load_state_dict(best["model_state"])
    logits = _predict(
        model,
        validation,
        batch_size=int(config["batch_size"]),
        device=device,
    )
    labels = [int(sample["label"]) for sample in validation]
    metrics = classification_metrics_from_logits(labels, logits)
    records = prediction_records(validation_raw, logits)
    atomic_write_json(fold_dir / "preprocess_stats.json", preprocessor.to_dict())
    atomic_write_json(fold_dir / "predictions.json", {"records": records})
    atomic_write_json(fold_dir / "metrics.json", metrics)
    return {
        "validation_sample_ids": tuple(record["sample_id"] for record in records),
        "metrics": {"supported_class_macro_f1": metrics["supported_class_macro_f1"]},
        "all_metrics": metrics,
        "valid_label_count": len(validation),
        "best_epoch": int(best["trainer_state"]["best_epoch"] + 1),
        "stopped_early": bool(state.stopped_early),
        "train_class_support": list(preprocessor.class_support),
        "validation_class_support": class_support(validation_raw).tolist(),
        "train_missing_class_ids": [
            index for index, count in enumerate(preprocessor.class_support) if count == 0
        ],
        "preprocess_stats": str((fold_dir / "preprocess_stats.json").relative_to(run_root)),
        "checkpoint": str((fold_dir / "checkpoint_best.pkl").relative_to(run_root)),
        "predictions": str((fold_dir / "predictions.json").relative_to(run_root)),
    }


def run_cv(run_root: Path, *, dataset_root: Path | None = None) -> dict[str, Any]:
    """Run development-only LOGO-4.  No test path/loader is accepted."""
    lifecycle = _load_lifecycle(run_root)
    lifecycle.require_development_access()
    if lifecycle.state != ExperimentState.SMOKE_PASSED:
        raise RuntimeError("CV requires SMOKE_PASSED")
    development = load_development_samples(dataset_root)
    manifest = _split_manifest_from_dict(_read_json(run_root / "split_manifest.json"))
    config = _read_json(run_root / "run_config.json")
    lookup = {sample_id(sample): sample for sample in development}
    if set(lookup) != set(manifest.development_sample_ids):
        raise RuntimeError("development HDF5 does not match the locked split manifest")

    def fold_runner(fold: Fold) -> Mapping[str, Any]:
        return _train_one_fold(
            fold,
            samples_by_id=lookup,
            manifest=manifest,
            config=config,
            run_root=run_root,
        )

    summary = run_development_cv(
        manifest,
        fold_runner,
        output_dir=run_root,
        primary_metric="supported_class_macro_f1",
        metric_direction="maximize",
    )
    records: list[dict[str, Any]] = []
    for fold in manifest.folds:
        payload = _read_json(run_root / "folds" / f"fold_{fold.fold_id}" / "predictions.json")
        records.extend(payload["records"])
    if sorted(record["sample_id"] for record in records) != sorted(manifest.development_sample_ids):
        raise RuntimeError("OOF prediction archive is not a one-to-one development cover")
    labels = np.asarray([int(record["true_class_id"]) for record in records], dtype=np.int64)
    logits = np.asarray([record["logits"] for record in records], dtype=np.float64)
    calibration = fit_temperature(labels, logits)
    calibrated_metrics = classification_metrics_from_logits(labels, logits / calibration["temperature"])
    calibrated_records_by_id = {
        record["sample_id"]: record
        for record in prediction_records(
            [lookup[record["sample_id"]] for record in records],
            logits,
            temperature=float(calibration["temperature"]),
        )
    }
    ordered_records = [calibrated_records_by_id[record["sample_id"]] for record in records]
    atomic_write_json(
        run_root / "oof" / "predictions.json",
        {
            "scope": "development_oof",
            "temperature": calibration["temperature"],
            "records": ordered_records,
        },
    )
    atomic_write_json(run_root / "oof" / "calibration.json", calibration)
    atomic_write_json(run_root / "oof" / "metrics.json", calibrated_metrics)
    lifecycle.advance(
        ExperimentState.CV_COMPLETE,
        {
            "oof_hash": hash_file(run_root / "oof" / "predictions.json"),
            "effective_n_splits": manifest.effective_n_splits,
            "oof_sample_count": len(records),
        },
    )
    _write_lifecycle(run_root, lifecycle)
    refresh_artifact_manifest(run_root)
    return {
        **summary,
        "calibration_temperature": calibration["temperature"],
        "calibrated_supported_class_macro_f1": calibrated_metrics["supported_class_macro_f1"],
    }


def freeze_configuration(run_root: Path) -> dict[str, Any]:
    lifecycle = _load_lifecycle(run_root)
    if lifecycle.state != ExperimentState.CV_COMPLETE:
        raise RuntimeError("freeze requires CV_COMPLETE")
    config = _read_json(run_root / "run_config.json")
    summary = _read_json(run_root / "oof" / "summary.json")
    calibration = _read_json(run_root / "oof" / "calibration.json")
    best_epochs = [int(fold["best_epoch"]) for fold in summary["folds"]]
    frozen = {
        **config,
        "temperature": float(calibration["temperature"]),
        "refit_epochs": max(1, int(round(float(np.median(best_epochs))))),
        "selection_scope": "development_oof_only",
        "test_metrics_seen": False,
        "best_epochs_by_fold": best_epochs,
    }
    config_hash = hash_payload(frozen)
    frozen["config_hash"] = config_hash
    atomic_write_json(run_root / "frozen_config.json", frozen)
    lifecycle.advance(ExperimentState.CONFIG_FROZEN, {"config_hash": config_hash})
    _write_lifecycle(run_root, lifecycle)
    refresh_artifact_manifest(run_root)
    return frozen


def refit_development(run_root: Path, *, dataset_root: Path | None = None) -> dict[str, Any]:
    lifecycle = _load_lifecycle(run_root)
    if lifecycle.state != ExperimentState.CONFIG_FROZEN:
        raise RuntimeError("refit requires CONFIG_FROZEN")
    torch = _require_torch()
    frozen = _read_json(run_root / "frozen_config.json")
    config_hash = str(frozen.pop("config_hash"))
    if hash_payload(frozen) != config_hash:
        raise RuntimeError("frozen configuration hash mismatch")
    frozen["config_hash"] = config_hash
    manifest = _split_manifest_from_dict(_read_json(run_root / "split_manifest.json"))
    development_raw = load_development_samples(dataset_root)
    preprocessor = fit_fold_preprocessor(development_raw)
    development = apply_fold_preprocessor(development_raw, preprocessor)
    root_seed = int(frozen["root_seed"])
    seed_report = seed_everything(root_seed, strict=True, include_torch=True)
    device = str(frozen["device"])
    model = _build_model(frozen, development).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(frozen["learning_rate"]),
        weight_decay=float(frozen["weight_decay"]),
    )
    weights = torch.as_tensor(preprocessor.class_weights, dtype=torch.float32, device=device)
    history: list[dict[str, Any]] = []
    global_step = 0
    model.train()
    for epoch in range(int(frozen["refit_epochs"])):
        loss_sum = 0.0
        count = 0
        for batch in _batches(
            development,
            batch_size=int(frozen["batch_size"]),
            device=device,
            shuffle_seed=derive_seed(root_seed, "loader", "refit", epoch),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy_loss(_forward(model, batch), batch, weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            count += len(batch.sample_ids)
            global_step += 1
        history.append({"epoch": epoch, "train_loss": loss_sum / count, "valid_count": count})
    trainer_state = {
        "next_epoch": int(frozen["refit_epochs"]),
        "global_step": global_step,
        "best_epoch": int(frozen["refit_epochs"]) - 1,
        "best_val_loss": float(history[-1]["train_loss"]),
        "epochs_without_improvement": 0,
        "stopped_early": False,
        "history": history,
    }
    checkpoint_path = run_root / "refit" / "checkpoint.pkl"
    save_checkpoint(
        checkpoint_path,
        epoch=int(frozen["refit_epochs"]) - 1,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        scaler_state=None,
        config_hash=config_hash,
        split_hash=manifest.stable_hash(),
        trainer_state=trainer_state,
        seed_report=seed_report.to_dict(),
        environment=_environment(),
        extra={
            "track_id": "lithofacies",
            "class_names": CLASS_NAMES,
            "preprocessor": preprocessor.to_dict(),
            "well_log_shape": tuple(int(value) for value in development[0]["well_log_seq"].shape),
            "seismic_shape": tuple(int(value) for value in development[0]["seismic_patch"].shape),
        },
    )
    atomic_write_json(run_root / "refit" / "preprocess_stats.json", preprocessor.to_dict())
    atomic_write_json(run_root / "refit" / "history.json", history)
    checkpoint_hash = hash_file(checkpoint_path)
    lifecycle.advance(
        ExperimentState.REFIT_COMPLETE,
        {"checkpoint_hash": checkpoint_hash, "config_hash": config_hash},
    )
    _write_lifecycle(run_root, lifecycle)
    refresh_artifact_manifest(run_root)
    return {
        "status": "REFIT_COMPLETE",
        "epochs": int(frozen["refit_epochs"]),
        "checkpoint_hash": checkpoint_hash,
    }


def run_frozen_test(run_root: Path, *, dataset_root: Path | None = None) -> dict[str, Any]:
    """Single-use F-5 test campaign; the lifecycle is consumed before HDF5 access."""
    lifecycle = _load_lifecycle(run_root)
    if lifecycle.state != ExperimentState.REFIT_COMPLETE:
        raise RuntimeError("frozen test requires REFIT_COMPLETE")
    frozen = _read_json(run_root / "frozen_config.json")
    config_hash = str(frozen["config_hash"])
    frozen_without_hash = dict(frozen)
    frozen_without_hash.pop("config_hash")
    if hash_payload(frozen_without_hash) != config_hash:
        raise RuntimeError("frozen configuration hash mismatch")
    manifest = _split_manifest_from_dict(_read_json(run_root / "split_manifest.json"))
    checkpoint_path = run_root / "refit" / "checkpoint.pkl"
    checkpoint_hash = hash_file(checkpoint_path)
    lifecycle.consume_test(
        config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        split_hash=manifest.stable_hash(),
    )
    _write_lifecycle(run_root, lifecycle)

    # The frozen HDF5 is opened only after the single-use state has been durably recorded.
    test_raw = load_frozen_test_samples(dataset_root)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["config_hash"] != config_hash or checkpoint["split_hash"] != manifest.stable_hash():
        raise RuntimeError("refit checkpoint envelope does not match frozen config/split")
    preprocessor = FoldPreprocessor.from_dict(
        _read_json(run_root / "refit" / "preprocess_stats.json")
    )
    test = apply_fold_preprocessor(test_raw, preprocessor)
    device = str(frozen["device"])
    model = _build_model(frozen, test).to(device)
    model.load_state_dict(checkpoint["model_state"])
    logits = _predict(
        model,
        test,
        batch_size=int(frozen["batch_size"]),
        device=device,
    )
    temperature = float(frozen["temperature"])
    labels = [int(sample["label"]) for sample in test]
    calibrated_logits = logits / temperature
    metrics = classification_metrics_from_logits(labels, calibrated_logits)
    records = prediction_records(test_raw, logits, temperature=temperature)
    atomic_write_json(
        run_root / "frozen_test" / "predictions.json",
        {
            "scope": "frozen_F5_test",
            "test_consumed_at": lifecycle.test_consumed_at,
            "config_hash": config_hash,
            "checkpoint_hash": checkpoint_hash,
            "split_hash": manifest.stable_hash(),
            "temperature": temperature,
            "records": records,
        },
    )
    atomic_write_json(run_root / "frozen_test" / "metrics.json", metrics)
    refresh_artifact_manifest(run_root)
    return {
        "status": "TEST_CONSUMED",
        "samples": len(test),
        "accuracy": metrics["accuracy"],
        "supported_class_macro_f1": metrics["supported_class_macro_f1"],
        "fixed_schema_macro_f1": metrics["fixed_schema_macro_f1"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "smoke", "cv", "hpo-plan", "freeze", "refit", "test"),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_id"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--device", default=DEFAULT_CONFIG["device"])
    parser.add_argument("--root-seed", type=int, default=DEFAULT_CONFIG["root_seed"])
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be >0")
    run_root = _ensure_track_owned_run_root(args.run_root)
    dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    config = {
        **DEFAULT_CONFIG,
        "model_id": args.model,
        "max_epochs": args.epochs,
        "min_epochs": min(int(DEFAULT_CONFIG["min_epochs"]), args.epochs),
        "device": args.device,
        "root_seed": args.root_seed,
    }
    started = time.monotonic()
    try:
        if args.command == "prepare":
            result = prepare_run(run_root, config=config, dataset_root=dataset_root)
        elif args.command == "smoke":
            result = run_real_smoke(run_root, dataset_root=dataset_root)
        elif args.command == "cv":
            result = run_cv(run_root, dataset_root=dataset_root)
        elif args.command == "hpo-plan":
            result = lithofacies_hpo_plan()
            create_run_layout(run_root)
            atomic_write_json(run_root / "hpo" / "plan.json", result)
            refresh_artifact_manifest(run_root)
        elif args.command == "freeze":
            result = freeze_configuration(run_root)
        elif args.command == "refit":
            result = refit_development(run_root, dataset_root=dataset_root)
        else:
            result = run_frozen_test(run_root, dataset_root=dataset_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if run_root.exists():
            _write_not_feasible(run_root, args.command, exc)
        print(json.dumps({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
    result = {**dict(result), "elapsed_seconds": time.monotonic() - started}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
