"""P4 facies training primitives; every callable is development-only by design."""
from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from _code.ml_framework.artifacts import atomic_write_json, hash_payload
from _code.ml_framework.checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from _code.ml_framework.contracts import ModelBatch
from _code.ml_framework.model_registry import get_model
from _code.ml_framework.seeding import SeedTree, derive_seed, seed_everything
from _code.ml_framework.lifecycle import ExperimentLifecycle, ExperimentState
from _code.ml_framework.splits import Fold, SplitManifest
from _code.ml_framework.trainer import (
    StepResult,
    TrainerConfig,
    TrainerState,
    train_with_validation,
)

from p4_data import (
    ArchiveRecord,
    FaciesArchive,
    FoldPreprocessor,
    records_by_id,
    select_records_with_all_classes,
)
from p4_losses import build_loss, softmax_probabilities
from p4_metrics import (
    confidence_entropy_error,
    evaluate_probabilities,
    sample_calibration_pixels,
)
from p4_tasks import fixed_baseline_config, get_task_spec
from pipeline_contract import ordered_spatial_split


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    }


def _atomic_savez(path: Path, **arrays: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary_name, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path


class _EpochBatchFactory:
    def __init__(
        self,
        archive: FaciesArchive,
        records: Sequence[ArchiveRecord],
        preprocessor: FoldPreprocessor,
        *,
        batch_size: int,
        root_seed: int,
        fold_id: int,
        start_epoch: int,
        shuffle: bool,
    ) -> None:
        self.archive = archive
        self.records = tuple(records)
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.root_seed = root_seed
        self.fold_id = fold_id
        self.epoch = start_epoch
        self.shuffle = shuffle

    def __call__(self):
        seed = derive_seed(self.root_seed, "loader", self.fold_id, self.epoch)
        self.epoch += 1
        return self.archive.iter_model_batches(
            self.records,
            self.preprocessor,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=seed,
            include_targets=True,
        )


def _torch_batch(batch: ModelBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.targets is None:
        raise ValueError("training/evaluation batch is missing facies targets")
    images = torch.as_tensor(batch.inputs["seismic"], dtype=torch.float32, device=device)
    labels = torch.as_tensor(batch.targets["facies"], dtype=torch.long, device=device)
    return images, labels


def train_development_model(
    *,
    archive: FaciesArchive,
    train_records: Sequence[ArchiveRecord],
    validation_records: Sequence[ArchiveRecord],
    preprocessor: FoldPreprocessor,
    run_config: Mapping[str, Any],
    split_hash: str,
    output_dir: Path,
    epochs: int,
    fold_id: int,
    device: torch.device,
    resume_path: Path | None = None,
    strict_determinism: bool = True,
) -> tuple[nn.Module, TrainerState, Path]:
    """Train one development fold; there is intentionally no test argument."""
    task_id = archive.task_id
    spec = get_task_spec(task_id)
    num_classes = int(spec.metadata["num_classes"])
    root_seed = int(run_config["root_seed"])
    seed_report = seed_everything(root_seed, strict=strict_determinism).to_dict()
    model = get_model(
        str(run_config["model_id"]),
        models_package="models",
        num_classes=num_classes,
        **dict(run_config.get("model_config", {})),
    )
    if not isinstance(model, nn.Module):
        raise TypeError("facies model registry did not return torch.nn.Module")
    model = model.to(device)
    criterion = build_loss(
        str(run_config["loss_id"]),
        num_classes=num_classes,
        class_weights=torch.tensor(preprocessor.class_weights, device=device),
        **dict(run_config.get("loss_config", {})),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(run_config["learning_rate"]),
        weight_decay=float(run_config["weight_decay"]),
    )
    scheduler = None
    scaler = None
    config_payload = {key: value for key, value in dict(run_config).items() if key != "config_hash"}
    computed_config_hash = hash_payload(config_payload)
    config_hash = str(run_config.get("config_hash", computed_config_hash))
    if config_hash != computed_config_hash:
        raise RuntimeError("run_config config_hash does not match its canonical payload")
    resume_state: TrainerState | None = None
    if resume_path is not None:
        payload = load_checkpoint(resume_path)
        if payload["config_hash"] != config_hash or payload["split_hash"] != split_hash:
            raise RuntimeError("resume checkpoint config/split hash mismatch")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload["scheduler_state"] is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        restore_rng_state(payload["rng_state"])
        resume_state = TrainerState.from_dict(payload["trainer_state"])

    def train_step(batch: ModelBatch) -> StepResult:
        model.train()
        images, labels = _torch_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.numel()
        correct = int((logits.detach().argmax(dim=1) == labels).sum())
        return StepResult(
            loss_sum=float(loss.detach()) * count,
            valid_count=count,
            metric_sums={"pixel_accuracy": float(correct)},
        )

    @torch.no_grad()
    def validation_step(batch: ModelBatch) -> StepResult:
        model.eval()
        images, labels = _torch_batch(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)
        count = labels.numel()
        correct = int((logits.argmax(dim=1) == labels).sum())
        return StepResult(
            loss_sum=float(loss) * count,
            valid_count=count,
            metric_sums={"pixel_accuracy": float(correct)},
        )

    start_epoch = 0 if resume_state is None else resume_state.next_epoch
    train_batches = _EpochBatchFactory(
        archive,
        train_records,
        preprocessor,
        batch_size=int(run_config["batch_size"]),
        root_seed=root_seed,
        fold_id=fold_id,
        start_epoch=start_epoch,
        shuffle=True,
    )
    validation_batches = _EpochBatchFactory(
        archive,
        validation_records,
        preprocessor,
        batch_size=int(run_config["batch_size"]),
        root_seed=root_seed,
        fold_id=fold_id,
        start_epoch=start_epoch,
        shuffle=False,
    )
    environment = environment_record(device)

    def checkpoint_writer(state: TrainerState, path: Path) -> None:
        save_checkpoint(
            path,
            epoch=max(0, state.next_epoch - 1),
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None if scheduler is None else scheduler.state_dict(),
            scaler_state=None if scaler is None else scaler.state_dict(),
            config_hash=config_hash,
            split_hash=split_hash,
            trainer_state=state.to_dict(),
            seed_report=seed_report,
            environment=environment,
            extra={
                "track_id": "facies",
                "task_id": task_id,
                "label_version": spec.label_version,
                "fold_id": fold_id,
                "preprocessor": preprocessor.to_dict(),
                "output_contract": "raw_logits_BCHW",
            },
        )

    state = train_with_validation(
        train_step=train_step,
        validation_step=validation_step,
        train_batches_fn=train_batches,
        validation_batches_fn=validation_batches,
        config=TrainerConfig(max_epochs=epochs, min_epochs=epochs, patience=None),
        output_dir=output_dir,
        checkpoint_writer=checkpoint_writer,
        resume_state=resume_state,
    )
    best_path = output_dir / "checkpoint_best.pkl"
    best = load_checkpoint(best_path)
    model.load_state_dict(best["model_state"])
    model.eval()
    return model, state, best_path


@torch.no_grad()
def _predict_records(
    *,
    model: nn.Module,
    archive: FaciesArchive,
    records: Sequence[ArchiveRecord],
    preprocessor: FoldPreprocessor,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
    require_all_classes: bool = True,
) -> dict[str, Any]:
    """Internal inference primitive wrapped by lifecycle-specific public APIs."""
    model.eval()
    all_seismic: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    sample_ids: list[str] = []
    inline: list[int] = []
    batches = archive.iter_model_batches(
        records,
        preprocessor,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        include_targets=True,
    )
    for batch in batches:
        images, labels = _torch_batch(batch, device)
        logits = model(images)
        if logits.ndim != 4 or logits.shape[0] != images.shape[0] or logits.shape[2:] != images.shape[2:]:
            raise ValueError("model output violates facies [B,C,H,W] alignment")
        if not torch.isfinite(logits).all():
            raise ValueError("model returned NaN/Inf logits")
        all_seismic.append(images[:, 0].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        sample_ids.extend(batch.sample_ids)
        inline.extend(int(value) for value in batch.coordinates["inline"])
    logits_array = np.concatenate(all_logits)
    labels_array = np.concatenate(all_labels)
    probabilities = softmax_probabilities(
        torch.from_numpy(logits_array), temperature=temperature
    ).numpy()
    metrics, _ = evaluate_probabilities(
        probabilities,
        labels_array,
        num_classes=probabilities.shape[1],
        require_all_classes=require_all_classes,
    )
    prediction, confidence, entropy, error = confidence_entropy_error(
        probabilities, labels_array
    )
    return {
        "sample_ids": np.asarray(sample_ids),
        "inline": np.asarray(inline, dtype=np.int64),
        "seismic": np.concatenate(all_seismic).astype(np.float32),
        "labels": labels_array.astype(np.uint8),
        "logits": logits_array.astype(np.float32),
        "prediction": prediction,
        "confidence": confidence,
        "entropy": entropy,
        "error": error,
        "metrics": metrics,
        "temperature": float(temperature),
    }


def predict_development_records(
    *,
    model: nn.Module,
    archive: FaciesArchive,
    records: Sequence[ArchiveRecord],
    preprocessor: FoldPreprocessor,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Predict development only; this API rejects frozen-test records."""
    if not records or any(record.split != "train" for record in records):
        raise ValueError("development prediction accepts only train-archive records")
    return _predict_records(
        model=model,
        archive=archive,
        records=records,
        preprocessor=preprocessor,
        batch_size=batch_size,
        device=device,
        temperature=temperature,
        require_all_classes=True,
    )


def predict_consumed_frozen_test(
    *,
    lifecycle: ExperimentLifecycle,
    model: nn.Module,
    archive: FaciesArchive,
    records: Sequence[ArchiveRecord],
    preprocessor: FoldPreprocessor,
    batch_size: int,
    device: torch.device,
    temperature: float,
) -> dict[str, Any]:
    """The only prediction API that accepts test, after atomic consumption."""
    if lifecycle.state != ExperimentState.TEST_CONSUMED:
        raise RuntimeError("frozen test prediction requires TEST_CONSUMED lifecycle state")
    if not records or any(record.split != "test" for record in records):
        raise ValueError("frozen test prediction requires only test-archive records")
    return _predict_records(
        model=model,
        archive=archive,
        records=records,
        preprocessor=preprocessor,
        batch_size=batch_size,
        device=device,
        temperature=temperature,
        require_all_classes=True,
    )


def archive_prediction_maps(
    prediction: Mapping[str, Any],
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Archive maps/metrics without creating or fitting any calibrator."""
    prediction_path = output_dir / "predictions.npz"
    metrics_path = output_dir / "metrics.json"
    _atomic_savez(
        prediction_path,
        sample_ids=np.asarray(prediction["sample_ids"], dtype=str),
        inline=np.asarray(prediction["inline"], dtype=np.int64),
        seismic=np.asarray(prediction["seismic"], dtype=np.float16),
        labels=np.asarray(prediction["labels"], dtype=np.uint8),
        prediction=np.asarray(prediction["prediction"], dtype=np.uint8),
        confidence=np.asarray(prediction["confidence"], dtype=np.float16),
        entropy=np.asarray(prediction["entropy"], dtype=np.float16),
        error=np.asarray(prediction["error"], dtype=np.uint8),
    )
    atomic_write_json(metrics_path, dict(prediction["metrics"]))
    return prediction_path, metrics_path


def archive_predictions(
    prediction: Mapping[str, Any],
    *,
    output_dir: Path,
    calibration_seed: int,
    calibration_max_pixels: int = 200_000,
) -> tuple[Path, Path, Path]:
    """Archive predictions plus a deterministic OOF calibration subset."""
    calibration_path = output_dir / "calibration_logits.npz"
    prediction_path, metrics_path = archive_prediction_maps(
        prediction, output_dir=output_dir
    )
    calibration_logits, calibration_labels = sample_calibration_pixels(
        np.asarray(prediction["logits"]),
        np.asarray(prediction["labels"]),
        max_pixels=calibration_max_pixels,
        seed=calibration_seed,
    )
    _atomic_savez(
        calibration_path,
        logits=calibration_logits,
        labels=calibration_labels,
        fit_scope=np.asarray("oof_validation_only"),
    )
    return prediction_path, metrics_path, calibration_path


def run_development_fold(
    *,
    fold: Fold,
    archive: FaciesArchive,
    development_records: Sequence[ArchiveRecord],
    run_config: Mapping[str, Any],
    manifest: SplitManifest,
    output_dir: Path,
    epochs: int,
    device: torch.device,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
) -> dict[str, Any]:
    """Train/evaluate a fold and return exactly its locked OOF IDs."""
    mapping = records_by_id(development_records)
    train_records = tuple(mapping[sample_id] for sample_id in fold.train_sample_ids)
    validation_records = tuple(mapping[sample_id] for sample_id in fold.validation_sample_ids)
    spec = get_task_spec(archive.task_id)
    num_classes = int(spec.metadata["num_classes"])
    formal_run = max_train_records is None and max_validation_records is None
    if max_train_records is not None:
        train_records = select_records_with_all_classes(
            train_records, num_classes=num_classes, max_records=max_train_records
        )
    if max_validation_records is not None:
        validation_records = select_records_with_all_classes(
            validation_records, num_classes=num_classes, max_records=max_validation_records
        )
    preprocessor = archive.fit_preprocessor(
        train_records, method=str(run_config["normalization"])
    )
    atomic_write_json(output_dir / "preprocess_stats.json", preprocessor.to_dict())
    fold_split_hash = hash_payload({"manifest_hash": manifest.stable_hash(), "fold_id": fold.fold_id})
    model, state, checkpoint_path = train_development_model(
        archive=archive,
        train_records=train_records,
        validation_records=validation_records,
        preprocessor=preprocessor,
        run_config=run_config,
        split_hash=fold_split_hash,
        output_dir=output_dir,
        epochs=epochs,
        fold_id=fold.fold_id,
        device=device,
    )
    prediction = predict_development_records(
        model=model,
        archive=archive,
        records=validation_records,
        preprocessor=preprocessor,
        batch_size=int(run_config["batch_size"]),
        device=device,
    )
    archive_predictions(
        prediction,
        output_dir=output_dir,
        calibration_seed=derive_seed(
            int(run_config["root_seed"]), "diagnostic", archive.task_id, fold.fold_id
        ),
    )
    metrics = dict(prediction["metrics"])
    returned_ids = tuple(str(value) for value in prediction["sample_ids"])
    if formal_run and (
        set(returned_ids) != set(fold.validation_sample_ids)
        or len(returned_ids) != len(fold.validation_sample_ids)
    ):
        raise RuntimeError(f"fold {fold.fold_id} prediction did not cover its locked OOF IDs")
    return {
        "validation_sample_ids": returned_ids,
        "metrics": metrics,
        "valid_label_count": int(np.asarray(prediction["labels"]).size),
        "best_epoch": state.best_epoch,
        "best_validation_loss": state.best_val_loss,
        "checkpoint": checkpoint_path.name,
        "preprocess_stats": "preprocess_stats.json",
        "prediction_archive": "predictions.npz",
        "calibration_archive": "calibration_logits.npz",
        "smoke_subset": not formal_run,
    }


def run_real_data_smoke(
    *,
    task_id: str,
    archive: FaciesArchive,
    manifest: SplitManifest,
    output_dir: Path,
    device: torch.device,
    epochs: int = 1,
    max_train_records: int = 32,
    max_validation_records: int = 16,
) -> dict[str, Any]:
    """Run a bounded real-data development smoke without consuming test labels."""
    config = fixed_baseline_config(task_id)
    fold = manifest.folds[0]
    development = archive.development_index()
    result = run_development_fold(
        fold=fold,
        archive=archive,
        development_records=development,
        run_config=config,
        manifest=manifest,
        output_dir=output_dir,
        epochs=epochs,
        device=device,
        max_train_records=max_train_records,
        max_validation_records=max_validation_records,
    )
    evidence = {
        "track_id": "facies",
        "task_id": task_id,
        "label_version": get_task_spec(task_id).label_version,
        "kind": "real_data_development_smoke",
        "epochs": epochs,
        "fold_id": fold.fold_id,
        "model_id": config["model_id"],
        "loss_id": config["loss_id"],
        "test_labels_read": False,
        "test_inference_run": False,
        "manifest_hash": manifest.stable_hash(),
        "metrics": result["metrics"],
        "train_subset_limit": max_train_records,
        "validation_subset_limit": max_validation_records,
    }
    atomic_write_json(output_dir / "smoke_evidence.json", evidence)
    return evidence


def run_fast_real_data_smoke(
    *,
    task_id: str,
    archive: FaciesArchive,
    output_dir: Path,
    device: torch.device,
    epochs: int = 1,
    max_candidates: int = 256,
    max_train_records: int = 32,
    max_validation_records: int = 16,
) -> dict[str, Any]:
    """Bounded integration smoke that never opens the frozen-test archive."""
    candidates = archive.sampled_development_index(max_candidates=max_candidates)
    lines = sorted({record.inline for record in candidates})
    train_lines, guard_lines, validation_lines = ordered_spatial_split(
        lines, holdout_fraction=0.20, guard_fraction=0.05
    )
    train_candidates = tuple(record for record in candidates if record.inline in train_lines)
    validation_candidates = tuple(
        record for record in candidates if record.inline in validation_lines
    )
    spec = get_task_spec(task_id)
    num_classes = int(spec.metadata["num_classes"])
    train_records = select_records_with_all_classes(
        train_candidates, num_classes=num_classes, max_records=max_train_records
    )
    validation_records = tuple(validation_candidates[:max_validation_records])
    if not validation_records:
        raise ValueError("bounded real-data smoke validation subset is empty")
    config = fixed_baseline_config(task_id)
    preprocessor = archive.fit_preprocessor(
        train_records, method=str(config["normalization"])
    )
    atomic_write_json(output_dir / "preprocess_stats.json", preprocessor.to_dict())
    smoke_split_hash = hash_payload(
        {
            "task_id": task_id,
            "train_sample_ids": [record.sample_id for record in train_records],
            "guard_lines": sorted(guard_lines),
            "validation_sample_ids": [record.sample_id for record in validation_records],
            "test_archive_opened": False,
        }
    )
    model, state, checkpoint_path = train_development_model(
        archive=archive,
        train_records=train_records,
        validation_records=validation_records,
        preprocessor=preprocessor,
        run_config=config,
        split_hash=smoke_split_hash,
        output_dir=output_dir,
        epochs=epochs,
        fold_id=0,
        device=device,
    )
    prediction = _predict_records(
        model=model,
        archive=archive,
        records=validation_records,
        preprocessor=preprocessor,
        batch_size=int(config["batch_size"]),
        device=device,
        require_all_classes=False,
    )
    archive_prediction_maps(prediction, output_dir=output_dir)
    evidence = {
        "track_id": "facies",
        "task_id": task_id,
        "label_version": spec.label_version,
        "kind": "bounded_real_data_development_smoke",
        "formal_cv": False,
        "epochs": epochs,
        "model_id": config["model_id"],
        "loss_id": config["loss_id"],
        "sampled_development_candidates": len(candidates),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "train_inline_range": [min(record.inline for record in train_records), max(record.inline for record in train_records)],
        "guard_inline_range": [min(guard_lines), max(guard_lines)],
        "validation_inline_range": [
            min(record.inline for record in validation_records),
            max(record.inline for record in validation_records),
        ],
        "test_archive_opened": False,
        "test_labels_read": False,
        "test_inference_run": False,
        "observed_support_metric_only": True,
        "best_epoch": state.best_epoch,
        "checkpoint": checkpoint_path.name,
        "metrics": prediction["metrics"],
    }
    atomic_write_json(output_dir / "smoke_evidence.json", evidence)
    return evidence
