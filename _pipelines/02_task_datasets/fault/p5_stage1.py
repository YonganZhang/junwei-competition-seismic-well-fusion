#!/usr/bin/env python3
"""Fail-closed P5 Stage-1 contract runner for the first ten fault candidates.

The runner never accepts a test loader or test path.  It uses only a caller-
supplied development HDF5, verifies that file against the frozen P4 train hash,
and refuses official loss/backward when audited verified negatives are absent.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import resource
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_DIR = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, TRACK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from _code.ml_framework.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from _code.ml_framework.contracts import ModelBatch, ModelOutput  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _code.ml_framework.trainer import TrainerState  # noqa: E402
from _models.fault.p5_lock import (  # noqa: E402
    P5ModelUnavailable,
    evaluate_runtime_gate,
    load_source_locks,
    lock_file_evidence,
)
from p4_contract import TARGET_NAME, adapt_fault_arrays, fault_task_spec, validate_fault_batch  # noqa: E402


FIRST_TEN_MODEL_IDS = (
    "monai_segresnet",
    "monai_dynunet",
    "nnunet_v2_3d_fullres",
    "pytorch3dunet_unet3d",
    "faultnet_md",
    "faultseg3d_keras",
    "monai_vnet",
    "mednext_v1_s_k3",
    "uxnet3d",
    "monai_swinunetr",
)
BUILD_SUMMARY = TRACK_DIR / "_outputs" / "runs" / "audited_v2" / "build_summary.json"
CV_PLAN = TRACK_DIR / "_outputs" / "p4_preflight" / "buffered_cv_plan.json"
BLIND_AUDIT = TRACK_DIR / "_outputs" / "p4_preflight" / "blind_test_not_feasible.json"


class DevelopmentDataUnavailable(RuntimeError):
    def __init__(self, reason_code: str, detail: str, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail
        self.evidence = dict(evidence or {})


@dataclass(frozen=True)
class DevelopmentProbe:
    batch: ModelBatch
    evidence: Mapping[str, Any]


def _attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if hasattr(value, "item") else value


def _crop_around_positive(
    patch: np.ndarray,
    label: np.ndarray,
    verified_negative: np.ndarray,
    *,
    output_shape: tuple[int, int] = (32, 64),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if patch.ndim != 3 or patch.shape[0] != 1 or label.shape != patch.shape[1:]:
        raise DevelopmentDataUnavailable(
            "DEVELOPMENT_SHAPE_INVALID",
            f"expected patch [1,H,W] and matching label, got {patch.shape}/{label.shape}",
        )
    height, width = label.shape
    out_h, out_w = output_shape
    if height < out_h or width < out_w:
        raise DevelopmentDataUnavailable(
            "DEVELOPMENT_SHAPE_TOO_SMALL",
            f"real patch {label.shape} cannot supply fixed smoke crop {output_shape}",
        )
    positive = np.argwhere(label == 1)
    if not len(positive):
        raise DevelopmentDataUnavailable("DEVELOPMENT_POSITIVE_MISSING", "selected sample has no fault voxel")
    centre_h, centre_w = (int(value) for value in positive[len(positive) // 2])
    start_h = min(max(centre_h - out_h // 2, 0), height - out_h)
    start_w = min(max(centre_w - out_w // 2, 0), width - out_w)
    slices = (slice(start_h, start_h + out_h), slice(start_w, start_w + out_w))
    return patch[:, slices[0], slices[1]], label[slices], verified_negative[slices]


def load_real_development_probe(path: Path, *, batch_size: int = 1) -> DevelopmentProbe:
    """Read a real train-only batch; never infer background from stored zeros."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not path.is_file():
        raise DevelopmentDataUnavailable(
            "DEVELOPMENT_ASSET_MISSING",
            "caller-supplied development HDF5 does not exist",
        )
    observed_hash = hash_file(path)
    summary = json.loads(BUILD_SUMMARY.read_text(encoding="utf-8"))
    expected_hash = str(summary["dataset_sha256"]["train"])
    if observed_hash != expected_hash:
        raise DevelopmentDataUnavailable(
            "DEVELOPMENT_HASH_MISMATCH",
            "development HDF5 does not match the frozen audited_v2 train hash",
            {"expected_sha256": expected_hash, "observed_sha256": observed_hash},
        )

    amplitudes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    positions: list[dict[str, Any]] = []
    sample_kinds: list[str] = []
    sample_keys: list[str] = []
    with h5py.File(path, "r") as handle:
        task = str(_attribute(handle.attrs.get("task", "")))
        split = str(_attribute(handle.attrs.get("split", "")))
        if task != "fault" or split != "train":
            raise DevelopmentDataUnavailable(
                "DEVELOPMENT_ROLE_INVALID",
                f"HDF5 must declare task=fault, split=train; got task={task!r}, split={split!r}",
            )
        audit_status = str(_attribute(handle.attrs.get("verified_negative_audit_status", "")))
        audit_hash = str(_attribute(handle.attrs.get("verified_negative_audit_sha256", "")))
        masks_are_audited = audit_status == "complete" and len(audit_hash) == 64
        for key in sorted(handle.keys()):
            group = handle[key]
            meta = json.loads(_attribute(group.attrs["meta"]))
            label = np.asarray(group["label"][()], dtype=np.uint8)
            if meta.get("sample_kind") != "fault" or not np.any(label == 1):
                continue
            patch = np.asarray(group["seismic_patch"][()], dtype=np.float32)
            if "verified_negative_mask" in group:
                if not masks_are_audited:
                    raise DevelopmentDataUnavailable(
                        "VERIFIED_NEGATIVE_AUDIT_INVALID",
                        "HDF5 contains a negative mask without complete audit provenance",
                    )
                verified = np.asarray(group["verified_negative_mask"][()], dtype=bool)
            else:
                verified = np.zeros_like(label, dtype=bool)
            patch, label, verified = _crop_around_positive(patch, label, verified)
            amplitudes.append(patch)
            labels.append(label)
            negatives.append(verified)
            positions.append(json.loads(_attribute(group.attrs["position"])))
            sample_kinds.append(str(meta["sample_kind"]))
            sample_keys.append(str(key))
            if len(amplitudes) == batch_size:
                break
        if not amplitudes:
            raise DevelopmentDataUnavailable(
                "DEVELOPMENT_POSITIVE_MISSING",
                "no positive train sample was found in the development HDF5",
            )

    batch = adapt_fault_arrays(
        np.stack(amplitudes),
        np.stack(labels),
        positions,
        sample_kinds,
        verified_negative_mask=np.stack(negatives),
    )
    validate_fault_batch(batch)
    valid = np.asarray(batch.target_masks[TARGET_NAME], dtype=bool)
    target = np.asarray(batch.targets[TARGET_NAME], dtype=bool)
    verified_negative = np.asarray(batch.input_masks["verified_negative_mask"], dtype=bool)
    evidence = {
        "source_role": "audited_v2_train_development",
        "source_sha256": observed_hash,
        "source_bytes": path.stat().st_size,
        "sample_keys": sample_keys,
        "batch_shape": list(np.asarray(batch.inputs["seismic_amplitude"]).shape),
        "positive_labels": int(np.sum(target & valid)),
        "verified_negative_labels": int(verified_negative.sum()),
        "valid_labels": int(valid.sum()),
        "unknown_labels": int((~valid).sum()),
        "verified_negative_audit_status": audit_status or "absent",
        "verified_negative_audit_sha256": audit_hash or None,
        "stored_zero_policy": "unknown unless covered by audited verified_negative_mask",
    }
    return DevelopmentProbe(batch=batch, evidence=evidence)


def synthetic_verified_batch(*, seed: int = 2693) -> ModelBatch:
    rng = np.random.default_rng(seed)
    amplitudes = rng.normal(0.0, 0.25, size=(1, 32, 32, 32)).astype(np.float32)
    labels = np.zeros((1, 32, 32, 32), dtype=np.uint8)
    verified = np.zeros_like(labels, dtype=bool)
    labels[0, 16, 14:19, 16] = 1
    verified[0, 8, 8:13, 8] = True
    return adapt_fault_arrays(
        amplitudes,
        labels,
        [{"inline": 100, "crossline": 200, "time_index": 300}],
        ["fault"],
        verified_negative_mask=verified,
        spatial_block_ids=["synthetic-contract-only"],
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _output_evidence(batch: ModelBatch, output: ModelOutput) -> dict[str, Any]:
    logits = output.raw[TARGET_NAME]
    probability = output.transformed[TARGET_NAME]
    if not isinstance(logits, torch.Tensor) or not isinstance(probability, torch.Tensor):
        raise TypeError("P5 Torch adapter must return Torch tensors")
    expected = tuple(np.asarray(batch.inputs["seismic_amplitude"]).shape)
    if tuple(logits.shape) != expected or tuple(probability.shape) != expected:
        raise RuntimeError(f"output shape mismatch: {tuple(logits.shape)}/{tuple(probability.shape)} vs {expected}")
    if not torch.isfinite(logits).all() or not torch.isfinite(probability).all():
        raise RuntimeError("model output contains non-finite values")
    if torch.any(probability < 0) or torch.any(probability > 1):
        raise RuntimeError("sigmoid probability lies outside [0,1]")
    return {
        "raw_shape": list(logits.shape),
        "raw_finite": True,
        "probability_shape": list(probability.shape),
        "probability_range": [float(probability.min().item()), float(probability.max().item())],
        "sigmoid_applied_only_by_adapter": True,
    }


def _state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _build_model(model_id: str, *, seed: int, device: torch.device) -> torch.nn.Module:
    model = discover_model("fault", model_id).build(fault_task_spec(), seed=seed)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{model_id} did not build a Torch module in torch-common")
    return model.to(device)


def run_trainable_contract(
    model_id: str,
    batch: ModelBatch,
    *,
    device: torch.device,
    seed: int,
    split_hash: str,
) -> dict[str, Any]:
    """Run one full build/forward/masked-loss/backward/checkpoint contract."""

    _seed_everything(seed)
    vram_telemetry_error: str | None = None
    if device.type == "cuda":
        try:
            # CUDA_VISIBLE_DEVICES may remap the caller's device index.  Resource
            # telemetry must never turn a valid model contract into a failure.
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError as exc:
            vram_telemetry_error = f"{type(exc).__name__}: {exc}"
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    first = _build_model(model_id, seed=seed, device=device)
    first.eval()
    with torch.no_grad():
        initial = first(batch)
        initial_logits = initial.raw[TARGET_NAME].detach().cpu()
    del first
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = _build_model(model_id, seed=seed, device=device)
    model.eval()
    with torch.no_grad():
        repeated = model(batch)
    deterministic_diff = float(
        torch.max(torch.abs(initial_logits - repeated.raw[TARGET_NAME].detach().cpu())).item()
    )
    if deterministic_diff > 1e-6:
        raise RuntimeError(f"same-seed build changed initial logits by {deterministic_diff}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    output_check = _output_evidence(batch, output)
    loss = model.masked_loss(batch, output)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("backward produced missing or non-finite gradients")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        reference = model(batch)
    reference_logits = reference.raw[TARGET_NAME].detach().cpu()
    trainer_state = TrainerState(
        next_epoch=1,
        global_step=1,
        best_epoch=0,
        best_val_loss=float(loss.detach().cpu().item()),
        epochs_without_improvement=0,
        stopped_early=False,
        history=[{"epoch": 0, "masked_loss": float(loss.detach().cpu().item())}],
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_type": device.type,
    }
    with tempfile.TemporaryDirectory(prefix="fault-p5-stage1-") as directory:
        checkpoint_path = Path(directory) / f"{model_id}.ckpt"
        save_checkpoint(
            checkpoint_path,
            epoch=0,
            model_state=_state_to_cpu(model),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=None,
            scaler_state=None,
            config_hash=hash_payload({"model_id": model_id, "stage": 1, "seed": seed}),
            split_hash=split_hash,
            trainer_state=trainer_state.to_dict(),
            seed_report={"root_seed": seed, "seed_tree": {"model": seed}},
            environment=environment,
            extra={"track_id": "fault", "frozen_test_accessed": False},
        )
        checkpoint_hash = hash_file(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
        loaded = load_checkpoint(checkpoint_path)
        restored = _build_model(model_id, seed=seed, device=device)
        restored.load_state_dict(loaded["model_state"], strict=True)
        restored.eval()
        with torch.no_grad():
            recovered = restored(batch)
        checkpoint_diff = float(
            torch.max(
                torch.abs(reference_logits - recovered.raw[TARGET_NAME].detach().cpu())
            ).item()
        )
        recovered_state = TrainerState.from_dict(loaded["trainer_state"])
        if recovered_state != trainer_state or checkpoint_diff > 1e-6:
            raise RuntimeError("checkpoint prediction/TrainerState did not round-trip")
        del restored

    if device.type == "cuda":
        try:
            torch.cuda.synchronize()
            peak_vram: int | None = int(torch.cuda.max_memory_allocated())
        except RuntimeError as exc:
            peak_vram = None
            detail = f"{type(exc).__name__}: {exc}"
            vram_telemetry_error = (
                detail if vram_telemetry_error is None else f"{vram_telemetry_error}; {detail}"
            )
    else:
        peak_vram = 0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    evidence = {
        "status": "passed",
        "model_id": model_id,
        "seed": seed,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": peak_vram,
        "vram_telemetry_error": vram_telemetry_error,
        "process_max_rss_kib_before": int(rss_before),
        "process_max_rss_kib_after": int(rss_after),
        "downloaded_bytes": 0,
        "output": output_check,
        "masked_loss": float(loss.detach().cpu().item()),
        "backward_finite": True,
        "same_seed_max_abs_diff": deterministic_diff,
        "checkpoint": {
            "sha256": checkpoint_hash,
            "bytes": checkpoint_bytes,
            "prediction_max_abs_diff": checkpoint_diff,
            "trainer_state_restored": True,
            "persisted": False,
        },
        "frozen_test_accessed": False,
    }
    del model, optimizer, output, reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return evidence


def run_real_forward(
    model_id: str,
    batch: ModelBatch,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = _build_model(model_id, seed=seed, device=device)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(batch)
    result = {
        "status": "passed",
        "wall_seconds": time.perf_counter() - started,
        "output": _output_evidence(batch, output),
        "loss_backward_checkpoint": "not_run_until_binary_supervision_gate",
        "frozen_test_accessed": False,
    }
    del model, output
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _environment(device: torch.device) -> dict[str, Any]:
    try:
        import monai

        monai_version = monai.__version__
    except ImportError:
        monai_version = None
    gpu_name = (
        torch.cuda.get_device_name(device.index if device.index is not None else torch.cuda.current_device())
        if device.type == "cuda"
        else None
    )
    return {
        "python": platform.python_version(),
        "executable_role": "caller-selected shared torch-common environment",
        "torch": torch.__version__,
        "monai": monai_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_name": gpu_name,
        "platform": platform.platform(),
    }


def run_stage1(
    output_dir: Path,
    *,
    development_hdf5: Path | None,
    model_ids: Sequence[str] = FIRST_TEN_MODEL_IDS,
    device_name: str = "cuda:0",
    root_seed: int = 2693,
) -> dict[str, Any]:
    """Run Stage-1 without any frozen-test argument or access path."""

    if tuple(model_ids) != FIRST_TEN_MODEL_IDS and not set(model_ids).issubset(FIRST_TEN_MODEL_IDS):
        raise ValueError("model_ids must be the first-ten list or a subset used by tests")
    locked = load_source_locks()
    if tuple(locked) != FIRST_TEN_MODEL_IDS:
        raise RuntimeError("source lock order does not match the frozen P5 fault candidate order")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_payload = json.loads(CV_PLAN.read_text(encoding="utf-8"))
    blind_payload = json.loads(BLIND_AUDIT.read_text(encoding="utf-8"))
    split_hash = hash_payload(cv_payload)

    development_probe: DevelopmentProbe | None = None
    development_gap: DevelopmentDataUnavailable | None = None
    if development_hdf5 is None:
        development_gap = DevelopmentDataUnavailable(
            "DEVELOPMENT_ASSET_MISSING",
            "--development-hdf5 was not supplied; no data path is guessed",
        )
    else:
        try:
            development_probe = load_real_development_probe(development_hdf5)
        except DevelopmentDataUnavailable as exc:
            development_gap = exc

    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        gate = evaluate_runtime_gate(model_id)
        record: dict[str, Any] = {
            "model_id": model_id,
            "source_lock": locked[model_id],
            "runtime_gate": gate,
            "status": "skipped",
            "evidence_state": "scouted",
            "frozen_test_accessed": False,
        }
        if gate["status"] != "ready":
            record["reason_code"] = gate["reason_code"]
            record["detail"] = gate["detail"]
            results.append(record)
            atomic_write_json(output_dir / "models" / f"{model_id}.json", record)
            continue
        try:
            synthetic = run_trainable_contract(
                model_id,
                synthetic_verified_batch(seed=root_seed),
                device=device,
                seed=root_seed,
                split_hash="synthetic-contract-only",
            )
            record["synthetic_contract"] = synthetic
            record["evidence_state"] = "synthetic_contract_smoked"
            if development_gap is not None:
                record["reason_code"] = development_gap.reason_code
                record["detail"] = development_gap.detail
                record["development"] = {
                    "status": "skipped",
                    "evidence": development_gap.evidence,
                }
            else:
                assert development_probe is not None
                real_forward = run_real_forward(
                    model_id,
                    development_probe.batch,
                    device=device,
                    seed=root_seed,
                )
                record["development"] = {
                    **dict(development_probe.evidence),
                    "forward": real_forward,
                }
                positives = int(development_probe.evidence["positive_labels"])
                negatives = int(development_probe.evidence["verified_negative_labels"])
                if positives == 0 or negatives == 0:
                    record["reason_code"] = "NO_AUDITED_VERIFIED_NEGATIVES"
                    record["detail"] = (
                        "real development forward passed, but official masked loss/backward/checkpoint "
                        f"was skipped: positive={positives}, verified_negative={negatives}"
                    )
                    record["development"]["status"] = "skipped"
                    record["development"]["loss_backward_checkpoint"] = "skipped"
                else:
                    real_contract = run_trainable_contract(
                        model_id,
                        development_probe.batch,
                        device=device,
                        seed=root_seed,
                        split_hash=split_hash,
                    )
                    record["development"]["status"] = "passed"
                    record["development"]["contract"] = real_contract
                    record["status"] = "contract_smoked"
                    record["evidence_state"] = "contract_smoked"
                    record["reason_code"] = None
                    record["detail"] = None
        except P5ModelUnavailable as exc:
            record["reason_code"] = exc.reason_code
            record["detail"] = exc.detail
            record["runtime_exception"] = exc.to_dict()
        except Exception as exc:  # unexpected failures remain visible and make the CLI non-zero
            record["status"] = "failed"
            record["reason_code"] = "RUNTIME_CONTRACT_FAILURE"
            record["detail"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
        atomic_write_json(output_dir / "models" / f"{model_id}.json", record)

    counts = {
        status: sum(record["status"] == status for record in results)
        for status in ("contract_smoked", "skipped", "failed")
    }
    summary = {
        "protocol": "fault-p5-stage1-v1",
        "track_id": "fault",
        "root_seed": root_seed,
        "status": "failed" if counts["failed"] else "completed_with_structured_skips",
        "candidate_count": len(results),
        "counts": counts,
        "model_ids": list(model_ids),
        "results": [
            {
                "model_id": record["model_id"],
                "status": record["status"],
                "evidence_state": record["evidence_state"],
                "reason_code": record.get("reason_code"),
                "evidence": f"models/{record['model_id']}.json",
            }
            for record in results
        ],
        "environment": _environment(device),
        "source_lock": lock_file_evidence(),
        "p4_evidence": {
            "cv_plan_sha256": hash_file(CV_PLAN),
            "cv_status": cv_payload["status"],
            "effective_n_splits": cv_payload["effective_n_splits"],
            "blind_audit_sha256": hash_file(BLIND_AUDIT),
            "blind_test_status": blind_payload["status"],
            "frozen_test_accessed": False,
        },
        "development_source": (
            dict(development_probe.evidence)
            if development_probe is not None
            else {
                "status": "unavailable",
                "reason_code": development_gap.reason_code if development_gap else None,
                "detail": development_gap.detail if development_gap else None,
            }
        ),
        "downloads_performed": False,
        "installed_packages_changed": False,
        "training_scope": "single Stage-1 contract step only; no pilot, HPO, CV, refit, or test",
    }
    atomic_write_json(output_dir / "summary.json", summary)
    manifest = ArtifactManifest("fault-p5-stage1", output_dir)
    for model_id in model_ids:
        manifest.register(f"models/{model_id}.json", role="stage1_model_evidence")
    manifest.register("summary.json", role="stage1_summary")
    manifest.write()
    manifest.verify()
    summary["manifest_sha256"] = hash_file(output_dir / "manifest.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_DIR / "_outputs" / "p5_stage1",
    )
    parser.add_argument(
        "--development-hdf5",
        type=Path,
        default=Path(os.environ["FAULT_P5_DEVELOPMENT_HDF5"])
        if "FAULT_P5_DEVELOPMENT_HDF5" in os.environ
        else None,
        help="audited train-only HDF5; its SHA-256 must match the P4 train hash",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root-seed", type=int, default=2693)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_stage1(
        args.output_dir,
        development_hdf5=args.development_hdf5,
        device_name=args.device,
        root_seed=args.root_seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary["counts"]["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
