"""P5 Stage-1 contract-smoke runner for the first ten property models.

This module deliberately has no frozen-test argument or loader.  ``prepare``
reads only an explicit ``train.h5`` plus the development guard archive and
creates a small normalized development batch.  ``run`` can then be invoked by
the shared tabular or torch interpreter without requiring HDF5 dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from _code.ml_framework.contracts import ModelBatch, ModelOutput  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.property._p5_common import (  # noqa: E402
    PROPERTY_TARGETS,
    Stage1GateError,
    source_lock_entry,
    source_lock_sha256,
)
from p5_contract import TARGET_INDICES, build_task_spec  # noqa: E402


FROZEN_TEST_FAMILY = "15/9-F-15"
DEVELOPMENT_FAMILIES = ("15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_sha256(path: Path) -> str:
    if path.is_file():
        return _sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(_sha256(child).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _decoded_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return dict(json.loads(str(value)))


def _train_hdf5_records(train_h5: Path) -> list[dict[str, Any]]:
    if train_h5.name != "train.h5":
        raise ValueError("Stage-1 accepts only an explicit file named train.h5")
    if not train_h5.is_file():
        raise FileNotFoundError(train_h5)
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("prepare requires an interpreter with h5py; run does not") from exc
    records: list[dict[str, Any]] = []
    with h5py.File(train_h5, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            meta = _decoded_json(group.attrs["meta"])
            position = _decoded_json(group.attrs["position"])
            family = str(meta["family_id"])
            if family == FROZEN_TEST_FAMILY or family not in DEVELOPMENT_FAMILIES:
                raise RuntimeError(f"train.h5 contains unauthorized Stage-1 family {family!r}")
            records.append(
                {
                    "sample_id": f"train-{key}",
                    "family_id": family,
                    "well_id": str(meta["well_id"]),
                    "depth_m": float(meta["depth_m"]),
                    "position": position,
                    "seismic_patch": np.asarray(group["seismic_patch"][()], dtype=np.float64),
                    "well_log_seq": np.asarray(group["well_log_seq"][()], dtype=np.float64),
                    "label": np.asarray(group["label"][()], dtype=np.float64).reshape(-1),
                }
            )
    return records


def _guard_records(guard_npz: Path) -> list[dict[str, Any]]:
    if not guard_npz.is_file():
        raise FileNotFoundError(guard_npz)
    records: list[dict[str, Any]] = []
    with np.load(guard_npz, allow_pickle=False) as archive:
        required = {"seismic_patch", "well_log_seq", "label", "position_json", "meta_json"}
        if not required <= set(archive.files):
            raise ValueError(f"guard archive missing {sorted(required - set(archive.files))}")
        for index in range(len(archive["label"])):
            meta = _decoded_json(str(archive["meta_json"][index]))
            position = _decoded_json(str(archive["position_json"][index]))
            family = str(meta["family_id"])
            if family == FROZEN_TEST_FAMILY or family not in DEVELOPMENT_FAMILIES:
                raise RuntimeError(f"guard archive contains unauthorized Stage-1 family {family!r}")
            records.append(
                {
                    "sample_id": f"guard-{index}",
                    "family_id": family,
                    "well_id": str(meta["well_id"]),
                    "depth_m": float(meta["depth_m"]),
                    "position": position,
                    "seismic_patch": np.asarray(archive["seismic_patch"][index], dtype=np.float64),
                    "well_log_seq": np.asarray(archive["well_log_seq"][index], dtype=np.float64),
                    "label": np.asarray(archive["label"][index], dtype=np.float64).reshape(-1),
                }
            )
    return records


def _balanced_development_subset(records: Sequence[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples < 8:
        raise ValueError("Stage-1 development batch requires at least 8 samples")
    families = sorted({str(record["family_id"]) for record in records})
    if len(families) < 2:
        raise RuntimeError("real Stage-1 batch requires at least two development mother-well families")
    per_family = max(2, max_samples // len(families))
    selected: list[dict[str, Any]] = []
    for family in families:
        family_records = sorted(
            (record for record in records if record["family_id"] == family),
            key=lambda record: (record["well_id"], record["depth_m"], record["sample_id"]),
        )
        if family_records:
            indices = np.linspace(0, len(family_records) - 1, min(per_family, len(family_records)), dtype=int)
            selected.extend(family_records[index] for index in indices)
    return sorted(selected[:max_samples], key=lambda record: (record["family_id"], record["sample_id"]))


def _normalize_inputs(seismic: np.ndarray, logs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if seismic.ndim != 4 or seismic.shape[1:] != (3, 3, 9):
        raise ValueError(f"unexpected seismic shape {seismic.shape}")
    if logs.ndim != 3 or logs.shape[1:] != (9, 8):
        raise ValueError(f"unexpected well-log shape {logs.shape}")
    if not np.isfinite(seismic).all() or not np.isfinite(logs).all():
        raise ValueError("development inputs must be finite")
    seismic_flat = seismic.reshape(len(seismic), -1)
    seismic_mean = seismic_flat.mean(axis=0)
    seismic_std = seismic_flat.std(axis=0) + 1e-8
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).reshape(len(logs), -1)
    log_mean = np.zeros(values.shape[1], dtype=np.float64)
    log_std = np.ones(values.shape[1], dtype=np.float64)
    for column in range(values.shape[1]):
        observed = values[masks[:, column], column]
        if observed.size:
            log_mean[column] = observed.mean()
            log_std[column] = observed.std() + 1e-8
    tabular = np.concatenate(
        [
            (seismic_flat - seismic_mean) / seismic_std,
            ((values - log_mean) / log_std) * masks,
            masks.astype(np.float64),
        ],
        axis=1,
    )
    if tabular.shape != (len(seismic), 153) or not np.isfinite(tabular).all():
        raise RuntimeError(f"invalid Stage-1 tabular view {tabular.shape}")
    stats = {
        "fit_scope": "selected real development samples only",
        "seismic_features": 81,
        "well_log_values": 36,
        "well_log_masks": 36,
        "denoise": "identity",
        "target_statistics_fitted": False,
    }
    return tabular, stats


def prepare_development_batch(
    *, train_h5: Path, guard_npz: Path, output_path: Path, max_samples: int = 64
) -> dict[str, Any]:
    """Create a small real-development NPZ without opening any frozen-test source."""
    records = _train_hdf5_records(Path(train_h5)) + _guard_records(Path(guard_npz))
    selected = _balanced_development_subset(records, max_samples)
    seismic = np.stack([record["seismic_patch"] for record in selected])
    logs = np.stack([record["well_log_seq"] for record in selected])
    labels = np.stack([record["label"] for record in selected])
    if labels.ndim != 2 or labels.shape[1] < len(PROPERTY_TARGETS):
        raise ValueError(f"real labels cannot provide {PROPERTY_TARGETS}: {labels.shape}")
    labels = labels[:, : len(PROPERTY_TARGETS)]
    target_masks = np.isfinite(labels)
    if not target_masks.any(axis=0).all():
        raise RuntimeError("real development batch does not support every independent property target")
    stored_labels = np.where(target_masks, labels, 0.0)
    tabular, normalization = _normalize_inputs(seismic, logs)
    families = np.asarray([record["family_id"] for record in selected])
    if FROZEN_TEST_FAMILY in set(families.tolist()):
        raise RuntimeError("frozen-test family reached the Stage-1 development batch")
    source_manifest = {
        "stage": 1,
        "test_access": False,
        "test_loader_implemented": False,
        "train_hdf5_sha256": _sha256(Path(train_h5)),
        "guard_npz_sha256": _sha256(Path(guard_npz)),
        "selected_samples": len(selected),
        "families": {family: int(np.sum(families == family)) for family in sorted(set(families))},
        "independent_target_valid_counts": {
            target: int(target_masks[:, TARGET_INDICES[target]].sum()) for target in PROPERTY_TARGETS
        },
        "normalization": normalization,
        "source_paths_persisted": False,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        seismic_patch=seismic,
        well_log_sequence=logs,
        tabular=tabular,
        labels_model_domain=stored_labels,
        target_masks=target_masks.astype(np.uint8),
        sample_ids=np.asarray([record["sample_id"] for record in selected]),
        family_ids=families,
        well_ids=np.asarray([record["well_id"] for record in selected]),
        depths_m=np.asarray([record["depth_m"] for record in selected], dtype=np.float64),
        source_manifest_json=np.asarray(json.dumps(source_manifest, sort_keys=True)),
    )
    return {**source_manifest, "development_batch_sha256": _sha256(output_path)}


def load_development_batch(path: Path) -> tuple[ModelBatch, dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        source_manifest = json.loads(str(archive["source_manifest_json"]))
        labels = np.asarray(archive["labels_model_domain"], dtype=np.float64)
        masks = np.asarray(archive["target_masks"], dtype=bool)
        sample_ids = archive["sample_ids"].astype(str).tolist()
        families = archive["family_ids"].astype(str).tolist()
        well_ids = archive["well_ids"].astype(str).tolist()
        depths = np.asarray(archive["depths_m"], dtype=np.float64)
        batch = ModelBatch(
            inputs={
                "tabular": np.asarray(archive["tabular"], dtype=np.float64),
                "seismic_patch": np.asarray(archive["seismic_patch"], dtype=np.float64),
                "well_log_sequence": np.asarray(archive["well_log_sequence"], dtype=np.float64),
            },
            targets={target: labels[:, index] for index, target in enumerate(PROPERTY_TARGETS)},
            input_masks={"well_log_observed": np.asarray(archive["well_log_sequence"][:, :, 4:8] > 0.5)},
            target_masks={target: masks[:, index] for index, target in enumerate(PROPERTY_TARGETS)},
            sample_ids=sample_ids,
            groups={"mother_well_family": families, "well_id": well_ids},
            coordinates={"depth_m": depths},
            metadata={"source": "real_development_npz", "test_access": False},
        )
    if source_manifest.get("test_access") is not False or FROZEN_TEST_FAMILY in set(families):
        raise RuntimeError("development batch violates the frozen-test firewall")
    return batch, {**source_manifest, "development_batch_sha256": _sha256(path)}


def synthetic_contract_batch(seed: int = 2693, sample_count: int = 12) -> ModelBatch:
    rng = np.random.default_rng(seed)
    seismic = rng.normal(size=(sample_count, 3, 3, 9))
    log_values = rng.normal(size=(sample_count, 9, 4))
    log_masks = np.ones((sample_count, 9, 4), dtype=np.float64)
    log_masks[::3, :, 1] = 0.0
    logs = np.concatenate([log_values * log_masks, log_masks], axis=2)
    tabular, _ = _normalize_inputs(seismic, logs)
    targets = {
        "PHIF": rng.uniform(0.08, 0.32, size=sample_count),
        "KLOGH": np.log1p(rng.uniform(1.0, 1500.0, size=sample_count)),
        "SW": rng.uniform(0.1, 0.95, size=sample_count),
    }
    masks = {target: np.ones(sample_count, dtype=bool) for target in PROPERTY_TARGETS}
    masks["PHIF"][0] = False
    masks["KLOGH"][1] = False
    masks["SW"][2] = False
    return ModelBatch(
        inputs={"tabular": tabular, "seismic_patch": seismic, "well_log_sequence": logs},
        targets=targets,
        input_masks={"well_log_observed": log_masks.astype(bool)},
        target_masks=masks,
        sample_ids=[f"synthetic-contract-{index}" for index in range(sample_count)],
        groups={"mother_well_family": ["synthetic-development"] * sample_count},
        coordinates={"depth_m": np.arange(sample_count, dtype=np.float64)},
        metadata={"synthetic_contract_only": True, "never_used_as_scientific_evidence": True},
    )


def _validate_output(output: ModelOutput, sample_count: int) -> dict[str, Any]:
    if tuple(output.raw) != PROPERTY_TARGETS or output.transformed is None:
        raise ValueError("property output must preserve ordered raw and physical views for all targets")
    raw_shapes: dict[str, list[int]] = {}
    physical_ranges: dict[str, list[float]] = {}
    for target in PROPERTY_TARGETS:
        raw = np.asarray(output.raw[target], dtype=np.float64)
        transformed = np.asarray(output.transformed[target], dtype=np.float64)
        if raw.shape != (sample_count,) or transformed.shape != (sample_count,):
            raise ValueError(f"{target} output has wrong shape")
        if not np.isfinite(raw).all() or not np.isfinite(transformed).all():
            raise FloatingPointError(f"{target} output is non-finite")
        raw_shapes[target] = list(raw.shape)
        physical_ranges[target] = [float(transformed.min()), float(transformed.max())]
    if physical_ranges["PHIF"][0] < 0 or physical_ranges["PHIF"][1] > 1:
        raise ValueError("PHIF physical view is outside [0,1]")
    if physical_ranges["SW"][0] < 0 or physical_ranges["SW"][1] > 1:
        raise ValueError("SW physical view is outside [0,1]")
    if physical_ranges["KLOGH"][0] < 0:
        raise ValueError("KLOGH physical view is negative")
    return {"raw_shapes": raw_shapes, "physical_ranges": physical_ranges, "finite": True}


def _dependency_versions(model_id: str) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for dependency in source_lock_entry(model_id).get("dependencies", []):
        distribution = dependency["distribution"]
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _raw_matrix(output: ModelOutput) -> np.ndarray:
    return np.column_stack([np.asarray(output.raw[target], dtype=np.float64) for target in PROPERTY_TARGETS])


def _model_config(model_id: str, *, seed: int, device: str) -> dict[str, Any]:
    config: dict[str, Any] = {"seed": seed, "n_features": 153, "device": device}
    if model_id == "realmlp_regressor":
        config.update({"n_epochs": 2, "batch_size": 64, "hidden_sizes": [64, 64], "n_threads": 1})
    if model_id == "monai_densenet3d_regressor":
        config.update({"init_features": 8, "growth_rate": 8, "block_config": (2, 2), "device": device})
    return config


def _reset_cuda_peak() -> None:
    # Must be set before torch initializes cuBLAS; adapters repeat this gate
    # immediately before module/optimizer construction.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _cuda_peak_bytes() -> int:
    try:
        import torch

        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


def contract_smoke_model(
    *, model_id: str, development: ModelBatch, output_dir: Path, seed: int, device: str
) -> dict[str, Any]:
    started = time.perf_counter()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _reset_cuda_peak()
    task_spec = build_task_spec()
    discovered = discover_model("property", model_id)
    config = _model_config(model_id, seed=seed, device=device)
    model = discovered.build(task_spec, **config)
    synthetic = synthetic_contract_batch(seed)
    synthetic_fit = model.fit(synthetic)
    synthetic_output = model.predict(synthetic)
    synthetic_report = _validate_output(synthetic_output, len(synthetic.sample_ids))
    real_fit = model.fit(development)
    real_output = model.predict(development)
    real_report = _validate_output(real_output, len(development.sample_ids))

    model_dir = output_dir / model_id
    checkpoint = model_dir / ("checkpoint" if model_id == "tabiclv2_regressor" else "checkpoint.bin")
    model.save_checkpoint(checkpoint)
    restored = discovered.build(task_spec, **config)
    restored.load_checkpoint(checkpoint)
    restored_prediction = _raw_matrix(restored.predict(development))
    original_prediction = _raw_matrix(real_output)
    if not np.allclose(restored_prediction, original_prediction, rtol=1e-6, atol=1e-7):
        raise RuntimeError("checkpoint round-trip changed real-development predictions")

    replay = discovered.build(task_spec, **config)
    replay.fit(synthetic)
    replay.fit(development)
    replay_prediction = _raw_matrix(replay.predict(development))
    deterministic = bool(np.allclose(replay_prediction, original_prediction, rtol=1e-5, atol=1e-6))
    if not deterministic:
        raise RuntimeError("same seed/environment Stage-1 replay exceeded determinism tolerance")
    wall_time = time.perf_counter() - started
    result = {
        "model_id": model_id,
        "status": "contract_smoked",
        "evidence_state": "contract_smoked",
        "capabilities": dict(discovered.capabilities),
        "source_lock": source_lock_entry(model_id),
        "dependencies": _dependency_versions(model_id),
        "environment": {
            "python": platform.python_version(),
            "executable_role": source_lock_entry(model_id)["dependency_group"],
            "device": device,
        },
        "synthetic": {"fit": synthetic_fit, "output": synthetic_report},
        "real_development": {"fit": real_fit, "output": real_report, "samples": len(development.sample_ids)},
        "checkpoint": {
            "roundtrip": True,
            "sha256": _checkpoint_sha256(checkpoint),
            "path_persisted": False,
        },
        "determinism": {"same_seed_replay": True, "rtol": 1e-5, "atol": 1e-6},
        "resources": {
            "wall_seconds": wall_time,
            "max_rss_kib_end": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "max_rss_kib_delta_lower_bound": int(
                max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before)
            ),
            "peak_cuda_bytes": _cuda_peak_bytes(),
            "download_bytes": 0,
        },
        "test_access": False,
    }
    _atomic_json(model_dir / "status.json", result)
    return result


def _load_existing_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {result["model_id"]: result for result in payload.get("results", [])}


def run_stage1(
    *,
    development_batch_path: Path,
    output_dir: Path,
    model_ids: Sequence[str],
    seed: int = 2693,
    device: str = "cpu",
) -> dict[str, Any]:
    development, development_manifest = load_development_batch(development_batch_path)
    output_dir = Path(output_dir)
    status_path = output_dir / "stage1_status.json"
    results = _load_existing_results(status_path)
    for model_id in model_ids:
        try:
            results[model_id] = contract_smoke_model(
                model_id=model_id,
                development=development,
                output_dir=output_dir,
                seed=seed,
                device=device,
            )
        except Stage1GateError as exc:
            results[model_id] = {
                "model_id": model_id,
                "status": "skipped",
                "evidence_state": "scouted",
                "skip": exc.to_dict(),
                "source_lock": source_lock_entry(model_id),
                "dependencies": _dependency_versions(model_id),
                "resources": {"download_bytes": 0},
                "test_access": False,
            }
            _atomic_json(output_dir / model_id / "status.json", results[model_id])
        except Exception as exc:
            results[model_id] = {
                "model_id": model_id,
                "status": "failed",
                "evidence_state": "scouted",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines(),
                },
                "source_lock": source_lock_entry(model_id),
                "dependencies": _dependency_versions(model_id),
                "test_access": False,
            }
            _atomic_json(output_dir / model_id / "status.json", results[model_id])
        ordered = [results[model] for model in _locked_model_order() if model in results]
        _atomic_json(
            status_path,
            {
                "schema_version": 1,
                "track_id": "property",
                "stage": 1,
                "root_seed": seed,
                "source_lock_sha256": source_lock_sha256(),
                "development_batch": development_manifest,
                "test_access": False,
                "results": ordered,
                "counts": _status_counts(ordered),
            },
        )
    return json.loads(status_path.read_text(encoding="utf-8"))


def _locked_model_order() -> list[str]:
    payload = json.loads(
        (PROJECT_ROOT / "_models/property/source_lock.json").read_text(encoding="utf-8")
    )
    return list(payload["model_order"])


def _status_counts(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {"contract_smoked": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result["status"]] += 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="prepare a real development-only small batch")
    prepare.add_argument("--train-h5", type=Path, required=True)
    prepare.add_argument("--guard-npz", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--max-samples", type=int, default=64)
    run = subparsers.add_parser("run", help="run model contract smokes from a prepared batch")
    run.add_argument("--development-batch", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--models", default=",".join(_locked_model_order()))
    run.add_argument("--seed", type=int, default=2693)
    run.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare_development_batch(
            train_h5=args.train_h5,
            guard_npz=args.guard_npz,
            output_path=args.output,
            max_samples=args.max_samples,
        )
    else:
        requested = [model.strip() for model in args.models.split(",") if model.strip()]
        unknown = sorted(set(requested) - set(_locked_model_order()))
        if unknown:
            raise ValueError(f"models are absent from source lock: {unknown}")
        report = run_stage1(
            development_batch_path=args.development_batch,
            output_dir=args.output_dir,
            model_ids=requested,
            seed=args.seed,
            device=args.device,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return int(report.get("counts", {}).get("failed", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
