"""Fixed-budget P5 Stage-2 pilot for the reservoir-property track.

Only P4-locked development IDs, real train.h5, and guard arrays are accepted.
There is no frozen-test data argument or loader.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import resource
import signal
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
from p5_contract import build_task_spec, model_to_physical  # noqa: E402

FROZEN_TEST_FAMILY = "15/9-F-15"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p5_stage2"
DEFAULT_BATCH = DEFAULT_OUTPUT_DIR / "runtime" / "fixed_fold.npz"
DEFAULT_BUDGET = HERE / "reservoir_p5_stage2_budget.json"
DEFAULT_P4_SPLIT = (
    PROJECT_ROOT
    / "_pipelines/02_task_datasets/sweetspot/targets/porosity/_outputs/phif/split_manifest.json"
)
GPU_LOCK_PATH = Path("/mnt/data/yongan-admin-2/.cache/volve-p5/locks/gpu0.lock")


class Stage2Timeout(TimeoutError):
    pass


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def stable_seed(root_seed: int, *parts: Any) -> int:
    value = "|".join([str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def load_budget(path: Path = DEFAULT_BUDGET) -> dict[str, Any]:
    budget = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_budget(budget)
    return budget


def validate_budget(budget: Mapping[str, Any]) -> None:
    if (budget.get("track_id"), budget.get("stage"), budget.get("root_seed")) != (
        "property",
        2,
        2693,
    ):
        raise ValueError("budget identity or root seed changed")
    locked = json.loads(
        (PROJECT_ROOT / "_models/property/source_lock.json").read_text(encoding="utf-8")
    )["model_order"]
    if budget.get("model_order") != locked:
        raise ValueError("Stage-2 candidates differ from the Stage-1 source lock")
    for model_id in locked:
        cell = budget["model_budgets"][model_id]
        kind, wall, updates = cell["kind"], cell["max_wall_seconds"], cell["update_steps"]
        if kind == "tree" and wall > 300:
            raise ValueError(f"{model_id} exceeds the CPU limit")
        if kind == "neural_tabular" and (wall > 600 or updates > 200):
            raise ValueError(f"{model_id} exceeds the tabular-neural limit")
        if kind == "neural_3d" and (wall > 900 or updates > 80):
            raise ValueError(f"{model_id} exceeds the 3D limit")
        if kind == "gated_weight" and updates != 0:
            raise ValueError("gated weight candidates cannot receive training updates")
    firewall = budget["test_firewall"]
    if (
        firewall["frozen_test_family"] != FROZEN_TEST_FAMILY
        or firewall["runner_accepts_test_path"]
        or firewall["test_metrics_allowed"]
    ):
        raise ValueError("Stage-2 test firewall changed")


def _decoded_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return dict(json.loads(str(value)))


def _p4_sample_id(
    task_id: str,
    family_id: str,
    well_id: str,
    depth_m: float,
    source_kind: str,
    source_key: str | int,
) -> str:
    raw = "|".join(
        [task_id, family_id, well_id, f"{depth_m:.6f}", source_kind, str(source_key)]
    )
    return f"{task_id}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def _read_records(train_h5: Path, guard_npz: Path, task_id: str) -> list[dict[str, Any]]:
    if Path(train_h5).name != "train.h5":
        raise ValueError("only an explicit development file named train.h5 is accepted")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("prepare requires h5py; pilot run does not") from exc
    rows: list[dict[str, Any]] = []
    with h5py.File(train_h5, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            meta = _decoded_json(group.attrs["meta"])
            family = str(meta["family_id"])
            if family == FROZEN_TEST_FAMILY:
                raise RuntimeError("frozen-test family found in development train data")
            rows.append(
                {
                    "sample_id": _p4_sample_id(
                        task_id,
                        family,
                        str(meta["well_id"]),
                        float(meta["depth_m"]),
                        "hdf5",
                        key,
                    ),
                    "family_id": family,
                    "well_id": str(meta["well_id"]),
                    "depth_m": float(meta["depth_m"]),
                    "seismic": np.asarray(group["seismic_patch"][()], dtype=np.float64),
                    "logs": np.asarray(group["well_log_seq"][()], dtype=np.float64),
                    "label": np.asarray(group["label"][()], dtype=np.float64).reshape(-1),
                }
            )
    with np.load(guard_npz, allow_pickle=False) as archive:
        required = {"seismic_patch", "well_log_seq", "label", "meta_json"}
        if not required <= set(archive.files):
            raise ValueError(f"guard archive missing {sorted(required - set(archive.files))}")
        for index in range(len(archive["label"])):
            meta = _decoded_json(str(archive["meta_json"][index]))
            family = str(meta["family_id"])
            if family == FROZEN_TEST_FAMILY:
                raise RuntimeError("frozen-test family found in development guard data")
            rows.append(
                {
                    "sample_id": _p4_sample_id(
                        task_id,
                        family,
                        str(meta["well_id"]),
                        float(meta["depth_m"]),
                        "guard_npz",
                        index,
                    ),
                    "family_id": family,
                    "well_id": str(meta["well_id"]),
                    "depth_m": float(meta["depth_m"]),
                    "seismic": np.asarray(archive["seismic_patch"][index], dtype=np.float64),
                    "logs": np.asarray(archive["well_log_seq"][index], dtype=np.float64),
                    "label": np.asarray(archive["label"][index], dtype=np.float64).reshape(-1),
                }
            )
    if len(rows) != len({row["sample_id"] for row in rows}):
        raise RuntimeError("development sample IDs are not unique")
    return rows


def validate_p4_fold(
    manifest: Mapping[str, Any], expected_hash: str, actual_hash: str
) -> Mapping[str, Any]:
    if actual_hash != expected_hash:
        raise ValueError("P4 split hash differs from the frozen budget")
    if manifest.get("group_key") != "mother_well_family":
        raise ValueError("P4 split does not isolate mother-well families")
    if manifest.get("test_groups") != [FROZEN_TEST_FAMILY]:
        raise ValueError("P4 frozen-test group changed")
    if not manifest.get("folds") or manifest["folds"][0].get("fold_id") != 0:
        raise ValueError("P4 first development fold is unavailable")
    fold = manifest["folds"][0]
    train_groups, val_groups = set(fold["train_groups"]), set(fold["validation_groups"])
    if not train_groups or not val_groups or train_groups & val_groups:
        raise ValueError("fold groups are empty or overlapping")
    if FROZEN_TEST_FAMILY in train_groups | val_groups:
        raise ValueError("frozen-test family reached the development fold")
    if set(fold["train_sample_ids"]) & set(fold["validation_sample_ids"]):
        raise ValueError("fold sample IDs overlap")
    return fold


def _depth_spread(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["well_id"], row["depth_m"], row["sample_id"]))
    if len(ordered) <= limit:
        return ordered
    return [ordered[index] for index in np.linspace(0, len(ordered) - 1, limit, dtype=int)]


def _select_fixed(
    rows: Sequence[dict[str, Any]], fold: Mapping[str, Any], max_train: int, max_val: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["sample_id"]: row for row in rows}
    locked = set(fold["train_sample_ids"]) | set(fold["validation_sample_ids"])
    missing = locked - set(by_id)
    if missing:
        raise RuntimeError(f"{len(missing)} locked development IDs are missing")
    train_all = [by_id[value] for value in fold["train_sample_ids"]]
    val_all = [by_id[value] for value in fold["validation_sample_ids"]]
    families = sorted({row["family_id"] for row in train_all})
    base, remainder = divmod(max_train, len(families))
    train: list[dict[str, Any]] = []
    for index, family in enumerate(families):
        family_rows = [row for row in train_all if row["family_id"] == family]
        train.extend(_depth_spread(family_rows, base + int(index < remainder)))
    train.sort(key=lambda row: (row["family_id"], row["well_id"], row["depth_m"]))
    return train, _depth_spread(val_all, max_val)


def prepare_fixed_fold(
    train_h5: Path,
    guard_npz: Path,
    p4_split_path: Path,
    output_path: Path,
    budget_path: Path = DEFAULT_BUDGET,
) -> dict[str, Any]:
    budget = load_budget(budget_path)
    p4_split_path = Path(p4_split_path)
    task_path = p4_split_path.with_name("task_spec.json")
    manifest = json.loads(p4_split_path.read_text(encoding="utf-8"))
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("targets") != ["PHIF"]:
        raise ValueError("the joint fold source must be the frozen P4 PHIF task")
    fold = validate_p4_fold(
        manifest,
        budget["joint_fold_policy"]["p4_split_manifest_sha256"],
        _hash_file(p4_split_path),
    )
    rows = _read_records(Path(train_h5), Path(guard_npz), str(task["task_id"]))
    sample_budget = budget["sample_budget"]
    train, validation = _select_fixed(
        rows,
        fold,
        int(sample_budget["max_train_samples"]),
        int(sample_budget["max_validation_samples"]),
    )
    selected = train + validation
    seismic = np.stack([row["seismic"] for row in selected])
    logs = np.stack([row["logs"] for row in selected])
    labels = np.stack([row["label"] for row in selected])[:, :3]
    masks = np.isfinite(labels)
    if seismic.shape[1:] != (3, 3, 9) or logs.shape[1:] != (9, 8):
        raise ValueError("real property input shape changed")
    if not masks.any(axis=0).all() or not np.isfinite(seismic).all() or not np.isfinite(logs).all():
        raise ValueError("fixed fold has invalid inputs or unsupported targets")
    split_payload: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "source": {
            "p4_split_manifest_sha256": _hash_file(p4_split_path),
            "p4_task_spec_sha256": _hash_file(task_path),
            "train_hdf5_sha256": _hash_file(train_h5),
            "guard_npz_sha256": _hash_file(guard_npz),
            "paths_persisted": False,
        },
        "source_fold_id": int(fold["fold_id"]),
        "selection_registered_before_modeling": True,
        "selection_policy": sample_budget["selection"],
        "train_groups": sorted({row["family_id"] for row in train}),
        "validation_groups": sorted({row["family_id"] for row in validation}),
        "train_sample_ids": [row["sample_id"] for row in train],
        "validation_sample_ids": [row["sample_id"] for row in validation],
        "family_counts": {
            split_name: {
                family: sum(row["family_id"] == family for row in values)
                for family in sorted({row["family_id"] for row in values})
            }
            for split_name, values in (("train", train), ("validation", validation))
        },
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "frozen_test_family": FROZEN_TEST_FAMILY,
            "frozen_test_ids_persisted": False,
        },
    }
    split_payload["split_hash"] = _hash_payload(split_payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        seismic_patch=seismic,
        well_log_sequence=logs,
        labels_model_domain=np.where(masks, labels, 0.0),
        target_masks=masks.astype(np.uint8),
        split=np.asarray(["train"] * len(train) + ["validation"] * len(validation)),
        sample_ids=np.asarray([row["sample_id"] for row in selected]),
        family_ids=np.asarray([row["family_id"] for row in selected]),
        well_ids=np.asarray([row["well_id"] for row in selected]),
        depths_m=np.asarray([row["depth_m"] for row in selected], dtype=np.float64),
        split_manifest_json=np.asarray(json.dumps(split_payload, sort_keys=True)),
    )
    return {
        **split_payload,
        "development_batch_sha256": _hash_file(output_path),
        "independent_target_valid_counts": {
            target: {
                "train": int(masks[: len(train), index].sum()),
                "validation": int(masks[len(train) :, index].sum()),
            }
            for index, target in enumerate(PROPERTY_TARGETS)
        },
    }


def _fit_stats(seismic: np.ndarray, logs: np.ndarray) -> dict[str, np.ndarray]:
    seismic_flat = seismic.reshape(len(seismic), -1)
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).reshape(len(logs), -1)
    mean, std = np.zeros(values.shape[1]), np.ones(values.shape[1])
    for column in range(values.shape[1]):
        observed = values[masks[:, column], column]
        if observed.size:
            mean[column], std[column] = observed.mean(), observed.std() + 1e-8
    return {
        "seismic_mean": seismic_flat.mean(axis=0),
        "seismic_std": seismic_flat.std(axis=0) + 1e-8,
        "log_mean": mean,
        "log_std": std,
    }


def _transform(
    seismic: np.ndarray, logs: np.ndarray, stats: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seismic_flat = seismic.reshape(len(seismic), -1)
    seismic_flat = (seismic_flat - stats["seismic_mean"]) / stats["seismic_std"]
    values = logs[:, :, :4].reshape(len(logs), -1)
    masks = (logs[:, :, 4:8] > 0.5).astype(float).reshape(len(logs), -1)
    values = ((values - stats["log_mean"]) / stats["log_std"]) * masks
    tabular = np.concatenate([seismic_flat, values, masks], axis=1)
    normalized_logs = np.concatenate(
        [values.reshape(len(logs), 9, 4), masks.reshape(len(logs), 9, 4)], axis=2
    )
    arrays = (seismic_flat.reshape(seismic.shape), normalized_logs, tabular)
    if tabular.shape[1] != 153 or not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("fold preprocessing produced invalid inputs")
    return arrays


def _make_batch(indices: np.ndarray, arrays: Mapping[str, Any], split_hash: str) -> ModelBatch:
    labels, masks = arrays["labels"], arrays["masks"]
    return ModelBatch(
        inputs={
            "tabular": arrays["tabular"][indices],
            "seismic_patch": arrays["seismic"][indices],
            "well_log_sequence": arrays["logs"][indices],
        },
        targets={target: labels[indices, i] for i, target in enumerate(PROPERTY_TARGETS)},
        input_masks={"well_log_observed": arrays["logs"][indices, :, 4:8] > 0.5},
        target_masks={target: masks[indices, i] for i, target in enumerate(PROPERTY_TARGETS)},
        sample_ids=[arrays["sample_ids"][i] for i in indices],
        groups={
            "mother_well_family": [arrays["families"][i] for i in indices],
            "well_id": [arrays["wells"][i] for i in indices],
        },
        coordinates={"depth_m": arrays["depths"][indices]},
        metadata={
            "stage": 2,
            "split_hash": split_hash,
            "preprocessing_fit": "fold_train_only",
            "test_access": False,
        },
    )


def load_fixed_fold(path: Path) -> tuple[ModelBatch, ModelBatch, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["split_manifest_json"]))
        split_hash = manifest.pop("split_hash")
        if _hash_payload(manifest) != split_hash:
            raise RuntimeError("portable split manifest hash mismatch")
        manifest["split_hash"] = split_hash
        split = archive["split"].astype(str)
        seismic_raw = np.asarray(archive["seismic_patch"], dtype=float)
        logs_raw = np.asarray(archive["well_log_sequence"], dtype=float)
        arrays: dict[str, Any] = {
            "labels": np.asarray(archive["labels_model_domain"], dtype=float),
            "masks": np.asarray(archive["target_masks"], dtype=bool),
            "sample_ids": archive["sample_ids"].astype(str).tolist(),
            "families": archive["family_ids"].astype(str).tolist(),
            "wells": archive["well_ids"].astype(str).tolist(),
            "depths": np.asarray(archive["depths_m"], dtype=float),
        }
    if manifest["test_firewall"]["test_access"] or FROZEN_TEST_FAMILY in arrays["families"]:
        raise RuntimeError("fixed fold violates the frozen-test firewall")
    train_indices, val_indices = np.flatnonzero(split == "train"), np.flatnonzero(split == "validation")
    if not len(train_indices) or not len(val_indices):
        raise RuntimeError("fixed fold is empty")
    stats = _fit_stats(seismic_raw[train_indices], logs_raw[train_indices])
    arrays["seismic"], arrays["logs"], arrays["tabular"] = _transform(
        seismic_raw, logs_raw, stats
    )
    train = _make_batch(train_indices, arrays, split_hash)
    validation = _make_batch(val_indices, arrays, split_hash)
    evidence = {
        **manifest,
        "development_batch_sha256": _hash_file(path),
        "preprocessing": {
            "fit_sample_ids_sha256": _hash_payload(sorted(train.sample_ids)),
            "validation_sample_ids_sha256": _hash_payload(sorted(validation.sample_ids)),
            "fit_validation_overlap": False,
            "target_statistics_fitted": False,
            "denoise": "identity",
            "stats_sha256": _hash_payload(
                {key: value.tolist() for key, value in stats.items()}
            ),
        },
    }
    return train, validation, evidence


def slice_batch(batch: ModelBatch, indices: Sequence[int]) -> ModelBatch:
    selected = np.asarray(indices, dtype=int)
    return ModelBatch(
        inputs={key: np.asarray(value)[selected] for key, value in batch.inputs.items()},
        targets={key: np.asarray(value)[selected] for key, value in (batch.targets or {}).items()},
        input_masks={key: np.asarray(value)[selected] for key, value in batch.input_masks.items()},
        target_masks={key: np.asarray(value)[selected] for key, value in batch.target_masks.items()},
        sample_ids=[batch.sample_ids[index] for index in selected],
        groups={key: [value[index] for index in selected] for key, value in batch.groups.items()},
        coordinates={key: np.asarray(value)[selected] for key, value in batch.coordinates.items()},
        metadata=dict(batch.metadata),
    )


def _raw_matrix(output: ModelOutput) -> np.ndarray:
    matrix = np.column_stack([np.asarray(output.raw[target], dtype=float) for target in PROPERTY_TARGETS])
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise ValueError("model output must be finite [N,3]")
    return matrix


def _regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("metric vectors are empty or misaligned")
    residual = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "R2": None if denominator <= 0 else float(1 - np.sum(residual**2) / denominator),
        "R2_reason": "undefined because ground truth is constant" if denominator <= 0 else None,
    }


def evaluate_targets(batch: ModelBatch, output: ModelOutput) -> dict[str, dict[str, Any]]:
    raw = _raw_matrix(output)
    families = np.asarray(batch.groups["mother_well_family"])
    result: dict[str, dict[str, Any]] = {}
    for index, target in enumerate(PROPERTY_TARGETS):
        mask = np.asarray(batch.target_masks[target], dtype=bool)
        truth_domain = np.asarray(batch.targets[target])[mask]
        predicted_domain = raw[mask, index]
        truth_physical = model_to_physical(target, truth_domain, prediction=False)
        predicted_physical = np.asarray(output.transformed[target])[mask]
        groups: dict[str, Any] = {}
        for family in sorted(set(families[mask])):
            selected = families[mask] == family
            groups[family] = {
                "valid_count": int(selected.sum()),
                "physical": _regression(truth_physical[selected], predicted_physical[selected]),
                "model_domain": _regression(truth_domain[selected], predicted_domain[selected]),
            }
        worst = max(groups, key=lambda family: (groups[family]["physical"]["RMSE"], family))
        outside = (
            ((predicted_domain < 0) | (predicted_domain > 1))
            if target in {"PHIF", "SW"}
            else predicted_domain < 0
        )
        result[target] = {
            "valid_count": int(mask.sum()),
            "mask_sha256": _hash_payload(mask.astype(np.uint8).tolist()),
            "unit": {"PHIF": "fraction", "KLOGH": "mD", "SW": "fraction"}[target],
            "model_domain_name": "log1p(KLOGH_mD)" if target == "KLOGH" else "identity",
            "model_domain": _regression(truth_domain, predicted_domain),
            "physical": _regression(truth_physical, predicted_physical),
            "raw_out_of_physical_range_rate": float(np.mean(outside)),
            "mother_families": groups,
            "worst_mother_family": {"family_id": worst, **groups[worst]},
        }
    return result


def _config(model_id: str, cell: Mapping[str, Any], seed: int, device: str) -> dict[str, Any]:
    return {"seed": seed, "n_features": 153, "device": device, **cell.get("config", {})}


def _batches(batch: ModelBatch, steps: int, batch_size: int, seed: int) -> Iterator[ModelBatch]:
    rng, order, cursor = np.random.default_rng(seed), np.arange(len(batch.sample_ids)), len(batch.sample_ids)
    for _ in range(steps):
        if cursor + batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        chosen = order[cursor : min(cursor + batch_size, len(order))]
        cursor += len(chosen)
        yield slice_batch(batch, chosen)


def _checkpoint_path(root: Path, model_id: str, prefix: str) -> Path:
    return root / model_id / (prefix if model_id == "tabiclv2_regressor" else prefix + ".bin")


def _checkpoint_hash(path: Path) -> str:
    if path.is_file():
        return _hash_file(path)
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(_hash_file(child).encode())
    return digest.hexdigest()


def tiny_gate(
    discovered: Any,
    task_spec: Any,
    model_id: str,
    train: ModelBatch,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    tiny = slice_batch(train, np.arange(min(budget["tiny_gate"]["samples"], len(train.sample_ids))))
    model = discovered.build(task_spec, **config)
    kind = budget["model_budgets"][model_id]["kind"]
    calls = budget["tiny_gate"]["optimizer_steps"] if kind in {"neural_tabular", "neural_3d"} and model_id != "realmlp_regressor" else 1
    reports = [model.fit(tiny) for _ in range(calls)]
    prediction = _raw_matrix(model.predict(tiny))
    checkpoint = _checkpoint_path(root, model_id, "tiny")
    model.save_checkpoint(checkpoint)
    restored = discovered.build(task_spec, **config)
    restored.load_checkpoint(checkpoint)
    if not np.allclose(prediction, _raw_matrix(restored.predict(tiny)), rtol=1e-6, atol=1e-7):
        raise RuntimeError("tiny checkpoint round-trip changed predictions")
    return {
        "status": "passed",
        "samples": len(tiny.sample_ids),
        "fit_reports": reports,
        "finite_output": True,
        "output_shape": list(prediction.shape),
        "checkpoint_roundtrip": True,
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
    }


def _cuda_reset() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _cuda_peak() -> int:
    try:
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


@contextlib.contextmanager
def _timeout(seconds: int) -> Iterator[None]:
    def handler(_signum: int, _frame: Any) -> None:
        raise Stage2Timeout(f"cell exceeded the frozen {seconds}-second budget")
    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def gpu_flock(device: str, lock_path: Path | None) -> Iterator[float]:
    if not device.startswith("cuda"):
        yield 0.0
        return
    if lock_path is None or Path(lock_path) != GPU_LOCK_PATH:
        raise ValueError("CUDA Stage-2 requires the frozen gpu0.lock path")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        waited = time.perf_counter() - started
        try:
            yield waited
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def pilot_model(
    model_id: str,
    train: ModelBatch,
    validation: ModelBatch,
    split: Mapping[str, Any],
    budget: Mapping[str, Any],
    output_dir: Path,
    root_seed: int,
    device: str,
) -> dict[str, Any]:
    cell = budget["model_budgets"][model_id]
    seeds = {
        role: stable_seed(root_seed, role, model_id)
        for role in ("model", "loader", "sampler")
    }
    config, task_spec = _config(model_id, cell, seeds["model"], device), build_task_spec()
    discovered = discover_model("property", model_id)
    input_modalities = list(discovered.capabilities["input_modalities"])
    expected_modalities = {
        "tabular_cpu": ["tabular"],
        "seismic_3d_gpu": ["seismic_patch"],
    }
    if input_modalities != expected_modalities[cell["lane"]]:
        raise RuntimeError(
            f"{model_id} input modalities {input_modalities} violate lane {cell['lane']}"
        )
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _cuda_reset()
    started = time.perf_counter()
    with _timeout(cell["max_wall_seconds"]):
        gate = tiny_gate(
            discovered,
            task_spec,
            model_id,
            train,
            config,
            budget,
            output_dir / "runtime/checkpoints",
        )
        model, reports = discovered.build(task_spec, **config), []
        if cell["kind"] == "tree" or model_id == "realmlp_regressor":
            reports.append(model.fit(train))
        else:
            for batch in _batches(
                train, cell["update_steps"], cell["batch_size"], seeds["sampler"]
            ):
                reports.append(model.fit(batch))
        output = model.predict(validation)
        target_metrics, prediction = evaluate_targets(validation, output), _raw_matrix(output)
        checkpoint = _checkpoint_path(
            output_dir / "runtime/checkpoints", model_id, "pilot"
        )
        model.save_checkpoint(checkpoint)
        restored = discovered.build(task_spec, **config)
        restored.load_checkpoint(checkpoint)
        if not np.allclose(
            prediction, _raw_matrix(restored.predict(validation)), rtol=1e-6, atol=1e-7
        ):
            raise RuntimeError("pilot checkpoint round-trip changed predictions")
    wall = time.perf_counter() - started
    return {
        "schema_version": 1,
        "model_id": model_id,
        "task_id": task_spec.task_id,
        "lane": cell["lane"],
        "status": "development_piloted",
        "reason": None,
        "evidence_state": "development_piloted",
        "seed": {"root": root_seed, **seeds},
        "split_hash": split["split_hash"],
        "input_budget": {
            **budget["sample_budget"],
            "actual_train_samples": len(train.sample_ids),
            "actual_validation_samples": len(validation.sample_ids),
            "batch_size": cell["batch_size"],
            "input_modalities": input_modalities,
        },
        "training_budget": {
            "kind": cell["kind"],
            "update_steps": cell["update_steps"],
            "update_unit": cell["update_unit"],
            "max_wall_seconds": cell["max_wall_seconds"],
            "hpo": False,
            "fit_calls": len(reports),
        },
        "tiny_gate": gate,
        "validation": {
            "targets": target_metrics,
            "independent_target_masks": True,
            "mother_well_families": sorted(set(validation.groups["mother_well_family"])),
        },
        "checkpoint": {
            "roundtrip": True,
            "sha256": _checkpoint_hash(checkpoint),
            "path_persisted": False,
        },
        "resources": {
            "wall_seconds": wall,
            "max_rss_kib_end": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "max_rss_kib_delta_lower_bound": int(
                max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before)
            ),
            "peak_cuda_bytes": _cuda_peak(),
            "download_bytes": 0,
        },
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family_seen": False,
        },
    }


def _dependencies(model_id: str) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for item in source_lock_entry(model_id).get("dependencies", []):
        try:
            result[item["distribution"]] = metadata.version(item["distribution"])
        except metadata.PackageNotFoundError:
            result[item["distribution"]] = None
    return result


def _safe_error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error).replace(str(PROJECT_ROOT), "<project>")[:1000],
    }


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return {value["model_id"]: value for value in values}


def _write_results(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
    )


def _lane_leaderboard(
    target: str,
    lane: str,
    values: Sequence[Mapping[str, Any]],
    split_hash: str,
) -> dict[str, Any]:
    entries = []
    for value in values:
        if value.get("lane") != lane or value["status"] != "development_piloted":
            continue
        metric = value["validation"]["targets"].get(target)
        if metric is None or not math.isfinite(metric["physical"]["RMSE"]):
            continue
        entries.append(
            {
                "model_id": value["model_id"],
                "physical_RMSE": metric["physical"]["RMSE"],
                "physical_MAE": metric["physical"]["MAE"],
                "physical_R2": metric["physical"]["R2"],
                "model_domain_RMSE": metric["model_domain"]["RMSE"],
                "model_domain_R2": metric["model_domain"]["R2"],
                "worst_mother_family": metric["worst_mother_family"],
                "valid_count": metric["valid_count"],
                "wall_seconds": value["resources"]["wall_seconds"],
                "input_modalities": value["input_budget"]["input_modalities"],
            }
        )
    entries.sort(
        key=lambda row: (
            row["physical_RMSE"],
            row["worst_mother_family"]["physical"]["RMSE"],
            row["wall_seconds"],
            row["model_id"],
        )
    )
    rankable = len(entries) >= 2
    if rankable:
        for rank, entry in enumerate(entries, 1):
            entry["rank"] = rank
    return {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "task_id": f"reservoir_property_stage2_{target.lower()}_{lane}",
        "target": target,
        "modality_lane": lane,
        "status": "rankable" if rankable else "not_rankable",
        "primary_metric": "physical_RMSE",
        "direction": "minimize",
        "split_hash": split_hash,
        "eligibility": "same modality lane, same frozen fold, development_piloted, finite target metric",
        "candidate_count": len(entries),
        "entries": entries,
        "cross_lane_ranking": False,
        "test_access": False,
    }


def _leaderboard(
    target: str,
    values: Sequence[Mapping[str, Any]],
    split_hash: str,
    lanes: Sequence[str],
) -> dict[str, Any]:
    boards = {
        lane: _lane_leaderboard(target, lane, values, split_hash) for lane in lanes
    }
    return {
        "schema_version": 2,
        "track_id": "property",
        "stage": 2,
        "task_id": f"reservoir_property_stage2_{target.lower()}",
        "target": target,
        "primary_metric": "physical_RMSE",
        "direction": "minimize",
        "split_hash": split_hash,
        "lane_isolation": "strict",
        "cross_lane_ranking": False,
        "lanes": boards,
        "test_access": False,
    }


def write_outputs(
    output_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    budget: Mapping[str, Any],
    split: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = [results[model_id] for model_id in budget["model_order"] if model_id in results]
    result_path = output_dir / "p5_stage2_results.jsonl"
    _write_results(result_path, ordered)
    lanes = list(
        dict.fromkeys(
            budget["model_budgets"][model_id]["lane"]
            for model_id in budget["model_order"]
        )
    )
    leaderboard_hashes = {}
    for target in PROPERTY_TARGETS:
        path = output_dir / f"leaderboard_{target.lower()}.json"
        _atomic_json(path, _leaderboard(target, ordered, split["split_hash"], lanes))
        leaderboard_hashes[target] = _hash_file(path)
    counts = {
        status: sum(value["status"] == status for value in ordered)
        for status in ("development_piloted", "skipped", "failed", "timeout")
    }
    lane_summary = {}
    for lane in lanes:
        lane_results = [value for value in ordered if value.get("lane") == lane]
        lane_counts = {
            status: sum(value["status"] == status for value in lane_results)
            for status in ("development_piloted", "skipped", "failed", "timeout")
        }
        lane_summary[lane] = {
            "expected_cells": sum(
                budget["model_budgets"][model_id]["lane"] == lane
                for model_id in budget["model_order"]
            ),
            "attempted_cells": len(lane_results),
            "counts": lane_counts,
            "leaderboard_status_by_target": {
                target: _lane_leaderboard(
                    target, lane, ordered, split["split_hash"]
                )["status"]
                for target in PROPERTY_TARGETS
            },
            "cross_lane_ranking": False,
        }
    summary = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 2,
        "root_seed": budget["root_seed"],
        "expected_cells": len(budget["model_order"]),
        "attempted_cells": len(ordered),
        "missing_cells": [
            model_id for model_id in budget["model_order"] if model_id not in results
        ],
        "counts": counts,
        "modality_lanes": lane_summary,
        "cross_lane_ranking": False,
        "split_hash": split["split_hash"],
        "source_hashes": {
            "budget_sha256": _hash_file(DEFAULT_BUDGET),
            "source_lock_sha256": source_lock_sha256(),
            "runner_sha256": _hash_file(Path(__file__)),
        },
        "results_sha256": _hash_file(result_path),
        "leaderboard_sha256": leaderboard_hashes,
        "portable": {
            "absolute_paths_persisted": False,
            "raw_checkpoints_persisted": False,
            "source_data_copied": False,
        },
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
        },
    }
    _atomic_json(output_dir / "p5_stage2_summary.json", summary)
    _atomic_json(output_dir / "p5_stage2_split_manifest.json", split)
    _atomic_json(output_dir / "p5_stage2_budget.json", budget)
    return summary


def run_stage2(
    development_batch: Path,
    output_dir: Path,
    model_ids: Sequence[str],
    seed: int,
    device: str,
    gpu_lock_path: Path | None,
) -> dict[str, Any]:
    budget = load_budget()
    if seed != budget["root_seed"]:
        raise ValueError("execution seed differs from the frozen budget")
    if set(model_ids) - set(budget["model_order"]):
        raise ValueError("requested model is absent from the frozen budget")
    train, validation, split = load_fixed_fold(development_batch)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = _load_results(output_dir / "p5_stage2_results.jsonl")
    for model_id in model_ids:
        cell = budget["model_budgets"][model_id]
        expected_cuda = cell["kind"] == "neural_3d"
        if expected_cuda != device.startswith("cuda"):
            raise ValueError(f"{model_id} is outside its frozen device lane")
        lock_wait = 0.0
        try:
            with gpu_flock(device, gpu_lock_path) as lock_wait:
                value = pilot_model(
                    model_id, train, validation, split, budget, output_dir, seed, device
                )
            value["resources"]["gpu_lock_wait_seconds_excluded_from_wall"] = lock_wait
            value["resources"]["gpu_lock"] = "gpu0.lock" if expected_cuda else None
            results[model_id] = value
        except Stage1GateError as exc:
            results[model_id] = {
                "schema_version": 1,
                "model_id": model_id,
                "task_id": build_task_spec().task_id,
                "lane": cell["lane"],
                "status": "skipped",
                "reason": exc.to_dict(),
                "evidence_state": "contract_smoked",
                "seed": {"root": seed, "model": stable_seed(seed, "model", model_id)},
                "split_hash": split["split_hash"],
                "dependencies": _dependencies(model_id),
                "resources": {
                    "download_bytes": 0,
                    "gpu_lock_wait_seconds_excluded_from_wall": lock_wait,
                },
                "test_firewall": {"test_access": False, "test_metrics": False},
            }
        except Stage2Timeout as exc:
            results[model_id] = {
                "schema_version": 1,
                "model_id": model_id,
                "task_id": build_task_spec().task_id,
                "lane": cell["lane"],
                "status": "timeout",
                "reason": _safe_error(exc),
                "seed": {"root": seed, "model": stable_seed(seed, "model", model_id)},
                "split_hash": split["split_hash"],
                "resources": {
                    "download_bytes": 0,
                    "gpu_lock_wait_seconds_excluded_from_wall": lock_wait,
                },
                "test_firewall": {"test_access": False, "test_metrics": False},
            }
        except Exception as exc:
            results[model_id] = {
                "schema_version": 1,
                "model_id": model_id,
                "task_id": build_task_spec().task_id,
                "lane": cell["lane"],
                "status": "failed",
                "reason": _safe_error(exc),
                "seed": {"root": seed, "model": stable_seed(seed, "model", model_id)},
                "split_hash": split["split_hash"],
                "resources": {
                    "download_bytes": 0,
                    "gpu_lock_wait_seconds_excluded_from_wall": lock_wait,
                },
                "test_firewall": {"test_access": False, "test_metrics": False},
            }
        write_outputs(output_dir, results, budget, split)
    return json.loads((output_dir / "p5_stage2_summary.json").read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="materialize the fixed P4 development fold")
    prepare.add_argument("--train-h5", type=Path, required=True)
    prepare.add_argument("--guard-npz", type=Path, required=True)
    prepare.add_argument("--p4-split-manifest", type=Path, default=DEFAULT_P4_SPLIT)
    prepare.add_argument("--output", type=Path, default=DEFAULT_BATCH)
    run = commands.add_parser("run", help="run fixed-budget development pilot cells")
    run.add_argument("--development-batch", type=Path, default=DEFAULT_BATCH)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run.add_argument("--models", required=True)
    run.add_argument("--seed", type=int, default=2693)
    run.add_argument("--device", default="cpu")
    run.add_argument("--gpu-lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        if args.output.resolve() != DEFAULT_BATCH.resolve():
            raise ValueError("prepared data must stay in the reservoir private runtime directory")
        report = prepare_fixed_fold(
            args.train_h5, args.guard_npz, args.p4_split_manifest, args.output
        )
    else:
        if args.output_dir.resolve() != DEFAULT_OUTPUT_DIR.resolve():
            raise ValueError("results must stay in the reservoir private output directory")
        report = run_stage2(
            args.development_batch,
            args.output_dir,
            [value.strip() for value in args.models.split(",") if value.strip()],
            args.seed,
            args.device,
            args.gpu_lock,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    counts = report.get("counts", {})
    return int(counts.get("failed", 0) + counts.get("timeout", 0) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
