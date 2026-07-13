"""Reservoir-prefixed P5 Stage-3 multiseed LOGO4 confirmation.

The runner consumes development-only train/guard assets and a hash-locked P4
manifest.  It has no frozen-test loader, path, command, or metric surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

import reservoir_p5_stage2 as stage2  # noqa: E402
from _code.ml_framework.contracts import ModelBatch, ModelOutput  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.property._p5_common import (  # noqa: E402
    PROPERTY_TARGETS,
    Stage1GateError,
    source_lock_sha256,
)
from p5_contract import build_task_spec, model_to_physical  # noqa: E402

DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p5_stage3"
DEFAULT_BATCH = DEFAULT_OUTPUT_DIR / "runtime" / "development_logo4.npz"
DEFAULT_BUDGET = HERE / "reservoir_p5_stage3_budget.json"
DEFAULT_P4_PHIF_SPLIT = (
    PROJECT_ROOT
    / "_pipelines/02_task_datasets/sweetspot/targets/porosity/_outputs/phif/split_manifest.json"
)
DEFAULT_P4_KLOGH_SPLIT = (
    PROJECT_ROOT
    / "_pipelines/02_task_datasets/sweetspot/targets/permeability/_outputs/klogh/split_manifest.json"
)
STAGE2_BUDGET = HERE / "reservoir_p5_stage2_budget.json"
STAGE2_RUNNER = HERE / "reservoir_p5_stage2.py"
STAGE2_RESULTS = HERE / "_outputs/p5_stage2/p5_stage2_results.jsonl"
FROZEN_TEST_FAMILY = "15/9-F-15"
REPEAT_SEEDS = (1867973658, 2137841944, 3902865753)
TABULAR_LANE = "tabular_cpu"
MONAI_LANE = "seismic_3d_gpu"
STATUS_VALUES = ("completed", "skipped", "failed", "timeout", "data_blocked")
UNIT = {"PHIF": "fraction", "KLOGH": "mD", "SW": "fraction"}


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
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_budget(path: Path = DEFAULT_BUDGET) -> dict[str, Any]:
    budget = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_budget(budget)
    return budget


def _flatten_target_models(budget: Mapping[str, Any]) -> set[str]:
    return {
        model_id
        for values in budget["targets"].values()
        for model_id in values
    }


def validate_budget(budget: Mapping[str, Any]) -> None:
    if (
        budget.get("track_id"),
        budget.get("stage"),
        budget.get("root_seed"),
        tuple(budget.get("repeat_seeds", ())),
    ) != ("property", 3, 2693, REPEAT_SEEDS):
        raise ValueError("Stage-3 identity or repeat seeds changed")
    expected_targets = {
        "PHIF": [
            "extra_trees_regressor",
            "lightgbm_regressor",
            "hist_gradient_boosting_regressor",
        ],
        "KLOGH": [
            "lightgbm_regressor",
            "extra_trees_regressor",
            "xgboost_regressor",
        ],
        "SW": [
            "lightgbm_regressor",
            "hist_gradient_boosting_regressor",
            "xgboost_regressor",
        ],
    }
    if budget.get("targets") != expected_targets:
        raise ValueError("Stage-3 top-3 target candidates changed")
    if budget.get("expected_cells") != 108 or budget.get("hpo") is not False:
        raise ValueError("Stage-3 expected cell count or HPO policy changed")
    if budget.get("lane") != TABULAR_LANE:
        raise ValueError("Stage-3 property cells must remain in the tabular lane")
    firewall = budget["test_firewall"]
    if (
        firewall.get("frozen_test_family") != FROZEN_TEST_FAMILY
        or firewall.get("runner_accepts_test_path")
        or firewall.get("test_loader_implemented")
        or firewall.get("test_metrics_allowed")
    ):
        raise ValueError("Stage-3 frozen-test firewall changed")
    if budget["monai_lane"] != {
        "lane": MONAI_LANE,
        "model_id": "monai_densenet3d_regressor",
        "stage3_cells": 0,
        "status": "not_rankable",
        "reason": "single Stage-2 candidate; no top-3 can be formed and cross-lane ranking is forbidden",
    }:
        raise ValueError("MONAI single-candidate lane changed")
    if "monai_densenet3d_regressor" in _flatten_target_models(budget):
        raise ValueError("MONAI cannot enter a tabular Stage-3 task")

    stage2_budget = json.loads(STAGE2_BUDGET.read_text(encoding="utf-8"))
    source = budget["stage2_source"]
    required_hashes = {
        "runner_sha256": _hash_file(STAGE2_RUNNER),
        "budget_sha256": _hash_file(STAGE2_BUDGET),
        "results_sha256": _hash_file(STAGE2_RESULTS),
    }
    if any(source.get(key) != value for key, value in required_hashes.items()):
        raise ValueError("Stage-2 source evidence differs from the Stage-3 lock")
    for model_id in _flatten_target_models(budget):
        inherited = stage2_budget["model_budgets"][model_id]
        frozen = budget["model_budgets"][model_id]
        for key in (
            "kind",
            "max_wall_seconds",
            "update_steps",
            "update_unit",
            "batch_size",
            "config",
        ):
            if frozen[key] != inherited[key]:
                raise ValueError(f"{model_id} Stage-2 budget field {key} changed")
    if _hash_file(DEFAULT_P4_PHIF_SPLIT) != budget["fold_policy"][
        "p4_phif_split_manifest_sha256"
    ]:
        raise ValueError("P4 PHIF split manifest differs from the Stage-3 lock")
    if _hash_file(DEFAULT_P4_KLOGH_SPLIT) != budget["fold_policy"][
        "p4_klogh_split_manifest_sha256"
    ]:
        raise ValueError("P4 KLOGH split manifest differs from the Stage-3 lock")


def expected_cells(budget: Mapping[str, Any]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for target in PROPERTY_TARGETS:
        for model_id in budget["targets"][target]:
            for fold_id in budget["fold_policy"]["fold_ids"]:
                for repeat_id, repeat_seed in enumerate(budget["repeat_seeds"]):
                    cell_id = (
                        f"{target.lower()}__{TABULAR_LANE}__{model_id}__"
                        f"fold{fold_id}__repeat{repeat_id}"
                    )
                    values.append(
                        {
                            "cell_id": cell_id,
                            "target": target,
                            "lane": TABULAR_LANE,
                            "model_id": model_id,
                            "fold_id": int(fold_id),
                            "repeat_id": int(repeat_id),
                            "repeat_seed": int(repeat_seed),
                        }
                    )
    if len(values) != budget["expected_cells"]:
        raise RuntimeError("frozen Stage-3 matrix does not contain 108 cells")
    if len({value["cell_id"] for value in values}) != len(values):
        raise RuntimeError("frozen Stage-3 matrix contains duplicate cells")
    return values


def _validate_p4_logo4(
    manifest: Mapping[str, Any], budget: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    policy = budget["fold_policy"]
    development = set(policy["development_groups"])
    if (
        manifest.get("group_key") != policy["group_key"]
        or set(manifest.get("development_groups", ())) != development
        or manifest.get("test_groups") != [FROZEN_TEST_FAMILY]
    ):
        raise ValueError("P4 manifest family boundary changed")
    folds = list(manifest.get("folds", ()))
    if len(folds) != 4 or [fold.get("fold_id") for fold in folds] != [0, 1, 2, 3]:
        raise ValueError("P4 manifest is not the frozen LOGO4 split")
    validation_groups: list[str] = []
    for fold in folds:
        train_groups = set(fold["train_groups"])
        validation = list(fold["validation_groups"])
        if (
            len(validation) != 1
            or train_groups != development - set(validation)
            or train_groups & set(validation)
            or FROZEN_TEST_FAMILY in train_groups | set(validation)
        ):
            raise ValueError(f"P4 fold {fold['fold_id']} violates LOGO4")
        if set(fold["train_sample_ids"]) & set(fold["validation_sample_ids"]):
            raise ValueError(f"P4 fold {fold['fold_id']} sample IDs overlap")
        validation_groups.extend(validation)
    if set(validation_groups) != development or len(validation_groups) != 4:
        raise ValueError("P4 validation groups do not cover development exactly once")
    return folds


def prepare_logo4(
    train_h5: Path,
    guard_npz: Path,
    p4_split_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    budget = load_budget()
    p4_split_path = Path(p4_split_path)
    if _hash_file(p4_split_path) != budget["fold_policy"][
        "p4_phif_split_manifest_sha256"
    ]:
        raise ValueError("prepare requires the frozen P4 PHIF manifest")
    task_path = p4_split_path.with_name("task_spec.json")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if task.get("targets") != ["PHIF"]:
        raise ValueError("LOGO4 source must be the PHIF P4 task")
    manifest = json.loads(p4_split_path.read_text(encoding="utf-8"))
    folds = _validate_p4_logo4(manifest, budget)
    rows = stage2._read_records(Path(train_h5), Path(guard_npz), str(task["task_id"]))
    sample_budget = budget["sample_budget"]
    archive: dict[str, Any] = {}
    portable_folds: list[dict[str, Any]] = []
    for fold in folds:
        train, validation = stage2._select_fixed(
            rows,
            fold,
            int(sample_budget["max_train_samples_per_fold"]),
            int(sample_budget["max_validation_samples_per_fold"]),
        )
        selected = train + validation
        prefix = f"fold_{fold['fold_id']}"
        labels = np.stack([row["label"] for row in selected])[:, :3]
        masks = np.isfinite(labels)
        seismic = np.stack([row["seismic"] for row in selected])
        logs = np.stack([row["logs"] for row in selected])
        if seismic.shape[1:] != (3, 3, 9) or logs.shape[1:] != (9, 8):
            raise ValueError("real property input shape changed")
        if (
            not masks.any(axis=0).all()
            or not np.isfinite(seismic).all()
            or not np.isfinite(logs).all()
        ):
            raise ValueError(f"fold {fold['fold_id']} contains invalid development data")
        archive.update(
            {
                f"{prefix}_seismic_patch": seismic,
                f"{prefix}_well_log_sequence": logs,
                f"{prefix}_labels_model_domain": np.where(masks, labels, 0.0),
                f"{prefix}_target_masks": masks.astype(np.uint8),
                f"{prefix}_split": np.asarray(
                    ["train"] * len(train) + ["validation"] * len(validation)
                ),
                f"{prefix}_sample_ids": np.asarray(
                    [row["sample_id"] for row in selected]
                ),
                f"{prefix}_family_ids": np.asarray(
                    [row["family_id"] for row in selected]
                ),
                f"{prefix}_well_ids": np.asarray([row["well_id"] for row in selected]),
                f"{prefix}_depths_m": np.asarray(
                    [row["depth_m"] for row in selected], dtype=np.float64
                ),
            }
        )
        portable_folds.append(
            {
                "fold_id": int(fold["fold_id"]),
                "train_groups": list(fold["train_groups"]),
                "validation_groups": list(fold["validation_groups"]),
                "train_sample_ids": [row["sample_id"] for row in train],
                "validation_sample_ids": [row["sample_id"] for row in validation],
                "family_counts": {
                    split_name: {
                        family: sum(row["family_id"] == family for row in values)
                        for family in sorted({row["family_id"] for row in values})
                    }
                    for split_name, values in (
                        ("train", train),
                        ("validation", validation),
                    )
                },
                "independent_target_valid_counts": {
                    target: {
                        "train": int(masks[: len(train), index].sum()),
                        "validation": int(masks[len(train) :, index].sum()),
                    }
                    for index, target in enumerate(PROPERTY_TARGETS)
                },
                "p4_fold_signature_sha256": _hash_payload(
                    {
                        "fold_id": fold["fold_id"],
                        "train_groups": fold["train_groups"],
                        "validation_groups": fold["validation_groups"],
                        "train_sample_ids_sha256": _hash_payload(
                            fold["train_sample_ids"]
                        ),
                        "validation_sample_ids_sha256": _hash_payload(
                            fold["validation_sample_ids"]
                        ),
                    }
                ),
            }
        )
    split_manifest: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "kind": "mother_well_family_LOGO4",
        "source": {
            "p4_phif_split_manifest_sha256": _hash_file(p4_split_path),
            "p4_klogh_split_manifest_sha256": _hash_file(DEFAULT_P4_KLOGH_SPLIT),
            "p4_task_spec_sha256": _hash_file(task_path),
            "train_hdf5_sha256": _hash_file(train_h5),
            "guard_npz_sha256": _hash_file(guard_npz),
            "paths_persisted": False,
        },
        "selection_registered_before_modeling": True,
        "selection_policy": sample_budget["selection"],
        "development_groups": budget["fold_policy"]["development_groups"],
        "folds": portable_folds,
        "temporary_fractional_split_used": False,
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family": FROZEN_TEST_FAMILY,
            "frozen_test_ids_persisted": False,
        },
    }
    split_manifest["split_hash"] = _hash_payload(split_manifest)
    archive["split_manifest_json"] = np.asarray(
        json.dumps(split_manifest, sort_keys=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **archive)
    return {
        **split_manifest,
        "development_batch_sha256": _hash_file(output_path),
    }


def _make_batch(
    indices: np.ndarray,
    arrays: Mapping[str, Any],
    split_hash: str,
    fold_id: int,
) -> ModelBatch:
    labels, masks = arrays["labels"], arrays["masks"]
    return ModelBatch(
        inputs={
            "tabular": arrays["tabular"][indices],
            "seismic_patch": arrays["seismic"][indices],
            "well_log_sequence": arrays["logs"][indices],
        },
        targets={
            target: labels[indices, index]
            for index, target in enumerate(PROPERTY_TARGETS)
        },
        input_masks={
            "well_log_observed": arrays["logs"][indices, :, 4:8] > 0.5
        },
        target_masks={
            target: masks[indices, index]
            for index, target in enumerate(PROPERTY_TARGETS)
        },
        sample_ids=[arrays["sample_ids"][index] for index in indices],
        groups={
            "mother_well_family": [
                arrays["families"][index] for index in indices
            ],
            "well_id": [arrays["wells"][index] for index in indices],
        },
        coordinates={"depth_m": arrays["depths"][indices]},
        metadata={
            "stage": 3,
            "fold_id": fold_id,
            "split_hash": split_hash,
            "preprocessing_fit": "fold_train_only",
            "test_access": False,
        },
    )


def load_fold(
    development_batch: Path, fold_id: int
) -> tuple[ModelBatch, ModelBatch, dict[str, Any]]:
    with np.load(development_batch, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["split_manifest_json"]))
        stored_hash = manifest.pop("split_hash")
        if _hash_payload(manifest) != stored_hash:
            raise RuntimeError("Stage-3 split manifest hash mismatch")
        manifest["split_hash"] = stored_hash
        fold_manifest = next(
            (fold for fold in manifest["folds"] if fold["fold_id"] == fold_id),
            None,
        )
        if fold_manifest is None:
            raise KeyError(f"fold {fold_id} is absent from Stage-3 data")
        prefix = f"fold_{fold_id}"
        split = archive[f"{prefix}_split"].astype(str)
        seismic_raw = np.asarray(archive[f"{prefix}_seismic_patch"], dtype=float)
        logs_raw = np.asarray(archive[f"{prefix}_well_log_sequence"], dtype=float)
        arrays: dict[str, Any] = {
            "labels": np.asarray(
                archive[f"{prefix}_labels_model_domain"], dtype=float
            ),
            "masks": np.asarray(archive[f"{prefix}_target_masks"], dtype=bool),
            "sample_ids": archive[f"{prefix}_sample_ids"].astype(str).tolist(),
            "families": archive[f"{prefix}_family_ids"].astype(str).tolist(),
            "wells": archive[f"{prefix}_well_ids"].astype(str).tolist(),
            "depths": np.asarray(archive[f"{prefix}_depths_m"], dtype=float),
        }
    if (
        manifest["test_firewall"]["test_access"]
        or FROZEN_TEST_FAMILY in arrays["families"]
    ):
        raise RuntimeError("Stage-3 fold violates the frozen-test firewall")
    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError(f"fold {fold_id} is empty")
    if set(np.asarray(arrays["families"])[train_indices]) & set(
        np.asarray(arrays["families"])[validation_indices]
    ):
        raise RuntimeError(f"fold {fold_id} family overlap")
    stats = stage2._fit_stats(seismic_raw[train_indices], logs_raw[train_indices])
    arrays["seismic"], arrays["logs"], arrays["tabular"] = stage2._transform(
        seismic_raw, logs_raw, stats
    )
    train = _make_batch(train_indices, arrays, stored_hash, fold_id)
    validation = _make_batch(validation_indices, arrays, stored_hash, fold_id)
    evidence = {
        "fold_id": fold_id,
        "split_hash": stored_hash,
        "train_groups": fold_manifest["train_groups"],
        "validation_groups": fold_manifest["validation_groups"],
        "train_sample_ids_sha256": _hash_payload(sorted(train.sample_ids)),
        "validation_sample_ids_sha256": _hash_payload(
            sorted(validation.sample_ids)
        ),
        "fit_validation_overlap": bool(
            set(train.sample_ids) & set(validation.sample_ids)
        ),
        "preprocessing": {
            "fit": "fold_train_only",
            "stats_sha256": _hash_payload(
                {key: value.tolist() for key, value in stats.items()}
            ),
            "target_statistics_fitted": False,
            "target_transform_fitted": False,
            "class_weights": "not_applicable_regression",
            "calibration": "none",
        },
    }
    if evidence["fit_validation_overlap"]:
        raise RuntimeError(f"fold {fold_id} preprocessing sets overlap")
    return train, validation, evidence


def _regression(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    return stage2._regression(np.asarray(actual), np.asarray(predicted))


def evaluate_target(
    target: str, batch: ModelBatch, output: ModelOutput
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mask = np.asarray(batch.target_masks[target], dtype=bool)
    if not mask.any():
        raise ValueError(f"{target} validation mask is empty")
    truth_model = np.asarray(batch.targets[target], dtype=float)[mask]
    prediction_model = np.asarray(output.raw[target], dtype=float)[mask]
    truth_physical = model_to_physical(target, truth_model, prediction=False)
    prediction_physical = np.asarray(output.transformed[target], dtype=float)[mask]
    if not all(
        np.isfinite(values).all()
        for values in (
            truth_model,
            prediction_model,
            truth_physical,
            prediction_physical,
        )
    ):
        raise FloatingPointError(f"{target} metric arrays are non-finite")
    families = np.asarray(batch.groups["mother_well_family"])[mask]
    per_family: dict[str, Any] = {}
    for family in sorted(set(families)):
        selected = families == family
        per_family[family] = {
            "valid_count": int(selected.sum()),
            "physical": _regression(
                truth_physical[selected], prediction_physical[selected]
            ),
            "model_domain": _regression(
                truth_model[selected], prediction_model[selected]
            ),
        }
    worst = max(
        per_family,
        key=lambda family: (per_family[family]["physical"]["RMSE"], family),
    )
    outside = (
        ((prediction_model < 0) | (prediction_model > 1))
        if target in {"PHIF", "SW"}
        else prediction_model < 0
    )
    metric = {
        "valid_count": int(mask.sum()),
        "target_mask_sha256": _hash_payload(mask.astype(np.uint8).tolist()),
        "unit": UNIT[target],
        "model_domain_name": (
            "log1p(KLOGH_mD)" if target == "KLOGH" else "identity"
        ),
        "physical": _regression(truth_physical, prediction_physical),
        "model_domain": _regression(truth_model, prediction_model),
        "raw_out_of_physical_range_rate": float(np.mean(outside)),
        "mother_families": per_family,
        "worst_mother_family": {"family_id": worst, **per_family[worst]},
    }
    arrays = {
        "mask": mask,
        "truth_model": truth_model,
        "prediction_model": prediction_model,
        "truth_physical": truth_physical,
        "prediction_physical": prediction_physical,
    }
    return metric, arrays


def _checkpoint_path(output_dir: Path, cell: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "runtime/checkpoints"
        / cell["target"].lower()
        / cell["model_id"]
        / f"fold{cell['fold_id']}__repeat{cell['repeat_id']}.bin"
    )


def _oof_path(output_dir: Path, cell: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "runtime/oof"
        / cell["target"].lower()
        / cell["model_id"]
        / f"fold{cell['fold_id']}__repeat{cell['repeat_id']}.npz"
    )


def _relative_runtime_path(path: Path, output_dir: Path) -> str:
    value = path.relative_to(output_dir).as_posix()
    if not value.startswith("runtime/"):
        raise ValueError("large Stage-3 artifacts must stay in private runtime")
    return value


def _safe_error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error).replace(str(PROJECT_ROOT), "<project>")[:1000],
    }


def run_cell(
    cell: Mapping[str, Any],
    train: ModelBatch,
    validation: ModelBatch,
    fold_evidence: Mapping[str, Any],
    budget: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    model_id, target = str(cell["model_id"]), str(cell["target"])
    cell_budget = budget["model_budgets"][model_id]
    repeat_seed = int(cell["repeat_seed"])
    config = stage2._config(model_id, cell_budget, repeat_seed, "cpu")
    task_spec = build_task_spec()
    discovered = discover_model("property", model_id)
    if list(discovered.capabilities["input_modalities"]) != ["tabular"]:
        raise RuntimeError(f"{model_id} is not eligible for the tabular lane")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    checkpoint = _checkpoint_path(output_dir, cell)
    oof_path = _oof_path(output_dir, cell)
    with stage2._timeout(int(cell_budget["max_wall_seconds"])):
        model = discovered.build(task_spec, **config)
        fit_report = model.fit(train)
        output = model.predict(validation)
        prediction_matrix = stage2._raw_matrix(output)
        metric, arrays = evaluate_target(target, validation, output)
        model.save_checkpoint(checkpoint)
        restored = discovered.build(task_spec, **config)
        restored.load_checkpoint(checkpoint)
        restored_matrix = stage2._raw_matrix(restored.predict(validation))
        if not np.allclose(
            prediction_matrix, restored_matrix, rtol=1e-6, atol=1e-7
        ):
            raise RuntimeError("Stage-3 checkpoint round-trip changed predictions")
        mask = arrays["mask"]
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            oof_path,
            sample_ids=np.asarray(validation.sample_ids)[mask],
            family_ids=np.asarray(validation.groups["mother_well_family"])[mask],
            well_ids=np.asarray(validation.groups["well_id"])[mask],
            depths_m=np.asarray(validation.coordinates["depth_m"])[mask],
            truth_model_domain=arrays["truth_model"],
            prediction_model_domain=arrays["prediction_model"],
            truth_physical=arrays["truth_physical"],
            prediction_physical=arrays["prediction_physical"],
        )
    wall = time.perf_counter() - started
    return {
        "schema_version": 1,
        **dict(cell),
        "task_id": f"reservoir_property_stage3_{target.lower()}",
        "status": "completed",
        "reason": None,
        "evidence_state": "development_logo4_multiseed_completed",
        "seed": {
            "root": budget["root_seed"],
            "repeat_id": cell["repeat_id"],
            "model": repeat_seed,
            "loader": "not_applicable_full_batch_tree",
            "sampler": "not_applicable_full_batch_tree",
        },
        "split": {
            "split_hash": fold_evidence["split_hash"],
            "fold_id": cell["fold_id"],
            "train_groups": fold_evidence["train_groups"],
            "validation_groups": fold_evidence["validation_groups"],
            "train_sample_ids_sha256": fold_evidence[
                "train_sample_ids_sha256"
            ],
            "validation_sample_ids_sha256": fold_evidence[
                "validation_sample_ids_sha256"
            ],
        },
        "preprocessing": fold_evidence["preprocessing"],
        "input_budget": {
            **budget["sample_budget"],
            "actual_train_samples": len(train.sample_ids),
            "actual_validation_samples": len(validation.sample_ids),
            "n_features": 153,
            "input_modalities": ["tabular"],
        },
        "training_budget": {
            "inherited_from_stage2": True,
            "kind": cell_budget["kind"],
            "update_steps": cell_budget["update_steps"],
            "update_unit": cell_budget["update_unit"],
            "batch_size": cell_budget["batch_size"],
            "max_wall_seconds": cell_budget["max_wall_seconds"],
            "config": cell_budget["config"],
            "config_sha256": _hash_payload(cell_budget["config"]),
            "hpo": False,
            "fit_calls": 1,
            "loss": budget["train_loss"],
            "ranked_target": target,
            "adapter_targets_fit_independently": list(PROPERTY_TARGETS),
        },
        "fit_report": fit_report,
        "validation": {
            "target": target,
            "metric": metric,
            "independent_target_mask": True,
            "validation_mother_well_families": fold_evidence[
                "validation_groups"
            ],
        },
        "checkpoint": {
            "roundtrip": True,
            "sha256": stage2._checkpoint_hash(checkpoint),
            "path_persisted": False,
            "git_ignored": True,
        },
        "oof": {
            "rows": metric["valid_count"],
            "sha256": _hash_file(oof_path),
            "relative_path": _relative_runtime_path(oof_path, output_dir),
            "git_ignored": True,
        },
        "resources": {
            "wall_seconds": wall,
            "max_rss_kib_end": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
            "max_rss_kib_delta_lower_bound": int(
                max(
                    0,
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    - rss_before,
                )
            ),
            "device": "cpu",
            "download_bytes": 0,
        },
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
            "frozen_test_family_seen": False,
        },
    }


def _failure_row(
    cell: Mapping[str, Any],
    budget: Mapping[str, Any],
    fold_evidence: Mapping[str, Any] | None,
    status: str,
    error: BaseException,
) -> dict[str, Any]:
    if status not in STATUS_VALUES or status == "completed":
        raise ValueError(f"invalid structured failure status {status}")
    return {
        "schema_version": 1,
        **dict(cell),
        "task_id": f"reservoir_property_stage3_{str(cell['target']).lower()}",
        "status": status,
        "reason": _safe_error(error),
        "evidence_state": "structured_noncompletion",
        "seed": {
            "root": budget["root_seed"],
            "repeat_id": cell["repeat_id"],
            "model": cell["repeat_seed"],
        },
        "split": (
            None
            if fold_evidence is None
            else {
                "split_hash": fold_evidence["split_hash"],
                "fold_id": cell["fold_id"],
                "train_groups": fold_evidence["train_groups"],
                "validation_groups": fold_evidence["validation_groups"],
            }
        ),
        "resources": {"device": "cpu", "download_bytes": 0},
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
        },
    }


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    cell_ids = [value.get("cell_id") for value in values]
    if len(cell_ids) != len(set(cell_ids)):
        raise RuntimeError("Stage-3 results contain duplicate cells")
    return {str(value["cell_id"]): value for value in values}


def _write_results(
    path: Path,
    results: Mapping[str, Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
) -> None:
    ordered = [results[cell["cell_id"]] for cell in matrix if cell["cell_id"] in results]
    _atomic_text(
        path,
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in ordered
        ),
    )


def _bootstrap_ci(
    values: Sequence[float], replicates: int, seed: int, confidence: float
) -> list[float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires finite nonempty values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return [float(value) for value in np.quantile(means, [alpha, 1.0 - alpha])]


def _tabular_leaderboard(
    target: str,
    values: Sequence[Mapping[str, Any]],
    budget: Mapping[str, Any],
    split_hash: str,
) -> dict[str, Any]:
    allowed_models = list(budget["targets"][target])
    expected = [
        cell
        for cell in expected_cells(budget)
        if cell["target"] == target and cell["lane"] == TABULAR_LANE
    ]
    expected_ids = {cell["cell_id"] for cell in expected}
    task_values = [value for value in values if value.get("target") == target]
    for value in task_values:
        if (
            value.get("lane") != TABULAR_LANE
            or value.get("model_id") not in allowed_models
            or value.get("cell_id") not in expected_ids
        ):
            raise RuntimeError(f"{target} leaderboard detected cross-lane/task pollution")
    completed = [value for value in task_values if value.get("status") == "completed"]
    task_completion = len(completed) / len(expected)
    minimum = float(budget["metric_policy"]["minimum_legal_completion_rate"])
    entries: list[dict[str, Any]] = []
    for model_id in allowed_models:
        model_values = [value for value in completed if value["model_id"] == model_id]
        model_expected = 4 * len(REPEAT_SEEDS)
        model_completion = len(model_values) / model_expected
        if model_completion < minimum:
            continue
        rmses = [value["validation"]["metric"]["physical"]["RMSE"] for value in model_values]
        fold_means = {
            str(fold_id): float(
                np.mean(
                    [
                        value["validation"]["metric"]["physical"]["RMSE"]
                        for value in model_values
                        if value["fold_id"] == fold_id
                    ]
                )
            )
            for fold_id in range(4)
        }
        seed_means = {
            str(seed): float(
                np.mean(
                    [
                        value["validation"]["metric"]["physical"]["RMSE"]
                        for value in model_values
                        if value["repeat_seed"] == seed
                    ]
                )
            )
            for seed in REPEAT_SEEDS
        }
        entries.append(
            {
                "model_id": model_id,
                "completed_cells": len(model_values),
                "expected_cells": model_expected,
                "completion_rate": model_completion,
                "mean_physical_RMSE": float(np.mean(rmses)),
                "mean_physical_RMSE_bootstrap_95_CI": _bootstrap_ci(
                    rmses,
                    int(budget["metric_policy"]["bootstrap_replicates"]),
                    stage2.stable_seed(
                        budget["root_seed"], "stage3", "bootstrap", target, model_id
                    ),
                    float(budget["metric_policy"]["bootstrap_confidence"]),
                ),
                "mean_physical_MAE": float(
                    np.mean(
                        [
                            value["validation"]["metric"]["physical"]["MAE"]
                            for value in model_values
                        ]
                    )
                ),
                "mean_physical_R2": float(
                    np.mean(
                        [
                            value["validation"]["metric"]["physical"]["R2"]
                            for value in model_values
                        ]
                    )
                ),
                "mean_model_domain_RMSE": float(
                    np.mean(
                        [
                            value["validation"]["metric"]["model_domain"]["RMSE"]
                            for value in model_values
                        ]
                    )
                ),
                "fold_mean_physical_RMSE": fold_means,
                "worst_fold": max(
                    fold_means,
                    key=lambda fold_id: (fold_means[fold_id], fold_id),
                ),
                "worst_fold_physical_RMSE": max(fold_means.values()),
                "seed_mean_physical_RMSE": seed_means,
                "seed_mean_RMSE_std": float(
                    np.std(list(seed_means.values()), ddof=0)
                ),
                "mean_wall_seconds": float(
                    np.mean([value["resources"]["wall_seconds"] for value in model_values])
                ),
                "max_wall_seconds": float(
                    np.max([value["resources"]["wall_seconds"] for value in model_values])
                ),
                "input_modalities": ["tabular"],
            }
        )
    entries.sort(
        key=lambda entry: (
            entry["mean_physical_RMSE"],
            entry["worst_fold_physical_RMSE"],
            entry["seed_mean_RMSE_std"],
            entry["mean_wall_seconds"],
            entry["model_id"],
        )
    )
    rankable = task_completion >= minimum and len(entries) >= 2
    if rankable:
        for rank, entry in enumerate(entries, 1):
            entry["rank"] = rank
    return {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "task_id": f"reservoir_property_stage3_{target.lower()}",
        "target": target,
        "lane": TABULAR_LANE,
        "status": "rankable" if rankable else "not_rankable",
        "expected_cells": len(expected),
        "completed_cells": len(completed),
        "legal_completion_rate": task_completion,
        "minimum_legal_completion_rate": minimum,
        "primary_metric": "mean_physical_RMSE",
        "direction": "minimize",
        "ranking_tiebreakers": budget["metric_policy"]["ranking_tiebreakers"],
        "split_hash": split_hash,
        "cross_lane_ranking": False,
        "entries": entries,
        "test_access": False,
    }


def _monai_leaderboard(
    target: str, budget: Mapping[str, Any], split_hash: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "task_id": f"reservoir_property_stage3_{target.lower()}",
        "target": target,
        "lane": MONAI_LANE,
        "status": "not_rankable",
        "candidate_count": 1,
        "stage3_expected_cells": 0,
        "stage3_completed_cells": 0,
        "candidates": [budget["monai_lane"]],
        "entries": [],
        "reason": budget["monai_lane"]["reason"],
        "split_hash": split_hash,
        "cross_lane_ranking": False,
        "test_access": False,
    }


def _oof_manifest(
    values: Sequence[Mapping[str, Any]], split_hash: str
) -> dict[str, Any]:
    entries = [
        {
            "cell_id": value["cell_id"],
            "target": value["target"],
            "lane": value["lane"],
            "model_id": value["model_id"],
            "fold_id": value["fold_id"],
            "repeat_id": value["repeat_id"],
            "repeat_seed": value["repeat_seed"],
            **value["oof"],
        }
        for value in values
        if value.get("status") == "completed"
    ]
    return {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "kind": "development_LOGO4_OOF",
        "split_hash": split_hash,
        "expected_entries": 108,
        "actual_entries": len(entries),
        "raw_predictions_git_ignored": True,
        "paths_are_project_relative": True,
        "entries": entries,
        "test_firewall": {"test_access": False, "test_metrics": False},
    }


def _read_oof(
    output_dir: Path, entry: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    path = output_dir / str(entry["relative_path"])
    if not path.is_file() or _hash_file(path) != entry["sha256"]:
        raise RuntimeError(f"OOF hash/path mismatch for {entry['cell_id']}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def build_visualization_data(
    output_dir: Path,
    values: Sequence[Mapping[str, Any]],
    boards: Mapping[str, Mapping[str, Any]],
    split_hash: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "split_hash": split_hash,
        "source": "development OOF only",
        "aggregation": "rank-1 model prediction mean and standard deviation across the three frozen repeat seeds",
        "targets": {},
        "test_firewall": {"test_access": False, "test_metrics": False},
    }
    for target in PROPERTY_TARGETS:
        board = boards[target]
        if board["status"] != "rankable" or not board["entries"]:
            payload["targets"][target] = {
                "status": "not_feasible_without_rankable_development_board"
            }
            continue
        winner = board["entries"][0]["model_id"]
        winner_values = [
            value
            for value in values
            if value.get("status") == "completed"
            and value.get("target") == target
            and value.get("model_id") == winner
        ]
        if len(winner_values) != 12:
            payload["targets"][target] = {
                "status": "not_feasible_incomplete_winner_oof",
                "winner_model_id": winner,
                "completed_winner_cells": len(winner_values),
                "expected_winner_cells": 12,
            }
            continue
        samples: dict[str, dict[str, Any]] = {}
        for value in winner_values:
            archive = _read_oof(output_dir, {"cell_id": value["cell_id"], **value["oof"]})
            count = len(archive["sample_ids"])
            for index in range(count):
                sample_id = str(archive["sample_ids"][index])
                record = samples.setdefault(
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "family_id": str(archive["family_ids"][index]),
                        "well_id": str(archive["well_ids"][index]),
                        "depth_m": float(archive["depths_m"][index]),
                        "fold_id": int(value["fold_id"]),
                        "truth_model_domain": float(
                            archive["truth_model_domain"][index]
                        ),
                        "truth_physical": float(archive["truth_physical"][index]),
                        "prediction_model_domain": [],
                        "prediction_physical": [],
                        "repeat_seeds": [],
                    },
                )
                if (
                    record["family_id"] != str(archive["family_ids"][index])
                    or record["well_id"] != str(archive["well_ids"][index])
                    or not math.isclose(
                        record["truth_physical"],
                        float(archive["truth_physical"][index]),
                    )
                ):
                    raise RuntimeError(f"OOF sample identity drift for {sample_id}")
                record["prediction_model_domain"].append(
                    float(archive["prediction_model_domain"][index])
                )
                record["prediction_physical"].append(
                    float(archive["prediction_physical"][index])
                )
                record["repeat_seeds"].append(int(value["repeat_seed"]))
        portable_samples: list[dict[str, Any]] = []
        for sample_id in sorted(samples):
            record = samples[sample_id]
            if sorted(record["repeat_seeds"]) != sorted(REPEAT_SEEDS):
                raise RuntimeError(f"OOF sample {sample_id} does not have three repeats")
            portable_samples.append(
                {
                    key: record[key]
                    for key in (
                        "sample_id",
                        "family_id",
                        "well_id",
                        "depth_m",
                        "fold_id",
                        "truth_model_domain",
                        "truth_physical",
                    )
                }
                | {
                    "prediction_model_domain_mean": float(
                        np.mean(record["prediction_model_domain"])
                    ),
                    "prediction_model_domain_std": float(
                        np.std(record["prediction_model_domain"], ddof=0)
                    ),
                    "prediction_physical_mean": float(
                        np.mean(record["prediction_physical"])
                    ),
                    "prediction_physical_std": float(
                        np.std(record["prediction_physical"], ddof=0)
                    ),
                }
            )
        fold_seed_metrics = [
            {
                "model_id": value["model_id"],
                "fold_id": value["fold_id"],
                "repeat_id": value["repeat_id"],
                "repeat_seed": value["repeat_seed"],
                "physical_RMSE": value["validation"]["metric"]["physical"][
                    "RMSE"
                ],
            }
            for value in values
            if value.get("status") == "completed" and value.get("target") == target
        ]
        family_model_rmse: list[dict[str, Any]] = []
        for entry in board["entries"]:
            model_id = entry["model_id"]
            model_values = [
                value
                for value in values
                if value.get("status") == "completed"
                and value.get("target") == target
                and value.get("model_id") == model_id
            ]
            grouped: dict[str, list[float]] = defaultdict(list)
            for value in model_values:
                metric = value["validation"]["metric"]
                family = metric["worst_mother_family"]["family_id"]
                grouped[family].append(metric["physical"]["RMSE"])
            for family in sorted(grouped):
                family_model_rmse.append(
                    {
                        "model_id": model_id,
                        "family_id": family,
                        "mean_physical_RMSE": float(np.mean(grouped[family])),
                    }
                )
        payload["targets"][target] = {
            "status": "ready",
            "unit": UNIT[target],
            "winner_model_id": winner,
            "winner_rank_evidence": board["entries"][0],
            "aggregated_sample_count": len(portable_samples),
            "expected_repeat_count_per_sample": 3,
            "samples": portable_samples,
            "family_model_RMSE": family_model_rmse,
            "fold_seed_metrics": fold_seed_metrics,
        }
    return payload


def _resolved_serif_font() -> tuple[str, str]:
    from matplotlib import font_manager

    requested = "Times New Roman"
    try:
        font_manager.findfont(requested, fallback_to_default=False)
        return requested, requested
    except ValueError:
        fallback = "Liberation Serif"
        font_manager.findfont(fallback, fallback_to_default=False)
        return requested, fallback


def normalize_fonts(fig: Any, family: str) -> None:
    """Normalize all existing matplotlib text before every save."""
    import matplotlib.text as mtext

    for item in fig.findobj(match=mtext.Text):
        item.set_fontfamily(family)
        text = item.get_text().strip()
        if len(text) == 1 and text in "abcdefgh":
            item.set_fontsize(max(float(item.get_fontsize()), 10.0))
            item.set_fontweight("bold")
        else:
            item.set_fontsize(max(float(item.get_fontsize()), 7.0))


def _panel(ax: Any, letter: str) -> None:
    ax.text(
        -0.10,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="#111111",
        va="bottom",
        ha="left",
    )


def _pretty_model(model_id: str) -> str:
    return {
        "extra_trees_regressor": "Extra Trees",
        "lightgbm_regressor": "LightGBM",
        "hist_gradient_boosting_regressor": "HistGradientBoosting",
        "xgboost_regressor": "XGBoost",
    }[model_id]


def _save_figure(fig: Any, path: Path, family: str) -> None:
    normalize_fonts(fig, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")


def _companion_markdown(
    target: str,
    kind: str,
    target_data: Mapping[str, Any],
    board: Mapping[str, Any],
) -> str:
    winner = target_data["winner_model_id"]
    top = board["entries"][0]
    lines = [
        f"# {target} Stage-3 {kind.replace('_', ' ')}",
        "",
        "## Meta",
        f"- Target: `{target}` ({UNIT[target]})",
        f"- Development-only winner shown: `{winner}`",
        f"- Aggregated OOF samples: {target_data['aggregated_sample_count']}",
        "- Repeats per sample: 3",
        "- Frozen test access: false",
        "",
        "## Quantitative Summary",
        "",
        "| Rank | Model | Mean RMSE | 95% bootstrap CI | Worst-fold RMSE | Seed SD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for entry in board["entries"]:
        ci = entry["mean_physical_RMSE_bootstrap_95_CI"]
        lines.append(
            f"| {entry['rank']} | {entry['model_id']} | "
            f"{entry['mean_physical_RMSE']:.8g} | "
            f"[{ci[0]:.8g}, {ci[1]:.8g}] | "
            f"{entry['worst_fold_physical_RMSE']:.8g} | "
            f"{entry['seed_mean_RMSE_std']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Visual Description",
            f"This panel is reconstructed from the portable mean-over-repeat OOF aggregate for {target}; it contains no frozen-test prediction or metric.",
            "",
            "## Boundary",
            f"The rank-1 development model has mean RMSE {top['mean_physical_RMSE']:.8g}. Stage-3 confirmation is not a frozen-test claim.",
            "",
        ]
    )
    return "\n".join(lines)


def render_figures(
    output_dir: Path,
    visualization_data: Mapping[str, Any] | None = None,
    boards: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if visualization_data is None:
        visualization_data = json.loads(
            (output_dir / "p5_stage3_visualization_data.json").read_text(
                encoding="utf-8"
            )
        )
    if visualization_data.get("test_firewall", {}).get("test_access"):
        raise RuntimeError("visualization data violates the frozen-test firewall")
    if boards is None:
        boards = {
            target: json.loads(
                (
                    output_dir
                    / f"leaderboard_{target.lower()}_{TABULAR_LANE}.json"
                ).read_text(encoding="utf-8")
            )
            for target in PROPERTY_TARGETS
        }
    requested_font, resolved_font = _resolved_serif_font()
    palette = ["#376795", "#72BCD5", "#E76254", "#FFD06F"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [resolved_font],
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
            "axes.linewidth": 0.5,
            "savefig.dpi": 300,
        }
    )
    figure_dir = output_dir / "figures"
    entries: list[dict[str, Any]] = []
    for target in PROPERTY_TARGETS:
        data = visualization_data["targets"][target]
        if data.get("status") != "ready":
            continue
        samples = data["samples"]
        truth = np.asarray([value["truth_physical"] for value in samples])
        prediction = np.asarray(
            [value["prediction_physical_mean"] for value in samples]
        )
        residual = prediction - truth
        board = boards[target]

        fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), squeeze=False)
        families = sorted({value["family_id"] for value in samples})
        for panel_index, (ax, family_id) in enumerate(zip(axes.flat, families)):
            family_rows = [
                value for value in samples if value["family_id"] == family_id
            ]
            for well_index, well_id in enumerate(
                sorted({value["well_id"] for value in family_rows})
            ):
                well_rows = sorted(
                    [value for value in family_rows if value["well_id"] == well_id],
                    key=lambda value: value["depth_m"],
                )
                depth = [value["depth_m"] for value in well_rows]
                ax.plot(
                    [value["truth_physical"] for value in well_rows],
                    depth,
                    color=palette[0],
                    label="Truth" if well_index == 0 else None,
                )
                ax.plot(
                    [value["prediction_physical_mean"] for value in well_rows],
                    depth,
                    color=palette[3],
                    linestyle="--",
                    label="Prediction" if well_index == 0 else None,
                )
            ax.invert_yaxis()
            ax.set_xlabel(f"{target} ({UNIT[target]}) · {family_id}")
            ax.set_ylabel("Depth (m)")
            ax.grid(alpha=0.2, linewidth=0.4)
            ax.legend(frameon=False, loc="best")
            _panel(ax, chr(ord("a") + panel_index))
        path = figure_dir / f"{target.lower()}_per_well_depth.png"
        fig.tight_layout()
        _save_figure(fig, path, resolved_font)
        plt.close(fig)
        entries.append({"target": target, "kind": "per_well_depth", "mode": "D", "path": path.relative_to(output_dir).as_posix(), "sha256": _hash_file(path)})

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.scatter(truth, prediction, s=10, alpha=0.55, color=palette[0], edgecolors="none")
        low, high = float(min(truth.min(), prediction.min())), float(max(truth.max(), prediction.max()))
        ax.plot([low, high], [low, high], color="#333333", linestyle="--")
        ax.set_xlabel(f"Truth ({UNIT[target]})")
        ax.set_ylabel(f"Prediction ({UNIT[target]})")
        ax.grid(alpha=0.2, linewidth=0.4)
        path = figure_dir / f"{target.lower()}_truth_prediction.png"
        fig.tight_layout()
        _save_figure(fig, path, resolved_font)
        plt.close(fig)
        entries.append({"target": target, "kind": "truth_prediction", "mode": "A", "path": path.relative_to(output_dir).as_posix(), "sha256": _hash_file(path)})

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.scatter(prediction, residual, s=10, alpha=0.55, color=palette[2], edgecolors="none")
        ax.axhline(0.0, color="#333333", linestyle="--")
        ax.set_xlabel(f"Prediction ({UNIT[target]})")
        ax.set_ylabel(f"Residual ({UNIT[target]})")
        ax.grid(alpha=0.2, linewidth=0.4)
        path = figure_dir / f"{target.lower()}_residual.png"
        fig.tight_layout()
        _save_figure(fig, path, resolved_font)
        plt.close(fig)
        entries.append({"target": target, "kind": "residual", "mode": "A", "path": path.relative_to(output_dir).as_posix(), "sha256": _hash_file(path)})

        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        models = list(board["entries"])
        family_values = data["family_model_RMSE"]
        positions = np.arange(len(families))
        width = 0.8 / len(models)
        for model_index, entry in enumerate(models):
            lookup = {
                value["family_id"]: value["mean_physical_RMSE"]
                for value in family_values
                if value["model_id"] == entry["model_id"]
            }
            ax.bar(
                positions + (model_index - (len(models) - 1) / 2) * width,
                [lookup[family] for family in families],
                width=width,
                color=palette[model_index],
                label=_pretty_model(entry["model_id"]),
            )
        ax.set_xticks(positions, families)
        ax.set_ylabel(f"Mean RMSE ({UNIT[target]})")
        ax.set_xlabel("Validation mother-well family")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
        ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
        path = figure_dir / f"{target.lower()}_worst_family.png"
        fig.tight_layout()
        _save_figure(fig, path, resolved_font)
        plt.close(fig)
        entries.append({"target": target, "kind": "worst_family", "mode": "B", "path": path.relative_to(output_dir).as_posix(), "sha256": _hash_file(path)})

        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        markers = ["o", "s", "^"]
        for model_index, entry in enumerate(models):
            model_rows = [value for value in data["fold_seed_metrics"] if value["model_id"] == entry["model_id"]]
            for row in model_rows:
                jitter = (model_index - 1) * 0.16 + (row["repeat_id"] - 1) * 0.035
                ax.scatter(row["fold_id"] + jitter, row["physical_RMSE"], color=palette[model_index], marker=markers[row["repeat_id"]], s=24, alpha=0.8)
        ax.set_xticks(range(4), [f"Fold {index}" for index in range(4)])
        ax.set_xlabel("Frozen LOGO4 fold")
        ax.set_ylabel(f"RMSE ({UNIT[target]})")
        ax.grid(axis="y", alpha=0.2, linewidth=0.4)
        model_handles = [Line2D([0], [0], color=palette[index], marker="o", linestyle="none", label=_pretty_model(entry["model_id"])) for index, entry in enumerate(models)]
        seed_handles = [Line2D([0], [0], color="#555555", marker=markers[index], linestyle="none", label=f"Repeat {index + 1}") for index in range(3)]
        first = ax.legend(handles=model_handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.add_artist(first)
        ax.legend(handles=seed_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0.0))
        path = figure_dir / f"{target.lower()}_fold_seed_distribution.png"
        fig.tight_layout()
        _save_figure(fig, path, resolved_font)
        plt.close(fig)
        entries.append({"target": target, "kind": "fold_seed_distribution", "mode": "B", "path": path.relative_to(output_dir).as_posix(), "sha256": _hash_file(path)})

        for entry in [value for value in entries if value["target"] == target]:
            image_path = output_dir / entry["path"]
            companion = image_path.with_suffix(".md")
            _atomic_text(
                companion,
                _companion_markdown(target, entry["kind"], data, board),
            )
            entry["companion_path"] = companion.relative_to(output_dir).as_posix()
            entry["companion_sha256"] = _hash_file(companion)
    return {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "source_visualization_data_sha256": _hash_payload(visualization_data),
        "requested_font": requested_font,
        "resolved_font": resolved_font,
        "font_fallback_reason": (
            None
            if requested_font == resolved_font
            else "Times New Roman is unavailable on the execution host; Liberation Serif is the metric-compatible local fallback"
        ),
        "palette": "Ukiyo-e",
        "dpi": 300,
        "figure_count": len(entries),
        "figures": entries,
        "rebuild_command": "python3 _pipelines/02_task_datasets/reservoir/reservoir_p5_stage3.py render",
        "inputs": ["p5_stage3_visualization_data.json", "leaderboard_<target>_tabular_cpu.json"],
        "test_firewall": {"test_access": False, "test_metrics": False},
    }


def write_outputs(
    output_dir: Path,
    results: Mapping[str, Mapping[str, Any]],
    budget: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = expected_cells(budget)
    _write_results(output_dir / "p5_stage3_results.jsonl", results, matrix)
    ordered = [
        results[cell["cell_id"]]
        for cell in matrix
        if cell["cell_id"] in results
    ]
    split_hash = str(split_manifest["split_hash"])
    boards: dict[str, dict[str, Any]] = {}
    leaderboard_hashes: dict[str, str] = {}
    for target in PROPERTY_TARGETS:
        tabular_board = _tabular_leaderboard(
            target, ordered, budget, split_hash
        )
        monai_board = _monai_leaderboard(target, budget, split_hash)
        tabular_path = (
            output_dir / f"leaderboard_{target.lower()}_{TABULAR_LANE}.json"
        )
        monai_path = output_dir / f"leaderboard_{target.lower()}_{MONAI_LANE}.json"
        _atomic_json(tabular_path, tabular_board)
        _atomic_json(monai_path, monai_board)
        boards[target] = tabular_board
        leaderboard_hashes[f"{target}/{TABULAR_LANE}"] = _hash_file(tabular_path)
        leaderboard_hashes[f"{target}/{MONAI_LANE}"] = _hash_file(monai_path)
    oof_manifest = _oof_manifest(ordered, split_hash)
    _atomic_json(output_dir / "p5_stage3_oof_manifest.json", oof_manifest)
    visualization_data = build_visualization_data(
        output_dir, ordered, boards, split_hash
    )
    visualization_data_path = output_dir / "p5_stage3_visualization_data.json"
    _atomic_json(visualization_data_path, visualization_data)
    visualization_manifest = render_figures(
        output_dir, visualization_data, boards
    )
    visualization_manifest["source_visualization_data_sha256"] = _hash_file(
        visualization_data_path
    )
    _atomic_json(
        output_dir / "p5_stage3_visualization_manifest.json",
        visualization_manifest,
    )
    seed_manifest = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "root_seed": budget["root_seed"],
        "repeat_seeds": budget["repeat_seeds"],
        "derivation": "frozen by P5 Stage-3 protocol",
        "passed_directly_to_every_estimator_random_state": True,
        "traditional_deterministic_models_repeated": True,
        "test_access": False,
    }
    _atomic_json(output_dir / "p5_stage3_seed_manifest.json", seed_manifest)
    _atomic_json(output_dir / "p5_stage3_split_manifest.json", split_manifest)
    _atomic_json(output_dir / "p5_stage3_budget.json", budget)
    counts = {
        status: sum(value.get("status") == status for value in ordered)
        for status in STATUS_VALUES
    }
    expected_count = int(budget["expected_cells"])
    completed_count = counts["completed"]
    task_status = {
        target: boards[target]["status"] for target in PROPERTY_TARGETS
    }
    summary = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 3,
        "baseline_commit": budget["baseline_commit"],
        "root_seed": budget["root_seed"],
        "repeat_seeds": budget["repeat_seeds"],
        "expected_cells": expected_count,
        "attempted_cells": len(ordered),
        "missing_cells": [
            cell["cell_id"] for cell in matrix if cell["cell_id"] not in results
        ],
        "counts": counts,
        "legal_completion_rate": completed_count / expected_count,
        "task_status": task_status,
        "lanes": {
            TABULAR_LANE: {
                "expected_cells": expected_count,
                "completed_cells": completed_count,
                "targets": task_status,
                "ranked": True,
            },
            MONAI_LANE: {
                "expected_cells": 0,
                "completed_cells": 0,
                "candidate_count": 1,
                "model_id": budget["monai_lane"]["model_id"],
                "status": "not_rankable",
                "ranked": False,
            },
        },
        "cross_lane_ranking": False,
        "split_hash": split_hash,
        "source_hashes": {
            "protocol_sha256": budget["protocol_sha256"],
            "runner_sha256": _hash_file(Path(__file__)),
            "budget_sha256": _hash_file(DEFAULT_BUDGET),
            "stage2_runner_sha256": _hash_file(STAGE2_RUNNER),
            "stage2_budget_sha256": _hash_file(STAGE2_BUDGET),
            "stage2_results_sha256": _hash_file(STAGE2_RESULTS),
            "source_lock_sha256": source_lock_sha256(),
        },
        "artifact_hashes": {
            "results_sha256": _hash_file(output_dir / "p5_stage3_results.jsonl"),
            "split_manifest_sha256": _hash_file(
                output_dir / "p5_stage3_split_manifest.json"
            ),
            "seed_manifest_sha256": _hash_file(
                output_dir / "p5_stage3_seed_manifest.json"
            ),
            "oof_manifest_sha256": _hash_file(
                output_dir / "p5_stage3_oof_manifest.json"
            ),
            "visualization_data_sha256": _hash_file(visualization_data_path),
            "visualization_manifest_sha256": _hash_file(
                output_dir / "p5_stage3_visualization_manifest.json"
            ),
            "leaderboards": leaderboard_hashes,
        },
        "portable": {
            "absolute_paths_persisted": False,
            "raw_checkpoints_persisted": False,
            "complete_predictions_persisted": False,
            "source_data_copied": False,
        },
        "test_firewall": {
            "test_access": False,
            "test_loader_implemented": False,
            "test_metrics": False,
        },
    }
    _atomic_json(output_dir / "p5_stage3_summary.json", summary)
    return summary


def run_stage3(development_batch: Path, output_dir: Path) -> dict[str, Any]:
    budget = load_budget()
    matrix = expected_cells(budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "p5_stage3_results.jsonl"
    results = _load_results(result_path)
    expected_ids = {cell["cell_id"] for cell in matrix}
    if set(results) - expected_ids:
        raise RuntimeError("existing Stage-3 results contain unregistered cells")
    with np.load(development_batch, allow_pickle=False) as archive:
        split_manifest = json.loads(str(archive["split_manifest_json"]))
    split_hash = split_manifest.pop("split_hash")
    if _hash_payload(split_manifest) != split_hash:
        raise RuntimeError("Stage-3 development archive split hash changed")
    split_manifest["split_hash"] = split_hash
    fold_cache: dict[int, tuple[ModelBatch, ModelBatch, dict[str, Any]]] = {}
    for cell in matrix:
        cell_id = cell["cell_id"]
        if cell_id in results:
            continue
        fold_id = int(cell["fold_id"])
        fold_evidence: dict[str, Any] | None = None
        try:
            if fold_id not in fold_cache:
                fold_cache[fold_id] = load_fold(development_batch, fold_id)
            train, validation, fold_evidence = fold_cache[fold_id]
            results[cell_id] = run_cell(
                cell,
                train,
                validation,
                fold_evidence,
                budget,
                output_dir,
            )
        except Stage1GateError as error:
            results[cell_id] = _failure_row(
                cell, budget, fold_evidence, "skipped", error
            )
        except stage2.Stage2Timeout as error:
            results[cell_id] = _failure_row(
                cell, budget, fold_evidence, "timeout", error
            )
        except (FileNotFoundError, KeyError) as error:
            results[cell_id] = _failure_row(
                cell, budget, fold_evidence, "data_blocked", error
            )
        except Exception as error:
            results[cell_id] = _failure_row(
                cell, budget, fold_evidence, "failed", error
            )
        _write_results(result_path, results, matrix)
    summary = write_outputs(output_dir, results, budget, split_manifest)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="materialize the frozen development-only LOGO4 archive"
    )
    prepare.add_argument("--train-h5", type=Path, required=True)
    prepare.add_argument("--guard-npz", type=Path, required=True)
    prepare.add_argument(
        "--p4-split-manifest", type=Path, default=DEFAULT_P4_PHIF_SPLIT
    )
    run = commands.add_parser("run", help="execute all 108 frozen Stage-3 cells")
    run.add_argument("--development-batch", type=Path, default=DEFAULT_BATCH)
    commands.add_parser(
        "render", help="rebuild figures from committed portable aggregates"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare_logo4(
            args.train_h5,
            args.guard_npz,
            args.p4_split_manifest,
            DEFAULT_BATCH,
        )
    elif args.command == "run":
        report = run_stage3(args.development_batch, DEFAULT_OUTPUT_DIR)
    else:
        data_path = DEFAULT_OUTPUT_DIR / "p5_stage3_visualization_data.json"
        if not data_path.is_file():
            raise FileNotFoundError("portable Stage-3 visualization data is absent")
        report = render_figures(DEFAULT_OUTPUT_DIR)
        report["source_visualization_data_sha256"] = _hash_file(data_path)
        _atomic_json(
            DEFAULT_OUTPUT_DIR / "p5_stage3_visualization_manifest.json", report
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    counts = report.get("counts", {})
    return int(
        counts.get("failed", 0)
        + counts.get("timeout", 0)
        + counts.get("data_blocked", 0)
        > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
