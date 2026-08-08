"""P5 Stage-4 confirmation on the explicitly known, reusable F-15 holdout.

This runner is intentionally track-private.  It verifies the frozen Stage-3
winner evidence and prior holdout exposure before fitting.  ``prepare`` only
materializes source arrays in the ignored runtime directory; ``confirm`` fits
on all four development mother-well families, advances a fail-closed single-
use state, and only then loads the known holdout arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from _code.ml_framework.contracts import ModelBatch  # noqa: E402
from _code.ml_framework.model_discovery import discover_model  # noqa: E402
from _models.property._p5_common import PROPERTY_TARGETS, source_lock_sha256  # noqa: E402
from p5_contract import build_task_spec, model_to_physical  # noqa: E402
import reservoir_p5_stage2 as stage2  # noqa: E402


ROOT_SEED = 2693
KNOWN_HOLDOUT_FAMILY = "15/9-F-15"
DEVELOPMENT_FAMILIES = ("15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12")
TARGET_INDEX = {target: index for index, target in enumerate(PROPERTY_TARGETS)}
FROZEN_WINNERS = {
    "PHIF": "extra_trees_regressor",
    "KLOGH": "extra_trees_regressor",
    "SW": "xgboost_regressor",
}
FROZEN_BOARD_HASHES = {
    "PHIF": "590ef021351b04264f0a6e889ac206ef1929aea9d54ca9d823a879f8265aa8bf",
    "KLOGH": "f80a05a039f56b9c9bd5bce9804d77ad811ba66d67e73315b0e6d1c251c5e539",
    "SW": "0d68817ee16e716a1b0ef18d52dc3e3988679b0f1f08a10e2f4cd3956c98c919",
}
EXPECTED_FEATURE_WHITELIST = (
    "ST0202_seismic_patch",
    "GR",
    "RT",
    "NPHI",
    "RHOB",
    "GR_observed_mask",
    "RT_observed_mask",
    "NPHI_observed_mask",
    "RHOB_observed_mask",
)
DEFAULT_CONTRACT = HERE / "reservoir_p5_stage4_contract.json"
DEFAULT_OUTPUT_DIR = HERE / "_outputs" / "p5_stage4_confirmation"
DEFAULT_DEVELOPMENT_ARCHIVE = DEFAULT_OUTPUT_DIR / "runtime" / "development_all_rows.npz"
DEFAULT_HOLDOUT_ARCHIVE = DEFAULT_OUTPUT_DIR / "runtime" / "known_holdout_f15.npz"
STAGE3_DIR = HERE / "_outputs" / "p5_stage3"
P4_PHIF_SPLIT = (
    PROJECT_ROOT
    / "_pipelines/02_task_datasets/sweetspot/targets/porosity/_outputs/phif/split_manifest.json"
)
P4_KLOGH_SPLIT = (
    PROJECT_ROOT
    / "_pipelines/02_task_datasets/sweetspot/targets/permeability/_outputs/klogh/split_manifest.json"
)
PALETTE = {"blue": "#376795", "sky": "#72BCD5", "yellow": "#FFD06F", "red": "#E76254"}
FIGURE_SIZE = (3.5, 3.5)


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_path(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_contract(path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        contract.get("track_id"),
        contract.get("stage"),
        contract.get("root_seed"),
        contract.get("confirmation_kind"),
    ) != ("property", 4, ROOT_SEED, "previously_seen_reusable_holdout"):
        raise ValueError("Stage-4 contract identity changed")
    if contract.get("fresh_blind") is not False or contract.get("prior_test_consumed") is not True:
        raise ValueError("known-holdout disclosure changed")
    if tuple(contract.get("development_families", ())) != DEVELOPMENT_FAMILIES:
        raise ValueError("Stage-4 development families changed")
    if contract.get("known_holdout_family") != KNOWN_HOLDOUT_FAMILY:
        raise ValueError("Stage-4 known holdout family changed")
    winners = contract.get("winners", {})
    if {target: winners.get(target, {}).get("model_id") for target in PROPERTY_TARGETS} != FROZEN_WINNERS:
        raise ValueError("Stage-4 winners changed")
    for target, digest in FROZEN_BOARD_HASHES.items():
        winner = winners[target]
        budget = winner.get("budget", {})
        if winner.get("leaderboard_sha256") != digest or winner.get("rank") != 1:
            raise ValueError(f"{target} leaderboard lock changed")
        if budget.get("n_estimators") != 32 or budget.get("update_steps") != 32:
            raise ValueError(f"{target} estimator/update budget changed")
    return contract


def _verify_hash(path: Path, expected: str, role: str) -> str:
    actual = _hash_file(path)
    if actual != expected:
        raise RuntimeError(f"{role} SHA-256 changed")
    return actual


def _verify_prior_exposure(contract: Mapping[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for role, evidence in contract["prior_exposure_evidence"].items():
        path = PROJECT_ROOT / evidence["path"]
        verified[role] = {
            "sha256": _verify_hash(path, evidence["sha256"], role),
            "path": evidence["path"],
        }
    historical_split = json.loads(
        (PROJECT_ROOT / contract["prior_exposure_evidence"]["historical_split_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    historical_run = json.loads(
        (PROJECT_ROOT / contract["prior_exposure_evidence"]["historical_run_manifest"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    historical_metrics = json.loads(
        (PROJECT_ROOT / contract["prior_exposure_evidence"]["historical_metrics"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if historical_split.get("family_partition", {}).get(KNOWN_HOLDOUT_FAMILY) != "test":
        raise RuntimeError("historical split does not identify F-15 as test")
    if historical_run.get("families", {}).get("test") != [KNOWN_HOLDOUT_FAMILY]:
        raise RuntimeError("historical run does not record the F-15 test family")
    if not historical_run.get("test_loaded_after_best_checkpoint"):
        raise RuntimeError("historical test exposure evidence is absent")
    if set(historical_metrics.get("per_target", {})) != {"PHIF", "log1p(KLOGH)", "SW"}:
        raise RuntimeError("historical metrics do not expose all property targets")
    for role in ("p4_phif_lifecycle", "p4_klogh_lifecycle"):
        lifecycle = json.loads((PROJECT_ROOT / verified[role]["path"]).read_text(encoding="utf-8"))
        if lifecycle.get("state") != "VERIFIED" or not lifecycle.get("test_consumed_at"):
            raise RuntimeError(f"{role} does not prove prior test consumption")
        if lifecycle.get("evidence", {}).get("SPLIT_LOCKED", {}).get("test_family") != KNOWN_HOLDOUT_FAMILY:
            raise RuntimeError(f"{role} used a different holdout family")
    return {
        "status": "verified_before_fitting",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "known_holdout_family": KNOWN_HOLDOUT_FAMILY,
        "evidence": verified,
    }


def validate_stage3_and_contract(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = load_contract(contract_path)
    locks = contract["stage3_locks"]
    current_source_lock_sha256 = source_lock_sha256()
    historical_source_lock_sha256 = locks["source_lock_sha256"]
    locked_files = {
        "budget": HERE / "reservoir_p5_stage3_budget.json",
        "summary": STAGE3_DIR / "p5_stage3_summary.json",
        "split_manifest": STAGE3_DIR / "p5_stage3_split_manifest.json",
        "visualization_data": STAGE3_DIR / "p5_stage3_visualization_data.json",
        "p5_contract": HERE / "p5_contract.py",
    }
    for role, path in locked_files.items():
        _verify_hash(path, locks[f"{role}_sha256"], f"Stage-3 {role}")
    summary = json.loads(locked_files["summary"].read_text(encoding="utf-8"))
    split = json.loads(locked_files["split_manifest"].read_text(encoding="utf-8"))
    budget = json.loads(locked_files["budget"].read_text(encoding="utf-8"))
    if summary.get("expected_cells") != 108 or summary.get("counts", {}).get("completed") != 108:
        raise RuntimeError("Stage-3 is not a complete 108-cell confirmation")
    if any(summary.get("test_firewall", {}).values()):
        raise RuntimeError("Stage-3 test firewall is not clean")
    if split.get("split_hash") != locks["split_hash"]:
        raise RuntimeError("Stage-3 split identity changed")
    if split.get("development_groups") != list(DEVELOPMENT_FAMILIES):
        raise RuntimeError("Stage-3 development groups changed")
    firewall = split.get("test_firewall", {})
    if firewall.get("frozen_test_family") != KNOWN_HOLDOUT_FAMILY or firewall.get("test_access"):
        raise RuntimeError("Stage-3 split firewall changed")
    task_spec = build_task_spec()
    feature_contract = contract["feature_contract"]
    if tuple(task_spec.input_whitelist) != EXPECTED_FEATURE_WHITELIST:
        raise RuntimeError("P5 TaskSpec feature whitelist changed")
    if list(task_spec.input_whitelist) != feature_contract["input_whitelist"]:
        raise RuntimeError("Stage-4 feature whitelist differs from P5")
    if set(task_spec.input_whitelist) & set(task_spec.forbidden_inputs):
        raise RuntimeError("feature whitelist contains a forbidden interpreted target")
    winner_report: dict[str, Any] = {}
    for target in PROPERTY_TARGETS:
        board_path = STAGE3_DIR / f"leaderboard_{target.lower()}_tabular_cpu.json"
        board_hash = _verify_hash(
            board_path, FROZEN_BOARD_HASHES[target], f"{target} Stage-3 leaderboard"
        )
        board = json.loads(board_path.read_text(encoding="utf-8"))
        rank_one = [entry for entry in board.get("entries", []) if entry.get("rank") == 1]
        if (
            board.get("status") != "rankable"
            or board.get("lane") != "tabular_cpu"
            or len(rank_one) != 1
            or rank_one[0].get("model_id") != FROZEN_WINNERS[target]
        ):
            raise RuntimeError(f"{target} does not have the frozen unique winner")
        inherited = budget["model_budgets"][FROZEN_WINNERS[target]]
        frozen = contract["winners"][target]["budget"]
        for key in ("update_steps", "update_unit"):
            if frozen[key] != inherited[key]:
                raise RuntimeError(f"{target} Stage-3 {key} changed")
        if inherited["config"].get("n_estimators") != 32:
            raise RuntimeError(f"{target} Stage-3 estimator count changed")
        discovered = discover_model("property", FROZEN_WINNERS[target])
        if list(discovered.capabilities["input_modalities"]) != ["tabular"]:
            raise RuntimeError(f"{target} winner left the tabular lane")
        winner_report[target] = {
            "model_id": FROZEN_WINNERS[target],
            "leaderboard_sha256": board_hash,
            "mean_physical_RMSE": rank_one[0]["mean_physical_RMSE"],
            "budget": frozen,
        }
    phif_split_hash = _hash_file(P4_PHIF_SPLIT)
    klogh_split_hash = _hash_file(P4_KLOGH_SPLIT)
    fold_policy = budget["fold_policy"]
    if phif_split_hash != fold_policy["p4_phif_split_manifest_sha256"]:
        raise RuntimeError("P4 PHIF split identity changed")
    if klogh_split_hash != fold_policy["p4_klogh_split_manifest_sha256"]:
        raise RuntimeError("P4 KLOGH split identity changed")
    for path in (P4_PHIF_SPLIT, P4_KLOGH_SPLIT):
        source_split = json.loads(path.read_text(encoding="utf-8"))
        if source_split.get("development_groups") != list(DEVELOPMENT_FAMILIES):
            raise RuntimeError("P4 development family identity changed")
        if source_split.get("test_groups") != [KNOWN_HOLDOUT_FAMILY]:
            raise RuntimeError("P4 test family identity changed")
    return {
        "contract_sha256": _hash_file(contract_path),
        "stage3_split_hash": locks["split_hash"],
        "p4_split_sha256": {"PHIF": phif_split_hash, "KLOGH": klogh_split_hash},
        "feature_whitelist_verified": True,
        "forbidden_feature_overlap": [],
        "winners": winner_report,
        "source_lock": {
            "historical_source_lock_sha256": historical_source_lock_sha256,
            "current_source_lock_sha256": current_source_lock_sha256,
            "historical_source_lock_mismatch": current_source_lock_sha256 != historical_source_lock_sha256,
        },
        "prior_exposure": _verify_prior_exposure(contract),
    }


def _decoded_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return dict(json.loads(str(value)))


def _record_id(partition: str, key: str | int, meta: Mapping[str, Any]) -> str:
    raw = "|".join(
        [partition, str(key), str(meta["family_id"]), str(meta["well_id"]), f"{float(meta['depth_m']):.6f}"]
    )
    return f"stage4-{partition}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def _read_hdf5(path: Path, partition: str, allowed_families: set[str]) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Stage-4 prepare requires an interpreter with h5py") from exc
    records: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        for key in sorted(handle.keys()):
            group = handle[key]
            meta = _decoded_json(group.attrs["meta"])
            position = _decoded_json(group.attrs["position"])
            family = str(meta["family_id"])
            if family not in allowed_families:
                raise RuntimeError(f"{partition} contains unauthorized mother-well family {family!r}")
            records.append(
                {
                    "sample_id": _record_id(partition, key, meta),
                    "family_id": family,
                    "well_id": str(meta["well_id"]),
                    "depth_m": float(meta["depth_m"]),
                    "position": position,
                    "seismic": np.asarray(group["seismic_patch"][()], dtype=np.float64),
                    "logs": np.asarray(group["well_log_seq"][()], dtype=np.float64),
                    "labels": np.asarray(group["label"][()], dtype=np.float64).reshape(-1)[:3],
                }
            )
    return records


def _read_guard(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        required = {"seismic_patch", "well_log_seq", "label", "meta_json", "position_json"}
        if not required <= set(archive.files):
            raise ValueError(f"guard archive missing {sorted(required - set(archive.files))}")
        for index in range(len(archive["label"])):
            meta = _decoded_json(str(archive["meta_json"][index]))
            position = _decoded_json(str(archive["position_json"][index]))
            family = str(meta["family_id"])
            if family not in DEVELOPMENT_FAMILIES or family == KNOWN_HOLDOUT_FAMILY:
                raise RuntimeError(f"guard contains unauthorized mother-well family {family!r}")
            records.append(
                {
                    "sample_id": _record_id("guard", index, meta),
                    "family_id": family,
                    "well_id": str(meta["well_id"]),
                    "depth_m": float(meta["depth_m"]),
                    "position": position,
                    "seismic": np.asarray(archive["seismic_patch"][index], dtype=np.float64),
                    "logs": np.asarray(archive["well_log_seq"][index], dtype=np.float64),
                    "labels": np.asarray(archive["label"][index], dtype=np.float64).reshape(-1)[:3],
                }
            )
    return records


def _validate_records(records: Sequence[Mapping[str, Any]], families: set[str], role: str) -> None:
    if not records or {str(row["family_id"]) for row in records} != families:
        raise RuntimeError(f"{role} mother-well families changed")
    if len({str(row["sample_id"]) for row in records}) != len(records):
        raise RuntimeError(f"{role} sample IDs are not unique")
    seismic = np.stack([row["seismic"] for row in records])
    logs = np.stack([row["logs"] for row in records])
    labels = np.stack([row["labels"] for row in records])
    if seismic.shape[1:] != (3, 3, 9) or logs.shape[1:] != (9, 8) or labels.shape[1:] != (3,):
        raise RuntimeError(f"{role} real input/label shape changed")
    if not np.isfinite(seismic).all() or not np.isfinite(logs).all() or not np.isfinite(labels).all():
        raise RuntimeError(f"{role} contains non-finite arrays")
    if np.any(labels[:, TARGET_INDEX["KLOGH"]] < 0):
        raise RuntimeError(f"{role} contains invalid log1p(KLOGH) values")


def _write_raw_archive(path: Path, records: Sequence[Mapping[str, Any]], source: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        seismic_patch=np.stack([row["seismic"] for row in records]),
        well_log_sequence=np.stack([row["logs"] for row in records]),
        labels_model_domain=np.stack([row["labels"] for row in records]),
        target_masks=np.isfinite(np.stack([row["labels"] for row in records])).astype(np.uint8),
        sample_ids=np.asarray([row["sample_id"] for row in records]),
        family_ids=np.asarray([row["family_id"] for row in records]),
        well_ids=np.asarray([row["well_id"] for row in records]),
        depths_m=np.asarray([row["depth_m"] for row in records], dtype=np.float64),
        position_json=np.asarray([json.dumps(row["position"], sort_keys=True) for row in records]),
        source_manifest_json=np.asarray(json.dumps(source, sort_keys=True)),
    )


def prepare_stage4(
    train_h5: Path,
    test_h5: Path,
    guard_npz: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    state_path = output_dir / "confirmation_state.json"
    if state_path.exists():
        raise RuntimeError("Stage-4 confirmation state already exists; refuse to prepare again")
    contract = load_contract(contract_path)
    locks = validate_stage3_and_contract(contract_path)
    sources = {"train_h5": Path(train_h5), "test_h5": Path(test_h5), "guard_npz": Path(guard_npz)}
    for role, path in sources.items():
        if path.name != {"train_h5": "train.h5", "test_h5": "test.h5", "guard_npz": "guard.npz"}[role]:
            raise ValueError(f"Stage-4 requires an explicit {role} source with its canonical filename")
        _verify_hash(path, contract["source_data_sha256"][role], f"Stage-4 {role}")
    development = _read_hdf5(sources["train_h5"], "train", set(DEVELOPMENT_FAMILIES))
    development.extend(_read_guard(sources["guard_npz"]))
    holdout = _read_hdf5(sources["test_h5"], "known_holdout", {KNOWN_HOLDOUT_FAMILY})
    _validate_records(development, set(DEVELOPMENT_FAMILIES), "development")
    _validate_records(holdout, {KNOWN_HOLDOUT_FAMILY}, "known holdout")
    if len(development) != contract["expected_counts"]["development"]:
        raise RuntimeError("all-development row count changed")
    if len(holdout) != contract["expected_counts"]["known_holdout"]:
        raise RuntimeError("known-holdout row count changed")
    if {row["sample_id"] for row in development} & {row["sample_id"] for row in holdout}:
        raise RuntimeError("development and known-holdout sample IDs overlap")
    source_common = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 4,
        "source_data_sha256": contract["source_data_sha256"],
        "stage3_split_hash": locks["stage3_split_hash"],
        "prior_test_consumed": True,
        "fresh_blind": False,
        "source_paths_persisted": False,
    }
    development_path = output_dir / "runtime" / "development_all_rows.npz"
    holdout_path = output_dir / "runtime" / "known_holdout_f15.npz"
    _write_raw_archive(
        development_path,
        development,
        {**source_common, "role": "all_legal_development_rows", "model_access": True},
    )
    _write_raw_archive(
        holdout_path,
        holdout,
        {
            **source_common,
            "role": "previously_seen_reusable_holdout",
            "model_access": "only_after_single_use_state_transition",
        },
    )
    family_counts = {
        family: sum(row["family_id"] == family for row in development)
        for family in DEVELOPMENT_FAMILIES
    }
    report = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 4,
        "status": "prepared",
        "confirmation_kind": "previously_seen_reusable_holdout",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "known_holdout_family": KNOWN_HOLDOUT_FAMILY,
        "development_families": list(DEVELOPMENT_FAMILIES),
        "development_rows": len(development),
        "known_holdout_rows": len(holdout),
        "development_family_counts": family_counts,
        "independent_target_valid_counts": {
            "development": {
                target: int(np.isfinite(np.stack([row["labels"] for row in development])[:, index]).sum())
                for index, target in enumerate(PROPERTY_TARGETS)
            },
            "known_holdout": {
                target: int(np.isfinite(np.stack([row["labels"] for row in holdout])[:, index]).sum())
                for index, target in enumerate(PROPERTY_TARGETS)
            },
        },
        "runtime_archives": {
            "development_sha256": _hash_file(development_path),
            "known_holdout_sha256": _hash_file(holdout_path),
            "paths_persisted": False,
            "git_ignored": True,
        },
        "contract_sha256": locks["contract_sha256"],
        "stage3_evidence": locks,
        "source_data_sha256": contract["source_data_sha256"],
        "preparation_wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "preparation_manifest.json", report)
    return report


def _load_raw_archive(path: Path, expected_hash: str, expected_role: str) -> dict[str, Any]:
    _verify_hash(path, expected_hash, expected_role)
    with np.load(path, allow_pickle=False) as archive:
        source = json.loads(str(archive["source_manifest_json"]))
        result = {
            "seismic": np.asarray(archive["seismic_patch"], dtype=np.float64),
            "logs": np.asarray(archive["well_log_sequence"], dtype=np.float64),
            "labels": np.asarray(archive["labels_model_domain"], dtype=np.float64),
            "masks": np.asarray(archive["target_masks"], dtype=bool),
            "sample_ids": archive["sample_ids"].astype(str).tolist(),
            "families": archive["family_ids"].astype(str).tolist(),
            "wells": archive["well_ids"].astype(str).tolist(),
            "depths": np.asarray(archive["depths_m"], dtype=np.float64),
            "positions": [json.loads(value) for value in archive["position_json"].astype(str)],
            "source": source,
        }
    if source.get("role") != expected_role:
        raise RuntimeError(f"runtime archive role changed: {source.get('role')!r}")
    return result


def _stats_payload(stats: Mapping[str, np.ndarray], sample_ids: Sequence[str]) -> dict[str, Any]:
    arrays = {key: np.asarray(value).tolist() for key, value in stats.items()}
    return {
        "method": "Stage-2 tabular zscore with explicit observed-log masks",
        "fit_scope": "all legal development rows only",
        "fit_rows": len(sample_ids),
        "fit_sample_ids_sha256": _hash_payload(sorted(sample_ids)),
        "target_statistics_fitted": False,
        "denoise": "identity",
        "feature_count": 153,
        "statistics": arrays,
        "statistics_sha256": _hash_payload(arrays),
    }


def _make_batch(raw: Mapping[str, Any], transformed: tuple[np.ndarray, np.ndarray, np.ndarray], role: str) -> ModelBatch:
    seismic, logs, tabular = transformed
    return ModelBatch(
        inputs={"tabular": tabular, "seismic_patch": seismic, "well_log_sequence": logs},
        targets={target: raw["labels"][:, index] for index, target in enumerate(PROPERTY_TARGETS)},
        input_masks={"well_log_observed": logs[:, :, 4:8] > 0.5},
        target_masks={target: raw["masks"][:, index] for index, target in enumerate(PROPERTY_TARGETS)},
        sample_ids=list(raw["sample_ids"]),
        groups={"mother_well_family": list(raw["families"]), "well_id": list(raw["wells"])},
        coordinates={"depth_m": np.asarray(raw["depths"])},
        metadata={
            "stage": 4,
            "role": role,
            "prior_test_consumed": True,
            "fresh_blind": False,
            "preprocessing_fit": "all_legal_development_rows_only",
        },
    )


def _model_config(model_id: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    target = next(target for target, winner in FROZEN_WINNERS.items() if winner == model_id)
    budget = contract["winners"][target]["budget"]
    return {
        "seed": ROOT_SEED,
        "n_features": 153,
        "device": "cpu",
        "n_estimators": int(budget["n_estimators"]),
        "n_jobs": int(budget["n_jobs"]),
    }


def _checkpoint_hash(path: Path) -> str:
    return stage2._checkpoint_hash(path)


def _regression(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("regression vectors are empty or misaligned")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("regression vectors contain non-finite values")
    residual = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": None if denominator == 0 else float(1.0 - np.sum(residual**2) / denominator),
        "R2_reason": "constant truth" if denominator == 0 else None,
        "bias": float(np.mean(residual)),
    }


def _oof_interval_q90(target: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    path = STAGE3_DIR / "p5_stage3_visualization_data.json"
    _verify_hash(path, contract["stage3_locks"]["visualization_data_sha256"], "Stage-3 OOF visualization data")
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_data = payload.get("targets", {}).get(target, {})
    if target_data.get("winner_model_id") != FROZEN_WINNERS[target]:
        raise RuntimeError(f"{target} OOF interval source winner changed")
    samples = target_data.get("samples", [])
    residuals = np.asarray(
        [abs(float(row["prediction_model_domain_mean"]) - float(row["truth_model_domain"])) for row in samples],
        dtype=np.float64,
    )
    if len(residuals) != target_data.get("aggregated_sample_count") or not np.isfinite(residuals).all():
        raise RuntimeError(f"{target} OOF interval source is incomplete")
    return {
        "absolute_residual_q90_model_domain": float(np.quantile(residuals, 0.9)),
        "development_oof_rows": len(residuals),
        "source_sha256": _hash_file(path),
        "method": "symmetric Stage-3 development OOF mean-over-seeds absolute-residual q90",
        "nominal_coverage": 0.9,
    }


def _target_metrics(
    target: str,
    truth_model: np.ndarray,
    prediction_model: np.ndarray,
    interval_q90: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    truth_model = np.asarray(truth_model, dtype=np.float64)
    prediction_model = np.asarray(prediction_model, dtype=np.float64)
    truth_physical = model_to_physical(target, truth_model, prediction=False)
    prediction_physical = model_to_physical(target, prediction_model, prediction=True)
    low_model, high_model = prediction_model - interval_q90, prediction_model + interval_q90
    low_physical = model_to_physical(target, low_model, prediction=True)
    high_physical = model_to_physical(target, high_model, prediction=True)
    covered = (truth_physical >= low_physical) & (truth_physical <= high_physical)
    outside = (
        ((prediction_model < 0) | (prediction_model > 1))
        if target in {"PHIF", "SW"}
        else prediction_model < 0
    )
    metrics = {
        "target": target,
        "confirmation_kind": "previously_seen_reusable_holdout",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "known_holdout_family": KNOWN_HOLDOUT_FAMILY,
        "valid_count": int(len(truth_model)),
        "unit": {"PHIF": "fraction", "KLOGH": "mD", "SW": "fraction"}[target],
        "model_domain_name": "log1p(KLOGH_mD)" if target == "KLOGH" else "identity",
        "model_domain": _regression(truth_model, prediction_model),
        "physical": _regression(truth_physical, prediction_physical),
        "raw_out_of_physical_range_rate": float(np.mean(outside)),
        "interval": {
            "method": "symmetric Stage-3 development OOF absolute-residual q90 in model domain",
            "nominal_coverage": 0.9,
            "empirical_known_holdout_coverage": float(np.mean(covered)),
            "mean_physical_interval_width": float(np.mean(high_physical - low_physical)),
            "absolute_residual_q90_model_domain": float(interval_q90),
        },
    }
    arrays = {
        "truth_model": truth_model,
        "prediction_model": prediction_model,
        "truth_physical": truth_physical,
        "prediction_physical": prediction_physical,
        "low_physical": low_physical,
        "high_physical": high_physical,
        "covered": covered,
    }
    return metrics, arrays


def _write_predictions(path: Path, raw: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "family_id",
                "well_id",
                "depth_m",
                "truth_model_domain",
                "prediction_model_domain",
                "truth_physical",
                "prediction_physical",
                "interval_low_physical",
                "interval_high_physical",
                "interval_covered",
            ]
        )
        for index, sample_id in enumerate(raw["sample_ids"]):
            writer.writerow(
                [
                    sample_id,
                    raw["families"][index],
                    raw["wells"][index],
                    float(raw["depths"][index]),
                    float(arrays["truth_model"][index]),
                    float(arrays["prediction_model"][index]),
                    float(arrays["truth_physical"][index]),
                    float(arrays["prediction_physical"][index]),
                    float(arrays["low_physical"][index]),
                    float(arrays["high_physical"][index]),
                    bool(arrays["covered"][index]),
                ]
            )


def _font_choice() -> tuple[str, str | None]:
    names = {font.name for font in font_manager.fontManager.ttflist}
    if "Times New Roman" in names:
        return "Times New Roman", None
    return "Liberation Serif", "Times New Roman is unavailable; used metrically compatible Liberation Serif"


def normalize_fonts(figure: Any, family: str) -> None:
    for axis in figure.axes:
        axis.xaxis.label.set_fontsize(8)
        axis.yaxis.label.set_fontsize(8)
        for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            label.set_fontsize(7)
            label.set_fontfamily(family)
        for text in axis.texts:
            text.set_fontsize(max(float(text.get_fontsize()), 7.0))
            text.set_fontfamily(family)
        for text in (axis.xaxis.label, axis.yaxis.label):
            text.set_fontfamily(family)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(7)
                text.set_fontfamily(family)


def _save_figure(figure: Any, path: Path, family: str) -> None:
    normalize_fonts(figure, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_figure_note(
    path: Path,
    target: str,
    figure_kind: str,
    predictions_hash: str,
    metrics: Mapping[str, Any],
) -> None:
    note = "\n".join(
        [
            f"# {target} Stage-4 {figure_kind}",
            "",
            "- Evidence status: `previously_seen_reusable_holdout`",
            "- Prior test consumed: `true`",
            "- Fresh blind: `false`",
            f"- Known holdout: `{KNOWN_HOLDOUT_FAMILY}`",
            f"- Rows: `{metrics['valid_count']}`",
            f"- Unit: `{metrics['unit']}`",
            f"- Prediction source SHA-256: `{predictions_hash}`",
            "",
            "The panel is generated only from the target-specific Stage-4 prediction CSV. "
            "It is a known-holdout confirmation and must not be described as a fresh-blind result.",
            "",
        ]
    )
    _atomic_text(path, note)


def _write_target_figures(
    target: str,
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    predictions_hash: str,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], str, str | None]:
    family, fallback = _font_choice()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family],
            "axes.linewidth": 0.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
        }
    )
    unit = metrics["unit"]
    truth, prediction = arrays["truth_physical"], arrays["prediction_physical"]
    figures: list[dict[str, Any]] = []

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.scatter(truth, prediction, s=10, alpha=0.55, color=PALETTE["blue"], edgecolors="none")
    lower = float(min(truth.min(), prediction.min()))
    upper = float(max(truth.max(), prediction.max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#333333")
    axis.set_xlabel(f"Truth ({unit})")
    axis.set_ylabel(f"Prediction ({unit})")
    axis.grid(alpha=0.18)
    name = f"{target.lower()}_predicted_vs_true.png"
    _save_figure(figure, output_dir / name, family)
    _write_figure_note(output_dir / name.replace(".png", ".md"), target, "predicted-vs-true", predictions_hash, metrics)
    figures.append({"kind": "predicted_vs_true", "path": name})

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    residual = prediction - truth
    axis.scatter(prediction, residual, s=10, alpha=0.55, color=PALETTE["red"], edgecolors="none")
    axis.axhline(0.0, linestyle="--", color="#333333")
    axis.set_xlabel(f"Prediction ({unit})")
    axis.set_ylabel(f"Residual ({unit})")
    axis.grid(alpha=0.18)
    name = f"{target.lower()}_residual.png"
    _save_figure(figure, output_dir / name, family)
    _write_figure_note(output_dir / name.replace(".png", ".md"), target, "residual", predictions_hash, metrics)
    figures.append({"kind": "residual", "path": name})

    order = np.argsort(truth)
    x = np.arange(len(order))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.fill_between(
        x,
        arrays["low_physical"][order],
        arrays["high_physical"][order],
        color=PALETTE["sky"],
        alpha=0.32,
        linewidth=0,
        label="OOF q90 interval",
    )
    axis.plot(x, truth[order], color=PALETTE["blue"], label="Truth")
    axis.plot(x, prediction[order], color=PALETTE["red"], alpha=0.85, label="Prediction")
    axis.set_xlabel("Known-holdout samples sorted by truth")
    axis.set_ylabel(f"Property ({unit})")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False)
    name = f"{target.lower()}_interval_diagnostic.png"
    _save_figure(figure, output_dir / name, family)
    _write_figure_note(output_dir / name.replace(".png", ".md"), target, "interval-diagnostic", predictions_hash, metrics)
    figures.append({"kind": "interval_diagnostic", "path": name})
    return figures, family, fallback


def _write_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or "runtime" in path.relative_to(output_dir).parts or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(output_dir).as_posix()
        artifacts[relative] = {"sha256": _hash_file(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 4,
        "confirmation_kind": "previously_seen_reusable_holdout",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "artifacts": artifacts,
        "runtime_artifacts_git_ignored": True,
        "absolute_paths_persisted": False,
    }
    _atomic_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def confirm_stage4(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    state_path = output_dir / "confirmation_state.json"
    if state_path.exists():
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Stage-4 confirmation is single-use; existing state={existing.get('state')!r}"
        )
    preparation_path = output_dir / "preparation_manifest.json"
    if not preparation_path.is_file():
        raise FileNotFoundError("Stage-4 preparation manifest is absent")
    contract = load_contract(contract_path)
    locks = validate_stage3_and_contract(contract_path)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    if preparation.get("prior_test_consumed") is not True or preparation.get("fresh_blind") is not False:
        raise RuntimeError("preparation lost the known-holdout disclosure")
    if preparation.get("contract_sha256") != locks["contract_sha256"]:
        raise RuntimeError("preparation contract hash changed")
    development_path = output_dir / "runtime" / "development_all_rows.npz"
    holdout_path = output_dir / "runtime" / "known_holdout_f15.npz"
    development = _load_raw_archive(
        development_path,
        preparation["runtime_archives"]["development_sha256"],
        "all_legal_development_rows",
    )
    if set(development["families"]) != set(DEVELOPMENT_FAMILIES):
        raise RuntimeError("refit development families changed")
    if KNOWN_HOLDOUT_FAMILY in development["families"]:
        raise RuntimeError("known holdout reached refit")
    stats = stage2._fit_stats(development["seismic"], development["logs"])
    development_transformed = stage2._transform(development["seismic"], development["logs"], stats)
    development_batch = _make_batch(development, development_transformed, "all_legal_development_rows")
    if development_batch.inputs["tabular"].shape != (1216, 153):
        raise RuntimeError("all-development tabular matrix changed")
    preprocessing = _stats_payload(stats, development_batch.sample_ids)
    _atomic_json(output_dir / "refit" / "preprocessing.json", preprocessing)
    task_spec = build_task_spec()
    fitted_models: dict[str, Any] = {}
    refit_models: dict[str, Any] = {}
    for model_id in sorted(set(FROZEN_WINNERS.values())):
        config = _model_config(model_id, contract)
        discovered = discover_model("property", model_id)
        model = discovered.build(task_spec, **config)
        fit_started = time.perf_counter()
        fit_report = model.fit(development_batch)
        fit_seconds = time.perf_counter() - fit_started
        checkpoint = output_dir / "runtime" / "checkpoints" / f"{model_id}.bin"
        model.save_checkpoint(checkpoint)
        restored = discovered.build(task_spec, **config)
        restored.load_checkpoint(checkpoint)
        original = model.predict(development_batch)
        replay = restored.predict(development_batch)
        for target in PROPERTY_TARGETS:
            if not np.allclose(original.raw[target], replay.raw[target], rtol=1e-7, atol=1e-8):
                raise RuntimeError(f"{model_id} development checkpoint replay changed {target}")
        fitted_models[model_id] = restored
        refit_models[model_id] = {
            "config": config,
            "config_sha256": _hash_payload(config),
            "checkpoint_sha256": _checkpoint_hash(checkpoint),
            "checkpoint_path_persisted": False,
            "checkpoint_git_ignored": True,
            "checkpoint_roundtrip": True,
            "fit_wall_seconds": fit_seconds,
            "fit_report": fit_report,
            "development_rows": len(development_batch.sample_ids),
            "development_family_counts": {
                family: development["families"].count(family) for family in DEVELOPMENT_FAMILIES
            },
            "known_holdout_rows_used_for_fit": 0,
            "seed": ROOT_SEED,
        }
    refit_hashes = {model_id: report["checkpoint_sha256"] for model_id, report in refit_models.items()}
    state = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 4,
        "state": "REFIT_COMPLETE",
        "confirmation_kind": "previously_seen_reusable_holdout",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "known_holdout_accessed_by_stage4_model": False,
        "contract_sha256": locks["contract_sha256"],
        "stage3_split_hash": locks["stage3_split_hash"],
        "refit_checkpoint_sha256": refit_hashes,
        "refit_completed_at": _utc_now(),
    }
    _atomic_json(state_path, state)
    state.update(
        {
            "state": "CONFIRMATION_CONSUMED",
            "known_holdout_accessed_by_stage4_model": True,
            "confirmation_consumed_at": _utc_now(),
        }
    )
    _atomic_json(state_path, state)

    holdout = _load_raw_archive(
        holdout_path,
        preparation["runtime_archives"]["known_holdout_sha256"],
        "previously_seen_reusable_holdout",
    )
    if set(holdout["families"]) != {KNOWN_HOLDOUT_FAMILY}:
        raise RuntimeError("known-holdout family changed after state transition")
    holdout_transformed = stage2._transform(holdout["seismic"], holdout["logs"], stats)
    holdout_batch = _make_batch(holdout, holdout_transformed, "previously_seen_reusable_holdout")
    model_outputs = {model_id: model.predict(holdout_batch) for model_id, model in fitted_models.items()}
    target_summaries: dict[str, Any] = {}
    figure_manifest: list[dict[str, Any]] = []
    resolved_font, font_fallback = _font_choice()
    for target in PROPERTY_TARGETS:
        model_id = FROZEN_WINNERS[target]
        mask = np.asarray(holdout_batch.target_masks[target], dtype=bool)
        truth_model = np.asarray(holdout_batch.targets[target], dtype=np.float64)[mask]
        prediction_model = np.asarray(model_outputs[model_id].raw[target], dtype=np.float64)[mask]
        interval_source = _oof_interval_q90(target, contract)
        metrics, arrays = _target_metrics(
            target,
            truth_model,
            prediction_model,
            interval_source["absolute_residual_q90_model_domain"],
        )
        metrics["model_id"] = model_id
        metrics["seed"] = ROOT_SEED
        metrics["stage3_leaderboard_sha256"] = FROZEN_BOARD_HASHES[target]
        metrics["interval_source"] = interval_source
        target_dir = output_dir / target.lower()
        config_payload = {
            "target": target,
            "model_id": model_id,
            "seed": ROOT_SEED,
            "budget": contract["winners"][target]["budget"],
            "model_config": refit_models[model_id]["config"],
            "model_domain": contract["winners"][target]["model_domain"],
            "prediction_transform": contract["winners"][target]["prediction_transform"],
            "feature_whitelist": contract["feature_contract"]["input_whitelist"],
            "winner_reselection": False,
            "hpo": False,
            "confirmation_kind": "previously_seen_reusable_holdout",
        }
        refit_payload = {
            "target": target,
            "selected_head_from_independent_multi-target_adapter": True,
            "adapter_targets_fit_independently": list(PROPERTY_TARGETS),
            "model_id": model_id,
            **refit_models[model_id],
            "target_valid_development_rows": int(development_batch.target_masks[target].sum()),
        }
        _atomic_json(target_dir / "config.json", config_payload)
        _atomic_json(target_dir / "refit.json", refit_payload)
        _atomic_json(target_dir / "metrics.json", metrics)
        predictions_path = target_dir / "predictions.csv"
        masked_holdout = {
            key: [value[index] for index in np.flatnonzero(mask)]
            if key in {"sample_ids", "families", "wells"}
            else np.asarray(value)[mask]
            if key == "depths"
            else value
            for key, value in holdout.items()
        }
        _write_predictions(predictions_path, masked_holdout, arrays)
        predictions_hash = _hash_file(predictions_path)
        figures, _font, _fallback = _write_target_figures(
            target,
            arrays,
            metrics,
            predictions_hash,
            target_dir / "figures",
        )
        for figure in figures:
            figure_manifest.append(
                {
                    "target": target,
                    "kind": figure["kind"],
                    "path": f"{target.lower()}/figures/{figure['path']}",
                    "sha256": _hash_file(target_dir / "figures" / figure["path"]),
                    "companion_md_sha256": _hash_file(
                        target_dir / "figures" / figure["path"].replace(".png", ".md")
                    ),
                }
            )
        target_summaries[target] = {
            "model_id": model_id,
            "development_rows": len(development_batch.sample_ids),
            "known_holdout_rows": metrics["valid_count"],
            "physical": metrics["physical"],
            "model_domain": metrics["model_domain"],
            "interval": metrics["interval"],
            "predictions_sha256": predictions_hash,
            "metrics_sha256": _hash_file(target_dir / "metrics.json"),
            "config_sha256": _hash_file(target_dir / "config.json"),
            "refit_sha256": _hash_file(target_dir / "refit.json"),
        }
    visualization_manifest = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 4,
        "confirmation_kind": "previously_seen_reusable_holdout",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "figure_mode": "A_square",
        "figure_size_inches": list(FIGURE_SIZE),
        "dpi": 300,
        "palette": PALETTE,
        "requested_font": "Times New Roman",
        "resolved_font": resolved_font,
        "font_fallback_reason": font_fallback,
        "figures": figure_manifest,
        "rebuild_command": "python3 _pipelines/02_task_datasets/reservoir/reservoir_p5_stage4.py render",
    }
    _atomic_json(output_dir / "visualization_manifest.json", visualization_manifest)
    state.update(
        {
            "state": "CONFIRMED",
            "confirmed_at": _utc_now(),
            "targets": {
                target: {
                    "model_id": target_summaries[target]["model_id"],
                    "metrics_sha256": target_summaries[target]["metrics_sha256"],
                    "predictions_sha256": target_summaries[target]["predictions_sha256"],
                }
                for target in PROPERTY_TARGETS
            },
        }
    )
    _atomic_json(state_path, state)
    summary = {
        "schema_version": 1,
        "track_id": "property",
        "stage": 4,
        "status": "confirmed",
        "confirmation_kind": "previously_seen_reusable_holdout",
        "prior_test_consumed": True,
        "fresh_blind": False,
        "known_holdout_family": KNOWN_HOLDOUT_FAMILY,
        "root_seed": ROOT_SEED,
        "winners_frozen_before_refit": True,
        "winner_reselection": False,
        "hpo": False,
        "development_rows": len(development_batch.sample_ids),
        "known_holdout_rows": len(holdout_batch.sample_ids),
        "development_families": list(DEVELOPMENT_FAMILIES),
        "target_summaries": target_summaries,
        "source_hashes": {
            "contract_sha256": locks["contract_sha256"],
            "stage3_split_hash": locks["stage3_split_hash"],
            "source_data_sha256": contract["source_data_sha256"],
            "preparation_manifest_sha256": _hash_file(preparation_path),
            "preprocessing_sha256": _hash_file(output_dir / "refit" / "preprocessing.json"),
            "source_lock_sha256": source_lock_sha256(),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": metadata.version("numpy"),
            "scikit_learn": metadata.version("scikit-learn"),
            "xgboost": metadata.version("xgboost"),
        },
        "timings": {
            "preparation_wall_seconds": preparation["preparation_wall_seconds"],
            "confirmation_wall_seconds": time.perf_counter() - started,
            "model_refit_wall_seconds": {
                model_id: report["fit_wall_seconds"] for model_id, report in refit_models.items()
            },
        },
        "single_use_state": "CONFIRMED",
        "portable": {
            "absolute_paths_persisted": False,
            "runtime_archives_persisted": False,
            "checkpoints_git_ignored": True,
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    manifest = _write_artifact_manifest(output_dir)
    return {
        **summary,
        "artifact_manifest_sha256": _hash_file(output_dir / "artifact_manifest.json"),
        "portable_artifact_count": len(manifest["artifacts"]),
    }


def render_existing(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    figures: list[dict[str, Any]] = []
    resolved_font, fallback = _font_choice()
    for target in PROPERTY_TARGETS:
        target_dir = output_dir / target.lower()
        metrics = json.loads((target_dir / "metrics.json").read_text(encoding="utf-8"))
        rows: list[dict[str, str]] = []
        with (target_dir / "predictions.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        arrays = {
            "truth_physical": np.asarray([float(row["truth_physical"]) for row in rows]),
            "prediction_physical": np.asarray([float(row["prediction_physical"]) for row in rows]),
            "low_physical": np.asarray([float(row["interval_low_physical"]) for row in rows]),
            "high_physical": np.asarray([float(row["interval_high_physical"]) for row in rows]),
            "covered": np.asarray([row["interval_covered"] == "True" for row in rows]),
        }
        target_figures, _font, _fallback = _write_target_figures(
            target,
            arrays,
            metrics,
            _hash_file(target_dir / "predictions.csv"),
            target_dir / "figures",
        )
        for figure in target_figures:
            figures.append(
                {
                    "target": target,
                    "kind": figure["kind"],
                    "path": f"{target.lower()}/figures/{figure['path']}",
                    "sha256": _hash_file(target_dir / "figures" / figure["path"]),
                    "companion_md_sha256": _hash_file(
                        target_dir / "figures" / figure["path"].replace(".png", ".md")
                    ),
                }
            )
    manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    manifest.update({"resolved_font": resolved_font, "font_fallback_reason": fallback, "figures": figures})
    _atomic_json(output_dir / "visualization_manifest.json", manifest)
    _write_artifact_manifest(output_dir)
    return manifest


def audit_existing(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = Path(output_dir)
    state = json.loads((output_dir / "confirmation_state.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    if state.get("state") != "CONFIRMED" or not state.get("known_holdout_accessed_by_stage4_model"):
        raise RuntimeError("Stage-4 state is not CONFIRMED")
    if summary.get("confirmation_kind") != "previously_seen_reusable_holdout":
        raise RuntimeError("Stage-4 summary lost the known-holdout label")
    if summary.get("fresh_blind") is not False or summary.get("prior_test_consumed") is not True:
        raise RuntimeError("Stage-4 summary disclosure changed")
    for relative, evidence in manifest.get("artifacts", {}).items():
        path = output_dir / relative
        if _hash_file(path) != evidence["sha256"] or path.stat().st_size != evidence["bytes"]:
            raise RuntimeError(f"Stage-4 artifact changed: {relative}")
    if any(str(output_dir.resolve()) in path.read_text(encoding="utf-8") for path in output_dir.rglob("*.json")):
        raise RuntimeError("Stage-4 JSON contains an absolute output path")
    return {
        "status": "verified",
        "state": state["state"],
        "artifact_count": len(manifest["artifacts"]),
        "artifact_manifest_sha256": _hash_file(output_dir / "artifact_manifest.json"),
        "confirmation_kind": summary["confirmation_kind"],
        "prior_test_consumed": summary["prior_test_consumed"],
        "fresh_blind": summary["fresh_blind"],
        "target_summaries": summary["target_summaries"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="materialize verified development/known-holdout runtime arrays")
    prepare.add_argument("--train-h5", type=Path, required=True)
    prepare.add_argument("--test-h5", type=Path, required=True)
    prepare.add_argument("--guard-npz", type=Path, required=True)
    commands.add_parser("confirm", help="single-use all-development refit and known-holdout confirmation")
    commands.add_parser("render", help="rebuild figures from target prediction CSV files")
    commands.add_parser("audit", help="verify the completed portable confirmation artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        report = prepare_stage4(args.train_h5, args.test_h5, args.guard_npz)
    elif args.command == "confirm":
        report = confirm_stage4()
    elif args.command == "render":
        report = render_existing()
    else:
        report = audit_existing()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
