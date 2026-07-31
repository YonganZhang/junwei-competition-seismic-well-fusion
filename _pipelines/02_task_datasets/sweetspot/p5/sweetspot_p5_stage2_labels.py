"""Audited P4-to-P5 label mapping for sweetspot Stage-2 development pilots.

This module approves only a *pilot use* of already-versioned P4 labels.  It
does not construct labels, does not upgrade proxies to field truth, and never
opens frozen-test predictions, metrics, checkpoints, or HDF5 files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_MAPPING_PATH = HERE / "sweetspot_p5_label_mapping.v1.json"
TARGET_ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
APPROVED_TARGETS = ("T1", "T2", "T3", "T4", "T6", "T7")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _forbidden_provenance_path(relative: str) -> bool:
    path = Path(relative)
    forbidden_names = {
        "status.json", "metrics.json", "predictions.csv", "test.h5", "test.hdf5",
        "test.npz", "checkpoint_best.pkl",
    }
    if path.name.lower() in forbidden_names:
        return True
    return any(part.lower() == "frozen_test" or part.lower().startswith("frozen_test") for part in path.parts)


def _resolve_repo_file(relative: str, project_root: Path) -> Path:
    if Path(relative).is_absolute():
        raise ValueError(f"mapping provenance path must be repository-relative: {relative}")
    if _forbidden_provenance_path(relative):
        raise PermissionError(f"mapping may not cite frozen-test/history artifacts: {relative}")
    root = Path(project_root).resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"mapping provenance escapes project root: {relative}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_reference(reference: Mapping[str, Any], project_root: Path) -> Path:
    if set(reference) != {"path", "sha256"}:
        raise ValueError("each provenance reference must contain exactly path and sha256")
    path = _resolve_repo_file(str(reference["path"]), project_root)
    actual = sha256_file(path)
    if actual != reference["sha256"]:
        raise ValueError(f"P4 provenance hash changed for {reference['path']}: {actual}")
    return path


@dataclass(frozen=True)
class LabelMappingAudit:
    mapping_path: Path
    mapping_sha256: str
    payload: Mapping[str, Any]
    registry: Mapping[str, Any]

    def target(self, target_id: str) -> Mapping[str, Any]:
        return self.payload["targets"][target_id]

    def split_manifest(self, target_id: str) -> Mapping[str, Any]:
        target = self.target(target_id)
        path = _resolve_repo_file(str(target["split_manifest"]["path"]), PROJECT_ROOT)
        return json.loads(path.read_text(encoding="utf-8"))


def validate_label_mapping(
    mapping_path: Path = DEFAULT_MAPPING_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
) -> LabelMappingAudit:
    """Validate the versioned mapping and every non-test P4 provenance hash."""
    path = Path(mapping_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sweetspot-p5-label-mapping/v1":
        raise ValueError("unsupported sweetspot P5 label-mapping schema")
    if payload.get("root_seed") != 2693:
        raise ValueError("label mapping must freeze root_seed=2693")
    targets = payload.get("targets")
    if not isinstance(targets, Mapping) or tuple(targets) != TARGET_ORDER:
        raise ValueError("label mapping must contain T1..T7 in frozen order")
    policy = payload.get("policy", {})
    if tuple(policy.get("approved_target_ids", ())) != APPROVED_TARGETS:
        raise ValueError("only T1-T4, T6 and T7 may be approved for Stage-2 pilot")
    if policy.get("not_feasible_target_ids") != ["T5"]:
        raise ValueError("T5 must remain the sole not-feasible target")
    if policy.get("frozen_test_access") != "forbidden":
        raise ValueError("frozen-test access must remain forbidden")
    if policy.get("historical_test_metrics_for_selection") != "forbidden":
        raise ValueError("historical test metrics must remain forbidden")
    if policy.get("label_generation") != "forbidden":
        raise ValueError("Stage-2 mapping may not authorize label generation")

    registry_path = _verify_reference(payload["registry"], project_root)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_number = {f"T{entry['target_number']}": entry for entry in registry["targets"]}
    if set(registry_by_number) != set(TARGET_ORDER):
        raise ValueError("P4 registry must provide seven independent targets")

    for target_id in TARGET_ORDER:
        target = targets[target_id]
        registry_entry = registry_by_number[target_id]
        if target["p4_status"] != registry_entry["status"]:
            raise ValueError(f"{target_id}: mapping status disagrees with P4 registry")
        if target_id == "T5":
            if target.get("status") != "not_feasible" or target.get("development_rebuild") != "forbidden":
                raise ValueError("T5 must be fail-closed not_feasible")
            _verify_reference(target["not_feasible_evidence"], project_root)
            continue
        if target.get("status") != "approved_for_development_pilot":
            raise ValueError(f"{target_id}: approved target lacks pilot-only approval")
        if target.get("label_version") != registry_entry.get("label_version"):
            raise ValueError(f"{target_id}: label version disagrees with P4 registry")
        task_path = _verify_reference(target["task_spec"], project_root)
        split_path = _verify_reference(target["split_manifest"], project_root)
        _verify_reference(target["label_evidence"], project_root)
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        if task_payload.get("label_version") != target["label_version"]:
            raise ValueError(f"{target_id}: task-spec label version changed")
        if task_payload.get("task_type") != target["task_type"]:
            raise ValueError(f"{target_id}: task type changed")
        if task_payload.get("targets") != [target["target_name"]]:
            raise ValueError(f"{target_id}: P4 target/head changed")
        development_groups = set(split_payload.get("development_groups", ()))
        development_ids = set(split_payload.get("development_sample_ids", ()))
        folds = split_payload.get("folds", ())
        if not development_groups or not development_ids or not folds:
            raise ValueError(f"{target_id}: P4 development split is incomplete")
        first = folds[0]
        train_groups = set(first.get("train_groups", ()))
        validation_groups = set(first.get("validation_groups", ()))
        train_ids = set(first.get("train_sample_ids", ()))
        validation_ids = set(first.get("validation_sample_ids", ()))
        if train_groups & validation_groups or not (train_groups | validation_groups) <= development_groups:
            raise ValueError(f"{target_id}: fold_0 group isolation failed")
        if train_ids & validation_ids or not (train_ids | validation_ids) <= development_ids:
            raise ValueError(f"{target_id}: fold_0 development IDs are invalid")
        if target.get("is_proxy") and not str(target.get("proxy_semantics", "")).strip():
            raise ValueError(f"{target_id}: proxy semantics must remain explicit")

    if targets["T6"]["label_version"] == targets["T7"]["label_version"]:
        raise ValueError("T6 porosity and T7 permeability must remain independent")
    if targets["T6"]["target_name"] == targets["T7"]["target_name"]:
        raise ValueError("T6 and T7 may not share a head")
    return LabelMappingAudit(path, sha256_file(path), payload, registry)


def build_pilot_task_spec(audit: LabelMappingAudit, target_id: str) -> TaskSpec:
    """Build one independent P5 estimator/head from the audited P4 TaskSpec."""
    target = audit.target(target_id)
    if target.get("status") != "approved_for_development_pilot":
        raise PermissionError(f"{target_id} is not approved for a development pilot")
    task_path = _resolve_repo_file(str(target["task_spec"]["path"]), PROJECT_ROOT)
    p4 = TaskSpec.from_dict(json.loads(task_path.read_text(encoding="utf-8")))
    metadata = dict(p4.metadata)
    metadata.update({
        "single_target_head": True,
        "class_count": 2 if p4.task_type == "binary" else 0,
        "p5_stage": "fixed_development_pilot",
        "p5_label_mapping_sha256": audit.mapping_sha256,
        "p4_status": target["p4_status"],
        "is_proxy": target["is_proxy"],
        "proxy_semantics": target["proxy_semantics"],
        "label_generated_by_p5": False,
        "test_access": "forbidden",
    })
    return replace(
        p4,
        task_id=f"sweetspot.p5.stage2.{target_id.lower()}.{target['slug']}",
        hpo={
            "stage": "P5 Stage-2 fixed development pilot",
            "root_seed": 2693,
            "selection_scope": "P4 fold_0 development validation only",
            "test_access": "forbidden",
        },
        metadata=metadata,
    )


def portable_mapping_payload(audit: LabelMappingAudit) -> dict[str, Any]:
    """Return the tracked mapping with its content hash; paths remain relative."""
    return {
        "schema_version": "sweetspot-p5-label-mapping-audit/v1",
        "mapping_path": audit.mapping_path.relative_to(PROJECT_ROOT).as_posix(),
        "mapping_sha256": audit.mapping_sha256,
        "mapping": audit.payload,
        "test_accessed": False,
        "historical_test_metrics_used": False,
        "labels_generated": False,
    }
