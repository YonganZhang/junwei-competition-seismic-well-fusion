"""Read-only approval gate for target-specific sweetspot label contracts.

This module intentionally does not import ``build_dataset`` or ``dataset_io``.
It validates approval metadata and the fields already recorded by the audited
inventory.  A failed gate is a structured skip, never an invitation to invent a
replacement label.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


HERE = Path(__file__).resolve().parent
TRACK_DIR = HERE.parent
DEFAULT_INVENTORY = TRACK_DIR / "audit" / "data_availability.json"
UNRESOLVED_MARKERS = ("<required", "todo", "tbd", "待定", "待批准")
TOP_LEVEL_FIELDS = {
    "schema_version", "spec_version", "status", "target_semantics", "output",
    "allowed_source_fields", "label_construction", "time_window", "spatial_scale",
    "class_rules", "split_strategy", "inference_allowed_inputs", "metrics",
    "approval", "notes",
}
REQUIRED_NESTED = {
    "output": {"type", "classes", "units", "probability_interpretation"},
    "label_construction": {
        "formula", "formula_field_refs", "thresholds", "thresholds_not_applicable_reason",
        "weights", "weights_not_applicable_reason", "fit_domain",
    },
    "time_window": {"definition", "start", "end", "timezone", "leakage_cutoff"},
    "spatial_scale": {
        "support", "coordinate_system", "vertical_domain", "resolution", "alignment_tolerance",
    },
    "class_rules": {"positive", "negative", "unlabeled"},
    "split_strategy": {
        "strategy", "group_key", "train_rule", "validation_rule", "test_rule",
        "fit_statistics_scope", "leakage_guards",
    },
    "approval": {"approved", "approved_by", "approved_role", "approved_at", "decision_record"},
}


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_unresolved(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or any(marker in text for marker in UNRESOLVED_MARKERS)


def _collect_unresolved(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.extend(_collect_unresolved(child, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_collect_unresolved(child, f"{prefix}.{index}"))
    elif isinstance(value, str) and _is_unresolved(value):
        result.append(prefix)
    return result


@dataclass(frozen=True)
class LabelGateResult:
    target_id: str
    approved: bool
    status: str
    reason_codes: tuple[str, ...]
    errors: tuple[str, ...]
    spec_path: str | None
    spec_sha256: str | None
    spec: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "approved": self.approved,
            "status": self.status,
            "reason_codes": list(self.reason_codes),
            "errors": list(self.errors),
            "spec_path": self.spec_path,
            "spec_sha256": self.spec_sha256,
        }


def _field_pair(entry: Mapping[str, Any]) -> tuple[str, str]:
    return str(entry.get("source", "")), str(entry.get("field", ""))


def _validate_spec(spec: Mapping[str, Any], inventory: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    reasons: list[str] = []
    missing = sorted(TOP_LEVEL_FIELDS - set(spec))
    unknown = sorted(set(spec) - TOP_LEVEL_FIELDS)
    if missing:
        errors.append(f"missing top-level fields: {missing}")
        reasons.append("incomplete_contract")
    if unknown:
        errors.append(f"unknown top-level fields: {unknown}")
        reasons.append("unknown_contract_field")
    if spec.get("schema_version") != "sweetspot-label-spec/v1":
        errors.append("schema_version must be sweetspot-label-spec/v1")
        reasons.append("schema_mismatch")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", str(spec.get("spec_version", ""))):
        errors.append("spec_version is not a valid semantic version")
        reasons.append("schema_mismatch")
    if spec.get("status") != "approved":
        errors.append("status must be approved")
        reasons.append("label_not_approved")
    if spec.get("target_semantics") not in {"geological", "engineering", "production", "joint"}:
        errors.append("target_semantics is outside the contract enum")
        reasons.append("schema_mismatch")

    for section, expected in REQUIRED_NESTED.items():
        value = spec.get(section)
        if not isinstance(value, Mapping):
            errors.append(f"{section} must be an object")
            reasons.append("incomplete_contract")
            continue
        nested_missing = sorted(expected - set(value))
        nested_unknown = sorted(set(value) - expected)
        if nested_missing:
            errors.append(f"{section} missing fields: {nested_missing}")
            reasons.append("incomplete_contract")
        if nested_unknown:
            errors.append(f"{section} has unknown fields: {nested_unknown}")
            reasons.append("unknown_contract_field")

    approval = spec.get("approval") if isinstance(spec.get("approval"), Mapping) else {}
    if approval.get("approved") is not True:
        errors.append("approval.approved must be true")
        reasons.append("label_not_approved")
    for field in ("approved_by", "approved_role", "approved_at", "decision_record"):
        if _is_unresolved(approval.get(field)):
            errors.append(f"approval.{field} is unresolved")
            reasons.append("approval_metadata_incomplete")

    unresolved = _collect_unresolved(spec)
    if unresolved:
        errors.append("contract contains unresolved placeholders: " + ", ".join(unresolved))
        reasons.append("unresolved_placeholder")

    construction = spec.get("label_construction") if isinstance(spec.get("label_construction"), Mapping) else {}
    fit_domain = construction.get("fit_domain") if isinstance(construction.get("fit_domain"), Mapping) else {}
    if fit_domain.get("uses_test_statistics") is not False or fit_domain.get("statistics_scope") == "test":
        errors.append("label construction must not use test statistics")
        reasons.append("test_statistics_forbidden")
    if re.search(
        r"test[_ -]?(mean|std|quantile|median|stat)|测试集.*统计",
        str(construction.get("formula", "")), re.I,
    ):
        errors.append("label formula appears to reference test statistics")
        reasons.append("test_statistics_forbidden")
    if not construction.get("thresholds") and _is_unresolved(construction.get("thresholds_not_applicable_reason")):
        errors.append("empty thresholds require a non-placeholder reason")
        reasons.append("threshold_definition_incomplete")
    if not construction.get("weights") and _is_unresolved(construction.get("weights_not_applicable_reason")):
        errors.append("empty weights require a non-placeholder reason")
        reasons.append("weight_definition_incomplete")

    class_rules = spec.get("class_rules") if isinstance(spec.get("class_rules"), Mapping) else {}
    for field in ("positive", "negative", "unlabeled"):
        if _is_unresolved(class_rules.get(field)):
            errors.append(f"class_rules.{field} must be explicit")
            reasons.append("sample_rule_incomplete")

    spatial = spec.get("spatial_scale") if isinstance(spec.get("spatial_scale"), Mapping) else {}
    if spatial.get("support") not in {"well_point", "well_interval", "2d_slice", "3d_voxel"}:
        errors.append("spatial_scale.support is outside the contract enum")
        reasons.append("spatial_scale_incomplete")
    for field in ("support", "coordinate_system", "vertical_domain", "resolution", "alignment_tolerance"):
        if _is_unresolved(spatial.get(field)):
            errors.append(f"spatial_scale.{field} must be explicit")
            reasons.append("spatial_scale_incomplete")

    split = spec.get("split_strategy") if isinstance(spec.get("split_strategy"), Mapping) else {}
    if split.get("strategy") not in {"well_holdout", "spatial_block", "well_and_spatial"}:
        errors.append("split_strategy.strategy is outside the contract enum")
        reasons.append("split_incomplete")
    for field in ("strategy", "group_key", "train_rule", "validation_rule", "test_rule"):
        if _is_unresolved(split.get(field)):
            errors.append(f"split_strategy.{field} must be explicit")
            reasons.append("split_incomplete")
    if split.get("fit_statistics_scope") != "train_only":
        errors.append("split_strategy.fit_statistics_scope must be train_only")
        reasons.append("test_statistics_forbidden")
    if not split.get("leakage_guards"):
        errors.append("split_strategy.leakage_guards must not be empty")
        reasons.append("split_incomplete")

    catalog = inventory.get("field_catalog", {})
    allowed = spec.get("allowed_source_fields") if isinstance(spec.get("allowed_source_fields"), list) else []
    allowed_pairs = {_field_pair(entry) for entry in allowed if isinstance(entry, Mapping)}
    allowed_roles: dict[tuple[str, str], str] = {}
    for entry in allowed:
        if not isinstance(entry, Mapping):
            continue
        role = str(entry.get("role", ""))
        if role not in {"candidate_evidence", "inference_input", "label_only", "join_key"}:
            errors.append(f"allowed_source_fields contains invalid role {role!r}")
            reasons.append("schema_mismatch")
        allowed_roles[_field_pair(entry)] = role
    referenced_sections = {
        "allowed_source_fields": allowed,
        "label_construction.formula_field_refs": construction.get("formula_field_refs", []),
        "inference_allowed_inputs": spec.get("inference_allowed_inputs", []),
    }
    for section, entries in referenced_sections.items():
        if not isinstance(entries, list) or not entries:
            errors.append(f"{section} must be a non-empty list")
            reasons.append("incomplete_contract")
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                errors.append(f"{section} contains a non-object field reference")
                reasons.append("unknown_source_field")
                continue
            source, field = _field_pair(entry)
            if source not in catalog or field not in catalog.get(source, []):
                errors.append(f"{section} references unavailable field {source}.{field}")
                reasons.append("unknown_source_field")
            if section != "allowed_source_fields" and (source, field) not in allowed_pairs:
                errors.append(f"{section} field {source}.{field} is not allowed")
                reasons.append("field_not_allowed")

    inference_pairs = {
        _field_pair(entry) for entry in spec.get("inference_allowed_inputs", [])
        if isinstance(entry, Mapping)
    }
    formula_pairs = {
        _field_pair(entry) for entry in construction.get("formula_field_refs", [])
        if isinstance(entry, Mapping)
    }
    leaked = sorted(inference_pairs & formula_pairs)
    if leaked:
        errors.append(f"label-construction fields cannot be inference inputs: {leaked}")
        reasons.append("label_input_leakage")
    invalid_inference_roles = sorted(
        pair for pair in inference_pairs if allowed_roles.get(pair) != "inference_input"
    )
    if invalid_inference_roles:
        errors.append(f"inference inputs must carry role=inference_input: {invalid_inference_roles}")
        reasons.append("label_input_leakage")

    if inventory.get("label_readiness", {}).get("sweetspot_truth_found"):
        errors.append("inventory contains an unreviewed possible direct sweetspot field")
        reasons.append("direct_label_requires_review")
    return list(dict.fromkeys(errors)), list(dict.fromkeys(reasons))


def evaluate_label_spec(
    target_id: str,
    spec_path: Path | None,
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
) -> LabelGateResult:
    if spec_path is None:
        return LabelGateResult(
            target_id, False, "SKIP", ("label_spec_missing",),
            ("no target-specific label_spec path was supplied",), None, None, None,
        )
    path = Path(spec_path)
    if not path.is_file():
        return LabelGateResult(
            target_id, False, "SKIP", ("label_spec_missing",),
            (f"label_spec does not exist: {path}",), path.as_posix(), None, None,
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return LabelGateResult(
            target_id, False, "SKIP", ("label_spec_unreadable",),
            (f"{type(exc).__name__}: {exc}",), path.as_posix(), None, None,
        )
    if not isinstance(loaded, Mapping):
        return LabelGateResult(
            target_id, False, "SKIP", ("label_spec_invalid",),
            ("label_spec root must be an object",), path.as_posix(), None, None,
        )
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    errors, reasons = _validate_spec(loaded, inventory)
    digest = _canonical_hash(loaded)
    return LabelGateResult(
        target_id=target_id,
        approved=not errors,
        status="PASS" if not errors else "SKIP",
        reason_codes=tuple(reasons),
        errors=tuple(errors),
        spec_path=path.as_posix(),
        spec_sha256=digest,
        spec=dict(loaded),
    )
