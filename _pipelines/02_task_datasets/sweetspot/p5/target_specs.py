"""Seven independent TaskSpec builders driven only by approved label contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from _code.ml_framework.contracts import TaskSpec

from .label_gate import LabelGateResult


@dataclass(frozen=True)
class TargetDefinition:
    target_id: str
    slug: str
    display_name: str
    head_name: str
    allowed_task_types: tuple[str, ...]
    required_figures: tuple[str, ...]


TARGETS: dict[str, TargetDefinition] = {
    "T1": TargetDefinition(
        "T1", "reservoir_quality", "储层质量", "T1_RESERVOIR_QUALITY",
        ("binary", "multiclass", "regression", "ranking"),
        ("t1_prediction", "t1_well_or_block_error"),
    ),
    "T2": TargetDefinition(
        "T2", "hydrocarbon_pay", "含烃有效层", "T2_HYDROCARBON_PAY",
        ("binary", "multiclass", "regression"),
        ("t2_interval_prediction", "t2_precision_recall"),
    ),
    "T3": TargetDefinition(
        "T3", "productivity", "产能", "T3_PRODUCTIVITY",
        ("regression", "ranking"),
        ("t3_forecast", "t3_well_error"),
    ),
    "T4": TargetDefinition(
        "T4", "water_breakthrough", "水突破风险", "T4_WATER_BREAKTHROUGH",
        ("binary", "regression", "survival"),
        ("t4_risk_timeline", "t4_calibration"),
    ),
    "T5": TargetDefinition(
        "T5", "remaining_oil_infill", "剩余油/加密井", "T5_REMAINING_OIL_INFILL",
        ("regression", "ranking", "multiclass"),
        ("t5_spatial_prediction", "t5_candidate_ranking"),
    ),
    "T6": TargetDefinition(
        "T6", "porosity", "孔隙度", "T6_POROSITY",
        ("regression",), ("t6_porosity_depth_track", "t6_porosity_residual"),
    ),
    "T7": TargetDefinition(
        "T7", "permeability", "渗透率", "T7_PERMEABILITY",
        ("regression",), ("t7_permeability_depth_track", "t7_permeability_residual"),
    ),
}


_METRIC_DIRECTIONS = {
    "average_precision": "maximize", "auprc": "maximize", "pr_auc": "maximize",
    "auroc": "maximize", "roc_auc": "maximize", "f1": "maximize", "macro_f1": "maximize",
    "r2": "maximize", "spearman": "maximize", "ndcg": "maximize", "top_k_hit": "maximize",
    "mae": "minimize", "rmse": "minimize", "brier": "minimize", "ece": "minimize",
    "crps": "minimize", "smape": "minimize", "time_error": "minimize",
}


def _task_type(spec: Mapping[str, Any]) -> str:
    output = spec["output"]
    kind = output["type"]
    if kind == "binary":
        return "binary"
    if kind == "multiclass":
        return "multiclass"
    if kind == "continuous_score":
        return "regression"
    if kind == "probability":
        return "binary" if len(output.get("classes", [])) == 2 else "regression"
    raise ValueError(f"unsupported approved output type: {kind}")


def _field_token(entry: Mapping[str, Any]) -> str:
    return f"{entry['source']}.{entry['field']}"


def build_task_spec(target_id: str, gate: LabelGateResult) -> TaskSpec:
    """Build one single-head TaskSpec; an unapproved gate can never build one."""
    if target_id not in TARGETS:
        raise KeyError(f"unknown sweetspot target {target_id!r}")
    if not gate.approved or gate.spec is None or not gate.spec_sha256:
        raise PermissionError(f"{target_id}: approved label_spec is required before TaskSpec construction")
    target = TARGETS[target_id]
    spec = gate.spec
    task_type = _task_type(spec)
    if task_type not in target.allowed_task_types:
        raise ValueError(f"{target_id}: approved output resolves to unsupported task type {task_type}")
    metric_names = tuple(str(item["name"]) for item in spec["metrics"])
    directions: dict[str, str] = {}
    for name in metric_names:
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        if normalized not in _METRIC_DIRECTIONS:
            raise ValueError(f"{target_id}: metric direction is not registered for {name!r}")
        directions[name] = _METRIC_DIRECTIONS[normalized]
    output = spec["output"]
    units = str(output["units"]) if output.get("units") is not None else "dimensionless"
    inference_inputs = tuple(_field_token(item) for item in spec["inference_allowed_inputs"])
    label_only = tuple(
        _field_token(item) for item in spec["allowed_source_fields"]
        if item.get("role") == "label_only"
    )
    split = spec["split_strategy"]
    spatial = spec["spatial_scale"]
    time_window = spec["time_window"]
    train_loss = (
        "bce_with_logits" if task_type == "binary"
        else "cross_entropy" if task_type == "multiclass"
        else "masked_mean_squared_error"
    )
    inference_transform = (
        "sigmoid" if task_type == "binary" else "softmax" if task_type == "multiclass" else "identity"
    )
    classes = list(output.get("classes", []))
    return TaskSpec(
        track_id="sweetspot",
        task_id=f"sweetspot.p5.{target.slug}.{spec['spec_version']}",
        task_type=task_type,
        input_modalities=(f"approved_{spatial['support']}",),
        targets=(target.head_name,),
        units={target.head_name: units},
        label_version=f"{target_id.lower()}-{spec['spec_version']}-{gate.spec_sha256[:12]}",
        target_masks={target.head_name: str(spec["class_rules"]["unlabeled"])},
        group_keys=(str(split["group_key"]),),
        target_transform={target.head_name: "identity_from_approved_builder_output"},
        inverse_transform={target.head_name: "identity_to_approved_output_space"},
        train_loss={target.head_name: train_loss},
        inference_transform={target.head_name: inference_transform},
        threshold_policy={"source": "approved_label_spec_or_development_only"},
        calibration_policy={"fit_scope": "development_only"},
        primary_metrics=metric_names,
        metric_directions=directions,
        spatial_buffer={
            "support": spatial["support"], "resolution": spatial["resolution"],
            "alignment_tolerance": spatial["alignment_tolerance"],
        },
        time_cutoff={
            "definition": time_window["definition"], "leakage_cutoff": time_window["leakage_cutoff"],
        },
        hpo={"stage": "P5 Stage 1 contract smoke only", "test_access": "forbidden"},
        visualizer_id=f"sweetspot_p5_{target.slug}",
        required_figures=target.required_figures,
        input_whitelist=inference_inputs,
        forbidden_inputs=label_only,
        metadata={
            "target_id": target_id,
            "target_name": target.display_name,
            "single_target_head": True,
            "class_count": len(classes),
            "approved_label_spec_sha256": gate.spec_sha256,
            "approval": dict(spec["approval"]),
            "label_formula_owned_by_contract": spec["label_construction"]["formula"],
            "no_proxy_fallback": True,
        },
    )
