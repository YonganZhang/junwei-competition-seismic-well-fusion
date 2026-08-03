#!/usr/bin/env python3
"""Verify the evidence-backed lifecycle contract for one of the six tracks.

This entrypoint does not retrain models.  It checks that the archived development
evidence supports the configured baseline, candidate execution and promotion
decision.  A rejected agent route is therefore a valid pipeline outcome, while a
missing or contradictory decision is a hard failure.

@role: entry
@pipeline: six_track_agentic_lifecycle
@produces: structured JSON verification on stdout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("validate", "prepare", "baseline", "optimize", "promote", "refit", "verify")


@dataclass(frozen=True)
class Check:
    source: str
    path: str
    operator: str
    expected: Any


@dataclass(frozen=True)
class TrackContract:
    summary: str
    actions: str
    final: str | None
    checks: dict[str, tuple[Check, ...]]
    promoted_default: str
    rejected_route: str
    required_files: tuple[str, ...] = ()
    verification: str | None = None
    optimization: str | None = None
    audit: str | None = None


def _c(source: str, path: str, operator: str, expected: Any) -> Check:
    return Check(source, path, operator, expected)


TRACKS: dict[str, TrackContract] = {
    "fault": TrackContract(
        summary="_pipelines/02_task_datasets/fault/_outputs/p29_agent_action_effect/summary.json",
        actions="_pipelines/02_task_datasets/fault/_outputs/p29_agent_action_effect/action_effects.json",
        final="_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/cigbench_vs_baseline_lift_tolerance_v2/comparison.json",
        checks={
            "validate": (
                _c("summary", "frozen_test_accessed", "eq", False),
                _c("summary", "data_gate_blocked", "eq", True),
                _c("final", "split.development_only", "eq", True),
                _c("final", "split.frozen_holdout_accessed", "eq", False),
            ),
            "prepare": (_c("actions", "records", "nonempty", True),),
            "baseline": (
                _c("summary", "policies.A0_static_baseline.policy_id", "eq", "A0_static_baseline"),
            ),
            "optimize": (
                _c("summary", "policies.A2D_deterministic_agent.decision_accuracy", "gt", 0.5),
                _c("summary", "selection_promotion_intersection_ok", "eq", True),
            ),
            "promote": (
                _c("summary", "retained_policy", "eq", "A2D_deterministic_agent"),
                _c("summary", "rejected_policy", "eq", "A2L_llm_agent_execute"),
            ),
            "refit": (
                _c("final", "status", "eq", "READY"),
                _c("final", "decision.default_recommendation", "eq", "do_not_advance"),
                _c("final", "comparison.guard_lift.precision_lift", "gt", 1.0),
                _c("final", "comparison.guard_lift.average_precision_lift", "gt", 1.0),
                _c("final", "comparison.guard_lift.predicted_positive_fraction", "gt", 0.5),
                _c("final", "comparison.tolerance_radius_2.cig_bench.f1", "lt", 0.05),
            ),
        },
        promoted_default="fault_local_logistic_with_A2D_governance",
        rejected_route="CIG-Bench FaultPredictor and direct LLM execution",
        required_files=(
            "_pipelines/02_task_datasets/fault/_outputs/runs/audited_v2/baseline_model.joblib",
            "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/dev_subvolume.npz",
            "_pipelines/02_task_datasets/fault/_outputs/p30_3d_dev_gate_st10010/cigbench_vs_baseline_lift_tolerance_v2/manifest.json",
        ),
    ),
    "facies": TrackContract(
        summary="_pipelines/02_task_datasets/facies/_outputs/p29_agent_action_effect/summary.json",
        actions="_pipelines/02_task_datasets/facies/_outputs/p29_agent_action_effect/action_effects.json",
        final="_pipelines/02_task_datasets/facies/_outputs/p32_hybrid_agent_optimizer/summary.json",
        checks={
            "validate": (
                _c("summary", "gate.checks.no_frozen_test_access", "eq", True),
                _c("final", "data.frozen_test_accessed", "eq", False),
                _c("verification", "verified", "eq", True),
            ),
            "prepare": (_c("actions", "actions", "nonempty", True),),
            "baseline": (_c("summary", "a0.selection.policy_id", "eq", "A0_static_baseline"),),
            "optimize": (
                _c("summary", "action_noop_check_passed", "eq", True),
                _c("summary", "gate.direct_agent_endpoint_superiority", "eq", False),
                _c("final", "matched_budget.equal", "eq", True),
            ),
            "promote": (
                _c("final", "promotion_gate.decision", "eq", "RETAIN_HYBRID"),
                _c("final", "promotion_gate.task_nondegradation_pass", "eq", True),
                _c("verification", "selected_decision_stable", "eq", True),
            ),
            "refit": (
                _c("final", "promotion_gate.agent_minus_deterministic_equal_mean_mIoU", "gt", 0.005),
                _c("final", "agent.promotion.config_by_dataset.F3.fusion_scale_initial", "eq", 0.8),
                _c("final", "agent.promotion.config_by_dataset.F3.fusion_lr", "eq", 0.0005),
                _c("final", "agent.promotion.config_by_dataset.F3.dice_weight", "eq", 0.75),
            ),
        },
        promoted_default="P32_hybrid_F3_joint_gate0.8_lr5e-4_dice0.75_plus_Penobscot_A0",
        rejected_route="direct LLM endpoint without deterministic candidate scheduling",
        verification="_pipelines/02_task_datasets/facies/_outputs/p32_hybrid_agent_optimizer/independent_verification.json",
    ),
    "property": TrackContract(
        summary="_pipelines/02_task_datasets/reservoir/_outputs/p29_agent_action_effect/summary.json",
        actions="_pipelines/02_task_datasets/reservoir/_outputs/p29_agent_action_effect/action_effects.json",
        final="_pipelines/02_task_datasets/reservoir/_outputs/p32_hybrid_agent_optimizer/summary.json",
        checks={
            "validate": (
                _c("summary", "gate.status", "eq", "blocked"),
                _c("final", "data.frozen_test_accessed", "eq", False),
                _c("verification", "verified", "eq", True),
            ),
            "prepare": (_c("actions", "rows", "nonempty", True),),
            "baseline": (_c("summary", "strategies.A1.selected_by", "eq", "identity_replay"),),
            "optimize": (
                _c("summary", "strategies.A2D.selection_primary_delta_rel", "lt", 0.0),
                _c("final", "matched_budget.equal", "eq", True),
            ),
            "promote": (
                _c("final", "promotion_gate.decision", "eq", "RETAIN_HYBRID"),
                _c("final", "promotion_gate.paired_seed_wins", "eq", 3),
                _c("final", "promotion_gate.worst_target_nondegradation_2pct", "eq", True),
                _c("verification", "selected_decision_stable", "eq", True),
            ),
            "refit": (
                _c("final", "promotion_gate.agent_minus_deterministic_relative_primary", "lt", -0.01),
                _c("final", "agent.selected_candidate.model_name", "eq", "reservoir_linear"),
                _c("final", "agent.selected_candidate.model_kwargs.learning_rate", "eq", 0.01),
            ),
        },
        promoted_default="P32_hybrid_reservoir_linear_lr0.01",
        rejected_route="direct LLM selector without deterministic candidate scheduling and unavailable CIG-Bench PropertyPredictor",
        verification="_pipelines/02_task_datasets/reservoir/_outputs/p32_hybrid_agent_optimizer/independent_verification.json",
    ),
    "lithofacies": TrackContract(
        summary="_pipelines/02_task_datasets/lithofacies/_outputs/p29_agent_action_effect/summary.json",
        actions="_pipelines/02_task_datasets/lithofacies/_outputs/p29_agent_action_effect/action_effects.json",
        final="_pipelines/02_task_datasets/lithofacies/_outputs/default_baseline/summary.json",
        checks={
            "validate": (_c("summary", "gates.safe_observation_firewall", "eq", True),),
            "prepare": (
                _c("actions", "all_nonbaseline_actions_change_prediction", "eq", True),
                _c("actions", "all_nonbaseline_configs_differ", "eq", True),
            ),
            "baseline": (
                _c("final", "default_config.max_depth", "eq", 3),
                _c("final", "default_config.eta", "eq", 0.1),
                _c("final", "default_config.rounds", "eq", 60),
            ),
            "optimize": (
                _c("optimization", "matched_budget.equal", "eq", True),
                _c("optimization", "data.selection_promotion_overlap", "eq", False),
                _c("verification", "verified", "eq", True),
            ),
            "promote": (
                _c("optimization", "promotion_gate.decision", "eq", "KEEP_CURRENT_DEFAULT"),
                _c("optimization", "promotion_gate.agent_minus_incumbent_promotion_macro_f1", "lt", 0.0),
                _c("final", "decision.status", "eq", "ACCEPT_AS_DEFAULT"),
                _c("final", "decision.moment_or_large_model_contribution", "eq", False),
            ),
            "refit": (
                _c("final", "comparison.default_minus_legacy", "gt", 0.005),
                _c("final", "comparison.default_wins", "eq", 12),
            ),
        },
        promoted_default="XGBoost_depth3_eta0.1_rounds60",
        rejected_route="P33 hybrid candidate depth3_eta0.2, P29 direct LLM policy, and MOMENT causal attribution",
        verification="_pipelines/02_task_datasets/lithofacies/_outputs/p33_hybrid_agent_optimizer/independent_verification.json",
        optimization="_pipelines/02_task_datasets/lithofacies/_outputs/p33_hybrid_agent_optimizer/summary.json",
    ),
    "sweetspot": TrackContract(
        summary="_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/summary.json",
        actions="_pipelines/02_task_datasets/sweetspot/_outputs/p29_agent_action_effect/action_effects.json",
        final=None,
        checks={
            "validate": (_c("summary", "a1_same_hash", "eq", True),),
            "prepare": (_c("actions", "actions", "nonempty", True),),
            "baseline": (_c("summary", "a0_action_id", "eq", "T3_XGB_D4_ETA_0_05_ROUNDS_96"),),
            "optimize": (
                _c("summary", "a2l.status", "eq", "STOPPED"),
                _c("summary", "a2l.stop_requested", "eq", True),
            ),
            "promote": (
                _c("summary", "verdict", "eq", "REJECT_AGENT"),
                _c("summary", "retain_llm", "eq", False),
            ),
            "refit": (_c("summary", "a2d.mean_signed_normalized_delta", "eq", 0.0),),
        },
        promoted_default="frozen_A0_xgboost",
        rejected_route="direct LLM policy after legal stop",
    ),
    "reconstruction": TrackContract(
        summary="_pipelines/02_task_datasets/reconstruction/_outputs/p29_agent_action_effect_repair_v2/summary.json",
        actions="_pipelines/02_task_datasets/reconstruction/_outputs/p29_agent_action_effect_repair_v2/action_effects.json",
        final="_pipelines/02_task_datasets/reconstruction/_outputs/p21_fixed_foundation_ensemble/summary.json",
        checks={
            "validate": (
                _c("summary", "frozen_holdout_opened", "eq", False),
                _c("summary", "held_fold_purge_reused.all_held_folds", "eq", True),
                _c("audit", "protocol.matched_to_p21_rows_and_budget", "eq", True),
                _c("audit", "protocol.frozen_holdout_opened", "eq", False),
                _c("audit", "default_evidence_audit.p29.authoritative_output_usable_for_new_promotion", "eq", True),
            ),
            "prepare": (_c("actions", "", "min_length", 8),),
            "baseline": (
                _c("final", "decision.default_enabled", "eq", True),
                _c("final", "protocol.holdout_opened", "eq", False),
            ),
            "optimize": (
                _c("summary", "policy.oracle_used_for_promotion", "eq", False),
                _c("summary", "interface_contract.feature_cache_is_explicit_runtime_input", "eq", True),
                _c("summary", "interface_contract.query_side_seismic_required", "eq", True),
                _c("summary", "interface_contract.query_side_foundation_embedding_required", "eq", True),
                _c("summary", "interface_contract.seismic_weights_are_scalar_ensemble_members", "eq", True),
            ),
            "promote": (
                _c("summary", "policy.promotion.verdict", "eq", "RETAIN_FROZEN_BASELINE"),
                _c("summary", "policy.promotion.positive_folds", "lt", 3),
                _c("audit", "decision.promote_over_p21", "eq", False),
                _c("audit", "decision.p21_remains_default", "eq", True),
            ),
            "refit": (
                _c("audit", "default_evidence_audit.p29.corrected_a0_max_abs_difference_vs_p21", "lt", 1e-12),
                _c("audit", "decision.cross_modal_foundation_claimed", "eq", False),
            ),
        },
        promoted_default="P21_fixed_three_kernel_ensemble",
        rejected_route="P29 LLM-selected numerical strategy",
        audit="_pipelines/02_task_datasets/reconstruction/_outputs/p30_bounded_geostatistics_feasibility_v2/summary.json",
        required_files=(
            "_pipelines/02_task_datasets/reconstruction/_outputs/p30_bounded_geostatistics_feasibility_v2/fusion_io_contract.json",
            "_pipelines/02_task_datasets/reconstruction/_outputs/p30_bounded_geostatistics_feasibility_v2/predictions.npz",
        ),
    ),
}


def _load(path: str) -> tuple[Path, Any, str]:
    file_path = PROJECT_ROOT / path
    if not file_path.is_file():
        raise AssertionError(f"missing evidence file: {path}")
    payload = file_path.read_bytes()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"invalid JSON evidence: {path}: {exc}") from exc
    return file_path, data, hashlib.sha256(payload).hexdigest()


def _resolve(data: Any, dotted_path: str) -> Any:
    current = data
    if not dotted_path:
        return current
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise AssertionError(f"missing evidence field: {dotted_path}")
        current = current[part]
    return current


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "eq":
        return actual == expected
    if operator == "gt":
        return isinstance(actual, (int, float)) and math.isfinite(actual) and actual > expected
    if operator == "lt":
        return isinstance(actual, (int, float)) and math.isfinite(actual) and actual < expected
    if operator == "nonempty":
        return bool(actual) is bool(expected)
    if operator == "min_length":
        return hasattr(actual, "__len__") and len(actual) >= expected
    if operator == "in":
        return actual in expected
    raise AssertionError(f"unknown check operator: {operator}")


def evaluate(track: str, stage: str) -> dict[str, Any]:
    if track not in TRACKS:
        raise AssertionError(f"unknown track: {track}")
    if stage not in STAGES:
        raise AssertionError(f"unknown stage: {stage}")
    contract = TRACKS[track]
    sources: dict[str, tuple[Path, Any, str]] = {
        "summary": _load(contract.summary),
        "actions": _load(contract.actions),
    }
    if contract.final:
        sources["final"] = _load(contract.final)
    if contract.verification:
        sources["verification"] = _load(contract.verification)
    if contract.optimization:
        sources["optimization"] = _load(contract.optimization)
    if contract.audit:
        sources["audit"] = _load(contract.audit)
    required_artifacts: dict[str, dict[str, Any]] = {}
    for artifact in contract.required_files:
        artifact_path = PROJECT_ROOT / artifact
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise AssertionError(f"missing required artifact: {artifact}")
        required_artifacts[artifact] = {
            "bytes": artifact_path.stat().st_size,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }

    if stage == "verify":
        stage_names: Iterable[str] = STAGES[:-1]
    else:
        stage_names = (stage,)

    results: list[dict[str, Any]] = []
    for stage_name in stage_names:
        for check in contract.checks[stage_name]:
            if check.source not in sources:
                raise AssertionError(f"{track}:{stage_name} references absent source {check.source}")
            actual = _resolve(sources[check.source][1], check.path)
            passed = _compare(actual, check.operator, check.expected)
            results.append(
                {
                    "stage": stage_name,
                    "source": check.source,
                    "field": check.path or "$",
                    "operator": check.operator,
                    "expected": check.expected,
                    "actual": actual,
                    "passed": passed,
                }
            )
            if not passed:
                raise AssertionError(
                    f"{track}:{stage_name} contract mismatch at {check.source}.{check.path}: "
                    f"expected {check.operator} {check.expected!r}, got {actual!r}"
                )

    return {
        "schema_version": "six_track_lifecycle/v1",
        "track": track,
        "stage": stage,
        "status": "PASS",
        "promoted_default": contract.promoted_default,
        "rejected_route": contract.rejected_route,
        "checks": results,
        "evidence": {
            name: {"path": str(item[0].relative_to(PROJECT_ROOT)), "sha256": item[2]}
            for name, item in sources.items()
        },
        "required_artifacts": required_artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=tuple(TRACKS), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate(args.track, args.stage)
    except AssertionError as exc:
        result = {
            "schema_version": "six_track_lifecycle/v1",
            "track": args.track,
            "stage": args.stage,
            "status": "FAIL",
            "error": str(exc),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
