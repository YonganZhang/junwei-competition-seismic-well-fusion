#!/usr/bin/env python3
"""Fail-closed Stage-2 data gate for the frozen fault candidate set.

This runner consumes only portable P4/Stage-1 evidence.  It does not accept a
dataset, model, loader, checkpoint, or test argument and contains no training
path.  Until coverage-audited negative and unknown masks make a development
split scientifically valid, every preregistered cell remains blocked and no
leaderboard is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACK_DIR = Path(__file__).resolve().parent
STAGE1_SUMMARY = TRACK_DIR / "_outputs" / "p5_stage1" / "summary.json"
SOURCE_LOCK = TRACK_DIR.parents[2] / "_models" / "fault" / "p5_source_locks.json"
CV_PLAN = TRACK_DIR / "_outputs" / "p4_preflight" / "buffered_cv_plan.json"
BLIND_AUDIT = TRACK_DIR / "_outputs" / "p4_preflight" / "blind_test_not_feasible.json"
DEFAULT_OUTPUT_DIR = TRACK_DIR / "_outputs" / "p5_stage2"

ROOT_SEED = 2693
TASK_ID = "fault_stick_segmentation"
LANE = "fault_3d_segmentation_data_gate"
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
RESULTS_FILENAME = "p5_stage2_results.jsonl"
SUMMARY_FILENAME = "p5_stage2_summary.json"
FIXED_BUDGET = {
    "lane_type": "3d_neural_or_operator",
    "max_parameter_updates": 80,
    "max_wall_seconds": 900,
    "train_sample_cap": None,
    "validation_sample_cap": None,
    "sample_caps_status": "unavailable_until_valid_development_split",
}


class Stage2EvidenceInvalid(RuntimeError):
    """Portable prerequisite evidence is missing, inconsistent, or unsafe."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2EvidenceInvalid(f"required portable evidence is missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2EvidenceInvalid(f"required portable evidence is not a JSON object: {path.name}")
    return payload


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def derive_seed(root_seed: int, model_id: str, role: str) -> int:
    """Derive stable, model-specific seeds without consuming a random stream."""

    material = f"fault|p5-stage2|{root_seed}|{model_id}|{role}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF
    return value or 1


def _validate_frozen_inputs(
    source_lock: Mapping[str, Any],
    stage1: Mapping[str, Any],
    cv_plan: Mapping[str, Any],
    blind_audit: Mapping[str, Any],
) -> None:
    locked_ids = tuple(item.get("model_id") for item in source_lock.get("models", ()))
    if (
        source_lock.get("protocol") != "fault-p5-source-lock-v1"
        or source_lock.get("candidate_count") != 10
        or locked_ids != FIRST_TEN_MODEL_IDS
    ):
        raise Stage2EvidenceInvalid("source lock does not match the frozen first-ten fault order")
    if (
        stage1.get("protocol") != "fault-p5-stage1-v1"
        or stage1.get("track_id") != "fault"
        or stage1.get("candidate_count") != 10
        or tuple(stage1.get("model_ids", ())) != FIRST_TEN_MODEL_IDS
        or tuple(item.get("model_id") for item in stage1.get("results", ()))
        != FIRST_TEN_MODEL_IDS
    ):
        raise Stage2EvidenceInvalid("Stage-1 summary does not cover the frozen first-ten fault order")
    if stage1.get("p4_evidence", {}).get("frozen_test_accessed") is not False:
        raise Stage2EvidenceInvalid("Stage-1 evidence does not prove a closed frozen-test firewall")
    if cv_plan.get("plan_version") != "fault-buffered-cv-v1":
        raise Stage2EvidenceInvalid("unexpected fault buffered-CV plan version")
    if blind_audit.get("audit_version") != "fault-blind-test-v1":
        raise Stage2EvidenceInvalid("unexpected fault blind-test audit version")


def _validate_source_hash_links(
    stage1: Mapping[str, Any], source_hashes: Mapping[str, str]
) -> None:
    expected_links = {
        "p5_source_locks.json": stage1.get("source_lock", {}).get("sha256"),
        "p4_preflight/buffered_cv_plan.json": stage1.get("p4_evidence", {}).get(
            "cv_plan_sha256"
        ),
        "p4_preflight/blind_test_not_feasible.json": stage1.get("p4_evidence", {}).get(
            "blind_audit_sha256"
        ),
    }
    mismatches = [
        name for name, expected in expected_links.items() if source_hashes.get(name) != expected
    ]
    if mismatches:
        raise Stage2EvidenceInvalid(
            "Stage-1 evidence hash links do not match current frozen sources: "
            + ", ".join(mismatches)
        )


def audit_data_gate(
    stage1: Mapping[str, Any],
    cv_plan: Mapping[str, Any],
    blind_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate label/split sufficiency without reading data or constructing labels."""

    development = dict(stage1.get("development_source", {}))
    verified_negative_count = int(development.get("verified_negative_labels", 0) or 0)
    verified_negative_status = str(
        development.get("verified_negative_audit_status", "absent") or "absent"
    )
    verified_negative_hash = development.get("verified_negative_audit_sha256")
    verified_negative_ready = (
        verified_negative_count > 0
        and verified_negative_status == "complete"
        and _is_sha256(verified_negative_hash)
    )

    unknown_count = int(development.get("unknown_labels", 0) or 0)
    unknown_status = str(development.get("unknown_mask_audit_status", "absent") or "absent")
    unknown_hash = development.get("unknown_mask_audit_sha256")
    unknown_source = str(development.get("unknown_mask_source", "absent") or "absent")
    explicit_unknown_ready = (
        unknown_status == "complete"
        and _is_sha256(unknown_hash)
        and unknown_source == "coverage_audited_source_mask"
    )

    effective_folds = int(cv_plan.get("effective_n_splits", 0) or 0)
    split_ready = cv_plan.get("status") == "ready" and effective_folds >= 2
    blockers: list[dict[str, Any]] = []
    if not verified_negative_ready:
        blockers.append(
            {
                "code": "AUDITED_VERIFIED_NEGATIVE_COVERAGE_MISSING",
                "detail": "no covered, audit-proven formal negative voxels are available",
            }
        )
    if not explicit_unknown_ready:
        blockers.append(
            {
                "code": "COVERAGE_AUDITED_UNKNOWN_MASK_MISSING",
                "detail": (
                    "P4 represents unlabelled voxels as unknown in memory, but the source evidence "
                    "has no explicit coverage-audited unknown mask/provenance"
                ),
            }
        )
    if not split_ready:
        blockers.append(
            {
                "code": "DEVELOPMENT_SPLIT_NOT_FEASIBLE",
                "detail": "the frozen buffered development CV plan has fewer than two valid folds",
            }
        )

    return {
        "status": "blocked" if blockers else "ready",
        "formal_training_allowed": not blockers,
        "verified_negative_coverage": {
            "status": "ready" if verified_negative_ready else "blocked",
            "observed_labels": verified_negative_count,
            "audit_status": verified_negative_status,
            "audit_sha256": verified_negative_hash,
            "required": "positive count plus complete coverage audit and SHA-256 provenance",
        },
        "unknown_mask": {
            "status": "ready" if explicit_unknown_ready else "blocked",
            "observed_unknown_labels": unknown_count,
            "in_memory_unknown_semantics_available": unknown_count > 0,
            "source_mask_audit_status": unknown_status,
            "source_mask_audit_sha256": unknown_hash,
            "source_mask_kind": unknown_source,
            "required": "explicit coverage_audited_source_mask with complete audit and SHA-256",
        },
        "development_split": {
            "status": "ready" if split_ready else "blocked",
            "plan_status": cv_plan.get("status"),
            "requested_n_splits": cv_plan.get("requested_n_splits"),
            "effective_n_splits": effective_folds,
            "fold_count": len(cv_plan.get("folds", ())),
            "downgrade_reason": cv_plan.get("downgrade_reason"),
        },
        "blind_test_audit": {
            "status": blind_audit.get("status"),
            "reason_code": blind_audit.get("reason_code"),
            "complete_candidate_block_count": len(blind_audit.get("complete_candidate_blocks", ())),
            "historical_test_role": "regression_evidence_only",
        },
        "blockers": blockers,
        "random_negative_generation": {
            "allowed": False,
            "performed": False,
            "generated_voxels": 0,
        },
    }


def _seed_tree(model_id: str, root_seed: int) -> dict[str, int]:
    return {
        "root": root_seed,
        "model": derive_seed(root_seed, model_id, "model"),
        "loader": derive_seed(root_seed, model_id, "loader"),
        "sampler": derive_seed(root_seed, model_id, "sampler"),
    }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _record_for_model(
    model_id: str,
    *,
    source_revision: str,
    stage1_result: Mapping[str, Any],
    data_gate: Mapping[str, Any],
    split_hash: str,
    root_seed: int,
) -> dict[str, Any]:
    blocked = data_gate["status"] == "blocked"
    seed_tree = _seed_tree(model_id, root_seed)
    return {
        "protocol": "fault-p5-stage2-cell-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "model_id": model_id,
        "source_revision": source_revision,
        "status": "blocked" if blocked else "eligible",
        "ranking_status": "not_rankable",
        "reason": {
            "code": "FAULT_DATA_GATE_BLOCKED" if blocked else None,
            "message": (
                "formal Stage-2 pilot is prohibited until every scientific data gate is ready"
                if blocked
                else "data gate is ready; this gate-only runner still performs no pilot"
            ),
            "blocking_codes": [item["code"] for item in data_gate["blockers"]],
        },
        "seed": seed_tree["model"],
        "seed_tree": seed_tree,
        "split_hash": split_hash,
        "input_budget": dict(FIXED_BUDGET),
        "allocated_input": {
            "train_samples": 0,
            "validation_samples": 0,
            "input_voxels": 0,
        },
        "updates_completed": 0,
        "wall_time_seconds": 0.0,
        "peak_resource": {
            "process_max_rss_kib": None,
            "gpu_vram_bytes": None,
        },
        "validation_metrics": None,
        "stage1_evidence": {
            "status": stage1_result.get("status"),
            "evidence_state": stage1_result.get("evidence_state"),
            "reason_code": stage1_result.get("reason_code"),
            "evidence": stage1_result.get("evidence"),
        },
        "data_gate": data_gate,
        "operations": {
            "model_built": False,
            "training_invoked": False,
            "validation_invoked": False,
            "checkpoint_written": False,
            "random_negative_generation_invoked": False,
        },
        "test_firewall": {
            "runner_accepts_test_inputs": False,
            "frozen_test_loader_created": False,
            "frozen_test_accessed": False,
            "test_metrics_computed": False,
            "historical_test_role": "regression_evidence_only",
        },
    }


def run_stage2_data_gate(output_dir: Path, *, root_seed: int = ROOT_SEED) -> dict[str, Any]:
    """Write ten deterministic gate records and a portable not-rankable summary."""

    if root_seed != ROOT_SEED:
        raise ValueError(f"fault Stage-2 root_seed is frozen at {ROOT_SEED}")
    source_lock = _read_json(SOURCE_LOCK)
    stage1 = _read_json(STAGE1_SUMMARY)
    cv_plan = _read_json(CV_PLAN)
    blind_audit = _read_json(BLIND_AUDIT)
    _validate_frozen_inputs(source_lock, stage1, cv_plan, blind_audit)

    source_hashes = {
        "p5_source_locks.json": _sha256(SOURCE_LOCK),
        "p5_stage1/summary.json": _sha256(STAGE1_SUMMARY),
        "p4_preflight/buffered_cv_plan.json": _sha256(CV_PLAN),
        "p4_preflight/blind_test_not_feasible.json": _sha256(BLIND_AUDIT),
    }
    _validate_source_hash_links(stage1, source_hashes)
    split_hash = source_hashes["p4_preflight/buffered_cv_plan.json"]
    data_snapshot_hash = _sha256_payload(source_hashes)
    data_gate = audit_data_gate(stage1, cv_plan, blind_audit)
    stage1_results = {item["model_id"]: item for item in stage1["results"]}
    source_revisions = {
        item["model_id"]: str(item["source"]["revision"])
        for item in source_lock["models"]
    }
    if set(stage1_results) != set(FIRST_TEN_MODEL_IDS):
        raise Stage2EvidenceInvalid("Stage-1 result rows do not cover exactly the frozen first ten")

    records = [
        _record_for_model(
            model_id,
            source_revision=source_revisions[model_id],
            stage1_result=stage1_results[model_id],
            data_gate=data_gate,
            split_hash=split_hash,
            root_seed=root_seed,
        )
        for model_id in FIRST_TEN_MODEL_IDS
    ]
    results_content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    results_path = output_dir / RESULTS_FILENAME
    _atomic_write_text(results_path, results_content)

    blocked_count = sum(record["status"] == "blocked" for record in records)
    eligible_count = sum(record["status"] == "eligible" for record in records)
    summary = {
        "protocol": "fault-p5-stage2-summary-v1",
        "track_id": "fault",
        "task_id": TASK_ID,
        "lane": LANE,
        "status": "not_rankable" if blocked_count else "ready_for_separate_pilot_runner",
        "reason_code": "FAULT_DATA_GATE_BLOCKED" if blocked_count else None,
        "root_seed": root_seed,
        "candidate_count": len(records),
        "model_ids": list(FIRST_TEN_MODEL_IDS),
        "counts": {
            "expected": 10,
            "data_gate_evaluated": len(records),
            "attempted": 0,
            "passed": 0,
            "blocked": blocked_count,
            "eligible": eligible_count,
            "skip": 0,
            "failed": 0,
            "timeout": 0,
        },
        "fixed_budget": dict(FIXED_BUDGET),
        "budget_consumed": {
            "parameter_updates": 0,
            "train_samples": 0,
            "validation_samples": 0,
            "wall_seconds": 0.0,
        },
        "split_hash": split_hash,
        "data_snapshot_hash": data_snapshot_hash,
        "source_hashes": source_hashes,
        "results_artifact": {
            "path": RESULTS_FILENAME,
            "sha256": _sha256(results_path),
            "bytes": results_path.stat().st_size,
            "line_count": len(records),
        },
        "data_gate": data_gate,
        "leaderboard": {
            "status": "not_rankable",
            "generated": False,
            "valid_validation_metric_count": 0,
            "reason": "no candidate may consume development training/validation data while the gate is blocked",
        },
        "test_firewall": {
            "runner_accepts_test_inputs": False,
            "frozen_test_accessed": False,
            "test_metrics_computed": False,
            "blind_audit_status": blind_audit.get("status"),
            "historical_test_role": "regression_evidence_only",
        },
        "prohibitions": {
            "training_performed": False,
            "random_negatives_generated": False,
            "proxy_promoted_to_formal_negative": False,
            "frozen_test_accessed": False,
        },
    }
    _atomic_write_text(
        output_dir / SUMMARY_FILENAME,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--root-seed", type=int, default=ROOT_SEED)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = run_stage2_data_gate(args.output_dir, root_seed=args.root_seed)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
