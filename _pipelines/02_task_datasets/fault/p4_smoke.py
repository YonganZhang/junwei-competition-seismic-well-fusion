#!/usr/bin/env python3
"""Fast reproducible P4 fault smoke: real audit plus tiny verified-label baseline."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACK_DIR = Path(__file__).resolve().parent
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "_code", TRACK_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from _code.ml_framework.artifacts import (  # noqa: E402
    ArtifactManifest,
    atomic_write_json,
    hash_file,
    hash_payload,
)
from p4_contract import fault_task_spec  # noqa: E402
from p4_split import (  # noqa: E402
    audit_blind_test,
    build_buffered_spatial_cv,
    spatial_samples_from_audited_manifest,
)
from p4_workflow import (  # noqa: E402
    assert_development_interfaces_have_no_test_argument,
    fault_hpo_plan,
    fixed_baseline_configs,
    run_tiny_baseline_smoke,
)


FAULT_POINTS = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs" / "fault_points.npz"
SEISMIC_INDEX = PROJECT_ROOT / "_pipelines" / "01_common_preprocess" / "outputs" / "seismic_index.npz"
ANNOTATION_COVERAGE = (
    PROJECT_ROOT
    / "_pipelines"
    / "01_common_preprocess"
    / "outputs"
    / "fault_annotation_coverage.json"
)
AUDITED_RUN = TRACK_DIR / "_outputs" / "runs" / "audited_v2"
BUILD_SUMMARY = AUDITED_RUN / "build_summary.json"
AUDITED_SPLIT = AUDITED_RUN / "split_manifest.json"


def run_preflight(output_dir: Path, *, root_seed: int = 2693) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    blind = audit_blind_test(
        fault_points_path=FAULT_POINTS,
        seismic_index_path=SEISMIC_INDEX,
        audited_build_summary_path=BUILD_SUMMARY,
        audited_split_manifest_path=AUDITED_SPLIT,
        annotation_coverage_path=ANNOTATION_COVERAGE,
    )
    blind_name = (
        "blind_test_manifest.json"
        if blind["status"] == "frozen"
        else "blind_test_not_feasible.json"
    )
    atomic_write_json(output_dir / blind_name, blind)

    audited_manifest = json.loads(AUDITED_SPLIT.read_text(encoding="utf-8"))
    spatial_samples = spatial_samples_from_audited_manifest(audited_manifest)
    cv_plan = build_buffered_spatial_cv(
        spatial_samples,
        requested_n_splits=5,
        buffer_inlines=8,
    )
    atomic_write_json(output_dir / "buffered_cv_plan.json", cv_plan.to_dict())
    spec = fault_task_spec()
    atomic_write_json(output_dir / "task_spec.json", spec.to_dict())
    atomic_write_json(output_dir / "hpo_plan.json", asdict(fault_hpo_plan()))
    atomic_write_json(output_dir / "fixed_baselines.json", list(fixed_baseline_configs()))
    assert_development_interfaces_have_no_test_argument()
    tiny = run_tiny_baseline_smoke(seed=root_seed, epochs=12)
    atomic_write_json(output_dir / "tiny_baseline_smoke.json", tiny)

    report = {
        "smoke_version": "fault-p4-smoke-v1",
        "status": "passed_with_scientific_block" if blind["status"] != "frozen" else "passed",
        "root_seed": root_seed,
        "shared_contract_commit": "954e06c8d6d5454891c77aa370d244ae0b7453fc",
        "fault_source_sha256": {
            name: hash_file(TRACK_DIR / name)
            for name in (
                "p4_contract.py",
                "p4_split.py",
                "p4_workflow.py",
                "p4_visualization.py",
                "p4_smoke.py",
            )
        },
        "task_spec_status": "validated",
        "task_spec_hash": hash_payload(spec.to_dict()),
        "fixed_io_contract": "ModelBatch/ModelOutput [B,D,H,W]",
        "real_data_smoke": {
            "status": "passed",
            "fault_points": blind["fault_point_count"],
            "seismic_inline_extent": blind["searched_seismic_inline_extent"],
            "source_sha256": blind["source_sha256"],
            "active_hdf5_required": False,
            "training_performed": False,
        },
        "label_semantics": blind["label_semantics"],
        "blind_test_status": blind["status"],
        "blind_test_evidence": blind_name,
        "blind_test_audit_hash": blind["audit_hash"],
        "audited_v2_role": "regression_evidence_only",
        "cv_status": cv_plan.status,
        "cv_plan_hash": cv_plan.stable_hash(),
        "requested_n_splits": cv_plan.requested_n_splits,
        "effective_n_splits": cv_plan.effective_n_splits,
        "cv_downgrade_reason": cv_plan.downgrade_reason,
        "tiny_verified_label_smoke": tiny,
        "hpo_executed": False,
        "test_consumed": False,
        "long_training_performed": False,
    }
    atomic_write_json(output_dir / "smoke_report.json", report)

    manifest = ArtifactManifest("fault-p4-preflight", output_dir)
    roles = {
        blind_name: "blind_test_audit",
        "buffered_cv_plan.json": "development_split_plan",
        "task_spec.json": "task_spec",
        "hpo_plan.json": "hpo_plan",
        "fixed_baselines.json": "fixed_baselines",
        "tiny_baseline_smoke.json": "tiny_overfit_smoke",
        "smoke_report.json": "real_data_smoke",
    }
    for relative, role in roles.items():
        manifest.register(relative, role=role)
    manifest.write()
    manifest.verify()
    report["manifest"] = "manifest.json"
    report["manifest_sha256"] = hash_file(output_dir / "manifest.json")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRACK_DIR / "_outputs" / "p4_preflight",
    )
    parser.add_argument("--root-seed", type=int, default=2693)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_preflight(args.output_dir, root_seed=args.root_seed)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
