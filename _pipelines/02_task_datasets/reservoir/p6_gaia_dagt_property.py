"""P6 property Gaia/DAGT evidence-pack builder.

This module only reads existing reservoir artifacts and materializes a private
evidence bundle.  It does not train, download, or call any external API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _models.gaia_dagt import (  # noqa: E402
    DEFAULT_SOURCE_MANIFEST,
    GaiaDAGTAdapter,
    ModelBatch,
    TrackSpec,
    supervisory_qc_agent,
    verify_default_source_manifest,
)

P5_CONTRACT_PATH = HERE / "p5_contract.py"
P5_CONTRACT_SPEC = importlib.util.spec_from_file_location("reservoir_p5_contract", P5_CONTRACT_PATH)
assert P5_CONTRACT_SPEC and P5_CONTRACT_SPEC.loader
P5_CONTRACT = importlib.util.module_from_spec(P5_CONTRACT_SPEC)
P5_CONTRACT_SPEC.loader.exec_module(P5_CONTRACT)

OUTPUT_DIR = HERE / "_outputs" / "p6_gaia_dagt_property"
QC_REPORT = "P6 property Gaia/DAGT supervisory QC report"

PROPERTY_SPLIT = {
    "development_families": ["15/9-19", "15/9-F-1", "15/9-F-11", "15/9-F-12"],
    "frozen_test_family": "15/9-F-15",
    "root_seed": 2693,
}

TARGET_ROWS = [
    {
        "target": "PHIF",
        "status": "no_verified_gain",
        "lane_status": "rankable",
        "baseline": "extra_trees_regressor",
    },
    {
        "target": "KLOGH",
        "status": "no_verified_gain",
        "lane_status": "rankable",
        "baseline": "extra_trees_regressor",
    },
    {
        "target": "SW",
        "status": "no_verified_gain",
        "lane_status": "rankable",
        "baseline": "xgboost_regressor",
    },
]

TABULAR_CANDIDATES = (
    "extra_trees_regressor",
    "xgboost_regressor",
    "tabm_regressor",
    "realmlp_regressor",
    "ft_transformer_regressor",
    "tabiclv2_regressor",
    "monai_densenet3d_regressor",
)


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(payload: dict[str, Any]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _qc_track_spec() -> TrackSpec:
    return TrackSpec(
        track_id="property_p6_gaia_dagt_qc",
        task_type="multitask",
        modality="tabular_qc",
        input_fields=("report", "source_manifest_digest", "split_hash"),
        target_fields=("PHIF", "KLOGH", "SW"),
        allowed_paths=(
            "_pipelines/02_task_datasets/reservoir/_outputs/p5_stage4_confirmation",
            "_pipelines/02_task_datasets/reservoir/_outputs/p5_stage3",
            "_models/property/source_lock.json",
        ),
        forbidden_paths=(
            "_pipelines/02_task_datasets/reservoir/_outputs/run_manifest.json",
            "_pipelines/02_task_datasets/reservoir/_outputs/test.h5",
            "_pipelines/02_task_datasets/reservoir/_outputs/guard.npz",
        ),
        metric_names=("mae",),
        base_seed=2693,
        source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        provenance={
            "agent_mode": "supervisory_qc_agent",
            "description": "P6 property supervisory QC evidence track",
        },
    )


def _qc_batch(track_spec: TrackSpec) -> ModelBatch:
    split_summary = _load_json(HERE / "_outputs" / "p5_stage4_confirmation" / "summary.json")
    return ModelBatch(
        track_spec=track_spec,
        features={
            "report_text": QC_REPORT,
            "source_manifest_digest": track_spec.source_manifest_digest,
            "split_hash": split_summary["source_hashes"]["stage3_split_hash"],
        },
        task_targets={
            "PHIF": [0.12, 0.18],
            "KLOGH": [1.0, 2.0],
            "SW": [0.30, 0.70],
        },
        task_metrics={"PHIF": "mae", "KLOGH": "mae", "SW": "mae"},
        task_masks={
            "PHIF": [True, True],
            "KLOGH": [True, True],
            "SW": [True, True],
        },
        feasibility={"PHIF": True, "KLOGH": True, "SW": True},
        metadata={
            "source_lock_digest": track_spec.source_manifest_digest,
            "split_family": PROPERTY_SPLIT["development_families"],
            "deny_list_expected": True,
            "predictive_text_agent_allowed": False,
        },
    )


def _candidate_state(model_id: str) -> dict[str, Any]:
    results_path = HERE / "_outputs" / "p5_stage2" / "p5_stage2_results.jsonl"
    rows: list[dict[str, Any]] = []
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            row = json.loads(line)
            if row.get("model_id") == model_id:
                rows.append(
                    {
                        "status": row.get("status"),
                        "lane": row.get("lane"),
                        "evidence_state": row.get("evidence_state"),
                        "reason": row.get("reason"),
                        "split_hash": row.get("split_hash"),
                        "test_firewall": row.get("test_firewall"),
                        "training_budget": row.get("training_budget"),
                        "resources": row.get("resources"),
                    }
                )
    return {
        "model_id": model_id,
        "state": rows[0]["status"] if rows else "missing",
        "records": rows,
    }


def _stage4_confirmation() -> dict[str, Any]:
    summary_path = HERE / "_outputs" / "p5_stage4_confirmation" / "summary.json"
    artifact_path = HERE / "_outputs" / "p5_stage4_confirmation" / "artifact_manifest.json"
    prep_path = HERE / "_outputs" / "p5_stage4_confirmation" / "preparation_manifest.json"
    vis_path = HERE / "_outputs" / "p5_stage4_confirmation" / "visualization_manifest.json"
    conf_path = HERE / "_outputs" / "p5_stage4_confirmation" / "confirmation_state.json"
    return {
        "summary": _load_json(summary_path),
        "artifact_manifest": _load_json(artifact_path),
        "preparation_manifest": _load_json(prep_path),
        "visualization_manifest": _load_json(vis_path),
        "confirmation_state": _load_json(conf_path),
        "hashes": {
            "summary": _hash_file(summary_path),
            "artifact_manifest": _hash_file(artifact_path),
            "preparation_manifest": _hash_file(prep_path),
            "visualization_manifest": _hash_file(vis_path),
            "confirmation_state": _hash_file(conf_path),
        },
    }


def _science_manifest() -> dict[str, Any]:
    manifest = _load_json(HERE / "_outputs" / "p5_stage4_confirmation" / "visualization_manifest.json")
    figures = [
        {
            "target": entry["target"],
            "kind": entry["kind"],
            "path": entry["path"],
            "sha256": entry["sha256"],
            "companion_md_sha256": entry["companion_md_sha256"],
        }
        for entry in manifest["figures"]
    ]
    return {
        "schema_version": 1,
        "track_id": "property",
        "stage": 6,
        "title": "P6 property SCI manifest",
        "figure_count": len(figures),
        "figures": figures,
        "build_command": "python3 _pipelines/02_task_datasets/reservoir/p6_gaia_dagt_property.py write",
        "source": _repo_relative(HERE / "_outputs" / "p5_stage4_confirmation" / "visualization_manifest.json"),
    }


def _portable_source_manifest_status() -> dict[str, Any]:
    upstream_root = Path(DEFAULT_SOURCE_MANIFEST.upstream_repo_root)
    statuses = []
    for record in verify_default_source_manifest():
        statuses.append(
            {
                "path": Path(record.path).relative_to(upstream_root).as_posix(),
                "expected_sha256": record.expected_sha256,
                "actual_sha256": record.actual_sha256,
                "status": record.status,
            }
        )
    return {
        "commit": DEFAULT_SOURCE_MANIFEST.commit,
        "digest": DEFAULT_SOURCE_MANIFEST.digest(),
        "files": statuses,
    }


def build_evidence_bundle() -> dict[str, Any]:
    track_spec = P5_CONTRACT.build_task_spec().to_dict()
    qc_track_spec = _qc_track_spec()
    batch = _qc_batch(qc_track_spec)
    adapter = GaiaDAGTAdapter.from_default_manifest(qc_track_spec, seed=2693)
    qc_evidence = adapter.build_agent_evidence(QC_REPORT, mode="supervisory_qc_agent")
    dry_run = adapter.dry_run(QC_REPORT, batch, mode="supervisory_qc_agent")

    stage4 = _stage4_confirmation()
    candidate_index = {model_id: _candidate_state(model_id) for model_id in TABULAR_CANDIDATES}
    state_table = [
        {
            "target": row["target"],
            "status": row["status"],
            "lane_status": row["lane_status"],
            "baseline": row["baseline"],
            "b0": "extra_trees_regressor" if row["target"] in {"PHIF", "KLOGH"} else "xgboost_regressor",
            "b1": "xgboost_regressor" if row["target"] in {"PHIF", "KLOGH"} else "extra_trees_regressor",
            "f0": "tabm_regressor",
            "f1": "realmlp_regressor / ft_transformer_regressor",
            "f2": "blocked_by_missing_sample_level_text",
            "c1": "disabled",
            "c2": "disabled",
        }
        for row in TARGET_ROWS
    ]

    conclusion = {
        "schema_version": 1,
        "track_id": "property",
        "status": "no_verified_gain",
        "status_reason": "No sample-level predictive text source exists, so only supervisory QC is admissible; F2/C1/C2 remain disabled.",
        "agent_mode": "supervisory_qc_agent",
        "predictive_text_agent": "blocked_by_missing_sample_level_text",
        "targets": {
            row["target"]: {
                "status": row["status"],
                "baseline": row["baseline"],
                "lane_status": row["lane_status"],
            }
            for row in TARGET_ROWS
        },
    }

    bundle = {
        "schema_version": 1,
        "track_id": "property",
        "root_seed": 2693,
        "source_manifest": {
            **_portable_source_manifest_status(),
        },
        "property_task_spec": track_spec,
        "qc_track_spec": qc_track_spec.to_dict(),
        "model_batch": batch.to_dict(),
        "agent_evidence": qc_evidence.to_dict(),
        "qc_dry_run": {
            "metric": dict(dry_run.metric),
            "prediction": dry_run.output.to_dict(),
            "svg": dry_run.svg,
            "svg_path": "qc_dry_run.svg",
            "svg_sha256": _hash_json({"svg": dry_run.svg}),
        },
        "property_split": PROPERTY_SPLIT,
        "candidate_state_index": {
            "B0": _candidate_state("extra_trees_regressor"),
            "B1": _candidate_state("xgboost_regressor"),
            "tabm_regressor": candidate_index["tabm_regressor"],
            "realmlp_regressor": candidate_index["realmlp_regressor"],
            "ft_transformer_regressor": candidate_index["ft_transformer_regressor"],
            "tabiclv2_regressor": candidate_index["tabiclv2_regressor"],
            "monai_densenet3d_regressor": candidate_index["monai_densenet3d_regressor"],
            "F0": candidate_index["tabm_regressor"],
            "F1": {
                "realmlp_regressor": candidate_index["realmlp_regressor"],
                "ft_transformer_regressor": candidate_index["ft_transformer_regressor"],
            },
            "TabICLv2": candidate_index["tabiclv2_regressor"],
            "MONAI": candidate_index["monai_densenet3d_regressor"],
        },
        "state_table": state_table,
        "science_manifest": _science_manifest(),
        "stage4_confirmation": stage4,
        "conclusion": conclusion,
    }
    bundle["bundle_sha256"] = _hash_json(bundle)
    return bundle


def _write_portable_outputs(bundle: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_svg = output_dir / "qc_dry_run.svg"
    _atomic_write(qc_svg, bundle["qc_dry_run"]["svg"])
    evidence_json = output_dir / "p6_gaia_dagt_property_evidence.json"
    conclusion_json = output_dir / "p6_gaia_dagt_property_conclusion.json"
    state_table_json = output_dir / "p6_gaia_dagt_property_state_table.json"
    science_manifest_json = output_dir / "p6_gaia_dagt_property_science_manifest.json"
    evidence_md = output_dir / "p6_gaia_dagt_property_evidence.md"

    _atomic_json(evidence_json, bundle)
    _atomic_json(conclusion_json, bundle["conclusion"])
    _atomic_json(state_table_json, {"rows": bundle["state_table"]})
    _atomic_json(science_manifest_json, bundle["science_manifest"])

    markdown = [
        "# P6 property Gaia/DAGT evidence pack",
        "",
        "## Summary",
        "",
        f"- status: `{bundle['conclusion']['status']}`",
        f"- agent mode: `{bundle['conclusion']['agent_mode']}`",
        f"- predictive text lane: `{bundle['conclusion']['predictive_text_agent']}`",
        f"- root seed: `{bundle['root_seed']}`",
        "",
        "## TrackSpec / ModelBatch",
        "",
        f"- property task spec digest: `{bundle['source_manifest']['digest']}`",
        f"- qc track spec digest: `{bundle['qc_track_spec']['source_manifest_digest']}`",
        f"- model batch keys: `{', '.join(bundle['model_batch'].keys())}`",
        "",
        "## Candidate state index",
        "",
        json.dumps(bundle["candidate_state_index"], indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## SCI manifest",
        "",
        json.dumps(bundle["science_manifest"], indent=2, sort_keys=True, ensure_ascii=False),
        "",
        "## Conclusion",
        "",
        json.dumps(bundle["conclusion"], indent=2, sort_keys=True, ensure_ascii=False),
        "",
    ]
    _atomic_write(evidence_md, "\n".join(markdown))

    return {
        "evidence_json": evidence_json,
        "conclusion_json": conclusion_json,
        "state_table_json": state_table_json,
        "science_manifest_json": science_manifest_json,
        "evidence_md": evidence_md,
        "qc_svg": qc_svg,
    }


def write_bundle(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    bundle = build_evidence_bundle()
    _write_portable_outputs(bundle, output_dir)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write",), nargs="?", default="write")
    args = parser.parse_args()
    if args.command != "write":  # pragma: no cover - defensive
        raise SystemExit(2)
    bundle = build_evidence_bundle()
    paths = _write_portable_outputs(bundle)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
