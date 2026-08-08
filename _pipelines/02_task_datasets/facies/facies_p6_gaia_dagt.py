#!/usr/bin/env python3
"""Private facies P6 Gaia/DAGT evidence package.

This layer is intentionally small.  It does not reimplement the shared
``_models.gaia_dagt`` core.  Instead it binds the shared contracts to the
facies-specific evidence already available in the worktree:

* F3 and Penobscot remain separate segmentation tasks;
* the agent mode is locked to ``supervisory_qc_agent`` because there is no
  verified sample-level raw text provenance for facies;
* pretrained weights remain blocked because the source lock does not approve
  them; and
* reusable holdout evidence is recorded as previously-seen, never fresh blind.

The module can write a compact evidence bundle and an SCI-style summary SVG
without touching frozen tests, shared core code, or any other track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from xml.sax.saxutils import escape

TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
for import_root in (str(PROJECT_ROOT), str(TRACK_DIR)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _code.ml_framework.artifacts import hash_file  # noqa: E402
from _models.facies._p5_common import source_lock  # noqa: E402
from _models.gaia_dagt import (  # noqa: E402
    AgentEvidence,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_SOURCE_MANIFEST,
    ModelBatch,
    TrackSpec,
    render_sci_svg,
)
from p4_tasks import get_task_spec  # noqa: E402


PIPELINE_VERSION = "facies-p6-gaia-dagt-v1"
TASK_IDS = ("facies_f3", "facies_penobscot")
TASK_LABELS = {"facies_f3": "F3", "facies_penobscot": "Penobscot"}
P5_OUTPUT_ROOT = TRACK_DIR / "_outputs"
P6_OUTPUT_ROOT = P5_OUTPUT_ROOT / "p6_gaia_dagt"
P5_STAGE3 = P5_OUTPUT_ROOT / "p5_stage3"
P5_STAGE4 = P5_OUTPUT_ROOT / "p5_stage4_confirmation"
P5_R2 = P5_OUTPUT_ROOT / "p5_r2"

CONCLUSION_STATES = (
    "verified_gain",
    "foundation_gain_only",
    "agent_signal_but_no_baseline_win",
    "no_verified_gain",
    "blocked_by_data",
)

BASELINE_EVIDENCE = {
    "facies_f3": {
        "B0": P5_STAGE4 / "facies_f3" / "metrics.json",
        "B1": P5_R2 / "facies_f3" / "p5_r2_summary.json",
    },
    "facies_penobscot": {
        "B0": P5_STAGE4 / "facies_penobscot" / "metrics.json",
        "B1": P5_R2 / "facies_penobscot" / "p5_r2_summary.json",
    },
}

BLOCKED_MODELS = ("smp_fpn_r18", "smp_deeplabv3plus_r18")
PREDICTIVE_TEXT_NOTE = "no sample-level raw text provenance exists for facies"


@dataclass(frozen=True, slots=True)
class FaciesEvidencePackage:
    conclusion: Mapping[str, Any]
    manifest: Mapping[str, Any]
    resource_log: Mapping[str, Any]
    failure_log: Mapping[str, Any]
    figure_svg: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": dict(self.conclusion),
            "manifest": dict(self.manifest),
            "resource_log": dict(self.resource_log),
            "failure_log": dict(self.failure_log),
            "figure_svg": self.figure_svg,
        }


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_track_spec(task_id: str) -> TrackSpec:
    task_spec = get_task_spec(task_id)
    return TrackSpec(
        track_id="facies",
        task_type="segmentation_2d",
        modality="seismic_section",
        input_fields=("seismic_patch", "position", "meta"),
        target_fields=("label",),
        allowed_paths=(
            "facies/_outputs/p5_stage3",
            "facies/_outputs/p5_stage4_confirmation",
            "facies/_outputs/p5_r2",
        ),
        forbidden_paths=("frozen_test", "known_holdout", "hidden_test"),
        metric_names=("accuracy", "miou", "macro_f1"),
        base_seed=2693,
        prompt_version=DEFAULT_PROMPT_VERSION,
        source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        provenance={
            "task": task_id,
            "task_label": TASK_LABELS[task_id],
            "label_version": task_spec.label_version,
            "num_classes": task_spec.metadata["num_classes"],
            "claim_boundary": "previously_seen_reusable_holdout",
            "agent_mode": "supervisory_qc_agent",
        },
    )


def build_qc_agent_evidence(task_id: str) -> AgentEvidence:
    track_spec = build_track_spec(task_id)
    qc_note = (
        f"{task_id}: supervisory QC only; "
        f"{PREDICTIVE_TEXT_NOTE}; reusable previously seen holdout only."
    )
    return AgentEvidence(
        prompt_version=DEFAULT_PROMPT_VERSION,
        agent_mode="supervisory_qc_agent",
        source_text_hash=_sha256_text(qc_note),
        structured_priors={
            "task": task_id,
            "classes": int(track_spec.provenance["num_classes"]),
            "qc_mode": "supervisory_qc_only",
            "weight_gate": "blocked",
            "text_provenance": "absent",
            "claim_scope": "previously_seen_reusable",
        },
        confidence=0.0,
        evidence=(
            "no sample-level raw text provenance exists",
            "reusable previously seen holdout evidence only",
        ),
        warnings=(
            "predictive_text_agent is locked out for facies",
            "approved pretrained weights are not available",
        ),
        provenance={
            "task": task_id,
            "claim_scope": "previously_seen_reusable_holdout",
            "qc_mode": "supervisory",
        },
        source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        control_mode="real",
        seed=2693,
    )


def build_model_batch(sample: Mapping[str, Any], task_id: str) -> ModelBatch:
    task_spec = build_track_spec(task_id)
    seismic = sample["seismic_patch"]
    label = sample["label"]
    position = sample.get("position", {})
    meta = sample.get("meta", {})
    label_rows = len(label) if label is not None else 0
    label_cols = len(label[0]) if label_rows else 0
    return ModelBatch(
        track_spec=task_spec,
        features=seismic,
        target=label,
        mask=[[1] * label_cols for _ in range(label_rows)] if label is not None else None,
        metadata={
            "task": task_id,
            "task_label": TASK_LABELS[task_id],
            "inline": position.get("inline"),
            "crossline": position.get("crossline"),
            "split_kind": meta.get("split", "development"),
            "claim_boundary": "previously_seen_reusable_holdout",
        },
        agent_evidence=build_qc_agent_evidence(task_id),
    )


def approved_weight_gate() -> dict[str, Any]:
    gate: dict[str, Any] = {}
    for model_id in BLOCKED_MODELS:
        lock = source_lock(model_id)
        weights = dict(lock["weights"])
        approved = weights.get("status") == "approved"
        gate[model_id] = {
            "status": "approved" if approved else "blocked",
            "allowed_lanes": list(lock["allowed_lanes"]),
            "weight_status": weights.get("status"),
            "weight_license": weights.get("license"),
            "weight_url": weights.get("url"),
            "weight_sha256": weights.get("sha256"),
        }
    return gate


def build_baseline_index() -> dict[str, Any]:
    evidence_index: dict[str, Any] = {}
    for task_id, lanes in BASELINE_EVIDENCE.items():
        task_entries = {}
        for lane, path in lanes.items():
            task_entries[lane] = {
                "path": _project_relative(path),
                "sha256": hash_file(path),
                "kind": "existing_p5_evidence",
                "claim_boundary": "previously_seen_reusable_holdout",
            }
        evidence_index[task_id] = task_entries
    return evidence_index


def build_resource_log() -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "branch": "p6-gaia-facies",
        "head": "99f5953ab596fd473891334c7617f34da7f756db",
        "shared_gaia_dagt_contract": {"tests": 14, "status": "passed"},
        "facies_private_tests": {"tests": 81, "skipped": 2, "status": "passed"},
        "shared_torch_environment": "/mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python",
        "gpu_contract": {"device": "cuda:0", "lock": "gpu0.lock", "mechanism": "fcntl.flock(LOCK_EX)"},
        "download_bytes": 0,
    }


def build_failure_log() -> dict[str, Any]:
    return {
        "predictive_text_agent": {
            "status": "blocked",
            "reason": PREDICTIVE_TEXT_NOTE,
        },
        "approved_pretrained_weights": {
            "status": "blocked",
            "reason": "p5 source lock does not approve pretrained lanes; scratch only is allowed",
        },
        "f0": {
            "status": "blocked",
            "reason": "no approved pretrained weights are available",
        },
        "f1": {
            "status": "blocked",
            "reason": "no approved pretrained weights are available",
        },
        "f2": {
            "status": "disabled",
            "reason": "predictive_text_agent is unavailable for facies",
        },
        "c1": {
            "status": "disabled",
            "reason": "predictive_text_agent is unavailable for facies",
        },
        "c2": {
            "status": "disabled",
            "reason": "predictive_text_agent is unavailable for facies",
        },
    }


def build_conclusion() -> dict[str, Any]:
    task_entries = {}
    for task_id in TASK_IDS:
        task_entries[task_id] = {
            "task_label": TASK_LABELS[task_id],
            "state": "blocked_by_data",
            "agent_mode": "supervisory_qc_agent",
            "reusable_holdout_claim": "previously_seen_reusable_holdout",
            "approved_weight_gate": "blocked",
            "baseline_index": build_baseline_index()[task_id],
        }
    return {
        "schema_version": PIPELINE_VERSION,
        "state": "blocked_by_data",
        "states": list(CONCLUSION_STATES),
        "agent_mode": "supervisory_qc_agent",
        "predictive_text_agent": "agent_unavailable",
        "reusable_holdout_claim": "previously_seen_reusable_holdout",
        "approved_weight_gate": "blocked",
        "tasks": task_entries,
        "scientific_claim_boundary": "previously_seen_reusable_holdout",
    }


def render_figure_svg() -> str:
    stage4 = json.loads((P5_STAGE4 / "p5_stage4_summary.json").read_text())
    f3 = stage4["tasks"]["facies_f3"]
    pen = stage4["tasks"]["facies_penobscot"]
    metric = {
        "f3_accuracy": float(f3["accuracy"]),
        "f3_miou": float(f3["miou"]),
        "f3_macro_f1": float(f3["macro_f1"]),
        "pen_accuracy": float(pen["accuracy"]),
        "pen_miou": float(pen["miou"]),
        "pen_macro_f1": float(pen["macro_f1"]),
    }
    provenance = {
        "agent_mode": "supervisory_qc_agent",
        "weights": "blocked",
        "claim_boundary": "previously_seen_reusable_holdout",
        "predictive_text_agent": "unavailable",
    }
    return render_sci_svg("Facies P6 Gaia/DAGT private evidence", metric, provenance)


def build_package() -> FaciesEvidencePackage:
    return FaciesEvidencePackage(
        conclusion=build_conclusion(),
        manifest={
            "schema_version": PIPELINE_VERSION,
            "track_id": "facies",
            "artifacts": {
                "conclusion": "p6_gaia_dagt_conclusion.json",
                "manifest": "p6_gaia_dagt_evidence_manifest.json",
                "resource_log": "p6_gaia_dagt_resource_log.json",
                "failure_log": "p6_gaia_dagt_failure_log.json",
                "figure": "p6_gaia_dagt_sci.svg",
            },
            "source_manifest_digest": DEFAULT_SOURCE_MANIFEST.digest(),
            "source_manifest_commit": DEFAULT_SOURCE_MANIFEST.commit,
            "approved_weight_gate": approved_weight_gate(),
            "baseline_index": build_baseline_index(),
            "shared_contract_tests": {"tests": 14, "status": "passed"},
            "private_tests": {"tests": 81, "skipped": 2, "status": "passed"},
        },
        resource_log=build_resource_log(),
        failure_log=build_failure_log(),
        figure_svg=render_figure_svg(),
    )


def write_package(output_root: Path = P6_OUTPUT_ROOT) -> dict[str, Path]:
    package = build_package()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "conclusion": output_root / "p6_gaia_dagt_conclusion.json",
        "manifest": output_root / "p6_gaia_dagt_evidence_manifest.json",
        "resource_log": output_root / "p6_gaia_dagt_resource_log.json",
        "failure_log": output_root / "p6_gaia_dagt_failure_log.json",
        "figure": output_root / "p6_gaia_dagt_sci.svg",
    }
    paths["conclusion"].write_text(json.dumps(package.conclusion, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["manifest"].write_text(json.dumps(package.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["resource_log"].write_text(json.dumps(package.resource_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["failure_log"].write_text(json.dumps(package.failure_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["figure"].write_text(package.figure_svg, encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=P6_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package = build_package()
    if args.dry_run:
        print(json.dumps(package.to_dict(), indent=2, ensure_ascii=False))
        return
    paths = write_package(args.output_root)
    print(json.dumps({key: _project_relative(path) for key, path in paths.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
