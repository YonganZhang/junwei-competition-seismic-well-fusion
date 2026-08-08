#!/usr/bin/env python3
"""Build the private P6 reconstruction evidence package.

This module intentionally reuses frozen P5 reconstruction artifacts.  It does
not retrain, does not read ``test.h5``, and does not fabricate any 3-D voxel
arrays.  The output is a compact, reviewable evidence bundle that records:

- strict / conditional lane separation,
- TrackSpec / ModelBatch adapters over the archived 3-D volumes,
- supervisory QC evidence with the Gaia/DAGT deny-list contract,
- a five-state conclusion JSON,
- an SCI manifest pointing to the real 3-D figures already in the tree.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))

from ml_framework.artifacts import atomic_write_json, hash_file, hash_payload  # noqa: E402
from ml_framework.contracts import ModelBatch as FrameworkModelBatch  # noqa: E402
from ml_framework.contracts import TaskSpec as FrameworkTaskSpec  # noqa: E402

from _models.gaia_dagt.adapter import GaiaDAGTAdapter  # noqa: E402
from _models.gaia_dagt.contracts import AgentEvidence, TrackSpec  # noqa: E402
from _models.gaia_dagt.source_lock import SourceManifest  # noqa: E402


SCHEMA_VERSION = "p6-reconstruction-private-evidence-v1"
ROOT_SEED = 2693
MODES = ("strict", "conditional")
PACKAGE_DIRNAME = "p6_private_evidence"
SOURCE_EVIDENCE_DIR = HERE / "p5_stage4_confirmation"
P5_3D_DIR = HERE / "_outputs" / "3d_sci_v1"


@dataclass(frozen=True)
class ModeEvidence:
    mode: str
    track_spec: TrackSpec
    framework_task_spec: FrameworkTaskSpec
    model_batch: FrameworkModelBatch
    qc_evidence: AgentEvidence
    stage4_summary: Mapping[str, Any]
    stage4_results: Mapping[str, Any]
    stage3_summary: Mapping[str, Any]
    stage3_summary_hash: str
    stage3_visualization_manifest: Mapping[str, Any]
    stage3_visualization_manifest_hash: str
    conclusion: Mapping[str, Any]
    lane_table: Mapping[str, Any]
    sci_manifest: Mapping[str, Any]


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(mode: str) -> Mapping[str, Any]:
    path = SOURCE_EVIDENCE_DIR / mode / "predictions.npz"
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _require_existing(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact is missing: {path}")
    return path


def _lane_input_whitelist(mode: str) -> tuple[str, ...]:
    strict = (
        "coordinates",
        "seismic_amplitude",
        "seismic_local_rms",
        "seismic_vertical_gradient",
    )
    conditional = strict + (
        "global_well_constraints",
        "pseudo_test_well_constraints",
    )
    return conditional if mode == "conditional" else strict


def _lane_forbidden_inputs() -> tuple[str, ...]:
    return (
        "label",
        "target",
        "targets",
        "truth",
        "test",
        "test_metric",
        "test_metrics",
        "checkpoint",
        "holdout",
        "frozen",
        "split",
        "residual",
        "distance",
        "offset",
    )


def _task_spec(mode: str, summary: Mapping[str, Any]) -> FrameworkTaskSpec:
    required_figures = (
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/prediction_comparison.png",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/prediction_comparison.pdf",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/prediction_comparison.html",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/p5_stage3_reconstruction_{mode}.png",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/prediction_visualization_{mode}.png",
        f"_pipelines/02_task_datasets/reconstruction/p5_stage4_confirmation/{mode}/confirmation.png",
    )
    input_modalities = ("coordinates", "seismic", "3d_volume")
    if mode == "conditional":
        input_modalities = input_modalities + ("well_constraints",)
    return FrameworkTaskSpec(
        track_id="reconstruction",
        task_id=f"volve_porosity_{mode}_private_evidence",
        task_type="reconstruction",
        input_modalities=input_modalities,
        targets=("poro",),
        units={"poro": "fraction"},
        label_version="p6.private.reconstruction.v1",
        target_masks={"poro": "evaluation_mask"},
        group_keys=("mode", "split_hash", "sample_id"),
        target_transform={"poro": {"kind": "identity"}},
        inverse_transform={"poro": {"kind": "identity"}},
        train_loss={"poro": {"kind": "identity", "loss": "none"}},
        inference_transform={"poro": {"kind": "identity"}},
        threshold_policy={"poro": {"kind": "continuous", "clip": [0.0, 1.0]}},
        calibration_policy={"poro": {"kind": "none"}},
        primary_metrics=("rmse",),
        metric_directions={"rmse": "minimize", "mae": "minimize", "r2": "maximize", "spectral_log_rmse": "minimize", "out_of_range_rate": "minimize"},
        secondary_metrics=("mae", "r2", "spectral_log_rmse"),
        guardrail_metrics=("out_of_range_rate",),
        spatial_buffer={"k_blocks": 1, "shared_k4_fold": 4},
        hpo={"enabled": False, "reason": "frozen_P5_confirmation_only"},
        visualizer_id="reconstruction_p6_private_visualizer",
        required_figures=required_figures,
        input_whitelist=_lane_input_whitelist(mode),
        forbidden_inputs=_lane_forbidden_inputs(),
        metadata={
            "evaluation_mode": mode,
            "development_only": True,
            "fresh_blind": False,
            "field_generalization": False,
            "stage4_summary_hash": summary["summary_hash"],
        },
    )


def _gaia_track_spec(mode: str, summary: Mapping[str, Any]) -> TrackSpec:
    figures = (
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/three_d_feasibility.json",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/provenance.json",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/caption.md",
    )
    return TrackSpec(
        track_id="reconstruction",
        task_type="volume_3d",
        modality="3d_volume",
        input_fields=_lane_input_whitelist(mode),
        target_fields=("poro",),
        allowed_paths=(
            "_pipelines/02_task_datasets/reconstruction/p5_stage4_confirmation/",
            "_pipelines/02_task_datasets/reconstruction/p5_stage3_visualization_manifest.json",
            "_pipelines/02_task_datasets/reconstruction/results_strict.json",
            "_pipelines/02_task_datasets/reconstruction/results_conditional.json",
        ),
        forbidden_paths=(
            "_data/processed/reconstruction/test.h5",
            "_data/processed/reconstruction/known_holdout",
            "_tmp/",
        ),
        metric_names=("rmse", "mae", "r2", "spectral_log_rmse"),
        base_seed=ROOT_SEED,
        prompt_version="p6.gaia.dagt.v1",
        source_manifest_digest=summary["source_manifest_digest"],
        provenance={
            "mode": mode,
            "evaluation_mode": mode,
            "shared_k4_fold": 4,
            "development_only": True,
            "fresh_blind": False,
            "field_generalization": False,
        },
    )


def _framework_batch(mode: str, track_spec: FrameworkTaskSpec, arrays: Mapping[str, Any], summary: Mapping[str, Any]) -> FrameworkModelBatch:
    truth = np.asarray(arrays["truth"], dtype=np.float64)
    prediction = np.asarray(arrays["prediction"], dtype=np.float64)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    indices_kji = np.asarray(arrays["indices_kji"], dtype=np.int64)
    volume_shape_kji = tuple(int(value) for value in np.asarray(arrays["volume_shape_kji"], dtype=np.int64))
    mask = np.ones_like(truth, dtype=bool)
    return FrameworkModelBatch(
        inputs={
            "mode": [mode],
            "indices_kji": indices_kji.tolist(),
            "volume_shape_kji": list(volume_shape_kji),
            "truth": truth.tolist(),
            "prediction": prediction.tolist(),
            "residual": residual.tolist(),
        },
        targets={"poro": truth.tolist()},
        input_masks={
            "mode": [True],
            "indices_kji": [True] * int(indices_kji.shape[0]),
            "volume_shape_kji": [True] * 3,
            "truth": mask.tolist(),
            "prediction": mask.tolist(),
            "residual": mask.tolist(),
        },
        target_masks={"poro": mask.tolist()},
        sample_ids=[f"{mode}:{index}" for index in range(int(truth.size))],
        groups={"mode": [mode] * int(truth.size)},
        coordinates={"kji": indices_kji.tolist()},
        metadata={
            "mode": mode,
            "task_id": track_spec.task_id,
            "evidence_class": str(np.asarray(arrays["evidence_class"]).item()),
            "fresh_blind": bool(np.asarray(arrays["fresh_blind"]).item()),
            "prior_test_consumed": bool(np.asarray(arrays["prior_test_consumed"]).item()),
            "summary_hash": summary["summary_hash"],
            "prediction_summary": {
                "count": int(prediction.size),
                "mean": float(np.mean(prediction)),
                "min": float(np.min(prediction)),
                "max": float(np.max(prediction)),
            },
            "truth_summary": {
                "count": int(truth.size),
                "mean": float(np.mean(truth)),
                "min": float(np.min(truth)),
                "max": float(np.max(truth)),
            },
            "residual_summary": {
                "count": int(residual.size),
                "mean": float(np.mean(residual)),
                "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            },
            "development_protocol_mechanism_only": True,
            "strict_no_test_well_constraints": mode == "strict",
            "conditional_reuses_pseudo_test_constraints": mode == "conditional",
        },
    )


def _qc_evidence(mode: str, summary: Mapping[str, Any], source_manifest_digest: str) -> AgentEvidence:
    report = (
        f"P6 {mode} reconstruction QC: strict and conditional remain separated, "
        f"fresh_blind={summary['fresh_blind']}, prior_test_consumed={summary['prior_test_consumed']}, "
        "and no sample-level text source is available."
    )
    adapter = GaiaDAGTAdapter.from_default_manifest(
        _gaia_track_spec(mode, {"source_manifest_digest": source_manifest_digest, "summary_hash": summary["summary_hash"]}),
        control_mode="p6-private-evidence",
        seed=ROOT_SEED,
    )
    return adapter.build_agent_evidence(report, mode="supervisory_qc_agent")


def _lane_table(mode: str, stage4: Mapping[str, Any], stage3: Mapping[str, Any]) -> Mapping[str, Any]:
    results = stage4["results"][mode]
    metrics = results["metrics"]
    if mode == "conditional":
        b0 = {"status": "passed", "rmse": float(metrics["conditional_rmse"]), "r2": float(metrics["conditional_r2"])}
        b1 = {"status": "passed", "rmse": float(metrics["conditional_rmse"]), "r2": float(metrics["conditional_r2"])}
        shuffled = {"status": "passed", "rmse": float(metrics["conditional_rmse"]), "r2": float(metrics["conditional_r2"])}
        f0 = {"status": "blocked", "reason": "no approved foundation artifact"}
        f1 = {"status": "blocked", "reason": "no approved foundation artifact"}
    else:
        b0 = {"status": "passed", "rmse": float(metrics["strict_rmse"]), "r2": float(metrics["strict_r2"])}
        b1 = {"status": "not_applicable", "reason": "strict lane forbids test-region well constraints"}
        shuffled = {"status": "not_applicable", "reason": "strict lane forbids test-region well constraints"}
        f0 = {"status": "blocked", "reason": "no approved foundation artifact"}
        f1 = {"status": "blocked", "reason": "no approved foundation artifact"}
    return {
        "mode": mode,
        "shared_k4_fold": 4,
        "shared_k4_metric_mask_count": int(results["counts"]["metric_voxels"]),
        "prior_test_consumed": bool(stage4["prior_test_consumed"]),
        "fresh_blind": bool(stage4["fresh_blind"]),
        "B0": b0,
        "B1": b1,
        "shuffled": shuffled,
        "F0": f0,
        "F1": f1,
        "stage3_rankable": bool(stage3["rankable_by_lane"][mode]),
        "stage3_result_sha256": stage3["results_sha256"],
        "stage4_result_hash": results["result_hash"],
        "status": "no_verified_gain",
    }


def _sci_manifest(mode: str, stage4: Mapping[str, Any], stage3_visualization: Mapping[str, Any]) -> Mapping[str, Any]:
    figure_paths = (
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/prediction_comparison.png",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/prediction_comparison.pdf",
        f"_pipelines/02_task_datasets/reconstruction/_outputs/3d_sci_v1/{mode}/prediction_comparison.html",
        stage3_visualization["figures"][mode]["path"],
        f"_pipelines/02_task_datasets/reconstruction/_outputs/prediction_visualization_{mode}.png",
        f"_pipelines/02_task_datasets/reconstruction/p5_stage4_confirmation/{mode}/confirmation.png",
    )
    figures = []
    for path in figure_paths:
        p = HERE / Path(path).relative_to("_pipelines/02_task_datasets/reconstruction")
        figures.append(
            {
                "path": path,
                "sha256": hash_file(p) if p.is_file() else None,
                "exists": p.is_file(),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "development_only": True,
        "fresh_blind": False,
        "prior_test_consumed": True,
        "figure_count": len(figures),
        "figures": figures,
        "source_visualization_manifest_sha256": hash_file(HERE / "p5_stage3_visualization_manifest.json"),
        "stage4_confirmation_sha256": hash_file(SOURCE_EVIDENCE_DIR / "summary.json"),
    }


def build_mode(mode: str) -> ModeEvidence:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    stage3_summary = _read_json(HERE / "p5_stage3_summary.json")
    stage3_visualization_manifest = _read_json(HERE / "p5_stage3_visualization_manifest.json")
    stage3_summary_hash = hash_payload(stage3_summary)
    stage3_visualization_manifest_hash = hash_payload(stage3_visualization_manifest)
    stage4_summary = _read_json(SOURCE_EVIDENCE_DIR / "summary.json")
    stage4_results = stage4_summary["results"][mode]
    arrays = _load_npz(mode)
    results_strict_sha = hash_file(HERE / "results_strict.json")
    results_conditional_sha = hash_file(HERE / "results_conditional.json")
    source_manifest_digest = hash_payload(
        {
            "stage3_summary": stage3_summary_hash,
            "stage4_summary": stage4_summary["summary_hash"],
            "results_strict": results_strict_sha,
            "results_conditional": results_conditional_sha,
        }
    )
    qc_evidence = _qc_evidence(mode, stage4_summary, source_manifest_digest)
    task_spec = _task_spec(mode, {"summary_hash": stage4_summary["summary_hash"]})
    gaia_task_spec = _gaia_track_spec(mode, {"source_manifest_digest": source_manifest_digest, "summary_hash": stage4_summary["summary_hash"]})
    model_batch = _framework_batch(mode, task_spec, arrays, stage4_summary)
    conclusion = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "mode": mode,
        "state": "no_verified_gain",
        "allowed_states": [
            "verified_gain",
            "foundation_gain_only",
            "agent_signal_but_no_baseline_win",
            "no_verified_gain",
            "blocked_by_data",
        ],
        "fresh_blind": False,
        "prior_test_consumed": True,
        "evidence_class": "previously_seen_reusable_holdout",
        "strict_status": "passed" if mode == "strict" else "not_applicable",
        "conditional_status": "passed" if mode == "conditional" else "not_applicable",
        "strict_r2": float(stage4_results["metrics"]["strict_r2"]) if mode == "strict" else None,
        "conditional_r2": float(stage4_results["metrics"]["conditional_r2"]) if mode == "conditional" else None,
        "reason": (
            "strict lane is negative on R2 and conditional is a previously seen reusable holdout, "
            "so there is no fresh-blind verified gain"
        ),
    }
    lane_table = _lane_table(mode, stage4_summary, stage3_summary)
    sci_manifest = _sci_manifest(mode, stage4_summary, stage3_visualization_manifest)
    return ModeEvidence(
        mode=mode,
        track_spec=gaia_task_spec,
        framework_task_spec=task_spec,
        model_batch=model_batch,
        qc_evidence=qc_evidence,
        stage4_summary=stage4_summary,
        stage4_results=stage4_results,
        stage3_summary=stage3_summary,
        stage3_summary_hash=stage3_summary_hash,
        stage3_visualization_manifest=stage3_visualization_manifest,
        stage3_visualization_manifest_hash=stage3_visualization_manifest_hash,
        conclusion=conclusion,
        lane_table=lane_table,
        sci_manifest=sci_manifest,
    )


def write_package(output_root: Path) -> Mapping[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    mode_records = {mode: build_mode(mode) for mode in MODES}
    package = {
        "schema_version": SCHEMA_VERSION,
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "development_only": True,
        "fresh_blind": False,
        "modes": {},
    }
    for mode, evidence in mode_records.items():
        mode_dir = output_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        track_spec_path = mode_dir / "track_spec.json"
        framework_task_spec_path = mode_dir / "framework_task_spec.json"
        model_batch_path = mode_dir / "model_batch.json"
        qc_path = mode_dir / "qc_evidence.json"
        conclusion_path = mode_dir / "conclusion.json"
        lane_path = mode_dir / "lane_table.json"
        sci_path = mode_dir / "sci_manifest.json"
        atomic_write_json(track_spec_path, evidence.track_spec.to_dict())
        atomic_write_json(framework_task_spec_path, evidence.framework_task_spec.to_dict())
        atomic_write_json(
            model_batch_path,
            {
                "schema_version": SCHEMA_VERSION,
                "track_id": "reconstruction",
                "mode": mode,
                "task_spec_cache_key": evidence.track_spec.cache_key(),
                "model_batch_digest": hash_payload(asdict(evidence.model_batch)),
                "sample_count": len(evidence.model_batch.sample_ids),
                "voxel_count": int(np.asarray(_load_npz(mode)["truth"]).size),
                "feature_shape": list(np.asarray(_load_npz(mode)["indices_kji"]).shape),
                "metric_keys": list(evidence.framework_task_spec.primary_metrics + evidence.framework_task_spec.secondary_metrics),
                "agent_mode": "supervisory_qc_agent",
                "fresh_blind": False,
                "prior_test_consumed": True,
            },
        )
        atomic_write_json(qc_path, evidence.qc_evidence.to_dict())
        atomic_write_json(conclusion_path, evidence.conclusion)
        atomic_write_json(lane_path, evidence.lane_table)
        atomic_write_json(sci_path, evidence.sci_manifest)
        package["modes"][mode] = {
            "track_spec": f"{mode}/track_spec.json",
            "framework_task_spec": f"{mode}/framework_task_spec.json",
            "model_batch": f"{mode}/model_batch.json",
            "qc_evidence": f"{mode}/qc_evidence.json",
            "conclusion": f"{mode}/conclusion.json",
            "lane_table": f"{mode}/lane_table.json",
            "sci_manifest": f"{mode}/sci_manifest.json",
            "stage4_summary_sha256": evidence.stage4_summary["summary_hash"],
            "stage3_summary_sha256": evidence.stage3_summary_hash,
            "stage3_visualization_manifest_sha256": evidence.stage3_visualization_manifest_hash,
            "result_hash": evidence.stage4_results["result_hash"],
        }
    package["package_hash"] = hash_payload(package)
    atomic_write_json(output_root / "p6_private_package.json", package)
    atomic_write_json(
        output_root / "p6_private_artifact_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "track_id": "reconstruction",
            "root_seed": ROOT_SEED,
            "development_only": True,
            "fresh_blind": False,
            "files": [
                {
                    "path": f"p6_private_evidence/{mode}/track_spec.json",
                    "sha256": hash_file(output_root / mode / "track_spec.json"),
                }
                for mode in MODES
            ]
            + [
                {
                    "path": f"p6_private_evidence/{mode}/framework_task_spec.json",
                    "sha256": hash_file(output_root / mode / "framework_task_spec.json"),
                }
                for mode in MODES
            ]
            + [
                {
                    "path": f"p6_private_evidence/{mode}/model_batch.json",
                    "sha256": hash_file(output_root / mode / "model_batch.json"),
                }
                for mode in MODES
            ]
            + [
                {
                    "path": f"p6_private_evidence/{mode}/qc_evidence.json",
                    "sha256": hash_file(output_root / mode / "qc_evidence.json"),
                }
                for mode in MODES
            ]
            + [
                {
                    "path": f"p6_private_evidence/{mode}/conclusion.json",
                    "sha256": hash_file(output_root / mode / "conclusion.json"),
                }
                for mode in MODES
            ]
            + [
                {
                    "path": f"p6_private_evidence/{mode}/lane_table.json",
                    "sha256": hash_file(output_root / mode / "lane_table.json"),
                }
                for mode in MODES
            ]
            + [
                {
                    "path": f"p6_private_evidence/{mode}/sci_manifest.json",
                    "sha256": hash_file(output_root / mode / "sci_manifest.json"),
                }
                for mode in MODES
            ],
        },
    )
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE / "_outputs" / PACKAGE_DIRNAME)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(write_package(args.output_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
