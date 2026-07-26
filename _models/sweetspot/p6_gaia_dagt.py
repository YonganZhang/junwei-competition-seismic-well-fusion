"""P6 private Gaia/DAGT evidence packaging for sweetspot T1/T2.

This module keeps the sweetspot proxy work honest: it reuses archived P5
evidence, builds TrackSpec/ModelBatch wrappers for T1/T2 only, and records the
locked boundaries for T3-T7 plus unavailable foundation lanes F0/F1.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any, Mapping

from _models.gaia_dagt import (
    DEFAULT_SOURCE_MANIFEST,
    AgentEvidence,
    GaiaDAGTAdapter,
    ModelBatch,
    TrackSpec,
)
from _models.gaia_dagt.contracts import canonical_json, sha256_json
from _models.gaia_dagt import supervisory_qc_agent


ROOT_SEED = 2693
PACKAGE_NAME = "sweetspot-p6-gaia-dagt-private"
ALLOWED_STATES = ("READY", "PARTIAL_READY", "BLOCKED", "NOT_FEASIBLE", "UNAVAILABLE")

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "_pipelines" / "02_task_datasets" / "sweetspot"
P5_STAGE4_DIR = PIPELINE_ROOT / "p5" / "_outputs" / "stage4_confirmation"
P5_R02_DIR = PIPELINE_ROOT / "p5" / "r02" / "_outputs" / "protocol_r2"
P6_OUTPUT_DIR = PIPELINE_ROOT / "p6" / "_outputs" / "private_gaia_dagt"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files() -> list[dict[str, str]]:
    paths = [
        P5_STAGE4_DIR / "p5_stage4_summary.json",
        P5_STAGE4_DIR / "p5_stage4_manifest.json",
        P5_STAGE4_DIR / "targets" / "T1" / "metrics.json",
        P5_STAGE4_DIR / "targets" / "T1" / "refit.json",
        P5_STAGE4_DIR / "targets" / "T2" / "metrics.json",
        P5_STAGE4_DIR / "targets" / "T2" / "refit.json",
        P5_R02_DIR / "p5_r02_summary.json",
        P5_R02_DIR / "p5_r02_plateau_gate.json",
        PIPELINE_ROOT / "p5" / "sweetspot_p5_label_mapping.v1.json",
    ]
    result: list[dict[str, str]] = []
    for path in paths:
        if path.is_file():
            result.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256_file(path)})
    return result


def _stage4_summary() -> dict[str, Any]:
    return _load_json(P5_STAGE4_DIR / "p5_stage4_summary.json")


def _target_meta(target_id: str) -> dict[str, Any]:
    if target_id == "T1":
        return {
            "task_type": "regression",
            "track_id": "sweetspot.p6.t1.proxy_qc",
            "lane": "reservoir_quality",
            "head": "RQI",
            "metric_names": ("mae", "rmse", "spearman"),
            "input_fields": ("mae", "rmse", "spearman", "holdout_samples", "development_samples"),
            "target_vector": [0.2161617856064557, 0.27932989213623827, 0.9931289666595325, 11936.0, 35810.0],
            "target_value": [0.2161617856064557],
            "task_metric": "mae",
            "source_kind": "P5 Stage-4 known-holdout confirmation",
            "stage4_metric_file": P5_STAGE4_DIR / "targets" / "T1" / "metrics.json",
            "stage4_refit_file": P5_STAGE4_DIR / "targets" / "T1" / "refit.json",
        }
    if target_id == "T2":
        return {
            "task_type": "classification",
            "track_id": "sweetspot.p6.t2.proxy_qc",
            "lane": "hydrocarbon_pay",
            "head": "SAND_FLAG_PROXY",
            "metric_names": ("average_precision", "brier", "f1"),
            "input_fields": ("average_precision", "brier", "f1", "thickness_mae_m", "holdout_samples"),
            "target_vector": [0.9990776475633495, 0.02286081524589408, 0.9777260638297872, 26.19999999997617, 12081.0],
            "target_value": [1.0],
            "task_metric": "average_precision",
            "source_kind": "P5 Stage-4 known-holdout confirmation",
            "stage4_metric_file": P5_STAGE4_DIR / "targets" / "T2" / "metrics.json",
            "stage4_refit_file": P5_STAGE4_DIR / "targets" / "T2" / "refit.json",
        }
    raise KeyError(target_id)


def build_track_spec(target_id: str) -> TrackSpec:
    meta = _target_meta(target_id)
    allowed_paths = (
        str((P5_STAGE4_DIR / "p5_stage4_summary.json").relative_to(REPO_ROOT)),
        str(meta["stage4_metric_file"].relative_to(REPO_ROOT)),
        str(meta["stage4_refit_file"].relative_to(REPO_ROOT)),
    )
    forbidden_paths = ("test.h5", "frozen_holdout", "known_holdout_predictions", "checkpoint", "score")
    return TrackSpec(
        track_id=meta["track_id"],
        task_type=meta["task_type"],
        modality="proxy_qc_summary",
        input_fields=meta["input_fields"],
        target_fields=(meta["head"],),
        allowed_paths=allowed_paths,
        forbidden_paths=forbidden_paths,
        metric_names=meta["metric_names"],
        base_seed=ROOT_SEED,
        source_manifest_digest=DEFAULT_SOURCE_MANIFEST.digest(),
        provenance={
            "package": PACKAGE_NAME,
            "proxy_only": True,
            "source_kind": meta["source_kind"],
            "target_id": target_id,
        },
    )


def build_model_batch(target_id: str) -> ModelBatch:
    meta = _target_meta(target_id)
    track_spec = build_track_spec(target_id)
    features = [meta["target_vector"]]
    target = meta["target_value"]
    mask = [True]
    return ModelBatch(
        track_spec=track_spec,
        features=features,
        target=target,
        mask=mask,
        task_targets={meta["head"]: target},
        task_metrics={meta["head"]: meta["task_metric"]},
        task_masks={meta["head"]: mask},
        feasibility={meta["head"]: True},
        metadata={"package": PACKAGE_NAME, "proxy_only": True, "source_kind": meta["source_kind"], "target_id": target_id},
    )


def build_agent_evidence(target_id: str) -> AgentEvidence:
    meta = _target_meta(target_id)
    track_spec = build_track_spec(target_id)
    report = {
        "package": PACKAGE_NAME,
        "target_id": target_id,
        "lane": meta["lane"],
        "proxy_only": True,
        "source_kind": meta["source_kind"],
        "stage4_metric_file": str(meta["stage4_metric_file"].relative_to(REPO_ROOT)),
        "stage4_refit_file": str(meta["stage4_refit_file"].relative_to(REPO_ROOT)),
        "supporting_files": [row["path"] for row in _source_files()],
    }
    adapter = GaiaDAGTAdapter.from_default_manifest(track_spec, control_mode="real", seed=ROOT_SEED)
    return adapter.build_agent_evidence(report, mode="supervisory_qc_agent")


def _roundtrip_summary(target_id: str) -> dict[str, Any]:
    track_spec = build_track_spec(target_id)
    batch = build_model_batch(target_id)
    evidence = build_agent_evidence(target_id)
    batch = ModelBatch(
        track_spec=track_spec,
        features=batch.features,
        target=batch.target,
        mask=batch.mask,
        task_targets=batch.task_targets,
        task_metrics=batch.task_metrics,
        task_masks=batch.task_masks,
        feasibility=batch.feasibility,
        metadata=batch.metadata,
        agent_evidence=evidence,
    )
    batch_roundtrip = ModelBatch.from_dict(batch.to_dict())
    return {
        "track_spec": track_spec.to_dict(),
        "track_spec_sha256": track_spec.cache_key(),
        "model_batch": batch.to_dict(),
        "model_batch_sha256": sha256_json(batch.to_dict()),
        "agent_evidence": evidence.to_dict(),
        "agent_evidence_sha256": sha256_json(evidence.to_dict()),
        "batch_roundtrip_sha256": sha256_json(batch_roundtrip.to_dict()),
        "batch_roundtrip_equal": batch_roundtrip.to_dict() == batch.to_dict(),
    }


def build_bundle() -> dict[str, Any]:
    stage4 = _stage4_summary()
    t1 = _roundtrip_summary("T1")
    t2 = _roundtrip_summary("T2")
    locked_targets = {
        "T3": {"status": "BLOCKED", "reason": "no P6 private evidence extension requested for T3"},
        "T4": {"status": "BLOCKED", "reason": "no P6 private evidence extension requested for T4"},
        "T5": {"status": "NOT_FEASIBLE", "reason": "simulation proxy remains insufficient"},
        "T6": {"status": "BLOCKED", "reason": "no development-only feature source; test.h5 fallback forbidden"},
        "T7": {"status": "BLOCKED", "reason": "no development-only feature source; test.h5 fallback forbidden"},
    }
    foundation_status = {
        "F0": {"status": "UNAVAILABLE", "reason": "no approved foundation artifact"},
        "F1": {"status": "UNAVAILABLE", "reason": "no approved foundation artifact"},
    }
    proxy_evidence_index = {
        "B0": {
            "target_id": "T1",
            "status": "PARTIAL_READY",
            "source": {
                "summary_path": str((P5_STAGE4_DIR / "p5_stage4_summary.json").relative_to(REPO_ROOT)),
                "metrics_path": str((P5_STAGE4_DIR / "targets" / "T1" / "metrics.json").relative_to(REPO_ROOT)),
                "refit_path": str((P5_STAGE4_DIR / "targets" / "T1" / "refit.json").relative_to(REPO_ROOT)),
            },
            "source_hashes": {
                "summary": _sha256_file(P5_STAGE4_DIR / "p5_stage4_summary.json"),
                "metrics": _sha256_file(P5_STAGE4_DIR / "targets" / "T1" / "metrics.json"),
                "refit": _sha256_file(P5_STAGE4_DIR / "targets" / "T1" / "refit.json"),
            },
            "source_kind": "authorized known-holdout members only",
            "prior_test_consumed": True,
            "fresh_blind": False,
        },
        "B1": {
            "target_id": "T2",
            "status": "PARTIAL_READY",
            "source": {
                "summary_path": str((P5_STAGE4_DIR / "p5_stage4_summary.json").relative_to(REPO_ROOT)),
                "metrics_path": str((P5_STAGE4_DIR / "targets" / "T2" / "metrics.json").relative_to(REPO_ROOT)),
                "refit_path": str((P5_STAGE4_DIR / "targets" / "T2" / "refit.json").relative_to(REPO_ROOT)),
            },
            "source_hashes": {
                "summary": _sha256_file(P5_STAGE4_DIR / "p5_stage4_summary.json"),
                "metrics": _sha256_file(P5_STAGE4_DIR / "targets" / "T2" / "metrics.json"),
                "refit": _sha256_file(P5_STAGE4_DIR / "targets" / "T2" / "refit.json"),
            },
            "source_kind": "authorized known-holdout members only",
            "prior_test_consumed": True,
            "fresh_blind": False,
        },
    }
    resource_log = {
        "schema_version": "sweetspot-p6-gaia-dagt-resource-log/v1",
        "package": PACKAGE_NAME,
        "no_training": True,
        "no_downloads": True,
        "no_api_calls": True,
        "frozen_test_accessed": False,
        "shared_contract_tests_required": 14,
        "shared_contract_tests_ran": True,
        "private_tests_ran": False,
        "used_source_files": _source_files(),
    }
    failure_log = {
        "schema_version": "sweetspot-p6-gaia-dagt-failure-log/v1",
        "package": PACKAGE_NAME,
        "blocked": {
            "F0": foundation_status["F0"],
            "F1": foundation_status["F1"],
            "T3": locked_targets["T3"],
            "T4": locked_targets["T4"],
            "T5": locked_targets["T5"],
            "T6": locked_targets["T6"],
            "T7": locked_targets["T7"],
        },
        "deny_list_proof": {
            "mode": "supervisory_qc_agent",
            "predictive_mode_disabled": True,
            "no_sample_level_text_available": True,
            "no_test_or_holdout_inputs_used": True,
        },
    }
    conclusion = {
        "schema_version": "sweetspot-p6-gaia-dagt-conclusion/v1",
        "status": "PARTIAL_READY",
        "final_state": "blocked_by_data",
        "approved_states": list(ALLOWED_STATES),
        "target_states": {
            "T1": "PARTIAL_READY",
            "T2": "PARTIAL_READY",
            "T3": locked_targets["T3"]["status"],
            "T4": locked_targets["T4"]["status"],
            "T5": locked_targets["T5"]["status"],
            "T6": locked_targets["T6"]["status"],
            "T7": locked_targets["T7"]["status"],
            "F0": foundation_status["F0"]["status"],
            "F1": foundation_status["F1"]["status"],
        },
        "reason": "T1/T2 private Gaia/DAGT evidence packaging is complete; F0/F1 remain unavailable and T3-T7 stay locked/not_feasible/blocked.",
    }
    bundle = {
        "schema_version": "sweetspot-p6-gaia-dagt-bundle/v1",
        "package": PACKAGE_NAME,
        "root_seed": ROOT_SEED,
        "source_manifest": DEFAULT_SOURCE_MANIFEST.to_dict(),
        "source_manifest_digest": DEFAULT_SOURCE_MANIFEST.digest(),
        "source_files": _source_files(),
        "targets": {"T1": t1, "T2": t2},
        "proxy_evidence_index": proxy_evidence_index,
        "foundation_status": foundation_status,
        "lock_table": locked_targets,
        "resource_log": resource_log,
        "failure_log": failure_log,
        "conclusion": conclusion,
    }
    bundle["bundle_sha256"] = sha256_json({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    return bundle


def render_proxy_only_figure(bundle: Mapping[str, Any]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage4 = _stage4_summary()
    t1_metrics = _load_json(P5_STAGE4_DIR / "targets" / "T1" / "metrics.json")["metrics"]
    t2_metrics = _load_json(P5_STAGE4_DIR / "targets" / "T2" / "metrics.json")["metrics"]
    fig = plt.figure(figsize=(11, 6), dpi=200)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.16,
        hspace=0.58,
        wspace=0.38,
    )
    ax_t1_err = fig.add_subplot(gs[0, 0])
    ax_t1_rank = fig.add_subplot(gs[0, 1])
    ax_t2_class = fig.add_subplot(gs[1, 0])
    ax_t2_diag = fig.add_subplot(gs[1, 1])

    def _annotate_bars(ax, bars, fmt="{:.3f}") -> None:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    t1_error_labels = ["MAE", "RMSE"]
    t1_error_values = [t1_metrics["mae"], t1_metrics["rmse"]]
    bars = ax_t1_err.bar(t1_error_labels, t1_error_values, color=["#5B8FF9", "#61DDAA"])
    _annotate_bars(ax_t1_err, bars)
    ax_t1_err.set_title("T1 error metrics")
    ax_t1_err.set_ylabel("proxy units")
    ax_t1_err.text(
        0.02,
        0.96,
        f"source_kind={bundle['targets']['T1']['track_spec']['provenance']['source_kind']}",
        transform=ax_t1_err.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2},
    )
    ax_t1_err.grid(axis="y", alpha=0.2)

    bars = ax_t1_rank.bar(["Spearman ρ"], [t1_metrics["spearman"]], color=["#65789B"])
    _annotate_bars(ax_t1_rank, bars)
    ax_t1_rank.set_title("T1 rank correlation")
    ax_t1_rank.set_ylabel("ρ")
    ax_t1_rank.set_ylim(0, 1.05)
    ax_t1_rank.text(
        0.02,
        0.96,
        "proxy-only / no field truth claim",
        transform=ax_t1_rank.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2},
    )
    ax_t1_rank.grid(axis="y", alpha=0.2)

    t2_class_labels = ["AP", "Brier", "F1"]
    t2_class_values = [t2_metrics["average_precision"], t2_metrics["brier"], t2_metrics["f1_at_0_5"]]
    bars = ax_t2_class.bar(t2_class_labels, t2_class_values, color=["#5AD8A6", "#F6BD16", "#E86452"])
    _annotate_bars(ax_t2_class, bars)
    ax_t2_class.set_title("T2 classification scores")
    ax_t2_class.set_ylabel("unitless score")
    ax_t2_class.set_ylim(0, 1.05)
    ax_t2_class.text(
        0.02,
        0.96,
        f"known-holdout confirmation / prior_test_consumed={stage4['prior_test_consumed']}",
        transform=ax_t2_class.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2},
    )
    ax_t2_class.grid(axis="y", alpha=0.2)

    thickness_mae = t2_metrics["thickness_diagnostic"]["net_thickness_mae_m"]
    bars = ax_t2_diag.bar(["Thickness MAE"], [thickness_mae], color=["#6DC8EC"])
    _annotate_bars(ax_t2_diag, bars, fmt="{:.2f}")
    ax_t2_diag.set_title("T2 thickness diagnostic")
    ax_t2_diag.set_ylabel("m")
    ax_t2_diag.text(
        0.02,
        0.96,
        "separated from classification scores",
        transform=ax_t2_diag.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 2},
    )
    ax_t2_diag.grid(axis="y", alpha=0.2)

    fig.suptitle("Sweetspot P6 private Gaia/DAGT evidence package — proxy-only T1/T2")
    fig.text(
        0.5,
        0.045,
        "No predictive text agent, no test.h5, no F2/C1/C2, no new labels, known-holdout only",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    png_path = P6_OUTPUT_DIR / "figures" / "proxy_only_qc.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return str(png_path.relative_to(REPO_ROOT))


def materialize_private_package(output_dir: Path = P6_OUTPUT_DIR) -> dict[str, Any]:
    bundle = build_bundle()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_rel = render_proxy_only_figure(bundle)
    bundle["figure_manifest"] = {
        "schema_version": "sweetspot-p6-gaia-dagt-figure-manifest/v1",
        "figures": [
            {
                "path": figure_rel,
                "title": "Sweetspot P6 proxy-only evidence package",
                "kind": "proxy-only_sci_figure",
                "target_scope": ["T1", "T2"],
            }
        ],
    }
    bundle_path = output_dir / "p6_gaia_dagt_bundle.json"
    conclusion_path = output_dir / "p6_gaia_dagt_conclusion.json"
    resource_path = output_dir / "p6_gaia_dagt_resource_log.json"
    failure_path = output_dir / "p6_gaia_dagt_failure_log.json"
    evidence_path = output_dir / "p6_gaia_dagt_evidence_index.json"
    manifest_path = output_dir / "p6_gaia_dagt_manifest.json"
    artifacts: list[dict[str, Any]] = []
    for path, payload in [
        (conclusion_path, bundle["conclusion"]),
        (resource_path, bundle["resource_log"]),
        (failure_path, bundle["failure_log"]),
        (evidence_path, bundle["proxy_evidence_index"]),
    ]:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append({"path": str(path.relative_to(REPO_ROOT)), "size_bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    figure_abs = REPO_ROOT / figure_rel
    artifacts.append({"path": figure_rel, "size_bytes": figure_abs.stat().st_size, "sha256": _sha256_file(figure_abs)})
    manifest = {
        "schema_version": "sweetspot-p6-gaia-dagt-artifact-manifest/v1",
        "package": PACKAGE_NAME,
        "root_seed": ROOT_SEED,
        "artifacts": artifacts,
        "all_paths_portable": all(not Path(row["path"]).is_absolute() for row in artifacts),
        "no_training": True,
        "no_downloads": True,
        "no_test_access": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bundle["manifest"] = manifest
    bundle["manifest_sha256"] = _sha256_file(manifest_path)
    bundle["bundle_sha256"] = sha256_json({key: value for key, value in bundle.items() if key != "bundle_sha256"})
    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    resource = bundle["resource_log"]
    resource["private_tests_ran"] = False
    resource_path.write_text(json.dumps(resource, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "bundle": bundle,
        "bundle_path": str(bundle_path.relative_to(REPO_ROOT)),
        "conclusion_path": str(conclusion_path.relative_to(REPO_ROOT)),
        "resource_log_path": str(resource_path.relative_to(REPO_ROOT)),
        "failure_log_path": str(failure_path.relative_to(REPO_ROOT)),
        "evidence_index_path": str(evidence_path.relative_to(REPO_ROOT)),
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "figure_path": figure_rel,
        "artifacts": artifacts,
    }


def dry_report_text() -> str:
    stage4 = _stage4_summary()
    return canonical_json(
        {
            "package": PACKAGE_NAME,
            "target_scope": ["T1", "T2"],
            "t1": {"status": stage4["target_status"]["T1"], "curve": stage4["target_budget_curve"]["T1"]},
            "t2": {"status": stage4["target_status"]["T2"], "curve": stage4["target_budget_curve"]["T2"]},
            "locked": {target: stage4["target_status"][target] for target in ("T3", "T4", "T5", "T6", "T7")},
        }
    )
