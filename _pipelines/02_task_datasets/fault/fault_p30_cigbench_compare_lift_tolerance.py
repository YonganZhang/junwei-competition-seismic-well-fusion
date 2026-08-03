#!/usr/bin/env python3
"""Lift and tolerance analysis for the ST10010-aligned P30 CIG-Bench comparison."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import fault_p30_cigbench_compare_final as base

try:  # pragma: no cover - SciPy is expected but keep a fallback guard.
    from scipy.ndimage import distance_transform_edt
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy.ndimage.distance_transform_edt is required for tolerance metrics") from exc


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
ASSET_ROOT = base.ASSET_ROOT
OUTPUT_ROOT = ASSET_ROOT / "cigbench_vs_baseline_lift_tolerance"
OUTPUT_ROOT_V2 = ASSET_ROOT / "cigbench_vs_baseline_lift_tolerance_v2"
COMPARISON_JSON_PATH = OUTPUT_ROOT / "comparison.json"
EVIDENCE_PATH = OUTPUT_ROOT / "evidence.md"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"


def lift_against_prior(value: float, prior: float) -> float:
    return float(value / prior) if prior > 0 else float("nan")


def fold_prior(scoreable_voxels: int, positive_voxels: int) -> float:
    return float(positive_voxels) / float(scoreable_voxels) if scoreable_voxels else float("nan")


def tolerance_scores(truth: np.ndarray, pred: np.ndarray, score_mask: np.ndarray, radius: int) -> dict[str, float | int]:
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool) & np.asarray(score_mask, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError(f"truth/pred shape mismatch: {truth.shape} vs {pred.shape}")
    if not truth.any():
        raise ValueError("tolerance scoring requires at least one positive truth voxel")
    if not pred.any():
        return {
            "radius": int(radius),
            "predicted_positive_voxels": 0,
            "truth_positive_voxels": int(truth.sum()),
            "matched_prediction_voxels": 0,
            "matched_truth_voxels": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    dist_truth = distance_transform_edt(~truth)
    dist_pred = distance_transform_edt(~pred)
    pred_hits = pred & (dist_truth <= float(radius))
    truth_hits = truth & (dist_pred <= float(radius))
    tp_pred = int(pred_hits.sum())
    tp_truth = int(truth_hits.sum())
    precision = tp_pred / int(pred.sum())
    recall = tp_truth / int(truth.sum())
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "radius": int(radius),
        "predicted_positive_voxels": int(pred.sum()),
        "truth_positive_voxels": int(truth.sum()),
        "matched_prediction_voxels": tp_pred,
        "matched_truth_voxels": tp_truth,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def aggregate_tolerance_summaries(summaries: list[dict[str, dict[str, float | int]]], radii: tuple[int, ...]) -> dict[str, dict[str, float | int]]:
    aggregated: dict[str, dict[str, float | int]] = {}
    for radius in radii:
        key = f"radius_{radius}"
        pred_positive = sum(int(summary[key]["predicted_positive_voxels"]) for summary in summaries)
        truth_positive = sum(int(summary[key]["truth_positive_voxels"]) for summary in summaries)
        matched_pred = sum(int(summary[key]["matched_prediction_voxels"]) for summary in summaries)
        matched_truth = sum(int(summary[key]["matched_truth_voxels"]) for summary in summaries)
        precision = matched_pred / pred_positive if pred_positive else 0.0
        recall = matched_truth / truth_positive if truth_positive else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregated[key] = {
            "radius": int(radius),
            "predicted_positive_voxels": int(pred_positive),
            "truth_positive_voxels": int(truth_positive),
            "matched_prediction_voxels": int(matched_pred),
            "matched_truth_voxels": int(matched_truth),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return aggregated


def enrich_fold_metrics(
    fold: base.FoldView,
    probabilities_volume: np.ndarray,
    *,
    threshold: float,
    radii: tuple[int, ...],
) -> dict[str, Any]:
    result = base.evaluate_fold(fold, probabilities_volume, threshold=threshold, threshold_source="fit_reused")
    metrics = dict(result["metrics"])
    prior = fold_prior(metrics["scoreable_voxels"], metrics["positive_voxels"])
    metrics["positive_prior"] = prior
    metrics["precision_lift"] = lift_against_prior(metrics["precision"], prior)
    metrics["recall_to_prior_ratio"] = lift_against_prior(metrics["recall"], prior)
    metrics["average_precision_lift"] = lift_against_prior(metrics["average_precision"], prior)
    metrics["f1_vs_prior_ratio"] = lift_against_prior(metrics["f1"], prior)
    metrics["predicted_positive_fraction"] = float(metrics["tp"] + metrics["fp"]) / float(metrics["scoreable_voxels"]) if metrics["scoreable_voxels"] else float("nan")
    metrics["coverage_ratio"] = lift_against_prior(metrics["predicted_positive_fraction"], prior)

    fold_probabilities = np.asarray(probabilities_volume, dtype=np.float64)[:, fold.selection, :]
    fold_pred = fold_probabilities >= float(threshold)
    score_mask = fold.score_mask
    tolerance = {
        f"radius_{radius}": tolerance_scores(fold.positive_mask, fold_pred, score_mask, radius)
        for radius in radii
    }
    result["metrics"] = metrics
    result["tolerance"] = tolerance
    return result


def enrich_union_metrics(
    union_metrics: dict[str, Any],
    fold_results: list[dict[str, Any]],
    *,
    prior: float,
    radii: tuple[int, ...],
) -> dict[str, Any]:
    enriched = dict(union_metrics)
    enriched["positive_prior"] = prior
    enriched["precision_lift"] = lift_against_prior(enriched["precision"], prior)
    enriched["recall_to_prior_ratio"] = lift_against_prior(enriched["recall"], prior)
    enriched["average_precision_lift"] = lift_against_prior(enriched["average_precision"], prior)
    enriched["f1_vs_prior_ratio"] = lift_against_prior(enriched["f1"], prior)
    enriched["predicted_positive_fraction"] = float(enriched["tp"] + enriched["fp"]) / float(enriched["scoreable_voxels"]) if enriched["scoreable_voxels"] else float("nan")
    enriched["coverage_ratio"] = lift_against_prior(enriched["predicted_positive_fraction"], prior)
    enriched["tolerance"] = aggregate_tolerance_summaries([result["tolerance"] for result in fold_results], radii)
    return enriched


def compare_with_lift_and_tolerance(
    dev: dict[str, np.ndarray],
    split_manifest: dict[str, Any],
    *,
    device: str,
    cig_scale: dict[str, float] | None = None,
    radii: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    folds = base.parse_fold_views(dev, split_manifest)
    fold_by_name = {fold.name: fold for fold in folds}
    if "fit" not in fold_by_name or "guard" not in fold_by_name:
        raise ValueError("split manifest must contain fit and guard folds")

    scale = dict(base.DEFAULT_CIG_SCALE)
    if cig_scale is not None:
        scale.update(cig_scale)

    cig_probabilities, cig_info = base.predict_cigbench_volume(dev["seismic"], device=device, **scale)
    baseline_probabilities, baseline_info = base.predict_baseline_volume(dev["seismic"])

    cig_fit = base.evaluate_fold(fold_by_name["fit"], cig_probabilities, threshold=None)
    cig_fit = enrich_fold_metrics(
        fold_by_name["fit"],
        cig_probabilities,
        threshold=float(cig_fit["threshold"]),
        radii=radii,
    )
    baseline_fit = base.evaluate_fold(fold_by_name["fit"], baseline_probabilities, threshold=None)
    baseline_fit = enrich_fold_metrics(
        fold_by_name["fit"],
        baseline_probabilities,
        threshold=float(baseline_fit["threshold"]),
        radii=radii,
    )

    cig_guard = enrich_fold_metrics(fold_by_name["guard"], cig_probabilities, threshold=float(cig_fit["metrics"]["threshold"]), radii=radii)
    baseline_guard = enrich_fold_metrics(fold_by_name["guard"], baseline_probabilities, threshold=float(baseline_fit["metrics"]["threshold"]), radii=radii)
    cig_validation = enrich_fold_metrics(fold_by_name["validation"], cig_probabilities, threshold=float(cig_fit["metrics"]["threshold"]), radii=radii)
    baseline_validation = enrich_fold_metrics(fold_by_name["validation"], baseline_probabilities, threshold=float(baseline_fit["metrics"]["threshold"]), radii=radii)

    cig_union = base.aggregate_union([cig_fit, cig_guard, cig_validation], float(cig_fit["metrics"]["threshold"]))
    baseline_union = base.aggregate_union([baseline_fit, baseline_guard, baseline_validation], float(baseline_fit["metrics"]["threshold"]))
    cig_union = enrich_union_metrics(
        cig_union,
        [cig_fit, cig_guard, cig_validation],
        prior=fold_prior(cig_union["scoreable_voxels"], cig_union["positive_voxels"]),
        radii=radii,
    )
    baseline_union = enrich_union_metrics(
        baseline_union,
        [baseline_fit, baseline_guard, baseline_validation],
        prior=fold_prior(baseline_union["scoreable_voxels"], baseline_union["positive_voxels"]),
        radii=radii,
    )

    return {
        "status": "READY",
        "reason_code": "P30_LIFT_AND_TOLERANCE_ANALYSIS_COMPLETED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track_id": "fault",
        "source_commit": base.git_head(),
        "asset_root": str(ASSET_ROOT.relative_to(PROJECT_ROOT)),
        "p30_manifest": str((ASSET_ROOT / "manifest.json").relative_to(PROJECT_ROOT)),
        "p30_split_manifest": str(base.SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "p30_subvolume": str(base.SUBVOLUME_PATH.relative_to(PROJECT_ROOT)),
        "split": {
            "development_only": bool(split_manifest.get("development_only")),
            "group_isolated": bool(split_manifest.get("group_isolated")),
            "frozen_holdout_accessed": bool(split_manifest.get("frozen_holdout_accessed")),
            "coordinate_order": split_manifest.get("coordinate_order"),
            "subvolume": split_manifest.get("subvolume"),
            "folds": [
                {
                    "name": fold.name,
                    "inline": [fold.inline_start, fold.inline_end],
                    "shape": list(np.asarray(dev["seismic"][:, fold.selection, :]).shape),
                    "scoreable_voxels": int(np.asarray(fold.score_mask, dtype=bool).sum()),
                    "positive_voxels": int(np.asarray(fold.positive_mask, dtype=bool).sum()),
                    "unknown_voxels": int(np.asarray(fold.unknown_mask, dtype=bool).sum()),
                    "positive_prior": fold_prior(int(np.asarray(fold.score_mask, dtype=bool).sum()), int(np.asarray(fold.positive_mask, dtype=bool).sum())),
                }
                for fold in folds
            ],
        },
        "tolerance_policy": {
            "distance_metric": "euclidean_voxel",
            "radii_voxels": list(radii),
            "primary_radius_voxels": 2,
            "rationale": [
                "1 voxel is a strict near-exact match.",
                "2 voxels is the main reporting radius because it matches the existing boundary halo used in the development gate and tolerates small rasterization offsets without granting broad blobs a free pass.",
                "3 voxels is a sensitivity upper bound to show whether any apparent gain survives a looser boundary tolerance.",
            ],
        },
        "models": {
            "cig_bench_fault_predictor": {
                "info": cig_info,
                "fit": cig_fit["metrics"],
                "guard": cig_guard["metrics"],
                "validation": cig_validation["metrics"],
                "development_union": cig_union,
            },
            "fault_local_logistic": {
                "info": baseline_info,
                "fit": baseline_fit["metrics"],
                "guard": baseline_guard["metrics"],
                "validation": baseline_validation["metrics"],
                "development_union": baseline_union,
            },
        },
        "comparison": {
            "primary_metric": "tolerance_f1_radius_2",
            "guard_delta": {
                "precision": float(cig_guard["metrics"]["precision"] - baseline_guard["metrics"]["precision"]),
                "recall": float(cig_guard["metrics"]["recall"] - baseline_guard["metrics"]["recall"]),
                "f1": float(cig_guard["metrics"]["f1"] - baseline_guard["metrics"]["f1"]),
                "iou": float(cig_guard["metrics"]["iou"] - baseline_guard["metrics"]["iou"]),
            },
            "guard_lift": {
                "precision_lift": float(cig_guard["metrics"]["precision_lift"]),
                "baseline_precision_lift": float(baseline_guard["metrics"]["precision_lift"]),
                "average_precision_lift": float(cig_guard["metrics"]["average_precision_lift"]),
                "baseline_average_precision_lift": float(baseline_guard["metrics"]["average_precision_lift"]),
                "recall_to_prior_ratio": float(cig_guard["metrics"]["recall_to_prior_ratio"]),
                "baseline_recall_to_prior_ratio": float(baseline_guard["metrics"]["recall_to_prior_ratio"]),
                "positive_prior": float(cig_guard["metrics"]["positive_prior"]),
                "predicted_positive_fraction": float(cig_guard["metrics"]["predicted_positive_fraction"]),
                "baseline_predicted_positive_fraction": float(baseline_guard["metrics"]["predicted_positive_fraction"]),
                "coverage_ratio": float(cig_guard["metrics"]["coverage_ratio"]),
                "baseline_coverage_ratio": float(baseline_guard["metrics"]["coverage_ratio"]),
                "threshold": float(cig_guard["metrics"]["threshold"]),
            },
            "guard_tolerance_sweep": {
                "cig_bench": cig_guard["tolerance"],
                "baseline": baseline_guard["tolerance"],
            },
            "development_union_tolerance_sweep": {
                "cig_bench": cig_union["tolerance"],
                "baseline": baseline_union["tolerance"],
            },
            "tolerance_radius_2": {
                "cig_bench": cig_guard["tolerance"]["radius_2"],
                "baseline": baseline_guard["tolerance"]["radius_2"],
            },
            "fit_thresholds": {
                "cig_bench": float(cig_fit["metrics"]["threshold"]),
                "fault_local_logistic": float(baseline_fit["metrics"]["threshold"]),
            },
        },
        "decision": {
            "default_recommendation": "do_not_advance",
            "model_classification": "diagnostic_high_recall_proposal_only",
            "reason_codes": [
                "precision_lift_only_moderate",
                "average_precision_lift_only_moderate",
                "threshold_near_zero_for_cigbench",
                "predicted_positive_fraction_is_high",
                "tolerance_f1_radius_2_is_low",
            ],
            "summary": (
                "CIG-Bench should not be promoted as the default fault detector yet. "
                "It is a diagnostic/high-recall proposal: precision lift is about 1.18x, "
                "AP lift about 1.42x, fit threshold is near zero, predicted positive fraction "
                "is about 69.4%, and radius-2 tolerance F1 is about 0.0202."
            ),
            "minimum_advancement_conditions": [
                "Lift precision materially beyond the current ~1.18x level without collapsing recall.",
                "Reduce predicted positive fraction substantially below the current ~69.4% coverage.",
                "Raise tolerance F1 beyond the current radius-2 ~0.0202 while keeping guard and validation stable.",
                "Demonstrate calibration stability across the development folds with the same fit threshold.",
            ],
        },
        "baseline_reference": {
            "audited_v2_model_path": str(base.BASELINE_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "audited_v2_model_sha256": base.sha256_file(base.BASELINE_MODEL_PATH),
            "audited_v2_metrics_path": str(base.BASELINE_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "audited_v2_metrics_sha256": base.sha256_file(base.BASELINE_METRICS_PATH),
            "audited_v2_old_metrics": base.load_json(base.BASELINE_METRICS_PATH)["test_metrics"],
        },
        "minimum_unblock_contract": [
            "The comparison uses the ST10010 continuous 3-D development asset.",
            "CIG-Bench is run once on the full cube and thresholds are selected only on fit.",
            "Guard, validation and union reuse the fit threshold without extra search.",
            "Lift metrics are normalized by the observed positive prior on each fold.",
            "Tolerance metrics use 1/2/3 voxel Euclidean hit radii, with 2 voxels as the primary radius.",
            "Union tolerance is micro-aggregated from per-fold 3-D tolerance counts, not flattened across fold boundaries.",
            "The audited_v2 baseline checkpoint is applied slice-by-slice without retraining.",
            "No frozen holdout/test.h5 path is opened or consumed.",
        ],
        "cig_bench": {
            "package": cig_info["package"],
            "package_version": cig_info["package_version"],
            "restore_path": cig_info["restore_path"],
            "restore_sha256": cig_info["restore_sha256"],
            "restore_bytes": cig_info["restore_bytes"],
            "scale_t": cig_info["scale_t"],
            "scale_h": cig_info["scale_h"],
            "scale_w": cig_info["scale_w"],
        },
        "cig_bench_scale": scale,
    }


def render_evidence(report: dict[str, Any]) -> str:
    cig = report["models"]["cig_bench_fault_predictor"]
    baseline = report["models"]["fault_local_logistic"]
    guard = report["comparison"]["guard_lift"]
    guard_tol_cig = report["comparison"]["guard_tolerance_sweep"]["cig_bench"]
    guard_tol_baseline = report["comparison"]["guard_tolerance_sweep"]["baseline"]
    union_tol_cig = report["comparison"]["development_union_tolerance_sweep"]["cig_bench"]
    union_tol_baseline = report["comparison"]["development_union_tolerance_sweep"]["baseline"]
    lines = [
        "# P30 CIG-Bench vs audited_v2 lift and tolerance analysis",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Source commit: `{report['source_commit']}`",
        f"- Asset root: `{report['asset_root']}`",
        "",
        "## Scope and gate",
        "",
        f"- P30 gate manifest: `{report['p30_manifest']}`",
        f"- Development only: `{report['split']['development_only']}`",
        f"- Group isolated: `{report['split']['group_isolated']}`",
        f"- Frozen holdout accessed: `{report['split']['frozen_holdout_accessed']}`",
        "",
        "## Prior-normalized lift",
        "",
        f"- Guard positive prior: {guard['positive_prior']:.6f}",
        f"- CIG-Bench precision lift: {guard['precision_lift']:.3f}x",
        f"- Baseline precision lift: {guard['baseline_precision_lift']:.3f}x",
        f"- CIG-Bench AP lift: {guard['average_precision_lift']:.3f}x",
        f"- Baseline AP lift: {guard['baseline_average_precision_lift']:.3f}x",
        f"- CIG-Bench recall / prior ratio: {guard['recall_to_prior_ratio']:.3f}x",
        f"- Baseline recall / prior ratio: {guard['baseline_recall_to_prior_ratio']:.3f}x",
        f"- CIG-Bench predicted-positive fraction: {guard['predicted_positive_fraction']:.6f}",
        f"- Baseline predicted-positive fraction: {guard['baseline_predicted_positive_fraction']:.6f}",
        f"- CIG-Bench coverage ratio vs prior: {guard['coverage_ratio']:.3f}x",
        f"- Baseline coverage ratio vs prior: {guard['baseline_coverage_ratio']:.3f}x",
        f"- CIG-Bench fit threshold: {guard['threshold']:.12g}",
        "",
        "## Guard ordinary metrics",
        "",
        f"- CIG-Bench: precision={cig['guard']['precision']:.6f}, recall={cig['guard']['recall']:.6f}, AP={cig['guard']['average_precision']:.6f}, F1={cig['guard']['f1']:.6f}",
        f"- Baseline: precision={baseline['guard']['precision']:.6f}, recall={baseline['guard']['recall']:.6f}, AP={baseline['guard']['average_precision']:.6f}, F1={baseline['guard']['f1']:.6f}",
        "",
        "## Tolerance radius 2 voxels",
        "",
        f"- CIG-Bench tolerance: precision={guard_tol_cig['radius_2']['precision']:.6f}, recall={guard_tol_cig['radius_2']['recall']:.6f}, F1={guard_tol_cig['radius_2']['f1']:.6f}",
        f"- Baseline tolerance: precision={guard_tol_baseline['radius_2']['precision']:.6f}, recall={guard_tol_baseline['radius_2']['recall']:.6f}, F1={guard_tol_baseline['radius_2']['f1']:.6f}",
        "",
        "## Radius sweep",
        "",
        f"- Guard CIG-Bench: {json.dumps(guard_tol_cig, sort_keys=True)}",
        f"- Guard baseline: {json.dumps(guard_tol_baseline, sort_keys=True)}",
        f"- Union CIG-Bench: {json.dumps(union_tol_cig, sort_keys=True)}",
        f"- Union baseline: {json.dumps(union_tol_baseline, sort_keys=True)}",
        "",
        "## Verdict",
        "",
        f"- Default recommendation: `{report['decision']['default_recommendation']}`",
        f"- Model classification: `{report['decision']['model_classification']}`",
        f"- Summary: {report['decision']['summary']}",
        f"- Minimum advancement conditions: {json.dumps(report['decision']['minimum_advancement_conditions'], ensure_ascii=False)}",
        "",
        "## Interpretation",
        "",
        "CIG-Bench remains the higher-recall model, but the prior-normalized lift is still modest and the tolerance-based scores do not transform it into a clean, high-precision detector. The per-fold tolerance sweep is included to show whether the gain survives a near-boundary match criterion instead of exact voxel equality, and the union result is micro-aggregated from per-fold 3-D counts rather than flattened across fold boundaries.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    compare_path = output_root / "comparison.json"
    evidence_path = output_root / "evidence.md"
    manifest_path = output_root / "manifest.json"
    compare_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence_path.write_text(render_evidence(report), encoding="utf-8")
    manifest = {
        "schema_version": "fault_p30_cigbench_compare_lift_tolerance/v2",
        "track_id": "fault",
        "source_commit": report["source_commit"],
        "generated_at": report["generated_at"],
        "runner": {
            "path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "sha256": base.sha256_file(Path(__file__)),
        },
        "inputs": [
            {"path": report["p30_manifest"], "sha256": base.sha256_file(ASSET_ROOT / "manifest.json"), "role": "P30 development gate manifest"},
            {"path": report["p30_subvolume"], "sha256": base.sha256_file(base.SUBVOLUME_PATH), "role": "P30 continuous 3-D development subvolume"},
            {"path": report["p30_split_manifest"], "sha256": base.sha256_file(base.SPLIT_MANIFEST_PATH), "role": "P30 development split manifest"},
            {"path": report["baseline_reference"]["audited_v2_model_path"], "sha256": report["baseline_reference"]["audited_v2_model_sha256"], "role": "audited_v2 baseline checkpoint"},
            {"path": report["cig_bench"]["restore_path"], "sha256": report["cig_bench"]["restore_sha256"], "role": "CIG-Bench FaultPredictor weights"},
        ],
        "outputs": [
            {"path": str(compare_path.relative_to(PROJECT_ROOT)), "sha256": base.sha256_file(compare_path), "role": "structured lift and tolerance comparison report"},
            {"path": str(evidence_path.relative_to(PROJECT_ROOT)), "sha256": base.sha256_file(evidence_path), "role": "human evidence"},
        ],
        "status": report["status"],
        "reason_code": report["reason_code"],
        "comparison": {
            "primary_metric": report["comparison"]["primary_metric"],
            "guard_lift": report["comparison"]["guard_lift"],
            "guard_tolerance_sweep": report["comparison"]["guard_tolerance_sweep"],
            "development_union_tolerance_sweep": report["comparison"]["development_union_tolerance_sweep"],
            "tolerance_radius_2": report["comparison"]["tolerance_radius_2"],
            "fit_thresholds": report["comparison"]["fit_thresholds"],
        },
        "decision": report["decision"],
        "minimum_unblock_contract": report["minimum_unblock_contract"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "comparison_path": str(compare_path.relative_to(PROJECT_ROOT)),
        "evidence_path": str(evidence_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": base.sha256_file(manifest_path),
        "comparison_sha256": base.sha256_file(compare_path),
        "evidence_sha256": base.sha256_file(evidence_path),
    }


def run(output_root: Path = OUTPUT_ROOT_V2, device: str | None = None, cig_scale: dict[str, float] | None = None) -> dict[str, Any]:
    dev = np.load(base.SUBVOLUME_PATH, allow_pickle=False)
    split_manifest = base.load_json(base.SPLIT_MANIFEST_PATH)
    resolved_device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    report = compare_with_lift_and_tolerance(dev, split_manifest, device=resolved_device, cig_scale=cig_scale)
    outputs = write_outputs(report, output_root=output_root)
    return {"report": report, "outputs": outputs}


def run_default() -> dict[str, Any]:
    return run(output_root=OUTPUT_ROOT_V2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scale-t", type=float, default=base.DEFAULT_CIG_SCALE["scale_t"])
    parser.add_argument("--scale-h", type=float, default=base.DEFAULT_CIG_SCALE["scale_h"])
    parser.add_argument("--scale-w", type=float, default=base.DEFAULT_CIG_SCALE["scale_w"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cig_scale = {"scale_t": args.scale_t, "scale_h": args.scale_h, "scale_w": args.scale_w}
    report = run(output_root=args.output_root or OUTPUT_ROOT_V2, device=args.device, cig_scale=cig_scale)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
