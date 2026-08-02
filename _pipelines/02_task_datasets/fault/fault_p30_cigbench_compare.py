#!/usr/bin/env python3
"""Run CIG-Bench FaultPredictor and the audited fault logistic baseline on the P30 dev cube."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
P30_OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p30_3d_dev_gate"
P30_MANIFEST_PATH = P30_OUTPUT_ROOT / "manifest.json"
P30_SUBVOLUME_PATH = P30_OUTPUT_ROOT / "dev_subvolume.npz"
P30_SPLIT_MANIFEST_PATH = P30_OUTPUT_ROOT / "split_manifest.json"
AUDITED_V2_DIR = TRACK_DIR / "_outputs" / "runs" / "audited_v2"
AUDITED_V2_MODEL_PATH = AUDITED_V2_DIR / "baseline_model.joblib"
AUDITED_V2_METRICS_PATH = AUDITED_V2_DIR / "baseline_metrics.json"
OUTPUT_ROOT = P30_OUTPUT_ROOT / "cigbench_vs_baseline"
FIXED_OUTPUT_ROOT = P30_OUTPUT_ROOT / "cigbench_vs_baseline_threshold_fix"
COMPARISON_JSON_PATH = OUTPUT_ROOT / "comparison.json"
EVIDENCE_PATH = OUTPUT_ROOT / "evidence.md"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"

sys.path.insert(0, str(TRACK_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))


def git_head() -> str:
    import subprocess

    return (
        subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        )
        .strip()
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_threshold_by_f1(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, str, float]:
    from baseline import select_validation_threshold

    return select_validation_threshold(np.asarray(y_true, dtype=np.uint8), np.asarray(probabilities, dtype=np.float64), "auto")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    from baseline import binary_metrics as _binary_metrics

    return _binary_metrics(np.asarray(y_true, dtype=np.uint8), np.asarray(y_pred, dtype=bool))


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from baseline import probability_metrics as _probability_metrics

    return _probability_metrics(np.asarray(y_true, dtype=np.uint8), np.asarray(probabilities, dtype=np.float64))


@dataclass(frozen=True)
class FoldView:
    name: str
    inline_start: int
    inline_end: int
    seismic: np.ndarray
    positive_mask: np.ndarray
    unknown_mask: np.ndarray
    verified_background_mask: np.ndarray

    @property
    def score_mask(self) -> np.ndarray:
        return np.asarray(self.positive_mask | self.verified_background_mask, dtype=bool)


def parse_fold_views(dev: dict[str, np.ndarray], split_manifest: dict[str, Any]) -> list[FoldView]:
    seismic = np.asarray(dev["seismic"], dtype=np.float32)
    positive = np.asarray(dev["positive_mask"], dtype=bool)
    unknown = np.asarray(dev["unknown_mask"], dtype=bool)
    verified_background = np.asarray(dev["verified_background_mask"], dtype=bool)
    inline_axis = np.asarray(dev["iline"], dtype=np.int32)
    if seismic.ndim != 3:
        raise ValueError(f"expected 3D seismic volume, received shape={seismic.shape}")
    views: list[FoldView] = []
    for block in split_manifest["blocks"]:
        start, end = map(int, block["inline"])
        if start > end:
            raise ValueError(f"invalid inline range for {block['name']}: {block['inline']}")
        selection = np.where((inline_axis >= start) & (inline_axis <= end))[0]
        if not len(selection):
            raise ValueError(f"no inline indices found for fold {block['name']} in [{start}, {end}]")
        views.append(
            FoldView(
                name=str(block["name"]),
                inline_start=start,
                inline_end=end,
                seismic=seismic[:, selection, :],
                positive_mask=positive[:, selection, :],
                unknown_mask=unknown[:, selection, :],
                verified_background_mask=verified_background[:, selection, :],
            )
        )
    return views


def scoreable_truth_and_probabilities(truth: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth_flat = np.asarray(truth, dtype=np.uint8)[mask]
    probabilities_flat = np.asarray(probabilities, dtype=np.float64)[mask]
    if truth_flat.size == 0:
        raise ValueError("no scoreable voxels available after applying verified-background/positive mask")
    return truth_flat, probabilities_flat


def predict_cigbench_volume(seismic: np.ndarray, *, device: str) -> tuple[np.ndarray, dict[str, Any]]:
    from cig_bench.predictor.fault import FaultPredictor

    predictor = FaultPredictor(device=device)
    weight_path = Path(predictor.restore_path)
    started = time.perf_counter()
    probabilities, _ = predictor.predict(np.asarray(seismic, dtype=np.float32), threshold=0.0, resize_back=True)
    elapsed = time.perf_counter() - started
    if probabilities.shape != seismic.shape:
        raise ValueError(f"FaultPredictor output shape {probabilities.shape} does not match input {seismic.shape}")
    return probabilities.astype(np.float32), {
        "package": "cig_bench",
        "package_version": importlib.metadata.version("cig_bench"),
        "predictor_class": "cig_bench.predictor.fault.FaultPredictor",
        "device": str(device),
        "restore_path": str(weight_path),
        "restore_sha256": sha256_file(weight_path),
        "restore_bytes": weight_path.stat().st_size,
        "elapsed_seconds": elapsed,
    }


def predict_baseline_volume(seismic: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    sys.path.insert(0, str(TRACK_DIR))
    sys.path.insert(0, str(PROJECT_ROOT / "_code"))
    started = time.perf_counter()
    model = joblib.load(AUDITED_V2_MODEL_PATH)
    # Preserve the same [channel, spatial, temporal] contract the audited 2-D
    # baseline was trained on: each inline slice is converted to [1, xline, time].
    probabilities = np.empty_like(seismic, dtype=np.float32)
    for inline_index in range(seismic.shape[1]):
        patch = np.asarray(seismic[:, inline_index, :], dtype=np.float32).T[None, None]
        slice_probabilities = np.asarray(model.predict_batch(patch)[0], dtype=np.float32)
        if slice_probabilities.shape != (seismic.shape[2], seismic.shape[0]):
            raise ValueError(
                "baseline predict_batch returned unexpected shape "
                f"{slice_probabilities.shape}; expected {(seismic.shape[2], seismic.shape[0])}"
            )
        probabilities[:, inline_index, :] = slice_probabilities.T
    elapsed = time.perf_counter() - started
    return probabilities, {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "model_builder": "models.fault_local_logistic.build_model",
        "model_description": getattr(model, "description", type(model).__name__),
        "elapsed_seconds": elapsed,
        "note": "audited_v2 joblib checkpoint evaluated slice-by-slice over the P30 development cube",
    }


def evaluate_model_on_fold(
    fold: FoldView,
    probabilities: np.ndarray,
    *,
    threshold: float | None = None,
    threshold_source: str = "validation_max_f1",
) -> dict[str, Any]:
    score_mask = fold.score_mask
    truth = np.asarray(fold.positive_mask, dtype=np.uint8)
    truth_flat, probabilities_flat = scoreable_truth_and_probabilities(truth, probabilities, score_mask)
    if threshold is None:
        threshold, threshold_source, fit_f1 = select_threshold_by_f1(truth_flat, probabilities_flat)
    else:
        fit_f1 = float(binary_metrics(truth_flat, probabilities_flat >= threshold)["f1"])
    predictions = probabilities_flat >= threshold
    metrics = binary_metrics(truth_flat, predictions)
    metrics.update(probability_metrics(truth_flat, probabilities_flat))
    metrics["threshold"] = float(threshold)
    metrics["threshold_source"] = threshold_source
    metrics["fit_selected_f1"] = float(fit_f1)
    metrics["scoreable_voxels"] = int(truth_flat.size)
    metrics["positive_voxels"] = int(truth_flat.sum())
    metrics["unknown_voxels_excluded"] = int(np.asarray(fold.unknown_mask, dtype=bool).sum())
    return {
        "threshold": threshold,
        "threshold_source": threshold_source,
        "fit_selected_f1": fit_f1,
        "metrics": metrics,
        "truth": truth_flat,
        "probabilities": probabilities_flat,
    }


def aggregate_union(fit_result: dict[str, Any], guard_result: dict[str, Any], threshold: float) -> dict[str, Any]:
    truth = np.concatenate([fit_result["truth"], guard_result["truth"]]).astype(np.uint8)
    probabilities = np.concatenate([fit_result["probabilities"], guard_result["probabilities"]]).astype(np.float64)
    predictions = probabilities >= threshold
    metrics = binary_metrics(truth, predictions)
    metrics.update(probability_metrics(truth, probabilities))
    metrics["threshold"] = float(threshold)
    metrics["scoreable_voxels"] = int(truth.size)
    metrics["positive_voxels"] = int(truth.sum())
    return metrics


def compare_models(dev: dict[str, np.ndarray], split_manifest: dict[str, Any], *, device: str) -> dict[str, Any]:
    folds = parse_fold_views(dev, split_manifest)
    by_name = {fold.name: fold for fold in folds}
    if "fit" not in by_name or "guard" not in by_name:
        raise ValueError("P30 split manifest must contain fit and guard folds")
    fit_fold = by_name["fit"]
    guard_fold = by_name["guard"]

    cig_fit_probabilities, cig_info = predict_cigbench_volume(fit_fold.seismic, device=device)
    cig_guard_probabilities, _ = predict_cigbench_volume(guard_fold.seismic, device=device)
    baseline_fit_probabilities, baseline_info = predict_baseline_volume(fit_fold.seismic)
    baseline_guard_probabilities, _ = predict_baseline_volume(guard_fold.seismic)

    cig_fit = evaluate_model_on_fold(fit_fold, cig_fit_probabilities)
    baseline_fit = evaluate_model_on_fold(fit_fold, baseline_fit_probabilities)
    cig_guard = evaluate_model_on_fold(
        guard_fold,
        cig_guard_probabilities,
        threshold=float(cig_fit["threshold"]),
        threshold_source="fit_reused",
    )
    baseline_guard = evaluate_model_on_fold(
        guard_fold,
        baseline_guard_probabilities,
        threshold=float(baseline_fit["threshold"]),
        threshold_source="fit_reused",
    )

    cig_union = aggregate_union(cig_fit, cig_guard, cig_fit["threshold"])
    baseline_union = aggregate_union(baseline_fit, baseline_guard, baseline_fit["threshold"])

    return {
        "status": "READY",
        "reason_code": "CIG_BENCH_AND_BASELINE_DEV_COMPARISON_COMPLETED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track_id": "fault",
        "source_commit": git_head(),
        "p30_manifest": str(P30_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "p30_split_manifest": str(P30_SPLIT_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "p30_subvolume": str(P30_SUBVOLUME_PATH.relative_to(PROJECT_ROOT)),
        "p30_gate": {
            "status": load_json(P30_MANIFEST_PATH)["status"],
            "reason_code": load_json(P30_MANIFEST_PATH)["reason_code"],
            "data_gate_blocked": load_json(P30_MANIFEST_PATH)["data_gate_blocked"],
        },
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
                    "shape": list(fold.seismic.shape),
                    "scoreable_voxels": int(np.asarray(fold.score_mask, dtype=bool).sum()),
                    "positive_voxels": int(np.asarray(fold.positive_mask, dtype=bool).sum()),
                    "unknown_voxels": int(np.asarray(fold.unknown_mask, dtype=bool).sum()),
                }
                for fold in folds
            ],
        },
        "models": {
            "cig_bench_fault_predictor": {
                "info": cig_info,
                "fit": cig_fit["metrics"],
                "guard": cig_guard["metrics"],
                "development_union": cig_union,
            },
            "fault_local_logistic": {
                "info": baseline_info,
                "fit": baseline_fit["metrics"],
                "guard": baseline_guard["metrics"],
                "development_union": baseline_union,
            },
        },
        "comparison": {
            "primary_metric": "f1",
            "guard_delta": {
                "f1": float(cig_guard["metrics"]["f1"] - baseline_guard["metrics"]["f1"]),
                "precision": float(cig_guard["metrics"]["precision"] - baseline_guard["metrics"]["precision"]),
                "recall": float(cig_guard["metrics"]["recall"] - baseline_guard["metrics"]["recall"]),
                "iou": float(cig_guard["metrics"]["iou"] - baseline_guard["metrics"]["iou"]),
            },
            "fit_thresholds": {
                "cig_bench": float(cig_fit["threshold"]),
                "fault_local_logistic": float(baseline_fit["threshold"]),
            },
        },
        "baseline_reference": {
            "audited_v2_model_path": str(AUDITED_V2_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "audited_v2_model_sha256": sha256_file(AUDITED_V2_MODEL_PATH),
            "audited_v2_metrics_path": str(AUDITED_V2_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "audited_v2_metrics_sha256": sha256_file(AUDITED_V2_METRICS_PATH),
            "audited_v2_old_metrics": load_json(AUDITED_V2_METRICS_PATH)["test_metrics"],
        },
        "minimum_unblock_contract": [
            "The comparison stays inside the P30 continuous 3-D development volume.",
            "Scoring excludes unknown voxels and uses verified background only.",
            "Thresholds are selected on the fit fold and then reused for guard reporting.",
            "The audited_v2 baseline checkpoint is applied slice-by-slice without retraining.",
            "No frozen holdout/test.h5 path is opened or consumed.",
        ],
        "cig_bench": {
            "package": cig_info["package"],
            "package_version": cig_info["package_version"],
            "restore_path": cig_info["restore_path"],
            "restore_sha256": cig_info["restore_sha256"],
            "restore_bytes": cig_info["restore_bytes"],
        },
    }


def render_evidence(report: dict[str, Any]) -> str:
    cig = report["models"]["cig_bench_fault_predictor"]
    baseline = report["models"]["fault_local_logistic"]
    lines = [
        "# P30 CIG-Bench vs audited_v2 fault baseline comparison",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Source commit: `{report['source_commit']}`",
        f"- P30 manifest: `{report['p30_manifest']}`",
        f"- P30 split manifest: `{report['p30_split_manifest']}`",
        "",
        "## Gate and scope",
        "",
        f"- P30 gate status: `{report['p30_gate']['status']}`",
        f"- P30 gate reason: `{report['p30_gate']['reason_code']}`",
        f"- Development only: `{report['split']['development_only']}`",
        f"- Group isolated: `{report['split']['group_isolated']}`",
        f"- Frozen holdout accessed: `{report['split']['frozen_holdout_accessed']}`",
        "",
        "## Model provenance",
        "",
        f"- CIG-Bench package: `{report['cig_bench']['package']} {report['cig_bench']['package_version']}`",
        f"- CIG-Bench weight path: `{report['cig_bench']['restore_path']}`",
        f"- CIG-Bench weight sha256: `{report['cig_bench']['restore_sha256']}`",
        f"- audited_v2 baseline model: `{report['baseline_reference']['audited_v2_model_path']}`",
        f"- audited_v2 baseline model sha256: `{report['baseline_reference']['audited_v2_model_sha256']}`",
        "",
        "## Fit-selected thresholds",
        "",
        f"- CIG-Bench threshold: {cig['fit']['threshold']:.6f}",
        f"- Baseline threshold: {baseline['fit']['threshold']:.6f}",
        "",
        "## Guard metrics",
        "",
        f"- CIG-Bench: precision={cig['guard']['precision']:.6f}, recall={cig['guard']['recall']:.6f}, "
        f"f1={cig['guard']['f1']:.6f}, iou={cig['guard']['iou']:.6f}",
        f"- Baseline: precision={baseline['guard']['precision']:.6f}, recall={baseline['guard']['recall']:.6f}, "
        f"f1={baseline['guard']['f1']:.6f}, iou={baseline['guard']['iou']:.6f}",
        "",
        "## Guard deltas (CIG - baseline)",
        "",
        f"- F1 delta: {report['comparison']['guard_delta']['f1']:.6f}",
        f"- Precision delta: {report['comparison']['guard_delta']['precision']:.6f}",
        f"- Recall delta: {report['comparison']['guard_delta']['recall']:.6f}",
        f"- IoU delta: {report['comparison']['guard_delta']['iou']:.6f}",
        "",
        "## Fold summary",
        "",
    ]
    for fold in report["split"]["folds"]:
        lines.append(
            f"- {fold['name']}: inline={fold['inline']}, shape={fold['shape']}, "
            f"scoreable_voxels={fold['scoreable_voxels']}, positive_voxels={fold['positive_voxels']}, "
            f"unknown_voxels={fold['unknown_voxels']}"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "This run compares the CIG-Bench FaultPredictor and the audited_v2 logistic baseline on the same "
            "P30 continuous 3-D development asset. Unknown voxels are excluded; thresholds are selected on the "
            "fit fold and reused on guard. The saved comparison report records the exact measured metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any], output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "comparison.json"
    evidence_path = output_root / "evidence.md"
    manifest_path = output_root / "manifest.json"
    comparison_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence_path.write_text(render_evidence(report), encoding="utf-8")
    manifest = {
        "schema_version": "fault_p30_cigbench_compare/v1",
        "track_id": "fault",
        "source_commit": report["source_commit"],
        "generated_at": report["generated_at"],
        "runner": {
            "path": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(Path(__file__)),
        },
        "inputs": [
            {
                "path": report["p30_manifest"],
                "sha256": sha256_file(P30_MANIFEST_PATH),
                "role": "P30 development gate manifest",
            },
            {
                "path": report["p30_subvolume"],
                "sha256": sha256_file(P30_SUBVOLUME_PATH),
                "role": "P30 continuous 3-D development subvolume",
            },
            {
                "path": report["p30_split_manifest"],
                "sha256": sha256_file(P30_SPLIT_MANIFEST_PATH),
                "role": "P30 development split manifest",
            },
            {
                "path": report["baseline_reference"]["audited_v2_model_path"],
                "sha256": report["baseline_reference"]["audited_v2_model_sha256"],
                "role": "audited_v2 logistic checkpoint",
            },
            {
                "path": report["baseline_reference"]["audited_v2_metrics_path"],
                "sha256": report["baseline_reference"]["audited_v2_metrics_sha256"],
                "role": "audited_v2 baseline metrics",
            },
            {
                "path": report["cig_bench"]["restore_path"],
                "sha256": report["cig_bench"]["restore_sha256"],
                "role": "CIG-Bench FaultPredictor weights",
            },
        ],
        "outputs": [
            {
                "path": str(comparison_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(comparison_path),
                "role": "structured comparison report",
            },
            {
                "path": str(evidence_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(evidence_path),
                "role": "human evidence",
            },
        ],
        "status": report["status"],
        "reason_code": report["reason_code"],
        "comparison": {
            "primary_metric": report["comparison"]["primary_metric"],
            "guard_delta": report["comparison"]["guard_delta"],
            "fit_thresholds": report["comparison"]["fit_thresholds"],
        },
        "minimum_unblock_contract": report["minimum_unblock_contract"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "comparison_path": str(comparison_path.relative_to(PROJECT_ROOT)),
        "evidence_path": str(evidence_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "comparison_sha256": sha256_file(comparison_path),
        "evidence_sha256": sha256_file(evidence_path),
    }


def run(output_root: Path = OUTPUT_ROOT, device: str | None = None) -> dict[str, Any]:
    dev = np.load(P30_SUBVOLUME_PATH, allow_pickle=False)
    split_manifest = load_json(P30_SPLIT_MANIFEST_PATH)
    resolved_device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    report = compare_models(dev, split_manifest, device=resolved_device)
    output_info = write_outputs(report, output_root=output_root)
    return {"report": report, "outputs": output_info}


def run_fixed(output_root: Path = FIXED_OUTPUT_ROOT, device: str | None = None) -> dict[str, Any]:
    dev = np.load(P30_SUBVOLUME_PATH, allow_pickle=False)
    split_manifest = load_json(P30_SPLIT_MANIFEST_PATH)
    resolved_device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    report = compare_models(dev, split_manifest, device=resolved_device)
    output_info = write_outputs(report, output_root=output_root)
    return {"report": report, "outputs": output_info}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fixed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    if args.fixed:
        report = run_fixed(
            output_root=output_root or FIXED_OUTPUT_ROOT,
            device=args.device,
        )
    else:
        report = run(output_root=output_root or OUTPUT_ROOT, device=args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
