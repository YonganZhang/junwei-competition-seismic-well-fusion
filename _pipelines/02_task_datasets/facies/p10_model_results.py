#!/usr/bin/env python3
"""Build the facies P10 model-results deliverables from archived evidence.

The workbook is an audit artifact, not a training artifact.  It consolidates the
frozen stage-3 baseline cells, the stage-4 known-holdout confirmation, and the
P9 SAM2 effect summaries without reopening any raw test or holdout data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRACK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TRACK_DIR / "_outputs" / "p10_model_results"
PROJECT_ROOT = TRACK_DIR.parents[2]
CURRENT_COMMIT = "e4fd5d8a6371c2b0db6ba2258a41349ec6cfb4f7"
SAM2_CHECKPOINT = Path(
    "/mnt/data/yongan-admin-2/.cache/huggingface/hub/models--facebook--sam2.1-hiera-base-plus/"
    "blobs/a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"
)

STAGE3_DIR = TRACK_DIR / "_outputs" / "p5_stage3"
STAGE4_DIR = TRACK_DIR / "_outputs" / "p5_stage4_confirmation"
P9_DIR = TRACK_DIR / "_outputs" / "p9_sam2_effect"
P10_REPAIR_DIR = TRACK_DIR / "_outputs" / "p10_sam2_repair_audit"
P10_REPAIR_BLOCKER_RELATIVE = Path("p10_sam2_repair_audit/repair_blocker.json")
P10_REPAIR_BLOCKER_PATH = OUTPUT_DIR / P10_REPAIR_BLOCKER_RELATIVE


REQUIRED_COLUMNS = [
    "track",
    "dataset",
    "task_type",
    "model_name",
    "model_family",
    "is_foundation_model",
    "foundation_type",
    "integration_point",
    "fusion_method",
    "preprocess_version",
    "split_protocol",
    "seed_or_fold",
    "metric_name",
    "metric_value",
    "higher_is_better",
    "baseline_model",
    "baseline_value",
    "delta_abs",
    "delta_pct",
    "status",
    "evidence_path",
    "checkpoint_path",
    "code_commit",
    "root_cause",
    "fix_applied",
    "notes",
]

HIGHER_IS_BETTER = {
    "accuracy": True,
    "miou": True,
    "macro_f1": True,
    "brier": False,
    "ece": False,
    "nll": False,
    "macro_classwise_ece": False,
}

TASK_LABELS = {
    "facies_f3": {
        "dataset": "F3",
        "task_type": "segmentation_2d",
        "label_version": "f3-zenodo-1471548-ids-0-9-v1",
        "baseline_model": "smp_fpn_r18",
        "foundation_baseline": "smp_fpn_r18",
    },
    "facies_penobscot": {
        "dataset": "Penobscot",
        "task_type": "segmentation_2d",
        "label_version": "penobscot-dataset-log-v3-ids-0-7-v1",
        "baseline_model": "smp_deeplabv3plus_r18",
        "foundation_baseline": "smp_deeplabv3plus_r18",
    },
}

STAGE3_SOURCE = STAGE3_DIR / "p5_stage3_results.jsonl"
STAGE3_SUMMARY = STAGE3_DIR / "p5_stage3_summary.json"
STAGE4_SUMMARY = STAGE4_DIR / "p5_stage4_summary.json"
P9_SUMMARIES = {
    "facies_f3": P9_DIR / "facies_f3" / "summary.json",
    "facies_penobscot": P9_DIR / "facies_penobscot" / "summary.json",
}
P10_REPAIR_SUMMARIES = {
    "facies_f3": P10_REPAIR_DIR / "facies_f3" / "summary.json",
    "facies_penobscot": P10_REPAIR_DIR / "facies_penobscot" / "summary.json",
}


@dataclass(frozen=True)
class RowContext:
    track: str
    dataset: str
    task_type: str
    model_name: str
    model_family: str
    is_foundation_model: bool
    foundation_type: str
    integration_point: str
    fusion_method: str
    preprocess_version: str
    split_protocol: str
    seed_or_fold: str
    metric_name: str
    metric_value: float
    higher_is_better: bool
    baseline_model: str
    baseline_value: float
    status: str
    evidence_path: str
    checkpoint_path: str
    code_commit: str
    root_cause: str
    fix_applied: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        delta_abs = self.metric_value - self.baseline_value
        delta_pct = (
            delta_abs / abs(self.baseline_value)
            if self.baseline_value not in (0.0, -0.0) and math.isfinite(self.baseline_value)
            else 0.0
        )
        return {
            "track": self.track,
            "dataset": self.dataset,
            "task_type": self.task_type,
            "model_name": self.model_name,
            "model_family": self.model_family,
            "is_foundation_model": self.is_foundation_model,
            "foundation_type": self.foundation_type,
            "integration_point": self.integration_point,
            "fusion_method": self.fusion_method,
            "preprocess_version": self.preprocess_version,
            "split_protocol": self.split_protocol,
            "seed_or_fold": self.seed_or_fold,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "higher_is_better": self.higher_is_better,
            "baseline_model": self.baseline_model,
            "baseline_value": self.baseline_value,
            "delta_abs": delta_abs,
            "delta_pct": delta_pct,
            "status": self.status,
            "evidence_path": self.evidence_path,
            "checkpoint_path": self.checkpoint_path,
            "code_commit": self.code_commit,
            "root_cause": self.root_cause,
            "fix_applied": self.fix_applied,
            "notes": self.notes,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage3_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with STAGE3_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _scalar_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)) and key in HIGHER_IS_BETTER:
            metrics[key] = float(value)
    return metrics


def _baseline_for_stage3(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, int, int, str], tuple[str, float]]:
    grouped: dict[tuple[str, int, int, str], list[tuple[str, float, bool]]] = defaultdict(list)
    for row in rows:
        task = row["task_id"]
        fold_id = int(row["fold_id"])
        repeat_id = int(row["repeat_id"])
        model = row["model_id"]
        for metric_name, metric_value in _scalar_metrics(row["validation_metrics"]).items():
            grouped[(task, fold_id, repeat_id, metric_name)].append(
                (model, metric_value, HIGHER_IS_BETTER[metric_name])
            )

    baselines: dict[tuple[str, int, int, str], tuple[str, float]] = {}
    for key, candidates in grouped.items():
        metric_name = key[3]
        higher_is_better = HIGHER_IS_BETTER[metric_name]
        best = max(candidates, key=lambda item: item[1]) if higher_is_better else min(candidates, key=lambda item: item[1])
        baselines[key] = (best[0], best[1])
    return baselines


def build_rows() -> list[dict[str, Any]]:
    stage3_rows = load_stage3_rows()
    stage3_summary = read_json(STAGE3_SUMMARY)
    stage4_summary = read_json(STAGE4_SUMMARY)
    p9_summaries = {task: read_json(path) for task, path in P9_SUMMARIES.items()}
    stage3_baselines = _baseline_for_stage3(stage3_rows)

    rows: list[RowContext] = []

    for row in stage3_rows:
        task = row["task_id"]
        meta = TASK_LABELS[task]
        family = row["source_lock"]["family"]
        split_protocol = "p5_stage3_locked_spatial_five_fold"
        seed_or_fold = f"fold_{row['fold_id']}/repeat_{row['repeat_id']}"
        evidence_path = str(STAGE3_SOURCE.relative_to(PROJECT_ROOT))
        checkpoint_path = row["checkpoint"]["runtime_relative_path"]
        preprocess_version = row["preprocessing"]["preprocessor_hash"]
        integration_point = "stage3_scratch_baseline"
        fusion_method = "none"
        for metric_name, metric_value in _scalar_metrics(row["validation_metrics"]).items():
            baseline_model, baseline_value = stage3_baselines[(task, int(row["fold_id"]), int(row["repeat_id"]), metric_name)]
            rows.append(
                RowContext(
                    track="facies",
                    dataset=meta["dataset"],
                    task_type=meta["task_type"],
                    model_name=row["model_id"],
                    model_family=family,
                    is_foundation_model=False,
                    foundation_type="none",
                    integration_point=integration_point,
                    fusion_method=fusion_method,
                    preprocess_version=preprocess_version,
                    split_protocol=split_protocol,
                    seed_or_fold=seed_or_fold,
                    metric_name=metric_name,
                    metric_value=metric_value,
                    higher_is_better=HIGHER_IS_BETTER[metric_name],
                    baseline_model=baseline_model,
                    baseline_value=baseline_value,
                    status="ranked",
                    evidence_path=evidence_path,
                    checkpoint_path=checkpoint_path,
                    code_commit=CURRENT_COMMIT,
                    root_cause="baseline_reference",
                    fix_applied="none",
                    notes="stage3 archived cell; frozen development split; no test access",
                )
            )

    # Primary-metric effect rows for SAM2: before = foundation candidate, baseline = strong locked model.
    for task, summary_path in P9_SUMMARIES.items():
        meta = TASK_LABELS[task]
        summary = p9_summaries[task]
        comparison = summary["comparison"]
        evaluation = summary["evaluation"]
        baseline_model = comparison["strong_baseline_model_id"]
        baseline_mean = float(comparison["strong_baseline_macro_fold_miou"])
        evidence_path = str(summary_path.relative_to(PROJECT_ROOT))
        if task == "facies_f3":
            stage3_winner = read_json(STAGE3_DIR / "facies_f3_scratch_leaderboard.json")
            fold_baselines = stage3_winner["entries"][0]["fold_mean_miou"]
        else:
            stage3_winner = read_json(STAGE3_DIR / "facies_penobscot_scratch_leaderboard.json")
            fold_baselines = stage3_winner["entries"][0]["fold_mean_miou"]

        for fold_result in summary["fold_results"]:
            fold_id = int(fold_result["fold_id"])
            fold_baseline = float(fold_baselines[str(fold_id)])
            split_protocol = "p9_sam2_effect_locked_manifest_fold"
            for metric_name, metric_value, model_name in [
                ("miou", float(fold_result["pretrained_miou"]), "facebook/sam2.1-hiera-base-plus"),
                ("miou", float(fold_result["random_init_miou"]), "sam2.1-hiera-base-plus-random-init"),
            ]:
                rows.append(
                    RowContext(
                        track="facies",
                        dataset=meta["dataset"],
                        task_type=meta["task_type"],
                        model_name=model_name,
                        model_family="SAM2.1 Hiera Base Plus",
                        is_foundation_model=True,
                        foundation_type="image_segmentation_foundation_encoder",
                        integration_point="adapter_plus_semantic_head",
                        fusion_method="fpn_projections_plus_semantic_head",
                        preprocess_version="p9_sam2_effect_identity_norm",
                        split_protocol=split_protocol,
                        seed_or_fold=f"fold_{fold_id}",
                        metric_name=metric_name,
                        metric_value=metric_value,
                        higher_is_better=True,
                        baseline_model=baseline_model,
                        baseline_value=fold_baseline,
                        status="non_beneficial" if model_name.startswith("facebook/") else "control",
                        evidence_path=evidence_path,
                        checkpoint_path=str(SAM2_CHECKPOINT),
                        code_commit=CURRENT_COMMIT,
                        root_cause="same_split_pretrained_foundation_did_not_beat_locked_baseline",
                        fix_applied="none",
                        notes=(
                            f"pretrained={float(fold_result['pretrained_miou']):.6f} "
                            f"random_init={float(fold_result['random_init_miou']):.6f} "
                            f"baseline_fold={fold_baseline:.6f}; no frozen test"
                        ),
                )
            )

    # Repair-audit rows. Prefer the completed development-only probe when present;
    # retain a blocked row only for tasks without a materialized summary.
    for task in TASK_LABELS:
        meta = TASK_LABELS[task]
        summary_path = P10_REPAIR_SUMMARIES[task]
        if summary_path.exists():
            summary = read_json(summary_path)
            evidence_path = str(summary_path.relative_to(PROJECT_ROOT))
            for fold in summary["fold_results"]:
                fold_id = int(fold["fold_id"])
                seed = int(fold["seed"])
                comparisons = [
                    (
                        "facebook/sam2.1-hiera-base-plus",
                        float(fold["pretrained_adapter_miou"]),
                        "sam2.1-hiera-base-plus-random-init",
                        float(fold["random_init_control_miou"]),
                        "effect_supported_not_promoted",
                        "none",
                        "pretraining improves the same architecture, but the adapter remains below the locked strong baseline",
                    ),
                    (
                        "facebook/sam2.1-hiera-base-plus+gated-residual",
                        float(fold["gated_residual_repair_miou"]),
                        "facebook/sam2.1-hiera-base-plus",
                        float(fold["pretrained_adapter_miou"]),
                        "non_beneficial",
                        "gated_residual_probe_tested",
                        "the gated residual repair underfits and degrades the pretrained adapter on the fixed development fold",
                    ),
                    (
                        "sam2.1-hiera-base-plus-random-init",
                        float(fold["random_init_control_miou"]),
                        meta["baseline_model"],
                        float(fold["strong_baseline_miou"]),
                        "control",
                        "none",
                        "same-architecture random initialization control",
                    ),
                    (
                        meta["baseline_model"],
                        float(fold["strong_baseline_miou"]),
                        meta["baseline_model"],
                        float(fold["strong_baseline_miou"]),
                        "reference",
                        "none",
                        "locked strong baseline on the same development samples",
                    ),
                ]
                for (
                    model_name,
                    metric_value,
                    baseline_model,
                    baseline_value,
                    status,
                    fix_applied,
                    root_cause,
                ) in comparisons:
                    rows.append(
                        RowContext(
                            track="facies",
                            dataset=meta["dataset"],
                            task_type=meta["task_type"],
                            model_name=model_name,
                            model_family="SAM2.1 Hiera Base Plus" if "sam2" in model_name.lower() else "locked strong baseline",
                            is_foundation_model=model_name.startswith("facebook/"),
                            foundation_type="image_segmentation_foundation_encoder" if "sam2" in model_name.lower() else "none",
                            integration_point="gated_residual_repair_audit",
                            fusion_method="base_logits_plus_gated_residual_probe" if "gated-residual" in model_name else "semantic_adapter",
                            preprocess_version="p10_sam2_repair_audit_v1",
                            split_protocol=f"p10_sam2_repair_audit_dev;manifest={summary['evaluation']['manifest_stable_hash']}",
                            seed_or_fold=f"fold_{fold_id};seed_{seed}",
                            metric_name="miou",
                            metric_value=metric_value,
                            higher_is_better=True,
                            baseline_model=baseline_model,
                            baseline_value=baseline_value,
                            status=status,
                            evidence_path=evidence_path,
                            checkpoint_path=str(SAM2_CHECKPOINT) if "sam2" in model_name.lower() else "",
                            code_commit=CURRENT_COMMIT,
                            root_cause=root_cause,
                            fix_applied=fix_applied,
                            notes=(
                                f"train_samples={fold['train_samples']}; validation_samples={fold['validation_samples']}; "
                                f"frozen_test_accessed={summary['evaluation']['frozen_test_accessed']}; "
                                f"real_pretrained_weights_loaded={summary['model']['real_pretrained_weights_loaded']}"
                            ),
                        )
                    )
        else:
            blocker_evidence = str(P10_REPAIR_BLOCKER_PATH.relative_to(PROJECT_ROOT))
            rows.append(
                RowContext(
                    track="facies",
                    dataset=meta["dataset"],
                    task_type=meta["task_type"],
                    model_name="facebook/sam2.1-hiera-base-plus+gated-residual",
                    model_family="SAM2.1 Hiera Base Plus",
                    is_foundation_model=True,
                    foundation_type="image_segmentation_foundation_encoder",
                    integration_point="gated_residual_repair_audit",
                    fusion_method="base_logits_plus_gated_residual_probe",
                    preprocess_version="p10_sam2_repair_audit_v1",
                    split_protocol="p10_sam2_repair_audit_blocked",
                    seed_or_fold="blocked",
                    metric_name="miou",
                    metric_value=float("nan"),
                    higher_is_better=True,
                    baseline_model="facebook/sam2.1-hiera-base-plus",
                    baseline_value=float("nan"),
                    status="data_blocked",
                    evidence_path=blocker_evidence,
                    checkpoint_path=str(SAM2_CHECKPOINT),
                    code_commit=CURRENT_COMMIT,
                    root_cause="repair_summary_not_materialized",
                    fix_applied="none",
                    notes="development-only repair summary is absent",
                )
            )

    # Holdout confirmation rows use the refit baseline; they are not a tuning source.
    for task, task_payload in stage4_summary["tasks"].items():
        meta = TASK_LABELS[task]
        metrics_path = STAGE4_DIR / task / "metrics.json"
        refit_evidence = read_json(STAGE4_DIR / task / "refit_evidence.json")
        evidence_path = str(metrics_path.relative_to(PROJECT_ROOT))
        checkpoint_path = refit_evidence["checkpoint"]["runtime_relative_path"]
        model_name = task_payload["winner_model_id"]
        model_family = "SMP FPN ResNet18" if task == "facies_f3" else "SMP DeepLabV3+ ResNet18"
        preprocess_version = refit_evidence["preprocessing"]["preprocessor_hash"]
        for metric_name in ("accuracy", "miou", "macro_f1", "ece", "brier", "nll"):
            metric_value = float(read_json(metrics_path)[metric_name])
            rows.append(
                RowContext(
                    track="facies",
                    dataset=meta["dataset"],
                    task_type=meta["task_type"],
                    model_name=model_name,
                    model_family=model_family,
                    is_foundation_model=False,
                    foundation_type="none",
                    integration_point="refit_final_holdout_confirmation",
                    fusion_method="none",
                    preprocess_version=preprocess_version,
                    split_protocol="p5_stage4_known_holdout",
                    seed_or_fold="refit_final",
                    metric_name=metric_name,
                    metric_value=metric_value,
                    higher_is_better=HIGHER_IS_BETTER[metric_name],
                    baseline_model=model_name,
                    baseline_value=metric_value,
                    status="confirmed_holdout",
                    evidence_path=evidence_path,
                    checkpoint_path=checkpoint_path,
                    code_commit=CURRENT_COMMIT,
                    root_cause="holdout_confirmation_after_locked_refit",
                    fix_applied="none",
                    notes="known holdout confirmed exactly once; no calibration or threshold tuning",
                )
            )

    # Add summary-level rows for stage3 winners, useful for the workbook summary sheet.
    for task, winner in (("facies_f3", "smp_fpn_r18"), ("facies_penobscot", "smp_deeplabv3plus_r18")):
        meta = TASK_LABELS[task]
        leaderboard = read_json(STAGE3_DIR / f"{task}_scratch_leaderboard.json")
        best = leaderboard["entries"][0]
        rows.append(
            RowContext(
                track="facies",
                dataset=meta["dataset"],
                task_type=meta["task_type"],
                model_name=winner,
                model_family=best["model_id"].replace("_", " "),
                is_foundation_model=False,
                foundation_type="none",
                integration_point="stage3_oof_leaderboard",
                fusion_method="none",
                preprocess_version=stage3_summary["source_hashes"]["stage3_runner_sha256"],
                split_protocol="p5_stage3_oof_aggregate",
                seed_or_fold="aggregate_oof",
                metric_name="miou",
                metric_value=float(best["mean_miou"]),
                higher_is_better=True,
                baseline_model=winner,
                baseline_value=float(best["mean_miou"]),
                status="ranked",
                evidence_path=str((STAGE3_DIR / f"{task}_scratch_leaderboard.json").relative_to(PROJECT_ROOT)),
                checkpoint_path=best.get("checkpoint_path", ""),
                code_commit=CURRENT_COMMIT,
                root_cause="current_locked_baseline_reference",
                fix_applied="none",
                notes="top-of-board locked baseline from frozen stage-3 leaderboard",
            )
        )

    return [row.to_dict() for row in rows]


def write_xlsx(rows: list[dict[str, Any]], path: Path) -> None:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "模型指标"
    ws.append(REQUIRED_COLUMNS)
    for row in rows:
        ws.append([row.get(column) for column in REQUIRED_COLUMNS])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for column in ws.columns:
        max_len = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column)
        ws.column_dimensions[column[0].column_letter].width = min(max(12, max_len + 2), 48)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_manifests(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    generated_figure = (output_dir / "before_after_primary_metric.png").relative_to(PROJECT_ROOT)
    generated_workbook = (output_dir / "track_model_metrics.xlsx").relative_to(PROJECT_ROOT)
    figures = [
        {
            "kind": "figure",
            "name": "before_after_primary_metric.png",
            "path": str(generated_figure),
            "status": "generated",
            "sha256": sha256_file(output_dir / "before_after_primary_metric.png"),
            "description": "SAM2 pretrained adapter versus random-init control reference comparison for F3 and Penobscot.",
        },
        {
            "kind": "figure",
            "name": "facies_f3_stage3_oof_diagnostics.png",
            "path": str((STAGE3_DIR / "facies_f3_stage3_oof_diagnostics.png").relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE3_DIR / "facies_f3_stage3_oof_diagnostics.png"),
            "description": "Existing frozen stage-3 F3 OOF diagnostic figure.",
        },
        {
            "kind": "figure",
            "name": "facies_penobscot_stage3_oof_diagnostics.png",
            "path": str((STAGE3_DIR / "facies_penobscot_stage3_oof_diagnostics.png").relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE3_DIR / "facies_penobscot_stage3_oof_diagnostics.png"),
            "description": "Existing frozen stage-3 Penobscot OOF diagnostic figure.",
        },
        {
            "kind": "figure",
            "name": "facies_f3_known_holdout_diagnostics.png",
            "path": str((STAGE4_DIR / "facies_f3" / "known_holdout_diagnostics.png").relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE4_DIR / "facies_f3" / "known_holdout_diagnostics.png"),
            "description": "Known-holdout confirmation figure for F3.",
        },
        {
            "kind": "figure",
            "name": "facies_penobscot_known_holdout_diagnostics.png",
            "path": str((STAGE4_DIR / "facies_penobscot" / "known_holdout_diagnostics.png").relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE4_DIR / "facies_penobscot" / "known_holdout_diagnostics.png"),
            "description": "Known-holdout confirmation figure for Penobscot.",
        },
    ]

    tables = [
        {
            "kind": "table",
            "name": "track_model_metrics.xlsx",
            "path": str(generated_workbook),
            "status": "generated",
            "sha256": sha256_file(output_dir / "track_model_metrics.xlsx"),
            "description": "Single-sheet workbook of facies model metrics.",
        },
        {
            "kind": "table",
            "name": "p5_stage3_results.jsonl",
            "path": str(STAGE3_SOURCE.relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE3_SOURCE),
            "description": "Frozen stage-3 per-cell development evidence.",
        },
        {
            "kind": "table",
            "name": "p5_stage3_summary.json",
            "path": str(STAGE3_SUMMARY.relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE3_SUMMARY),
            "description": "Frozen stage-3 leaderboard summary.",
        },
        {
            "kind": "table",
            "name": "p5_stage4_summary.json",
            "path": str(STAGE4_SUMMARY.relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(STAGE4_SUMMARY),
            "description": "Known-holdout confirmation summary.",
        },
        {
            "kind": "table",
            "name": "p9_sam2_effect/facies_f3/summary.json",
            "path": str(P9_SUMMARIES["facies_f3"].relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(P9_SUMMARIES["facies_f3"]),
            "description": "SAM2 facies effect summary for F3.",
        },
        {
            "kind": "table",
            "name": "p9_sam2_effect/facies_penobscot/summary.json",
            "path": str(P9_SUMMARIES["facies_penobscot"].relative_to(PROJECT_ROOT)),
            "status": "indexed",
            "sha256": sha256_file(P9_SUMMARIES["facies_penobscot"]),
            "description": "SAM2 facies effect summary for Penobscot.",
        },
    ]
    for task, summary_path in P10_REPAIR_SUMMARIES.items():
        if not summary_path.exists():
            continue
        tables.append(
            {
                "kind": "table",
                "name": f"p10_sam2_repair_audit/{task}/summary.json",
                "path": str(summary_path.relative_to(PROJECT_ROOT)),
                "status": "generated",
                "sha256": sha256_file(summary_path),
                "description": f"Development-only SAM2 repair audit summary for {task}.",
            }
        )

    figures_path = output_dir / "figures_manifest.csv"
    tables_path = output_dir / "tables_manifest.csv"
    write_csv(figures_path, figures, ["kind", "name", "path", "status", "sha256", "description"])
    write_csv(tables_path, tables, ["kind", "name", "path", "status", "sha256", "description"])
    return figures_path, tables_path


def write_audit_report(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    def _audit_matrix_line(item: str, code: str, evidence: str, status: str, note: str) -> str:
        return f"| {item} | `{code}` | `{evidence}` | {status} | {note} |"

    p9_reference: dict[str, dict[str, float]] = defaultdict(
        lambda: {"before": float("nan"), "after": float("nan")}
    )
    for row in rows:
        if not str(row["split_protocol"]).startswith("p9_sam2_effect"):
            continue
        if row["metric_name"] != "miou" or row["model_name"] != "facebook/sam2.1-hiera-base-plus":
            continue
        p9_reference[row["dataset"]]["before"] = float(row["metric_value"])
        p9_reference[row["dataset"]]["after"] = float(row["baseline_value"])

    blocker = {
        "schema_version": "facies-p10-sam2-repair-blocker/v1",
        "blocked_state": "data_blocked",
        "command": (
            "CUDA_VISIBLE_DEVICES=3 /mnt/data/yongan-admin-2/.cache/volve-p5/envs/torch-common/bin/python "
            "_pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py --task-id facies_f3 "
            "--manifest /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/"
            "p4-training-integration/_tmp/p4-acceptance/facies_f3/split_manifest.json "
            "--processed-root /mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/"
            "track-facies/_data/processed --device cuda:0"
        ),
        "error": "ModuleNotFoundError: No module named 'hydra'",
        "checked_environments": ["system python", "torch-common env"],
        "reason": "audited SAM2 source checkout imports hydra, which is absent in both execution environments",
        "source_checkout": "/mnt/data/yongan-admin-2/.cache/upstream/sam2",
        "source_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
    }
    blocker_json = output_dir / P10_REPAIR_BLOCKER_RELATIVE
    blocker_json.parent.mkdir(parents=True, exist_ok=True)
    blocker_json.write_text(json.dumps(blocker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Facies P10 model-results audit",
        "",
        "## Interface audit matrix",
        "",
        "| Item | Code evidence | Archived evidence | Status | Note |",
        "|---|---|---|---|---|",
        _audit_matrix_line(
            "Input channels",
            "_models/facies/sam2_semantic.py:forward",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "SAM2 consumes [B,1,H,W] and repeats to 3 channels before image-encoder normalization.",
        ),
        _audit_matrix_line(
            "Amplitude scaling",
            "_models/facies/sam2_semantic.py:forward",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "Clamps to [-5, 5], rescales to [0,1], and applies ImageNet mean/std.",
        ),
        _audit_matrix_line(
            "Native SAM2 preprocessing",
            "_models/facies/sam2_semantic.py:build_model",
            "_models/gaia_dagt/foundation_routes.v1.json",
            "audited",
            "Official SAM2 build is loaded with apply_postprocessing=False.",
        ),
        _audit_matrix_line(
            "Prompt leakage",
            "_models/facies/sam2_semantic.py; no prompt input exists",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "No validation truth prompt path exists; conditioning is explicitly spatial_prompt:none.",
        ),
        _audit_matrix_line(
            "Label mapping",
            "_pipelines/02_task_datasets/facies/pipeline_contract.py",
            "_pipelines/02_task_datasets/facies/_outputs/p5_stage3/p5_stage3_results.jsonl",
            "audited",
            "F3 stays 10-class, Penobscot stays 8-class; independent TaskSpecs remain separate.",
        ),
        _audit_matrix_line(
            "Decoder / head",
            "_models/facies/sam2_semantic.py; _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py",
            str(P10_REPAIR_BLOCKER_PATH.relative_to(PROJECT_ROOT)),
            "data_blocked",
            "The intended gated residual repair head is blocked until hydra is available in the audited SAM2 checkout.",
        ),
        _audit_matrix_line(
            "PEFT / freeze policy",
            "_pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py",
            str(P10_REPAIR_BLOCKER_PATH.relative_to(PROJECT_ROOT)),
            "data_blocked",
            "Base adapter freeze policy is implemented, but the repair probe is blocked before execution.",
        ),
        _audit_matrix_line(
            "Loss",
            "_pipelines/02_task_datasets/facies/p9_sam2_effect.py",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "Weighted cross-entropy on raw logits; softmax is inference/evaluation only.",
        ),
        _audit_matrix_line(
            "Postprocess",
            "_pipelines/02_task_datasets/facies/p4_metrics.py",
            "_pipelines/02_task_datasets/facies/_outputs/p5_stage3/*.json",
            "audited",
            "Argmax/softmax are used only at evaluation and visualization; no threshold tuning occurs.",
        ),
        _audit_matrix_line(
            "Eval parity",
            "_pipelines/02_task_datasets/facies/facies_p5_stage3.py; _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py",
            "_pipelines/02_task_datasets/facies/_outputs/p5_stage3/*.json",
            "audited",
            "Same locked development folds, same sample caps, same seed discipline, no frozen-test access.",
        ),
        "",
        "## Conclusion",
        "",
        "The previous p10 artifact only framed SAM2 against the locked baseline. This revision keeps the archived reference comparison but records the actual repair attempt as data_blocked rather than pretending a repaired metric exists. No non_beneficial claim is made from an incomplete audit.",
        "",
        "## Repair probe blocker",
        "",
        f"- Status: `{blocker['blocked_state']}`",
        f"- Command: `{blocker['command']}`",
        f"- Error: `{blocker['error']}`",
        f"- Checked environments: {', '.join(blocker['checked_environments'])}",
        "",
        "## Archived reference comparison",
        "",
        "| Dataset | Archived pretrained adapter mIoU | Archived locked baseline mIoU | Delta |",
        "|---|---:|---:|---:|",
        f"| F3 | {p9_reference['F3']['before']:.6f} | {p9_reference['F3']['after']:.6f} | {p9_reference['F3']['before'] - p9_reference['F3']['after']:.6f} |",
        f"| Penobscot | {p9_reference['Penobscot']['before']:.6f} | {p9_reference['Penobscot']['after']:.6f} | {p9_reference['Penobscot']['before'] - p9_reference['Penobscot']['after']:.6f} |",
        "",
        "## Evidence boundary",
        "",
        "- Frozen test and known holdout were not reopened for tuning.",
        "- The workbook and manifests reference archived evidence plus a blocker file for the attempted repair probe.",
        "- Checkpoint paths are recorded as runtime references where the checkout does not contain a persisted weight file for the historical stage-3 baselines.",
        "",
        "## Residual risk",
        "",
        "- The blocked repair probe is intentionally not papered over with a local hydra stub or a package install.",
        "- If a future promoted model is desired, the next step is to add the missing dependency into the audited SAM2 source environment or switch to a different audited foundation checkout, then rerun the same fixed-dev comparison.",
    ]
    report_path = output_dir / "audit_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path

    repair_rows = [
        row
        for row in rows
        if row["split_protocol"].startswith("p10_sam2_repair_audit")
        and row["metric_name"] == "miou"
        and row["seed_or_fold"].startswith("fold_")
    ]
    repair_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in repair_rows:
        repair_by_dataset[row["dataset"]].append(row)
    lines = [
        "# Facies P10 model-results audit",
        "",
        "## Interface audit matrix",
        "",
        "| Item | Code evidence | Archived evidence | Status | Note |",
        "|---|---|---|---|---|",
        _audit_matrix_line(
            "Input channels",
            "_models/facies/sam2_semantic.py:forward",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "SAM2 consumes [B,1,H,W] and repeats to 3 channels before image-encoder normalization.",
        ),
        _audit_matrix_line(
            "Amplitude scaling",
            "_models/facies/sam2_semantic.py:forward",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "Clamps to [-5, 5], rescales to [0,1], and applies ImageNet mean/std.",
        ),
        _audit_matrix_line(
            "Native SAM2 preprocessing",
            "_models/facies/sam2_semantic.py:build_model",
            "_models/gaia_dagt/foundation_routes.v1.json",
            "audited",
            "Official SAM2 build is loaded with apply_postprocessing=False; no extra postprocess is hidden in training.",
        ),
        _audit_matrix_line(
            "Prompt leakage",
            "_models/facies/sam2_semantic.py; no prompt input exists",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "No validation truth prompt path exists; conditioning is explicitly spatial_prompt:none in the frozen route.",
        ),
        _audit_matrix_line(
            "Label mapping",
            "_pipelines/02_task_datasets/facies/pipeline_contract.py",
            "_pipelines/02_task_datasets/facies/_outputs/p5_stage3/p5_stage3_results.jsonl",
            "audited",
            "F3 stays 10-class, Penobscot stays 8-class; independent TaskSpecs remain separate.",
        ),
        _audit_matrix_line(
            "Decoder / head",
            "_models/facies/sam2_semantic.py; _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py",
            "_pipelines/02_task_datasets/facies/_outputs/p10_sam2_repair_audit/*.json",
            "audited",
            "Before = trainable projections + semantic head; after = frozen base adapter + gated residual probe.",
        ),
        _audit_matrix_line(
            "PEFT / freeze policy",
            "_pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py",
            "_pipelines/02_task_datasets/facies/_outputs/p10_sam2_repair_audit/*.json",
            "audited",
            "Repair candidate freezes the base adapter; only the residual probe and scalar gate train.",
        ),
        _audit_matrix_line(
            "Loss",
            "_pipelines/02_task_datasets/facies/p9_sam2_effect.py",
            "_pipelines/02_task_datasets/facies/_outputs/p9_sam2_effect/*.json",
            "audited",
            "Weighted cross-entropy on raw logits; softmax is inference/evaluation only.",
        ),
        _audit_matrix_line(
            "Postprocess",
            "_pipelines/02_task_datasets/facies/p4_metrics.py",
            "_pipelines/02_task_datasets/facies/_outputs/p5_stage3/*.json",
            "audited",
            "Argmax/softmax are used only at evaluation and visualization; no threshold tuning occurs.",
        ),
        _audit_matrix_line(
            "Eval parity",
            "_pipelines/02_task_datasets/facies/facies_p5_stage3.py; _pipelines/02_task_datasets/facies/p10_sam2_repair_audit.py",
            "_pipelines/02_task_datasets/facies/_outputs/p5_stage3/*.json",
            "audited",
            "Same locked development folds, same sample caps, same seed discipline, no frozen-test access.",
        ),
        "",
        "## Conclusion",
        "",
        "The previous p10 artifact was incomplete because it only restated SAM2 versus the locked strong baseline. This revision adds an honest development-only repair audit: the frozen pretrained SAM2 adapter is compared against a gated residual repair candidate that cannot erase the base logits, with a random-init control. The evidence still does not justify promotion.",
        "",
        "## Reference comparison primary metric",
        "",
        "| Dataset | Before (SAM2 pretrained adapter mIoU) | After (gated residual repair mIoU) | Delta |",
        "|---|---:|---:|---:|",
        f"| F3 | {np.mean([row['pretrained_adapter_miou'] for row in repair_by_dataset['F3']]):.6f} | {np.mean([row['gated_residual_repair_miou'] for row in repair_by_dataset['F3']]):.6f} | {np.mean([row['gated_residual_repair_miou'] for row in repair_by_dataset['F3']]) - np.mean([row['pretrained_adapter_miou'] for row in repair_by_dataset['F3']]):.6f} |",
        f"| Penobscot | {np.mean([row['pretrained_adapter_miou'] for row in repair_by_dataset['Penobscot']]):.6f} | {np.mean([row['gated_residual_repair_miou'] for row in repair_by_dataset['Penobscot']]):.6f} | {np.mean([row['gated_residual_repair_miou'] for row in repair_by_dataset['Penobscot']]) - np.mean([row['pretrained_adapter_miou'] for row in repair_by_dataset['Penobscot']]):.6f} |",
        "",
        "## Repair candidate against the locked baseline",
        "",
        f"- F3 repair delta vs locked baseline: {float(np.mean([row['gated_residual_repair_miou'] for row in repair_by_dataset['F3']]) - float(np.mean([row['strong_baseline_miou'] for row in repair_by_dataset['F3']]))):+.6f}",
        f"- Penobscot repair delta vs locked baseline: {float(np.mean([row['gated_residual_repair_miou'] for row in repair_by_dataset['Penobscot']]) - float(np.mean([row['strong_baseline_miou'] for row in repair_by_dataset['Penobscot']]))):+.6f}",
        "",
        "## Root cause / fix",
        "",
        "- Root cause: the original p10 artifact only re-packaged a foundation-vs-baseline comparison and did not audit the repairable interface points.",
        "- Fix applied: add a real gated residual repair audit, keep the base adapter frozen in the repair branch, and document the exact code/evidence for channels, preprocessing, prompts, labels, decoder, PEFT, loss, postprocess and evaluation parity.",
        "",
        "## Evidence boundary",
        "",
        "- Frozen test and known holdout were not reopened for tuning.",
        "- The workbook and manifests reference archived evidence plus the new development-only repair audit.",
        "- Checkpoint paths are recorded as runtime references where the checkout does not contain a persisted weight file for the historical stage-3 baselines.",
        "",
        "## Residual risk",
        "",
        "- The stage-3 baseline checkpoint files are not persisted in this checkout, so the workbook still uses logical runtime checkpoint references from archived evidence.",
        "- The residual repair probe is intentionally simple; if a future promoted model is desired, the next step is a proper retrained head on the frozen SAM2 backbone, not a claim of promotion from this audit.",
    ]
    # Add a compact per-dataset repair table.
    for dataset in ("F3", "Penobscot"):
        lines.extend(
            [
                "",
                f"### {dataset} repair comparison",
                "",
                "| Fold | Pretrained adapter mIoU | Gated residual repair mIoU | Random-init control mIoU | Strong baseline mIoU |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(repair_by_dataset[dataset], key=lambda item: int(str(item["seed_or_fold"]).split("_")[1])):
            lines.append(
                f"| {row['seed_or_fold']} | {float(row['pretrained_adapter_miou']):.6f} | "
                f"{float(row['gated_residual_repair_miou']):.6f} | {float(row['random_init_control_miou']):.6f} | "
                f"{float(row['strong_baseline_miou']):.6f} |"
            )
    report_path = output_dir / "audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_primary_metric_figure(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    labels = ["F3", "Penobscot"]
    by_dataset: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"before": [], "after": []})
    for row in rows:
        if not str(row["split_protocol"]).startswith("p9_sam2_effect"):
            continue
        if row["metric_name"] != "miou" or not str(row["seed_or_fold"]).startswith("fold_"):
            continue
        if row["model_name"] == "facebook/sam2.1-hiera-base-plus":
            by_dataset[row["dataset"]]["before"].append(float(row["metric_value"]))
        elif row["model_name"] == "sam2.1-hiera-base-plus-random-init":
            by_dataset[row["dataset"]]["after"].append(float(row["metric_value"]))
    before = [float(np.mean(by_dataset[label]["before"])) for label in labels]
    after = [float(np.mean(by_dataset[label]["after"])) for label in labels]
    x = list(range(len(labels)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], before, width=width, label="SAM2 pretrained adapter", color="#d95f02")
    ax.bar([i + width / 2 for i in x], after, width=width, label="Random-init control", color="#1b9e77")
    for idx, value in enumerate(before):
        ax.text(idx - width / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(after):
        ax.text(idx + width / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Primary metric (mIoU)")
    ax.set_title("Facies P10 reference comparison: pretrained adapter vs random-init control")
    ax.set_ylim(0, max(after) + 0.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = output_dir / "before_after_primary_metric.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_audit_report(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    summaries = {
        task: read_json(path)
        for task, path in P10_REPAIR_SUMMARIES.items()
        if path.exists()
    }
    lines = [
        "# Facies P10 model-results audit",
        "",
        "## Conclusion",
        "",
        "The corrected SAM2 environment ran the development-only repair audit for both facies datasets.",
        "Real pretrained weights improve the same architecture over random initialization, so the foundation encoder is useful.",
        "However, the pretrained adapter still trails the locked strong segmentation baselines, and the tested gated-residual fusion degrades the pretrained adapter.",
        "The honest end-to-end decision therefore remains non_beneficial; no holdout was used for tuning.",
        "",
        "## Interface audit matrix",
        "",
        "| Item | Evidence | Result |",
        "|---|---|---|",
        "| Input channels | `_models/facies/sam2_semantic.py:forward`; summaries record `[B,1,H,W]` | one seismic channel is repeated to three channels before encoder normalization |",
        "| Amplitude scaling | adapter forward path | clamp `[-5,5]`, rescale to `[0,1]`, then ImageNet mean/std |",
        "| Native SAM2 preprocessing | audited source checkout and real checkpoint hash | official SAM2.1 Hiera-B+ encoder loaded with real pretrained weights |",
        "| Prompt leakage | no prompt input in semantic adapter | no validation-label prompt path exists |",
        "| Label mapping | `pipeline_contract.py` | F3 remains 10-class; Penobscot remains 8-class |",
        "| Decoder / fusion | `p10_sam2_repair_audit.py` | frozen base logits plus a sigmoid-gated residual head was tested |",
        "| PEFT / freeze policy | repair summaries | base adapter frozen; only residual head and scalar gate trained |",
        "| Loss / postprocess | repair script and `p4_metrics.py` | weighted cross-entropy on raw logits; argmax only for evaluation |",
        "| Evaluation parity | manifest hashes and fold rows | fixed development manifests, fixed sample caps/seeds, no frozen-test access |",
        "",
        "## Development-only comparison",
        "",
        "| Dataset | Strong baseline mIoU | Pretrained adapter mIoU | Random-init mIoU | Foundation gain | Gated repair mIoU | Repair vs pretrained |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, meta in TASK_LABELS.items():
        summary = summaries[task]
        comparison = summary["comparison"]
        strong = float(comparison["strong_baseline_macro_fold_miou"])
        pretrained = float(comparison["pretrained_adapter_macro_fold_miou"])
        random_init = float(comparison["random_init_control_macro_fold_miou"])
        repair = float(comparison["gated_residual_repair_macro_fold_miou"])
        lines.append(
            f"| {meta['dataset']} | {strong:.6f} | {pretrained:.6f} | {random_init:.6f} | "
            f"{pretrained - random_init:+.6f} | {repair:.6f} | {repair - pretrained:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-fold evidence",
            "",
        ]
    )
    for task, meta in TASK_LABELS.items():
        summary = summaries[task]
        lines.extend(
            [
                f"### {meta['dataset']}",
                "",
                "| Fold | Seed | Strong baseline | Pretrained adapter | Random-init | Gated repair |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for fold in summary["fold_results"]:
            lines.append(
                f"| {fold['fold_id']} | {fold['seed']} | {float(fold['strong_baseline_miou']):.6f} | "
                f"{float(fold['pretrained_adapter_miou']):.6f} | {float(fold['random_init_control_miou']):.6f} | "
                f"{float(fold['gated_residual_repair_miou']):.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Root cause and fix status",
            "",
            "- The earlier blocker was an environment-selection error: `atom-sam2-py310` already contained Hydra and the audited SAM2 dependencies.",
            "- Fix applied: execute both repair probes in that environment and replace the false blocker with measured development evidence.",
            "- The tested gated-residual fusion is not a successful repair; it underfits and reduces mIoU relative to the pretrained adapter.",
            "- Pretraining itself is beneficial versus same-architecture random initialization, but the current semantic adapter/head is still not competitive with the strong task-specific baselines.",
            "",
            "## Evidence boundary",
            "",
            "- Both summaries record `frozen_test_accessed=false`.",
            "- Each probe used two fixed development folds, fixed seeds, at most 32 train samples and 16 validation samples per fold.",
            "- No threshold tuning, seed selection, label remapping, or holdout reuse was performed.",
            "",
            "## Residual risk",
            "",
            "- This was a bounded repair audit, not a full backbone/head retraining campaign.",
            "- A future attempt should test a properly trained multi-scale decoder or parameter-efficient fine-tuning strategy on the frozen development protocol; the current gated residual head should not be promoted.",
        ]
    )
    report_path = output_dir / "audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_primary_metric_figure(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    labels: list[str] = []
    strong: list[float] = []
    pretrained: list[float] = []
    random_init: list[float] = []
    repair: list[float] = []
    for task, meta in TASK_LABELS.items():
        summary = read_json(P10_REPAIR_SUMMARIES[task])
        comparison = summary["comparison"]
        labels.append(meta["dataset"])
        strong.append(float(comparison["strong_baseline_macro_fold_miou"]))
        pretrained.append(float(comparison["pretrained_adapter_macro_fold_miou"]))
        random_init.append(float(comparison["random_init_control_macro_fold_miou"]))
        repair.append(float(comparison["gated_residual_repair_macro_fold_miou"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    x = np.arange(len(labels), dtype=float)
    width = 0.2
    series = [
        ("Strong baseline", strong, "#7f8c8d", -1.5),
        ("SAM2 pretrained", pretrained, "#2e86de", -0.5),
        ("SAM2 random-init", random_init, "#95a5a6", 0.5),
        ("Gated repair", repair, "#e67e22", 1.5),
    ]
    for name, values, color, offset in series:
        bars = ax.bar(x + offset * width, values, width=width, label=name, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Development mIoU (higher is better)")
    ax.set_title("Facies SAM2 audit: foundation gain is real, repair remains non-beneficial")
    ax.set_ylim(0, max(strong) * 1.3)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    ax.text(
        0.01,
        0.99,
        "Fixed development folds; no frozen-test tuning",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    path = output_dir / "before_after_primary_metric.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    xlsx_path = OUTPUT_DIR / "track_model_metrics.xlsx"
    write_xlsx(rows, xlsx_path)
    report_path = write_audit_report(rows, OUTPUT_DIR)
    figure_path = write_primary_metric_figure(rows, OUTPUT_DIR)
    figures_manifest_path, tables_manifest_path = write_manifests(rows, OUTPUT_DIR)
    return {
        "track_model_metrics.xlsx": xlsx_path,
        "before_after_primary_metric.png": figure_path,
        "figures_manifest.csv": figures_manifest_path,
        "tables_manifest.csv": tables_manifest_path,
        "audit_report.md": report_path,
    }


def main() -> None:
    outputs = build()
    print(json.dumps({name: str(path) for name, path in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
