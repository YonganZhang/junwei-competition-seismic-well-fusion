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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook, load_workbook


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

        rows.append(
            RowContext(
                track="facies",
                dataset=meta["dataset"],
                task_type=meta["task_type"],
                model_name="facebook/sam2.1-hiera-base-plus",
                model_family="SAM2.1 Hiera Base Plus",
                is_foundation_model=True,
                foundation_type="image_segmentation_foundation_encoder",
                integration_point="adapter_plus_semantic_head",
                fusion_method="fpn_projections_plus_semantic_head",
                preprocess_version="p9_sam2_effect_identity_norm",
                split_protocol="p9_sam2_effect_macro_fold",
                seed_or_fold="aggregate_macro_fold",
                metric_name="miou",
                metric_value=float(comparison["pretrained_macro_fold_miou"]),
                higher_is_better=True,
                baseline_model=baseline_model,
                baseline_value=baseline_mean,
                status="non_beneficial",
                evidence_path=evidence_path,
                checkpoint_path=str(SAM2_CHECKPOINT),
                code_commit=CURRENT_COMMIT,
                root_cause="pretrained_foundation_underperforms_locked_baseline_on_same_split",
                fix_applied="none",
                notes=(
                    f"pretrained={comparison['pretrained_macro_fold_miou']:.6f} "
                    f"random_init={comparison['same_architecture_random_init_macro_fold_miou']:.6f}; "
                    "frozen test not opened"
                ),
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
            "description": "Primary metric before/after comparison for F3 and Penobscot.",
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

    figures_path = output_dir / "figures_manifest.csv"
    tables_path = output_dir / "tables_manifest.csv"
    write_csv(figures_path, figures, ["kind", "name", "path", "status", "sha256", "description"])
    write_csv(tables_path, tables, ["kind", "name", "path", "status", "sha256", "description"])
    return figures_path, tables_path


def write_audit_report(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    by_task: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["metric_name"] == "miou" and row["split_protocol"].startswith("p9_sam2_effect"):
            key = f"{row['dataset']}::{row['status']}"
            by_task[key][row["model_name"]] = float(row["metric_value"])
    lines = [
        "# Facies P10 model-results audit",
        "",
        "## Conclusion",
        "",
        "The archived evidence shows a non-beneficial SAM2 integration: same-split SAM2 remains below the locked strong baselines on both F3 and Penobscot. No reproducible code defect was proven from the archived artifacts, so no model repair was applied in this pass.",
        "",
        "## Before / after primary metric",
        "",
        "| Dataset | Before (SAM2 pretrained mIoU) | After (locked strong baseline mIoU) | Delta |",
        "|---|---:|---:|---:|",
        "| F3 | 0.082017 | 0.131316 | -0.049299 |",
        "| Penobscot | 0.076754 | 0.132021 | -0.055267 |",
        "",
        "## Root cause / fix",
        "",
        "- Root cause: no gain on the locked same-split development evidence; the integration is honest but non-beneficial.",
        "- Fix applied: none in this pass; the right conclusion is `non_beneficial`, not a fabricated repair.",
        "",
        "## Evidence boundary",
        "",
        "- Frozen test and known holdout were not reopened for tuning.",
        "- The workbook and manifests reference archived evidence only.",
        "- Checkpoint paths are recorded as runtime references where the checkout does not contain a persisted weight file.",
        "",
        "## Residual risk",
        "",
        "- Because no persisted checkpoint files exist in this checkout, the workbook uses logical runtime checkpoint references from the archived JSON evidence rather than a file on disk.",
    ]
    report_path = output_dir / "audit_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_primary_metric_figure(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    labels = ["F3", "Penobscot"]
    before = [0.0820173614809763, 0.07675446038321675]
    after = [0.13131642202022092, 0.13202141174689058]
    x = list(range(len(labels)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], before, width=width, label="SAM2 pretrained", color="#d95f02")
    ax.bar([i + width / 2 for i in x], after, width=width, label="Locked strong baseline", color="#1b9e77")
    for idx, value in enumerate(before):
        ax.text(idx - width / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(after):
        ax.text(idx + width / 2, value + 0.003, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Primary metric (mIoU)")
    ax.set_title("Facies P10 primary metric before/after")
    ax.set_ylim(0, max(after) + 0.05)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    path = output_dir / "before_after_primary_metric.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build() -> dict[str, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    xlsx_path = OUTPUT_DIR / "track_model_metrics.xlsx"
    write_xlsx(rows, xlsx_path)
    figure_path = write_primary_metric_figure(OUTPUT_DIR)
    figures_manifest_path, tables_manifest_path = write_manifests(rows, OUTPUT_DIR)
    report_path = write_audit_report(rows, OUTPUT_DIR)
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
