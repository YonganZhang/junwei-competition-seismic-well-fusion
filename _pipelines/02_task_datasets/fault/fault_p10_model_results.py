#!/usr/bin/env python3
"""Assemble fault p10 model-results evidence without training or holdout access."""
from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from openpyxl import Workbook  # noqa: E402


TRACK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TRACK_DIR / "_outputs" / "p10_model_results"
WORKBOOK_NAME = "track_model_metrics.xlsx"
WORKBOOK_SHEET_NAME = "模型指标"
FIGURES_MANIFEST_NAME = "figures_manifest.csv"
TABLES_MANIFEST_NAME = "tables_manifest.csv"
AUDIT_REPORT_NAME = "audit_report.md"
FIGURE_NAME = "before_after_primary_metric.png"
PRIMARY_METRIC = "average_precision"
SOURCE_COMMIT = "e4fd5d8a6371c2b0db6ba2258a41349ec6cfb4f7"

AUDITED_V2_DIR = TRACK_DIR / "_outputs" / "runs" / "audited_v2"
AUDITED_V2_BASELINE = AUDITED_V2_DIR / "baseline_metrics.json"
AUDITED_V2_BUILD = AUDITED_V2_DIR / "build_summary.json"
AUDITED_V2_VISUALIZATION = AUDITED_V2_DIR / "visualization_report.json"
AUDITED_V2_PREDICTION = AUDITED_V2_DIR / "prediction_visualization.png"
AUDITED_V2_LOSS = AUDITED_V2_DIR / "loss_curve.png"
P5_STAGE3_SUMMARY = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_summary.json"
P5_STAGE3_DATA = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_data_manifest.json"
P5_STAGE3_VISUALIZATION = TRACK_DIR / "_outputs" / "p5_stage3" / "p5_stage3_visualization_manifest.json"
P5_STAGE3_READINESS_FIGS = [
    TRACK_DIR / "_outputs" / "p5_stage3" / "figures" / "fault_readiness.svg",
    TRACK_DIR / "_outputs" / "p5_stage3" / "figures" / "fault_negative_coverage.svg",
    TRACK_DIR / "_outputs" / "p5_stage3" / "figures" / "fault_unknown_coverage.svg",
]
P5_STAGE4_CONFIRMATION = TRACK_DIR / "_outputs" / "p5_stage4_confirmation" / "p5_stage4_confirmation.json"
P9_SAMMED3D_SUMMARY = TRACK_DIR / "_outputs" / "p9_sammed3d_gate" / "summary.json"

HEADER = [
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected object JSON in {path}")
    return payload


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required fault p10 evidence: " + ", ".join(missing))


def _source_commit() -> str:
    result = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=TRACK_DIR,
        text=True,
    )
    return result.strip()


def _portable_path(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _baseline_metrics() -> dict[str, float]:
    payload = _load_json(AUDITED_V2_BASELINE)
    test_metrics = payload["test_metrics"]
    probability_metrics = payload["test_probability_metrics"]
    return {
        "average_precision": float(probability_metrics["average_precision"]),
        "pr_auc": float(probability_metrics["pr_auc"]),
        "precision": float(test_metrics["precision"]),
        "recall": float(test_metrics["recall"]),
        "dice": float(test_metrics["dice"]),
        "iou": float(test_metrics["iou"]),
        "f1": float(test_metrics["f1"]),
    }


def build_rows(source_commit: str) -> list[dict[str, Any]]:
    baseline_metrics = _baseline_metrics()
    baseline_row_common = {
        "track": "fault",
        "dataset": "fault/audited_v2",
        "task_type": "3d_fault_stick_segmentation",
        "model_name": "fault_local_logistic",
        "model_family": "sgd_linear_pixel_classifier",
        "is_foundation_model": False,
        "foundation_type": "",
        "integration_point": "baseline.py::audited_v2",
        "fusion_method": "local_features_only",
        "preprocess_version": "audited_v2_train_fit_zscore",
        "split_protocol": "audited_v2 train/validation/test",
        "seed_or_fold": "seed=2693",
        "higher_is_better": True,
        "baseline_model": "fault_local_logistic",
        "checkpoint_path": "",
        "code_commit": source_commit,
        "root_cause": "canonical audited_v2 regression evidence only; no production checkpoint file exists in this checkout",
        "fix_applied": "none",
        "notes": "reference metrics preserved for p10 comparison; checkpoint_path intentionally blank because no checkpoint artifact is present",
    }
    rows = []
    for metric_name in ("average_precision", "pr_auc", "precision", "recall", "dice", "iou", "f1"):
        value = baseline_metrics[metric_name]
        rows.append(
            {
                **baseline_row_common,
                "metric_name": metric_name,
                "metric_value": value,
                "baseline_value": value,
                "delta_abs": 0.0,
                "delta_pct": 0.0,
                "status": "reference_only",
                "evidence_path": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            }
        )
    rows.append(
        {
            "track": "fault",
            "dataset": "fault/p9_sammed3d_gate",
            "task_type": "3d_fault_stick_segmentation",
            "model_name": "blueyo0/SAM-Med3D:sam_med3d_turbo.pth",
            "model_family": "foundation_3d_segmentation",
            "is_foundation_model": True,
            "foundation_type": "SAM-Med3D",
            "integration_point": "_models.fault.sam_med3d_semantic",
            "fusion_method": "prompted_3d_adapter",
            "preprocess_version": "blocked_no_legal_3d_development_fold",
            "split_protocol": "frozen_stage3_zero_fold",
            "seed_or_fold": "seed=2693",
            "metric_name": PRIMARY_METRIC,
            "metric_value": None,
            "higher_is_better": True,
            "baseline_model": "fault_local_logistic",
            "baseline_value": baseline_metrics[PRIMARY_METRIC],
            "delta_abs": None,
            "delta_pct": None,
            "status": "data_blocked",
            "evidence_path": str(P9_SAMMED3D_SUMMARY.relative_to(TRACK_DIR)),
            "checkpoint_path": "",
            "code_commit": source_commit,
            "root_cause": (
                "NO_VALID_FAULT_DEVELOPMENT_FOLDS; verified negatives remain 0; explicit unknown mask absent; "
                "2D slices are not a legal continuous 3D development volume"
            ),
            "fix_applied": "none; preserved fail-closed gate and evidence-only SAM-Med3D audit",
            "notes": "real SAM-Med3D checkpoint and forward path are documented in blocked evidence only; no 3D metric emitted",
        }
    )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_workbook(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = WORKBOOK_SHEET_NAME
    sheet.append(HEADER)
    for row in rows:
        sheet.append([row.get(column) for column in HEADER])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column, width in {
        "A": 10,
        "B": 24,
        "C": 26,
        "D": 36,
        "E": 30,
        "F": 18,
        "G": 16,
        "H": 30,
        "I": 22,
        "J": 28,
        "K": 26,
        "L": 14,
        "M": 26,
        "N": 16,
        "O": 16,
        "P": 18,
        "Q": 16,
        "R": 12,
        "S": 12,
        "T": 16,
        "U": 48,
        "V": 34,
        "W": 18,
        "X": 42,
        "Y": 24,
        "Z": 52,
    }.items():
        sheet.column_dimensions[column].width = width
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)


def _plot_before_after(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    baseline_row = next(row for row in rows if row["model_name"] == "fault_local_logistic" and row["metric_name"] == PRIMARY_METRIC)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    labels = ["Before\n(audited_v2)", "After\n(SAM-Med3D gate)"]
    values = [float(baseline_row["metric_value"]), 0.0]
    bars = ax.bar(labels, values, color=["#2A6F97", "#B8B8B8"], edgecolor=["#1B4965", "#6B6B6B"])
    bars[1].set_hatch("///")
    bars[1].set_alpha(0.75)
    ax.text(
        0,
        values[0] + max(values[0] * 0.03, 0.0002),
        f"{values[0]:.6f}",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#1B4965",
    )
    ax.text(
        1,
        0.0002,
        "data_blocked\nno legal 3D fold",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#404040",
    )
    ax.set_ylabel("Average precision")
    ax.set_title("Fault primary metric before/after comparison")
    ax.set_ylim(0.0, max(values[0] * 1.35, 0.01))
    ax.grid(axis="y", alpha=0.2)
    ax.text(
        0.5,
        -0.18,
        "Before uses the canonical audited_v2 reference; after remains blocked because verified negatives and legal 3D development folds are missing.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#505050",
        wrap=True,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _rows_to_report(rows: Sequence[dict[str, Any]], source_commit: str) -> dict[str, Any]:
    baseline_rows = [row for row in rows if row["status"] == "reference_only"]
    blocked_rows = [row for row in rows if row["status"] == "data_blocked"]
    return {
        "track": "fault",
        "task_id": "fault_stick_segmentation",
        "output_dir": str(OUTPUT_DIR.relative_to(TRACK_DIR)),
        "source_commit": source_commit,
        "primary_metric": PRIMARY_METRIC,
        "baseline_rows": len(baseline_rows),
        "blocked_rows": len(blocked_rows),
        "data_blocked": bool(blocked_rows),
        "production_checkpoint_present": False,
        "production_checkpoint_paths": [],
        "evidence_paths": [row["evidence_path"] for row in rows],
        "checkpoint_paths": [row["checkpoint_path"] for row in rows if row["checkpoint_path"]],
    }


def _render_audit_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fault P10 model-results audit",
        "",
        f"- Source commit: `{report['source_commit']}`",
        f"- Primary metric: `{report['primary_metric']}`",
        f"- Workbook: `{report['workbook']}`",
        f"- Figures manifest: `{report['figures_manifest']}`",
        f"- Tables manifest: `{report['tables_manifest']}`",
        f"- Primary figure: `{report['primary_figure']}`",
        "",
        "## Conclusion",
        "",
        "The fault lane remains `data_blocked` for any real 3D SAM-Med3D scoring attempt.",
        "No production checkpoint exists in this checkout, and no fabricated 3D improvement is claimed.",
        "",
        "## Evidence summary",
        "",
        f"- Baseline-only rows: {report['baseline_rows']}",
        f"- Blocked foundation rows: {report['blocked_rows']}",
        f"- Production checkpoint present: {report['production_checkpoint_present']}",
        "",
        "| role | path | note |",
        "|---|---|---|",
        f"| canonical baseline metrics | `{report['baseline_reference']}` | preserved reference evidence |",
        f"| blocked SAM-Med3D gate | `{report['blocked_evidence']}` | no legal 3D development fold |",
        f"| workbook | `{report['workbook']}` | single-sheet metrics ledger |",
        f"| figure | `{report['primary_figure']}` | before/after comparison with blocked after-side |",
        "",
        "## Preserved scientific conclusion",
        "",
        "- `fault_local_logistic` remains the canonical audited reference.",
        "- `blueyo0/SAM-Med3D:sam_med3d_turbo.pth` stays `data_blocked` because verified negatives are still 0 and there is no explicit unknown mask with a legal 3D development fold.",
        "- The report intentionally reuses the blocked evidence rather than inventing a score.",
        "",
        "## Validation scope",
        "",
        "- workbook can be reopened by openpyxl;",
        "- workbook has exactly one sheet named `模型指标`;",
        "- row-level evidence paths are existing local files;",
        "- figures/tables manifests index the rendered artifacts.",
        "",
    ]
    return "\n".join(lines)


def run(output_root: Path = OUTPUT_DIR) -> dict[str, Any]:
    _require_files(
        [
            AUDITED_V2_BASELINE,
            AUDITED_V2_BUILD,
            AUDITED_V2_VISUALIZATION,
            AUDITED_V2_PREDICTION,
            AUDITED_V2_LOSS,
            P5_STAGE3_SUMMARY,
            P5_STAGE3_DATA,
            P5_STAGE3_VISUALIZATION,
            P5_STAGE4_CONFIRMATION,
            P9_SAMMED3D_SUMMARY,
            *P5_STAGE3_READINESS_FIGS,
        ]
    )
    source_commit = _source_commit()
    rows = build_rows(source_commit)
    output_root.mkdir(parents=True, exist_ok=True)
    workbook_path = output_root / WORKBOOK_NAME
    figures_manifest_path = output_root / FIGURES_MANIFEST_NAME
    tables_manifest_path = output_root / TABLES_MANIFEST_NAME
    audit_report_path = output_root / AUDIT_REPORT_NAME
    primary_figure_path = output_root / FIGURE_NAME

    _write_workbook(workbook_path, rows)
    _plot_before_after(primary_figure_path, rows)

    figures_rows = [
        {
            "kind": "existing_chart",
            "path": str(AUDITED_V2_LOSS.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Audited baseline loss curve",
            "source_evidence": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            "notes": "canonical audited_v2 regression chart",
        },
        {
            "kind": "existing_chart",
            "path": str(AUDITED_V2_PREDICTION.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Audited baseline prediction visualization",
            "source_evidence": str(AUDITED_V2_VISUALIZATION.relative_to(TRACK_DIR)),
            "notes": "canonical audited_v2 regression chart",
        },
        {
            "kind": "existing_chart",
            "path": str(P5_STAGE3_READINESS_FIGS[0].relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault readiness coverage",
            "source_evidence": str(P5_STAGE3_DATA.relative_to(TRACK_DIR)),
            "notes": "readiness evidence reused by the blocked gate",
        },
        {
            "kind": "existing_chart",
            "path": str(P5_STAGE3_READINESS_FIGS[1].relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault negative coverage",
            "source_evidence": str(P5_STAGE3_DATA.relative_to(TRACK_DIR)),
            "notes": "readiness evidence reused by the blocked gate",
        },
        {
            "kind": "existing_chart",
            "path": str(P5_STAGE3_READINESS_FIGS[2].relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault unknown coverage",
            "source_evidence": str(P5_STAGE3_DATA.relative_to(TRACK_DIR)),
            "notes": "readiness evidence reused by the blocked gate",
        },
        {
            "kind": "generated_chart",
            "path": _portable_path(primary_figure_path, TRACK_DIR),
            "status": "valid",
            "title": "Fault primary metric before/after comparison",
            "source_evidence": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            "notes": "after side is explicitly blocked rather than an observed metric",
        },
    ]
    table_rows = [
        {
            "kind": "metric_workbook",
            "path": _portable_path(workbook_path, TRACK_DIR),
            "status": "valid",
            "title": "Fault model metrics workbook",
            "source_evidence": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            "notes": "single-sheet workbook only",
        },
        {
            "kind": "source_table",
            "path": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Audited baseline metrics",
            "source_evidence": str(AUDITED_V2_BUILD.relative_to(TRACK_DIR)),
            "notes": "reference-only baseline evidence",
        },
        {
            "kind": "source_table",
            "path": str(AUDITED_V2_BUILD.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Audited baseline build summary",
            "source_evidence": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            "notes": "build provenance and split hashes",
        },
        {
            "kind": "source_table",
            "path": str(P5_STAGE3_SUMMARY.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault P5 Stage-3 summary",
            "source_evidence": str(P5_STAGE3_DATA.relative_to(TRACK_DIR)),
            "notes": "zero-fold readiness gate",
        },
        {
            "kind": "source_table",
            "path": str(P5_STAGE3_DATA.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault P5 Stage-3 data manifest",
            "source_evidence": str(P5_STAGE3_SUMMARY.relative_to(TRACK_DIR)),
            "notes": "verified-negative and unknown-mask blockers",
        },
        {
            "kind": "source_table",
            "path": str(P5_STAGE3_VISUALIZATION.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault P5 Stage-3 visualization manifest",
            "source_evidence": str(P5_STAGE3_DATA.relative_to(TRACK_DIR)),
            "notes": "readiness figures only",
        },
        {
            "kind": "source_table",
            "path": str(P5_STAGE4_CONFIRMATION.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault P5 Stage-4 confirmation",
            "source_evidence": str(P5_STAGE3_SUMMARY.relative_to(TRACK_DIR)),
            "notes": "blocked confirmation; no winner/refit/holdout",
        },
        {
            "kind": "source_table",
            "path": str(P9_SAMMED3D_SUMMARY.relative_to(TRACK_DIR)),
            "status": "valid",
            "title": "Fault SAM-Med3D gate summary",
            "source_evidence": str(P5_STAGE4_CONFIRMATION.relative_to(TRACK_DIR)),
            "notes": "foundation route is connected-data blocked",
        },
    ]

    _write_csv(figures_manifest_path, figures_rows, ["kind", "path", "status", "title", "source_evidence", "notes"])
    _write_csv(tables_manifest_path, table_rows, ["kind", "path", "status", "title", "source_evidence", "notes"])

    report = _rows_to_report(rows, source_commit)
    report.update(
        {
            "baseline_reference": str(AUDITED_V2_BASELINE.relative_to(TRACK_DIR)),
            "blocked_evidence": str(P9_SAMMED3D_SUMMARY.relative_to(TRACK_DIR)),
            "workbook": _portable_path(workbook_path, TRACK_DIR),
            "figures_manifest": _portable_path(figures_manifest_path, TRACK_DIR),
            "tables_manifest": _portable_path(tables_manifest_path, TRACK_DIR),
            "primary_figure": _portable_path(primary_figure_path, TRACK_DIR),
            "rows": rows,
        }
    )
    audit_report_path.write_text(_render_audit_markdown(report), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = run(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
