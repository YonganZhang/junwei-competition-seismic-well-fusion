"""Track-local P12 visualizer for sweetspot.

This module is read-only with respect to upstream data and model artifacts.
It assembles seven-target visual evidence from archived JSON payloads and
original target plots into a compact, track-local deliverable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import textwrap
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
SWEETSPOT_ROOT = REPO_ROOT / "_pipelines" / "02_task_datasets" / "sweetspot"
OUTPUT_DIR = SWEETSPOT_ROOT / "_outputs" / "p12_visualization"
FIGURE_DIR = OUTPUT_DIR / "figures"

P5_STAGE3_DATA = SWEETSPOT_ROOT / "p5" / "_outputs" / "stage3_cv" / "visualization_data"
P5_STAGE3_FIGURES = SWEETSPOT_ROOT / "p5" / "_outputs" / "stage3_cv" / "figures"
P5_STAGE4_SUMMARY = SWEETSPOT_ROOT / "p5" / "_outputs" / "stage4_confirmation" / "p5_stage4_summary.json"
P7_T3_SUMMARY = SWEETSPOT_ROOT / "p7" / "_outputs" / "t3_chronos2_cv" / "summary.json"
P8_T3_SUMMARY = SWEETSPOT_ROOT / "p8" / "_outputs" / "t3_chronos2_calendar_cv" / "summary.json"

PALETTE_AKUN_4 = ["#376795", "#72BCD5", "#FFD06F", "#E76254"]
PALETTE_AKUN_7 = ["#3951A2", "#72AACF", "#CAE8F2", "#FEFBBA", "#FDB96B", "#EC5D3B", "#A80326"]

MODEL_COLORS = {
    "lightgbm": "#376795",
    "catboost": "#72BCD5",
    "xgboost": "#E76254",
    "inceptiontime": "#EF8A47",
}

TARGET_ORDER = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")

TARGET_META: dict[str, dict[str, Any]] = {
    "T1": {
        "name": "reservoir quality",
        "task_type": "regression",
        "primary_metric": "mae",
        "direction": "minimize",
        "status": "confirmed_known_holdout",
        "evidence_class": "previously_seen_reusable_holdout",
        "source_figure": P5_STAGE3_FIGURES / "T1_regression_scatter.png",
        "stage3_json": P5_STAGE3_DATA / "T1.json",
        "split_scope": "known-holdout confirmation",
        "caveat": "known holdout already consumed in stage 4; no fresh blind test is claimed.",
        "detail_kind": "regression",
    },
    "T2": {
        "name": "hydrocarbon pay",
        "task_type": "binary",
        "primary_metric": "average_precision",
        "secondary_metric": "brier",
        "direction": "maximize_primary_minimize_secondary",
        "status": "confirmed_known_holdout",
        "evidence_class": "previously_seen_reusable_holdout",
        "source_figure": P5_STAGE3_FIGURES / "T2_pr_calibration.png",
        "stage3_json": P5_STAGE3_DATA / "T2.json",
        "split_scope": "known-holdout confirmation",
        "caveat": "classification evidence comes from archived PR / calibration curves and is not a field truth claim.",
        "detail_kind": "classification",
    },
    "T3": {
        "name": "productivity",
        "task_type": "regression",
        "primary_metric": "mae",
        "direction": "minimize",
        "status": "confirmed_known_holdout",
        "evidence_class": "previously_seen_reusable_holdout",
        "source_figure": P5_STAGE3_FIGURES / "T3_regression_scatter.png",
        "stage3_json": P5_STAGE3_DATA / "T3.json",
        "split_scope": "known-holdout confirmation",
        "caveat": "foundation comparison is informative but default promotion remains governed by the archived decision gates.",
        "detail_kind": "regression",
    },
    "T4": {
        "name": "water breakthrough",
        "task_type": "binary",
        "primary_metric": "average_precision",
        "secondary_metric": "brier",
        "direction": "maximize_primary_minimize_secondary",
        "status": "confirmed_known_holdout",
        "evidence_class": "previously_seen_reusable_holdout",
        "source_figure": P5_STAGE3_FIGURES / "T4_pr_calibration.png",
        "stage3_json": P5_STAGE3_DATA / "T4.json",
        "split_scope": "known-holdout confirmation",
        "caveat": "Chronos risk evidence remains rejected/no-gain in the archived foundation summaries.",
        "detail_kind": "classification",
    },
    "T5": {
        "name": "remaining oil infill",
        "task_type": "status",
        "primary_metric": None,
        "direction": "n/a",
        "status": "not_feasible",
        "evidence_class": "blocked_by_contract",
        "source_figure": P5_STAGE3_FIGURES / "T5_status_gate.png",
        "stage3_json": P5_STAGE3_DATA / "T5.json",
        "split_scope": "no valid label contract",
        "caveat": "simulation proxy is not approved as field truth and must remain non-usable.",
        "detail_kind": "status",
    },
    "T6": {
        "name": "porosity",
        "task_type": "status",
        "primary_metric": None,
        "direction": "n/a",
        "status": "blocked",
        "evidence_class": "blocked_by_data",
        "source_figure": P5_STAGE3_FIGURES / "T6_data_gate.png",
        "stage3_json": P5_STAGE3_DATA / "T6.json",
        "split_scope": "no development-only feature source",
        "caveat": "test.h5 fallback is forbidden; this target stays blocked without a dev-only reconstruction source.",
        "detail_kind": "status",
    },
    "T7": {
        "name": "permeability",
        "task_type": "status",
        "primary_metric": None,
        "direction": "n/a",
        "status": "blocked",
        "evidence_class": "blocked_by_data",
        "source_figure": P5_STAGE3_FIGURES / "T7_data_gate.png",
        "stage3_json": P5_STAGE3_DATA / "T7.json",
        "split_scope": "no development-only feature source",
        "caveat": "test.h5 fallback is forbidden; this target stays blocked without a dev-only reconstruction source.",
        "detail_kind": "status",
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_commit(path: Path) -> str:
    return _git_output("log", "--follow", "--format=%H", "-n", "1", "--", str(path.relative_to(REPO_ROOT)))


def _installed_serif_family() -> str:
    candidates = ["Times New Roman", "TeX Gyre Termes"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            return candidate
    raise RuntimeError("required serif fonts are unavailable: Times New Roman / TeX Gyre Termes")


def _apply_style() -> str:
    serif = _installed_serif_family()
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif, "TeX Gyre Termes", "Times New Roman"],
            "svg.hashsalt": "sweetspot-p12-viz-v1",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
        }
    )
    return serif


def _panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.03,
        0.97,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111111",
    )


def _normalize_fonts(fig: plt.Figure) -> None:
    sizes = {"label": 8, "tick": 7, "legend": 7, "panel": 11}
    for ax in fig.get_axes():
        ax.xaxis.label.set_fontsize(sizes["label"])
        ax.yaxis.label.set_fontsize(sizes["label"])
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lbl.set_fontsize(sizes["tick"])
        lg = ax.get_legend()
        if lg is not None:
            for text in lg.get_texts():
                text.set_fontsize(sizes["legend"])
            if lg.get_title() is not None:
                lg.get_title().set_fontsize(sizes["legend"])
        for txt in ax.texts:
            raw = txt.get_text()
            if raw in {f"({chr(code)})" for code in range(ord("a"), ord("z") + 1)}:
                txt.set_fontsize(sizes["panel"])


def _assert_no_titles(fig: plt.Figure) -> None:
    if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text().strip():
        raise AssertionError("figure suptitle is disallowed")
    for ax in fig.get_axes():
        if ax.get_title().strip():
            raise AssertionError("axis titles are disallowed")


def _load_rgba(path: Path) -> Any:
    return Image.open(path).convert("RGBA")


def _save_bundle(fig: plt.Figure, stem: str, output_dir: Path) -> dict[str, Any]:
    _assert_no_titles(fig)
    _normalize_fonts(fig)
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_datetime = datetime(2000, 1, 1, tzinfo=timezone.utc)
    stable_svg_metadata = {"Date": "2000-01-01T00:00:00"}
    stable_pdf_metadata = {
        "Title": stem,
        "Author": "Codex",
        "Subject": "sweetspot p12 visualization",
        "Keywords": "deterministic, scientific visualization",
        "CreationDate": stable_datetime,
        "ModDate": stable_datetime,
    }
    paths = {}
    for ext in ("png", "svg", "pdf"):
        path = output_dir / f"{stem}.{ext}"
        if ext == "svg":
            fig.savefig(path, dpi=300, metadata=stable_svg_metadata)
        elif ext == "pdf":
            fig.savefig(path, dpi=300, metadata=stable_pdf_metadata)
        else:
            fig.savefig(path, dpi=300)
        paths[ext] = path
    svg_text = paths["svg"].read_text(encoding="utf-8")
    svg_text = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    paths["svg"].write_text(svg_text, encoding="utf-8")
    plt.close(fig)
    with Image.open(paths["png"]) as image:
        width_px, height_px = image.size
    return {
        "png": _repo_path(paths["png"]),
        "svg": _repo_path(paths["svg"]),
        "pdf": _repo_path(paths["pdf"]),
        "png_sha256": _sha256_file(paths["png"]),
        "svg_sha256": _sha256_file(paths["svg"]),
        "pdf_sha256": _sha256_file(paths["pdf"]),
        "width_px": width_px,
        "height_px": height_px,
    }


def _scatter_points(data: list[dict[str, Any]], model_id: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in data:
        if row["model_id"] != model_id:
            continue
        xs.append(float(row["actual"]))
        ys.append(float(row["prediction"]))
    return xs, ys


def _make_identity_axis(ax: plt.Axes, values: Iterable[float]) -> None:
    vals = list(values)
    lo = min(vals)
    hi = max(vals)
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#444444", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _bar_summary(ax: plt.Axes, labels: list[str], values: list[float], color: str, xlabel: str) -> None:
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, alpha=0.85)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.4)


def _metric_direction_text(meta: dict[str, Any]) -> str:
    if meta["detail_kind"] == "status":
        return "status only"
    if meta["task_type"] == "regression":
        return "↓ MAE"
    return "↑ AP / ↓ Brier"


def _short_commit(commit: str) -> str:
    return commit[:8]


def _compact_lines(lines: Iterable[str], width: int = 34) -> str:
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(textwrap.fill(line, width=width, break_long_words=True, break_on_hyphens=False).splitlines())
    return "\n".join(wrapped)


def _human_status(status: str) -> str:
    mapping = {
        "confirmed_known_holdout": "confirmed holdout",
        "not_feasible": "not feasible",
        "blocked": "blocked",
    }
    return mapping.get(status, status.replace("_", " "))


def _human_evidence_class(evidence_class: str) -> str:
    mapping = {
        "previously_seen_reusable_holdout": "reusable holdout",
        "blocked_by_contract": "blocked by contract",
        "blocked_by_data": "blocked by data",
    }
    return mapping.get(evidence_class, evidence_class.replace("_", " "))


def _human_task_type(task_type: str) -> str:
    return {"regression": "regression", "binary": "classification", "status": "status"}.get(task_type, task_type)


def _human_split_scope(scope: str) -> str:
    mapping = {
        "known-holdout confirmation": "known-holdout confirmation",
        "no valid label contract": "no valid label contract",
        "no development-only feature source": "no dev-only feature source",
        "mixed seven-target overview": "seven-target overview",
    }
    return mapping.get(scope, scope.replace("_", " "))


def _card_frame(ax: plt.Axes, facecolor: str, edgecolor: str) -> None:
    ax.set_axis_off()
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.01, 0.02),
            0.98,
            0.96,
            boxstyle="round,pad=0.018,rounding_size=0.03",
            linewidth=1.0,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False,
        )
    )


def _state_chip(ax: plt.Axes, label: str, color: str) -> None:
    width = min(0.74, max(0.54, 0.04 * len(label) + 0.14))
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.05, 0.80),
            width,
            0.11,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0,
            facecolor=color,
            transform=ax.transAxes,
        )
    )
    ax.text(0.08, 0.855, label, transform=ax.transAxes, ha="left", va="center", fontsize=8.3, fontweight="bold", color="white")


def _pill(ax: plt.Axes, xy: tuple[float, float], width: float, label: str, color: str, fontsize: float = 8.3) -> None:
    x, y = xy
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            width,
            0.11,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=0,
            facecolor=color,
            transform=ax.transAxes,
        )
    )
    ax.text(x + 0.02, y + 0.055, label, transform=ax.transAxes, ha="left", va="center", fontsize=fontsize, fontweight="bold", color="white")


def _target_card(ax: plt.Axes, letter: str, target_id: str, meta: dict[str, Any]) -> None:
    colors = {
        "confirmed_known_holdout": "#3951A2",
        "not_feasible": "#9A7B29",
        "blocked": "#A80326",
    }
    edge = {
        "confirmed_known_holdout": "#C7D8F0",
        "not_feasible": "#F0DCA1",
        "blocked": "#F3B4B0",
    }
    _card_frame(ax, "#FFFFFF", edge.get(meta["status"], "#D8D8D8"))
    _panel(ax, letter)
    _state_chip(ax, _human_status(meta["status"]), colors.get(meta["status"], "#6B7280"))
    name = meta["name"]
    lines = [
        _human_task_type(meta["task_type"]),
        f"metric: {_metric_direction_text(meta)}",
        f"evidence: {_human_evidence_class(meta['evidence_class'])}",
        f"scope: {_human_split_scope(meta['split_scope'])}",
    ]
    if meta["status"] == "blocked":
        lines.append("reason: no dev-only source")
    elif meta["status"] == "not_feasible":
        lines.append("reason: proxy only")
    ax.text(0.08, 0.70, f"{target_id}  {name}", transform=ax.transAxes, ha="left", va="top", fontsize=9.2, fontweight="bold", color="#111111")
    ax.text(0.08, 0.60, _compact_lines(lines, width=21), transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color="#222222", linespacing=1.35)


def _target_card_image(target_id: str) -> Path:
    return TARGET_META[target_id]["source_figure"]


def _t1_or_t3_target(data: dict[str, Any], target_id: str, output_dir: Path) -> dict[str, Any]:
    meta = TARGET_META[target_id]
    models = list(data["models"])
    scatter = list(data["scatter"])
    group_error = list(data["group_error"])
    cell_metrics = list(data["cell_metrics"])
    fold_labels = [f"fold {fold}" for fold in data["folds"]]
    fig = plt.figure(figsize=(12.6, 7.2))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 0.78], hspace=0.28, wspace=0.20)
    for idx, model_id in enumerate(models[:3]):
        ax = fig.add_subplot(gs[0, idx])
        xs, ys = _scatter_points(scatter, model_id)
        if not xs:
            raise ValueError(f"{target_id} missing scatter rows for model={model_id}")
        color = MODEL_COLORS.get(model_id, PALETTE_AKUN_4[idx % len(PALETTE_AKUN_4)])
        ax.scatter(xs, ys, s=8, alpha=0.26, color=color, edgecolors="none")
        _make_identity_axis(ax, xs + ys)
        ax.set_xlabel("actual")
        ax.set_ylabel("prediction")
        metric_value = float(np.mean([row["primary_value"] for row in cell_metrics if row["model_id"] == model_id]))
        ax.text(
            0.05,
            0.05,
            f"{model_id}\n{meta['primary_metric']}: {metric_value:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
        )
        _panel(ax, chr(ord("a") + idx))
    heat = np.full((len(models), len(fold_labels)), np.nan, dtype=float)
    for row in cell_metrics:
        model_idx = models.index(row["model_id"])
        fold_idx = data["folds"].index(row["fold_id"])
        heat[model_idx, fold_idx] = float(row["primary_value"])
    ax_heat = fig.add_subplot(gs[1, :2])
    cmap = plt.colormaps.get_cmap("Blues")
    im = ax_heat.imshow(heat, cmap=cmap, aspect="auto")
    ax_heat.set_xticks(np.arange(len(fold_labels)), fold_labels)
    ax_heat.set_yticks(np.arange(len(models)), models)
    ax_heat.set_xlabel("fold")
    ax_heat.set_ylabel("model family")
    for i in range(len(models)):
        for j in range(len(fold_labels)):
            value = heat[i, j]
            if not np.isnan(value):
                ax_heat.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7.0, color="#111111")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.034, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(meta["primary_metric"], fontsize=7.5)
    _panel(ax_heat, "d")
    ax = fig.add_subplot(gs[1, 2])
    if target_id == "T1":
        note = _json_load(P5_STAGE4_SUMMARY)
        stage4_commit = note.get("baseline_commit") or note.get("stage3_commit") or "unknown"
        evidence_lines = [
            "stage4: known holdout",
            f"prior test: {note['contract']['prior_exposure']['T1']['prior_test_consumed']}",
            "split: known-holdout confirmation",
            f"commit: {_short_commit(stage4_commit)}",
        ]
    else:
        p7 = _json_load(P7_T3_SUMMARY)
        p8 = _json_load(P8_T3_SUMMARY)
        evidence_lines = [
            "foundation evidence",
            f"xgb MAE: {p7['archived_p5_baseline']['primary_mean']:.3f}",
            f"chronos2 MAE: {p7['decision']['selected_macro_fold_mae']:.3f}",
            f"p8 state: {p8['decision']['state'].replace('_', ' ')}",
        ]
    _card_frame(ax, "#FBFBFC", "#D0D0D0")
    _panel(ax, "e")
    ax.text(0.08, 0.88, "evidence", transform=ax.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", color="#111111")
    ax.text(0.08, 0.78, _compact_lines([target_id, meta["name"], _human_status(meta["status"]), _human_evidence_class(meta["evidence_class"]), meta["split_scope"], *evidence_lines], width=21), transform=ax.transAxes, ha="left", va="top", fontsize=6.8, linespacing=1.26, color="#222222")
    return _save_bundle(fig, f"{target_id}_{meta['detail_kind']}_diagnostics", output_dir)


def _classification_summary_table(ax: plt.Axes, data: dict[str, Any]) -> None:
    rows = []
    group_error = list(data["group_error"])
    cell_metrics = list(data["cell_metrics"])
    models = sorted({row["model_id"] for row in cell_metrics})
    for model in models:
        model_cells = [row for row in cell_metrics if row["model_id"] == model]
        model_groups = [row for row in group_error if row.get("model_id") == model]
        ap = float(np.mean([row["primary_value"] for row in model_cells]))
        brier = float(np.mean([row["metrics"]["brier"] for row in model_groups]))
        f1 = float(np.mean([row["metrics"]["f1_at_0_5"] for row in model_groups]))
        rows.append([model, f"{ap:.3f}", f"{brier:.3f}", f"{f1:.3f}", str(len(model_cells))])
    table = ax.table(
        cellText=rows,
        colLabels=["model", "AP", "Brier", "F1@0.5", "cells"],
        loc="center",
        cellLoc="center",
        bbox=[0.03, 0.12, 0.62, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.2)
    table.set_zorder(3)
    ax.axis("off")


def _classification_target(data: dict[str, Any], target_id: str, output_dir: Path) -> dict[str, Any]:
    meta = TARGET_META[target_id]
    fig = plt.figure(figsize=(12.2, 7.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.9], hspace=0.28, wspace=0.22)
    ax_pr = fig.add_subplot(gs[0, 0])
    ax_cal = fig.add_subplot(gs[0, 1])
    ax_tbl = fig.add_subplot(gs[1, 0])
    ax_note = fig.add_subplot(gs[1, 1])
    precision_recall = {row["model_id"]: row for row in data["precision_recall"]}
    calibration = {row["model_id"]: row for row in data["calibration"]}
    model_order = [row["model_id"] for row in data["precision_recall"]]
    seen: list[str] = []
    for model in model_order:
        if model in seen:
            continue
        seen.append(model)
    for idx, model in enumerate(seen):
        pr = precision_recall.get(model)
        cal = calibration.get(model)
        if pr is None or cal is None:
            continue
        color = MODEL_COLORS.get(model, PALETTE_AKUN_4[idx % len(PALETTE_AKUN_4)])
        ax_pr.plot(pr["recall"], pr["precision"], color=color, linewidth=1.2, label=model)
        bins = cal["bins"]
        mean_probability = [float(bin_row["mean_probability"]) for bin_row in bins]
        positive_fraction = [float(bin_row["positive_fraction"]) for bin_row in bins]
        ax_cal.plot(mean_probability, positive_fraction, marker="o", markersize=3.5, linewidth=1.1, color=color, label=model)
    ax_pr.plot([0, 1], [0.5, 0.5], color="#BBBBBB", linestyle=":", linewidth=0.8)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.legend(frameon=False, loc="lower left")
    _panel(ax_pr, "a")
    ax_cal.plot([0, 1], [0, 1], color="#444444", linestyle="--", linewidth=1)
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel("mean predicted risk")
    ax_cal.set_ylabel("event fraction")
    ax_cal.legend(frameon=False, loc="lower right")
    _panel(ax_cal, "b")
    _card_frame(ax_tbl, "#FBFBFC", "#D0D0D0")
    _panel(ax_tbl, "c")
    _classification_summary_table(ax_tbl, data)
    _card_frame(ax_note, "#FBFBFC", "#D0D0D0")
    _panel(ax_note, "d")
    ax_note.text(0.08, 0.88, "evidence", transform=ax_note.transAxes, ha="left", va="top", fontsize=8.5, fontweight="bold", color="#111111")
    ax_note.text(
        0.08,
        0.78,
        _compact_lines(
            [
                _human_status(meta["status"]),
                f"split: {_human_split_scope(meta['split_scope'])}",
                f"evidence: {_human_evidence_class(meta['evidence_class'])}",
                f"commit: {_short_commit(_json_load(P5_STAGE4_SUMMARY).get('baseline_commit') or _json_load(P5_STAGE4_SUMMARY).get('stage3_commit') or 'unknown')}",
                "PR/calibration only; not field truth.",
            ],
            width=20,
        ),
        transform=ax_note.transAxes,
        va="top",
        ha="left",
        fontsize=6.8,
        linespacing=1.22,
        color="#222222",
    )
    return _save_bundle(fig, f"{target_id}_{meta['detail_kind']}_diagnostics", output_dir)


def _status_target(data: dict[str, Any], target_id: str, output_dir: Path) -> dict[str, Any]:
    meta = TARGET_META[target_id]
    fig = plt.figure(figsize=(9.6, 5.4))
    ax = fig.add_subplot(111)
    _card_frame(ax, "#FFFFFF", "#D4D4D4")
    _panel(ax, "a")
    state = _human_status(meta["status"])
    badge_color = {"not feasible": "#9A7B29", "blocked": "#A80326"}.get(state, "#3951A2")
    _pill(ax, (0.12, 0.82), 0.16, target_id, "#3951A2", fontsize=9.0)
    _pill(ax, (0.31, 0.82), 0.34 if len(state) < 15 else 0.42, state, badge_color, fontsize=8.7)
    ax.plot([0.08, 0.92], [0.74, 0.74], transform=ax.transAxes, color="#D6D6D6", linewidth=1.0, solid_capstyle="round")

    left_lines = [
        f"name: {meta['name']}",
        f"state: {state}",
        f"cause: {_human_evidence_class(meta['evidence_class'])}",
        f"scope: {_human_split_scope(meta['split_scope'])}",
        f"commit: {_short_commit(_source_commit(meta['stage3_json']))}",
    ]
    right_lines = [meta["caveat"]]
    if target_id in {"T6", "T7"}:
        right_lines.insert(0, f"dev source: {'available' if data['development_feature_source_available'] else 'unavailable'}")
        right_lines.insert(1, "test h5 fallback forbidden")
    if target_id == "T5":
        right_lines.insert(0, f"label generated: {'yes' if data['label_generated'] else 'no'}")
        right_lines.insert(1, f"expected cells: {data['expected_training_cells']}")

    ax.text(0.10, 0.67, _compact_lines(left_lines, width=26), transform=ax.transAxes, va="top", ha="left", fontsize=8.0, linespacing=1.28, color="#222222")
    ax.text(0.56, 0.67, _compact_lines(right_lines, width=24), transform=ax.transAxes, va="top", ha="left", fontsize=7.9, linespacing=1.26, color="#222222")
    ax.plot([0.50, 0.50], [0.14, 0.72], transform=ax.transAxes, color="#E4E4E4", linewidth=1.0)
    return _save_bundle(fig, f"{target_id}_{meta['detail_kind']}_card", output_dir)


def _overview(output_dir: Path) -> dict[str, Any]:
    fig = plt.figure(figsize=(14.2, 7.8))
    gs = GridSpec(2, 4, figure=fig, hspace=0.16, wspace=0.10)
    for idx, target_id in enumerate(TARGET_ORDER):
        r, c = divmod(idx, 4)
        ax = fig.add_subplot(gs[r, c])
        _target_card(ax, chr(ord("a") + idx), target_id, TARGET_META[target_id])
    legend_ax = fig.add_subplot(gs[1, 3])
    _card_frame(legend_ax, "#FBFBFC", "#D0D0D0")
    _panel(legend_ax, "h")
    legend_ax.text(0.08, 0.88, "legend", transform=legend_ax.transAxes, ha="left", va="top", fontsize=8.7, fontweight="bold")
    legend_lines = [
        "regression: actual vs prediction",
        "classification: PR + calibration",
        "status: contract gate only",
        "green = confirmed known holdout",
        "amber = not feasible",
        "red = blocked by data",
        "no legacy rasters",
    ]
    legend_ax.text(0.08, 0.76, _compact_lines(legend_lines, width=28), transform=legend_ax.transAxes, ha="left", va="top", fontsize=7.2, linespacing=1.20, color="#222222")
    return _save_bundle(fig, "overview_seven_targets", output_dir)


def _collect_source_shapes(data: dict[str, Any], target_id: str) -> dict[str, Any]:
    if target_id in {"T1", "T2", "T3", "T4"}:
        return {
            "scatter_points": len(data["scatter"]),
            "models": len(data["models"]),
            "folds": len(data["folds"]),
            "cell_metrics": len(data["cell_metrics"]),
        }
    return {"status_rows": len(data)}


def _input_record(path: Path, role: str, split_scope: str, shape_or_row_count: Any, scientific_role: str | None = None) -> dict[str, Any]:
    return {
        "path": _repo_path(path),
        "sha256": _sha256_file(path),
        "shape_or_row_count": shape_or_row_count,
        "scientific_role": scientific_role or role.replace("_", " "),
        "split_scope": split_scope,
        "role": role,
        "source_commit": _source_commit(path),
    }


def _source_inputs_for_target(target_id: str, data: dict[str, Any], figure_entry: dict[str, Any]) -> list[dict[str, Any]]:
    meta = TARGET_META[target_id]
    inputs = [
        _input_record(
            meta["stage3_json"],
            "archived_visualization_data",
            meta["split_scope"],
            _collect_source_shapes(data, target_id),
            "archived stage3 evidence data",
        ),
        _input_record(
            meta["source_figure"],
            "original_target_plot",
            meta["split_scope"],
            _image_size(meta["source_figure"]),
            "original archived plot",
        ),
    ]
    if target_id in {"T1", "T2", "T3", "T4"}:
        inputs.append(
            _input_record(
                P5_STAGE4_SUMMARY,
                "stage4_confirmation_summary",
                "known-holdout confirmation",
                {"keys": len(_json_load(P5_STAGE4_SUMMARY))},
                "stage 4 confirmation summary",
            )
        )
    if target_id == "T3":
        inputs.append(
            _input_record(
                P7_T3_SUMMARY,
                "foundation_evidence_summary_p7",
                "known-holdout confirmation",
                {"methods": len(_json_load(P7_T3_SUMMARY)["methods"])},
                "foundation evidence summary p7",
            )
        )
        inputs.append(
            _input_record(
                P8_T3_SUMMARY,
                "foundation_evidence_summary_p8",
                "known-holdout confirmation",
                {"methods": len(_json_load(P8_T3_SUMMARY)["methods"])},
                "foundation evidence summary p8",
            )
        )
    if target_id == "T4":
        inputs.append(
            _input_record(
                P7_T3_SUMMARY,
                "foundation_evidence_summary_p7",
                "known-holdout confirmation",
                {"methods": len(_json_load(P7_T3_SUMMARY)["methods"])},
                "foundation evidence summary p7",
            )
        )
    return inputs


def _image_size(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        return {"width_px": image.size[0], "height_px": image.size[1]}


def _output_record(role: str, bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "path": bundle["png"],
        "sha256": bundle["png_sha256"],
        "width_px": bundle["width_px"],
        "height_px": bundle["height_px"],
        "dpi": 300,
        "vector_companions": [bundle["svg"], bundle["pdf"]],
    }


def _figure_entry(target_id: str, data: dict[str, Any], bundle: dict[str, Any], source_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    meta = TARGET_META[target_id]
    entry = {
        "figure_name": f"{target_id}_{meta['detail_kind']}_diagnostics",
        "target_id": target_id,
        "target_name": meta["name"],
        "task_type": meta["task_type"],
        "status": meta["status"],
        "evidence_class": meta["evidence_class"],
        "split_scope": meta["split_scope"],
        "metric_direction": _metric_direction_text(meta),
        "primary_metric": meta["primary_metric"],
        "source_commit": _source_commit(meta["stage3_json"]),
        "figure_paths": bundle,
        "source_inputs": source_inputs,
        "visual_qa": {
            "no_titles": True,
            "fonts_normalized": True,
            "palette": "Akun_UKIYOE_4",
            "vector_outputs": ["svg", "pdf"],
            "original_figure_reused": False,
            "manual_review_pending": True,
        },
        "caveat": meta["caveat"],
        "dimensions_px": {"width_px": bundle["width_px"], "height_px": bundle["height_px"]},
    }
    if target_id in {"T3", "T4"}:
        entry["foundation_context"] = {
            "p7_summary": str(P7_T3_SUMMARY.relative_to(REPO_ROOT)),
            "p8_summary": str(P8_T3_SUMMARY.relative_to(REPO_ROOT)) if target_id == "T3" else None,
        }
    return entry


def _validate_target_payload(target_id: str, data: dict[str, Any]) -> None:
    meta = TARGET_META[target_id]
    if target_id in {"T5", "T6", "T7"}:
        required = {"status", "label_generated", "expected_training_cells", "development_feature_source_available"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{target_id} missing required keys: {missing}")
        if data["status"] not in {"not_feasible", "blocked"}:
            raise ValueError(f"{target_id} invalid gate status: {data['status']}")
        if data["label_generated"]:
            raise ValueError(f"{target_id} must not generate labels")
        return
    required = {"task_type", "primary_metric", "cell_metrics", "folds", "models"}
    if target_id in {"T1", "T2", "T3", "T4"}:
        required |= {"scatter", "group_error"}
    if target_id in {"T2", "T4"}:
        required |= {"precision_recall", "calibration"}
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{target_id} missing required keys: {missing}")
    if data["task_type"] != meta["task_type"]:
        raise ValueError(f"{target_id} task_type mismatch: {data['task_type']} != {meta['task_type']}")
    if data["primary_metric"] != meta["primary_metric"]:
        raise ValueError(f"{target_id} primary_metric mismatch: {data['primary_metric']} != {meta['primary_metric']}")
    if target_id in {"T1", "T3"} and len(data["scatter"]) < 100:
        raise ValueError(f"{target_id} scatter evidence too small to be trustworthy")
    if target_id in {"T2", "T4"} and len(data["precision_recall"]) < 3:
        raise ValueError(f"{target_id} missing per-model precision-recall evidence")


def build(output_dir: Path | None = None) -> dict[str, Any]:
    serif = _apply_style()
    output_dir = output_dir or OUTPUT_DIR
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    stage3 = {target_id: _json_load(TARGET_META[target_id]["stage3_json"]) for target_id in TARGET_ORDER}
    for target_id, data in stage3.items():
        _validate_target_payload(target_id, data)

    generated_figures: list[dict[str, Any]] = []

    overview_bundle = _overview(figure_dir)
    generated_figures.append(
        {
            "figure_name": "overview_seven_targets",
            "figure_paths": overview_bundle,
            "target_ids": list(TARGET_ORDER),
            "source_inputs": [
                _input_record(
                    TARGET_META[target_id]["source_figure"],
                    "thumbnail_source",
                    TARGET_META[target_id]["split_scope"],
                    _image_size(TARGET_META[target_id]["source_figure"]),
                    "archived target plot",
                )
                for target_id in TARGET_ORDER
            ],
            "split_scope": "mixed seven-target overview",
            "caveat": "overview is a compact guide; use target-specific figures for detailed evidence.",
            "visual_qa": {
                "no_titles": True,
                "fonts_normalized": True,
                "palette": "Akun_UKIYOE_4",
                "vector_outputs": ["svg", "pdf"],
                "original_figure_reused": False,
                "manual_review_pending": True,
            },
            "dimensions_px": {"width_px": overview_bundle["width_px"], "height_px": overview_bundle["height_px"]},
        }
    )

    for target_id in TARGET_ORDER:
        data = stage3[target_id]
        if target_id in {"T1", "T3"}:
            bundle = _t1_or_t3_target(data, target_id, figure_dir)
        elif target_id in {"T2", "T4"}:
            bundle = _classification_target(data, target_id, figure_dir)
        else:
            bundle = _status_target(data, target_id, figure_dir)
        generated_figures.append(
            _figure_entry(
                target_id,
                data,
                bundle,
                _source_inputs_for_target(target_id, data, bundle),
            )
        )

    source_commit = _git_output("rev-parse", "HEAD")
    renderer_path = Path(__file__).resolve()
    renderer_sha = _sha256_file(renderer_path)
    contract_inputs = []
    seen_input_paths: set[str] = set()
    for figure in generated_figures:
        for input_record in figure["source_inputs"]:
            if input_record["path"] not in seen_input_paths:
                seen_input_paths.add(input_record["path"])
                contract_inputs.append(
                    {
                        "path": input_record["path"],
                        "sha256": input_record["sha256"],
                        "shape_or_row_count": input_record["shape_or_row_count"],
                        "scientific_role": input_record["scientific_role"],
                        "split_scope": input_record["split_scope"],
                    }
                )
    contract_outputs = [_output_record(figure["figure_name"], figure["figure_paths"]) for figure in generated_figures]
    manifest = {
        "schema_version": "sweetspot-p12-visualization-manifest/v1",
        "track": "sweetspot",
        "track_id": "sweetspot",
        "branch": _git_output("rev-parse", "--abbrev-ref", "HEAD"),
        "head": source_commit,
        "source_commit": source_commit,
        "generator_commit": source_commit,
        "renderer": {
            "path": _repo_path(renderer_path),
            "sha256": renderer_sha,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_caveat": "This deliverable is a visualization contract only; it does not alter labels, splits, or target semantics.",
        "script_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "script_sha256": renderer_sha,
        "output_dir": _repo_path(output_dir),
        "font_family": serif,
        "palette_name": "Akun_UKIYOE_4",
        "figure_count": len(generated_figures),
        "figures": generated_figures,
        "source_registry": {
            "p5_stage3_data": str(P5_STAGE3_DATA.relative_to(REPO_ROOT)),
            "p5_stage3_figures": str(P5_STAGE3_FIGURES.relative_to(REPO_ROOT)),
            "p5_stage4_summary": str(P5_STAGE4_SUMMARY.relative_to(REPO_ROOT)),
            "p7_t3_summary": str(P7_T3_SUMMARY.relative_to(REPO_ROOT)),
            "p8_t3_summary": str(P8_T3_SUMMARY.relative_to(REPO_ROOT)),
        },
        "p12_contract": {
            "schema_version": "scientific-visualization-contract/v1",
            "profile": "p12_tracks_1_3_5",
            "track_id": "sweetspot",
            "source_commit": source_commit,
            "renderer": {
                "path": _repo_path(renderer_path),
                "sha256": renderer_sha,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scientific_caveat": "Visualization only; no label, split, or metric definitions were changed.",
            "inputs": contract_inputs,
            "outputs": contract_outputs,
            "manual_review": {
                "reviewed": False,
                "status": "pending",
                "reviewer": None,
                "reviewed_at": None,
                "reviewed_sha256": None,
                "colors_consistent": None,
                "labels_legible": None,
                "no_clipping": None,
                "no_overlap": None,
                "scientific_boundary_preserved": None,
                "notes": None,
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest["manifest_path"] = _repo_path(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "figures": generated_figures,
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    build(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
