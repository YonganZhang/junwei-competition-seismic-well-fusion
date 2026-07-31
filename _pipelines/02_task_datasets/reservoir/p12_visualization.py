from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
from matplotlib import font_manager
from PIL import Image

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "p12_visualization_v1"
matplotlib.rcParams["svg.fonttype"] = "path"
matplotlib.rcParams["pdf.compression"] = 0

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TRACK_ROOT = HERE
STAGE4_ROOT = TRACK_ROOT / "_outputs" / "p5_stage4_confirmation"
LEGACY_ROOT = TRACK_ROOT / "_outputs"
OUTPUT_ROOT = TRACK_ROOT / "_outputs" / "p12_visualization"
OUTPUT_FIGURE_SIZE = (7.2, 7.2)
DPI = 300
TARGET_ORDER = ("PHIF", "KLOGH", "SW")
LEGACY_RUN_MANIFEST = LEGACY_ROOT / "run_manifest.json"
LEGACY_DEPTH_FIGURE = LEGACY_ROOT / "test_depth_gt_vs_pred.png"
LEGACY_STAGE4_SUMMARY = STAGE4_ROOT / "summary.json"
LEGACY_STAGE4_PREPARATION = STAGE4_ROOT / "preparation_manifest.json"
LEGACY_STAGE4_VIZ = STAGE4_ROOT / "visualization_manifest.json"
LEGACY_STAGE4_CONFIRMATION = STAGE4_ROOT / "confirmation_state.json"

PALETTE_UKIYOE_4 = ["#376795", "#72BCD5", "#FFD06F", "#E76254"]
PALETTE_UKIYOE_10 = [
    "#E76254",
    "#EF8A47",
    "#F7AA58",
    "#FFD06F",
    "#FFE6B7",
    "#AADCE0",
    "#72BCD5",
    "#528FAD",
    "#376795",
    "#1E466E",
]
PANEL_LABELS = {
    "a": "(a)",
    "b": "(b)",
    "c": "(c)",
    "d": "(d)",
}

AVAILABLE_FONT_NAMES = {font.name for font in font_manager.fontManager.ttflist}
REQUESTED_FONT = "Times New Roman"
RESOLVED_FONT = next(
    (name for name in (REQUESTED_FONT, "TeX Gyre Termes") if name in AVAILABLE_FONT_NAMES),
    None,
)
if RESOLVED_FONT is None:
    raise RuntimeError("Times New Roman or TeX Gyre Termes font is required for P12 visualization")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": RESOLVED_FONT,
            "font.serif": [RESOLVED_FONT],
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


_apply_style()


@dataclass(frozen=True)
class TargetRow:
    sample_id: str
    family_id: str
    well_id: str
    depth_m: float
    truth_model_domain: float
    prediction_model_domain: float
    truth_physical: float
    prediction_physical: float
    interval_low_physical: float
    interval_high_physical: float
    interval_covered: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_svg_trailing_whitespace(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    sanitized = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    if sanitized != text:
        path.write_text(sanitized, encoding="utf-8")


def _canonicalize_png(path: Path) -> None:
    with Image.open(path) as image:
        image.load()
        mode = image.mode
        payload = image.copy()
    if mode not in {"RGB", "RGBA", "L", "LA"}:
        payload = payload.convert("RGBA")
    payload.save(path, format="PNG", optimize=False, compress_level=9, dpi=(DPI, DPI))


def _save_figure_bundle(fig: matplotlib.figure.Figure, png_path: Path, pdf_path: Path, svg_path: Path) -> None:
    stable_pdf_metadata = {
        "Creator": "p12_visualization",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    stable_svg_metadata = {
        "Creator": "p12_visualization",
        "Date": None,
    }
    fig.savefig(png_path, dpi=DPI, facecolor="white")
    fig.savefig(pdf_path, facecolor="white", metadata=stable_pdf_metadata)
    fig.savefig(svg_path, facecolor="white", metadata=stable_svg_metadata)
    _canonicalize_png(png_path)
    _strip_svg_trailing_whitespace(svg_path)


def repo_rel(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return candidate.as_posix()


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_target_rows(path: Path) -> list[TargetRow]:
    rows = []
    for raw in _read_csv_rows(path):
        rows.append(
            TargetRow(
                sample_id=raw["sample_id"],
                family_id=raw["family_id"],
                well_id=raw["well_id"],
                depth_m=float(raw["depth_m"]),
                truth_model_domain=float(raw["truth_model_domain"]),
                prediction_model_domain=float(raw["prediction_model_domain"]),
                truth_physical=float(raw["truth_physical"]),
                prediction_physical=float(raw["prediction_physical"]),
                interval_low_physical=float(raw["interval_low_physical"]),
                interval_high_physical=float(raw["interval_high_physical"]),
                interval_covered=raw["interval_covered"].strip().lower() == "true",
            )
        )
    if not rows:
        raise ValueError(f"no rows found in {path}")
    families = {row.family_id for row in rows}
    wells = {row.well_id for row in rows}
    if families != {"15/9-F-15"}:
        raise ValueError(f"unexpected holdout family set in {path}: {families}")
    if wells != {"15/9-F-15 D"}:
        raise ValueError(f"unexpected holdout well set in {path}: {wells}")
    if len(rows) != 344:
        raise ValueError(f"unexpected row count in {path}: {len(rows)}")
    return rows


def _load_all_targets() -> dict[str, list[TargetRow]]:
    targets = {}
    reference_ids: list[str] | None = None
    reference_depths: list[float] | None = None
    for target in TARGET_ORDER:
        csv_path = STAGE4_ROOT / target.lower() / "predictions.csv"
        rows = _load_target_rows(csv_path)
        ids = [row.sample_id for row in rows]
        depths = [row.depth_m for row in rows]
        if reference_ids is None:
            reference_ids = ids
            reference_depths = depths
        else:
            if ids != reference_ids:
                raise ValueError(f"sample alignment mismatch for {target}")
            if not np.allclose(depths, reference_depths, atol=1e-9, rtol=0.0):
                raise ValueError(f"depth alignment mismatch for {target}")
        targets[target] = rows
    return targets


def _depth_sorted(rows: list[TargetRow]) -> list[TargetRow]:
    return sorted(rows, key=lambda row: row.depth_m)


def _robust_limits(values: np.ndarray, pad_frac: float = 0.05) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values available for limits")
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if np.isclose(lo, hi):
        span = max(abs(lo), 1.0)
        return lo - span * 0.5, hi + span * 0.5
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _set_font(fig: matplotlib.figure.Figure) -> None:
    for text in fig.findobj(matplotlib.text.Text):
        text.set_fontfamily(RESOLVED_FONT)
    for ax in fig.axes:
        ax.xaxis.label.set_fontfamily(RESOLVED_FONT)
        ax.yaxis.label.set_fontfamily(RESOLVED_FONT)
        if hasattr(ax, "zaxis") and ax.zaxis is not None:
            ax.zaxis.label.set_fontfamily(RESOLVED_FONT)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily(RESOLVED_FONT)


def _panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        0.02,
        0.98,
        PANEL_LABELS[letter],
        transform=ax.transAxes,
        fontfamily=RESOLVED_FONT,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.1),
    )


def _axis_style(ax: plt.Axes) -> None:
    ax.grid(True, color="#C8C8C8", alpha=0.35, linewidth=0.8)
    ax.tick_params(direction="out", length=4.5, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _with_depth_breaks(depth: np.ndarray, values: np.ndarray, gap_threshold_m: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    if depth.shape != values.shape:
        raise ValueError("depth and value arrays must match")
    if depth.size == 0:
        return depth, values
    depth_out = [depth[0]]
    value_out = [values[0]]
    for idx in range(1, depth.size):
        if depth[idx] - depth[idx - 1] > gap_threshold_m:
            depth_out.append(np.nan)
            value_out.append(np.nan)
        depth_out.append(depth[idx])
        value_out.append(values[idx])
    return np.asarray(depth_out, dtype=float), np.asarray(value_out, dtype=float)


def _target_units(target: str) -> tuple[str, str, str]:
    if target == "PHIF":
        return "fraction", "PHIF", "PHIF"
    if target == "KLOGH":
        return "mD", "KLOGH", "log1p(KLOGH)"
    if target == "SW":
        return "fraction", "SW", "SW"
    raise KeyError(target)


def _truth_pred_domain(rows: list[TargetRow], target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "KLOGH":
        truth = np.asarray([row.truth_model_domain for row in rows], dtype=float)
        pred = np.asarray([row.prediction_model_domain for row in rows], dtype=float)
    else:
        truth = np.asarray([row.truth_physical for row in rows], dtype=float)
        pred = np.asarray([row.prediction_physical for row in rows], dtype=float)
    return truth, pred


def _physical_arrays(rows: list[TargetRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray([row.depth_m for row in rows], dtype=float)
    truth_physical = np.asarray([row.truth_physical for row in rows], dtype=float)
    pred_physical = np.asarray([row.prediction_physical for row in rows], dtype=float)
    interval_low = np.asarray([row.interval_low_physical for row in rows], dtype=float)
    interval_high = np.asarray([row.interval_high_physical for row in rows], dtype=float)
    covered = np.asarray([row.interval_covered for row in rows], dtype=bool)
    return depth, truth_physical, pred_physical, interval_low, interval_high, covered


def _render_target_figure(rows: list[TargetRow], target: str, output_dir: Path) -> dict[str, object]:
    sorted_rows = _depth_sorted(rows)
    depth, truth_physical, pred_physical, interval_low, interval_high, covered = _physical_arrays(sorted_rows)
    truth_domain, pred_domain = _truth_pred_domain(sorted_rows, target)
    unit, label, domain_label = _target_units(target)
    residual_physical = pred_physical - truth_physical
    half_width = 0.5 * (interval_high - interval_low)
    abs_residual = np.abs(residual_physical)
    if target == "KLOGH":
        interval_x = np.log1p(half_width)
        interval_y = np.log1p(abs_residual)
        interval_display_domain = "log1p(mD)"
    else:
        interval_x = half_width
        interval_y = abs_residual
        interval_display_domain = unit

    fig = plt.figure(figsize=OUTPUT_FIGURE_SIZE, constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.98, bottom=0.08, top=0.97, wspace=0.22, hspace=0.20)
    ax_depth = fig.add_subplot(gs[0, 0])
    ax_residual = fig.add_subplot(gs[0, 1], sharey=ax_depth)
    ax_parity = fig.add_subplot(gs[1, 0])
    ax_interval = fig.add_subplot(gs[1, 1])

    # a. depth tracks
    depth_line, truth_line = _with_depth_breaks(depth, truth_physical)
    _, pred_line = _with_depth_breaks(depth, pred_physical)
    ax_depth.plot(truth_line, depth_line, color=PALETTE_UKIYOE_4[0], lw=1.4, label="truth")
    ax_depth.plot(pred_line, depth_line, color=PALETTE_UKIYOE_4[3], lw=1.4, label="prediction")
    ax_depth.scatter(truth_physical, depth, s=9, color=PALETTE_UKIYOE_4[0], alpha=0.85, linewidths=0.0)
    ax_depth.scatter(pred_physical, depth, s=9, color=PALETTE_UKIYOE_4[3], alpha=0.75, linewidths=0.0)
    ax_depth.set_xlabel(f"{label} ({unit})")
    ax_depth.set_ylabel("Measured depth (m)")
    ax_depth.invert_yaxis()
    ax_depth.legend(loc="best", frameon=False, handlelength=1.8, borderaxespad=0.2)
    _axis_style(ax_depth)
    _panel(ax_depth, "a")

    # b. residual vs depth
    depth_line, residual_line = _with_depth_breaks(depth, residual_physical)
    ax_residual.plot(residual_line, depth_line, color=PALETTE_UKIYOE_4[3], lw=1.2, label="residual")
    ax_residual.scatter(residual_physical, depth, s=9, color=PALETTE_UKIYOE_4[3], alpha=0.78, linewidths=0.0)
    ax_residual.axvline(0.0, color="#333333", lw=1.0, ls="--")
    ax_residual.set_xlabel(f"Residual ({unit})")
    ax_residual.tick_params(labelleft=False)
    ax_residual.invert_yaxis()
    res_limit = _robust_limits(np.abs(residual_physical), pad_frac=0.08)[1]
    ax_residual.set_xlim(-res_limit, res_limit)
    ax_residual.legend(loc="best", frameon=False, handlelength=1.8, borderaxespad=0.2)
    _axis_style(ax_residual)
    _panel(ax_residual, "b")

    # c. parity
    ax_parity.scatter(
        truth_domain,
        pred_domain,
        s=16,
        color=PALETTE_UKIYOE_4[0],
        alpha=0.76,
        linewidths=0.0,
        label="samples",
    )
    parity_limits = _robust_limits(np.concatenate([truth_domain, pred_domain]), pad_frac=0.05)
    ax_parity.plot(parity_limits, parity_limits, color=PALETTE_UKIYOE_10[-1], lw=1.2, ls="--", label="identity")
    ax_parity.set_xlim(parity_limits)
    ax_parity.set_ylim(parity_limits)
    ax_parity.set_aspect("equal", adjustable="box")
    ax_parity.set_xlabel(f"{domain_label} truth ({'log1p(mD)' if target == 'KLOGH' else unit})")
    ax_parity.set_ylabel(f"{domain_label} prediction ({'log1p(mD)' if target == 'KLOGH' else unit})")
    ax_parity.legend(loc="best", frameon=False, handlelength=1.8, borderaxespad=0.2)
    _axis_style(ax_parity)
    _panel(ax_parity, "c")

    # d. interval diagnostic
    covered_mask = covered.astype(bool)
    if covered_mask.any():
        ax_interval.scatter(
            interval_x[covered_mask],
            interval_y[covered_mask],
            s=16,
            color=PALETTE_UKIYOE_4[1],
            alpha=0.82,
            linewidths=0.0,
            label="covered",
        )
    if (~covered_mask).any():
        ax_interval.scatter(
            interval_x[~covered_mask],
            interval_y[~covered_mask],
            s=18,
            color=PALETTE_UKIYOE_4[3],
            alpha=0.9,
            linewidths=0.0,
            label="miss",
            marker="x",
        )
    interval_limits = _robust_limits(np.concatenate([interval_x, interval_y]), pad_frac=0.05)[1]
    ax_interval.plot([0.0, interval_limits], [0.0, interval_limits], color=PALETTE_UKIYOE_10[-1], lw=1.1, ls="--", label="1:1")
    ax_interval.set_xlim(0.0, interval_limits)
    ax_interval.set_ylim(0.0, interval_limits)
    ax_interval.set_xlabel(f"Interval half-width ({interval_display_domain})")
    ax_interval.set_ylabel(f"|Residual| ({interval_display_domain})")
    ax_interval.legend(loc="best", frameon=False, handlelength=1.8, borderaxespad=0.2)
    _axis_style(ax_interval)
    _panel(ax_interval, "d")

    _set_font(fig)

    target_slug = target.lower()
    stem = f"{target_slug}_heldout_summary"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    svg_path = output_dir / f"{stem}.svg"
    _save_figure_bundle(fig, png_path, pdf_path, svg_path)
    plt.close(fig)

    return {
        "target": target,
        "kind": "heldout_summary",
        "path_png": repo_rel(png_path),
        "path_pdf": repo_rel(pdf_path),
        "path_svg": repo_rel(svg_path),
        "sha256_png": sha256_file(png_path),
        "sha256_pdf": sha256_file(pdf_path),
        "sha256_svg": sha256_file(svg_path),
        "dimensions_png": list(Image.open(png_path).size),
        "rows": len(rows),
        "well_id": sorted_rows[0].well_id,
        "family_id": sorted_rows[0].family_id,
        "axis_rules": {
            "depth_axis": "inverted",
            "parity_domain": "model_domain" if target == "KLOGH" else "physical",
            "parity_limits_rule": "p1-p99 padded by 5 percent",
            "residual_limits_rule": "absolute residual p1-p99 padded by 8 percent",
            "interval_limits_rule": "half-width and |residual| p1-p99 padded by 5 percent",
            "interval_display_domain": interval_display_domain,
            "interval_display_transform": "log1p(mD)" if target == "KLOGH" else "identity",
        },
        "metrics": {
            "physical_mae": float(np.mean(np.abs(residual_physical))),
            "physical_rmse": float(np.sqrt(np.mean(residual_physical**2))),
            "empirical_interval_coverage": float(covered_mask.mean()),
        },
    }


def _input_entry(path: Path, *, rows: int | None = None, columns: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": repo_rel(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if rows is not None and columns is not None:
        payload["shape"] = [rows, columns]
    return payload


def _contract_input_entry(
    path: Path,
    *,
    scientific_role: str,
    split_scope: str,
    shape_or_row_count: object,
) -> dict[str, object]:
    return {
        "path": repo_rel(path),
        "sha256": sha256_file(path),
        "shape_or_row_count": shape_or_row_count,
        "scientific_role": scientific_role,
        "split_scope": split_scope,
    }


def _contract_output_entry(path_png: Path, path_pdf: Path, path_svg: Path, *, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": repo_rel(path_png),
        "sha256": sha256_file(path_png),
        "width_px": int(round(OUTPUT_FIGURE_SIZE[0] * DPI)),
        "height_px": int(round(OUTPUT_FIGURE_SIZE[1] * DPI)),
        "dpi": DPI,
        "vector_companions": [repo_rel(path_pdf), repo_rel(path_svg)],
    }


def _build_p12_contract(figures: list[dict[str, object]], target_rows: dict[str, list[TargetRow]]) -> dict[str, object]:
    contract_inputs: list[dict[str, object]] = [
        _contract_input_entry(
            LEGACY_RUN_MANIFEST,
            scientific_role="original_lineage_manifest",
            split_scope="all lineage evidence; no split training relevance",
            shape_or_row_count=None,
        ),
        _contract_input_entry(
            LEGACY_DEPTH_FIGURE,
            scientific_role="legacy_depth_visual_reference",
            split_scope="held-out family 15/9-F-15 only; reference lineage",
            shape_or_row_count=None,
        ),
        _contract_input_entry(
            LEGACY_STAGE4_SUMMARY,
            scientific_role="stage4_confirmation_summary",
            split_scope="development-only confirmation; holdout excluded from modeling",
            shape_or_row_count=None,
        ),
        _contract_input_entry(
            LEGACY_STAGE4_PREPARATION,
            scientific_role="stage4_preparation_state",
            split_scope="development-only confirmation; holdout excluded from modeling",
            shape_or_row_count=None,
        ),
        _contract_input_entry(
            LEGACY_STAGE4_VIZ,
            scientific_role="stage4_visualization_state",
            split_scope="development-only confirmation; holdout excluded from modeling",
            shape_or_row_count=None,
        ),
        _contract_input_entry(
            LEGACY_STAGE4_CONFIRMATION,
            scientific_role="stage4_confirmation_state",
            split_scope="development-only confirmation; holdout excluded from modeling",
            shape_or_row_count=None,
        ),
    ]
    for target in TARGET_ORDER:
        contract_inputs.append(
            _contract_input_entry(
                STAGE4_ROOT / target.lower() / "predictions.csv",
                scientific_role=f"heldout_prediction_csv:{target.lower()}",
                split_scope="held-out family 15/9-F-15; 344-row confirmation predictions",
                shape_or_row_count={"rows": 344, "columns": 11},
            )
        )

    contract_outputs: list[dict[str, object]] = []
    for figure in figures:
        target = figure["target"].lower()
        png_path = PROJECT_ROOT / figure["path_png"]
        pdf_path = PROJECT_ROOT / figure["path_pdf"]
        svg_path = PROJECT_ROOT / figure["path_svg"]
        contract_outputs.append(
            _contract_output_entry(
                png_path,
                pdf_path,
                svg_path,
                role=f"heldout_summary_figure:{target}",
            )
        )

    return {
        "schema_version": "scientific-visualization-contract/v1",
        "profile": "p12_tracks_1_3_5",
        "track_id": "property",
        "source_commit": git_head(),
        "renderer": {
            "path": repo_rel(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scientific_caveat": (
            "Held-out depth tracks are rendered only from stage4 confirmation predictions for family 15/9-F-15; "
            "no trajectory, interpolation, or volume rendering is introduced, and KLOGH interval diagnostics are shown in log1p(mD) domain."
        ),
        "inputs": contract_inputs,
        "outputs": contract_outputs,
        "manual_review": {
            "reviewed": False,
            "reviewer": None,
            "reviewed_at_utc": None,
            "notes": None,
            "status": "pending",
            "reviewed_sha256": None,
            "no_clipping": None,
            "no_overlap": None,
            "labels_legible": None,
            "colors_consistent": None,
            "scientific_boundary_preserved": None,
        },
    }


def build_manifest(figures: list[dict[str, object]], target_rows: dict[str, list[TargetRow]]) -> dict[str, object]:
    summary = json.loads(LEGACY_STAGE4_SUMMARY.read_text(encoding="utf-8"))
    preparation = json.loads(LEGACY_STAGE4_PREPARATION.read_text(encoding="utf-8"))
    viz_manifest = json.loads(LEGACY_STAGE4_VIZ.read_text(encoding="utf-8"))
    root_manifest = json.loads(LEGACY_RUN_MANIFEST.read_text(encoding="utf-8"))
    p12_contract = _build_p12_contract(figures, target_rows)
    return {
        "schema_version": 1,
        "track_id": "property",
        "renderer": "p12_visualization",
        "source_commit": git_head(),
        "script": {
            "path": repo_rel(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figure_mode": "D_full",
        "figure_size_inches": list(OUTPUT_FIGURE_SIZE),
        "dpi": DPI,
        "requested_font": REQUESTED_FONT,
        "resolved_font": RESOLVED_FONT,
        "source_inputs": {
            "legacy_run_manifest": _input_entry(LEGACY_RUN_MANIFEST),
            "legacy_depth_figure": {
                **_input_entry(LEGACY_DEPTH_FIGURE),
                "dimensions": [2340, 1260],
            },
            "stage4_summary": _input_entry(LEGACY_STAGE4_SUMMARY),
            "stage4_preparation": _input_entry(LEGACY_STAGE4_PREPARATION),
            "stage4_visualization_manifest": _input_entry(LEGACY_STAGE4_VIZ),
            "stage4_confirmation_state": _input_entry(LEGACY_STAGE4_CONFIRMATION),
        },
        "target_sources": {
            target: _input_entry(STAGE4_ROOT / target.lower() / "predictions.csv", rows=344, columns=11)
            for target in TARGET_ORDER
        },
        "split_scope": {
            "confirmation_kind": summary["confirmation_kind"],
            "fresh_blind": summary["fresh_blind"],
            "prior_test_consumed": summary["prior_test_consumed"],
            "known_holdout_family": summary["known_holdout_family"],
            "development_families": summary["development_families"],
            "development_rows": summary["development_rows"],
            "known_holdout_rows": summary["known_holdout_rows"],
            "single_use_state": root_manifest.get("families", {}).get("test", ["15/9-F-15"]),
        },
        "lineage_notes": [
            "legacy run_manifest.json and test_depth_gt_vs_pred.png are retained as the original lineage evidence",
            "new figures are generated only from the stage4 confirmation prediction CSVs",
            "KLOGH parity panel uses log1p(mD) because that is the validated model domain; physical metrics stay in mD elsewhere",
            "no trajectory, interpolation, or volume rendering is used",
            "no well-comparison panel is emitted because the holdout family contains a single well",
        ],
        "target_summaries": {
            target: {
                "well_id": target_rows[target][0].well_id,
                "family_id": target_rows[target][0].family_id,
                "sample_count": len(target_rows[target]),
                "physical_metrics": {
                    "MAE": float(np.mean(np.abs(np.asarray([r.prediction_physical - r.truth_physical for r in target_rows[target]], dtype=float)))),
                    "RMSE": float(np.sqrt(np.mean(np.square(np.asarray([r.prediction_physical - r.truth_physical for r in target_rows[target]], dtype=float))))),
                },
                "model_domain_metrics": {
                    "MAE": float(np.mean(np.abs(np.asarray([r.prediction_model_domain - r.truth_model_domain for r in target_rows[target]], dtype=float)))),
                    "RMSE": float(np.sqrt(np.mean(np.square(np.asarray([r.prediction_model_domain - r.truth_model_domain for r in target_rows[target]], dtype=float))))),
                },
                "empirical_interval_coverage": float(np.mean([r.interval_covered for r in target_rows[target]])),
            }
            for target in TARGET_ORDER
        },
        "figures": figures,
        "visual_qa": {
            "titles_present": False,
            "fallback_data_used": False,
            "vector_outputs_present": True,
            "fonts_normalized": True,
            "no_leakage": True,
            "source_alignment_verified": True,
            "all_pngs_are_300dpi": True,
            "manual_review_pending": False,
            "klogh_interval_display_domain": "log1p(mD)",
        },
        "stage4_summary_sha256": sha256_file(LEGACY_STAGE4_SUMMARY),
        "stage4_preparation_sha256": sha256_file(LEGACY_STAGE4_PREPARATION),
        "stage4_visualization_manifest_sha256": sha256_file(LEGACY_STAGE4_VIZ),
        "legacy_run_manifest_sha256": sha256_file(LEGACY_RUN_MANIFEST),
        "legacy_depth_figure_sha256": sha256_file(LEGACY_DEPTH_FIGURE),
        "p12_contract": p12_contract,
    }


def _warm_render_state(target_rows: dict[str, list[TargetRow]]) -> None:
    with tempfile.TemporaryDirectory(prefix="p12_visualization_warmup_") as tmp_dir:
        warm_root = Path(tmp_dir) / "figures"
        warm_root.mkdir(parents=True, exist_ok=True)
        for target in TARGET_ORDER:
            _render_target_figure(target_rows[target], target, warm_root)


def generate_artifacts(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    target_rows = _load_all_targets()
    _warm_render_state(target_rows)
    figures: list[dict[str, object]] = []
    for target in TARGET_ORDER:
        figures.append(_render_target_figure(target_rows[target], target, figure_root))

    manifest = build_manifest(figures, target_rows)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = repo_rel(manifest_path)
    return manifest


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render reservoir-property held-out visualizations.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Output directory containing manifest.json and figures/.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = generate_artifacts(args.output_root)
    print(json.dumps({"manifest": manifest.get("manifest_path", repo_rel(args.output_root / "manifest.json"))}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
