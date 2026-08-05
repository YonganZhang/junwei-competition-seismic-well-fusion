#!/usr/bin/env python3
"""Build P5 R2 protocol/performance summaries from pinned Git blobs.

These figures are not domain visualizations and are never eligible for the
six-track card-rendering delivery pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_TMP = PROJECT_ROOT / "_tmp" / "p5_r2_visualization"
PROJECT_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(PROJECT_TMP))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_TMP / "mplconfig"))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402


BLUE = "#376795"
CYAN = "#72BCD5"
YELLOW = "#FFD06F"
RED = "#E76254"
DARK = "#1E466E"
GRID = "#D9E2E8"
LIGHT = "#EEF3F6"
COLORS = [BLUE, CYAN, YELLOW, RED]

TRACK_LABELS = [
    "Fault",
    "Seismic facies",
    "Property",
    "Lithofacies",
    "Sweet spot",
    "Reconstruction",
]

SOURCES: dict[str, dict[str, str]] = {
    "fault": {
        "commit": "0ff064d8e8850611df2f6eea0eab1aeebf03721c",
        "worktree": "p5-r2-fault-v2",
        "path": (
            "_pipelines/02_task_datasets/fault/_outputs/"
            "p5_r2_data_acquisition/p5_r2_summary.json"
        ),
    },
    "facies": {
        "commit": "450f3d5840ac8d25bc8f3d5e9029753f1cdbe591",
        "worktree": "p5-r2-facies",
        "path": (
            "_pipelines/02_task_datasets/facies/_outputs/"
            "p5_r2/p5_r2_summary.json"
        ),
    },
    "property": {
        "commit": "65740d49b479b1716f8c1a1d807ca4915d7a1dba",
        "worktree": "p5-r2-property",
        "path": (
            "_pipelines/02_task_datasets/reservoir/_outputs/"
            "p5_r2/p5_r2_summary.json"
        ),
    },
    "lithofacies": {
        "commit": "e5d5cbce2d2e26ce8479d4e6731b32d0bec16362",
        "worktree": "p5-r2-lithofacies",
        "path": (
            "_pipelines/02_task_datasets/lithofacies/_outputs/"
            "p5_r2/p5_r2_summary.json"
        ),
    },
    "sweetspot": {
        "commit": "5f80b5c6fb6fbf80b2af9781809ab48def6b932d",
        "worktree": "p5-r2-sweetspot",
        "path": (
            "_pipelines/02_task_datasets/sweetspot/p5/r02/_outputs/"
            "protocol_r2/p5_r02_summary.json"
        ),
    },
    "reconstruction": {
        "commit": "f6e6efbac1663c3fe6f768ea79c2f1189c45b751",
        "worktree": "p5-r2-reconstruction-v2",
        "path": (
            "_pipelines/02_task_datasets/reconstruction/"
            "p5_r2_evidence/p5_r2_summary.json"
        ),
    },
}


def _run_git(repo_root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def load_sources(
    repo_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads: dict[str, Any] = {}
    manifest: dict[str, Any] = {
        "schema_version": "p5-r2-protocol-visualization-sources/v2",
        "artifact_class": "protocol_and_model_performance_summary",
        "domain_visualization_eligible": False,
        "card_render_policy": "blocked_as_domain_visualization",
        "sources": {},
    }
    snapshot_dir = output_dir / "source_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for track, source in SOURCES.items():
        object_ref = f"{source['commit']}:{source['path']}"
        acquisition = "git_blob"
        try:
            raw = _run_git(repo_root, "show", object_ref)
        except subprocess.CalledProcessError:
            worktree_path = repo_root / ".claude" / "worktrees" / source["worktree"] / source["path"]
            snapshot_path = snapshot_dir / f"{track}.json"
            if worktree_path.is_file():
                raw = worktree_path.read_bytes()
                acquisition = "worktree_file"
            elif snapshot_path.is_file():
                raw = snapshot_path.read_bytes()
                acquisition = "saved_snapshot"
            else:
                raise FileNotFoundError(
                    f"{track}: neither {object_ref}, {worktree_path}, nor {snapshot_path} exists"
                )
        snapshot_path = snapshot_dir / f"{track}.json"
        snapshot_path.write_bytes(raw)
        payloads[track] = json.loads(raw)
        manifest["sources"][track] = {
            **source,
            "object_ref": object_ref,
            "acquisition": acquisition,
            "snapshot_path": str(snapshot_path.relative_to(output_dir)),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return payloads, manifest


def choose_font() -> tuple[str, bool]:
    for family in ("Times New Roman", "TeX Gyre Termes"):
        try:
            findfont(FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        return family, family == "Times New Roman"
    raise RuntimeError(
        "Neither Times New Roman nor the metric-compatible TeX Gyre Termes fallback is installed."
    )


def configure_style(font_family: str) -> None:
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "axes.edgecolor": DARK,
            "axes.facecolor": "white",
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "axes.labelcolor": DARK,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_fonts(figure: mpl.figure.Figure, font_family: str) -> None:
    for item in figure.findobj(match=mpl.text.Text):
        item.set_fontfamily(font_family)


def panel_label(axis: mpl.axes.Axes, label: str) -> None:
    axis.text(
        -0.13,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=DARK,
    )


def clean_axis(axis: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def save_figure(
    figure: mpl.figure.Figure,
    output_dir: Path,
    stem: str,
    font_family: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalize_fonts(figure, font_family)
    outputs: list[dict[str, Any]] = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        metadata = (
            {"Software": "p5_r2_six_track.py"}
            if suffix == "png"
            else {
                "Creator": "p5_r2_six_track.py",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        )
        figure.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        raw = path.read_bytes()
        outputs.append(
            {
                "path": path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    plt.close(figure)
    return outputs


def coverage_values(data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fault = data["fault"]
    facies = data["facies"]
    prop = data["property"]
    litho = data["lithofacies"]
    sweet = data["sweetspot"]
    recon = data["reconstruction"]

    complete = np.asarray(
        [
            fault["observed_acquisition_baseline_count"],
            facies["status_counts"]["completed"],
            prop["counts"]["completed"],
            litho["completed_cells"],
            sweet["counts"]["PASS"],
            recon["conditional"]["counts"]["passed"] + recon["strict"]["counts"]["passed"],
        ],
        dtype=float,
    )
    total = np.asarray(
        [
            fault["minimum_unblock_contract_count"],
            facies["result_count"],
            prop["expected_cells"],
            litho["expected_cells"],
            sweet["expected_cells"],
            (
                recon["conditional"]["counts"]["passed"]
                + recon["conditional"]["counts"]["blocked"]
                + recon["conditional"]["counts"]["not_rankable"]
                + recon["strict"]["counts"]["passed"]
                + recon["strict"]["counts"]["blocked"]
                + recon["strict"]["counts"]["not_rankable"]
            ),
        ],
        dtype=float,
    )
    kinds = [
        "acquisition evidence",
        "completed experiment cells",
        "completed experiment cells",
        "completed experiment cells",
        "passed experiment cells",
        "passed mode/cell outcomes",
    ]
    return complete, total, kinds


def render_readiness(data: Mapping[str, Any], font_family: str) -> mpl.figure.Figure:
    figure = plt.figure(figsize=(7.2, 3.5))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.08, 1.0), wspace=0.34)
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])

    complete, total, _ = coverage_values(data)
    fractions = 100.0 * complete / total
    y = np.arange(len(TRACK_LABELS))
    axis_a.barh(y, fractions, color=BLUE, height=0.58, edgecolor="none")
    axis_a.barh(
        y,
        100.0 - fractions,
        left=fractions,
        color=LIGHT,
        height=0.58,
        edgecolor=GRID,
        linewidth=0.4,
    )
    axis_a.set_yticks(y, TRACK_LABELS)
    axis_a.invert_yaxis()
    axis_a.set_xlim(0, 100)
    axis_a.set_xticks([0, 25, 50, 75, 100])
    axis_a.set_xlabel("Protocol cell/evidence coverage (%)")
    clean_axis(axis_a, grid_axis="x")
    panel_label(axis_a, "a")

    gate_matrix = np.asarray(
        [
            [0.5, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5, 1.0, 1.0],
            [1.0, 0.5, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    color_map = ListedColormap([RED, YELLOW, BLUE])
    norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], color_map.N)
    image = axis_b.imshow(gate_matrix, aspect="auto", cmap=color_map, norm=norm)
    axis_b.set_yticks(np.arange(len(TRACK_LABELS)), TRACK_LABELS)
    axis_b.set_xticks(
        np.arange(5),
        ["Legal dev\ndata", "Controlled\nsweep", "Model\ndiversity", "Rankable\noutput", "Test\nfirewall"],
    )
    axis_b.tick_params(axis="x", length=0)
    axis_b.tick_params(axis="y", length=0)
    for spine in axis_b.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis_b, fraction=0.045, pad=0.04, ticks=[0, 0.5, 1])
    colorbar.ax.set_yticklabels(["no", "partial", "yes"])
    colorbar.outline.set_linewidth(0.5)
    panel_label(axis_b, "b")

    figure.subplots_adjust(left=0.12, right=0.95, bottom=0.22, top=0.92)
    return figure


def plot_fault(axis: mpl.axes.Axes, fault: Mapping[str, Any]) -> None:
    observed = float(fault["observed_acquisition_baseline_count"])
    required = float(fault["minimum_unblock_contract_count"])
    axis.barh(
        [0, 1],
        [observed, required],
        color=[CYAN, BLUE],
        height=0.5,
        edgecolor="none",
    )
    axis.set_yticks([0, 1], ["Observed baseline", "Minimum contract"])
    axis.invert_yaxis()
    axis.set_xlabel("Acquisition-contract items")
    axis.set_xlim(0, required * 1.08)
    clean_axis(axis, grid_axis="x")


def plot_facies(axis: mpl.axes.Axes, facies: Mapping[str, Any]) -> None:
    budgets = np.asarray([40, 400, 1000], dtype=float)
    for task_id, label, color, marker in (
        ("facies_f3", "F3", BLUE, "o"),
        ("facies_penobscot", "Penobscot", RED, "s"),
    ):
        values = [
            facies["tasks"][task_id]["endpoint_means"][str(int(budget))]["miou"]
            for budget in budgets
        ]
        axis.plot(
            budgets,
            values,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=label,
        )
    axis.set_xlabel("Training updates")
    axis.set_ylabel("Validation mIoU")
    axis.set_xticks(budgets)
    axis.set_ylim(0, 1.0)
    axis.legend(loc="lower right")
    clean_axis(axis, grid_axis="y")


def plot_property(axis: mpl.axes.Axes, prop: Mapping[str, Any]) -> None:
    budgets = np.asarray([40, 160, 640], dtype=float)
    curves = prop["curve_metrics"]["tabm_regressor"]
    for target, color, marker in (("PHIF", BLUE, "o"), ("SW", CYAN, "s")):
        raw = np.asarray(
            [curves[target][str(int(b))]["mean_physical_RMSE"] for b in budgets],
            dtype=float,
        )
        axis.plot(
            budgets,
            raw / raw[0],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=target,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, ["40", "160", "640"])
    axis.set_xlabel("Training budget")
    axis.set_ylabel("RMSE / RMSE at budget 40")
    axis.set_ylim(0.5, 1.05)
    clean_axis(axis, grid_axis="y")

    secondary = axis.twinx()
    klogh = np.asarray(
        [curves["KLOGH"][str(int(b))]["mean_physical_RMSE"] for b in budgets],
        dtype=float,
    )
    secondary.plot(
        budgets,
        np.log10(klogh),
        color=RED,
        marker="x",
        markersize=4,
        linewidth=1.0,
        linestyle="--",
        label="KLOGH",
    )
    secondary.set_ylabel("KLOGH log10(RMSE)", color=RED)
    secondary.tick_params(axis="y", colors=RED)
    secondary.spines["top"].set_visible(False)
    secondary.spines["right"].set_color(RED)

    handles_a, labels_a = axis.get_legend_handles_labels()
    handles_b, labels_b = secondary.get_legend_handles_labels()
    axis.legend(handles_a + handles_b, labels_a + labels_b, loc="center right")


def plot_lithofacies(axis: mpl.axes.Axes, litho: Mapping[str, Any]) -> None:
    matrix = np.asarray(litho["fold_seed_matrix"], dtype=float)
    x = np.arange(matrix.shape[0])
    jitter = np.linspace(-0.16, 0.16, matrix.shape[1])
    for index, color in enumerate((BLUE, CYAN, RED)):
        axis.scatter(
            np.full(matrix.shape[1], x[index]) + jitter,
            matrix[index],
            s=12,
            color=color,
            alpha=0.72,
            edgecolors="none",
        )
        axis.plot(
            [x[index] - 0.22, x[index] + 0.22],
            [matrix[index].mean(), matrix[index].mean()],
            color=DARK,
            linewidth=1.4,
        )
    labels = ["MLP", "XGBoost\nwindow", "InceptionTime\nwindow"]
    axis.set_xticks(x, labels)
    axis.set_ylabel("Fixed-schema macro-F1")
    axis.set_xlim(-0.55, len(x) - 0.45)
    axis.set_ylim(0, max(0.30, float(matrix.max()) * 1.12))
    clean_axis(axis, grid_axis="y")


def _relative_improvement(curve: Mapping[str, Any]) -> np.ndarray:
    budgets = ["64", "256", "1024"]
    values = np.asarray([curve["budgets"][budget] for budget in budgets], dtype=float)
    baseline = values[0]
    if curve["direction"] == "minimize":
        return 100.0 * (baseline - values) / abs(baseline)
    return 100.0 * (values - baseline) / abs(baseline)


def plot_sweetspot(axis: mpl.axes.Axes, sweet: Mapping[str, Any]) -> None:
    budgets = np.asarray([64, 256, 1024], dtype=float)
    for target, color, marker in (
        ("T1", BLUE, "o"),
        ("T2", CYAN, "s"),
        ("T3", RED, "^"),
    ):
        curve = sweet["target_budget_curve"][target]
        axis.plot(
            budgets,
            _relative_improvement(curve),
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.4,
            label=target,
        )
    axis.axhline(0, color=DARK, linewidth=0.7, linestyle=":")
    axis.set_xscale("log", base=2)
    axis.set_xticks(budgets, ["64", "256", "1024"])
    axis.set_xlabel("Training budget")
    axis.set_ylabel("Relative improvement vs 64 (%)")
    axis.legend(loc="lower left", ncol=3)
    clean_axis(axis, grid_axis="y")


def plot_reconstruction(axis: mpl.axes.Axes, recon: Mapping[str, Any]) -> None:
    updates = np.asarray([100, 400], dtype=float)
    styles = [
        (BLUE, "o", "-"),
        (CYAN, "s", "--"),
        (YELLOW, "^", "-"),
        (RED, "D", "--"),
    ]
    for row, (color, marker, line_style) in zip(
        recon["conditional"]["r3_gate"],
        styles,
        strict=True,
    ):
        label = (
            row["model_id"]
            .replace("reconstruction_", "")
            .replace("_sgd", "")
            .replace("_", " ")
        )
        label = f"{label}—{row['loss_name'].upper()}"
        values = [row["rmse_100"], row["rmse_400"]]
        axis.plot(
            updates,
            values,
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.3,
            linestyle=line_style,
            label=label,
        )
    axis.set_yscale("log")
    axis.set_ylim(0.03, 1.5)
    axis.yaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1.0]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks(updates)
    axis.set_xlabel("Training updates")
    axis.set_ylabel("Conditional B1 RMSE")
    axis.legend(
        loc="center right",
        ncol=1,
        fontsize=5.8,
        handlelength=1.8,
        labelspacing=0.3,
    )
    clean_axis(axis, grid_axis="y")


TRACK_PLOTS: list[tuple[str, Callable[[mpl.axes.Axes, Mapping[str, Any]], None]]] = [
    ("fault", plot_fault),
    ("facies", plot_facies),
    ("property", plot_property),
    ("lithofacies", plot_lithofacies),
    ("sweetspot", plot_sweetspot),
    ("reconstruction", plot_reconstruction),
]


def render_scientific_results(
    data: Mapping[str, Any],
    font_family: str,
) -> mpl.figure.Figure:
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 7.2))
    for index, ((track, plotter), axis) in enumerate(zip(TRACK_PLOTS, axes.flat, strict=True)):
        plotter(axis, data[track])
        panel_label(axis, chr(ord("a") + index))
    figure.subplots_adjust(left=0.09, right=0.93, bottom=0.08, top=0.96, wspace=0.58, hspace=0.42)
    return figure


def render_track_detail(
    track: str,
    plotter: Callable[[mpl.axes.Axes, Mapping[str, Any]], None],
    payload: Mapping[str, Any],
) -> mpl.figure.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    plotter(axis, payload)
    panel_label(axis, "a")
    figure.subplots_adjust(left=0.16, right=0.88, bottom=0.22, top=0.91)
    return figure


def write_manifest(
    output_dir: Path,
    source_manifest: dict[str, Any],
    figure_entries: Sequence[dict[str, Any]],
    font_family: str,
    exact_tnr: bool,
    data: Mapping[str, Any],
) -> Path:
    complete, total, kinds = coverage_values(data)
    source_manifest["font"] = {
        "requested": "Times New Roman",
        "resolved": font_family,
        "exact_match": exact_tnr,
    }
    source_manifest["derived_fields"] = {
        "coverage": [
            {
                "track": track,
                "numerator": int(numerator),
                "denominator": int(denominator),
                "kind": kind,
            }
            for track, numerator, denominator, kind in zip(
                SOURCES, complete, total, kinds, strict=True
            )
        ],
        "readiness_gate_matrix": {
            "columns": [
                "legal_development_data",
                "controlled_sweep",
                "model_diversity",
                "rankable_output",
                "test_firewall",
            ],
            "encoding": {"no": 0.0, "partial": 0.5, "yes": 1.0},
            "rows": {
                "fault": [0.5, 0.0, 0.0, 0.0, 1.0],
                "facies": [1.0, 1.0, 0.5, 1.0, 1.0],
                "property": [1.0, 0.5, 0.5, 0.5, 1.0],
                "lithofacies": [1.0, 1.0, 1.0, 1.0, 1.0],
                "sweetspot": [0.5, 0.5, 0.5, 0.5, 1.0],
                "reconstruction": [0.5, 0.5, 1.0, 0.0, 1.0],
            },
        },
        "property_panel": {
            "PHIF_and_SW": "RMSE divided by the same target's budget-40 RMSE",
            "KLOGH": "shown on an independent log10(RMSE) right axis because values span 1e96–1e106",
        },
        "sweetspot_panel": (
            "relative improvement from budget 64; sign inverted only for minimize-direction metrics"
        ),
    }
    source_manifest["figures"] = [
        {
            **entry,
            "artifact_class": "protocol_and_model_performance_summary",
            "domain_visualization_eligible": False,
        }
        for entry in figure_entries
    ]
    path = output_dir / "visualization_manifest.json"
    path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "_outputs" / "p5_r2_visualization",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    data, source_manifest = load_sources(repo_root, output_dir)
    font_family, exact_tnr = choose_font()
    configure_style(font_family)

    figure_entries: list[dict[str, Any]] = []
    outputs = save_figure(
        render_readiness(data, font_family),
        output_dir,
        "figure_01_protocol_readiness",
        font_family,
    )
    figure_entries.append({"figure_id": "protocol_readiness", "outputs": outputs})

    outputs = save_figure(
        render_scientific_results(data, font_family),
        output_dir,
        "figure_02_r2_scientific_results",
        font_family,
    )
    figure_entries.append({"figure_id": "r2_scientific_results", "outputs": outputs})

    detail_dir = output_dir / "tracks"
    for index, (track, plotter) in enumerate(TRACK_PLOTS, start=1):
        outputs = save_figure(
            render_track_detail(track, plotter, data[track]),
            detail_dir,
            f"track_{index:02d}_{track}",
            font_family,
        )
        figure_entries.append({"figure_id": f"track_{track}", "outputs": outputs})

    manifest_path = write_manifest(
        output_dir,
        source_manifest,
        figure_entries,
        font_family,
        exact_tnr,
        data,
    )
    print(f"Wrote {len(figure_entries)} figure groups to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Font: {font_family} (exact Times New Roman={exact_tnr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
