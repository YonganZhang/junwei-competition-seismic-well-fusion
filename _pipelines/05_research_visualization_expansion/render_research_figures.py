#!/usr/bin/env python3
"""Render real-data research figures for the six geoscience tracks.

The script does not train or select a model.  It reads already archived
observations/predictions and creates spatial or along-well figures whose
evidence mode is declared in the output manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from PIL import Image
import segyio

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - surfaced by interactive render calls
    go = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "_code" / "visualization"))
from geo3d_viz import (  # noqa: E402
    AKUN,
    EvidenceSource,
    POROSITY_CMAP,
    RESIDUAL_CMAP,
    SEISMIC_CMAP,
    output_record,
    panel_label,
    plotly_layout,
    project_relative,
    publication_style,
    save_figure_bundle,
    sha256_file,
    write_json,
    write_plotly_html,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "_outputs" / "research_visualization_expansion" / "v1"
FAULT_POINTS = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/fault_points.npz"
HORIZON_POINTS = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/horizon_bcu_points.npz"
SEISMIC_INDEX = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
SEISMIC_INDEX_META = (
    PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index_meta.json"
)
SEGY_PATH = Path(
    json.loads(SEISMIC_INDEX_META.read_text(encoding="utf-8"))["segy_path"]
)
F3_INLINES = PROJECT_ROOT / "_sandbox/f3_penobscot/f3demo/inlines.zip"
F3_MASKS = PROJECT_ROOT / "_sandbox/f3_penobscot/f3demo/masks.tar.gz"
PENOBSCOT = PROJECT_ROOT / "_sandbox/f3_penobscot/penobscot/dataset.h5"
PROPERTY_ROOT = (
    PROJECT_ROOT
    / "_pipelines/02_task_datasets/reservoir/_outputs/p5_stage4_confirmation"
)
LITHOFACIES_PREDICTIONS = (
    PROJECT_ROOT
    / ".claude/worktrees/p10-results-lithofacies"
    / "_pipelines/02_task_datasets/lithofacies/_outputs/p5_stage4_confirmation/predictions.json"
)
SWEETSPOT_ROOT = (
    PROJECT_ROOT
    / ".claude/worktrees/p10-results-sweetspot"
    / "_pipelines/02_task_datasets/sweetspot/targets"
)
SWEETSPOT_T3_CONFIRMATION = (
    PROJECT_ROOT
    / ".claude/worktrees/p10-results-sweetspot"
    / "_pipelines/02_task_datasets/sweetspot/p5/_outputs/stage4_confirmation"
    / "targets/T3/predictions.csv.gz"
)
RECONSTRUCTION_ROOT = (
    PROJECT_ROOT
    / ".claude/worktrees/p5-r2-reconstruction-v2"
    / "_pipelines/02_task_datasets/reconstruction/_outputs/3d_property_volume_v2"
)
FOUNDATION_MODEL_CONTRACT = (
    PROJECT_ROOT
    / "_pipelines/05_research_visualization_expansion"
    / "foundation_model_experiment_contract.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-interactive",
        action="store_true",
        help="Render only PNG/PDF outputs; useful for quick local checks.",
    )
    return parser.parse_args()


def _register_bundle(
    outputs: list[dict[str, Any]],
    bundle: dict[str, Path],
    *,
    role: str,
    evidence_mode: str,
    caption: str,
) -> None:
    for path in bundle.values():
        outputs.append(
            output_record(
                path,
                PROJECT_ROOT,
                role=role,
                evidence_mode=evidence_mode,
                caption=caption,
            )
        )


def _register_html(
    outputs: list[dict[str, Any]],
    path: Path,
    *,
    role: str,
    evidence_mode: str,
    caption: str,
) -> None:
    outputs.append(
        output_record(
            path,
            PROJECT_ROOT,
            role=role,
            evidence_mode=evidence_mode,
            caption=caption,
        )
    )


def render_fault(output_dir: Path, *, interactive: bool) -> dict[str, Any]:
    with np.load(FAULT_POINTS, allow_pickle=True) as archive:
        fault = {key: archive[key] for key in archive.files}
    with np.load(HORIZON_POINTS) as archive:
        horizon = {key: archive[key] for key in archive.files}

    publication_style()
    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    horizon_step = max(1, len(horizon["utmx"]) // 9000)
    hs = slice(None, None, horizon_step)
    surface = ax.plot_trisurf(
        horizon["utmx"][hs] / 1000.0,
        horizon["utmy"][hs] / 1000.0,
        horizon["twt_ms"][hs],
        cmap=SEISMIC_CMAP,
        linewidth=0,
        alpha=0.34,
        antialiased=True,
        shade=False,
    )
    names = np.asarray(fault["fault_name"], dtype=str)
    sticks = np.asarray(fault["stick_no"], dtype=int)
    unique_names = sorted(np.unique(names))
    palette = [
        AKUN["red"],
        AKUN["orange"],
        AKUN["blue"],
        AKUN["cyan"],
        AKUN["green"],
        "#7B6FA8",
        "#A65C85",
        "#6E7F80",
    ]
    for fault_index, name in enumerate(unique_names):
        selection = names == name
        for stick in np.unique(sticks[selection]):
            mask = selection & (sticks == stick)
            if np.count_nonzero(mask) < 2:
                continue
            order = np.argsort(fault["twt_ms"][mask])
            ax.plot(
                fault["utmx"][mask][order] / 1000.0,
                fault["utmy"][mask][order] / 1000.0,
                fault["twt_ms"][mask][order],
                color=palette[fault_index % len(palette)],
                lw=1.05,
                alpha=0.92,
            )
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_zlabel("TWT (ms)")
    ax.invert_zaxis()
    ax.view_init(elev=24, azim=-54)
    ax.grid(False)
    scalar = cm.ScalarMappable(
        norm=colors.Normalize(
            float(np.nanpercentile(horizon["twt_ms"], 2)),
            float(np.nanpercentile(horizon["twt_ms"], 98)),
        ),
        cmap=SEISMIC_CMAP,
    )
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, shrink=0.58, pad=0.08)
    cbar.set_label("BCU TWT (ms)")
    panel_label(ax, "a", x=-0.02, y=0.98)
    fig.subplots_adjust(left=0.0, right=0.88, bottom=0.02, top=0.98)

    caption = (
        "Volve fault sticks in real UTM–TWT coordinates above the interpreted BCU "
        "horizon. The horizon is spatial context; no dense fault-probability volume "
        "is implied."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(fig, output_dir / "figures/fault_geological_context")
    _register_bundle(
        outputs,
        bundle,
        role="fault_spatial_context",
        evidence_mode="spatial_context",
        caption=caption,
    )

    # An inline section validates that projected fault-stick control points follow
    # reflectors in the source seismic, instead of judging the 3-D geometry alone.
    selected_inline = 10243
    with np.load(SEISMIC_INDEX) as archive:
        seismic_index = {key: archive[key] for key in archive.files}
    xl_min = int(seismic_index["xl_min"])
    xl_max = int(seismic_index["xl_max"])
    n_xl = int(seismic_index["n_xl"])
    il_min = int(seismic_index["il_min"])
    samples_ms = np.asarray(seismic_index["samples_ms"], dtype=float)
    trace_start = (selected_inline - il_min) * n_xl
    with segyio.open(str(SEGY_PATH), "r", ignore_geometry=True) as handle:
        section = np.stack(
            [
                np.asarray(handle.trace[trace_start + offset], dtype=np.float32)
                for offset in range(n_xl)
            ],
            axis=0,
        )
    amplitude_limit = float(np.nanpercentile(np.abs(section), 99.2))
    publication_style()
    fig2, section_ax = plt.subplots(figsize=(9.4, 5.4))
    section_ax.imshow(
        section.T,
        cmap=SEISMIC_CMAP,
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
        extent=[xl_min, xl_max, samples_ms[-1], samples_ms[0]],
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    inline_mask = np.asarray(fault["inline"], dtype=int) == selected_inline
    for fault_index, name in enumerate(unique_names):
        name_mask = inline_mask & (names == name)
        for stick in np.unique(sticks[name_mask]):
            mask = name_mask & (sticks == stick)
            if np.count_nonzero(mask) < 2:
                continue
            order = np.argsort(fault["crossline"][mask])
            section_ax.plot(
                fault["crossline"][mask][order],
                fault["twt_ms"][mask][order],
                color=palette[fault_index % len(palette)],
                lw=1.25,
                marker="o",
                ms=2.4,
                markeredgecolor="white",
                markeredgewidth=0.25,
                alpha=0.95,
            )
    section_ax.set_xlim(2060, 2525)
    section_ax.set_ylim(3400, 2300)
    section_ax.set_xlabel(f"Crossline (inline {selected_inline} section)")
    section_ax.set_ylabel("TWT (ms)")
    panel_label(section_ax, "a", x=-0.06, y=1.01)
    fig2.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.98)
    section_caption = (
        f"Source-seismic section along crossline direction at inline {selected_inline}, "
        "with official fault-stick points that lie on the same inline. Only stick "
        "segments with at least two control points are connected."
    )
    section_bundle = save_figure_bundle(
        fig2,
        output_dir / "figures/fault_seismic_section_overlay",
    )
    _register_bundle(
        outputs,
        section_bundle,
        role="fault_source_seismic_validation",
        evidence_mode="spatial_context",
        caption=section_caption,
    )

    if interactive:
        if go is None:
            raise RuntimeError("plotly is required for interactive fault rendering")
        scene = go.Figure()
        hstep = max(1, len(horizon["utmx"]) // 15000)
        scene.add_trace(
            go.Scatter3d(
                x=horizon["utmx"][::hstep] / 1000.0,
                y=horizon["utmy"][::hstep] / 1000.0,
                z=horizon["twt_ms"][::hstep],
                mode="markers",
                marker={
                    "size": 1.5,
                    "color": horizon["twt_ms"][::hstep],
                    "colorscale": [
                        [0.0, "#243B53"],
                        [0.5, "#FFFFFF"],
                        [1.0, "#B9503F"],
                    ],
                    "opacity": 0.26,
                    "colorbar": {"title": "BCU TWT (ms)", "len": 0.55},
                },
                name="BCU horizon",
                hovertemplate=(
                    "E %{x:.2f} km<br>N %{y:.2f} km<br>TWT %{z:.1f} ms<extra></extra>"
                ),
            )
        )
        for fault_index, name in enumerate(unique_names):
            selection = names == name
            first = True
            for stick in np.unique(sticks[selection]):
                mask = selection & (sticks == stick)
                if np.count_nonzero(mask) < 2:
                    continue
                order = np.argsort(fault["twt_ms"][mask])
                scene.add_trace(
                    go.Scatter3d(
                        x=fault["utmx"][mask][order] / 1000.0,
                        y=fault["utmy"][mask][order] / 1000.0,
                        z=fault["twt_ms"][mask][order],
                        mode="lines",
                        line={
                            "color": palette[fault_index % len(palette)],
                            "width": 4,
                        },
                        name=name,
                        legendgroup=name,
                        showlegend=first,
                        hovertemplate=(
                            f"{name}<br>E %{{x:.2f}} km<br>N %{{y:.2f}} km"
                            "<br>TWT %{z:.1f} ms<extra></extra>"
                        ),
                    )
                )
                first = False
        layout = plotly_layout(z_title="TWT (ms)")
        layout["scene"]["xaxis"]["title"] = "Easting (km)"
        layout["scene"]["yaxis"]["title"] = "Northing (km)"
        scene.update_layout(**layout)
        html = output_dir / "interactive/fault_geological_context.html"
        write_plotly_html(scene, html)
        _register_html(
            outputs,
            html,
            role="fault_interactive_scene",
            evidence_mode="spatial_context",
            caption=caption,
        )

    sources = [
        EvidenceSource(
            FAULT_POINTS,
            "official interpreted fault-stick control points",
            "full interpretation set; not a model split",
            f"{len(fault['utmx'])} points",
        ),
        EvidenceSource(
            HORIZON_POINTS,
            "official interpreted BCU horizon control points",
            "full interpretation set; not a model split",
            f"{len(horizon['utmx'])} points",
        ),
        EvidenceSource(
            SEGY_PATH,
            "Volve ST0202 post-stack 3-D source seismic",
            "source volume; inline section read lazily without loading the full cube",
            "385×605 traces, 1126 time samples",
        ),
        EvidenceSource(
            SEISMIC_INDEX,
            "measured SEG-Y inline/crossline/time index",
            "source coordinate registration; not a model split",
            "385×605 regular trace grid",
        ),
    ]
    return {
        "track": "fault",
        "evidence_mode": "spatial_context",
        "sources": [item.as_record(PROJECT_ROOT) for item in sources],
        "outputs": outputs,
        "scientific_boundary": (
            "The figure shows interpreted geometry and does not claim a dense model "
            "prediction volume."
        ),
    }


def _archive_member_by_inline(names: list[str], inline_id: int) -> str:
    pattern = re.compile(rf"(?:^|/)inline_{inline_id}(?:_mask)?\.(?:tiff?|png)$")
    for name in names:
        if "/._" not in name and pattern.search(name):
            return name
    raise KeyError(f"inline {inline_id} not found")


def _load_f3_curtains(inline_ids: list[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    records: list[tuple[int, np.ndarray, np.ndarray]] = []
    with zipfile.ZipFile(F3_INLINES) as inline_zip, tarfile.open(
        F3_MASKS, "r:gz"
    ) as mask_tar:
        inline_names = inline_zip.namelist()
        mask_names = mask_tar.getnames()
        for inline_id in inline_ids:
            inline_name = _archive_member_by_inline(inline_names, inline_id)
            mask_name = _archive_member_by_inline(mask_names, inline_id)
            with Image.open(io.BytesIO(inline_zip.read(inline_name))) as image:
                seismic = np.asarray(image, dtype=np.float32)
            extracted = mask_tar.extractfile(mask_name)
            if extracted is None:
                raise FileNotFoundError(mask_name)
            with Image.open(io.BytesIO(extracted.read())) as image:
                labels = np.asarray(image, dtype=np.uint8)
            if seismic.shape != labels.shape:
                raise ValueError(
                    f"F3 seismic/mask shape mismatch at inline {inline_id}: "
                    f"{seismic.shape} vs {labels.shape}"
                )
            records.append((inline_id, seismic, labels))
    return records


def render_facies(output_dir: Path, *, interactive: bool) -> dict[str, Any]:
    inline_ids = [220, 425, 650]
    records = _load_f3_curtains(inline_ids)
    publication_style()
    fig = plt.figure(figsize=(11.6, 5.7))
    axes = [
        fig.add_subplot(121, projection="3d"),
        fig.add_subplot(122, projection="3d"),
    ]
    label_cmap = matplotlib.colormaps["tab10"]
    amplitude_limit = float(
        np.nanpercentile(
            np.abs(np.concatenate([row[1].reshape(-1) for row in records])),
            99.0,
        )
    )
    amplitude_norm = colors.Normalize(-amplitude_limit, amplitude_limit)
    label_norm = colors.BoundaryNorm(np.arange(-0.5, 10.5, 1.0), 10)
    for inline_id, seismic, labels in records:
        row_step, column_step = 3, 6
        seismic_d = seismic[::row_step, ::column_step]
        labels_d = labels[::row_step, ::column_step]
        time_index = np.arange(seismic.shape[0])[::row_step]
        crossline_index = np.arange(seismic.shape[1])[::column_step]
        yy, zz = np.meshgrid(crossline_index, time_index)
        xx = np.full_like(yy, inline_id, dtype=float)
        axes[0].plot_surface(
            xx,
            yy,
            zz,
            facecolors=SEISMIC_CMAP(amplitude_norm(seismic_d)),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.93,
        )
        axes[1].plot_surface(
            xx,
            yy,
            zz,
            facecolors=label_cmap(label_norm(labels_d)),
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=0.92,
        )
    for ax in axes:
        ax.set_xlabel("Inline")
        ax.set_ylabel("Crossline index")
        ax.set_zlabel("Sample index")
        ax.invert_zaxis()
        ax.view_init(elev=22, azim=-58)
        ax.grid(False)
        ax.set_box_aspect((1.2, 1.8, 1.3))
    panel_label(axes[0], "a", x=-0.02, y=0.98)
    panel_label(axes[1], "b", x=-0.02, y=0.98)
    seismic_scalar = cm.ScalarMappable(norm=amplitude_norm, cmap=SEISMIC_CMAP)
    seismic_scalar.set_array([])
    cb0 = fig.colorbar(seismic_scalar, ax=axes[0], shrink=0.55, pad=0.05)
    cb0.set_label("Seismic amplitude")
    label_scalar = cm.ScalarMappable(norm=label_norm, cmap=label_cmap)
    label_scalar.set_array([])
    cb1 = fig.colorbar(
        label_scalar,
        ax=axes[1],
        shrink=0.55,
        pad=0.05,
        ticks=np.arange(10),
    )
    cb1.set_label("F3 interval class")
    fig.subplots_adjust(left=0.01, right=0.96, bottom=0.02, top=0.98, wspace=0.08)

    caption = (
        "Registered F3 inline curtains sampled from the continuous seismic volume: "
        "(a) seismic amplitude and (b) the corresponding ten-class interpretation. "
        "These are reference data, not dense predictions from the current SAM2 model."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(fig, output_dir / "figures/facies_f3_curtains")
    _register_bundle(
        outputs,
        bundle,
        role="facies_reference_volume_context",
        evidence_mode="native_volume",
        caption=caption,
    )

    if interactive:
        if go is None:
            raise RuntimeError("plotly is required for interactive facies rendering")
        scene = go.Figure()
        seismic_scale = [
            [0.0, "#243B53"],
            [0.5, "#FFFFFF"],
            [1.0, "#B9503F"],
        ]
        for inline_id, seismic, labels in records:
            row_step, column_step = 6, 10
            time_index = np.arange(seismic.shape[0])[::row_step]
            crossline_index = np.arange(seismic.shape[1])[::column_step]
            yy, zz = np.meshgrid(crossline_index, time_index)
            xx = np.full_like(yy, inline_id, dtype=float)
            scene.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=zz,
                    surfacecolor=seismic[::row_step, ::column_step],
                    cmin=-amplitude_limit,
                    cmax=amplitude_limit,
                    colorscale=seismic_scale,
                    opacity=0.88,
                    showscale=inline_id == inline_ids[0],
                    colorbar={"title": "Amplitude", "len": 0.48},
                    name=f"Seismic IL {inline_id}",
                    hovertemplate=(
                        "Inline %{x:.0f}<br>Crossline index %{y:.0f}"
                        "<br>Sample %{z:.0f}<br>Amplitude %{surfacecolor:.1f}<extra></extra>"
                    ),
                )
            )
            boundary = np.zeros_like(labels, dtype=bool)
            boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
            boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
            points = np.argwhere(boundary)
            if len(points) > 2500:
                points = points[:: max(1, len(points) // 2500)]
            scene.add_trace(
                go.Scatter3d(
                    x=np.full(len(points), inline_id),
                    y=points[:, 1],
                    z=points[:, 0],
                    mode="markers",
                    marker={
                        "size": 1.6,
                        "color": labels[points[:, 0], points[:, 1]],
                        "colorscale": "Turbo",
                        "cmin": 0,
                        "cmax": 9,
                        "opacity": 0.75,
                    },
                    name=f"Class boundaries IL {inline_id}",
                    hovertemplate=(
                        "Inline %{x:.0f}<br>Crossline index %{y:.0f}"
                        "<br>Sample %{z:.0f}<extra></extra>"
                    ),
                )
            )
        layout = plotly_layout(z_title="Sample index")
        layout["scene"]["xaxis"]["title"] = "Inline"
        layout["scene"]["yaxis"]["title"] = "Crossline index"
        scene.update_layout(**layout)
        html = output_dir / "interactive/facies_f3_curtains.html"
        write_plotly_html(scene, html)
        _register_html(
            outputs,
            html,
            role="facies_interactive_reference_volume",
            evidence_mode="native_volume",
            caption=caption,
        )

    with h5py.File(PENOBSCOT, "r") as handle:
        penobscot_shape = tuple(int(value) for value in handle["features"].shape)
    sources = [
        EvidenceSource(
            F3_INLINES,
            "continuous F3 inline seismic sections",
            "reference interpretation data; not a model split",
            "651 inline TIFF sections, each 462×951",
        ),
        EvidenceSource(
            F3_MASKS,
            "F3 inline/crossline interval-class interpretations",
            "reference interpretation data; not a model split",
            "1602 masks; rendered inlines use 462×951 labels 0–9",
        ),
        EvidenceSource(
            PENOBSCOT,
            "registered Penobscot seismic and eight-class label volume",
            "reference dataset; read-only dimensional audit",
            str(penobscot_shape),
        ),
    ]
    return {
        "track": "facies",
        "evidence_mode": "native_volume",
        "sources": [item.as_record(PROJECT_ROOT) for item in sources],
        "outputs": outputs,
        "scientific_boundary": (
            "F3/Penobscot provide true three-dimensional reference context, but the "
            "current archived SAM2 run does not persist a dense prediction volume."
        ),
    }


def render_property(output_dir: Path) -> dict[str, Any]:
    specs = [
        ("PHIF", PROPERTY_ROOT / "phif/predictions.csv", False, "fraction"),
        ("KLOGH", PROPERTY_ROOT / "klogh/predictions.csv", True, "mD"),
        ("SW", PROPERTY_ROOT / "sw/predictions.csv", False, "fraction"),
    ]
    publication_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 7.2),
        sharey="col",
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    sources: list[EvidenceSource] = []
    target_frames: dict[str, pd.DataFrame] = {}
    for column, (target, path, use_log, unit) in enumerate(specs):
        frame = pd.read_csv(path).sort_values("depth_m")
        target_frames[target] = frame
        depth = frame["depth_m"].to_numpy(float)
        truth = frame["truth_physical"].to_numpy(float)
        prediction = frame["prediction_physical"].to_numpy(float)
        low = frame["interval_low_physical"].to_numpy(float)
        high = frame["interval_high_physical"].to_numpy(float)
        residual = prediction - truth
        ax = axes[0, column]
        ax.fill_betweenx(
            depth,
            low,
            high,
            color=AKUN["sand"],
            alpha=0.30,
            linewidth=0,
            label="90% interval",
        )
        ax.plot(truth, depth, color=AKUN["ink"], lw=1.15, label="Observed")
        ax.plot(prediction, depth, color=AKUN["orange"], lw=1.05, label="Predicted")
        if use_log:
            ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(f"{target} ({unit})")
        ax.grid(axis="x", color=AKUN["grid"], lw=0.5, alpha=0.8)
        if column == 0:
            ax.set_ylabel("Measured depth (m)")
        ax.legend(frameon=False, loc="best")
        panel_label(ax, chr(ord("a") + column))

        residual_ax = axes[1, column]
        residual_ax.plot(residual, depth, color=AKUN["blue"], lw=0.9)
        residual_ax.axvline(0.0, color=AKUN["muted"], lw=0.8, ls="--")
        residual_ax.fill_betweenx(
            depth,
            0.0,
            residual,
            where=residual >= 0,
            color=AKUN["orange"],
            alpha=0.34,
        )
        residual_ax.fill_betweenx(
            depth,
            0.0,
            residual,
            where=residual < 0,
            color=AKUN["blue"],
            alpha=0.30,
        )
        residual_ax.invert_yaxis()
        residual_ax.set_xlabel(f"Residual ({unit})")
        residual_ax.grid(axis="x", color=AKUN["grid"], lw=0.5, alpha=0.8)
        if column == 0:
            residual_ax.set_ylabel("Measured depth (m)")
        panel_label(residual_ax, chr(ord("d") + column))
        sources.append(
            EvidenceSource(
                path,
                f"{target} known-holdout predictions and OOF residual interval",
                "previously seen reusable holdout: 15/9-F-15 D",
                f"{len(frame)} rows",
            )
        )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.98, hspace=0.20, wspace=0.28)
    caption = (
        "Along-well observed and predicted PHIF, KLOGH and SW with archived 90% "
        "residual intervals (a–c), and depth-resolved residuals (d–f). KLOGH is "
        "shown on a logarithmic physical scale; no samples are removed."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(fig, output_dir / "figures/property_well_diagnostics")
    _register_bundle(
        outputs,
        bundle,
        role="property_along_well_diagnostics",
        evidence_mode="section_only",
        caption=caption,
    )

    # Paired PHIF–KLOGH views test whether the two independently fitted targets
    # retain the observed rock-physics association on the same holdout samples.
    paired = target_frames["PHIF"][
        ["sample_id", "depth_m", "truth_physical", "prediction_physical"]
    ].merge(
        target_frames["KLOGH"][
            ["sample_id", "truth_physical", "prediction_physical"]
        ],
        on="sample_id",
        suffixes=("_phif", "_klogh"),
        validate="one_to_one",
    )
    publication_style()
    fig2, cross_axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)
    scatter = None
    for index, (x_name, y_name) in enumerate(
        (
            ("truth_physical_phif", "truth_physical_klogh"),
            ("prediction_physical_phif", "prediction_physical_klogh"),
        )
    ):
        ax = cross_axes[index]
        scatter = ax.scatter(
            paired[x_name],
            paired[y_name],
            c=paired["depth_m"],
            cmap=matplotlib.colormaps["viridis_r"],
            s=16,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        ax.set_yscale("log")
        ax.set_xlabel("PHIF (fraction)")
        if index == 0:
            ax.set_ylabel("KLOGH (mD)")
        ax.grid(color=AKUN["grid"], lw=0.5, alpha=0.8)
        panel_label(ax, chr(ord("a") + index))
    if scatter is not None:
        cbar_ax = fig2.add_axes([0.91, 0.20, 0.018, 0.62])
        cbar = fig2.colorbar(scatter, cax=cbar_ax)
        cbar.set_label("Measured depth (m)")
    fig2.subplots_adjust(left=0.09, right=0.88, bottom=0.13, top=0.97, wspace=0.16)
    cross_caption = (
        "Paired PHIF–KLOGH relationship on identical known-holdout samples: "
        "(a) observations and (b) independently predicted properties. Colour "
        "encodes measured depth, and permeability remains on its physical log scale."
    )
    cross_bundle = save_figure_bundle(
        fig2,
        output_dir / "figures/property_rock_physics_crossplot",
    )
    _register_bundle(
        outputs,
        cross_bundle,
        role="property_rock_physics_consistency",
        evidence_mode="section_only",
        caption=cross_caption,
    )
    return {
        "track": "property",
        "evidence_mode": "section_only",
        "sources": [item.as_record(PROJECT_ROOT) for item in sources],
        "outputs": outputs,
        "scientific_boundary": (
            "The figure is an along-well holdout diagnostic; intervals are empirical "
            "OOF-residual intervals, not Bayesian posterior uncertainty."
        ),
    }


def render_lithofacies(output_dir: Path) -> dict[str, Any]:
    payload = json.loads(LITHOFACIES_PREDICTIONS.read_text(encoding="utf-8"))
    records = payload["records"]
    frame = pd.DataFrame(records).sort_values("twt_ms")
    twt = frame["twt_ms"].to_numpy(float)
    truth = frame["true_class_id"].to_numpy(int)
    prediction = frame["predicted_class_id"].to_numpy(int)
    confidence = frame["confidence"].to_numpy(float)
    error = frame["error"].to_numpy(bool)
    twt_edges = np.empty(len(twt) + 1, dtype=float)
    twt_edges[1:-1] = 0.5 * (twt[:-1] + twt[1:])
    twt_edges[0] = twt[0] - 0.5 * (twt[1] - twt[0])
    twt_edges[-1] = twt[-1] + 0.5 * (twt[-1] - twt[-2])

    publication_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 6.2),
        gridspec_kw={"width_ratios": [0.8, 0.8, 2.3]},
        sharey=True,
    )
    label_cmap = matplotlib.colormaps["tab10"]
    axes[0].pcolormesh(
        [0, 1],
        twt_edges,
        truth[:, None],
        cmap=label_cmap,
        vmin=-0.5,
        vmax=8.5,
        shading="flat",
    )
    axes[1].pcolormesh(
        [0, 1],
        twt_edges,
        prediction[:, None],
        cmap=label_cmap,
        vmin=-0.5,
        vmax=8.5,
        shading="flat",
    )
    for index, ax in enumerate(axes[:2]):
        ax.set_xlim(0, 1)
        ax.set_xticks([0.5], ["Observed" if index == 0 else "Predicted"])
        ax.invert_yaxis()
        ax.set_ylabel("TWT (ms)" if index == 0 else "")
        panel_label(ax, chr(ord("a") + index))
    axes[2].plot(confidence, twt, color=AKUN["blue"], lw=1.05, label="Confidence")
    axes[2].scatter(
        confidence[error],
        twt[error],
        s=14,
        facecolor=AKUN["red"],
        edgecolor="white",
        linewidth=0.3,
        label="Misclassified",
        zorder=3,
    )
    axes[2].axvline(1.0 / 9.0, color=AKUN["muted"], lw=0.8, ls="--")
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Maximum class probability")
    axes[2].grid(color=AKUN["grid"], lw=0.5, alpha=0.8)
    axes[2].legend(frameon=False, loc="lower right")
    axes[2].invert_yaxis()
    panel_label(axes[2], "c")
    scalar = cm.ScalarMappable(
        norm=colors.Normalize(-0.5, 8.5), cmap=label_cmap
    )
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=axes[:2], shrink=0.64, pad=0.08, ticks=np.arange(9))
    cbar.set_label("Fixed genetic-facies class")
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.08, top=0.98, wspace=0.28)

    caption = (
        "Fixed-nine lithofacies sequence for the known holdout well 15/9-F-5: "
        "(a) observed class, (b) XGBoost prediction and (c) maximum probability "
        "with all misclassified samples marked. The vertical axis is TWT because "
        "the archived records do not contain verified MD or XYZ coordinates."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(fig, output_dir / "figures/lithofacies_well_sequence")
    _register_bundle(
        outputs,
        bundle,
        role="lithofacies_sequence_diagnostics",
        evidence_mode="section_only",
        caption=caption,
    )
    source = EvidenceSource(
        LITHOFACIES_PREDICTIONS,
        "fixed-nine known-holdout class probabilities and labels",
        "previously seen reusable holdout: 15/9-F-5",
        f"{len(frame)} records",
    )
    return {
        "track": "lithofacies",
        "evidence_mode": "section_only",
        "sources": [source.as_record(PROJECT_ROOT)],
        "outputs": outputs,
        "scientific_boundary": (
            "The archived records contain TWT but no verified trajectory/XYZ; the "
            "figure is therefore a one-dimensional sequence, not a 3-D facies body."
        ),
    }


def render_sweetspot(output_dir: Path) -> dict[str, Any]:
    rqi_path = (
        SWEETSPOT_ROOT
        / "reservoir_quality/_outputs/baseline_v1/frozen_test/predictions.csv"
    )
    productivity_path = SWEETSPOT_T3_CONFIRMATION
    rqi = pd.read_csv(rqi_path).sort_values("depth_m")
    productivity = pd.read_csv(productivity_path).rename(columns={"actual": "observed"})
    productivity["cutoff_date"] = pd.to_datetime(productivity["cutoff_date"])
    productivity = productivity.sort_values("cutoff_date")

    publication_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    depth_step = max(1, len(rqi) // 3500)
    rqi_view = rqi.iloc[::depth_step]
    axes[0, 0].plot(
        rqi_view["observed"],
        rqi_view["depth_m"],
        color=AKUN["ink"],
        lw=0.85,
        label="Observed",
    )
    axes[0, 0].plot(
        rqi_view["prediction"],
        rqi_view["depth_m"],
        color=AKUN["orange"],
        lw=0.85,
        label="Predicted",
    )
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("T1 reservoir-quality proxy")
    axes[0, 0].set_ylabel("Measured depth (m)")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="x", color=AKUN["grid"], lw=0.5)

    rqi_residual = rqi_view["prediction"] - rqi_view["observed"]
    axes[1, 0].plot(
        rqi_residual,
        rqi_view["depth_m"],
        color=AKUN["blue"],
        lw=0.75,
    )
    axes[1, 0].axvline(0.0, color=AKUN["muted"], lw=0.8, ls="--")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("T1 residual")
    axes[1, 0].set_ylabel("Measured depth (m)")
    axes[1, 0].grid(axis="x", color=AKUN["grid"], lw=0.5)

    axes[0, 1].plot(
        productivity["cutoff_date"],
        productivity["observed"],
        color=AKUN["ink"],
        lw=1.05,
        label="Observed",
    )
    axes[0, 1].plot(
        productivity["cutoff_date"],
        productivity["prediction"],
        color=AKUN["cyan"],
        lw=1.05,
        label="Predicted",
    )
    axes[0, 1].set_ylabel("T3 mean oil rate")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(color=AKUN["grid"], lw=0.5)

    prod_residual = productivity["prediction"] - productivity["observed"]
    axes[1, 1].axhline(0.0, color=AKUN["muted"], lw=0.8, ls="--")
    axes[1, 1].plot(
        productivity["cutoff_date"],
        prod_residual,
        color=AKUN["blue"],
        lw=0.9,
    )
    axes[1, 1].fill_between(
        productivity["cutoff_date"],
        0,
        prod_residual,
        where=prod_residual >= 0,
        color=AKUN["orange"],
        alpha=0.32,
    )
    axes[1, 1].fill_between(
        productivity["cutoff_date"],
        0,
        prod_residual,
        where=prod_residual < 0,
        color=AKUN["blue"],
        alpha=0.25,
    )
    axes[1, 1].set_xlabel("Forecast cutoff date")
    axes[1, 1].set_ylabel("T3 residual")
    axes[1, 1].grid(color=AKUN["grid"], lw=0.5)
    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        panel_label(ax, label)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.98, hspace=0.24, wspace=0.24)

    caption = (
        "Sweetspot tasks remain target-specific: T1 depth-domain reservoir-quality "
        "proxy and residuals (a,b), and T3 causal-history productivity forecasts and "
        "residuals (c,d). They are not collapsed into an unsupported composite score."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(fig, output_dir / "figures/sweetspot_target_diagnostics")
    _register_bundle(
        outputs,
        bundle,
        role="sweetspot_target_specific_diagnostics",
        evidence_mode="section_only",
        caption=caption,
    )

    observed = productivity["observed"].to_numpy(float)
    predicted = productivity["prediction"].to_numpy(float)
    residual = predicted - observed
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else np.nan
    residual_std = float(np.std(residual, ddof=1))
    standardized = (
        np.sort((residual - residual.mean()) / residual_std)
        if residual_std > 0
        else np.zeros_like(residual)
    )
    probabilities = (np.arange(len(standardized), dtype=float) + 0.5) / len(
        standardized
    )
    theoretical = np.asarray(
        [NormalDist().inv_cdf(float(value)) for value in probabilities],
        dtype=float,
    )

    publication_style()
    fig2, diagnostic_axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    scatter_ax, qq_ax = diagnostic_axes
    scatter_ax.scatter(
        observed,
        predicted,
        c=np.arange(len(observed)),
        cmap=matplotlib.colormaps["viridis"],
        s=22,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )
    lower = float(min(np.nanmin(observed), np.nanmin(predicted)))
    upper = float(max(np.nanmax(observed), np.nanmax(predicted)))
    scatter_ax.plot([lower, upper], [lower, upper], color=AKUN["muted"], lw=0.9, ls="--")
    scatter_ax.set_xlabel("Observed mean oil rate")
    scatter_ax.set_ylabel("Predicted mean oil rate")
    scatter_ax.text(
        0.04,
        0.95,
        rf"$R^2={r2:.3f}$",
        transform=scatter_ax.transAxes,
        ha="left",
        va="top",
        color=AKUN["ink"],
    )
    scatter_ax.grid(color=AKUN["grid"], lw=0.5, alpha=0.8)
    panel_label(scatter_ax, "a")

    qq_ax.scatter(
        theoretical,
        standardized,
        color=AKUN["blue"],
        s=18,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )
    qq_limit = float(
        max(np.nanmax(np.abs(theoretical)), np.nanmax(np.abs(standardized)))
    )
    qq_ax.plot(
        [-qq_limit, qq_limit],
        [-qq_limit, qq_limit],
        color=AKUN["muted"],
        lw=0.9,
        ls="--",
    )
    qq_ax.set_xlabel("Theoretical normal quantile")
    qq_ax.set_ylabel("Standardized residual quantile")
    qq_ax.grid(color=AKUN["grid"], lw=0.5, alpha=0.8)
    panel_label(qq_ax, "b")
    fig2.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.97, wspace=0.25)
    performance_caption = (
        "T3 frozen temporal-holdout diagnostics: (a) observed versus predicted mean "
        f"oil rate with the identity line and R²={r2:.3f}; (b) residual normal "
        "Q–Q diagnostic. The colour ordering in (a) follows forecast time."
    )
    performance_bundle = save_figure_bundle(
        fig2,
        output_dir / "figures/sweetspot_productivity_performance",
    )
    _register_bundle(
        outputs,
        performance_bundle,
        role="sweetspot_productivity_performance",
        evidence_mode="section_only",
        caption=performance_caption,
    )
    sources = [
        EvidenceSource(
            rqi_path,
            "T1 reservoir-quality proxy observed/predicted depth samples",
            "previously seen reusable holdout: 15/9-F-15 D",
            f"{len(rqi)} rows",
        ),
        EvidenceSource(
            productivity_path,
            "T3 causal-history productivity observed/predicted samples",
            "previously seen reusable temporal holdout",
            f"{len(productivity)} rows",
        ),
    ]
    return {
        "track": "sweetspot",
        "evidence_mode": "section_only",
        "sources": [item.as_record(PROJECT_ROOT) for item in sources],
        "outputs": outputs,
        "scientific_boundary": (
            "T1 is a proxy and T3 is an observed production target. The two axes are "
            "different evidence domains and are not combined into one score."
        ),
    }


def _sample_active(
    archive: dict[str, np.ndarray],
    field: str,
    *,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.asarray(archive["active"], dtype=bool)
    values = np.asarray(archive[field], dtype=float)
    mask = active & np.isfinite(values)
    indices = np.flatnonzero(mask)
    if len(indices) > maximum:
        indices = indices[:: max(1, len(indices) // maximum)][:maximum]
    return (
        np.asarray(archive["easting_m"]).reshape(-1)[indices] / 1000.0,
        np.asarray(archive["northing_m"]).reshape(-1)[indices] / 1000.0,
        np.asarray(archive["depth_m"]).reshape(-1)[indices],
        values.reshape(-1)[indices],
    )


def _directional_semivariogram(
    volume: np.ndarray,
    active: np.ndarray,
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    axes: tuple[int, ...],
    maximum_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic axis-aligned empirical semivariogram points."""

    distances: list[float] = []
    semivariances: list[float] = []
    finite = np.isfinite(volume)
    for lag in range(1, maximum_lag + 1):
        lag_distances: list[np.ndarray] = []
        lag_semivariances: list[np.ndarray] = []
        for axis in axes:
            head = [slice(None)] * 3
            tail = [slice(None)] * 3
            head[axis] = slice(lag, None)
            tail[axis] = slice(None, -lag)
            head_t = tuple(head)
            tail_t = tuple(tail)
            pair_mask = (
                active[head_t]
                & active[tail_t]
                & finite[head_t]
                & finite[tail_t]
            )
            if not np.any(pair_mask):
                continue
            coordinate_delta = [
                coordinate[head_t][pair_mask] - coordinate[tail_t][pair_mask]
                for coordinate in coordinates
            ]
            lag_distances.append(
                np.sqrt(sum(delta**2 for delta in coordinate_delta))
            )
            difference = volume[head_t][pair_mask] - volume[tail_t][pair_mask]
            lag_semivariances.append(0.5 * difference**2)
        if lag_distances:
            distance_values = np.concatenate(lag_distances)
            semivariance_values = np.concatenate(lag_semivariances)
            distances.append(float(np.nanmedian(distance_values)))
            semivariances.append(float(np.nanmean(semivariance_values)))
    return np.asarray(distances), np.asarray(semivariances)


def render_reconstruction(output_dir: Path, *, interactive: bool) -> dict[str, Any]:
    reference_path = RECONSTRUCTION_ROOT / "reference/reference_property_volume.npz"
    strict_path = RECONSTRUCTION_ROOT / "strict/heldout_reconstruction_volume.npz"
    conditional_path = (
        RECONSTRUCTION_ROOT / "conditional/heldout_reconstruction_volume.npz"
    )
    with np.load(reference_path, allow_pickle=True) as archive:
        reference = {key: archive[key] for key in archive.files}
    with np.load(conditional_path, allow_pickle=True) as archive:
        conditional = {key: archive[key] for key in archive.files}

    publication_style()
    fig = plt.figure(figsize=(14.2, 4.8))
    fields = [
        (reference, "reference_porosity", POROSITY_CMAP, "Reference PHIF"),
        (conditional, "truth", POROSITY_CMAP, "Held-out truth"),
        (conditional, "prediction", POROSITY_CMAP, "Reconstruction"),
        (conditional, "residual", RESIDUAL_CMAP, "Residual"),
    ]
    all_axes: list[plt.Axes] = []
    for index, (archive, field, cmap, colorbar_label) in enumerate(fields):
        ax = fig.add_subplot(1, 4, index + 1, projection="3d")
        x, y, z, value = _sample_active(archive, field, maximum=10500)
        if field == "residual":
            limit = float(np.nanpercentile(np.abs(value), 99.0))
            norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
        else:
            lower, upper = np.nanpercentile(value, [1.0, 99.0])
            norm = colors.Normalize(float(lower), float(upper))
        scatter = ax.scatter(
            x,
            y,
            z,
            c=value,
            cmap=cmap,
            norm=norm,
            s=1.7,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
        )
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_zlabel("Depth (m)")
        ax.invert_zaxis()
        ax.view_init(elev=22, azim=-57)
        ax.grid(False)
        ax.set_box_aspect((1.15, 1.0, 1.2))
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.50, pad=0.02)
        cbar.set_label(colorbar_label)
        panel_label(ax, chr(ord("a") + index), x=-0.02, y=0.98)
        all_axes.append(ax)
    fig.subplots_adjust(left=0.0, right=0.995, bottom=0.01, top=0.99, wspace=0.05)
    caption = (
        "Registered Volve MAPAXES volumes: (a) reference porosity, (b) held-out "
        "conditional truth, (c) conditional reconstruction and (d) prediction-minus-"
        "truth residual. Residual is an observed error field, not posterior uncertainty."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(
        fig,
        output_dir / "figures/reconstruction_native_volume",
    )
    _register_bundle(
        outputs,
        bundle,
        role="reconstruction_native_volume_diagnostics",
        evidence_mode="native_volume",
        caption=caption,
    )

    # Orthogonal slice diagnostics retain voxel topology more clearly than a point cloud.
    active = np.asarray(conditional["active"], dtype=bool)
    truth = np.asarray(conditional["truth"], dtype=float)
    prediction = np.asarray(conditional["prediction"], dtype=float)
    residual = np.asarray(conditional["residual"], dtype=float)
    index_candidates = np.argwhere(active)
    center = np.median(index_candidates, axis=0).astype(int)
    rows = [
        ("truth", truth, POROSITY_CMAP),
        ("prediction", prediction, POROSITY_CMAP),
        ("residual", residual, RESIDUAL_CMAP),
    ]
    publication_style()
    fig2, axes = plt.subplots(3, 3, figsize=(10.8, 8.8))
    for row_index, (field, volume, cmap) in enumerate(rows):
        finite = np.isfinite(volume) & active
        values = volume[finite]
        if field == "residual":
            limit = float(np.nanpercentile(np.abs(values), 99.0))
            norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
        else:
            lower, upper = np.nanpercentile(values, [1.0, 99.0])
            norm = colors.Normalize(float(lower), float(upper))
        slices = [
            np.ma.masked_invalid(volume[center[0], :, :]),
            np.ma.masked_invalid(volume[:, center[1], :]),
            np.ma.masked_invalid(volume[:, :, center[2]]),
        ]
        labels = [
            ("I index", "J index"),
            ("I index", "K index"),
            ("J index", "K index"),
        ]
        image = None
        for column_index, (section, (xlabel, ylabel)) in enumerate(
            zip(slices, labels, strict=True)
        ):
            ax = axes[row_index, column_index]
            image = ax.imshow(
                section,
                cmap=cmap,
                norm=norm,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            panel_label(
                ax,
                chr(ord("a") + row_index * 3 + column_index),
            )
        if image is not None:
            cbar = fig2.colorbar(image, ax=axes[row_index, :], shrink=0.72, pad=0.02)
            cbar.set_label(field.capitalize())
    fig2.subplots_adjust(left=0.06, right=0.93, bottom=0.06, top=0.98, hspace=0.28, wspace=0.24)
    slice_caption = (
        "Orthogonal conditional-holdout sections through the registered grid. Rows "
        "show truth, reconstruction and residual; columns show K-, J- and I-normal "
        "cuts through the median active-cell location."
    )
    slice_bundle = save_figure_bundle(
        fig2,
        output_dir / "figures/reconstruction_orthogonal_diagnostics",
    )
    _register_bundle(
        outputs,
        slice_bundle,
        role="reconstruction_orthogonal_slices",
        evidence_mode="native_volume",
        caption=slice_caption,
    )

    # Directional variograms measure whether reconstruction retains spatial
    # correlation, rather than judging the volume only by pointwise error.
    coordinates = (
        np.asarray(conditional["easting_m"], dtype=float),
        np.asarray(conditional["northing_m"], dtype=float),
        np.asarray(conditional["depth_m"], dtype=float),
    )
    publication_style()
    fig3, variogram_axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    fields_for_variogram = [
        ("Truth", truth, AKUN["ink"], "-"),
        ("Reconstruction", prediction, AKUN["orange"], "-"),
        ("Residual", residual, AKUN["blue"], "--"),
    ]
    for ax, axes_group, label in (
        (variogram_axes[0], (0,), "K direction"),
        (variogram_axes[1], (1, 2), "I–J directions"),
    ):
        for field_label, volume, color, line_style in fields_for_variogram:
            lag_distance, semivariance = _directional_semivariogram(
                volume,
                active,
                coordinates,
                axes=axes_group,
                maximum_lag=12,
            )
            ax.plot(
                lag_distance,
                semivariance,
                color=color,
                ls=line_style,
                marker="o",
                ms=3.0,
                lw=1.0,
                label=field_label,
            )
        ax.set_xlabel(f"Physical lag distance (m), {label}")
        ax.set_ylabel(r"Empirical semivariance (PHIF$^2$)")
        ax.grid(color=AKUN["grid"], lw=0.5, alpha=0.8)
    variogram_axes[0].legend(frameon=False)
    panel_label(variogram_axes[0], "a")
    panel_label(variogram_axes[1], "b")
    fig3.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.97, wspace=0.24)
    variogram_caption = (
        "Directional empirical semivariograms on the conditional holdout: "
        "(a) K-index direction and (b) pooled I–J index directions, with physical "
        "lag distances derived from MAPAXES coordinates. Truth, reconstruction and "
        "observed residual are shown in the same PHIF² units."
    )
    variogram_bundle = save_figure_bundle(
        fig3,
        output_dir / "figures/reconstruction_directional_variogram",
    )
    _register_bundle(
        outputs,
        variogram_bundle,
        role="reconstruction_spatial_structure_diagnostics",
        evidence_mode="native_volume",
        caption=variogram_caption,
    )

    if interactive:
        if go is None:
            raise RuntimeError(
                "plotly is required for interactive reconstruction rendering"
            )
        scene = go.Figure()
        for field, color_scale, visible in [
            ("truth", "Viridis", True),
            ("prediction", "Viridis", True),
            ("residual", "RdBu", True),
        ]:
            x, y, z, value = _sample_active(conditional, field, maximum=14000)
            marker: dict[str, Any] = {
                "size": 2.1,
                "color": value,
                "colorscale": color_scale,
                "opacity": 0.45,
            }
            if field == "residual":
                limit = float(np.nanpercentile(np.abs(value), 99.0))
                marker.update({"cmin": -limit, "cmax": limit, "cmid": 0.0})
            scene.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=marker,
                    name=field.capitalize(),
                    visible=visible,
                    hovertemplate=(
                        f"{field}<br>E %{{x:.2f}} km<br>N %{{y:.2f}} km"
                        "<br>Depth %{z:.1f} m<br>Value %{marker.color:.4f}<extra></extra>"
                    ),
                )
            )
        layout = plotly_layout(z_title="Depth (m)")
        layout["scene"]["xaxis"]["title"] = "Easting (km)"
        layout["scene"]["yaxis"]["title"] = "Northing (km)"
        scene.update_layout(**layout)
        html = output_dir / "interactive/reconstruction_native_volume.html"
        write_plotly_html(scene, html)
        _register_html(
            outputs,
            html,
            role="reconstruction_interactive_volume",
            evidence_mode="native_volume",
            caption=caption,
        )

    sources = [
        EvidenceSource(
            reference_path,
            "registered Volve reference porosity grid",
            "reference full grid; not a model split",
            str(tuple(int(value) for value in reference["active"].shape)),
        ),
        EvidenceSource(
            strict_path,
            "strict held-out reconstruction volume",
            "previously seen reusable strict holdout",
            "63×56×41 registered cells",
        ),
        EvidenceSource(
            conditional_path,
            "conditional held-out truth, prediction and residual volumes",
            "previously seen reusable conditional holdout",
            str(tuple(int(value) for value in conditional["active"].shape)),
        ),
    ]
    return {
        "track": "reconstruction",
        "evidence_mode": "native_volume",
        "sources": [item.as_record(PROJECT_ROOT) for item in sources],
        "outputs": outputs,
        "scientific_boundary": (
            "Strict and conditional holdouts are previously seen reusable holdouts; "
            "residual is not relabeled as uncertainty."
        ),
    }


def render_research_matrix(output_dir: Path) -> dict[str, Any]:
    """Render an auditable six-track foundation-model research map.

    This is a structured evidence chart rather than a network architecture
    schematic.  Detailed per-track frameworks are generated separately with the
    SCI neural-network diagram workflow.
    """

    rows = [
        ("Fault", "Local logistic", "SAM-Med3D", "prompted 3-D adapter", "data gate"),
        ("Facies", "FPN / DeepLabV3+", "SAM2 Hiera", "cross-attention", "attribution"),
        ("Property", "ExtraTrees / XGBoost", "TabICL", "parallel comparator", "no promotion"),
        ("Lithofacies", "XGBoost", "MOMENT", "sequence encoder", "non-beneficial"),
        ("Sweetspot", "GBDT routing", "Chronos-2", "causal forecast", "T3 development"),
        ("Reconstruction", "OK3D", "OpenMind-MAE", "gated residual", "diagnostic"),
    ]
    publication_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    columns = [
        (0.02, 0.18, "Track"),
        (0.20, 0.43, "Scientific baseline"),
        (0.45, 0.62, "Foundation model"),
        (0.64, 0.82, "Controlled interface"),
        (0.84, 0.98, "Evidence state"),
    ]
    for left, right, label in columns:
        ax.add_patch(
            plt.Rectangle(
                (left, len(rows) - 0.18),
                right - left,
                0.50,
                facecolor=AKUN["ink"],
                edgecolor="none",
            )
        )
        ax.text(
            (left + right) / 2,
            len(rows) + 0.07,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
    for row_index, row in enumerate(rows):
        y = len(rows) - 1 - row_index
        background = AKUN["warm"] if row_index % 2 == 0 else "#FFFFFF"
        ax.add_patch(
            plt.Rectangle(
                (0.02, y - 0.34),
                0.96,
                0.68,
                facecolor=background,
                edgecolor=AKUN["grid"],
                lw=0.6,
            )
        )
        values = [
            (0.10, row[0], AKUN["ink"]),
            (0.315, row[1], AKUN["blue"]),
            (0.535, row[2], AKUN["orange"]),
            (0.73, row[3], AKUN["cyan"]),
            (0.91, row[4], AKUN["muted"]),
        ]
        for x, text, color in values:
            ax.text(x, y, text, ha="center", va="center", color=color, fontsize=8.5)
        ax.annotate(
            "",
            xy=(0.445, y),
            xytext=(0.425, y),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": AKUN["muted"]},
        )
        ax.annotate(
            "",
            xy=(0.635, y),
            xytext=(0.615, y),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": AKUN["muted"]},
        )
        ax.annotate(
            "",
            xy=(0.835, y),
            xytext=(0.815, y),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": AKUN["muted"]},
        )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.91)
    caption = (
        "Six-track foundation-model research map. Each pretrained model is attached "
        "to a reproducible scientific baseline through a controlled interface; the "
        "rightmost state is an evidence status rather than a performance claim."
    )
    outputs: list[dict[str, Any]] = []
    bundle = save_figure_bundle(fig, output_dir / "figures/foundation_model_research_map")
    _register_bundle(
        outputs,
        bundle,
        role="foundation_model_research_map",
        evidence_mode="section_only",
        caption=caption,
    )
    return {
        "track": "cross_track",
        "evidence_mode": "section_only",
        "sources": [
            EvidenceSource(
                FOUNDATION_MODEL_CONTRACT,
                "pre-registered six-track foundation-model ablation and promotion contract",
                "methodological contract; no test labels or generated evidence",
                "6 track protocols",
            ).as_record(PROJECT_ROOT)
        ],
        "outputs": outputs,
        "scientific_boundary": (
            "The chart records interfaces and evidence states only; it does not rank "
            "models or convert pending evidence into a positive result."
        ),
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    tracks = manifest["tracks"]
    expected = {
        "fault",
        "facies",
        "property",
        "lithofacies",
        "sweetspot",
        "reconstruction",
        "cross_track",
    }
    observed = {record["track"] for record in tracks}
    if observed != expected:
        raise ValueError(f"track coverage mismatch: {observed}")
    outputs = [item for track in tracks for item in track["outputs"]]
    if len(outputs) < 17:
        raise ValueError(f"expected at least 17 outputs, found {len(outputs)}")
    paths = [item["path"] for item in outputs]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate output paths")
    for record in outputs:
        path = PROJECT_ROOT / record["path"]
        if not path.is_file() or path.stat().st_size < 10_000:
            raise ValueError(f"missing or too-small output: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"output hash mismatch: {path}")
        if record["evidence_mode"] not in {
            "native_volume",
            "spatial_context",
            "section_only",
        }:
            raise ValueError(f"invalid evidence mode: {record}")
    for track in tracks:
        for source in track["sources"]:
            path = PROJECT_ROOT / source["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            if source["sha256"] != sha256_file(path):
                raise ValueError(f"source hash mismatch: {path}")


def render_all(output_dir: Path, *, interactive: bool = True) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    tracks = [
        render_fault(output_dir, interactive=interactive),
        render_facies(output_dir, interactive=interactive),
        render_property(output_dir),
        render_lithofacies(output_dir),
        render_sweetspot(output_dir),
        render_reconstruction(output_dir, interactive=interactive),
        render_research_matrix(output_dir),
    ]
    manifest = {
        "schema_version": "junwei-research-visualization-expansion/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": project_relative(Path(__file__), PROJECT_ROOT),
        "scientific_contract": {
            "synthetic_geology": False,
            "cross_field_coordinate_fusion": False,
            "reference_prediction_separation": True,
            "large_model_role": (
                "controlled encoder, comparator, or residual/forecast branch; "
                "never visual evidence generation"
            ),
        },
        "tracks": tracks,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    write_json(manifest_path, manifest)
    validate_manifest(manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = render_all(
        args.output_dir,
        interactive=not args.skip_interactive,
    )
    outputs = [
        output
        for track in manifest["tracks"]
        for output in track["outputs"]
    ]
    print(
        json.dumps(
            {
                "manifest": project_relative(
                    args.output_dir / "artifact_manifest.json", PROJECT_ROOT
                ),
                "track_records": len(manifest["tracks"]),
                "outputs": len(outputs),
                "png": sum(item["path"].endswith(".png") for item in outputs),
                "pdf": sum(item["path"].endswith(".pdf") for item in outputs),
                "html": sum(item["path"].endswith(".html") for item in outputs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
