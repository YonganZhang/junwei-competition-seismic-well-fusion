#!/usr/bin/env python3
"""Real 3-D PORO visualization for the reconstruction track.

This module only consumes archived strict/conditional prediction evidence.
It never reads frozen test assets and it never fabricates physical XYZ.
The spatial axes are the native voxel indices K/J/I from the archived NPZs.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
import numpy as np

try:  # Plotly is used only for HTML output.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover - the tests exercise the numpy/matplotlib path.
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TRACK_ROOT = HERE
ARCHIVE_ROOT = TRACK_ROOT / "p5_stage4_confirmation"
OUTPUT_ROOT = TRACK_ROOT / "_outputs" / "3d_sci_v1"
MODES = ("strict", "conditional")
MAX_RENDER_POINTS = 12_000
ROOT_SEED = 2693
NATIVE_AXES = ("K", "J", "I")
UKIYO_POROSITY = colors.LinearSegmentedColormap.from_list(
    "ukiyo_porosity",
    ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"],
    N=256,
)
UKIYO_RESIDUAL = colors.LinearSegmentedColormap.from_list(
    "ukiyo_residual",
    ["#264653", "#F7F3EA", "#E76F51"],
    N=256,
)
PLOTLY_UKIYO_POROSITY = [
    [0.0, "#264653"],
    [0.38, "#2A9D8F"],
    [0.72, "#E9C46A"],
    [1.0, "#E76F51"],
]
PLOTLY_UKIYO_RESIDUAL = [
    [0.0, "#264653"],
    [0.5, "#F7F3EA"],
    [1.0, "#E76F51"],
]


def _resolve_font() -> tuple[str, dict[str, Any]]:
    requested = "Times New Roman"
    try:
        font_manager.findfont(requested, fallback_to_default=False)
        selected = requested
        available = True
    except ValueError:
        selected = "Liberation Serif"
        available = False
    return selected, {
        "requested": requested,
        "selected": selected,
        "times_new_roman_available": available,
        "limitation": None if available else "Times New Roman is not installed on this host.",
    }


FONT_FAMILY, FONT_STATUS = _resolve_font()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def normalize_fonts(fig: plt.Figure) -> None:
    """Normalize every visible text object with a deterministic serif fallback."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY, "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily(FONT_FAMILY)


def _load_archive(mode: str) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES!r}, got {mode!r}")
    archive_path = ARCHIVE_ROOT / mode / "predictions.npz"
    manifest_path = ARCHIVE_ROOT / mode / "manifest.json"
    metrics_path = ARCHIVE_ROOT / mode / "metrics.json"
    state_path = ARCHIVE_ROOT / mode / "confirmation_state.json"
    for path in (archive_path, manifest_path, metrics_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(f"required archived evidence missing: {path}")
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {
            "mode",
            "task_id",
            "evidence_class",
            "prior_test_consumed",
            "fresh_blind",
            "indices_kji",
            "volume_shape_kji",
            "truth",
            "prediction",
            "residual",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"prediction archive missing required fields: {missing}")
        payload = {name: archive[name].copy() for name in archive.files}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    archive_mode = str(np.asarray(payload["mode"]).item())
    if archive_mode != mode:
        raise ValueError(f"archive mode mismatch: expected {mode!r}, got {archive_mode!r}")
    if not bool(np.asarray(payload["prior_test_consumed"]).item()):
        raise ValueError("archived evidence must mark prior_test_consumed=true")
    if bool(np.asarray(payload["fresh_blind"]).item()):
        raise ValueError("archived evidence must not claim fresh_blind=true")
    if str(np.asarray(payload["evidence_class"]).item()) != "previously_seen_reusable_holdout":
        raise ValueError("unexpected evidence class in archived prediction evidence")
    if manifest.get("prior_test_consumed") is not True or manifest.get("fresh_blind") is not False:
        raise ValueError("manifest does not preserve previously seen reusable holdout boundary")
    if state.get("state") != "CONFIRMATION_COMPLETE":
        raise ValueError("confirmation state is not complete")
    return {
        "mode": mode,
        "archive_path": archive_path,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "state_path": state_path,
        "manifest": manifest,
        "metrics": metrics,
        "state": state,
        "indices_kji": np.asarray(payload["indices_kji"], dtype=np.int64),
        "volume_shape_kji": tuple(int(value) for value in np.asarray(payload["volume_shape_kji"], dtype=np.int64)),
        "truth": np.asarray(payload["truth"], dtype=np.float64),
        "prediction": np.asarray(payload["prediction"], dtype=np.float64),
        "residual": np.asarray(payload["residual"], dtype=np.float64),
    }


def _lexsort_sample(indices_kji: np.ndarray, max_points: int = MAX_RENDER_POINTS) -> np.ndarray:
    indices = np.asarray(indices_kji, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices_kji must have shape [N, 3]")
    total = indices.shape[0]
    if total <= max_points:
        return np.arange(total, dtype=np.int64)
    order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
    selected = np.linspace(0, total - 1, max_points, dtype=np.int64)
    return np.asarray(order[selected], dtype=np.int64)


def _format_caveat(mode: str) -> str:
    if mode == "conditional":
        return (
            "conditional reconstruction, not strict holdout; test-region constraints supplied, "
            "exact well cells excluded from metrics"
        )
    return "strict spatial holdout; no test-region well constraints used"


def _create_caption(mode: str, archive: dict[str, Any]) -> str:
    metrics = archive["metrics"]
    metric_key = "conditional_rmse" if mode == "conditional" else "strict_rmse"
    mae_key = "conditional_mae" if mode == "conditional" else "strict_mae"
    r2_key = "conditional_r2" if mode == "conditional" else "strict_r2"
    spec_key = "conditional_spectral_log_rmse" if mode == "conditional" else "strict_spectral_log_rmse"
    return (
        f"Native-volume 3-D porosity comparison for {mode}.\n"
        f"Axes: native voxel K/J/I only; no physical XYZ were reconstructed.\n"
        f"{_format_caveat(mode)}.\n"
        f"Metrics on archived holdout voxels: RMSE={metrics['metrics'][metric_key]:.6f}, "
        f"MAE={metrics['metrics'][mae_key]:.6f}, R²={metrics['metrics'][r2_key]:.6f}, "
        f"spectral log-RMSE={metrics['metrics'][spec_key]:.6f}.\n"
        f"Evidence class: {archive['state']['evidence_class']}."
    )


def _feasibility_record(mode: str, archive: dict[str, Any], sample_indices: np.ndarray) -> dict[str, Any]:
    return {
        "schema_version": "p5-three-d-sci-visualization-v1",
        "track_id": "reconstruction",
        "mode": mode,
        "feasibility": "native_volume",
        "native_volume": True,
        "spatial_context": False,
        "native_axes": list(NATIVE_AXES),
        "coordinate_system": "native voxel K/J/I",
        "coordinate_units": "voxel index",
        "volume_shape_kji": list(archive["volume_shape_kji"]),
        "field_names": ["truth", "prediction", "residual"],
        "source_paths": {
            "predictions": _project_relative(archive["archive_path"]),
            "manifest": _project_relative(archive["manifest_path"]),
            "metrics": _project_relative(archive["metrics_path"]),
            "state": _project_relative(archive["state_path"]),
        },
        "render_sampling": {
            "rule": "lexsort by K/J/I then evenly spaced sample indices",
            "selected_points": int(sample_indices.size),
            "max_points": MAX_RENDER_POINTS,
        },
        "boundary": {
            "strict": "no test-region well constraints used",
            "conditional": "test-region constraints supplied; exact well cells excluded from metrics",
        },
        "limitations": [
            "native voxel indices only",
            "no physical XYZ coordinates were introduced",
            "static figure uses a deterministic subsample for readability",
        ],
    }


def _provenance_record(mode: str, archive: dict[str, Any], sample_indices: np.ndarray) -> dict[str, Any]:
    truth = archive["truth"].astype(np.float64, copy=False)
    prediction = archive["prediction"].astype(np.float64, copy=False)
    residual = archive["residual"].astype(np.float64, copy=False)
    indices = archive["indices_kji"].astype(np.int64, copy=False)
    return {
        "schema_version": "p5-three-d-sci-visualization-v1",
        "track_id": "reconstruction",
        "mode": mode,
        "root_seed": ROOT_SEED,
        "code_path": _project_relative(HERE / "visualize_3d_sci.py"),
        "code_sha256": _sha256_file(HERE / "visualize_3d_sci.py"),
        "inputs": {
            "predictions_archive": {
                "path": _project_relative(archive["archive_path"]),
                "sha256": _sha256_file(archive["archive_path"]),
            },
            "manifest": {
                "path": _project_relative(archive["manifest_path"]),
                "sha256": _sha256_file(archive["manifest_path"]),
            },
            "metrics": {
                "path": _project_relative(archive["metrics_path"]),
                "sha256": _sha256_file(archive["metrics_path"]),
            },
            "confirmation_state": {
                "path": _project_relative(archive["state_path"]),
                "sha256": _sha256_file(archive["state_path"]),
            },
        },
        "array_hashes": {
            "indices_kji": _sha256_bytes(indices.tobytes()),
            "truth": _sha256_bytes(truth.tobytes()),
            "prediction": _sha256_bytes(prediction.tobytes()),
            "residual": _sha256_bytes(residual.tobytes()),
        },
        "native_volume": {
            "present": True,
            "shape_kji": list(archive["volume_shape_kji"]),
            "axes": list(NATIVE_AXES),
            "coordinates": "native voxel K/J/I only",
        },
        "sampling": {
            "rule": "lexsort by K/J/I then evenly spaced indices",
            "selected_points": int(sample_indices.size),
            "max_points": MAX_RENDER_POINTS,
        },
        "render_engine": {
            "matplotlib": matplotlib.__version__,
            "plotly": getattr(sys.modules.get("plotly"), "__version__", None),
            "interactive_html": True,
        },
        "font_status": FONT_STATUS,
    }


def _panel_label(ax: Any, label: str) -> None:
    ax.text2D(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _apply_axes_style(ax: Any, shape_kji: tuple[int, int, int]) -> None:
    ax.set_xlim(-0.5, shape_kji[0] - 0.5)
    ax.set_ylim(-0.5, shape_kji[1] - 0.5)
    ax.set_zlim(-0.5, shape_kji[2] - 0.5)
    ax.set_box_aspect((shape_kji[0], shape_kji[1], shape_kji[2]))
    ax.view_init(elev=20, azim=-55)
    ax.set_xlabel("K voxel index")
    ax.set_ylabel("J voxel index")
    ax.set_zlabel("I voxel index")
    ax.grid(False)
    ax.tick_params(labelsize=8, pad=1)


def _static_figure(mode: str, archive: dict[str, Any], sample_indices: np.ndarray, output_png: Path, output_pdf: Path) -> None:
    indices = archive["indices_kji"][sample_indices]
    truth = archive["truth"][sample_indices]
    prediction = archive["prediction"][sample_indices]
    residual = archive["residual"][sample_indices]
    shape = archive["volume_shape_kji"]
    truth_pred_min = float(min(truth.min(), prediction.min()))
    truth_pred_max = float(max(truth.max(), prediction.max()))
    residual_bound = float(max(1e-12, np.max(np.abs(residual))))

    fig = plt.figure(figsize=(15.8, 5.8))
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    scatter_kwargs = dict(s=3.0, alpha=0.85, depthshade=False, linewidths=0.0)
    panels = (
        (axes[0], truth, "truth", UKIYO_POROSITY, truth_pred_min, truth_pred_max),
        (axes[1], prediction, "reconstruction", UKIYO_POROSITY, truth_pred_min, truth_pred_max),
        (axes[2], residual, "residual", UKIYO_RESIDUAL, -residual_bound, residual_bound),
    )
    for index, (ax, values, label, cmap, vmin, vmax) in enumerate(panels, start=1):
        _panel_label(ax, chr(ord("a") + index - 1))
        _apply_axes_style(ax, shape)
        scatter = ax.scatter(
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
            c=values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **scatter_kwargs,
        )
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02, shrink=0.88)
        cbar.ax.tick_params(labelsize=8)
        if label == "residual":
            cbar.set_label("prediction - truth", fontsize=9)
        else:
            cbar.set_label("porosity", fontsize=9)
    normalize_fonts(fig)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    fig.savefig(output_pdf, dpi=300, facecolor="white")
    plt.close(fig)


def _single_panel_figure(
    *,
    mode: str,
    archive: dict[str, Any],
    sample_indices: np.ndarray,
    values: np.ndarray,
    label: str,
    panel_letter: str,
    cmap: colors.Colormap,
    vmin: float,
    vmax: float,
    output_png: Path,
    output_pdf: Path,
) -> None:
    indices = archive["indices_kji"][sample_indices]
    shape = archive["volume_shape_kji"]
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    _panel_label(ax, panel_letter)
    _apply_axes_style(ax, shape)
    scatter = ax.scatter(
        indices[:, 0],
        indices[:, 1],
        indices[:, 2],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=3.0,
        alpha=0.9,
        depthshade=False,
        linewidths=0.0,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.03, shrink=0.86)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("prediction - truth" if label == "residual" else "porosity", fontsize=9)
    normalize_fonts(fig)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, facecolor="white")
    fig.savefig(output_pdf, dpi=300, facecolor="white")
    plt.close(fig)


def _plotly_figure(mode: str, archive: dict[str, Any], sample_indices: np.ndarray, output_html: Path) -> None:
    if go is None or make_subplots is None:
        raise RuntimeError("plotly is unavailable; cannot write interactive HTML")
    indices = archive["indices_kji"][sample_indices]
    truth = archive["truth"][sample_indices]
    prediction = archive["prediction"][sample_indices]
    residual = archive["residual"][sample_indices]
    shape = archive["volume_shape_kji"]
    truth_pred_min = float(min(truth.min(), prediction.min()))
    truth_pred_max = float(max(truth.max(), prediction.max()))
    residual_bound = float(max(1e-12, np.max(np.abs(residual))))
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.03,
    )
    traces = (
        (1, 1, truth, "truth", PLOTLY_UKIYO_POROSITY, truth_pred_min, truth_pred_max),
        (1, 2, prediction, "reconstruction", PLOTLY_UKIYO_POROSITY, truth_pred_min, truth_pred_max),
        (1, 3, residual, "residual", PLOTLY_UKIYO_RESIDUAL, -residual_bound, residual_bound),
    )
    for col, _, values, name, colorscale, vmin, vmax in traces:
        scatter = go.Scatter3d(
            x=indices[:, 0],
            y=indices[:, 1],
            z=indices[:, 2],
            mode="markers",
            name=name,
            marker={
                "size": 2.6,
                "opacity": 0.85,
                "color": values,
                "colorscale": colorscale,
                "cmin": vmin,
                "cmax": vmax,
                "colorbar": {"title": "porosity" if name != "residual" else "prediction - truth"},
            },
            showlegend=False,
        )
        fig.add_trace(scatter, row=1, col=col)
    scene_template = dict(
        xaxis_title="K voxel index",
        yaxis_title="J voxel index",
        zaxis_title="I voxel index",
        xaxis=dict(range=[-0.5, shape[0] - 0.5], backgroundcolor="white", gridcolor="#dddddd", showbackground=True),
        yaxis=dict(range=[-0.5, shape[1] - 0.5], backgroundcolor="white", gridcolor="#dddddd", showbackground=True),
        zaxis=dict(range=[-0.5, shape[2] - 0.5], backgroundcolor="white", gridcolor="#dddddd", showbackground=True),
        aspectmode="data",
        camera=dict(eye=dict(x=1.55, y=1.35, z=0.95)),
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Times New Roman, Times, serif", size=13, color="#111111"),
        showlegend=False,
    )
    fig.update_scenes(scene_template, selector=dict(type="scene"))
    fig.add_annotation(
        text="<b>a</b>",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.985,
        showarrow=False,
        font=dict(family="Times New Roman, Times, serif", size=16, color="#111111"),
    )
    fig.add_annotation(
        text="<b>b</b>",
        xref="paper",
        yref="paper",
        x=0.344,
        y=0.985,
        showarrow=False,
        font=dict(family="Times New Roman, Times, serif", size=16, color="#111111"),
    )
    fig.add_annotation(
        text="<b>c</b>",
        xref="paper",
        yref="paper",
        x=0.678,
        y=0.985,
        showarrow=False,
        font=dict(family="Times New Roman, Times, serif", size=16, color="#111111"),
    )
    html_body = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True},
    )
    header = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<link rel='icon' href='data:,'>"
        "<style>body{font-family:Times New Roman,Times,serif;background:#fff;color:#111;margin:0;padding:12px;}"
        ".note{max-width:1200px;margin:0 auto 10px auto;font-size:14px;line-height:1.35;}</style>"
        "</head><body>"
        f"<div class='note'>{_create_caption(mode, archive).replace(chr(10), '<br>')}</div>"
    )
    footer = "</body></html>"
    output_html.parent.mkdir(parents=True, exist_ok=True)
    document = header + html_body + footer
    # Plotly's generated JavaScript contains trailing spaces on some releases.
    # Normalize them so the committed interactive artifact is reproducible and
    # passes the repository-wide whitespace check after every regeneration.
    document = "\n".join(line.rstrip() for line in document.splitlines()) + "\n"
    output_html.write_text(document, encoding="utf-8")


def render_mode(mode: str, output_root: Path = OUTPUT_ROOT, max_points: int = MAX_RENDER_POINTS) -> dict[str, Any]:
    archive = _load_archive(mode)
    sample_indices = _lexsort_sample(archive["indices_kji"], max_points=max_points)
    mode_dir = output_root / mode
    png = mode_dir / "prediction_comparison.png"
    pdf = mode_dir / "prediction_comparison.pdf"
    html = mode_dir / "prediction_comparison.html"
    caption = mode_dir / "caption.md"
    provenance = mode_dir / "provenance.json"
    feasibility = mode_dir / "three_d_feasibility.json"
    _static_figure(mode, archive, sample_indices, png, pdf)
    metrics = archive["metrics"]["metrics"]
    mode_metrics = {
        "strict": {
            "truth": archive["truth"],
            "reconstruction": archive["prediction"],
            "residual": archive["residual"],
            "truth_png": mode_dir / "truth.png",
            "truth_pdf": mode_dir / "truth.pdf",
            "reconstruction_png": mode_dir / "reconstruction.png",
            "reconstruction_pdf": mode_dir / "reconstruction.pdf",
            "residual_png": mode_dir / "residual.png",
            "residual_pdf": mode_dir / "residual.pdf",
        },
        "conditional": {
            "truth": archive["truth"],
            "reconstruction": archive["prediction"],
            "residual": archive["residual"],
            "truth_png": mode_dir / "truth.png",
            "truth_pdf": mode_dir / "truth.pdf",
            "reconstruction_png": mode_dir / "reconstruction.png",
            "reconstruction_pdf": mode_dir / "reconstruction.pdf",
            "residual_png": mode_dir / "residual.png",
            "residual_pdf": mode_dir / "residual.pdf",
        },
    }[mode]
    truth = mode_metrics["truth"][sample_indices]
    prediction = mode_metrics["reconstruction"][sample_indices]
    residual = mode_metrics["residual"][sample_indices]
    truth_pred_min = float(min(truth.min(), prediction.min()))
    truth_pred_max = float(max(truth.max(), prediction.max()))
    residual_bound = float(max(1e-12, np.max(np.abs(residual))))
    _single_panel_figure(
        mode=mode,
        archive=archive,
        sample_indices=sample_indices,
        values=truth,
        label="truth",
        panel_letter="a",
        cmap=UKIYO_POROSITY,
        vmin=truth_pred_min,
        vmax=truth_pred_max,
        output_png=mode_metrics["truth_png"],
        output_pdf=mode_metrics["truth_pdf"],
    )
    _single_panel_figure(
        mode=mode,
        archive=archive,
        sample_indices=sample_indices,
        values=prediction,
        label="reconstruction",
        panel_letter="b",
        cmap=UKIYO_POROSITY,
        vmin=truth_pred_min,
        vmax=truth_pred_max,
        output_png=mode_metrics["reconstruction_png"],
        output_pdf=mode_metrics["reconstruction_pdf"],
    )
    _single_panel_figure(
        mode=mode,
        archive=archive,
        sample_indices=sample_indices,
        values=residual,
        label="residual",
        panel_letter="c",
        cmap=UKIYO_RESIDUAL,
        vmin=-residual_bound,
        vmax=residual_bound,
        output_png=mode_metrics["residual_png"],
        output_pdf=mode_metrics["residual_pdf"],
    )
    _plotly_figure(mode, archive, sample_indices, html)
    caption.write_text(_create_caption(mode, archive) + "\n", encoding="utf-8")
    provenance_payload = _provenance_record(mode, archive, sample_indices)
    feasibility_payload = _feasibility_record(mode, archive, sample_indices)
    provenance.write_text(json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    feasibility.write_text(json.dumps(feasibility_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "mode": mode,
        "output_dir": _project_relative(mode_dir),
        "png": _project_relative(png),
        "pdf": _project_relative(pdf),
        "html": _project_relative(html),
        "panel_pngs": {
            "truth": _project_relative(mode_dir / "truth.png"),
            "reconstruction": _project_relative(mode_dir / "reconstruction.png"),
            "residual": _project_relative(mode_dir / "residual.png"),
        },
        "panel_pdfs": {
            "truth": _project_relative(mode_dir / "truth.pdf"),
            "reconstruction": _project_relative(mode_dir / "reconstruction.pdf"),
            "residual": _project_relative(mode_dir / "residual.pdf"),
        },
        "caption": _project_relative(caption),
        "provenance": _project_relative(provenance),
        "feasibility": _project_relative(feasibility),
        "sample_points": int(sample_indices.size),
        "source_sha256": {
            "predictions": _sha256_file(archive["archive_path"]),
            "manifest": _sha256_file(archive["manifest_path"]),
            "metrics": _sha256_file(archive["metrics_path"]),
            "state": _sha256_file(archive["state_path"]),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=MODES,
        nargs="*",
        default=list(MODES),
        help="render one or both reconstruction modes",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="root directory for 3-D figure outputs",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=MAX_RENDER_POINTS,
        help="maximum number of voxel points to render per mode",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    selected_modes = args.mode if isinstance(args.mode, list) else [args.mode]
    for mode in selected_modes:
        render_mode(mode, output_root=args.output_root, max_points=args.max_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
