#!/usr/bin/env python3
"""Render petroleum-domain 3-D porosity property volumes in physical coordinates.

This is the physical-volume successor to ``visualize_3d_sci.py``.  It reads the
native Volve Eclipse GRID/INIT archive, reconstructs MAPAXES cell-centre
coordinates, verifies PORO against the ASCII GRDECL export, and keeps the
Stage-4 strict/conditional evidence boundary intact.

Outputs deliberately distinguish:

* the full-field *reference* Eclipse porosity volume; and
* regional, previously-seen holdout *model* truth/prediction/residual volumes.

No scatter trace is used.  Static figures use VTK volume ray casting and
orthogonal slices; interactive figures use Plotly Volume/Isosurface traces.
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
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover - optional for data/static-only execution.
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

try:
    import pyvista as pv
except Exception:  # pragma: no cover - optional for data/interactive-only execution.
    pv = None  # type: ignore[assignment]

import build_dataset as dataset_builder
import visualize_3d_sci as archived_visualization


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
OUTPUT_ROOT = HERE / "_outputs" / "3d_property_volume_v2"
MODES = ("strict", "conditional")
ROOT_SEED = 2693
PORO_HEX = ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"]
RESIDUAL_HEX = ["#264653", "#F7F3EA", "#E76F51"]
PORO_CMAP = colors.LinearSegmentedColormap.from_list("ukiyo_porosity", PORO_HEX, N=256)
RESIDUAL_CMAP = colors.LinearSegmentedColormap.from_list(
    "ukiyo_residual", RESIDUAL_HEX, N=256
)
PLOTLY_PORO = [[0.0, PORO_HEX[0]], [0.38, PORO_HEX[1]], [0.72, PORO_HEX[2]], [1.0, PORO_HEX[3]]]
PLOTLY_RESIDUAL = [[0.0, RESIDUAL_HEX[0]], [0.5, RESIDUAL_HEX[1]], [1.0, RESIDUAL_HEX[2]]]


@dataclasses.dataclass(frozen=True)
class EclipsePropertyVolume:
    shape_kji: tuple[int, int, int]
    easting_m: np.ndarray
    northing_m: np.ndarray
    depth_m: np.ndarray
    active: np.ndarray
    porosity: np.ndarray
    inspection: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class RegionalModelVolume:
    mode: str
    source_shape_kji: tuple[int, int, int]
    source_slice_kji: tuple[slice, slice, slice]
    easting_m: np.ndarray
    northing_m: np.ndarray
    depth_m: np.ndarray
    active: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    residual: np.ndarray
    archive: dict[str, Any]


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
        "limitation": None if available else "Times New Roman is not installed; Liberation Serif used.",
    }


FONT_FAMILY, FONT_STATUS = _resolve_font()


def _normalize_fonts(fig: plt.Figure) -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY, "Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily(FONT_FAMILY)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _discover_eclipse_zip(explicit: Path | None = None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    relative = Path("_sandbox/volve_data/Volve_Reservoir_Model-Eclipse_model.zip")
    candidates = [PROJECT_ROOT / relative]
    candidates.extend(parent / relative for parent in PROJECT_ROOT.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Volve Eclipse ZIP not found. Pass --eclipse-zip explicitly; searched: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_eclipse_property_volume(eclipse_zip: Path) -> EclipsePropertyVolume:
    """Load and cross-check native Eclipse cell-centre geometry and PORO."""
    with ZipFile(eclipse_zip) as zf:
        shape, centres, active_flat = dataset_builder._parse_eclipse_grid(zf)
        init_poro = dataset_builder._parse_eclipse_init_poro(zf)
        ascii_poro = dataset_builder._parse_ascii_grdecl_property(
            zf,
            dataset_builder.ECLIPSE_ASCII_PORO_MEMBER,
            "PORO",
        )
    n_cells = int(np.prod(shape))
    if centres.shape != (n_cells, 3):
        raise ValueError(f"unexpected centre array shape {centres.shape}")
    if active_flat.shape != (n_cells,):
        raise ValueError(f"unexpected active-mask shape {active_flat.shape}")
    if ascii_poro.size != n_cells:
        raise ValueError(f"ASCII PORO has {ascii_poro.size} values; expected {n_cells}")
    if init_poro.size != int(active_flat.sum()):
        raise ValueError(
            f"INIT PORO has {init_poro.size} values; active grid has {active_flat.sum()} cells"
        )
    if not np.array_equal(ascii_poro[active_flat], init_poro):
        raise ValueError("INIT PORO and ASCII PORO disagree on active cells")
    porosity_flat = np.full(n_cells, np.nan, dtype=np.float32)
    porosity_flat[active_flat] = init_poro
    easting = centres[:, 0].reshape(shape).astype(np.float32)
    northing = centres[:, 1].reshape(shape).astype(np.float32)
    depth = centres[:, 2].reshape(shape).astype(np.float32)
    active = active_flat.reshape(shape)
    porosity = porosity_flat.reshape(shape)
    finite = porosity[active]
    inspection = {
        "schema_version": "volve-eclipse-property-volume-v2",
        "source_zip": _portable_path(eclipse_zip),
        "source_sha256": _sha256_file(eclipse_zip),
        "grid_member": dataset_builder.ECLIPSE_GRID_MEMBER,
        "init_member": dataset_builder.ECLIPSE_INIT_MEMBER,
        "ascii_porosity_member": dataset_builder.ECLIPSE_ASCII_PORO_MEMBER,
        "shape_kji": list(shape),
        "n_cells": n_cells,
        "n_active": int(active.sum()),
        "coordinate_system": "Eclipse MAPAXES-projected cell-centre coordinates",
        "horizontal_coordinates": "Easting/Northing in metres",
        "vertical_coordinate": "Eclipse grid depth coordinate in metres, positive downward",
        "porosity": {
            "name": "PHIF_NW / PORO",
            "unit": "fraction",
            "min": float(finite.min()),
            "max": float(finite.max()),
            "mean": float(finite.mean()),
        },
        "coordinate_extents": {
            "easting_m": [float(easting[active].min()), float(easting[active].max())],
            "northing_m": [float(northing[active].min()), float(northing[active].max())],
            "depth_m": [float(depth[active].min()), float(depth[active].max())],
        },
        "validation": {
            "ascii_matches_final_init_on_active_cells": True,
            "no_synthetic_coordinates": True,
            "no_synthetic_porosity": True,
        },
    }
    return EclipsePropertyVolume(shape, easting, northing, depth, active, porosity, inspection)


def _regional_model_volume(mode: str, volume: EclipsePropertyVolume) -> RegionalModelVolume:
    archive = archived_visualization._load_archive(mode)
    archive_shape = tuple(int(value) for value in archive["volume_shape_kji"])
    if len(archive_shape) != 3 or any(
        archive_size > eclipse_size
        for archive_size, eclipse_size in zip(archive_shape, volume.shape_kji)
    ):
        raise ValueError(f"{mode} archive shape {archive_shape} is incompatible with Eclipse {volume.shape_kji}")
    indices = np.asarray(archive["indices_kji"], dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError("indices_kji must be [N,3]")
    if np.unique(indices, axis=0).shape[0] != indices.shape[0]:
        raise ValueError(f"{mode} archive contains duplicate cell indices")
    for axis, size in enumerate(volume.shape_kji):
        if indices[:, axis].min() < 0 or indices[:, axis].max() >= size:
            raise ValueError(f"{mode} indices fall outside Eclipse axis {axis}")
        if indices[:, axis].max() >= archive_shape[axis]:
            raise ValueError(f"{mode} indices fall outside its declared archive shape on axis {axis}")
    if not volume.active[tuple(indices.T)].all():
        raise ValueError(f"{mode} archive includes inactive Eclipse cells")
    reference_truth = volume.porosity[tuple(indices.T)]
    if not np.array_equal(np.asarray(archive["truth"], dtype=np.float32), reference_truth):
        raise ValueError(
            f"{mode} archive truth does not exactly map to the native Eclipse PORO cells"
        )
    mins = indices.min(axis=0)
    maxs = indices.max(axis=0)
    region = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(mins, maxs))
    local = indices - mins
    regional_shape = tuple(int(hi - lo + 1) for lo, hi in zip(mins, maxs))
    fields: dict[str, np.ndarray] = {}
    for name in ("truth", "prediction", "residual"):
        values = np.asarray(archive[name], dtype=np.float32)
        if values.shape != (indices.shape[0],):
            raise ValueError(f"{mode} {name} shape mismatch")
        dense = np.full(regional_shape, np.nan, dtype=np.float32)
        dense[tuple(local.T)] = values
        fields[name] = dense
    if not np.allclose(
        fields["residual"][np.isfinite(fields["residual"])],
        (
            fields["prediction"] - fields["truth"]
        )[np.isfinite(fields["residual"])],
        rtol=0,
        atol=1e-7,
    ):
        raise ValueError(f"{mode} residual is not prediction - truth")
    return RegionalModelVolume(
        mode=mode,
        source_shape_kji=volume.shape_kji,
        source_slice_kji=region,  # type: ignore[arg-type]
        easting_m=volume.easting_m[region],
        northing_m=volume.northing_m[region],
        depth_m=volume.depth_m[region],
        active=volume.active[region],
        truth=fields["truth"],
        prediction=fields["prediction"],
        residual=fields["residual"],
        archive=archive,
    )


def _slice_bounds(region: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(item.start), int(item.stop)] for item in region]


def _save_npz_assets(
    volume: EclipsePropertyVolume,
    regional: dict[str, RegionalModelVolume],
    output_root: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    reference_dir = output_root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_npz = reference_dir / "reference_property_volume.npz"
    np.savez_compressed(
        reference_npz,
        shape_kji=np.asarray(volume.shape_kji, dtype=np.int32),
        easting_m=volume.easting_m,
        northing_m=volume.northing_m,
        depth_m=volume.depth_m,
        active=volume.active,
        reference_porosity=volume.porosity,
        coordinate_contract=np.asarray(
            "MAPAXES cell centres; easting/northing/depth metres; depth positive downward"
        ),
        property_contract=np.asarray("reference PHIF_NW/PORO fraction; NaN means inactive"),
    )
    paths["reference_npz"] = reference_npz
    for mode, item in regional.items():
        mode_dir = output_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        model_npz = mode_dir / "heldout_reconstruction_volume.npz"
        np.savez_compressed(
            model_npz,
            source_shape_kji=np.asarray(item.source_shape_kji, dtype=np.int32),
            source_slice_kji=np.asarray(_slice_bounds(item.source_slice_kji), dtype=np.int32),
            easting_m=item.easting_m,
            northing_m=item.northing_m,
            depth_m=item.depth_m,
            active=item.active,
            truth=item.truth,
            prediction=item.prediction,
            residual=item.residual,
            mode=np.asarray(mode),
            evidence_class=np.asarray(item.archive["state"]["evidence_class"]),
            prior_test_consumed=np.asarray(True),
            fresh_blind=np.asarray(False),
            coordinate_contract=np.asarray(
                "MAPAXES cell centres; easting/northing/depth metres; depth positive downward"
            ),
            residual_contract=np.asarray("prediction - truth"),
        )
        paths[f"{mode}_npz"] = model_npz
    return paths


def _structured_grid(
    easting: np.ndarray,
    northing: np.ndarray,
    depth: np.ndarray,
    fields: dict[str, np.ndarray],
) -> Any:
    if pv is None:
        raise RuntimeError("PyVista is unavailable")
    if not (easting.shape == northing.shape == depth.shape):
        raise ValueError("coordinate arrays must share a shape")
    grid = pv.StructuredGrid(easting, northing, -depth)
    for name, values in fields.items():
        if values.shape != easting.shape:
            raise ValueError(f"{name} shape {values.shape} != coordinate shape {easting.shape}")
        grid.point_data[name] = np.asarray(values).ravel(order="F")
    return grid


def _save_vtk_assets(
    volume: EclipsePropertyVolume,
    regional: dict[str, RegionalModelVolume],
    output_root: Path,
) -> dict[str, Path]:
    reference = _structured_grid(
        volume.easting_m,
        volume.northing_m,
        volume.depth_m,
        {
            "reference_porosity": volume.porosity,
            "active": volume.active.astype(np.uint8),
            "eclipse_depth_m": volume.depth_m,
        },
    )
    reference_path = output_root / "reference" / "reference_property_volume.vts"
    reference.save(reference_path, binary=True)
    paths = {"reference_vts": reference_path}
    for mode, item in regional.items():
        grid = _structured_grid(
            item.easting_m,
            item.northing_m,
            item.depth_m,
            {
                "truth": item.truth,
                "prediction": item.prediction,
                "residual_prediction_minus_truth": item.residual,
                "active": item.active.astype(np.uint8),
                "eclipse_depth_m": item.depth_m,
            },
        )
        path = output_root / mode / "heldout_reconstruction_volume.vts"
        grid.save(path, binary=True)
        paths[f"{mode}_vts"] = path
    return paths


def _scalar_limits(values: np.ndarray, *, symmetric: bool = False) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite values for scalar limits")
    if symmetric:
        bound = float(np.max(np.abs(finite)))
        return -max(bound, 1e-12), max(bound, 1e-12)
    return float(finite.min()), float(finite.max())


def _plotter_grid(
    plotter: Any,
    *,
    xlabel: str = "Easting (m)",
    ylabel: str = "Northing (m)",
    zlabel: str = "Elevation proxy, -depth (m)",
) -> None:
    plotter.show_grid(
        xtitle=xlabel,
        ytitle=ylabel,
        ztitle=zlabel,
        font_family="times",
        font_size=8,
        n_xlabels=4,
        n_ylabels=4,
        n_zlabels=4,
        color="#333333",
        grid="back",
        location="outer",
    )


def _volume_actor(
    plotter: Any,
    grid: Any,
    field: str,
    *,
    cmap: Any,
    clim: tuple[float, float],
    scalar_title: str,
) -> None:
    plotter.add_volume(
        grid,
        scalars=field,
        cmap=cmap,
        clim=clim,
        opacity="sigmoid",
        shade=True,
        ambient=0.25,
        diffuse=0.75,
        specular=0.12,
        scalar_bar_args={
            "title": scalar_title,
            "title_font_size": 11,
            "label_font_size": 9,
            "font_family": "times",
            "vertical": False,
            "position_x": 0.14,
            "position_y": 0.02,
            "width": 0.72,
            "height": 0.09,
            "fmt": "%.3f",
            "color": "#222222",
        },
    )


def _save_pdf_wrapper(png: Path, pdf: Path, figsize: tuple[float, float]) -> None:
    image = plt.imread(png)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_axis_off()
    _normalize_fonts(fig)
    fig.savefig(pdf, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _render_reference_static(
    volume: EclipsePropertyVolume,
    output_root: Path,
) -> dict[str, Path]:
    output_dir = output_root / "reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = _structured_grid(
        volume.easting_m,
        volume.northing_m,
        volume.depth_m,
        {"reference_porosity": volume.porosity},
    )
    clim = _scalar_limits(volume.porosity)
    png = output_dir / "reference_volume.png"
    pdf = output_dir / "reference_volume.pdf"
    plotter = pv.Plotter(off_screen=True, window_size=(2160, 1500))
    plotter.set_background("white")
    _volume_actor(
        plotter,
        grid,
        "reference_porosity",
        cmap=PORO_CMAP,
        clim=clim,
        scalar_title="Reference porosity (fraction)",
    )
    plotter.add_text("a", position=(20, 20), font="times", font_size=14, color="#111111")
    _plotter_grid(plotter)
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.15)
    plotter.show(screenshot=png, auto_close=True)
    _save_pdf_wrapper(png, pdf, (7.2, 5.0))

    slice_png = output_dir / "reference_orthogonal_slices.png"
    slice_pdf = output_dir / "reference_orthogonal_slices.pdf"
    active = volume.active
    origin = (
        float(np.median(volume.easting_m[active])),
        float(np.median(volume.northing_m[active])),
        float(-np.median(volume.depth_m[active])),
    )
    slices = grid.slice_orthogonal(x=origin[0], y=origin[1], z=origin[2])
    plotter = pv.Plotter(off_screen=True, window_size=(2160, 1500))
    plotter.set_background("white")
    plotter.add_mesh(
        slices,
        scalars="reference_porosity",
        cmap=PORO_CMAP,
        clim=clim,
        nan_opacity=0.0,
        show_edges=False,
        scalar_bar_args={
            "title": "Reference porosity (fraction)",
            "title_font_size": 11,
            "label_font_size": 9,
            "font_family": "times",
            "vertical": False,
            "position_x": 0.14,
            "position_y": 0.02,
            "width": 0.72,
            "height": 0.09,
            "fmt": "%.3f",
            "color": "#222222",
        },
    )
    plotter.add_mesh(grid.outline(), color="#666666", line_width=1.0)
    plotter.add_text("b", position=(20, 20), font="times", font_size=14, color="#111111")
    _plotter_grid(plotter)
    plotter.view_isometric()
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.15)
    plotter.show(screenshot=slice_png, auto_close=True)
    _save_pdf_wrapper(slice_png, slice_pdf, (7.2, 5.0))
    return {
        "reference_png": png,
        "reference_pdf": pdf,
        "reference_slices_png": slice_png,
        "reference_slices_pdf": slice_pdf,
    }


def _render_regional_static(
    item: RegionalModelVolume,
    output_root: Path,
) -> dict[str, Path]:
    output_dir = output_root / item.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = _structured_grid(
        item.easting_m,
        item.northing_m,
        item.depth_m,
        {"truth": item.truth, "prediction": item.prediction, "residual": item.residual},
    )
    poro_clim = (
        min(_scalar_limits(item.truth)[0], _scalar_limits(item.prediction)[0]),
        max(_scalar_limits(item.truth)[1], _scalar_limits(item.prediction)[1]),
    )
    residual_clim = _scalar_limits(item.residual, symmetric=True)
    png = output_dir / "heldout_volume_comparison.png"
    pdf = output_dir / "heldout_volume_comparison.pdf"
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(2160, 780), border=False)
    plotter.set_background("white")
    panels = [
        ("truth", PORO_CMAP, poro_clim, "Truth porosity (fraction)"),
        ("prediction", PORO_CMAP, poro_clim, "Predicted porosity (fraction)"),
        ("residual", RESIDUAL_CMAP, residual_clim, "Prediction - truth"),
    ]
    for index, (field, cmap, clim, scalar_title) in enumerate(panels):
        plotter.subplot(0, index)
        _volume_actor(
            plotter,
            grid,
            field,
            cmap=cmap,
            clim=clim,
            scalar_title=scalar_title,
        )
        plotter.add_text(
            chr(ord("a") + index),
            position=(12, 12),
            font="times",
            font_size=13,
            color="#111111",
        )
        _plotter_grid(plotter)
        plotter.view_isometric()
        plotter.enable_parallel_projection()
        plotter.camera.zoom(1.08)
    plotter.link_views()
    plotter.show(screenshot=png, auto_close=True)
    _save_pdf_wrapper(png, pdf, (7.2, 2.6))
    return {f"{item.mode}_png": png, f"{item.mode}_pdf": pdf}


def _volume_points(
    easting: np.ndarray,
    northing: np.ndarray,
    depth: np.ndarray,
    values: np.ndarray,
    *,
    stride_kji: tuple[int, int, int],
    fill_missing_with: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stride = tuple(slice(None, None, max(1, int(value))) for value in stride_kji)
    x = easting[stride]
    y = northing[stride]
    z = depth[stride]
    v = values[stride]
    valid_coordinates = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    filled = np.where(np.isfinite(v), v, np.float32(fill_missing_with))
    return (
        x[valid_coordinates],
        y[valid_coordinates],
        z[valid_coordinates],
        filled[valid_coordinates],
    )


def _rectilinear_display_coordinates(
    easting: np.ndarray,
    northing: np.ndarray,
    *,
    global_k_start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regularize a corner-point centre lattice for Plotly's rectilinear mapper.

    Plotly Volume does not support curvilinear/corner-point physical geometry.
    Along-grid I/J distances retain measured median horizontal spacing, while
    K remains the global stratigraphic layer index. Exact MAPAXES centres remain
    in the NPZ/VTS assets and in the VTK static render.
    """
    shape = easting.shape
    if northing.shape != shape or len(shape) != 3:
        raise ValueError("rectilinear display inputs must share a K/J/I shape")
    i_steps = np.nanmedian(
        np.hypot(np.diff(easting, axis=2), np.diff(northing, axis=2)),
        axis=(0, 1),
    )
    j_steps = np.nanmedian(
        np.hypot(np.diff(easting, axis=1), np.diff(northing, axis=1)),
        axis=(0, 2),
    )
    if not (np.isfinite(i_steps).all() and np.isfinite(j_steps).all()):
        raise ValueError("non-finite along-grid spacing")
    i_distance = np.concatenate([[0.0], np.cumsum(i_steps, dtype=np.float64)])
    j_distance = np.concatenate([[0.0], np.cumsum(j_steps, dtype=np.float64)])
    k_index = np.arange(global_k_start, global_k_start + shape[0], dtype=np.float64)
    x = np.broadcast_to(i_distance[None, None, :], shape)
    y = np.broadcast_to(j_distance[None, :, None], shape)
    z = np.broadcast_to(k_index[:, None, None], shape)
    return x, y, z


def _plotly_layout(fig: Any, caption_html: str) -> str:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Times New Roman, Times, serif", "size": 13, "color": "#111111"},
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        showlegend=False,
    )
    body = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<link rel='icon' href='data:,'>"
        "<style>body{margin:0;padding:14px;background:#fff;color:#111;"
        "font-family:Times New Roman,Times,serif}.note{max-width:1180px;margin:0 auto 10px;"
        "font-size:14px;line-height:1.42}.plotly-graph-div{min-height:760px}</style>"
        "</head><body><div class='note'>"
        + caption_html
        + "</div>"
        + body
        + "</body></html>"
    )


def _render_reference_interactive(
    volume: EclipsePropertyVolume,
    output_root: Path,
) -> Path:
    if go is None:
        raise RuntimeError("Plotly is unavailable")
    vmin, vmax = _scalar_limits(volume.porosity)
    fill = vmin - max(0.01 * (vmax - vmin), 1e-6)
    display_x, display_y, display_z = _rectilinear_display_coordinates(
        volume.easting_m,
        volume.northing_m,
        global_k_start=0,
    )
    x, y, z, values = _volume_points(
        display_x,
        display_y,
        display_z,
        volume.porosity,
        stride_kji=(1, 2, 2),
        fill_missing_with=fill,
    )
    finite_values = volume.porosity[np.isfinite(volume.porosity)]
    q = np.quantile(finite_values, [0.25, 0.75])
    midpoint = {
        "x": float(np.median(x)),
        "y": float(np.median(y)),
        "z": float(np.median(z)),
    }
    base = dict(
        x=x,
        y=y,
        z=z,
        value=values,
        colorscale=PLOTLY_PORO,
        isomin=vmin,
        isomax=vmax,
        cmin=vmin,
        cmax=vmax,
        caps={"x_show": False, "y_show": False, "z_show": False},
        colorbar={"title": "Porosity<br>(fraction)", "len": 0.65},
        hovertemplate=(
            "Along-I distance %{x:.1f} m<br>Along-J distance %{y:.1f} m<br>"
            "K layer %{z:.0f}<br>Porosity %{value:.4f}<extra></extra>"
        ),
    )
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            **base,
            opacity=0.12,
            surface_count=18,
            name="volume",
            visible=True,
        )
    )
    iso_base = dict(base)
    iso_base["isomin"] = float(q[0])
    iso_base["isomax"] = float(q[1])
    fig.add_trace(
        go.Isosurface(
            **iso_base,
            surface_count=6,
            opacity=0.42,
            name="isosurfaces",
            visible=False,
        )
    )
    fig.add_trace(
        go.Volume(
            **base,
            opacity=0.05,
            surface_count=10,
            slices={
                "x": {"show": True, "locations": [midpoint["x"]]},
                "y": {"show": True, "locations": [midpoint["y"]]},
                "z": {"show": True, "locations": [midpoint["z"]]},
            },
            name="orthogonal slices",
            visible=False,
        )
    )
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.02,
                "buttons": [
                    {"label": "体渲染", "method": "update", "args": [{"visible": [True, False, False]}]},
                    {"label": "等值面", "method": "update", "args": [{"visible": [False, True, False]}]},
                    {"label": "正交切片", "method": "update", "args": [{"visible": [False, False, True]}]},
                ],
            }
        ],
        scene={
            "xaxis_title": "Along-I distance (m)",
            "yaxis_title": "Along-J distance (m)",
            "zaxis_title": "Global K layer index",
            "zaxis": {"autorange": "reversed"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.9}},
        },
    )
    caption = (
        "<b>全场参考属性体。</b> Eclipse MAPAXES 物理坐标；PHIF_NW/PORO 为参考属性，"
        "不是模型预测。浏览器体渲染使用规则化的沿 I/J 物理距离与 K 层号；"
        "精确 MAPAXES Easting/Northing/depth 保存在 NPZ/VTS，并用于静态 VTK 图。"
        "规则化仅用于显示，定量分析请使用 NPZ/VTS 精确坐标。浏览器为性能采用 "
        "K/J/I 步长 1/2/2，完整分辨率见 VTS。Eclipse 深度基准未在 GRID 中独立标定，"
        "因此不冒充 TVDSS。按钮可切换连续体渲染、等值面和三向正交切片；K 轴向下为正。"
    )
    html = output_root / "reference" / "reference_volume.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    html.write_text(_plotly_layout(fig, caption), encoding="utf-8")
    return html


def _render_regional_interactive(
    item: RegionalModelVolume,
    output_root: Path,
) -> Path:
    if go is None or make_subplots is None:
        raise RuntimeError("Plotly is unavailable")
    poro_clim = (
        min(_scalar_limits(item.truth)[0], _scalar_limits(item.prediction)[0]),
        max(_scalar_limits(item.truth)[1], _scalar_limits(item.prediction)[1]),
    )
    residual_clim = _scalar_limits(item.residual, symmetric=True)
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.025,
    )
    panels = [
        ("truth", item.truth, PLOTLY_PORO, poro_clim, "Truth porosity"),
        ("prediction", item.prediction, PLOTLY_PORO, poro_clim, "Predicted porosity"),
        ("residual", item.residual, PLOTLY_RESIDUAL, residual_clim, "Prediction - truth"),
    ]
    for column, (_, values, cmap, clim, colorbar_title) in enumerate(panels, start=1):
        fill = clim[0] - max(0.01 * (clim[1] - clim[0]), 1e-6)
        display_x, display_y, display_z = _rectilinear_display_coordinates(
            item.easting_m,
            item.northing_m,
            global_k_start=int(item.source_slice_kji[0].start),
        )
        x, y, z, v = _volume_points(
            display_x,
            display_y,
            display_z,
            values,
            stride_kji=(1, 1, 1),
            fill_missing_with=fill,
        )
        trace = go.Volume(
            x=x,
            y=y,
            z=z,
            value=v,
            colorscale=cmap,
            isomin=clim[0],
            isomax=clim[1],
            cmin=clim[0],
            cmax=clim[1],
            opacity=0.13,
            surface_count=16,
            caps={"x_show": False, "y_show": False, "z_show": False},
            colorbar={
                "title": colorbar_title,
                "len": 0.48,
                "x": 0.30 if column == 1 else (0.64 if column == 2 else 0.99),
            },
            hovertemplate=(
                "Along-I distance %{x:.1f} m<br>Along-J distance %{y:.1f} m<br>"
                "K layer %{z:.0f}<br>Value %{value:.4f}<extra></extra>"
            ),
            name=colorbar_title,
            showlegend=False,
        )
        fig.add_trace(trace, row=1, col=column)
    scene = {
        "xaxis_title": "Along-I distance (m)",
        "yaxis_title": "Along-J distance (m)",
        "zaxis_title": "Global K layer index",
        "zaxis": {"autorange": "reversed"},
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.9}},
    }
    fig.update_layout(scene=scene, scene2=scene, scene3=scene)
    boundary = (
        "严格空间留出；未使用测试区井约束"
        if item.mode == "strict"
        else "条件重建；提供测试区约束，指标排除精确井格"
    )
    skill = _model_skill_interpretation(item.mode, item)
    caption = (
        f"<b>{item.mode} 留出区重建。</b> {boundary}。三幅依次为真实值、预测值、"
        "预测−真实残差；这是 previously-seen reusable holdout，不是新的盲测，"
        "也不是全场预测体。浏览器体渲染使用规则化的沿 I/J 物理距离与全局 K 层号；"
        "精确 MAPAXES 坐标见同目录 NPZ/VTS；规则化仅用于显示，定量分析请用 NPZ/VTS。"
        f"<br><b>模型技能警告：</b>RMSE={skill['rmse']:.6f}，MAE={skill['mae']:.6f}，"
        f"R²={skill['r2']:.6f}；{skill['reader_warning_zh']}拖动旋转，滚轮缩放。"
    )
    html = output_root / item.mode / "heldout_volume_comparison.html"
    html.write_text(_plotly_layout(fig, caption), encoding="utf-8")
    return html


def _caption_text(mode: str, item: RegionalModelVolume) -> str:
    metrics = item.archive["metrics"]["metrics"]
    prefix = "strict" if mode == "strict" else "conditional"
    boundary = (
        "Strict spatial holdout; no test-region well constraints were used."
        if mode == "strict"
        else "Conditional reconstruction; test-region constraints were supplied and exact well cells were excluded."
    )
    skill = _model_skill_interpretation(mode, item)
    return (
        f"Physical-coordinate 3-D porosity reconstruction for {mode}.\n"
        "Axes are Eclipse MAPAXES cell-centre Easting/Northing and Eclipse grid depth in metres.\n"
        f"{boundary}\n"
        "Panels: truth, prediction, and prediction-minus-truth residual.\n"
        f"Archived holdout metrics: RMSE={metrics[prefix + '_rmse']:.6f}, "
        f"MAE={metrics[prefix + '_mae']:.6f}, R²={metrics[prefix + '_r2']:.6f}.\n"
        f"Model-skill warning: {skill['reader_warning_en']}\n"
        "Evidence class: previously_seen_reusable_holdout; prior_test_consumed=true; fresh_blind=false.\n"
        "No ensemble or posterior samples are archived, so no uncertainty body is claimed."
    )


def _model_skill_interpretation(mode: str, item: RegionalModelVolume) -> dict[str, Any]:
    metrics = item.archive["metrics"]["metrics"]
    prefix = "strict" if mode == "strict" else "conditional"
    rmse = float(metrics[prefix + "_rmse"])
    mae = float(metrics[prefix + "_mae"])
    r2 = float(metrics[prefix + "_r2"])
    prediction_std = float(np.nanstd(item.prediction))
    truth_std = float(np.nanstd(item.truth))
    variability_ratio = prediction_std / truth_std if truth_std > 0 else math.nan
    if r2 < 0:
        interpretation_en = (
            "R²<0 means the prediction is worse than a constant-mean baseline; "
            "this regional reconstruction has no effective predictive skill."
        )
        interpretation_zh = (
            "R²<0，说明预测还不如常数均值基线；当前区域重建没有有效预测技能。"
        )
    else:
        interpretation_en = (
            "R² is near zero and the prediction retains only "
            f"{variability_ratio:.1%} of the truth standard deviation; "
            "spatial heterogeneity is essentially not reconstructed."
        )
        interpretation_zh = (
            f"R²接近 0，预测标准差仅为真实值的 {variability_ratio:.1%}；"
            "空间非均质性基本没有被重建。"
        )
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "prediction_std": prediction_std,
        "truth_std": truth_std,
        "prediction_to_truth_std_ratio": variability_ratio,
        "reader_warning_en": (
            f"prediction std={prediction_std:.6f} versus truth std={truth_std:.6f}; "
            "the prediction is nearly constant. "
            + interpretation_en
        ),
        "reader_warning_zh": (
            f"预测标准差={prediction_std:.6f}，真实值标准差={truth_std:.6f}，"
            "预测近乎常数。"
            + interpretation_zh
        ),
        "practical_skill_verdict": "no_practically_useful_spatial_reconstruction_skill",
    }


def _write_metadata(
    eclipse_zip: Path,
    volume: EclipsePropertyVolume,
    regional: dict[str, RegionalModelVolume],
    assets: dict[str, Path],
    output_root: Path,
) -> Path:
    reference_caption = (
        "Full-field Volve Eclipse reference porosity property volume.\n"
        "The geometry comes from GRID DIMENS/MAPAXES/CORNERS-derived cell centres; "
        "PORO comes from INIT and exactly matches the active-cell ASCII export.\n"
        "This is a reference property body, not a model prediction and not uncertainty.\n"
        "The Eclipse vertical coordinate is positive-down model depth in metres; its datum "
        "is not independently identified in the source GRID, so it is not labelled TVDSS."
    )
    (output_root / "reference" / "caption.md").write_text(reference_caption + "\n", encoding="utf-8")
    for mode, item in regional.items():
        (output_root / mode / "caption.md").write_text(
            _caption_text(mode, item) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "p5-physical-property-volume-v2",
        "track_id": "reconstruction",
        "root_seed": ROOT_SEED,
        "code": {
            "path": _portable_path(HERE / "visualize_property_volume.py"),
            "sha256": _sha256_file(HERE / "visualize_property_volume.py"),
        },
        "eclipse_source": {
            "path": _portable_path(eclipse_zip),
            "sha256": _sha256_file(eclipse_zip),
            "inspection": volume.inspection,
        },
        "coordinate_contract": {
            "horizontal": "Eclipse MAPAXES-projected cell-centre Easting/Northing, metres",
            "vertical": "Eclipse grid depth coordinate, metres, positive downward",
            "static_display_vertical": "-depth so shallower cells plot above deeper cells",
            "interactive_display_vertical": (
                "regularized along-I/along-J physical distances and global K layer index; "
                "exact curvilinear MAPAXES/depth coordinates remain in NPZ/VTS/static VTK"
            ),
            "crs_note": "GRID provides MAPAXES but no EPSG code; no unverified EPSG claim is made.",
        },
        "volume_contract": {
            "reference": "full-field active-cell PHIF_NW/PORO reference volume",
            "model": "regional previously-seen reusable holdout truth/prediction/residual",
            "residual": "prediction - truth",
            "uncertainty_available": False,
            "uncertainty_note": "No ensemble or posterior samples are archived; residual is error, not uncertainty.",
            "forbidden_mislabels": [
                "regional model volume as full-field prediction",
                "reference PORO as model prediction",
                "residual as uncertainty",
                "scatter/point cloud as continuous property volume",
                "previously-seen holdout as fresh blind evidence",
            ],
        },
        "modes": {
            mode: {
                "archive_volume_shape_kji": list(item.archive["volume_shape_kji"]),
                "source_slice_kji_half_open": _slice_bounds(item.source_slice_kji),
                "regional_shape_kji": list(item.truth.shape),
                "n_scored_cells": int(np.isfinite(item.truth).sum()),
                "truth_exactly_matches_native_eclipse_poro_at_archived_indices": True,
                "evidence_class": item.archive["state"]["evidence_class"],
                "prior_test_consumed": True,
                "fresh_blind": False,
                "source_predictions": {
                    "path": _portable_path(item.archive["archive_path"]),
                    "sha256": _sha256_file(item.archive["archive_path"]),
                },
                "prediction_statistics": {
                    "min": float(np.nanmin(item.prediction)),
                    "max": float(np.nanmax(item.prediction)),
                    "mean": float(np.nanmean(item.prediction)),
                    "std": float(np.nanstd(item.prediction)),
                },
                "model_skill_interpretation": _model_skill_interpretation(mode, item),
            }
            for mode, item in regional.items()
        },
        "rendering": {
            "static": "PyVista/VTK volume ray casting and orthogonal slices; no scatter",
            "interactive": "Plotly Volume and Isosurface traces; no scatter traces",
            "reference_interactive_stride_kji": [1, 2, 2],
            "regional_interactive_stride_kji": [1, 1, 1],
            "font_status": FONT_STATUS,
            "no_vertical_exaggeration": True,
        },
        "assets": {
            name: {
                "path": _portable_path(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(assets.items())
            if path.is_file()
        },
    }
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def render_all(
    *,
    eclipse_zip: Path,
    output_root: Path = OUTPUT_ROOT,
    skip_static: bool = False,
    skip_interactive: bool = False,
    skip_vtk: bool = False,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    volume = _load_eclipse_property_volume(eclipse_zip)
    regional = {mode: _regional_model_volume(mode, volume) for mode in MODES}
    assets = _save_npz_assets(volume, regional, output_root)
    if not skip_vtk:
        assets.update(_save_vtk_assets(volume, regional, output_root))
    if not skip_static:
        assets.update(_render_reference_static(volume, output_root))
        for item in regional.values():
            assets.update(_render_regional_static(item, output_root))
    if not skip_interactive:
        assets["reference_html"] = _render_reference_interactive(volume, output_root)
        for mode, item in regional.items():
            assets[f"{mode}_html"] = _render_regional_interactive(item, output_root)
    manifest = _write_metadata(eclipse_zip, volume, regional, assets, output_root)
    return {
        "output_root": _portable_path(output_root),
        "manifest": _portable_path(manifest),
        "shape_kji": list(volume.shape_kji),
        "active_cells": int(volume.active.sum()),
        "assets": {name: _portable_path(path) for name, path in sorted(assets.items())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eclipse-zip", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--skip-interactive", action="store_true")
    parser.add_argument("--skip-vtk", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    eclipse_zip = _discover_eclipse_zip(args.eclipse_zip)
    result = render_all(
        eclipse_zip=eclipse_zip,
        output_root=args.output_root,
        skip_static=args.skip_static,
        skip_interactive=args.skip_interactive,
        skip_vtk=args.skip_vtk,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
