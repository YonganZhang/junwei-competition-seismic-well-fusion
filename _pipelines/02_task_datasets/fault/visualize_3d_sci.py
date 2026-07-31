#!/usr/bin/env python3
"""Render real fault-test patches as spatially registered 3-D planes.

This is a spatial-context visualization, not a reconstructed seismic volume.
Every plane comes from one archived real test patch and is placed at its
recorded inline, crossline and two-way-time coordinates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors, font_manager
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover - reported explicitly at render time
    go = None
    make_subplots = None


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RUN_DIR = HERE / "_outputs" / "runs" / "audited_v2"
REPORT_PATH = RUN_DIR / "visualization_report.json"
CHECKPOINT_PATH = RUN_DIR / "checkpoints" / "best.ckpt"
OUTPUT_ROOT = HERE / "_outputs" / "3d_sci_v1"
FIELDS = ("seismic", "truth", "probability")

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "_code"))
from baseline import load_samples  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def resolve_font() -> tuple[str, dict[str, Any]]:
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


def normalize_fonts(fig: plt.Figure, family: str) -> None:
    plt.rcParams.update({"font.family": family, "mathtext.fontset": "stix"})
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily(family)


def _colormaps() -> dict[str, colors.Colormap]:
    return {
        "seismic": colors.LinearSegmentedColormap.from_list(
            "ukiyo_seismic", ["#264653", "#F7F3EA", "#E76F51"], N=256
        ),
        "truth": colors.ListedColormap(["#E8E2D5", "#C44536"]),
        "probability": colors.LinearSegmentedColormap.from_list(
            "ukiyo_probability", ["#264653", "#2A9D8F", "#E9C46A", "#E76F51"], N=256
        ),
    }


def load_context() -> dict[str, Any]:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    selected = [int(value) for value in report["selected_sample_indices"]]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("archived visualization report must identify exactly three unique test patches")

    test = load_samples("test")
    model = joblib.load(CHECKPOINT_PATH)
    probabilities = np.asarray(model.predict_batch(test.patches), dtype=np.float64)
    if probabilities.shape != test.labels.shape:
        raise ValueError("checkpoint prediction shape does not match archived test labels")

    positions = [test.positions[index] for index in selected]
    report_positions = report["selected_positions"]
    for observed, archived in zip(positions, report_positions):
        for key in ("inline", "crossline", "time_ms", "time_index"):
            if not np.isclose(float(observed[key]), float(archived[key]), rtol=0.0, atol=1e-9):
                raise ValueError(f"selected position disagrees with archived report for {key}")

    ratios = [
        float(position["time_ms"]) / int(position["time_index"])
        for position in positions
        if int(position["time_index"]) > 0
    ]
    time_step_ms = float(np.median(ratios))
    if not np.isfinite(time_step_ms) or not 0.0 < time_step_ms < 20.0:
        raise ValueError(f"invalid time sampling inferred from archived coordinates: {time_step_ms}")

    return {
        "selected_indices": selected,
        "positions": positions,
        "patches": np.asarray(test.patches[selected, 0], dtype=np.float64),
        "truth": np.asarray(test.labels[selected], dtype=np.float64),
        "probability": probabilities[selected],
        "time_step_ms": time_step_ms,
        "test_samples": int(len(test.patches)),
        "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
        "report_sha256": sha256_file(REPORT_PATH),
    }


def plane_coordinates(position: dict[str, Any], shape: tuple[int, int], time_step_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = shape
    crosslines = int(position["crossline"]) + np.arange(rows) - rows // 2
    times = float(position["time_ms"]) + (np.arange(columns) - columns // 2) * time_step_ms
    yy, zz = np.meshgrid(crosslines, times, indexing="ij")
    xx = np.full_like(yy, float(position["inline"]), dtype=np.float64)
    return xx, yy.astype(np.float64), zz.astype(np.float64)


def _field_spec(context: dict[str, Any], field: str) -> tuple[np.ndarray, colors.Colormap, colors.Normalize, str]:
    maps = _colormaps()
    if field == "seismic":
        values = context["patches"]
        bound = float(np.percentile(np.abs(values), 99.0))
        return values, maps[field], colors.Normalize(-bound, bound), "seismic amplitude (z-score)"
    if field == "truth":
        return context["truth"], maps[field], colors.Normalize(0.0, 1.0), "fault label"
    if field == "probability":
        return context["probability"], maps[field], colors.Normalize(0.0, 1.0), "fault probability"
    raise ValueError(f"unknown field: {field}")


def render_static(context: dict[str, Any], field: str, output_root: Path, font_family: str) -> tuple[Path, Path]:
    values, cmap, norm, colorbar_label = _field_spec(context, field)
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    for patch, position in zip(values, context["positions"]):
        xx, yy, zz = plane_coordinates(position, patch.shape, context["time_step_ms"])
        facecolors = cmap(norm(patch))
        if field == "truth":
            facecolors[..., 3] = np.where(patch > 0.5, 0.98, 0.16)
        else:
            facecolors[..., 3] = 0.90
        ax.plot_surface(
            xx,
            yy,
            zz,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    ax.set_xlabel("Inline")
    ax.set_ylabel("Crossline")
    ax.set_zlabel("TWT (ms)")
    ax.invert_zaxis()
    ax.view_init(elev=23, azim=-58)
    ax.grid(False)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, shrink=0.64, pad=0.08)
    colorbar.set_label(colorbar_label)
    normalize_fonts(fig, font_family)
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.05, top=0.98)

    output_root.mkdir(parents=True, exist_ok=True)
    png = output_root / f"{field}_spatial_context.png"
    pdf = output_root / f"{field}_spatial_context.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, dpi=300, facecolor="white")
    plt.close(fig)
    return png, pdf


def render_html(context: dict[str, Any], output_root: Path) -> Path:
    if go is None or make_subplots is None:
        raise RuntimeError("plotly is required for the interactive HTML")
    maps = {
        "seismic": [[0.0, "#264653"], [0.5, "#F7F3EA"], [1.0, "#E76F51"]],
        "truth": [[0.0, "#E8E2D5"], [1.0, "#C44536"]],
        "probability": [[0.0, "#264653"], [0.4, "#2A9D8F"], [0.75, "#E9C46A"], [1.0, "#E76F51"]],
    }
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.03,
    )
    for column, field in enumerate(FIELDS, start=1):
        values, _, norm, colorbar_label = _field_spec(context, field)
        for patch_index, (patch, position) in enumerate(zip(values, context["positions"])):
            xx, yy, zz = plane_coordinates(position, patch.shape, context["time_step_ms"])
            fig.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=zz,
                    surfacecolor=patch,
                    cmin=float(norm.vmin),
                    cmax=float(norm.vmax),
                    colorscale=maps[field],
                    opacity=0.96 if field != "truth" else 0.78,
                    showscale=patch_index == 0,
                    colorbar={"title": colorbar_label, "len": 0.55},
                    name=f"{field} IL {int(position['inline'])}",
                    hovertemplate=(
                        "Inline %{x:.0f}<br>Crossline %{y:.0f}<br>TWT %{z:.1f} ms"
                        f"<br>{colorbar_label} %{{surfacecolor:.4g}}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    scene = {
        "xaxis_title": "Inline",
        "yaxis_title": "Crossline",
        "zaxis_title": "TWT (ms)",
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.55, "y": 1.35, "z": 0.9}},
    }
    fig.update_layout(
        template="plotly_white",
        margin={"l": 5, "r": 5, "t": 5, "b": 5},
        paper_bgcolor="white",
        font={"family": "Times New Roman, Liberation Serif, serif", "size": 13, "color": "#111111"},
        showlegend=False,
    )
    fig.update_scenes(scene, selector={"type": "scene"})
    for label, x in zip(("a", "b", "c"), (0.01, 0.344, 0.678)):
        fig.add_annotation(
            text=f"<b>{label}</b>",
            xref="paper",
            yref="paper",
            x=x,
            y=0.99,
            showarrow=False,
            font={"family": "Times New Roman, Liberation Serif, serif", "size": 18, "color": "#111111"},
        )
    output_root.mkdir(parents=True, exist_ok=True)
    html = output_root / "spatial_context.html"
    body = fig.to_html(full_html=True, include_plotlyjs=True, config={"displaylogo": False, "responsive": True})
    body = body.replace("<head>", "<head><link rel=\"icon\" href=\"data:,\"><meta name=\"description\" content=\"spatial context only; no native volume reconstruction\">", 1)
    body = re.sub(r"<title>.*?</title>", "", body, flags=re.IGNORECASE | re.DOTALL)
    html.write_text(body, encoding="utf-8")
    return html


def render_all(output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    context = load_context()
    font_family, font_status = resolve_font()
    outputs: dict[str, str] = {}
    for field in FIELDS:
        png, pdf = render_static(context, field, output_root, font_family)
        outputs[f"{field}_png"] = relative_path(png)
        outputs[f"{field}_pdf"] = relative_path(pdf)
    html = render_html(context, output_root)
    outputs["html"] = relative_path(html)

    feasibility = {
        "schema_version": "fault-3d-sci-v1",
        "track_id": "fault",
        "verdict": "spatial_context",
        "native_volume": False,
        "coordinate_system": {"x": "Inline", "y": "Crossline", "z": "TWT (ms)"},
        "selected_sample_indices": context["selected_indices"],
        "selected_positions": context["positions"],
        "patch_shape_crossline_time": list(context["patches"].shape[1:]),
        "time_step_ms": context["time_step_ms"],
        "limitations": [
            "Three independent real test patches are shown at their archived coordinates.",
            "The planes are not interpolated or presented as a continuous seismic volume.",
        ],
    }
    provenance = {
        "schema_version": "fault-3d-sci-v1",
        "code_head": "db84205",
        "code_path": relative_path(Path(__file__)),
        "code_sha256": sha256_file(Path(__file__)),
        "inputs": {
            "visualization_report": {"path": relative_path(REPORT_PATH), "sha256": context["report_sha256"]},
            "checkpoint": {"path": relative_path(CHECKPOINT_PATH), "sha256": context["checkpoint_sha256"]},
            "test_h5": {
                "path": "_data/processed/fault/test.h5",
                "sha256": json.loads((RUN_DIR / "build_summary.json").read_text())["dataset_sha256"]["test"],
            },
        },
        "sampling": {
            "rule": "archived spatial-quantile selection from the audited_v2 visualization report",
            "selected_sample_indices": context["selected_indices"],
            "deterministic": True,
        },
        "font_status": font_status,
        "outputs": outputs,
    }
    caption = (
        "Three real fault-test patches positioned at their archived Inline, Crossline and TWT coordinates. "
        "Each static figure shows the same registered planes colored by seismic amplitude, ground-truth fault "
        "label or persisted-checkpoint fault probability. This is spatial context only: the independent patches "
        "were neither stacked nor interpolated into a continuous volume."
    )
    (output_root / "three_d_feasibility.json").write_text(
        json.dumps(feasibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "caption.md").write_text(caption + "\n", encoding="utf-8")
    return {"feasibility": feasibility, "provenance": provenance, "outputs": outputs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(render_all(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
