from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text

try:
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover - import availability is environment dependent
    go = None
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_H5 = PROJECT_ROOT / "_data/processed/reservoir/test.h5"
DEFAULT_PREDICTIONS = HERE / "_outputs/test_predictions.csv"
DEFAULT_OUTPUT_DIR = HERE / "_outputs/3d_sci_v1"
AVAILABLE_FONT_NAMES = {font.name for font in font_manager.fontManager.ttflist}
UKIYO_RESIDUAL = LinearSegmentedColormap.from_list(
    "ukiyo_residual",
    ["#264653", "#F7F3EA", "#E76F51"],
    N=256,
)
PRIMARY_SERIF_FONT = "Times New Roman" if "Times New Roman" in AVAILABLE_FONT_NAMES else "DejaVu Serif"

TARGET_SPECS = (
    ("PHIF", "PHIF_gt", "PHIF_pred", "fraction"),
    ("KLOGH", "log1p_KLOGH_gt", "log1p_KLOGH_pred", "log1p(mD)"),
    ("SW", "SW_gt", "SW_pred", "fraction"),
)


@dataclass(frozen=True)
class SampleRecord:
    well_id: str
    family_id: str
    depth_m: float
    inline: float
    crossline: float
    time_ms: float
    gt: dict[str, float]
    pred: dict[str, float]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _decode_json_attr(raw: object) -> dict[str, object]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unsupported attribute payload: {type(raw)!r}")


def load_real_records(h5_path: Path = DEFAULT_H5, predictions_csv: Path = DEFAULT_PREDICTIONS) -> list[SampleRecord]:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not predictions_csv.exists():
        raise FileNotFoundError(predictions_csv)

    rows: list[dict[str, str]] = []
    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    records: list[SampleRecord] = []
    with h5py.File(h5_path, "r") as handle:
        keys = sorted(handle.keys())
        if len(keys) != len(rows):
            raise ValueError(f"row count mismatch: h5={len(keys)} csv={len(rows)}")
        for key, row in zip(keys, rows, strict=True):
            group = handle[key]
            meta = _decode_json_attr(group.attrs["meta"])
            position = _decode_json_attr(group.attrs["position"])
            label = np.asarray(group["label"], dtype=float).reshape(-1)
            if label.shape != (3,):
                raise ValueError(f"unexpected label shape for {key}: {label.shape}")
            well_id = str(meta["well_id"])
            family_id = str(meta["family_id"])
            depth_m = float(meta["depth_m"])
            if row["well_id"] != well_id or row["family_id"] != family_id:
                raise ValueError(f"identifier mismatch for {key}")
            if not np.isclose(float(row["depth_m"]), depth_m, atol=1e-6, rtol=0.0):
                raise ValueError(f"depth mismatch for {key}: csv={row['depth_m']} h5={depth_m}")
            records.append(
                SampleRecord(
                    well_id=well_id,
                    family_id=family_id,
                    depth_m=depth_m,
                    inline=float(position["inline"]),
                    crossline=float(position["crossline"]),
                    time_ms=float(position["time_ms"]),
                    gt={
                        "PHIF": float(label[0]),
                        "KLOGH": float(label[1]),
                        "SW": float(label[2]),
                    },
                    pred={
                        "PHIF": float(row["PHIF_pred"]),
                        "KLOGH": float(row["log1p_KLOGH_pred"]),
                        "SW": float(row["SW_pred"]),
                    },
                )
            )
    return records


def _family_summary(records: Sequence[SampleRecord]) -> dict[str, object]:
    families = sorted({record.family_id for record in records})
    return {
        "family_count": len(families),
        "families": families,
        "well_count": len({record.well_id for record in records}),
        "record_count": len(records),
    }


def build_feasibility(records: Sequence[SampleRecord]) -> dict[str, object]:
    return {
        "mode": "spatial_context",
        "trajectory_used": False,
        "volume_used": False,
        "interpolation_used": False,
        "coordinate_evidence": {
            "fields": ["inline", "crossline", "time_ms"],
            "uses_real_sample_points": True,
            "depth_only_for_hover_and_caption": True,
        },
        "sources": {
            "test_h5": portable_path(DEFAULT_H5),
            "test_predictions_csv": portable_path(DEFAULT_PREDICTIONS),
        },
        "targets": [spec[0] for spec in TARGET_SPECS],
        "summary": _family_summary(records),
    }


def build_provenance(records: Sequence[SampleRecord], output_dir: Path) -> dict[str, object]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "inputs": {
            "test_h5": {
                "path": portable_path(DEFAULT_H5),
                "sha256": sha256_file(DEFAULT_H5),
            },
            "test_predictions_csv": {
                "path": portable_path(DEFAULT_PREDICTIONS),
                "sha256": sha256_file(DEFAULT_PREDICTIONS),
            },
        },
        "records": len(records),
        "fields": {
            "sample_coordinates": ["inline", "crossline", "time_ms"],
            "depth_field": "depth_m",
            "gt_fields": ["PHIF_gt", "log1p_KLOGH_gt", "SW_gt"],
            "pred_fields": ["PHIF_pred", "log1p_KLOGH_pred", "SW_pred"],
        },
        "output_dir": portable_path(output_dir),
    }


def build_caption(records: Sequence[SampleRecord]) -> str:
    family_info = _family_summary(records)
    return "\n".join(
        [
            "Spatial-context sample-point visualization for reservoir property prediction.",
            "PHIF, log1p(KLOGH), and SW are rendered as three separate D_full PNG/PDF figures, one 3D axis and one colorbar per file.",
            "Axes are real Inline, Crossline, and TWT (ms) coordinates from test.h5; depth_m appears only in hover and caption metadata.",
            "No trajectory, interpolation, or volume rendering is used.",
            f"Records: {family_info['record_count']} across {family_info['well_count']} wells and {family_info['family_count']} mother families.",
        ]
    )


def _panel_label(ax, letter: str) -> None:
    ax.text2D(
        0.02,
        0.98,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _slugify_target(target: str) -> str:
    return target.lower().replace("log1p(", "").replace(")", "").replace("/", "_")


def normalize_fonts(fig: matplotlib.figure.Figure) -> None:
    plt.rcParams.update(
        {
            "font.family": PRIMARY_SERIF_FONT,
            "font.serif": [PRIMARY_SERIF_FONT, "Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    for text in fig.findobj(Text):
        text.set_fontfamily(PRIMARY_SERIF_FONT)
        if len(text.get_text()) == 1 and text.get_text().islower():
            text.set_fontsize(max(text.get_fontsize(), 11))
            text.set_fontweight("bold")
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label, getattr(ax, "zaxis", None) and ax.zaxis.label):
            if label is not None:
                label.set_fontfamily(PRIMARY_SERIF_FONT)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily(PRIMARY_SERIF_FONT)
        if hasattr(ax, "get_zticklabels"):
            for tick in ax.get_zticklabels():
                tick.set_fontfamily(PRIMARY_SERIF_FONT)


def _residual_limits(records: Sequence[SampleRecord], target: str) -> float:
    residuals = np.asarray([record.pred[target] - record.gt[target] for record in records], dtype=float)
    if target == "KLOGH":
        residuals = residuals
    limit = float(np.nanpercentile(np.abs(residuals), 98))
    return max(limit, 1e-6)


def build_matplotlib_figure(
    records: Sequence[SampleRecord],
    target: str,
    unit: str,
    panel_letter: str,
) -> matplotlib.figure.Figure:
    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    x = np.asarray([record.inline for record in records], dtype=float)
    y = np.asarray([record.crossline for record in records], dtype=float)
    z = np.asarray([record.time_ms for record in records], dtype=float)
    residual = np.asarray([record.pred[target] - record.gt[target] for record in records], dtype=float)
    limit = _residual_limits(records, target)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    scatter = ax.scatter(
        x,
        y,
        z,
        c=residual,
        cmap=UKIYO_RESIDUAL,
        norm=norm,
        s=9,
        alpha=0.9,
        linewidths=0.0,
    )
    ax.set_xlabel("Inline")
    ax.set_ylabel("Crossline")
    ax.set_zlabel("TWT (ms)")
    ax.view_init(elev=18, azim=-64)
    try:
        ax.set_box_aspect((1, 1, 0.8))
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useOffset=False))
        except Exception:
            pass
    ax.tick_params(labelsize=7)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.04)
    cbar.set_label(f"{target} residual ({unit})")
    _panel_label(ax, panel_letter)
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.05, top=0.98)
    normalize_fonts(fig)
    return fig


def _plotly_trace(records: Sequence[SampleRecord], target: str, unit: str) -> "go.Scatter3d":
    x = np.asarray([record.inline for record in records], dtype=float)
    y = np.asarray([record.crossline for record in records], dtype=float)
    z = np.asarray([record.time_ms for record in records], dtype=float)
    residual = np.asarray([record.pred[target] - record.gt[target] for record in records], dtype=float)
    limit = _residual_limits(records, target)
    customdata = np.column_stack(
        [
            np.asarray([record.well_id for record in records]),
            np.asarray([record.family_id for record in records]),
            np.asarray([record.depth_m for record in records], dtype=float),
            np.asarray([record.gt[target] for record in records], dtype=float),
            np.asarray([record.pred[target] for record in records], dtype=float),
            residual,
        ]
    )
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=4,
            color=residual,
            colorscale="RdBu",
            cmin=-limit,
            cmax=limit,
            colorbar=dict(title=f"{target} residual ({unit})", thickness=12),
            opacity=0.85,
        ),
        hovertemplate=(
            "well_id=%{customdata[0]}<br>"
            "family_id=%{customdata[1]}<br>"
            "depth_m=%{customdata[2]:.1f}<br>"
            "gt=%{customdata[3]:.6f}<br>"
            "pred=%{customdata[4]:.6f}<br>"
            "residual=%{customdata[5]:.6f}<extra></extra>"
        ),
        customdata=customdata,
        name=target,
        showlegend=False,
    )


def build_html(records: Sequence[SampleRecord], output_path: Path) -> None:
    if go is None:  # pragma: no cover - dependency gate
        raise RuntimeError(f"plotly is required to write HTML visualization: {_PLOTLY_IMPORT_ERROR}")

    fragments: list[str] = []
    for idx, (letter, (target, _, _, unit)) in enumerate(zip(("a", "b", "c"), TARGET_SPECS, strict=True)):
        fig = go.Figure(data=[_plotly_trace(records, target, unit)])
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            scene=dict(
                xaxis_title="Inline",
                yaxis_title="Crossline",
                zaxis_title="TWT (ms)",
                aspectmode="cube",
            ),
            annotations=[
                dict(
                    text=f"<b>{letter}</b>",
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=16, family="Times New Roman"),
                )
            ],
        )
        fragments.append(
            fig.to_html(
                full_html=False,
                include_plotlyjs="inline" if idx == 0 else False,
                config={"displaylogo": False, "responsive": True},
            )
        )

    html = "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<link rel=\"icon\" href=\"data:,\"></head>",
            "<body style=\"margin:0;padding:0;background:#fff;font-family:'Times New Roman',serif;\">",
            "<div style=\"display:flex;flex-direction:column;gap:12px;padding:12px;\">",
            *[f"<div style=\"width:100%;\">{fragment}</div>" for fragment in fragments],
            "</div>",
            "</body></html>",
        ]
    )
    output_path.write_text(html, encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_caption(path: Path, caption: str) -> None:
    path.write_text(caption.strip() + "\n", encoding="utf-8")


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    records = load_real_records()
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_png = output_dir / "spatial_context.png"
    legacy_pdf = output_dir / "spatial_context.pdf"
    legacy_png.unlink(missing_ok=True)
    legacy_pdf.unlink(missing_ok=True)

    html_path = output_dir / "spatial_context.html"
    caption_path = output_dir / "caption.md"
    feasibility_path = output_dir / "three_d_feasibility.json"
    provenance_path = output_dir / "provenance.json"

    image_paths: dict[str, Path] = {}
    for panel_letter, (target, _, _, unit) in zip(("a", "b", "c"), TARGET_SPECS):
        slug = _slugify_target(target)
        fig = build_matplotlib_figure(records, target, unit, panel_letter)
        png_path = output_dir / f"{slug}_spatial_context.png"
        pdf_path = output_dir / f"{slug}_spatial_context.pdf"
        fig.savefig(png_path, dpi=300, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        image_paths[f"{target.lower()}_png"] = png_path
        image_paths[f"{target.lower()}_pdf"] = pdf_path

    build_html(records, html_path)
    write_caption(caption_path, build_caption(records))
    write_json(feasibility_path, build_feasibility(records))
    write_json(provenance_path, build_provenance(records, output_dir))

    return {
        **image_paths,
        "html": html_path,
        "caption": caption_path,
        "feasibility": feasibility_path,
        "provenance": provenance_path,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build spatial-context 3D SCI artifacts for reservoir property samples.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Destination directory for generated artifacts.")
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5, help="Real test.h5 path.")
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS, help="Real test_predictions.csv path.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    global DEFAULT_H5, DEFAULT_PREDICTIONS
    DEFAULT_H5 = args.h5_path
    DEFAULT_PREDICTIONS = args.predictions_csv
    generate_artifacts(args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
