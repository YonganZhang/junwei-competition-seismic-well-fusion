"""Shared publication and provenance utilities for six-track geoscience figures.

The module intentionally separates three evidence modes:

``native_volume``
    A registered 3-D grid or voxel volume is available.
``spatial_context``
    Real spatial coordinates are available, but no dense prediction volume exists.
``section_only``
    Only a depth/time section or an ordered sample sequence is defensible.

These labels prevent a visually attractive figure from silently upgrading a
single well, independent patches, or an interpolation into a measured volume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text


AKUN = {
    "blue": "#3D5A80",
    "cyan": "#4C9F9A",
    "green": "#74A57F",
    "sand": "#E9C46A",
    "orange": "#E07A5F",
    "red": "#B94A48",
    "ink": "#24313A",
    "muted": "#6B7780",
    "grid": "#D8DEE3",
    "paper": "#FFFFFF",
    "warm": "#F4F1EA",
}

SEISMIC_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "junwei_seismic",
    ["#243B53", "#EAF0F3", "#FFFFFF", "#F6E5D4", "#B9503F"],
    N=256,
)
RESIDUAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "junwei_residual",
    ["#3D5A80", "#EAF0F3", "#FFFFFF", "#F4E3D7", "#B94A48"],
    N=256,
)
POROSITY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "junwei_porosity",
    ["#203A5F", "#3D7C8A", "#74A57F", "#E9C46A", "#E07A5F"],
    N=256,
)


def _font_family() -> str:
    names = {item.name for item in font_manager.fontManager.ttflist}
    if "Times New Roman" in names:
        return "Times New Roman"
    if "TeX Gyre Termes" in names:
        return "TeX Gyre Termes"
    return "Liberation Serif"


FONT_FAMILY = _font_family()


def publication_style() -> None:
    """Apply the shared camera-ready style before axes are created."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [
                FONT_FAMILY,
                "Times New Roman",
                "TeX Gyre Termes",
                "Liberation Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.facecolor": AKUN["paper"],
            "axes.facecolor": AKUN["paper"],
            "savefig.facecolor": AKUN["paper"],
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_fonts(fig: plt.Figure) -> None:
    """Normalize all existing text objects after figure construction."""
    for text in fig.findobj(Text):
        text.set_fontfamily(FONT_FAMILY)
    for ax in fig.axes:
        ax.tick_params(width=0.8, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


def panel_label(ax: plt.Axes, label: str, *, x: float = -0.08, y: float = 1.04) -> None:
    kwargs = {
        "transform": ax.transAxes,
        "ha": "left",
        "va": "bottom",
        "fontsize": 11,
        "fontweight": "bold",
        "color": AKUN["ink"],
    }
    if hasattr(ax, "text2D"):
        ax.text2D(x, y, label, **kwargs)
    else:
        ax.text(x, y, label, **kwargs)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


@dataclass(frozen=True)
class EvidenceSource:
    path: Path
    scientific_role: str
    split_scope: str
    shape_or_row_count: str

    def as_record(self, project_root: Path) -> dict[str, Any]:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        return {
            "path": project_relative(self.path, project_root),
            "sha256": sha256_file(self.path),
            "shape_or_row_count": self.shape_or_row_count,
            "scientific_role": self.scientific_role,
            "split_scope": self.split_scope,
        }


def save_figure_bundle(
    fig: plt.Figure,
    stem: Path,
    *,
    dpi: int = 300,
    formats: Iterable[str] = ("png", "pdf"),
) -> dict[str, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    normalize_fonts(fig)
    outputs: dict[str, Path] = {}
    for extension in formats:
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi)
        outputs[extension] = path
    plt.close(fig)
    return outputs


def output_record(
    path: Path,
    project_root: Path,
    *,
    role: str,
    evidence_mode: str,
    caption: str,
) -> dict[str, Any]:
    if evidence_mode not in {"native_volume", "spatial_context", "section_only"}:
        raise ValueError(f"unsupported evidence mode: {evidence_mode}")
    record: dict[str, Any] = {
        "role": role,
        "path": project_relative(path, project_root),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "evidence_mode": evidence_mode,
        "caption": caption,
    }
    if path.suffix.lower() == ".png":
        from PIL import Image

        with Image.open(path) as image:
            record.update(
                {
                    "width_px": image.width,
                    "height_px": image.height,
                    "dpi": [
                        float(value)
                        for value in image.info.get("dpi", (300.0, 300.0))
                    ],
                }
            )
    return record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def plotly_layout(*, z_title: str, reverse_z: bool = True) -> dict[str, Any]:
    zaxis: dict[str, Any] = {
        "title": z_title,
        "showbackground": True,
        "backgroundcolor": "rgba(244,241,234,0.45)",
        "gridcolor": "#D8DEE3",
    }
    if reverse_z:
        zaxis["autorange"] = "reversed"
    scene = {
        "xaxis": {
            "title": "X",
            "showbackground": True,
            "backgroundcolor": "rgba(244,241,234,0.45)",
            "gridcolor": "#D8DEE3",
        },
        "yaxis": {
            "title": "Y",
            "showbackground": True,
            "backgroundcolor": "rgba(244,241,234,0.45)",
            "gridcolor": "#D8DEE3",
        },
        "zaxis": zaxis,
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 0.95}},
    }
    return {
        "template": "plotly_white",
        "paper_bgcolor": "#FFFFFF",
        "plot_bgcolor": "#FFFFFF",
        "font": {
            "family": f"{FONT_FAMILY}, Times New Roman, serif",
            "size": 13,
            "color": AKUN["ink"],
        },
        "margin": {"l": 0, "r": 0, "t": 5, "b": 0},
        "scene": scene,
        "showlegend": True,
        "legend": {"orientation": "h", "y": 0.02, "x": 0.02},
    }


def write_plotly_html(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )
    body = body.replace(
        "<head>",
        '<head><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<link rel="icon" href="data:,"><style>canvas{touch-action:none!important}</style>',
        1,
    )
    path.write_text(body, encoding="utf-8")
