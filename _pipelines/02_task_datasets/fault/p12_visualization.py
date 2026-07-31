#!/usr/bin/env python3
"""Track-local publication renderer for the fault track.

The renderer is intentionally conservative:
- it reuses the archived audited_v2 test prediction evidence;
- it does not change any split, metric, checkpoint or label content;
- it emits a compact real-test panel plus a spatial-context figure only
  because the archived coordinates support it;
- it records all provenance needed to reproduce or invalidate the outputs.
"""
from __future__ import annotations

import argparse
import json
import h5py
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm, colors, font_manager
import numpy as np


TRACK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRACK_DIR.parents[2]
PUBLISHED_OUTPUT_ROOT = TRACK_DIR / "_outputs" / "p12_visualization"
COMMON_DIR = Path(
    subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
).resolve()
SHARED_PROJECT_ROOT = COMMON_DIR.parent
SOURCE_WORKTREE = SHARED_PROJECT_ROOT / ".claude" / "worktrees" / "track-fault"
SOURCE_TRACK_DIR = SOURCE_WORKTREE / "_pipelines" / "02_task_datasets" / "fault"
RUN_DIR = SOURCE_TRACK_DIR / "_outputs" / "runs" / "audited_v2"
SPATIAL_SOURCE_DIR = SOURCE_TRACK_DIR / "_outputs" / "3d_sci_v1"
SOURCE_TEST_H5 = SOURCE_WORKTREE / "_data" / "processed" / "fault" / "test.h5"

AUDITED_VISUALIZATION_REPORT = RUN_DIR / "visualization_report.json"
AUDITED_BASELINE_METRICS = RUN_DIR / "baseline_metrics.json"
AUDITED_BUILD_SUMMARY = RUN_DIR / "build_summary.json"
AUDITED_CHECKPOINT = RUN_DIR / "checkpoints" / "best.ckpt"
AUDITED_PREDICTION_VISUALIZATION = RUN_DIR / "prediction_visualization.png"
AUDITED_SPLIT_MANIFEST = RUN_DIR / "split_manifest.json"
HISTORICAL_SEISMIC_PNG = SPATIAL_SOURCE_DIR / "seismic_spatial_context.png"
HISTORICAL_TRUTH_PNG = SPATIAL_SOURCE_DIR / "truth_spatial_context.png"
HISTORICAL_PROBABILITY_PNG = SPATIAL_SOURCE_DIR / "probability_spatial_context.png"
HISTORICAL_SPATIAL_PROVENANCE = SPATIAL_SOURCE_DIR / "provenance.json"

FIELDS = ("seismic", "truth", "probability")
PALETTE = {
    "ink": "#111111",
    "paper": "#FFFFFF",
    "blue": "#376795",
    "cyan": "#72BCD5",
    "yellow": "#FFD06F",
    "red": "#E76254",
    "support_dark": "#1E466E",
    "support_grid": "#D9E2E8",
    "support_light": "#EEF3F6",
    "truth_negative": "#EEF3F6",
    "truth_positive": "#E76254",
    "tn": "#D9E2E8",
    "tp": "#72BCD5",
    "fp": "#FFD06F",
    "fn": "#E76254",
}
FONT_CANDIDATES = ("Times New Roman", "TeX Gyre Termes")
SVG_HASH_SALT = "fault-p12-visualization-v1"
matplotlib.rcParams["svg.hashsalt"] = SVG_HASH_SALT

sys.path.insert(0, str(SOURCE_WORKTREE / "_code"))
sys.path.insert(0, str(SOURCE_TRACK_DIR))
from audit_utils import sha256_file, verify_historical_artifacts_if_present  # noqa: E402
from baseline import aggregate_physical_voxels, binary_metrics  # noqa: E402


@dataclass(frozen=True)
class SampleSet:
    patches: np.ndarray
    labels: np.ndarray
    positions: list[dict[str, Any]]
    normalization_stats: dict[str, Any] | None


@dataclass(frozen=True)
class PublicationContext:
    report: dict[str, Any]
    metrics: dict[str, Any]
    build_summary: dict[str, Any]
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_model: Any
    test_samples: Any
    probabilities: np.ndarray
    representative_index: int
    spatial_indices: list[int]
    selected_positions: list[dict[str, Any]]
    time_step_ms: float
    font_family: str
    font_status: dict[str, Any]
    source_worktree: str


def project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def source_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(SOURCE_WORKTREE.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def source_manifest_relative(path: Path) -> str:
    return f".claude/worktrees/track-fault/{source_relative(path)}"


shared_relative = project_relative


def git_head() -> str:
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_font() -> tuple[str, dict[str, Any]]:
    for candidate in FONT_CANDIDATES:
        try:
            font_manager.findfont(candidate, fallback_to_default=False)
            font_status = {
                "requested": "Times New Roman",
                "selected": candidate,
                "available": True,
                "fallback_order": [font for font in FONT_CANDIDATES if font != candidate],
            }
            if candidate != "Times New Roman":
                font_status["limitation"] = "Times New Roman is not installed on this host."
            else:
                font_status["limitation"] = None
            return candidate, font_status
        except ValueError:
            continue
    raise RuntimeError(f"none of the candidate serif fonts are available: {FONT_CANDIDATES}")


def configure_render_style(family: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [family, "TeX Gyre Termes"],
            "mathtext.fontset": "stix",
        }
    )


def normalize_fonts(fig: plt.Figure, family: str) -> None:
    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily(family)


def _no_titles(fig: plt.Figure) -> None:
    for axis in fig.axes:
        if axis.get_title():
            raise RuntimeError(f"figure must not contain axis titles: {axis.get_title()!r}")
    if fig._suptitle is not None and fig._suptitle.get_text().strip():  # type: ignore[attr-defined]
        raise RuntimeError(f"figure must not contain a suptitle: {fig._suptitle.get_text()!r}")  # type: ignore[attr-defined]


def _source_metadata_path(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing audited source artifact: {path}")
    return path


def _load_source_samples() -> SampleSet:
    _source_metadata_path(SOURCE_TEST_H5)
    patches: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    positions: list[dict[str, Any]] = []
    normalization_stats: dict[str, Any] | None = None
    expected_patch_shape: tuple[int, ...] | None = None
    expected_label_shape: tuple[int, ...] | None = None
    with h5py.File(SOURCE_TEST_H5, "r") as handle:
        if int(handle.attrs.get("n_samples", 0)) != 96:
            raise ValueError(f"unexpected source sample count in {SOURCE_TEST_H5}: {handle.attrs.get('n_samples')}")
        for key in sorted(handle.keys()):
            group = handle[key]
            patch = np.asarray(group["seismic_patch"][()], dtype=np.float32)
            label = np.asarray(group["label"][()], dtype=np.uint8)
            if patch.ndim != 3 or patch.shape[0] != 1:
                raise ValueError(f"expected seismic patch [1,H,W], received {patch.shape}")
            if patch.shape[1:] != label.shape:
                raise ValueError(f"patch/label shape mismatch in source data: {patch.shape} vs {label.shape}")
            if expected_patch_shape is None:
                expected_patch_shape, expected_label_shape = patch.shape, label.shape
            elif patch.shape != expected_patch_shape or label.shape != expected_label_shape:
                raise ValueError("inconsistent source sample shapes in fault/test")
            meta = json.loads(group.attrs["meta"])
            if meta.get("task") != "fault" or meta.get("split") != "test":
                raise ValueError(f"unexpected source metadata for {key}: {meta}")
            if meta.get("normalization_fit_split") != "train_fit":
                raise ValueError("source samples must preserve train_fit normalization provenance")
            if normalization_stats is None:
                normalization_stats = meta.get("normalization")
            elif meta.get("normalization") != normalization_stats:
                raise ValueError("source test samples contain inconsistent normalization statistics")
            position = json.loads(group.attrs["position"])
            if "time_index" not in position:
                raise ValueError(f"source sample {key} is missing time_index")
            patches.append(patch)
            labels.append(label)
            positions.append(position)
    if not patches:
        raise ValueError(f"no samples found in source dataset: {SOURCE_TEST_H5}")
    return SampleSet(
        patches=np.stack(patches),
        labels=np.stack(labels),
        positions=positions,
        normalization_stats=normalization_stats,
    )


def _source_file_sha256(path: Path) -> str:
    return sha256_file(_source_metadata_path(path))


def _load_reported_context() -> PublicationContext:
    verify_historical_artifacts_if_present()
    for required in (SOURCE_WORKTREE, SOURCE_TRACK_DIR, SOURCE_WORKTREE / "_code"):
        if not required.exists():
            raise FileNotFoundError(f"canonical source worktree is missing required path: {required}")
    report = json.loads(AUDITED_VISUALIZATION_REPORT.read_text(encoding="utf-8"))
    metrics = json.loads(AUDITED_BASELINE_METRICS.read_text(encoding="utf-8"))
    build_summary = json.loads(AUDITED_BUILD_SUMMARY.read_text(encoding="utf-8"))
    checkpoint_sha256 = _source_file_sha256(AUDITED_CHECKPOINT)
    if checkpoint_sha256 != report["checkpoint_sha256"]:
        raise RuntimeError(
            "source checkpoint hash changed: "
            f"expected {report['checkpoint_sha256']}, observed {checkpoint_sha256}"
        )
    source_test_sha256 = _source_file_sha256(SOURCE_TEST_H5)
    if source_test_sha256 != build_summary["dataset_sha256"]["test"]:
        raise RuntimeError(
            "source audited_v2 test.h5 hash changed: "
            f"expected {build_summary['dataset_sha256']['test']}, observed {source_test_sha256}"
        )
    if report["output_sha256"] != _source_file_sha256(AUDITED_PREDICTION_VISUALIZATION):
        raise RuntimeError("archived regression visual evidence hash mismatch")

    selected = [int(value) for value in report["selected_sample_indices"]]
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError("audited visualization report must contain exactly three unique selections")

    test_samples = _load_source_samples()
    checkpoint_model = joblib.load(AUDITED_CHECKPOINT)
    probabilities = np.asarray(checkpoint_model.predict_batch(test_samples.patches), dtype=np.float64)
    if probabilities.shape != test_samples.labels.shape:
        raise ValueError(
            f"checkpoint probabilities shape mismatch: {probabilities.shape} vs {test_samples.labels.shape}"
        )
    physical_labels, physical_probabilities, coverage = aggregate_physical_voxels(test_samples, probabilities)
    threshold = float(metrics["threshold"])
    recomputed = binary_metrics(physical_labels, physical_probabilities >= threshold)
    for key in ("precision", "recall", "f1"):
        if not np.isclose(recomputed[key], metrics["test_metrics"][key], rtol=0.0, atol=1e-15):
            raise AssertionError(
                f"recomputed {key}={recomputed[key]} disagrees with archived audited_v2 metric "
                f"{metrics['test_metrics'][key]}"
            )
    positions = [test_samples.positions[index] for index in selected]
    for observed, archived in zip(positions, report["selected_positions"]):
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

    font_family, font_status = resolve_font()
    font_status["coverage"] = coverage
    font_status["threshold"] = threshold
    return PublicationContext(
        report=report,
        metrics=metrics,
        build_summary=build_summary,
        checkpoint_path=AUDITED_CHECKPOINT,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_model=checkpoint_model,
        test_samples=test_samples,
        probabilities=probabilities,
        representative_index=selected[len(selected) // 2],
        spatial_indices=selected,
        selected_positions=positions,
        time_step_ms=time_step_ms,
        font_family=font_family,
        font_status=font_status,
        source_worktree=".claude/worktrees/track-fault",
    )


def _akum_prob_cmap() -> colors.Colormap:
    return colors.LinearSegmentedColormap.from_list(
        "fault_akun_probability",
        [PALETTE["blue"], PALETTE["cyan"], PALETTE["yellow"], PALETTE["red"]],
        N=256,
    )


def _akum_seismic_cmap() -> colors.Colormap:
    return colors.LinearSegmentedColormap.from_list(
        "fault_akun_seismic",
        [PALETTE["support_dark"], PALETTE["support_light"], PALETTE["blue"], PALETTE["red"]],
        N=256,
    )


def _akum_truth_cmap() -> colors.Colormap:
    return colors.ListedColormap([PALETTE["truth_negative"], PALETTE["truth_positive"]], name="fault_truth")


def _akum_confusion_cmap() -> colors.Colormap:
    return colors.ListedColormap([PALETTE["tn"], PALETTE["tp"], PALETTE["fp"], PALETTE["fn"]], name="fault_confusion")


def _field_spec(context: PublicationContext, field: str) -> tuple[np.ndarray, colors.Colormap, colors.Normalize, str]:
    if field == "seismic":
        values = np.asarray(context.test_samples.patches[context.spatial_indices, 0], dtype=np.float64)
        bound = float(np.percentile(np.abs(values), 99.0))
        return values, _akum_seismic_cmap(), colors.Normalize(-bound, bound), "seismic amplitude (z-score)"
    if field == "truth":
        values = np.asarray(context.test_samples.labels[context.spatial_indices], dtype=np.float64)
        return values, _akum_truth_cmap(), colors.Normalize(0.0, 1.0), "fault label"
    if field == "probability":
        values = np.asarray(context.probabilities[context.spatial_indices], dtype=np.float64)
        return values, _akum_prob_cmap(), colors.Normalize(0.0, 1.0), "fault probability"
    raise ValueError(f"unknown field: {field}")


def _sample_plane_coordinates(position: dict[str, Any], shape: tuple[int, int], time_step_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = shape
    crosslines = int(position["crossline"]) + np.arange(rows) - rows // 2
    times = float(position["time_ms"]) + (np.arange(columns) - columns // 2) * time_step_ms
    yy, zz = np.meshgrid(crosslines, times, indexing="ij")
    xx = np.full_like(yy, float(position["inline"]), dtype=np.float64)
    return xx, yy.astype(np.float64), zz.astype(np.float64)


def _set_panel_label(ax: plt.Axes, label: str) -> None:
    text_kwargs = {
        "va": "top",
        "ha": "left",
        "fontweight": "bold",
        "fontsize": 16,
        "color": PALETTE["ink"],
    }
    if hasattr(ax, "text2D") and getattr(ax, "name", "") == "3d":
        ax.text2D(0.02, 0.98, f"({label})", transform=ax.transAxes, **text_kwargs)
    else:
        ax.text(0.02, 0.98, f"({label})", transform=ax.transAxes, **text_kwargs)


def build_real_test_panel(context: PublicationContext) -> plt.Figure:
    configure_render_style(context.font_family)
    sample_index = context.representative_index
    patch = np.asarray(context.test_samples.patches[sample_index, 0], dtype=np.float64)
    truth = np.asarray(context.test_samples.labels[sample_index], dtype=np.float64)
    probability = np.asarray(context.probabilities[sample_index], dtype=np.float64)
    threshold = float(context.metrics["threshold"])
    prediction = probability >= threshold
    confusion = np.full(truth.shape, np.nan, dtype=np.float64)
    confusion[(truth == 0) & (~prediction)] = 0.0
    confusion[(truth == 1) & (prediction)] = 1.0
    confusion[(truth == 0) & (prediction)] = 2.0
    confusion[(truth == 1) & (~prediction)] = 3.0
    valid = np.isfinite(confusion)

    truth_overlay = np.full(truth.shape, np.nan, dtype=np.float64)
    truth_overlay[truth > 0.5] = 1.0
    truth_overlay[truth <= 0.5] = 0.0

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.55), constrained_layout=True)
    seismic_bound = float(np.percentile(np.abs(patch), 99.0))
    seismic = axes[0].imshow(
        patch,
        cmap=_akum_seismic_cmap(),
        aspect="auto",
        vmin=-seismic_bound,
        vmax=seismic_bound,
        interpolation="nearest",
    )
    truth_image = axes[1].imshow(
        truth_overlay,
        cmap=_akum_truth_cmap(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    probability_image = axes[2].imshow(
        probability,
        cmap=_akum_prob_cmap(),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    confusion_image = axes[3].imshow(
        np.where(valid, confusion, np.nan),
        cmap=_akum_confusion_cmap(),
        aspect="auto",
        vmin=0.0,
        vmax=3.0,
        interpolation="nearest",
    )

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    axes[0].set_xlabel("time")
    axes[1].set_xlabel("time")
    axes[2].set_xlabel("time")
    axes[3].set_xlabel("time")
    axes[0].set_ylabel(f"IL {int(context.selected_positions[1]['inline'])}\nXL {int(context.selected_positions[1]['crossline'])}")

    _set_panel_label(axes[0], "a")
    _set_panel_label(axes[1], "b")
    _set_panel_label(axes[2], "c")
    _set_panel_label(axes[3], "d")

    sample_position = context.test_samples.positions[sample_index]
    tp = int(np.sum((truth == 1) & prediction))
    fp = int(np.sum((truth == 0) & prediction))
    fn = int(np.sum((truth == 1) & (~prediction)))
    tn = int(np.sum((truth == 0) & (~prediction)))
    metrics_text = (
        f"τ={threshold:.4f}\n"
        f"TP {tp}  FP {fp}\n"
        f"FN {fn}  TN {tn}\n"
        f"IL {int(sample_position['inline'])}  XL {int(sample_position['crossline'])}\n"
        f"TWT {float(sample_position['time_ms']):.1f} ms"
    )
    axes[3].text(
        0.03,
        0.03,
        metrics_text,
        transform=axes[3].transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        color=PALETTE["ink"],
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
    )

    cbar0 = fig.colorbar(seismic, ax=axes[0], fraction=0.046, pad=0.02)
    cbar0.set_label("seismic amplitude (z-score)")
    cbar1 = fig.colorbar(truth_image, ax=axes[1], fraction=0.046, pad=0.02)
    cbar1.set_ticks([0.0, 1.0])
    cbar1.set_ticklabels(["0", "1"])
    cbar1.set_label("fault label")
    cbar2 = fig.colorbar(probability_image, ax=axes[2], fraction=0.046, pad=0.02)
    cbar2.set_label("fault probability")
    cbar3 = fig.colorbar(confusion_image, ax=axes[3], fraction=0.046, pad=0.02)
    cbar3.set_ticks([0.0, 1.0, 2.0, 3.0])
    cbar3.set_ticklabels(["TN", "TP", "FP", "FN"])
    cbar3.set_label(f"thresholded at τ={threshold:.4f}")

    normalize_fonts(fig, context.font_family)
    _no_titles(fig)
    return fig


def build_spatial_context_figure(context: PublicationContext) -> plt.Figure:
    configure_render_style(context.font_family)
    fig = plt.figure(figsize=(14.5, 4.6), constrained_layout=True)
    field_specs = [
        _field_spec(context, "seismic"),
        _field_spec(context, "truth"),
        _field_spec(context, "probability"),
    ]
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    for axis, (values, cmap, norm, colorbar_label), label in zip(axes, field_specs, ("a", "b", "c")):
        for patch, position in zip(values, context.selected_positions):
            xx, yy, zz = _sample_plane_coordinates(position, patch.shape, context.time_step_ms)
            facecolors = cmap(norm(patch))
            if colorbar_label == "fault label":
                facecolors[..., 3] = np.where(patch > 0.5, 0.96, 0.15)
            else:
                facecolors[..., 3] = 0.90
            axis.plot_surface(
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
        axis.set_xlabel("Inline")
        axis.set_ylabel("Crossline")
        axis.set_zlabel("TWT (ms)")
        axis.invert_zaxis()
        axis.view_init(elev=23, azim=-58)
        axis.grid(False)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        _set_panel_label(axis, label)
        scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar = fig.colorbar(scalar, ax=axis, shrink=0.63, pad=0.08)
        colorbar.set_label(colorbar_label)
    normalize_fonts(fig, context.font_family)
    _no_titles(fig)
    return fig


def _save_figure(fig: plt.Figure, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any]
    suffix = path.suffix.lower()
    if suffix == ".svg":
        metadata = {
            "Date": None,
            "Creator": "fault-p12-visualization",
            "Description": "fault p12 publication figure",
        }
    elif suffix == ".pdf":
        metadata = {
            "CreationDate": None,
            "ModDate": None,
            "Creator": "fault-p12-visualization",
            "Producer": "matplotlib",
        }
    else:
        metadata = {"Software": "fault-p12-visualization"}
    fig.savefig(path, dpi=300, facecolor=PALETTE["paper"], metadata=metadata)
    return {
        "path": project_relative(path),
        "sha256": sha256_file(path),
        "exists": path.exists(),
    }


def _svg_has_no_title(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "<title" not in text.lower()


def _output_dimensions(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".png":
        raster = mpimg.imread(path)
        return int(raster.shape[1]), int(raster.shape[0])
    return 0, 0


def _output_record(path: Path, dpi: int, vector_companions: list[str]) -> dict[str, Any]:
    width_px, height_px = _output_dimensions(path)
    return {
        "path": project_relative(path),
        "sha256": sha256_file(path),
        "width_px": width_px,
        "height_px": height_px,
        "dpi": dpi,
        "vector_companions": vector_companions,
    }


def build_publication(output_root: Path = PUBLISHED_OUTPUT_ROOT) -> dict[str, Any]:
    context = _load_reported_context()
    real_test_panel = build_real_test_panel(context)
    spatial_context = build_spatial_context_figure(context)

    figures_dir = output_root / "figures"
    real_png = figures_dir / "real_test_panel.png"
    real_pdf = figures_dir / "real_test_panel.pdf"
    real_svg = figures_dir / "real_test_panel.svg"
    spatial_png = figures_dir / "spatial_context.png"
    spatial_pdf = figures_dir / "spatial_context.pdf"
    spatial_svg = figures_dir / "spatial_context.svg"

    outputs = {
        "real_test_panel": {
            "png": _save_figure(real_test_panel, real_png),
            "pdf": _save_figure(real_test_panel, real_pdf),
            "svg": _save_figure(real_test_panel, real_svg),
            "figure_inches": [float(value) for value in real_test_panel.get_size_inches()],
        },
        "spatial_context": {
            "png": _save_figure(spatial_context, spatial_png),
            "pdf": _save_figure(spatial_context, spatial_pdf),
            "svg": _save_figure(spatial_context, spatial_svg),
            "figure_inches": [float(value) for value in spatial_context.get_size_inches()],
        },
    }
    plt.close(real_test_panel)
    plt.close(spatial_context)

    for svg_path in (real_svg, spatial_svg):
        if not _svg_has_no_title(svg_path):
            raise RuntimeError(f"{svg_path} unexpectedly contains a title tag")

    generated_at = datetime.now(timezone.utc).isoformat()
    p12_contract = {
        "schema_version": "scientific-visualization-contract/v1",
        "profile": "p12_tracks_1_3_5",
        "track_id": "fault",
        "source_commit": git_head(),
        "renderer": {
            "path": project_relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "generated_at": generated_at,
        "scientific_caveat": "Spatial-context support exists from archived coordinates, but the figure does not claim native 3-D volume reconstruction.",
        "inputs": [
            {
                "path": source_manifest_relative(SOURCE_TRACK_DIR / "baseline.py"),
                "sha256": _source_file_sha256(SOURCE_TRACK_DIR / "baseline.py"),
                "shape_or_row_count": 1,
                "scientific_role": "canonical audited_v2 source baseline implementation",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(SOURCE_WORKTREE / "_code" / "dataset_io.py"),
                "sha256": _source_file_sha256(SOURCE_WORKTREE / "_code" / "dataset_io.py"),
                "shape_or_row_count": 1,
                "scientific_role": "canonical audited_v2 shared data interface",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(AUDITED_VISUALIZATION_REPORT),
                "sha256": sha256_file(AUDITED_VISUALIZATION_REPORT),
                "shape_or_row_count": {
                    "test_samples": int(context.report["test_samples"]),
                    "selected_sample_indices": len(context.report["selected_sample_indices"]),
                },
                "scientific_role": "archived audited_v2 selection report",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(AUDITED_BASELINE_METRICS),
                "sha256": sha256_file(AUDITED_BASELINE_METRICS),
                "shape_or_row_count": 1,
                "scientific_role": "archived audited_v2 metric provenance",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(AUDITED_BUILD_SUMMARY),
                "sha256": sha256_file(AUDITED_BUILD_SUMMARY),
                "shape_or_row_count": 1,
                "scientific_role": "archived audited_v2 dataset provenance",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(AUDITED_SPLIT_MANIFEST),
                "sha256": sha256_file(AUDITED_SPLIT_MANIFEST),
                "shape_or_row_count": 1,
                "scientific_role": "archived audited_v2 split manifest",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(AUDITED_CHECKPOINT),
                "sha256": context.checkpoint_sha256,
                "shape_or_row_count": {
                    "test_patch_shape": list(context.test_samples.patches.shape[1:]),
                    "selected_patch_shape": list(context.test_samples.patches[context.representative_index].shape),
                },
                "scientific_role": "archived audited_v2 persisted checkpoint",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(SOURCE_TEST_H5),
                "sha256": _source_file_sha256(SOURCE_TEST_H5),
                "shape_or_row_count": int(context.test_samples.patches.shape[0]),
                "scientific_role": "canonical audited_v2 test evidence source",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(AUDITED_PREDICTION_VISUALIZATION),
                "sha256": _source_file_sha256(AUDITED_PREDICTION_VISUALIZATION),
                "shape_or_row_count": {
                    "width_px": int(mpimg.imread(AUDITED_PREDICTION_VISUALIZATION).shape[1]),
                    "height_px": int(mpimg.imread(AUDITED_PREDICTION_VISUALIZATION).shape[0]),
                },
                "scientific_role": "archived audited_v2 regression evidence",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(HISTORICAL_SEISMIC_PNG),
                "sha256": _source_file_sha256(HISTORICAL_SEISMIC_PNG),
                "shape_or_row_count": {
                    "width_px": int(mpimg.imread(HISTORICAL_SEISMIC_PNG).shape[1]),
                    "height_px": int(mpimg.imread(HISTORICAL_SEISMIC_PNG).shape[0]),
                },
                "scientific_role": "archived spatial-context seismic evidence",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(HISTORICAL_TRUTH_PNG),
                "sha256": _source_file_sha256(HISTORICAL_TRUTH_PNG),
                "shape_or_row_count": {
                    "width_px": int(mpimg.imread(HISTORICAL_TRUTH_PNG).shape[1]),
                    "height_px": int(mpimg.imread(HISTORICAL_TRUTH_PNG).shape[0]),
                },
                "scientific_role": "archived spatial-context truth evidence",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(HISTORICAL_PROBABILITY_PNG),
                "sha256": _source_file_sha256(HISTORICAL_PROBABILITY_PNG),
                "shape_or_row_count": {
                    "width_px": int(mpimg.imread(HISTORICAL_PROBABILITY_PNG).shape[1]),
                    "height_px": int(mpimg.imread(HISTORICAL_PROBABILITY_PNG).shape[0]),
                },
                "scientific_role": "archived spatial-context probability evidence",
                "split_scope": "audited_v2 test split only",
            },
            {
                "path": source_manifest_relative(HISTORICAL_SPATIAL_PROVENANCE),
                "sha256": _source_file_sha256(HISTORICAL_SPATIAL_PROVENANCE),
                "shape_or_row_count": 1,
                "scientific_role": "archived spatial-context provenance",
                "split_scope": "audited_v2 test split only",
            },
        ],
        "outputs": [
            {
                "role": "real_test_qualitative",
                "path": project_relative(real_png),
                "sha256": outputs["real_test_panel"]["png"]["sha256"],
                "width_px": _output_dimensions(real_png)[0],
                "height_px": _output_dimensions(real_png)[1],
                "dpi": 300,
                "vector_companions": ["svg", "pdf"],
            },
            {
                "role": "spatial_context_3d",
                "path": project_relative(spatial_png),
                "sha256": outputs["spatial_context"]["png"]["sha256"],
                "width_px": _output_dimensions(spatial_png)[0],
                "height_px": _output_dimensions(spatial_png)[1],
                "dpi": 300,
                "vector_companions": ["svg", "pdf"],
            },
        ],
        "manual_review": {
            "reviewed": False,
            "reviewed_sha256": None,
            "reviewer": None,
            "no_clipping": None,
            "no_overlap": None,
            "labels_legible": None,
            "colors_consistent": None,
            "scientific_boundary_preserved": None,
        },
    }
    manifest = {
        "schema_version": "fault-p12-publication-v1",
        "track_id": "fault",
        "source_commit": git_head(),
        "source_worktree": ".claude/worktrees/track-fault",
        "provenance": {
            "source_commit": git_head(),
            "renderer": project_relative(Path(__file__)),
            "renderer_sha256": sha256_file(Path(__file__)),
            "generated_at": generated_at,
            "scientific_caveat": "Spatial-context support exists from archived coordinates, but the figure does not claim native 3-D volume reconstruction.",
        },
        "renderer": {
            "path": project_relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "inputs": {
            "audited_visualization_report": {
                "path": source_manifest_relative(AUDITED_VISUALIZATION_REPORT),
                "sha256": sha256_file(AUDITED_VISUALIZATION_REPORT),
                "shape_or_row_count": {
                    "test_samples": int(context.report["test_samples"]),
                    "selected_sample_indices": len(context.report["selected_sample_indices"]),
                },
                "scientific_role": "archived audited_v2 selection report",
                "split_scope": "audited_v2 test split only",
            },
            "audited_baseline_metrics": {
                "path": source_manifest_relative(AUDITED_BASELINE_METRICS),
                "sha256": sha256_file(AUDITED_BASELINE_METRICS),
                "shape_or_row_count": 1,
                "scientific_role": "archived audited_v2 metric provenance",
                "split_scope": "audited_v2 test split only",
            },
            "audited_build_summary": {
                "path": source_manifest_relative(AUDITED_BUILD_SUMMARY),
                "sha256": sha256_file(AUDITED_BUILD_SUMMARY),
                "shape_or_row_count": 1,
                "scientific_role": "archived audited_v2 dataset provenance",
                "split_scope": "audited_v2 test split only",
            },
            "audited_checkpoint": {
                "path": source_manifest_relative(AUDITED_CHECKPOINT),
                "sha256": context.checkpoint_sha256,
                "shape_or_row_count": {
                    "test_patch_shape": list(context.test_samples.patches.shape[1:]),
                    "selected_patch_shape": list(context.test_samples.patches[context.representative_index].shape),
                },
                "scientific_role": "archived audited_v2 persisted checkpoint",
                "split_scope": "audited_v2 test split only",
            },
            "lineage_visualization_png": {
                "path": source_manifest_relative(AUDITED_PREDICTION_VISUALIZATION),
                "sha256": sha256_file(AUDITED_PREDICTION_VISUALIZATION),
                "shape_or_row_count": {
                    "width_px": int(mpimg.imread(AUDITED_PREDICTION_VISUALIZATION).shape[1]),
                    "height_px": int(mpimg.imread(AUDITED_PREDICTION_VISUALIZATION).shape[0]),
                },
                "scientific_role": "archived audited_v2 regression evidence",
                "split_scope": "audited_v2 test split only",
            },
        },
        "selection": {
            "representative_sample_index": context.representative_index,
            "spatial_sample_indices": context.spatial_indices,
            "selected_positions": context.selected_positions,
            "selection_rule": "median of archived spatial-quantile selections",
        },
        "visual_qa_metadata": {
            "font_status": context.font_status,
            "palette": "Akun",
            "probability_scale": [0.0, 1.0],
            "vector_outputs": ["svg", "pdf"],
            "png_dpi": 300,
            "no_titles": True,
            "panel_counts": {"real_test_panel": 4, "spatial_context": 3},
            "manual_review": {
                "reviewed": False,
                "reviewed_sha256": None,
                "reviewer": None,
                "no_clipping": None,
                "no_overlap": None,
                "labels_legible": None,
                "colors_consistent": None,
                "scientific_boundary_preserved": None,
            },
        },
        "split_scope": {
            "scope": "audited_v2 test split only",
            "notes": [
                "No source data, predictions, metrics, or split boundaries are changed.",
                "No holdout, frozen test or new split is consumed.",
                "The spatial figure is contextual, not a reconstructed continuous volume.",
            ],
        },
        "caveat": "Spatial-context support exists from archived coordinates, but the figure does not claim native 3-D volume reconstruction.",
        "p12_contract": p12_contract,
        "figures": [
            {
                "role": "real_test_qualitative",
                "scientific_role": "compact real-test panel with seismic, ground truth, probability and threshold diagnostics",
                "split_scope": "audited_v2 test split only",
                "manual_review": {
                    "reviewed": False,
                    "reviewed_sha256": None,
                    "reviewer": None,
                    "no_clipping": None,
                    "no_overlap": None,
                    "labels_legible": None,
                    "colors_consistent": None,
                    "scientific_boundary_preserved": None,
                },
                "outputs": [
                    {
                        "role": "real_test_qualitative",
                        "path": outputs["real_test_panel"]["png"]["path"],
                        "sha256": outputs["real_test_panel"]["png"]["sha256"],
                        "width_px": _output_dimensions(real_png)[0],
                        "height_px": _output_dimensions(real_png)[1],
                        "dpi": 300,
                        "vector_companions": ["svg", "pdf"],
                    },
                    {
                        "role": "real_test_qualitative",
                        "path": outputs["real_test_panel"]["svg"]["path"],
                        "sha256": outputs["real_test_panel"]["svg"]["sha256"],
                        "width_px": _output_dimensions(real_png)[0],
                        "height_px": _output_dimensions(real_png)[1],
                        "dpi": 300,
                        "vector_companions": ["png", "pdf"],
                    },
                    {
                        "role": "real_test_qualitative",
                        "path": outputs["real_test_panel"]["pdf"]["path"],
                        "sha256": outputs["real_test_panel"]["pdf"]["sha256"],
                        "width_px": _output_dimensions(real_png)[0],
                        "height_px": _output_dimensions(real_png)[1],
                        "dpi": 300,
                        "vector_companions": ["png", "svg"],
                    },
                ],
            },
            {
                "role": "spatial_context_3d",
                "scientific_role": "archived-coordinate spatial context figure",
                "split_scope": "audited_v2 test split only",
                "manual_review": {
                    "reviewed": False,
                    "reviewed_sha256": None,
                    "reviewer": None,
                    "no_clipping": None,
                    "no_overlap": None,
                    "labels_legible": None,
                    "colors_consistent": None,
                    "scientific_boundary_preserved": None,
                },
                "outputs": [
                    {
                        "role": "spatial_context_3d",
                        "path": outputs["spatial_context"]["png"]["path"],
                        "sha256": outputs["spatial_context"]["png"]["sha256"],
                        "width_px": _output_dimensions(spatial_png)[0],
                        "height_px": _output_dimensions(spatial_png)[1],
                        "dpi": 300,
                        "vector_companions": ["svg", "pdf"],
                    },
                    {
                        "role": "spatial_context_3d",
                        "path": outputs["spatial_context"]["svg"]["path"],
                        "sha256": outputs["spatial_context"]["svg"]["sha256"],
                        "width_px": _output_dimensions(spatial_png)[0],
                        "height_px": _output_dimensions(spatial_png)[1],
                        "dpi": 300,
                        "vector_companions": ["png", "pdf"],
                    },
                    {
                        "role": "spatial_context_3d",
                        "path": outputs["spatial_context"]["pdf"]["path"],
                        "sha256": outputs["spatial_context"]["pdf"]["sha256"],
                        "width_px": _output_dimensions(spatial_png)[0],
                        "height_px": _output_dimensions(spatial_png)[1],
                        "dpi": 300,
                        "vector_companions": ["png", "svg"],
                    },
                ],
            },
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PUBLISHED_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_publication(args.output_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
