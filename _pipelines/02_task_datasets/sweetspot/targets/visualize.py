"""Read-only, target-specific sweetspot visualizers.

The module consumes archived prediction tables only.  It never imports model or
training code, refits thresholds, or accesses raw labels outside the archive.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve


TARGET_ALIASES = {
    "reservoir_quality": "reservoir_quality",
    "hydrocarbon_pay": "hydrocarbon_pay",
    "productivity": "productivity",
    "water_breakthrough": "water_breakthrough",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"prediction archive missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction archive is empty")


def _finish(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _box_by_well(frame: pd.DataFrame, values: np.ndarray, title: str, path: Path) -> None:
    wells = sorted(frame["well"].astype(str).unique())
    groups = [values[frame["well"].astype(str).to_numpy() == well] for well in wells]
    figure, axis = plt.subplots(figsize=(max(6, len(wells) * 1.1), 4))
    axis.boxplot(groups, tick_labels=wells, showfliers=False)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_ylabel("prediction - observed")
    axis.tick_params(axis="x", rotation=30)
    _finish(figure, path)


def _scatter(frame: pd.DataFrame, path: Path, title: str) -> None:
    observed = frame["observed"].to_numpy(float)
    prediction = frame["prediction"].to_numpy(float)
    finite = np.isfinite(observed) & np.isfinite(prediction)
    if not finite.any():
        raise ValueError("no finite observed/prediction pairs")
    lo = float(min(observed[finite].min(), prediction[finite].min()))
    hi = float(max(observed[finite].max(), prediction[finite].max()))
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(observed[finite], prediction[finite], s=10, alpha=0.45)
    axis.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axis.set(xlabel="observed", ylabel="prediction", title=title)
    _finish(figure, path)


def _render_reservoir_quality(frame: pd.DataFrame, output: Path, threshold: float) -> list[Path]:
    del threshold
    _require(frame, ("well", "depth_m", "observed", "prediction"))
    paths = [output / name for name in (
        "depth_track.png", "observed_predicted.png", "residual_by_well.png", "rank_by_well.png",
    )]
    wells = sorted(frame["well"].astype(str).unique())
    figure, axes = plt.subplots(1, len(wells), figsize=(max(6, 3 * len(wells)), 6), squeeze=False)
    for axis, well in zip(axes[0], wells):
        part = frame[frame["well"].astype(str) == well].sort_values("depth_m")
        axis.plot(part["observed"], part["depth_m"], label="observed", linewidth=1)
        axis.plot(part["prediction"], part["depth_m"], label="prediction", linewidth=1)
        axis.invert_yaxis(); axis.set_title(well); axis.set_xlabel("RQI")
    axes[0][0].set_ylabel("measured depth (m)"); axes[0][0].legend(fontsize=8)
    _finish(figure, paths[0])
    _scatter(frame, paths[1], "Reservoir quality: archived predictions")
    _box_by_well(frame, frame["prediction"].to_numpy(float) - frame["observed"].to_numpy(float), "RQI residuals by well", paths[2])
    ranks = frame.groupby("well")[["observed", "prediction"]].mean().sort_values("observed")
    figure, axis = plt.subplots(figsize=(max(6, len(ranks) * 1.1), 4))
    x = np.arange(len(ranks)); axis.bar(x - .18, ranks["observed"], .36, label="observed"); axis.bar(x + .18, ranks["prediction"], .36, label="prediction")
    axis.set_xticks(x, ranks.index, rotation=30); axis.set_ylabel("mean RQI"); axis.set_title("Well ranking"); axis.legend()
    _finish(figure, paths[3])
    return paths


def _pr_plot(frame: pd.DataFrame, probability: np.ndarray, path: Path, title: str) -> None:
    observed = frame["observed"].to_numpy(int)
    if set(np.unique(observed)) != {0, 1}:
        raise ValueError("PR plot requires both classes")
    precision, recall, _ = precision_recall_curve(observed, probability)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision)
    axis.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1), title=title)
    _finish(figure, path)


def _render_hydrocarbon_pay(frame: pd.DataFrame, output: Path, threshold: float) -> list[Path]:
    _require(frame, ("well", "depth_m", "observed", "probability"))
    paths = [output / name for name in (
        "depth_probability_track.png", "pr_curve.png", "confusion_matrix.png", "net_thickness_by_well.png",
    )]
    wells = sorted(frame["well"].astype(str).unique())
    figure, axes = plt.subplots(1, len(wells), figsize=(max(6, 3 * len(wells)), 6), squeeze=False)
    for axis, well in zip(axes[0], wells):
        part = frame[frame["well"].astype(str) == well].sort_values("depth_m")
        axis.plot(part["probability"], part["depth_m"], label="P(sand)")
        axis.step(part["observed"], part["depth_m"], label="flag", where="mid", alpha=.7)
        axis.axvline(threshold, color="black", linestyle="--", linewidth=.8)
        axis.invert_yaxis(); axis.set(xlim=(0, 1), title=well, xlabel="probability")
    axes[0][0].set_ylabel("measured depth (m)"); axes[0][0].legend(fontsize=8)
    _finish(figure, paths[0])
    probability = frame["probability"].to_numpy(float)
    _pr_plot(frame, probability, paths[1], "Sand/net-reservoir proxy PR")
    observed = frame["observed"].to_numpy(int); predicted = (probability >= threshold).astype(int)
    figure, axis = plt.subplots(figsize=(4.5, 4)); ConfusionMatrixDisplay(confusion_matrix(observed, predicted)).plot(ax=axis, colorbar=False); axis.set_title(f"Frozen threshold={threshold:g}")
    _finish(figure, paths[2])
    rows = []
    for well, part in frame.groupby("well"):
        depths = np.sort(part["depth_m"].to_numpy(float)); spacing = float(np.median(np.diff(depths))) if len(depths) > 1 else 0.0
        rows.append((well, spacing * part["observed"].sum(), spacing * (part["probability"] >= threshold).sum()))
    net = pd.DataFrame(rows, columns=["well", "observed", "prediction"]).set_index("well")
    figure, axis = plt.subplots(figsize=(max(6, len(net) * 1.1), 4)); net.plot.bar(ax=axis); axis.set_ylabel("proxy net thickness (m)"); axis.set_title("Net thickness by well")
    _finish(figure, paths[3])
    return paths


def _render_productivity(frame: pd.DataFrame, output: Path, threshold: float) -> list[Path]:
    del threshold
    _require(frame, ("well", "cutoff_date", "observed", "prediction"))
    paths = [output / name for name in (
        "time_series_forecast.png", "observed_predicted.png", "residual_by_well.png", "topk_ranking.png",
    )]
    frame = frame.copy(); frame["cutoff_date"] = pd.to_datetime(frame["cutoff_date"])
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for well, part in frame.groupby("well"):
        part = part.sort_values("cutoff_date")
        axis.plot(part["cutoff_date"], part["observed"], linewidth=1, label=f"{well} obs")
        axis.plot(part["cutoff_date"], part["prediction"], linestyle="--", linewidth=1, label=f"{well} pred")
    axis.set(ylabel="future 30-day mean oil (Sm3/day)", title="Causal productivity forecasts"); axis.legend(ncol=2, fontsize=7)
    _finish(figure, paths[0])
    _scatter(frame, paths[1], "Productivity: archived predictions")
    _box_by_well(frame, frame["prediction"].to_numpy(float) - frame["observed"].to_numpy(float), "Productivity residuals by well", paths[2])
    top = frame.groupby("well")[["observed", "prediction"]].mean().sort_values("observed", ascending=False)
    figure, axis = plt.subplots(figsize=(max(6, len(top) * 1.1), 4)); top.plot.bar(ax=axis); axis.set_ylabel("mean future oil"); axis.set_title("Top-well ranking")
    _finish(figure, paths[3])
    return paths


def _render_water_breakthrough(frame: pd.DataFrame, output: Path, threshold: float) -> list[Path]:
    _require(frame, ("well", "cutoff_date", "observed", "probability"))
    paths = [output / name for name in (
        "risk_timeline.png", "pr_curve.png", "calibration.png", "confusion_by_well.png",
    )]
    frame = frame.copy(); frame["cutoff_date"] = pd.to_datetime(frame["cutoff_date"])
    figure, axis = plt.subplots(figsize=(10, 4.5))
    for well, part in frame.groupby("well"):
        part = part.sort_values("cutoff_date")
        axis.plot(part["cutoff_date"], part["probability"], marker=".", linewidth=1, label=well)
        positive = part[part["observed"].astype(int) == 1]
        axis.scatter(positive["cutoff_date"], positive["probability"], marker="x", s=35)
    axis.axhline(threshold, color="black", linestyle="--", linewidth=.8); axis.set(ylim=(0, 1), ylabel="30-day event risk", title="Pre-event water risk timeline"); axis.legend(fontsize=8)
    _finish(figure, paths[0])
    probability = frame["probability"].to_numpy(float)
    _pr_plot(frame, probability, paths[1], "Water-event risk PR")
    observed = frame["observed"].to_numpy(int)
    frac, mean = calibration_curve(observed, probability, n_bins=min(8, max(2, len(frame) // 10)), strategy="quantile")
    figure, axis = plt.subplots(figsize=(5, 4)); axis.plot([0, 1], [0, 1], "k--"); axis.plot(mean, frac, marker="o"); axis.set(xlabel="mean predicted risk", ylabel="event fraction", xlim=(0, 1), ylim=(0, 1), title="OOF/frozen-test calibration")
    _finish(figure, paths[2])
    rows = []
    for well, part in frame.groupby("well"):
        pred = (part["probability"].to_numpy(float) >= threshold).astype(int); obs = part["observed"].to_numpy(int)
        matrix = confusion_matrix(obs, pred, labels=[0, 1]); rows.append((well, int(matrix[1, 1]), int(matrix[0, 1]), int(matrix[1, 0])))
    counts = pd.DataFrame(rows, columns=["well", "TP", "FP", "FN"]).set_index("well")
    figure, axis = plt.subplots(figsize=(max(6, len(counts) * 1.1), 4)); counts.plot.bar(ax=axis); axis.set_ylabel("sample count"); axis.set_title("Risk confusion counts by well")
    _finish(figure, paths[3])
    return paths


RENDERERS: dict[str, Callable[[pd.DataFrame, Path, float], list[Path]]] = {
    "reservoir_quality": _render_reservoir_quality,
    "hydrocarbon_pay": _render_hydrocarbon_pay,
    "productivity": _render_productivity,
    "water_breakthrough": _render_water_breakthrough,
}


def render(target_id: str, prediction_csv: Path, output_dir: Path, *, frozen_threshold: float = 0.5) -> dict[str, object]:
    """Render a target-specific archive and write a provenance manifest."""
    if target_id not in RENDERERS:
        raise ValueError(f"unsupported feasible target_id={target_id!r}")
    prediction_csv = prediction_csv.resolve()
    before = _sha256(prediction_csv)
    frame = pd.read_csv(prediction_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = RENDERERS[target_id](frame, output_dir, frozen_threshold)
    after = _sha256(prediction_csv)
    if before != after:
        raise RuntimeError("prediction archive changed during visualization")
    manifest: dict[str, object] = {
        "target_id": target_id,
        # Keep archives portable: provenance stores the local role/name and
        # content hash, never a host- or worktree-specific absolute path.
        "prediction_path": prediction_csv.name,
        "prediction_sha256": before,
        "frozen_threshold": frozen_threshold,
        "figures": [{"name": path.name, "sha256": _sha256(path)} for path in figures],
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
