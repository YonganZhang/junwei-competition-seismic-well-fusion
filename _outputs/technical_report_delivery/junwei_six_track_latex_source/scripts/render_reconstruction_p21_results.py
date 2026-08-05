#!/usr/bin/env python3
"""Render the report-facing P21/P24 reconstruction evidence figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "navy": "#1E466E",
    "blue": "#376795",
    "sky": "#72BCD5",
    "yellow": "#FFD06F",
    "red": "#E76254",
    "grey": "#6B7280",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    p21 = load_json(
        root
        / "_pipelines/02_task_datasets/reconstruction/_outputs/"
        "p21_fixed_foundation_ensemble/summary.json"
    )
    p24 = load_json(
        root
        / "_pipelines/02_task_datasets/reconstruction/_outputs/"
        "p24_historical_transfer/summary.json"
    )

    dev = p21["comparison"]
    transfer = p24["comparison"]
    dev_names = ["PyKrige", "P19", "P21 fixed"]
    dev_values = [
        dev["pykrige"]["rmse"],
        dev["p19"]["rmse"],
        dev["candidate"]["rmse"],
    ]
    transfer_names = ["PyKrige", "Frozen P21"]
    transfer_values = [
        transfer["pykrige"]["rmse"],
        transfer["candidate"]["rmse"],
    ]

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    panels = [
        (
            axes[0],
            dev_names,
            dev_values,
            [PALETTE["grey"], PALETTE["sky"], PALETTE["blue"]],
            "Development spatial OOF",
            "P21 vs P19: 1 win, 4 ties, 0 losses",
        ),
        (
            axes[1],
            transfer_names,
            transfer_values,
            [PALETTE["grey"], PALETTE["red"]],
            "Historical-version transfer",
            "Frozen P21 vs PyKrige: 4 wins, 1 loss",
        ),
    ]

    for ax, names, values, colors, title, note in panels:
        x = np.arange(len(names))
        bars = ax.bar(x, values, width=0.62, color=colors, edgecolor="#263238", linewidth=0.7)
        lower = min(values) - 0.00045
        upper = max(values) + 0.00042
        ax.set_ylim(lower, upper)
        ax.set_xticks(x, names)
        ax.set_ylabel("RMSE (lower is better)")
        ax.set_title(title, color=PALETTE["navy"], fontweight="bold", pad=10)
        ax.grid(axis="y", color="#D9DEE3", linewidth=0.65, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.02,
            0.96,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=PALETTE["navy"],
            fontsize=9,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.000045,
                f"{value:.6f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#202124",
            )

    rel_dev = 100 * (dev_values[0] - dev_values[-1]) / dev_values[0]
    rel_transfer = 100 * (transfer_values[0] - transfer_values[-1]) / transfer_values[0]
    axes[0].annotate(
        f"{rel_dev:.2f}% lower",
        xy=(2, dev_values[-1]),
        xytext=(1.35, dev_values[0] + 0.00020),
        arrowprops={"arrowstyle": "->", "color": PALETTE["blue"], "lw": 1.2},
        color=PALETTE["blue"],
        fontweight="bold",
    )
    axes[1].annotate(
        f"{rel_transfer:.2f}% lower",
        xy=(1, transfer_values[-1]),
        xytext=(0.45, transfer_values[0] + 0.00019),
        arrowprops={"arrowstyle": "->", "color": PALETTE["red"], "lw": 1.2},
        color=PALETTE["red"],
        fontweight="bold",
    )

    fig.suptitle(
        "Foundation-informed reconstruction under fixed spatial protocols",
        color=PALETTE["navy"],
        fontsize=13,
        fontweight="bold",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "before_after_primary_metric"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
