#!/usr/bin/env python3
"""Render the P6 foundation-pretraining ablation from audited JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "baseline": "#376795",
    "pretrained": "#E76254",
    "random": "#72BCD5",
    "edge": "#1E466E",
}


def parse_args() -> argparse.Namespace:
    pipeline_dir = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve().parents[2]
    output_root = project_root / "_figures" / "p6_foundation_adaptation"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--property-json",
        type=Path,
        default=pipeline_dir
        / "_outputs"
        / "property_lora_multiseed"
        / "property_timellm_pilot.json",
    )
    parser.add_argument(
        "--lithofacies-json",
        type=Path,
        default=pipeline_dir
        / "_outputs"
        / "lithofacies_lora_multiseed"
        / "lithofacies_timellm_pilot.json",
    )
    parser.add_argument("--output-dir", type=Path, default=output_root)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return path.name


def normalize_fonts(fig: mpl.figure.Figure) -> None:
    """Apply a single serif family to every rendered text object."""
    for text in fig.findobj(mpl.text.Text):
        text.set_fontfamily("serif")
        text.set_fontname("Liberation Serif")


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#5B6770")
    ax.spines["bottom"].set_color("#5B6770")
    ax.tick_params(axis="both", colors="#28323C", labelsize=8.5, length=3)
    ax.grid(axis="y", color="#D7DEE5", linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)


def draw_panel(
    ax: mpl.axes.Axes,
    labels: list[str],
    values: list[float],
    errors: list[float],
    ylabel: str,
    panel: str,
) -> None:
    x = np.arange(len(labels), dtype=float)
    ax.bar(
        x,
        values,
        yerr=errors,
        width=0.62,
        color=[PALETTE["baseline"], PALETTE["pretrained"], PALETTE["random"]],
        edgecolor=PALETTE["edge"],
        linewidth=0.7,
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_ylim(bottom=0)
    ax.text(
        -0.12,
        1.04,
        panel,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    style_axis(ax)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    property_data = load_json(args.property_json)
    lithofacies_data = load_json(args.lithofacies_json)

    property_agg = property_data["aggregate"]
    lithofacies_agg = lithofacies_data["aggregate"]

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelcolor": "#28323C",
            "text.color": "#28323C",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    draw_panel(
        axes[0],
        ["Ridge", "GPT-2\n+ LoRA", "Random GPT-2\n+ LoRA"],
        [
            property_agg["ridge_same_sequence"]["macro_standardized_RMSE_mean"],
            property_agg["timellm_pretrained_lora_gpt2"]["macro_standardized_RMSE_mean"],
            property_agg["timellm_random_lora_gpt2"]["macro_standardized_RMSE_mean"],
        ],
        [
            property_agg["ridge_same_sequence"]["macro_standardized_RMSE_std"],
            property_agg["timellm_pretrained_lora_gpt2"]["macro_standardized_RMSE_std"],
            property_agg["timellm_random_lora_gpt2"]["macro_standardized_RMSE_std"],
        ],
        "Macro standardized RMSE",
        "a",
    )
    draw_panel(
        axes[1],
        ["Logistic", "GPT-2\n+ LoRA", "Random GPT-2\n+ LoRA"],
        [
            lithofacies_agg["logistic_same_sequence"]["accuracy_mean"],
            lithofacies_agg["timellm_pretrained_lora_gpt2"]["accuracy_mean"],
            lithofacies_agg["timellm_random_lora_gpt2"]["accuracy_mean"],
        ],
        [
            lithofacies_agg["logistic_same_sequence"]["accuracy_std"],
            lithofacies_agg["timellm_pretrained_lora_gpt2"]["accuracy_std"],
            lithofacies_agg["timellm_random_lora_gpt2"]["accuracy_std"],
        ],
        "Accuracy",
        "b",
    )
    fig.tight_layout(w_pad=2.3, pad=0.8)
    normalize_fonts(fig)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "fig1_foundation_pretraining_ablation.png"
    pdf_path = args.output_dir / "fig1_foundation_pretraining_ablation.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    figure_data = {
        "schema_version": "1.0",
        "figure": "fig1_foundation_pretraining_ablation",
        "source_evidence": {
            "property_json": {
                "path": portable_path(args.property_json, project_root),
                "sha256": sha256(args.property_json),
            },
            "lithofacies_json": {
                "path": portable_path(args.lithofacies_json, project_root),
                "sha256": sha256(args.lithofacies_json),
            },
        },
        "panels": {
            "a": {
                "track": "property",
                "metric": "macro_standardized_RMSE",
                "lower_is_better": True,
                "values": [
                    property_agg["ridge_same_sequence"],
                    property_agg["timellm_pretrained_lora_gpt2"],
                    property_agg["timellm_random_lora_gpt2"],
                ],
            },
            "b": {
                "track": "lithofacies",
                "metric": "accuracy",
                "higher_is_better": True,
                "values": [
                    lithofacies_agg["logistic_same_sequence"],
                    lithofacies_agg["timellm_pretrained_lora_gpt2"],
                    lithofacies_agg["timellm_random_lora_gpt2"],
                ],
            },
        },
        "font_request": "Times New Roman",
        "font_rendered": "Liberation Serif",
        "note": "Error bars are standard deviations across three training seeds; baselines are deterministic single fits.",
    }
    with (args.output_dir / "figure_data.json").open("w", encoding="utf-8") as handle:
        json.dump(figure_data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
