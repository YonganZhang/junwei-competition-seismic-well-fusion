"""Create the minimum P4 run artifact layout without fabricating results."""
from __future__ import annotations

from pathlib import Path


REQUIRED_DIRECTORIES = (
    "folds",
    "oof",
    "hpo",
    "refit",
    "frozen_test",
    "visualizations",
)


def create_run_layout(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    for relative in REQUIRED_DIRECTORIES:
        (run_root / relative).mkdir(exist_ok=True)
    return run_root


def assert_visualization_is_read_only(*, prediction_path: Path, metrics_path: Path) -> None:
    if not prediction_path.is_file():
        raise FileNotFoundError(f"archived prediction is required: {prediction_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"archived metrics are required: {metrics_path}")
