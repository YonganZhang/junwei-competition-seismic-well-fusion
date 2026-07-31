"""Pytest gate for real-data integration checks.

Default test runs stay portable in a clean checkout.  Real output checks are
opt-in because their HDF5/guard prerequisites are intentionally gitignored.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run reservoir integration/data-dependent tests after build and train",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: requires real reservoir HDF5/guard/training artifacts",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-integration"):
        return
    gated = pytest.mark.skip(
        reason=(
            "integration/data-dependent gate disabled; run build -> train -> "
            "pytest --run-integration"
        )
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(gated)
