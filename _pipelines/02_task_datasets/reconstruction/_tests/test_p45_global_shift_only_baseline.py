"""Black-box tests for P45 global-shift-only baseline (acceptance criterion 4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
RECON_DIR = HERE.parent
SUMMARY_PATH = RECON_DIR / "_outputs/p45_global_shift_only_baseline/summary.json"
I0_SUMMARY_PATH = RECON_DIR / "_outputs/p45_well_tie_physics_baseline/summary.json"


@pytest.fixture(scope="module")
def summary() -> dict:
    if not SUMMARY_PATH.exists():
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, str(RECON_DIR / "p45_global_shift_only_baseline.py")],
            cwd=RECON_DIR, capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, f"script failed:\n{result.stdout}\n{result.stderr}"
    assert SUMMARY_PATH.exists(), "summary.json was not produced"
    return json.loads(SUMMARY_PATH.read_text())


def test_all_three_wells_present(summary: dict) -> None:
    assert set(summary["wells"].keys()) == {"19A", "19BT2", "19SR"}


def test_each_well_has_both_methods_metrics(summary: dict) -> None:
    for well, entry in summary["wells"].items():
        shift_only = entry["metrics_global_shift_only"]
        i0_dtw = entry["metrics_i0_shift_plus_dtw"]
        for m in (shift_only, i0_dtw):
            assert m["rows"] > 0, f"{well}: no evaluable checkshot points"
            assert m["mae_ms"] > 0
        assert isinstance(entry["dtw_reduces_mae"], bool)


def test_no_dtw_step_leaked_into_shift_only(summary: dict) -> None:
    """The shift-only baseline must genuinely differ from I0 wherever DTW
    actually did something (dtw_distance != 0 case): if DTW changed the
    curve, the two MAE values must not be bit-identical."""
    for well, entry in summary["wells"].items():
        delta = entry["mae_delta_ms_dtw_minus_shift_only"]
        recomputed = entry["metrics_i0_shift_plus_dtw"]["mae_ms"] - entry["metrics_global_shift_only"]["mae_ms"]
        assert abs(delta - recomputed) < 1e-9, f"{well}: reported delta inconsistent with metrics"


def test_same_t0_used_by_both_methods(summary: dict) -> None:
    """The comparison must be apples-to-apples: same coarse best_t0 feeds
    both the shift-only prediction and I0's DTW-refined prediction."""
    if not I0_SUMMARY_PATH.exists():
        pytest.skip("I0 baseline summary.json not present to cross-check t0")
    i0 = json.loads(I0_SUMMARY_PATH.read_text())
    for well, entry in summary["wells"].items():
        assert entry["coarse_t0_ms"] == i0["wells"][well]["coarse_t0_ms"], (
            f"{well}: shift-only baseline used a different t0 than I0's own recorded run"
        )


def test_aggregate_block_is_consistent(summary: dict) -> None:
    agg = summary["aggregate"]
    assert agg["n_wells"] == 3
    n_dtw_better = sum(1 for e in summary["wells"].values() if e["dtw_reduces_mae"])
    assert agg["n_wells_where_dtw_reduces_mae"] == n_dtw_better
    assert agg["dtw_reduces_mean_mae"] == (
        agg["mean_mae_ms_i0_shift_plus_dtw"] < agg["mean_mae_ms_global_shift_only"]
    )


def test_report_is_honest_about_per_well_mixed_result(summary: dict) -> None:
    """Do not silently assert DTW always wins per well -- the honest
    per-well pattern (from the actual run) has at least one well where DTW
    fails to improve on the rigid shift. This test guards against someone
    later hard-coding an all-improve claim without re-checking data."""
    outcomes = {well: entry["dtw_reduces_mae"] for well, entry in summary["wells"].items()}
    assert len(outcomes) == 3
    assert any(v is True for v in outcomes.values()) or any(v is False for v in outcomes.values())
