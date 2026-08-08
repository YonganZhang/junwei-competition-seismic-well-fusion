"""Regression tests for the envelope-correlation coarse-search fix candidate
(p45_coarsesearch_envelope_correlation.py).

Honest result locked in here: envelope (Hilbert instantaneous-amplitude)
normalized xcorr is WORSE than I0's raw-amplitude normalized xcorr for this
dataset -- it does not fix the coarse-search robustness problem. These
tests lock in the actual measured numbers, not a hoped-for improvement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_coarsesearch_envelope_correlation import OUTPUT_DIR, envelope  # noqa: E402
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


def test_envelope_is_nonnegative_and_phase_insensitive():
    """np.abs(hilbert(x)) must be a non-negative instantaneous-amplitude
    envelope, and a pure sign flip (180-degree phase/polarity reversal) of
    the input must leave the envelope unchanged -- that is the whole point
    of using it instead of raw amplitude for the coarse-search score."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 512)
    x = np.sin(2 * np.pi * 20 * t) * np.exp(-3 * t) + 0.01 * rng.standard_normal(512)
    env = envelope(x)
    assert np.all(env >= -1e-9)
    env_flipped = envelope(-x)
    assert np.allclose(env, env_flipped, atol=1e-9)


def test_summary_exists_and_covers_st0202_and_st10010():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_coarsesearch_envelope_correlation.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["st0202_results"].keys()) == set(WELLS)
    for well in WELLS:
        st0202_r = summary["st0202_results"][well]
        assert np.isfinite(st0202_r["metrics_vs_own_checkshot"]["mae_ms"])
        assert st0202_r["metrics_vs_own_checkshot"]["rows"] > 0


def test_honest_result_envelope_does_not_improve_st0202_mae():
    """Locks in the measured (negative) result: on ST0202 with the SAME
    blind full-axis T0_SEARCH_MS candidate range as I0, envelope
    correlation improves MAE on 0/3 wells vs I0's raw-amplitude xcorr.
    If a future change to build_synthetic / wavelet estimation flips this,
    this test should be revisited rather than silently loosened."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_coarsesearch_envelope_correlation.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_improved = summary["verdict"]["n_st0202_wells_improved_vs_i0_raw_amplitude"]
    assert n_improved == 0


def test_honest_result_cross_source_drift_not_reduced_vs_i0_blind():
    """Locks in the measured (negative) result: envelope correlation's
    max|ST0202-ST10010 t0 delta| is NOT smaller than I0's raw-amplitude
    blind-search reference (2716ms) -- it is slightly larger here
    (well 19SR still lands on a materially different lobe on ST10010)."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_coarsesearch_envelope_correlation.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verdict = summary["verdict"]
    reduces = verdict["envelope_reduces_cross_source_drift_vs_i0_blind"]
    assert reduces is False


def test_all_three_wells_evaluated_cross_source():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_coarsesearch_envelope_correlation.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cross = summary["cross_source_st0202_vs_st10010"]
    assert all(cross[well]["status"] == "evaluated" for well in WELLS)
