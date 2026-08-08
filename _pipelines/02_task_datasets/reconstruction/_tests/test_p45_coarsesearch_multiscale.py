"""Regression tests for P45 coarse-to-fine multiscale search."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_coarsesearch_multiscale import (  # noqa: E402
    OUTPUT_DIR, FINE_HALF_WIDTH_MS, COARSE_NMS_GAP_MS, bandpass_filter,
)
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


def _load():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_coarsesearch_multiscale.py main() first")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_bandpass_filter_removes_high_frequency_content():
    """Sanity check on the filter itself (no seismic data needed): a signal
    that is pure 60Hz content should be almost entirely removed by a 2-15Hz
    band-pass, while a pure 8Hz signal should mostly survive."""
    dt_s = 0.004
    t = np.arange(0, 2.0, dt_s)
    high_freq = np.sin(2 * np.pi * 60.0 * t)
    low_freq = np.sin(2 * np.pi * 8.0 * t)
    filtered_high = bandpass_filter(high_freq, dt_s, 2.0, 15.0)
    filtered_low = bandpass_filter(low_freq, dt_s, 2.0, 15.0)
    # compare energy in the steady-state middle of the signal, away from
    # filtfilt edge transients
    mid = slice(len(t) // 4, 3 * len(t) // 4)
    high_energy_ratio = np.std(filtered_high[mid]) / np.std(high_freq[mid])
    low_energy_ratio = np.std(filtered_low[mid]) / np.std(low_freq[mid])
    assert high_energy_ratio < 0.1
    assert low_energy_ratio > 0.5


def test_fine_search_window_is_centered_on_coarse_stage1_pick():
    """The stage-2 fine search window must be built around the stage-1
    coarse center (not some other value), and must respect FINE_HALF_WIDTH_MS
    (clipped only at the survey's own time-axis edges)."""
    summary = _load()
    for well in WELLS:
        r = summary["st0202_results"][well]
        lo, hi = r["fine_search_window_ms"]
        center = r["coarse_center_t0_ms"]
        # window should be within [center-W, center+W], possibly clipped
        assert lo >= center - FINE_HALF_WIDTH_MS - 1e-6
        assert hi <= center + FINE_HALF_WIDTH_MS + 1e-6
        assert lo <= r["chosen_t0_ms"] <= hi


def test_st0202_summary_reports_honest_per_well_comparison():
    summary = _load()
    n_improved = sum(1 for w in WELLS if summary["st0202_results"][w]["improved_vs_i0_blind"])
    assert summary["verdict"]["st0202_n_wells_improved_vs_i0_blind_search"] == n_improved
    n_avoided = sum(1 for w in WELLS if summary["st0202_results"][w]["avoided_i0s_cycle_jump"])
    assert summary["verdict"]["st0202_n_wells_avoided_i0s_cycle_jump"] == n_avoided
    for well in WELLS:
        r = summary["st0202_results"][well]
        assert np.isfinite(r["metrics_multiscale"]["mae_ms"])
        assert np.isfinite(r["mae_delta_vs_i0_blind_ms"])
        # avoided_i0s_cycle_jump implies BOTH a >gap-ms different coarse center
        # AND a strict MAE improvement -- it must not be claimed on MAE
        # improvement alone (that would not demonstrate escaping a cycle jump)
        if r["avoided_i0s_cycle_jump"]:
            assert abs(r["coarse_center_t0_ms"] - r["i0_blind_t0_ms"]) > COARSE_NMS_GAP_MS
            assert r["improved_vs_i0_blind"]


def test_st10010_all_wells_evaluated_and_cross_source_delta_is_internally_consistent():
    summary = _load()
    assert summary["verdict"]["st10010_n_wells_evaluated"] == len(WELLS)
    for well in WELLS:
        r = summary["st10010_results"][well]
        assert r["status"] == "evaluated"
        assert np.isfinite(r["metrics_multiscale"]["mae_ms"])
        expected_delta = r["chosen_t0_ms"] - summary["st0202_results"][well]["chosen_t0_ms"]
        assert r["multiscale_t0_delta_ms_st0202_vs_st10010"] == pytest.approx(expected_delta, abs=1e-6)

    reported_max = summary["verdict"]["max_abs_t0_delta_ms_multiscale_st0202_vs_st10010"]
    computed_max = max(
        abs(summary["st10010_results"][w]["multiscale_t0_delta_ms_st0202_vs_st10010"]) for w in WELLS
    )
    assert reported_max == pytest.approx(computed_max, abs=1e-6)


def test_band_sensitivity_diagnostic_verdict_is_internally_consistent():
    """This is the honesty check on the coarse-search-by-lowpass idea itself:
    the top-level verdict flag must match what the per-band per-well data
    actually shows, not be asserted independently of it."""
    summary = _load()
    diag = summary["band_sensitivity_diagnostic"]
    any_band_wins = False
    bands = set()
    for well in WELLS:
        bands |= set(diag[well]["bands"].keys())
    for band_key in bands:
        if all(diag[w]["bands"].get(band_key, {}).get("within_fine_half_width") for w in WELLS):
            any_band_wins = True
            break
    assert diag["_verdict"]["any_single_band_lands_all_3_wells_within_fine_half_width_ms"] == any_band_wins
    assert summary["verdict"]["any_single_band_lands_all_3_wells_within_fine_half_width_ms"] == any_band_wins
