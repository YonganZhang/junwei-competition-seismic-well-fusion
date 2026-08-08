"""Black-box + honesty tests for p45_coarsesearch_bandpass_correlation.py.

Runs the real comparison against real well/seismic data (ST0202 + ST10010,
no mocking of physics) and checks: (1) structural completeness, (2) every
derived/aggregate number in summary.json is independently reproducible from
the per-well rows it claims to summarize, and (3) the pass/fail verdict
fields are internally consistent -- this script does NOT assert the new
method "wins"; it asserts the reported verdict matches the underlying data,
whatever that data says.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

import p45_coarsesearch_bandpass_correlation as mod  # noqa: E402
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    mod.main()
    path = mod.OUTPUT_DIR / "summary.json"
    assert path.exists()
    return json.loads(path.read_text(encoding="utf-8"))


# --- unit-level: filter design helpers -------------------------------------

def test_design_bandpass_matches_direct_butter_call():
    dt_s = 0.004  # 4ms, i.e. ST0202/ST10010 sample interval
    ba = mod.design_bandpass(dt_s, 8.0, 60.0, order=mod.BANDPASS_ORDER)
    assert ba is not None
    fs = 1.0 / dt_s
    nyq = 0.5 * fs
    expected = butter(mod.BANDPASS_ORDER, [8.0 / nyq, 60.0 / nyq], btype="band")
    np.testing.assert_allclose(ba[0], expected[0])
    np.testing.assert_allclose(ba[1], expected[1])


def test_design_bandpass_returns_none_for_unrepresentable_band():
    # low_hz above Nyquist makes low >= 1.0 -> band collapses, caller must
    # fall back to raw (unfiltered) signal instead of crashing.
    dt_s = 0.004
    ba = mod.design_bandpass(dt_s, 10_000.0, 20_000.0, order=mod.BANDPASS_ORDER)
    assert ba is None


def test_apply_bandpass_none_returns_unfiltered_copy():
    trace = np.array([1.0, 2.0, 3.0, 4.0])
    out = mod.apply_bandpass(trace, None)
    np.testing.assert_array_equal(out, trace)
    assert out is not trace  # must be a copy, not the same array object


def test_apply_bandpass_changes_signal_when_filter_is_active():
    rng = np.random.default_rng(0)
    dt_s = 0.004
    trace = rng.normal(size=512)
    ba = mod.design_bandpass(dt_s, 8.0, 60.0)
    filtered = mod.apply_bandpass(trace, ba)
    assert filtered.shape == trace.shape
    assert not np.allclose(filtered, trace)


# --- structural completeness ------------------------------------------------

def test_both_surveys_cover_all_wells(summary):
    for survey_key in ("st0202", "st10010"):
        assert set(summary[survey_key].keys()) == set(WELLS)
        for well in WELLS:
            row = summary[survey_key][well]
            for method_key in ("raw_amplitude", "bandpass_correlation"):
                m = row[method_key]
                assert np.isfinite(m["coarse_t0_ms"])
                assert np.isfinite(m["metrics_vs_own_checkshot"]["mae_ms"])


def test_bandpass_band_reported_matches_config(summary):
    assert summary["band_hz"] == [mod.BANDPASS_LOW_HZ, mod.BANDPASS_HIGH_HZ]
    for survey_key in ("st0202", "st10010"):
        for well in WELLS:
            assert summary[survey_key][well]["bandpass_correlation"]["band_hz"] == [
                mod.BANDPASS_LOW_HZ, mod.BANDPASS_HIGH_HZ,
            ]


# --- per-well delta reproducibility ----------------------------------------

def test_mae_delta_matches_independent_recomputation(summary):
    for survey_key in ("st0202", "st10010"):
        for well in WELLS:
            row = summary[survey_key][well]
            recomputed = (
                row["bandpass_correlation"]["metrics_vs_own_checkshot"]["mae_ms"]
                - row["raw_amplitude"]["metrics_vs_own_checkshot"]["mae_ms"]
            )
            assert row["mae_delta_ms_bandpass_minus_raw"] == pytest.approx(recomputed, abs=1e-9)
            assert row["bandpass_improved"] == (recomputed < 0)


def test_cross_source_t0_delta_matches_independent_recomputation(summary):
    for well in WELLS:
        raw_delta = abs(
            summary["st0202"][well]["raw_amplitude"]["coarse_t0_ms"]
            - summary["st10010"][well]["raw_amplitude"]["coarse_t0_ms"]
        )
        bp_delta = abs(
            summary["st0202"][well]["bandpass_correlation"]["coarse_t0_ms"]
            - summary["st10010"][well]["bandpass_correlation"]["coarse_t0_ms"]
        )
        row = summary["cross_source_t0_robustness"][well]
        assert row["raw_amplitude_abs_t0_delta_ms"] == pytest.approx(raw_delta, abs=1e-6)
        assert row["bandpass_abs_t0_delta_ms"] == pytest.approx(bp_delta, abs=1e-6)
        assert row["bandpass_reduces_t0_delta"] == (bp_delta < raw_delta)


# --- verdict internal consistency (no "must win" assertions) --------------

def test_verdict_counts_match_underlying_rows(summary):
    v = summary["verdict"]
    n_st0202 = sum(1 for well in WELLS if summary["st0202"][well]["bandpass_improved"])
    n_st10010 = sum(1 for well in WELLS if summary["st10010"][well]["bandpass_improved"])
    assert v["n_st0202_wells_improved_by_bandpass"] == n_st0202
    assert v["n_st10010_wells_improved_by_bandpass"] == n_st10010

    sr19_row = summary["st0202"]["19SR"]
    expected_sr19 = bool(
        sr19_row["bandpass_correlation"]["metrics_vs_own_checkshot"]["mae_ms"]
        < sr19_row["raw_amplitude"]["metrics_vs_own_checkshot"]["mae_ms"]
    )
    assert v["well_19SR_st0202_periodic_jump_avoided"] == expected_sr19


def test_verdict_max_deltas_match_cross_source_table(summary):
    v = summary["verdict"]
    cross = summary["cross_source_t0_robustness"]
    max_raw = max(row["raw_amplitude_abs_t0_delta_ms"] for row in cross.values())
    max_bp = max(row["bandpass_abs_t0_delta_ms"] for row in cross.values())
    assert v["max_abs_t0_delta_ms_raw_amplitude_this_run"] == pytest.approx(max_raw, abs=1e-6)
    assert v["max_abs_t0_delta_ms_bandpass_this_run"] == pytest.approx(max_bp, abs=1e-6)

    worst_raw = max(cross, key=lambda w: cross[w]["raw_amplitude_abs_t0_delta_ms"])
    worst_bp = max(cross, key=lambda w: cross[w]["bandpass_abs_t0_delta_ms"])
    assert v["worst_well_raw_amplitude"] == worst_raw
    assert v["worst_well_bandpass"] == worst_bp


def test_verdict_reference_comparisons_are_honest_booleans(summary):
    """These flags must reflect a plain numeric comparison against the fixed
    reference constants -- not be hand-set to True regardless of outcome."""
    v = summary["verdict"]
    ref = summary["reference_numbers"]
    assert v["bandpass_beats_blind_search_reference_2716ms"] == (
        v["max_abs_t0_delta_ms_bandpass_this_run"] < ref["blind_search_max_abs_t0_delta_ms"]
    )
    assert v["bandpass_beats_regional_prior_reference_316ms"] == (
        v["max_abs_t0_delta_ms_bandpass_this_run"] < ref["regional_prior_max_abs_t0_delta_ms"]
    )


def test_reference_numbers_match_previously_established_diagnostics(summary):
    # These are fixed facts already reported by earlier scripts in this
    # series -- pinning them here catches accidental drift of the constants.
    ref = summary["reference_numbers"]
    assert ref["blind_search_max_abs_t0_delta_ms"] == pytest.approx(2716.0)
    assert ref["regional_prior_max_abs_t0_delta_ms"] == pytest.approx(316.0)
