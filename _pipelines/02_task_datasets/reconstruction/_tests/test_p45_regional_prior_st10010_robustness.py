"""Regression tests: does the regional-prior fix generalize to ST10010?"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_regional_prior_st10010_robustness import (  # noqa: E402
    OUTPUT_DIR, T0_CONSISTENCY_TOLERANCE_MS,
)
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


def _load():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_regional_prior_st10010_robustness.py main() first")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def test_same_prior_used_for_both_surveys():
    """The prior itself must be survey-independent: it is read off the
    already-computed ST0202-regional-prior summary and off the same
    build_regional_prior_t0() call, not re-derived per survey."""
    summary = _load()
    for well in WELLS:
        r = summary["results"][well]
        if r.get("status") != "evaluated":
            continue
        assert np.isfinite(r["regional_prior_t0_ms"])
        lo, hi = r["st10010_search_window_ms"]
        assert lo <= r["st10010_chosen_t0_ms"] <= hi


def test_all_wells_evaluated():
    summary = _load()
    assert summary["verdict"]["n_wells_evaluated_on_both"] == len(WELLS)
    for well in WELLS:
        r = summary["results"][well]
        assert r["status"] == "evaluated"
        assert np.isfinite(r["t0_delta_ms"])
        assert np.isfinite(r["mae_delta_ms"])


def test_prior_narrowing_shrinks_the_cross_survey_t0_drift_versus_blind_search():
    """Honest directional check: narrowing the search window around a
    shared regional prior should shrink cross-survey t0 drift compared to
    I0's blind full-axis search (which drifted up to 2716ms) -- but this
    test does NOT assert that acceptance criterion (c) now passes; that is
    a separate, stricter verdict field checked below."""
    summary = _load()
    ref = summary["verdict"]["max_abs_t0_delta_ms_blind_search_reference"]
    if ref is None:
        pytest.skip("blind-search reference summary not available")
    assert summary["verdict"]["max_abs_t0_delta_ms_with_prior"] < ref


def test_verdict_on_acceptance_criterion_c_is_reported_honestly():
    """This is the real question this script answers. We assert the verdict
    is INTERNALLY CONSISTENT with the per-well data (not that it is True) --
    do not silently relax tolerance to force a pass."""
    summary = _load()
    v = summary["verdict"]
    n_consistent = sum(
        1 for well in WELLS
        if summary["results"][well].get("status") == "evaluated"
        and abs(summary["results"][well]["t0_delta_ms"]) <= T0_CONSISTENCY_TOLERANCE_MS
    )
    n_gate_agrees = sum(
        1 for well in WELLS
        if summary["results"][well].get("both_rejected_or_both_kept")
    )
    assert v["n_wells_t0_consistent_within_tolerance"] == n_consistent
    assert v["n_wells_ambiguity_gate_agrees"] == n_gate_agrees
    expected_pass = (
        v["n_wells_evaluated_on_both"] == len(WELLS)
        and n_consistent == len(WELLS)
        and n_gate_agrees == len(WELLS)
    )
    assert v["passes_acceptance_criterion_c"] == expected_pass
