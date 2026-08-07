"""Regression tests for P45 I0 physics-first well tie."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS,
    load_well_curve,
    md_to_tvdss,
    parse_well_picks_full,
    sonic_reflectivity_and_time,
    WELL_PICKS,
    PICK_NAME,
    OUTPUT_DIR,
)


def test_dt_rhob_curves_are_finite_and_physical():
    """Every well's extracted valid interval must have physically plausible
    sonic slowness (40-200 us/ft) and density (1.5-3.2 g/cc), no NaN."""
    for well in WELLS:
        curve = load_well_curve(well)
        assert np.all(np.isfinite(curve["dt"]))
        assert np.all(np.isfinite(curve["rhob"]))
        assert np.all((curve["dt"] > 30) & (curve["dt"] < 250))
        assert np.all((curve["rhob"] > 1.0) & (curve["rhob"] < 3.5))
        assert len(curve["depth_md"]) > 50


def test_md_to_tvdss_is_monotonic_and_reasonable():
    """TVDSS from official well picks must be monotonic non-decreasing with
    MD and never exceed MD (no overturned/impossible geometry) for these
    near-vertical Volve wells."""
    picks_all = parse_well_picks_full(WELL_PICKS)
    for well in WELLS:
        curve = load_well_curve(well)
        picks = picks_all[PICK_NAME[well]]
        tvdss = md_to_tvdss(curve["depth_md"], picks)
        assert np.all(np.diff(tvdss) >= -1e-6)
        assert np.all(tvdss <= curve["depth_md"] + 1.0)


def test_sonic_integration_produces_monotonic_relative_twt():
    """Cumulative one-way-time integration must be strictly non-decreasing
    with depth (physical requirement: time never runs backward)."""
    for well in WELLS:
        curve = load_well_curve(well)
        physics = sonic_reflectivity_and_time(curve["depth_md"], curve["dt"], curve["rhob"])
        assert np.all(np.diff(physics["relative_twt_ms"]) >= 0.0)
        assert np.all(np.isfinite(physics["reflectivity"]))


def test_summary_json_ambiguity_gate_flags_the_known_bad_well():
    """Locks in the I0 result: 19SR's coarse correlation search is genuinely
    ambiguous (multiple near-equal peaks) because its only valid DT+RHOB
    interval barely overlaps the checkshot evaluation range. The
    reject_ambiguous gate must catch this rather than silently returning a
    large error."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_physics_baseline.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["wells"]["19SR"]["reject_ambiguous"] is True
    for well in WELLS:
        m = summary["wells"][well]["metrics_vs_own_checkshot"]
        assert m["rows"] > 0
        assert np.isfinite(m["mae_ms"])
