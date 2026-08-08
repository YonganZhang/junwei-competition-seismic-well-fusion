"""Tests for the P45 acceptance-criteria (2)+(6) audit.

Two layers:
  - unit tests on audit_curve()/spearman_report() with synthetic arrays
    (deterministic, no seismic I/O) to lock in the monotonicity/velocity
    detection logic itself.
  - regression checks on the actual summary.json produced by main() against
    the real 3-well data, if it has already been run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_monotonicity_and_uncertainty_audit import (  # noqa: E402
    OUTPUT_DIR, VELOCITY_MIN_M_S, VELOCITY_MAX_M_S, audit_curve,
    collect_uncertainty_points, spearman_report,
)
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


def test_audit_curve_detects_strictly_monotonic_plausible_velocity_curve():
    depth = np.linspace(2700.0, 3200.0, 200)
    # constant 2500 m/s one-way -> twt_ms = 2000*depth/2500 = 0.8*depth (+const)
    twt = 0.8 * depth
    result = audit_curve("SYN", "synthetic_ok", depth, twt)
    assert result["strictly_monotonic_increasing"] is True
    assert result["n_non_monotonic_steps"] == 0
    assert result["n_velocity_out_of_range"] == 0
    assert VELOCITY_MIN_M_S <= result["velocity_m_s_median_observed"] <= VELOCITY_MAX_M_S


def test_audit_curve_flags_non_monotonic_steps():
    depth = np.linspace(2700.0, 2710.0, 6)
    # inject a reversal: twt goes down at index 3
    twt = np.array([2000.0, 2010.0, 2020.0, 2015.0, 2030.0, 2040.0])
    result = audit_curve("SYN", "synthetic_reversal", depth, twt)
    assert result["strictly_monotonic_increasing"] is False
    assert result["n_non_monotonic_steps"] == 1
    assert result["non_monotonic_fraction"] == pytest.approx(1.0 / 5.0)
    assert len(result["examples_non_monotonic"]) == 1
    assert result["examples_non_monotonic"][0]["twt_delta_ms"] < 0


def test_audit_curve_flags_implausible_velocity():
    depth = np.linspace(2700.0, 2701.0, 3)  # 0.5 m steps
    # 0.5 m in 0.001 ms one-way -> absurd velocity, way above 6000 m/s
    twt = np.array([2000.0, 2000.002, 2000.004])
    result = audit_curve("SYN", "synthetic_too_fast", depth, twt)
    assert result["strictly_monotonic_increasing"] is True  # still increasing
    assert result["n_velocity_out_of_range"] == result["n_velocity_steps_checked"]
    assert result["n_velocity_out_of_range"] > 0


def test_spearman_report_low_n_handled_honestly():
    r = spearman_report([1.0, 2.0], [3.0, 1.0], "x", "y")
    assert r["n"] == 2
    assert r["spearman_r"] is None
    r2 = spearman_report([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], "x", "y")
    assert r2["spearman_r"] == pytest.approx(-1.0)
    assert "LOW CONFIDENCE" in r2["note"]


def test_collect_uncertainty_points_reads_existing_summaries_without_rerunning():
    points = collect_uncertainty_points()
    # 3 wells x 3 already-run experiments (I0/ST0202, regional_prior, I0/ST10010)
    assert len(points) == 9
    methods = {p["method"] for p in points}
    assert methods == {"I0_blind_ST0202", "regional_prior", "I0_blind_ST10010"}
    for p in points:
        assert p["well"] in WELLS
        assert np.isfinite(p["ambiguity_gap"])
        assert np.isfinite(p["coarse_xcorr"])
        assert np.isfinite(p["mae_ms"])


def test_output_summary_structure_and_honesty_flags():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_monotonicity_and_uncertainty_audit.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    c2 = summary["criterion_2_monotonicity_and_velocity"]
    assert c2["aggregate"]["n_curves"] == 6  # 3 wells x 2 methods (I0 + regional_prior)
    for curve in c2["curves"]:
        assert curve["n_steps"] == curve["n_non_monotonic_steps"] + (
            curve["n_steps"] - curve["n_non_monotonic_steps"]
        )
        assert 0.0 <= curve["non_monotonic_fraction"] <= 1.0

    c6 = summary["criterion_6_uncertainty_vs_error"]
    assert c6["n_points"] == 9
    assert "caveat" in c6 and "low-confidence" in c6["caveat"].lower()
    assert c6["spearman_ambiguity_gap_vs_mae"]["n"] == 9
    assert c6["spearman_coarse_xcorr_vs_mae"]["n"] == 9

    # honesty gate: verdict must not silently claim criterion_2 passed if any
    # non-monotonic step or out-of-range velocity was actually found.
    agg = c2["aggregate"]
    expected_pass = (agg["n_curves_non_monotonic"] == 0 and agg["total_velocity_out_of_range"] == 0)
    assert summary["verdict"]["criterion_2_passed"] == expected_pass
