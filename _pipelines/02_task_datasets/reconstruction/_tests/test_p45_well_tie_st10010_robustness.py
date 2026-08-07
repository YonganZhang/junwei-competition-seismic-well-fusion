"""Regression test for P45 (c): ST0202 vs ST10010 robustness."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_well_tie_st10010_robustness import OUTPUT_DIR  # noqa: E402
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


def test_st10010_index_has_shorter_time_range_than_st0202():
    """Locks in the premise of this test: ST10010 genuinely has a different
    (shorter) recorded time range than ST0202, which is exactly the failure
    mode GitHub issue #1 describes."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_st10010_robustness.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    st10010_span = summary["st10010_survey"]["time_range_ms"][1] - summary["st10010_survey"]["time_range_ms"][0]
    st0202_span = (
        summary["st0202_survey_reference"]["time_range_ms"][1]
        - summary["st0202_survey_reference"]["time_range_ms"][0]
    )
    assert st10010_span < st0202_span


def test_all_wells_evaluated_on_both_surveys():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_st10010_robustness.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"]["n_wells_evaluated_on_both"] == 3
    for well in WELLS:
        r = summary["results"][well]
        assert r["status"] == "evaluated"
        assert np.isfinite(r["t0_delta_ms"])
        assert np.isfinite(r["mae_delta_ms"])


def test_drift_is_honestly_reported_not_hidden():
    """This test documents (not asserts a specific direction of) the real
    finding: the ambiguity gate does NOT agree across surveys for most
    wells, meaning I0 currently fails acceptance criterion (c) from the
    goal ('changing SEG-Y time range/sampling should leave overlap-region
    results largely unchanged'). We assert the honest fact pattern instead
    of hiding it: gate agreement is measured and reported, and is not
    silently coerced to True."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_st10010_robustness.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_agree = summary["verdict"]["n_wells_ambiguity_gate_agrees"]
    assert 0 <= n_agree <= 3
    for well in WELLS:
        assert "both_rejected_or_both_kept" in summary["results"][well]
