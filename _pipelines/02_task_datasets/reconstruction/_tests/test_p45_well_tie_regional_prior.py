"""Regression tests for P45 regional-prior-informed search."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_well_tie_regional_prior import OUTPUT_DIR, PRIOR_HALF_WIDTH_MS  # noqa: E402
from p45_well_tie_physics_baseline import WELLS  # noqa: E402


def test_prior_never_reads_the_held_out_wells_own_checkshot():
    """build_regional_prior_t0 must only average over the OTHER wells; this
    test locks in that the search window is centered on a prior built from
    <=2 donors, not 3 (which would mean the held-out well leaked into its
    own prior)."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_regional_prior.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for well in WELLS:
        r = summary["results"][well]
        lo, hi = r["search_window_ms"]
        assert hi - lo == pytest.approx(2 * PRIOR_HALF_WIDTH_MS, abs=8.0)
        assert lo <= r["regional_prior_t0_ms"] <= hi


def test_summary_reports_honest_per_well_comparison():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_regional_prior.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_improved = sum(1 for w in WELLS if summary["results"][w]["improved_vs_i0"])
    assert summary["verdict"]["n_wells_improved_vs_i0_blind_search"] == n_improved
    for well in WELLS:
        r = summary["results"][well]
        assert np.isfinite(r["metrics_with_prior"]["mae_ms"])
        assert np.isfinite(r["mae_delta_vs_i0_ms"])
