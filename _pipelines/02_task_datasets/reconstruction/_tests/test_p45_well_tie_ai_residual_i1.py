"""Regression tests for P45 I1 learned residual correction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_well_tie_ai_residual_i1 import OUTPUT_DIR, WELLS  # noqa: E402


def test_summary_has_all_three_wells_evaluated():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_ai_residual_i1.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"]["n_wells_evaluated"] == 3
    for well in WELLS:
        assert summary["results"][well]["status"] == "evaluated"


def test_no_catastrophic_extrapolation():
    """Locks in the fix for the well-identity-leakage bug: corrected MAE
    must stay within the same order of magnitude as the physics baseline
    (survey record length is 4500ms total), not blow up into seconds."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_ai_residual_i1.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for well in WELLS:
        r = summary["results"][well]
        assert r["i1_corrected_metrics"]["mae_ms"] < 4500.0
        assert np.isfinite(r["i1_corrected_metrics"]["mae_ms"])


def test_well_identity_features_are_excluded():
    """The point-level feature set must not include a well-constant
    covariate (see finding: this caused a 32-second extrapolation blowup
    in the first draft)."""
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_ai_residual_i1.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "well_xcorr" not in summary["feature_names"]
    assert "well_ambiguity_gap" not in summary["feature_names"]


def test_verdict_reflects_actual_no_improvement():
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_well_tie_ai_residual_i1.py main() first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    n_improved = sum(1 for w in WELLS if summary["results"][w]["improved"])
    assert summary["verdict"]["n_wells_improved"] == n_improved
    if n_improved < 3:
        assert summary["verdict"]["decision"] == "I1_NOT_PROMOTED_INSUFFICIENT_EVIDENCE"
