"""Regression tests for the unified, configurable well-tie pipeline
(p45_unified_well_tie_pipeline.py).

Covers: default config (regional prior + amplitude scoring) runs
end-to-end and reproduces the known regional-prior numbers; scoring_method
switches without crashing (amplitude <-> ssl_embedding, and an invalid
value raises cleanly); use_regional_prior=False correctly degrades to the
full blind T0_SEARCH_MS axis instead of a narrowed window.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_unified_well_tie_pipeline import (  # noqa: E402
    OUTPUT_DIR, SSL_ENCODER_PATH, run_well_tie_pipeline, ssl_encoder_available,
    stage2_t0_search_window,
)
from p45_well_tie_physics_baseline import (  # noqa: E402
    SEGY_PATH, T0_SEARCH_MS, WELLS, CHECKSHOT_MEMBER, VSP_ZIP, WELL_PICKS,
    load_well_curve, parse_checkshots, parse_well_picks_full,
)
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402


def _require_summary() -> dict:
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_unified_well_tie_pipeline.py main() first")
    return json.loads(summary_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage 2 unit-level: search-window narrowing / blind fallback wiring
# ---------------------------------------------------------------------------

def test_stage2_falls_back_to_full_range_when_prior_disabled():
    curve = load_well_curve("19A")
    candidates, info = stage2_t0_search_window("19A", False, None, None, curve)
    assert info["used_regional_prior"] is False
    assert info["search_window_ms"] == [float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1])]
    assert len(candidates) == len(T0_SEARCH_MS)


def test_stage2_narrows_window_when_prior_enabled():
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)
    curve = load_well_curve("19A")
    candidates, info = stage2_t0_search_window("19A", True, checkshots, all_picks, curve)
    assert info["used_regional_prior"] is True
    lo, hi = info["search_window_ms"]
    assert hi - lo < (T0_SEARCH_MS[-1] - T0_SEARCH_MS[0])
    assert lo <= info["prior_t0_ms"] <= hi
    assert candidates[0] >= 0.0 and candidates[-1] <= hi + 1.0
    # never leaks the held-out well's own checkshot into its own prior
    assert "19A" not in info["donor_wells"]


def test_stage2_donor_wells_restricts_to_requested_subset():
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)
    curve = load_well_curve("19A")
    _, info = stage2_t0_search_window("19A", True, checkshots, all_picks, curve, donor_wells=["19BT2"])
    assert info["donor_wells"] == ["19BT2"]


# ---------------------------------------------------------------------------
# run_well_tie_pipeline: direct calls, narrow candidate range for speed
# ---------------------------------------------------------------------------

def test_default_config_end_to_end_returns_complete_result():
    """Default config (use_regional_prior=True, scoring_method='amplitude')
    must run start to finish and return a predicted curve, a confidence
    score, and an explicit reject flag."""
    index = load_index()
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        result = run_well_tie_pipeline("19BT2", index, vol)
    assert result["config"]["use_regional_prior"] is True
    assert result["config"]["scoring_method"] == "amplitude"
    assert len(result["predicted_twt_ms"]) == len(result["depth_md"])
    assert np.all(np.isfinite(result["predicted_twt_ms"]))
    assert -1.0 <= result["confidence"] <= 1.0
    assert isinstance(result["reject_ambiguous"], bool)
    assert result["stage2_search_window"]["used_regional_prior"] is True


def test_scoring_method_amplitude_vs_ssl_embedding_both_run_without_crashing():
    """The Stage-3 slot must be switchable: both accepted values must
    complete and return a scoring-method-tagged confidence, on the exact
    same (narrow, fast) t0 candidate window."""
    index = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)
    curve = load_well_curve("19A")
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        amp = run_well_tie_pipeline(
            "19A", index, vol, scoring_method="amplitude",
            checkshots=checkshots, all_picks=all_picks, curve=curve,
        )
        assert amp["config"]["scoring_method"] == "amplitude"
        assert np.isfinite(amp["confidence"])

        if not ssl_encoder_available():
            pytest.skip(f"pretrained encoder missing at {SSL_ENCODER_PATH}")
        emb = run_well_tie_pipeline(
            "19A", index, vol, scoring_method="ssl_embedding",
            checkshots=checkshots, all_picks=all_picks, curve=curve,
        )
    assert emb["config"]["scoring_method"] == "ssl_embedding"
    assert -1.0 <= emb["confidence"] <= 1.0
    # both methods searched the SAME stage-2 window (scoring choice must not
    # change what candidates are considered)
    assert amp["stage2_search_window"]["search_window_ms"] == emb["stage2_search_window"]["search_window_ms"]


def test_invalid_scoring_method_raises_clean_value_error():
    index = load_index()
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        with pytest.raises(ValueError):
            run_well_tie_pipeline("19A", index, vol, scoring_method="not_a_real_method")


def test_use_regional_prior_false_degrades_to_full_blind_search():
    """use_regional_prior=False must NOT silently narrow the window -- it
    must produce the exact full T0_SEARCH_MS axis, same as I0's blind
    search."""
    index = load_index()
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        result = run_well_tie_pipeline("19A", index, vol, use_regional_prior=False)
    assert result["stage2_search_window"]["used_regional_prior"] is False
    assert result["stage2_search_window"]["search_window_ms"] == [
        float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1]),
    ]


# ---------------------------------------------------------------------------
# summary.json: full run coverage on all 3 wells, default config
# ---------------------------------------------------------------------------

def test_summary_default_config_matches_known_regional_prior_numbers():
    """Locks in that the unified pipeline's default config reproduces the
    already-measured best-known combination (regional prior + amplitude
    scoring) from p45_well_tie_regional_prior.py: 19A ~88.4ms, 19BT2
    ~312.8ms, 19SR ~326.2ms MAE."""
    summary = _require_summary()
    assert summary["status"] == "OK"
    known_mae = {"19A": 88.4, "19BT2": 312.8, "19SR": 326.2}
    for well in WELLS:
        r = summary["default_config_results"][well]
        assert np.isfinite(r["metrics_vs_own_checkshot"]["mae_ms"])
        assert r["metrics_vs_own_checkshot"]["mae_ms"] == pytest.approx(known_mae[well], abs=2.0)
        assert r["stage2_search_window"]["used_regional_prior"] is True


def test_summary_reports_ssl_embedding_slot_status_honestly():
    summary = _require_summary()
    slot = summary["ssl_embedding_slot"]
    assert isinstance(slot["available"], bool)
    if slot["available"]:
        for well in WELLS:
            r = slot["results_with_regional_prior_window"].get(well, {})
            if r.get("status") == "evaluated":
                assert np.isfinite(r["metrics_vs_own_checkshot"]["mae_ms"])
    else:
        assert slot["results_with_regional_prior_window"].get("status") == "SKIPPED_ENCODER_MISSING"


def test_summary_blind_fallback_check_matches_full_range():
    summary = _require_summary()
    check = summary["stage2_switch_check_blind_fallback"]["19A"]
    assert check["used_regional_prior"] is False
    assert check["matches_full_t0_search_range"] is True
