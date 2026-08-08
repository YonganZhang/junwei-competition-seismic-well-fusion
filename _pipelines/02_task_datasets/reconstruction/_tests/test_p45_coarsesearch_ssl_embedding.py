"""Regression tests for the SSL-encoder-embedding coarse-search fix
candidate (p45_coarsesearch_ssl_embedding.py).

Honest, measured result locked in here (not a hoped-for improvement):
embedding-cosine scoring is a MIXED result on this dataset, not a clean
win. On ST0202 it improves MAE on 2/3 wells (worse on 19A: 589ms vs
265ms; better on 19BT2 and 19SR). On ST10010 cross-source robustness it
gives a smaller |t0 drift| on 2/3 wells (worse on 19BT2: 1740ms vs
772ms). These tests lock in the actual measured counts, not a rounded-up
verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from p45_coarsesearch_ssl_embedding import (  # noqa: E402
    ENCODER_PATH, OUTPUT_DIR, cosine_similarity, embedding_score, encode_window,
    load_encoder, predict_well_embedding,
)
from p45_well_tie_physics_baseline import SEGY_PATH, WELLS  # noqa: E402
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402


def _require_encoder():
    if not ENCODER_PATH.exists():
        pytest.skip("pretrained encoder missing -- run p45_seismic_trace_ssl_pretrain.py first")


def _require_summary() -> dict:
    summary_path = OUTPUT_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("run p45_coarsesearch_ssl_embedding.py main() first")
    return json.loads(summary_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# unit-level: cosine scoring behaves sanely
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([0.3, -1.2, 4.0, 0.7])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    v = np.array([0.3, -1.2, 4.0, 0.7])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_zero_vector_returns_sentinel_not_nan():
    v = np.array([0.3, -1.2, 4.0, 0.7])
    z = np.zeros(4)
    assert cosine_similarity(v, z) == -1.0
    assert not np.isnan(cosine_similarity(v, z))


def test_embedding_score_identical_real_window_is_near_one():
    """Encoding the exact same window against itself must score close to
    the maximum -- the sanity floor for 'the score function does what it
    says'."""
    _require_encoder()
    encoder = load_encoder(ENCODER_PATH)
    rng = np.random.default_rng(0)
    window = np.cumsum(rng.standard_normal(200)) * 0.1  # smooth-ish fake trace
    score = embedding_score(encoder, window, window.copy())
    assert score > 0.99


def test_embedding_score_too_short_window_returns_sentinel():
    _require_encoder()
    encoder = load_encoder(ENCODER_PATH)
    short = np.array([1.0, 2.0, 3.0])
    assert embedding_score(encoder, short, short) == -1.0


def test_encode_window_is_deterministic_in_eval_mode():
    _require_encoder()
    encoder = load_encoder(ENCODER_PATH)
    rng = np.random.default_rng(1)
    window = rng.standard_normal(180).astype(np.float32)
    e1 = encode_window(encoder, window)
    e2 = encode_window(encoder, window)
    assert np.allclose(e1, e2)
    assert e1.shape == (encoder.embed_dim,)


# ---------------------------------------------------------------------------
# predict_well_embedding: structural sanity on a small (fast) candidate range
# ---------------------------------------------------------------------------

def test_predict_well_embedding_structure_on_narrow_range():
    _require_encoder()
    encoder = load_encoder(ENCODER_PATH)
    index = load_index()
    candidates = np.arange(0.0, 3600.0, 80.0)  # coarse/fast for the test
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        pred = predict_well_embedding("19BT2", index, vol, encoder, t0_candidates=candidates)
    assert pred["score_type"] == "embedding_cosine_similarity"
    assert candidates[0] <= pred["coarse_t0_ms"] <= candidates[-1]
    assert -1.0 <= pred["coarse_score"] <= 1.0
    assert len(pred["top_k_peaks"]) >= 1
    assert len(pred["predicted_twt_ms"]) == len(pred["depth_md"])
    assert np.all(np.isfinite(pred["predicted_twt_ms"]))


# ---------------------------------------------------------------------------
# summary.json: real full-run coverage + the honest measured result
# ---------------------------------------------------------------------------

def test_summary_covers_st0202_and_st10010_all_wells():
    summary = _require_summary()
    assert set(summary["st0202_results"].keys()) == set(WELLS)
    for well in WELLS:
        m = summary["st0202_results"][well]["metrics_vs_own_checkshot"]
        assert np.isfinite(m["mae_ms"])
        assert m["rows"] > 0
    for well in WELLS:
        r10 = summary["st10010_results"][well]
        assert r10["status"] == "evaluated"
        assert np.isfinite(r10["metrics_vs_own_checkshot"]["mae_ms"])


def test_honest_result_st0202_mixed_not_universal_improvement():
    """Locks in the measured count: embedding-cosine beats xcorr MAE on
    2/3 ST0202 wells, NOT 3/3 -- it is not a clean universal win."""
    summary = _require_summary()
    assert summary["verdict"]["st0202_n_wells_embedding_beats_xcorr_mae"] == 2
    assert summary["verdict"]["st0202_n_wells_total"] == 3
    # 19A is the one that got worse -- pin the specific regression, not just the count.
    assert summary["st0202_vs_xcorr_baseline"]["19A"]["embedding_improves_on_xcorr"] is False


def test_honest_result_st10010_cross_source_drift_mixed():
    """Locks in the measured count: embedding |t0 drift| is smaller than
    xcorr's on 2/3 wells, NOT all three -- 19BT2 drifts MORE (1740ms vs
    772ms) under the embedding score."""
    summary = _require_summary()
    assert summary["verdict"]["st10010_n_wells_embedding_drift_smaller_than_xcorr"] == 2
    assert summary["verdict"]["st10010_n_wells_evaluated"] == 3
    assert summary["st10010_vs_xcorr_baseline"]["19BT2"]["embedding_drift_smaller_than_xcorr"] is False


def test_ambiguity_gate_threshold_and_search_range_unchanged_from_i0():
    summary = _require_summary()
    assert summary["config"]["ambiguity_gap_threshold"] == 0.05
    assert summary["config"]["t0_search_range_ms"] == [0.0, 3596.0]
