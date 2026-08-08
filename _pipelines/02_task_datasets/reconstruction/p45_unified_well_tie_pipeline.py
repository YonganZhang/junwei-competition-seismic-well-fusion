#!/usr/bin/env python3
"""P45: unified, configurable well-seismic tie pipeline.

GitHub issue #1 asks for an end-to-end well-tie capability for an unknown
survey with no checkshot / no time-depth table. This module does not add a
new method -- every stage below is a thin orchestration layer that REUSES
the already-tested building blocks in this directory:

  Stage 1 (physics)        -> p45_well_tie_physics_baseline.sonic_reflectivity_and_time
                               / build_synthetic / estimate_wavelet_frequency
                               (embedded inside predict_well, reused whole)
  Stage 2 (search-window)  -> p45_well_tie_regional_prior.build_regional_prior_t0
                               narrows the t0 search window using OTHER
                               wells' real checkshot (inverse-distance
                               weighted); falls back to the full blind
                               T0_SEARCH_MS axis when no prior is available
                               or use_regional_prior=False.
  Stage 3 (scoring)        -> switchable: "amplitude" (I0's normalized
                               xcorr, default -- the best-known combination
                               per the user's own measured numbers) or
                               "ssl_embedding" (p45_coarsesearch_ssl_embedding's
                               self-supervised-encoder cosine similarity --
                               an optional slot, off by default, wired in
                               but NOT proven to help; see that module's
                               own honest mixed-result verdict).
  Stage 4 (refine)         -> p45_well_tie_physics_baseline.dtw_refine,
                               slope-constrained DTW (bundled inside
                               predict_well / predict_well_embedding).
  Stage 5 (ambiguity gate) -> reject_ambiguous when the top-2 t0 peaks are
                               closer than 0.05 in score (bundled inside the
                               same predict_* functions, unchanged).

Stages 3+4+5 are not reimplemented here: predict_well (amplitude scoring)
and predict_well_embedding (ssl_embedding scoring) each already bundle
scoring -> DTW refine -> ambiguity gate as one call, with byte-identical
DTW/gate logic between the two (locked in by p45_coarsesearch_ssl_embedding's
own tests). This module's job is ONLY to pick which one runs, and to build
the Stage 2 search window that gets handed to it -- that is the "complete,
configurable pipeline" deliverable, not a new modeling method.

Single entry point:

    run_well_tie_pipeline(well, index, get_trace,
                           use_regional_prior=True,
                           scoring_method="amplitude",
                           donor_wells=None) -> dict

Default config (use_regional_prior=True, scoring_method="amplitude") is the
best-known combination measured so far (p45_well_tie_regional_prior.py):
19A MAE ~88.4ms, 19BT2 ~312.8ms, 19SR ~326.2ms on ST0202, all three flagged
reject_ambiguous=True by the Stage-5 gate (an honest "low confidence,
don't trust this blindly" signal, not swept under -- this pipeline reports
that gate faithfully rather than hiding it).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, SEGY_PATH, WELL_PICKS, PICK_NAME,
    T0_SEARCH_MS, predict_well, metrics, parse_checkshots, parse_well_picks_full,
    load_well_curve, md_to_tvdss, sha256,
)
from p45_well_tie_regional_prior import (  # noqa: E402
    build_regional_prior_t0, well_xy, PRIOR_HALF_WIDTH_MS, PRIOR_SEARCH_STEP_MS,
)
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_unified_well_tie_pipeline"
SSL_ENCODER_PATH = HERE / "_outputs/p45_seismic_trace_ssl_pretrain/encoder.pt"

SCORING_METHODS = ("amplitude", "ssl_embedding")

# process-local cache so a multi-well run only loads the SSL encoder once.
_SSL_ENCODER_CACHE: dict = {}


def _load_ssl_encoder():
    """Stage-3 optional slot: self-supervised encoder, lazily imported so
    that scoring_method='amplitude' (the default) never requires torch to
    even be importable at call time."""
    if "encoder" not in _SSL_ENCODER_CACHE:
        from p45_coarsesearch_ssl_embedding import load_encoder  # noqa: E402 (lazy)
        _SSL_ENCODER_CACHE["encoder"] = load_encoder(SSL_ENCODER_PATH)
    return _SSL_ENCODER_CACHE["encoder"]


def ssl_encoder_available() -> bool:
    return SSL_ENCODER_PATH.exists()


# ---------------------------------------------------------------------------
# Stage 2: search-window narrowing (regional prior) or blind fallback
# ---------------------------------------------------------------------------

def _prior_t0_from_donors(held_out: str, checkshots: dict, all_picks: dict,
                           held_curve: dict, donors: list[str]) -> float:
    """Same inverse-distance-weighted math as build_regional_prior_t0, but
    restricted to a caller-supplied donor subset. build_regional_prior_t0
    itself always averages over ALL other WELLS, so this thin variant is
    what lets run_well_tie_pipeline's donor_wells= actually restrict which
    neighbours contribute -- not a rewrite of the core function, just its
    donor-loop generalized to an arbitrary list."""
    held_picks = all_picks[PICK_NAME[held_out]]
    hx, hy = well_xy(held_out, held_curve, held_picks)
    target_tvdss = float(np.median(md_to_tvdss(held_curve["depth_md"], held_picks)))
    numerator, denominator = 0.0, 0.0
    for donor in donors:
        depth_true, twt_true = checkshots[donor]
        donor_picks = all_picks[PICK_NAME[donor]]
        donor_curve = load_well_curve(donor)
        dx, dy = well_xy(donor, donor_curve, donor_picks)
        distance2 = (hx - dx) ** 2 + (hy - dy) ** 2
        weight = 1.0 / (distance2 + 100.0 ** 2)
        twt_at_target = float(np.interp(target_tvdss, depth_true, twt_true))
        numerator += weight * twt_at_target
        denominator += weight
    return numerator / denominator if denominator > 0 else 1500.0


def stage2_t0_search_window(well: str, use_regional_prior: bool, checkshots: dict | None,
                             all_picks: dict | None, curve: dict,
                             donor_wells: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Returns (t0_candidates, info). Falls back to the full blind
    T0_SEARCH_MS axis whenever a prior cannot be built (prior disabled, or
    no checkshot data available for any donor)."""
    full_range_info = {
        "used_regional_prior": False,
        "prior_t0_ms": None,
        "donor_wells": [],
        "search_window_ms": [float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1])],
    }
    if not use_regional_prior or checkshots is None or all_picks is None:
        return T0_SEARCH_MS, {**full_range_info, "reason": "regional_prior_disabled" if not use_regional_prior
                               else "no_checkshot_data_supplied"}

    candidate_donors = [w for w in (donor_wells if donor_wells is not None else WELLS)
                        if w != well and w in checkshots]
    if not candidate_donors:
        return T0_SEARCH_MS, {**full_range_info, "reason": "no_donor_wells_available"}

    if set(candidate_donors) == {w for w in WELLS if w != well}:
        prior_t0 = build_regional_prior_t0(well, checkshots, all_picks, curve)
    else:
        prior_t0 = _prior_t0_from_donors(well, checkshots, all_picks, curve, candidate_donors)

    lo = max(0.0, prior_t0 - PRIOR_HALF_WIDTH_MS)
    hi = prior_t0 + PRIOR_HALF_WIDTH_MS
    candidates = np.arange(lo, hi, PRIOR_SEARCH_STEP_MS)
    return candidates, {
        "used_regional_prior": True,
        "prior_t0_ms": prior_t0,
        "donor_wells": candidate_donors,
        "search_window_ms": [float(lo), float(hi)],
    }


# ---------------------------------------------------------------------------
# Stage 3+4+5: scoring (switchable) + DTW refine + ambiguity gate, delegated
# whole to the already-tested predict_well / predict_well_embedding.
# ---------------------------------------------------------------------------

def stage345_score_refine_gate(well: str, index: dict, get_trace, t0_candidates: np.ndarray,
                                scoring_method: str) -> dict:
    if scoring_method == "amplitude":
        pred = predict_well(well, index, get_trace, t0_candidates=t0_candidates)
        pred["confidence"] = pred["coarse_xcorr"]
        pred["score_type"] = "amplitude_normalized_xcorr"
    elif scoring_method == "ssl_embedding":
        from p45_coarsesearch_ssl_embedding import predict_well_embedding  # noqa: E402 (lazy)
        encoder = _load_ssl_encoder()
        pred = predict_well_embedding(well, index, get_trace, encoder, t0_candidates=t0_candidates)
        pred["confidence"] = pred["coarse_score"]
    else:
        raise ValueError(f"unknown scoring_method={scoring_method!r}, expected one of {SCORING_METHODS}")
    return pred


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def run_well_tie_pipeline(well: str, index: dict, get_trace, *,
                           use_regional_prior: bool = True,
                           scoring_method: str = "amplitude",
                           donor_wells: list[str] | None = None,
                           checkshots: dict | None = None,
                           all_picks: dict | None = None,
                           curve: dict | None = None) -> dict:
    """Full 5-stage well-tie pipeline for one well, no checkshot/time-depth
    table for the well itself required.

    Parameters
    ----------
    use_regional_prior : narrow the Stage-2 t0 search window using nearby
        wells' real checkshot (default True, the best-known combination
        measured so far). False -> full blind T0_SEARCH_MS axis.
    scoring_method : "amplitude" (default, I0's normalized xcorr) or
        "ssl_embedding" (optional self-supervised-encoder cosine slot,
        wired in but off by default -- its own module reports a mixed,
        not universally better, result).
    donor_wells : restrict which OTHER wells contribute to the Stage-2
        prior. None (default) -> all other WELLS with checkshot data.
    checkshots / all_picks / curve : optional pre-loaded caches so a
        multi-well caller (see main() below) does not reparse the same
        VSP zip / well-picks file / logs per well.

    Returns
    -------
    dict with the predicted depth->TWT curve, a scoring-method-agnostic
    "confidence" score, the Stage-5 reject flag, and per-stage bookkeeping
    (stage2 search window / prior info) for auditability.
    """
    if scoring_method not in SCORING_METHODS:
        raise ValueError(f"unknown scoring_method={scoring_method!r}, expected one of {SCORING_METHODS}")

    curve = curve if curve is not None else load_well_curve(well)
    if use_regional_prior:
        if checkshots is None:
            checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
        if all_picks is None:
            all_picks = parse_well_picks_full(WELL_PICKS)

    t0_candidates, stage2_info = stage2_t0_search_window(
        well, use_regional_prior, checkshots, all_picks, curve, donor_wells,
    )
    pred = stage345_score_refine_gate(well, index, get_trace, t0_candidates, scoring_method)

    return {
        "well": well,
        "config": {
            "use_regional_prior": use_regional_prior,
            "scoring_method": scoring_method,
            "donor_wells_requested": donor_wells,
        },
        "stage2_search_window": stage2_info,
        "depth_md": pred["depth_md"],
        "depth_tvdss": pred["depth_tvdss"],
        "predicted_twt_ms": pred["predicted_twt_ms"],
        "coarse_t0_ms": pred["coarse_t0_ms"],
        "confidence": pred["confidence"],
        "top_k_peaks": pred["top_k_peaks"],
        "ambiguity_gap": pred["ambiguity_gap"],
        "reject_ambiguous": pred["reject_ambiguous"],
        "dtw_distance": pred["dtw_distance"],
        "wavelet_freq_hz": pred["wavelet_freq_hz"],
        "inline": pred["inline"],
        "crossline": pred["crossline"],
    }


def _score_against_checkshot(result: dict, checkshots: dict) -> dict:
    depth_true, twt_true = checkshots[result["well"]]
    pred_at_checkshot_depth = np.interp(
        depth_true, result["depth_tvdss"], result["predicted_twt_ms"],
        left=np.nan, right=np.nan,
    )
    valid = np.isfinite(pred_at_checkshot_depth)
    valid &= (depth_true >= result["depth_tvdss"].min()) & (depth_true <= result["depth_tvdss"].max())
    return metrics(twt_true[valid], pred_at_checkshot_depth[valid])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    index = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)
    curves = {well: load_well_curve(well) for well in WELLS}

    encoder_available = ssl_encoder_available()
    print(f"ssl_embedding slot: {'available (' + str(SSL_ENCODER_PATH) + ')' if encoder_available else 'NOT available -- slot stays off/placeholder'}")

    default_results = {}
    ssl_results = {}
    blind_fallback_window_check = {}
    donor_subset_check = {}

    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        # --- default config: regional prior + amplitude scoring (best-known combo) ---
        print("\n=== default config: use_regional_prior=True, scoring_method='amplitude' ===")
        for well in WELLS:
            pred = run_well_tie_pipeline(
                well, index, vol, use_regional_prior=True, scoring_method="amplitude",
                checkshots=checkshots, all_picks=all_picks, curve=curves[well],
            )
            m = _score_against_checkshot(pred, checkshots)
            default_results[well] = {
                "stage2_search_window": pred["stage2_search_window"],
                "coarse_t0_ms": pred["coarse_t0_ms"],
                "confidence": pred["confidence"],
                "ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "dtw_distance": pred["dtw_distance"],
                "metrics_vs_own_checkshot": m,
            }
            print(f"  {well}: t0={pred['coarse_t0_ms']:.0f}ms conf={pred['confidence']:.3f} "
                  f"MAE={m['mae_ms']:.2f}ms reject={pred['reject_ambiguous']}")

        # --- Stage 3 switch check: ssl_embedding scoring (same Stage-2 window), optional slot ---
        print("\n=== scoring_method switch: 'ssl_embedding' (same regional-prior window) ===")
        if encoder_available:
            for well in WELLS:
                try:
                    pred = run_well_tie_pipeline(
                        well, index, vol, use_regional_prior=True, scoring_method="ssl_embedding",
                        checkshots=checkshots, all_picks=all_picks, curve=curves[well],
                    )
                    m = _score_against_checkshot(pred, checkshots)
                    ssl_results[well] = {
                        "status": "evaluated",
                        "coarse_t0_ms": pred["coarse_t0_ms"],
                        "confidence": pred["confidence"],
                        "reject_ambiguous": pred["reject_ambiguous"],
                        "metrics_vs_own_checkshot": m,
                        "mae_delta_vs_default_ms": m["mae_ms"] - default_results[well]["metrics_vs_own_checkshot"]["mae_ms"],
                    }
                    print(f"  {well}: t0={pred['coarse_t0_ms']:.0f}ms conf={pred['confidence']:.3f} "
                          f"MAE={m['mae_ms']:.2f}ms (default MAE={default_results[well]['metrics_vs_own_checkshot']['mae_ms']:.2f}ms)")
                except Exception as exc:  # honest: record the failure, do not hide it
                    ssl_results[well] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                    print(f"  {well}: ssl_embedding scoring FAILED -- {exc}")
        else:
            ssl_results = {"status": "SKIPPED_ENCODER_MISSING", "encoder_path": str(SSL_ENCODER_PATH)}
            print(f"  skipped: encoder missing at {SSL_ENCODER_PATH}")

        # --- Stage 2 switch check: use_regional_prior=False -> full blind axis ---
        print("\n=== use_regional_prior=False switch check (search window only, structural) ===")
        for well in ("19A",):  # one well is enough to prove the fallback wiring; full blind
            # search for all 3 wells is already covered by p45_well_tie_physics_baseline.
            pred = run_well_tie_pipeline(well, index, vol, use_regional_prior=False, scoring_method="amplitude")
            blind_fallback_window_check[well] = {
                "search_window_ms": pred["stage2_search_window"]["search_window_ms"],
                "used_regional_prior": pred["stage2_search_window"]["used_regional_prior"],
                "matches_full_t0_search_range": pred["stage2_search_window"]["search_window_ms"]
                    == [float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1])],
            }
            print(f"  {well}: window={blind_fallback_window_check[well]['search_window_ms']} "
                  f"(full T0_SEARCH_MS=[{T0_SEARCH_MS[0]:.0f},{T0_SEARCH_MS[-1]:.0f}])")

        # --- donor_wells restriction check: single-donor prior for 19A ---
        print("\n=== donor_wells= restriction check (19A, single donor) ===")
        single_donor = "19BT2"
        pred = run_well_tie_pipeline(
            "19A", index, vol, use_regional_prior=True, scoring_method="amplitude",
            donor_wells=[single_donor], checkshots=checkshots, all_picks=all_picks, curve=curves["19A"],
        )
        donor_subset_check = {
            "well": "19A",
            "requested_donor_wells": [single_donor],
            "stage2_search_window": pred["stage2_search_window"],
        }
        print(f"  19A (donor={single_donor} only): window={pred['stage2_search_window']['search_window_ms']}")

    elapsed_s = time.time() - t_start
    n_default_evaluated = sum(1 for w in WELLS if np.isfinite(default_results[w]["metrics_vs_own_checkshot"]["mae_ms"]))

    summary = {
        "status": "OK",
        "method": (
            "5-stage configurable well-tie pipeline: (1) sonic-integration "
            "physics [reused from p45_well_tie_physics_baseline], (2) "
            "regional-checkshot-prior t0 search-window narrowing or blind "
            "fallback [reused from p45_well_tie_regional_prior], (3) "
            "switchable scoring: amplitude xcorr (default) or ssl_embedding "
            "(optional, off by default) [reused from "
            "p45_coarsesearch_ssl_embedding], (4) slope-constrained DTW "
            "refine, (5) ambiguity gate -- (4)+(5) bundled inside the "
            "stage-3 predict_* call, unchanged."
        ),
        "default_config": {"use_regional_prior": True, "scoring_method": "amplitude"},
        "default_config_results": default_results,
        "default_config_n_wells_evaluated": n_default_evaluated,
        "ssl_embedding_slot": {
            "available": encoder_available,
            "encoder_path": str(SSL_ENCODER_PATH),
            "note": (
                "wired into stage3_score_refine_gate via scoring_method='ssl_embedding'; "
                "own module (p45_coarsesearch_ssl_embedding) reports a MIXED accuracy "
                "result vs amplitude xcorr (2/3 wells better, 1/3 worse), so it stays "
                "off by default here, not enabled by default -- an available-but-not-"
                "proven-better optional slot, per the user's own honesty requirement."
            ),
            "results_with_regional_prior_window": ssl_results,
        },
        "stage2_switch_check_blind_fallback": blind_fallback_window_check,
        "stage2_donor_wells_restriction_check": donor_subset_check,
        "reference_numbers_from_regional_prior_module": {
            well: default_results[well]["metrics_vs_own_checkshot"]["mae_ms"] for well in WELLS
        },
        "elapsed_seconds": elapsed_s,
        "inputs": {
            "well_picks_sha256": sha256(WELL_PICKS),
            "checkshot_zip_sha256": sha256(VSP_ZIP),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nelapsed: {elapsed_s:.1f}s")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
