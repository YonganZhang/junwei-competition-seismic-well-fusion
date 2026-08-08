#!/usr/bin/env python3
"""P45: replace the coarse-search xcorr scoring signal with a band-pass
filtered version, instead of raw amplitude.

Diagnosed problem (see p45_well_tie_regional_prior.py, p45_well_tie_st10010
_robustness.py, p45_dtw_window_sensitivity.py): the coarse-search step in
predict_well() (p45_well_tie_physics_baseline.py) scores every candidate t0
by normalized cross-correlation of the RAW-AMPLITUDE synthetic trace against
the RAW-AMPLITUDE real trace. Raw amplitude carries broadband noise and
low-frequency drift that is unrelated to the reflectivity series the
synthetic actually encodes, so the correlation surface can have spurious
peaks a full seismic cycle away from the true tie point (19SR locks onto
t0=612ms blind-search on ST0202, when the checkshot-consistent tie is much
later -- MAE=1863.7ms). Because DTW refinement afterwards only searches a
+/-200ms window around whatever coarse t0 was chosen, a wrong coarse peak
cannot be corrected downstream -- this is the mechanism GitHub issue #1
flags as "锁定虚假相关峰".

Fix under test: band-pass filter (scipy.signal.butter + filtfilt, zero
phase) BOTH the synthetic and the real trace to a fixed seismic band
(8-60 Hz by default) before computing the normalized cross-correlation
score in the coarse search. This suppresses out-of-band noise/drift that
create false correlation peaks, while leaving the actual reflectivity-band
content (which both synthetic and real trace should genuinely share) intact.
Everything else -- top-3 non-maximum-suppressed peak selection, the 0.05
ambiguity gate, and the slope-constrained DTW refinement step -- is
UNCHANGED and still operates on the ORIGINAL (unfiltered) waveforms, exactly
like the baseline. Only the coarse-search *scoring* signal changes.

Evaluation protocol (matches the project's established honest-comparison
pattern):
  1. ST0202, all 3 wells (19A/19BT2/19SR): run the EXISTING raw-amplitude
     predict_well() and the NEW predict_well_bandpass(), both with the
     SAME full blind t0 search range (T0_SEARCH_MS, 0-3600ms step 4ms; no
     leave-one-well-out is needed here because neither method reads any
     well's own checkshot -- coarse search only touches sonic/density/
     seismic). Compare coarse t0, ambiguity gap, MAE vs each well's own
     real VSP checkshot, and specifically whether 19SR's periodic-jump
     failure (MAE=1863.7ms at t0=612ms) is avoided.
  2. ST10010, same 3 wells, same blind full-range search, both methods,
     via p45_well_tie_st10010_robustness.ST10010Volume/build_index.
     Compare each method's cross-source |t0_ST0202 - t0_ST10010| against
     the two already-reported reference numbers: 2716ms (raw-amplitude
     blind search, worst well) and 316ms (regional-prior-narrowed search,
     worst well, from p45_regional_prior_st10010_robustness.py).

No threshold is loosened and no favorable subset is cherry-picked to
manufacture a pass; every well's number is reported, improved or not.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, PICK_NAME, WELL_PICKS,
    T0_SEARCH_MS, DTW_WINDOW_MS,
    load_well_curve, parse_well_picks_full, md_to_tvdss,
    sonic_reflectivity_and_time, estimate_wavelet_frequency, build_synthetic,
    dtw_refine, predict_well, metrics, parse_checkshots,
)
from p45_well_tie_st10010_robustness import ST10010Volume, build_index, ST10010_SEGY  # noqa: E402
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_coarsesearch_bandpass_correlation"

BANDPASS_LOW_HZ = 8.0
BANDPASS_HIGH_HZ = 60.0
BANDPASS_ORDER = 4

# already-established reference numbers from prior honest diagnostics, used
# ONLY as comparison points -- never recomputed by relaxing this script's
# own methodology.
REF_BLIND_SEARCH_MAX_ABS_T0_DELTA_MS = 2716.0  # p45_well_tie_st10010_robustness (19SR)
REF_REGIONAL_PRIOR_MAX_ABS_T0_DELTA_MS = 316.0  # p45_regional_prior_st10010_robustness (19SR)


def design_bandpass(dt_s: float, low_hz: float, high_hz: float, order: int = BANDPASS_ORDER):
    """Butterworth band-pass coefficients for the survey's own sample rate.
    Returns None if the requested band is not representable (e.g. Nyquist
    too low), in which case callers fall back to unfiltered (raw) signal."""
    fs = 1.0 / dt_s
    nyq = 0.5 * fs
    low = max(low_hz, 0.1) / nyq
    high = min(high_hz, nyq * 0.98) / nyq
    if not (0.0 < low < high < 1.0):
        return None
    return butter(order, [low, high], btype="band")


def apply_bandpass(trace: np.ndarray, ba) -> np.ndarray:
    trace64 = np.asarray(trace, dtype=np.float64)
    if ba is None:
        return trace64.copy()
    b, a = ba
    return filtfilt(b, a, trace64)


def predict_well_bandpass(well: str, index: dict, get_trace, t0_candidates: np.ndarray | None = None,
                           band_low_hz: float = BANDPASS_LOW_HZ, band_high_hz: float = BANDPASS_HIGH_HZ) -> dict:
    """Drop-in replacement for p45_well_tie_physics_baseline.predict_well
    that band-pass filters synthetic and real trace before the coarse-search
    xcorr scoring. All other steps (peak NMS, ambiguity gate, DTW
    refinement on the ORIGINAL unfiltered waveforms) are identical to the
    baseline predict_well()."""
    search_candidates = T0_SEARCH_MS if t0_candidates is None else t0_candidates
    curve = load_well_curve(well)
    picks = parse_well_picks_full(WELL_PICKS)[PICK_NAME[well]]
    tvdss = md_to_tvdss(curve["depth_md"], picks)
    physics = sonic_reflectivity_and_time(curve["depth_md"], curve["dt"], curve["rhob"])

    order = np.argsort(picks.md)
    x = float(np.interp(np.median(curve["depth_md"]), picks.md[order], picks.easting[order]))
    y = float(np.interp(np.median(curve["depth_md"]), picks.md[order], picks.northing[order]))
    il, xl = get_trace.utm_to_il_xl(x, y)
    real_trace = get_trace.get_trace(il, xl)
    sample_ms = index["samples_ms"]
    dt_s = float(sample_ms[1] - sample_ms[0]) / 1000.0

    wavelet_freq = estimate_wavelet_frequency(real_trace, dt_s)
    span_ms = physics["relative_twt_ms"][-1]

    ba = design_bandpass(dt_s, band_low_hz, band_high_hz)
    real_trace_filt = apply_bandpass(real_trace, ba)

    # coarse search: build RAW synthetic per candidate t0 (unchanged), but
    # score the candidate by band-pass filtered xcorr instead of raw xcorr.
    scored: list[tuple[float, float]] = []
    synthetics: dict[float, np.ndarray] = {}
    for t0 in search_candidates:
        synthetic = build_synthetic(physics["reflectivity"], physics["relative_twt_ms"], t0, sample_ms, wavelet_freq)
        window_mask = (sample_ms >= t0 - 2) & (sample_ms <= t0 + span_ms + 2)
        idx = np.where(window_mask)[0]
        if len(idx) < 8:
            continue
        synthetic_filt = apply_bandpass(synthetic, ba)
        syn_win = synthetic_filt[idx]
        real_win = real_trace_filt[idx]
        s = syn_win - syn_win.mean()
        r = real_win - real_win.mean()
        denom = np.linalg.norm(s) * np.linalg.norm(r)
        score = float(np.dot(s, r) / denom) if denom > 0 else -1.0
        scored.append((float(t0), score))
        synthetics[float(t0)] = synthetic  # keep the RAW synthetic for downstream DTW

    scored.sort(key=lambda p: p[1], reverse=True)
    peaks: list[tuple[float, float]] = []
    for t0, score in scored:
        if all(abs(t0 - p[0]) > 40.0 for p in peaks):
            peaks.append((t0, score))
        if len(peaks) >= 3:
            break
    best_t0, best_score = peaks[0]
    second_score = peaks[1][1] if len(peaks) > 1 else float("nan")
    ambiguity_gap = best_score - second_score if len(peaks) > 1 else float("nan")
    reject_ambiguous = bool(len(peaks) > 1 and ambiguity_gap < 0.05)

    # DTW refinement: identical to baseline, on ORIGINAL unfiltered waveforms.
    synthetic = synthetics[best_t0]
    window_mask = (sample_ms >= best_t0 - DTW_WINDOW_MS) & (sample_ms <= best_t0 + span_ms + DTW_WINDOW_MS)
    idx = np.where(window_mask)[0]
    syn_local = synthetic[idx]
    real_local = real_trace[idx]
    path, dtw_dist = dtw_refine(syn_local, real_local)

    synth_sample_ms = sample_ms[idx]
    path = np.array(path)
    dtw_map_from = synth_sample_ms[path[:, 0]]
    dtw_map_to = synth_sample_ms[path[:, 1]]
    order2 = np.argsort(dtw_map_from)
    depth_twt_raw = physics["relative_twt_ms"] + best_t0
    predicted_twt = np.interp(depth_twt_raw, dtw_map_from[order2], dtw_map_to[order2])

    return {
        "well": well,
        "depth_md": curve["depth_md"],
        "depth_tvdss": tvdss,
        "predicted_twt_ms": predicted_twt,
        "coarse_t0_ms": best_t0,
        "coarse_xcorr": best_score,
        "top_k_peaks": peaks,
        "ambiguity_gap": ambiguity_gap,
        "reject_ambiguous": reject_ambiguous,
        "dtw_distance": dtw_dist,
        "wavelet_freq_hz": wavelet_freq,
        "inline": int(il),
        "crossline": int(xl),
        "band_hz": [band_low_hz, band_high_hz],
    }


def score_against_checkshot(well: str, pred: dict, checkshots: dict) -> dict:
    depth_true, twt_true = checkshots[well]
    pred_at_checkshot_depth = np.interp(
        depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
        left=np.nan, right=np.nan,
    )
    valid = np.isfinite(pred_at_checkshot_depth)
    valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
    return metrics(twt_true[valid], pred_at_checkshot_depth[valid])


def run_survey(survey_name: str, index: dict, vol, checkshots: dict) -> dict:
    """For every well, run BOTH the raw-amplitude baseline and the new
    band-pass-scored coarse search, full blind T0_SEARCH_MS range, and
    score both against the well's own real checkshot."""
    out: dict = {}
    for well in WELLS:
        t0 = time.time()
        pred_raw = predict_well(well, index, vol)
        t1 = time.time()
        pred_bp = predict_well_bandpass(well, index, vol)
        t2 = time.time()
        m_raw = score_against_checkshot(well, pred_raw, checkshots)
        m_bp = score_against_checkshot(well, pred_bp, checkshots)
        out[well] = {
            "raw_amplitude": {
                "coarse_t0_ms": pred_raw["coarse_t0_ms"],
                "coarse_xcorr": pred_raw["coarse_xcorr"],
                "top_k_peaks": pred_raw["top_k_peaks"],
                "ambiguity_gap": pred_raw["ambiguity_gap"],
                "reject_ambiguous": pred_raw["reject_ambiguous"],
                "metrics_vs_own_checkshot": m_raw,
                "wall_time_s": t1 - t0,
            },
            "bandpass_correlation": {
                "coarse_t0_ms": pred_bp["coarse_t0_ms"],
                "coarse_xcorr": pred_bp["coarse_xcorr"],
                "top_k_peaks": pred_bp["top_k_peaks"],
                "ambiguity_gap": pred_bp["ambiguity_gap"],
                "reject_ambiguous": pred_bp["reject_ambiguous"],
                "metrics_vs_own_checkshot": m_bp,
                "band_hz": pred_bp["band_hz"],
                "wall_time_s": t2 - t1,
            },
            "mae_delta_ms_bandpass_minus_raw": m_bp["mae_ms"] - m_raw["mae_ms"],
            "bandpass_improved": bool(m_bp["mae_ms"] < m_raw["mae_ms"]),
        }
        print(f"[{survey_name}] {well}: RAW t0={pred_raw['coarse_t0_ms']:.0f}ms MAE={m_raw['mae_ms']:.1f}ms "
              f"gap={pred_raw['ambiguity_gap']:.3f} reject={pred_raw['reject_ambiguous']}  |  "
              f"BANDPASS t0={pred_bp['coarse_t0_ms']:.0f}ms MAE={m_bp['mae_ms']:.1f}ms "
              f"gap={pred_bp['ambiguity_gap']:.3f} reject={pred_bp['reject_ambiguous']}  "
              f"({'IMPROVED' if out[well]['bandpass_improved'] else 'NOT IMPROVED'}, "
              f"delta={out[well]['mae_delta_ms_bandpass_minus_raw']:+.1f}ms)")
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index_st0202 = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    print("=== ST0202: raw-amplitude baseline vs band-pass-scored coarse search (blind full-range) ===")
    with SeismicVolume(index=index_st0202) as vol:
        st0202_results = run_survey("ST0202", index_st0202, vol, checkshots)

    print("\n=== ST10010: same comparison ===")
    index_st10010 = build_index(ST10010_SEGY)
    with ST10010Volume(segy_path=ST10010_SEGY, index=index_st10010) as vol:
        st10010_results = run_survey("ST10010", index_st10010, vol, checkshots)

    # cross-source robustness: |t0_ST0202 - t0_ST10010| per method
    cross_source = {}
    for well in WELLS:
        t0_raw_st0202 = st0202_results[well]["raw_amplitude"]["coarse_t0_ms"]
        t0_raw_st10010 = st10010_results[well]["raw_amplitude"]["coarse_t0_ms"]
        t0_bp_st0202 = st0202_results[well]["bandpass_correlation"]["coarse_t0_ms"]
        t0_bp_st10010 = st10010_results[well]["bandpass_correlation"]["coarse_t0_ms"]
        cross_source[well] = {
            "raw_amplitude_abs_t0_delta_ms": abs(t0_raw_st0202 - t0_raw_st10010),
            "bandpass_abs_t0_delta_ms": abs(t0_bp_st0202 - t0_bp_st10010),
        }
        cross_source[well]["bandpass_reduces_t0_delta"] = bool(
            cross_source[well]["bandpass_abs_t0_delta_ms"] < cross_source[well]["raw_amplitude_abs_t0_delta_ms"]
        )

    max_raw_delta = max(v["raw_amplitude_abs_t0_delta_ms"] for v in cross_source.values())
    max_bp_delta = max(v["bandpass_abs_t0_delta_ms"] for v in cross_source.values())
    worst_raw_well = max(cross_source, key=lambda w: cross_source[w]["raw_amplitude_abs_t0_delta_ms"])
    worst_bp_well = max(cross_source, key=lambda w: cross_source[w]["bandpass_abs_t0_delta_ms"])

    n_st0202_improved = sum(1 for r in st0202_results.values() if r["bandpass_improved"])
    n_st10010_improved = sum(1 for r in st10010_results.values() if r["bandpass_improved"])
    sr19_st0202_avoids_jump = bool(
        st0202_results["19SR"]["bandpass_correlation"]["metrics_vs_own_checkshot"]["mae_ms"]
        < st0202_results["19SR"]["raw_amplitude"]["metrics_vs_own_checkshot"]["mae_ms"]
    )

    summary = {
        "method": "bandpass_filter_synthetic_and_real_trace_before_coarse_xcorr_scoring",
        "band_hz": [BANDPASS_LOW_HZ, BANDPASS_HIGH_HZ],
        "bandpass_order": BANDPASS_ORDER,
        "t0_search_range_ms_used": [float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1])],
        "st0202": st0202_results,
        "st10010": st10010_results,
        "cross_source_t0_robustness": cross_source,
        "reference_numbers": {
            "blind_search_max_abs_t0_delta_ms": REF_BLIND_SEARCH_MAX_ABS_T0_DELTA_MS,
            "regional_prior_max_abs_t0_delta_ms": REF_REGIONAL_PRIOR_MAX_ABS_T0_DELTA_MS,
            "note": "from p45_well_tie_st10010_robustness.py and "
                    "p45_regional_prior_st10010_robustness.py, reported here for "
                    "comparison only -- not recomputed by this script",
        },
        "verdict": {
            "n_st0202_wells_improved_by_bandpass": n_st0202_improved,
            "n_st10010_wells_improved_by_bandpass": n_st10010_improved,
            "well_19SR_st0202_periodic_jump_avoided": sr19_st0202_avoids_jump,
            "max_abs_t0_delta_ms_raw_amplitude_this_run": max_raw_delta,
            "max_abs_t0_delta_ms_bandpass_this_run": max_bp_delta,
            "worst_well_raw_amplitude": worst_raw_well,
            "worst_well_bandpass": worst_bp_well,
            "bandpass_beats_blind_search_reference_2716ms": bool(max_bp_delta < REF_BLIND_SEARCH_MAX_ABS_T0_DELTA_MS),
            "bandpass_beats_regional_prior_reference_316ms": bool(max_bp_delta < REF_REGIONAL_PRIOR_MAX_ABS_T0_DELTA_MS),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nST0202: {n_st0202_improved}/3 wells improved by bandpass scoring "
          f"(19SR periodic-jump avoided: {sr19_st0202_avoids_jump})")
    print(f"ST10010: {n_st10010_improved}/3 wells improved by bandpass scoring")
    print(f"cross-source max|t0 delta|: raw_amplitude={max_raw_delta:.0f}ms (worst={worst_raw_well}) "
          f"vs bandpass={max_bp_delta:.0f}ms (worst={worst_bp_well})  "
          f"[reference: blind_search=2716ms, regional_prior=316ms]")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
