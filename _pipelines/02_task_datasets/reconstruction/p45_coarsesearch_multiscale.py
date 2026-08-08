#!/usr/bin/env python3
"""P45: coarse-to-fine multiscale search -- fix the coarse-search step itself
instead of the DTW refinement or borrowing a neighbor-well prior.

GitHub issue #1's diagnosed root cause (this session, prior scripts): I0's
coarse search (`normalized_xcorr_best_shift` / the coarse loop inside
`predict_well`) correlates the RAW, full-bandwidth synthetic against the RAW
real trace over the whole 0-3600ms axis in one shot. High-frequency
reflectivity content is locally repetitive (near-periodic packages of
reflectors), so a single-scale xcorr on raw amplitude is prone to locking
onto a side lobe exactly one (or more) cycle away from the true tie -- this
is the classic seismic "cycle skip" and is exactly what happened to 19SR
(I0 blind search: t0=612ms, MAE=1863.7ms against its own real checkshot;
picked a peak with xcorr=0.547 vs the *rejected* peak's own top_k score of
0.508 -- the true tie was not even in the reported top-3).

Multiscale idea (does NOT touch neighbor wells' checkshots -- purely a
signal-processing change to I0's own search, complementary to and stackable
with p45_well_tie_regional_prior.py's neighbor-well prior):

  Stage 1 (coarse, low-frequency, whole-axis): band-pass BOTH the synthetic
    and the real trace to a low band (default 2-15Hz, i.e. keep only the
    long-wavelength reflectivity structure -- individual thin reflectors
    average out, but the gross formation-scale impedance trend survives).
    Run the SAME coarse xcorr + non-max-suppression peak search I0 already
    uses, but on the smoothed pair, over the SAME full T0_SEARCH_MS axis.
    Low-frequency correlation has far fewer, farther-apart side lobes than
    full-bandwidth correlation, so a genuine cycle skip in the raw-amplitude
    search should mostly disappear here; we only trust the WINNING REGION
    (not the exact ms), by design.

  Stage 2 (fine, full-bandwidth, local window): take the stage-1 coarse
    center and re-run I0's ORIGINAL, UNMODIFIED predict_well() (raw-
    amplitude xcorr + slope-constrained DTW) restricted to a
    +/-FINE_HALF_WIDTH_MS window around that center -- i.e. call
    predict_well(..., t0_candidates=narrowed_range) exactly like
    p45_well_tie_regional_prior.py does for its neighbor-well prior, just
    with a self-supplied (not neighbor-borrowed) center.

Evaluation protocol (matches p45_well_tie_regional_prior.py /
p45_regional_prior_st10010_robustness.py so numbers are directly
comparable): leave-one-well-out is not needed here (nothing is borrowed
from other wells at all -- the coarse pass only uses the well's own sonic-
derived synthetic and its own real trace), so all 3 ST0202 wells are
evaluated directly against I0's existing blind-search summary, then the
identical multiscale method is rerun on ST10010 (via
p45_well_tie_st10010_robustness.ST10010Volume/build_index) and the two
surveys' results are compared for cross-source t0 consistency, exactly as
p45_regional_prior_st10010_robustness.py does for the neighbor-prior fix.

Honesty: this script reports whatever numbers come out, including the case
where the multiscale coarse stage still lands on the wrong lobe (e.g. because
2-15Hz is itself aliased/periodic for a given well), and does not narrow
FINE_HALF_WIDTH_MS or relax the ambiguity gate to manufacture a pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, PICK_NAME, WELL_PICKS, T0_SEARCH_MS,
    predict_well, metrics, parse_checkshots, parse_well_picks_full,
    load_well_curve, md_to_tvdss, sonic_reflectivity_and_time,
    estimate_wavelet_frequency, build_synthetic,
)
from p45_well_tie_st10010_robustness import ST10010_SEGY, build_index, ST10010Volume  # noqa: E402
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_coarsesearch_multiscale"
I0_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"
ST10010_BLIND_SUMMARY = HERE / "_outputs/p45_well_tie_st10010_robustness/summary.json"
REGIONAL_PRIOR_ST10010_SUMMARY = HERE / "_outputs/p45_regional_prior_st10010_robustness/summary.json"

# Stage 1 (coarse) band: only long-wavelength reflectivity structure.
COARSE_LOW_HZ = 2.0
COARSE_HIGH_HZ = 15.0
COARSE_FILTER_ORDER = 4
COARSE_STEP_MS = 16.0  # coarser than I0's 4ms default -- stage 1 only needs a region, not a point
COARSE_NMS_GAP_MS = 40.0  # same non-max-suppression gap I0 uses

# Stage 2 (fine): re-run I0's UNMODIFIED full-bandwidth search+DTW, narrowed
# to this window around the stage-1 coarse center. 200ms == I0's own
# DTW_WINDOW_MS, i.e. the same local scale I0 already trusts its DTW step to
# operate within, rather than an arbitrary new number.
FINE_HALF_WIDTH_MS = 200.0
FINE_STEP_MS = 4.0  # same step as I0's own T0_SEARCH_MS default -- no unfair resolution advantage


def bandpass_filter(trace: np.ndarray, dt_s: float, low_hz: float, high_hz: float,
                     order: int = COARSE_FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth band-pass. Falls back to a plain mean-removed
    signal if the trace is too short for filtfilt's padding requirement."""
    nyquist = 0.5 / dt_s
    low = max(low_hz / nyquist, 1e-4)
    high = min(high_hz / nyquist, 0.999)
    b, a = butter(order, [low, high], btype="band")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(trace) <= padlen:
        return trace - trace.mean()
    return filtfilt(b, a, trace)


def multiscale_coarse_region(well: str, index: dict, get_trace,
                              low_hz: float = COARSE_LOW_HZ, high_hz: float = COARSE_HIGH_HZ) -> dict:
    """Stage 1: band-passed whole-axis xcorr search. Returns the winning
    region (top-3 NMS peaks, same rule I0 uses) but the caller is only meant
    to trust the WINNING CENTER, not its exact ms value. low_hz/high_hz are
    parameterized (not hardcoded to the module defaults) so
    band_sensitivity_diagnostic() can sweep them without duplicating this
    function."""
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
    real_smooth = bandpass_filter(real_trace, dt_s, low_hz, high_hz)

    lo_bound = float(sample_ms.min())
    hi_bound = float(sample_ms.max())
    candidates = T0_SEARCH_MS[(T0_SEARCH_MS >= lo_bound) & (T0_SEARCH_MS <= hi_bound)]
    if COARSE_STEP_MS > (T0_SEARCH_MS[1] - T0_SEARCH_MS[0]):
        candidates = np.arange(max(lo_bound, candidates[0]), min(hi_bound, candidates[-1]) + 1e-9, COARSE_STEP_MS)

    scored: list[tuple[float, float]] = []
    for t0 in candidates:
        synthetic = build_synthetic(physics["reflectivity"], physics["relative_twt_ms"], t0, sample_ms, wavelet_freq)
        window_mask = (sample_ms >= t0 - 2) & (sample_ms <= t0 + span_ms + 2)
        idx = np.where(window_mask)[0]
        if len(idx) < 8:
            continue
        synthetic_smooth = bandpass_filter(synthetic, dt_s, low_hz, high_hz)
        syn_win = synthetic_smooth[idx]
        real_win = real_smooth[idx]
        s = syn_win - syn_win.mean()
        r = real_win - real_win.mean()
        denom = np.linalg.norm(s) * np.linalg.norm(r)
        score = float(np.dot(s, r) / denom) if denom > 0 else -1.0
        scored.append((float(t0), score))

    if not scored:
        raise RuntimeError(f"{well}: no valid coarse candidates on this survey's time axis")

    scored.sort(key=lambda p: p[1], reverse=True)
    peaks: list[tuple[float, float]] = []
    for t0, score in scored:
        if all(abs(t0 - p[0]) > COARSE_NMS_GAP_MS for p in peaks):
            peaks.append((t0, score))
        if len(peaks) >= 3:
            break
    best_t0, best_score = peaks[0]
    second_score = peaks[1][1] if len(peaks) > 1 else float("nan")
    coarse_ambiguity_gap = best_score - second_score if len(peaks) > 1 else float("nan")

    return {
        "well": well,
        "coarse_center_t0_ms": best_t0,
        "coarse_score": best_score,
        "coarse_top_k_peaks": peaks,
        "coarse_ambiguity_gap": coarse_ambiguity_gap,
        "wavelet_freq_hz": wavelet_freq,
        "span_ms": span_ms,
        "own_log_depth_tvdss_range": [float(tvdss.min()), float(tvdss.max())],
        "inline": int(il),
        "crossline": int(xl),
        "lo_bound_ms": lo_bound,
        "hi_bound_ms": hi_bound,
    }


def approx_true_t0_ms(tvdss: np.ndarray, checkshots: dict, well: str) -> float:
    """Diagnostic-only ground truth reference: the well's own real checkshot
    interpolated at the TVDSS depth of the TOP of its own valid DT/RHOB
    interval, i.e. where relative_twt_ms==0 so predicted_twt(top) ~= t0
    (ignoring the small nonlinear DTW warp). Used ONLY to report how far a
    coarse candidate's pick is from the true region in band_sensitivity_
    diagnostic(); never fed back into any search."""
    depth_true, twt_true = checkshots[well]
    top_tvdss = float(tvdss.min())
    return float(np.interp(top_tvdss, depth_true, twt_true))


# small set of alternative low-pass bands, tried ONLY as an honesty check on
# whether 2-15Hz specifically was a mistuned choice, or whether "coarse
# search via a low-pass band" fails to generalize across wells regardless of
# the exact band. All results are reported; none is smuggled in as "the"
# shipped method (that stays COARSE_LOW_HZ/COARSE_HIGH_HZ = 2-15Hz, per the
# task's own suggested default).
BAND_SWEEP = [(2.0, 15.0), (1.0, 10.0), (2.0, 20.0), (2.0, 25.0), (5.0, 20.0), (3.0, 12.0)]


def band_sensitivity_diagnostic(index: dict, get_trace, checkshots: dict) -> dict:
    out: dict = {}
    for well in WELLS:
        curve = load_well_curve(well)
        picks = parse_well_picks_full(WELL_PICKS)[PICK_NAME[well]]
        tvdss = md_to_tvdss(curve["depth_md"], picks)
        true_t0 = approx_true_t0_ms(tvdss, checkshots, well)
        per_band = {}
        for lo, hi in BAND_SWEEP:
            coarse = multiscale_coarse_region(well, index, get_trace, low_hz=lo, high_hz=hi)
            err = coarse["coarse_center_t0_ms"] - true_t0
            per_band[f"{lo:g}-{hi:g}Hz"] = {
                "low_hz": lo, "high_hz": hi,
                "coarse_center_t0_ms": coarse["coarse_center_t0_ms"],
                "coarse_score": coarse["coarse_score"],
                "err_vs_checkshot_reference_ms": err,
                "within_fine_half_width": bool(abs(err) <= FINE_HALF_WIDTH_MS),
            }
        out[well] = {"approx_true_t0_ms_from_checkshot": true_t0, "bands": per_band}
        print(f"  [band sweep] {well}: true_t0~={true_t0:.0f}ms  " +
              "  ".join(f"{k}: t0={v['coarse_center_t0_ms']:.0f}ms err={v['err_vs_checkshot_reference_ms']:+.0f}ms"
                         for k, v in per_band.items()))
    n_wells = len(WELLS)
    any_band_all_wells_within = any(
        all(out[well]["bands"][f"{lo:g}-{hi:g}Hz"]["within_fine_half_width"] for well in WELLS)
        for lo, hi in BAND_SWEEP
    )
    out["_verdict"] = {
        "any_single_band_lands_all_3_wells_within_fine_half_width_ms": bool(any_band_all_wells_within),
        "fine_half_width_ms": FINE_HALF_WIDTH_MS,
    }
    return out


def fine_window_candidates(center_ms: float, lo_bound: float, hi_bound: float) -> np.ndarray:
    lo = max(lo_bound, center_ms - FINE_HALF_WIDTH_MS)
    hi = min(hi_bound, center_ms + FINE_HALF_WIDTH_MS)
    return np.arange(lo, hi, FINE_STEP_MS)


def run_coarse_to_fine(well: str, index: dict, get_trace) -> dict:
    coarse = multiscale_coarse_region(well, index, get_trace)
    candidates = fine_window_candidates(coarse["coarse_center_t0_ms"], coarse["lo_bound_ms"], coarse["hi_bound_ms"])
    pred = predict_well(well, index, get_trace, t0_candidates=candidates)
    return {"coarse": coarse, "fine_search_window_ms": [float(candidates[0]), float(candidates[-1])], "pred": pred}


def score_against_checkshot(well: str, pred: dict, checkshots: dict) -> dict:
    depth_true, twt_true = checkshots[well]
    pred_at_checkshot_depth = np.interp(
        depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
        left=np.nan, right=np.nan,
    )
    valid = np.isfinite(pred_at_checkshot_depth)
    valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
    return metrics(twt_true[valid], pred_at_checkshot_depth[valid])


def run_st0202(checkshots: dict) -> dict:
    index = load_index()
    i0_summary = json.loads(I0_SUMMARY.read_text(encoding="utf-8"))
    results = {}
    with SeismicVolume(index=index) as vol:
        for well in WELLS:
            out = run_coarse_to_fine(well, index, vol)
            pred = out["pred"]
            m = score_against_checkshot(well, pred, checkshots)
            i0_r = i0_summary["wells"][well]
            results[well] = {
                "coarse_center_t0_ms": out["coarse"]["coarse_center_t0_ms"],
                "coarse_score": out["coarse"]["coarse_score"],
                "coarse_top_k_peaks": out["coarse"]["coarse_top_k_peaks"],
                "coarse_ambiguity_gap": out["coarse"]["coarse_ambiguity_gap"],
                "fine_search_window_ms": out["fine_search_window_ms"],
                "chosen_t0_ms": pred["coarse_t0_ms"],
                "fine_xcorr": pred["coarse_xcorr"],
                "fine_top_k_peaks": pred["top_k_peaks"],
                "fine_ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "dtw_distance": pred["dtw_distance"],
                "metrics_multiscale": m,
                "i0_blind_t0_ms": i0_r["coarse_t0_ms"],
                "i0_blind_metrics": i0_r["metrics_vs_own_checkshot"],
                "mae_delta_vs_i0_blind_ms": m["mae_ms"] - i0_r["metrics_vs_own_checkshot"]["mae_ms"],
                "improved_vs_i0_blind": bool(m["mae_ms"] < i0_r["metrics_vs_own_checkshot"]["mae_ms"]),
                "avoided_i0s_cycle_jump": bool(
                    abs(out["coarse"]["coarse_center_t0_ms"] - i0_r["coarse_t0_ms"]) > COARSE_NMS_GAP_MS
                    and m["mae_ms"] < i0_r["metrics_vs_own_checkshot"]["mae_ms"]
                ),
            }
            print(f"{well}: coarse_center={out['coarse']['coarse_center_t0_ms']:.0f}ms "
                  f"(score={out['coarse']['coarse_score']:.3f}) -> fine window="
                  f"[{out['fine_search_window_ms'][0]:.0f},{out['fine_search_window_ms'][1]:.0f}]ms "
                  f"chosen_t0={pred['coarse_t0_ms']:.0f}ms MAE={m['mae_ms']:.1f}ms "
                  f"(I0 blind t0={i0_r['coarse_t0_ms']:.0f}ms MAE={i0_r['metrics_vs_own_checkshot']['mae_ms']:.1f}ms, "
                  f"delta={results[well]['mae_delta_vs_i0_blind_ms']:+.1f}ms, "
                  f"{'IMPROVED' if results[well]['improved_vs_i0_blind'] else 'NOT_IMPROVED'})")
    return results


def run_st10010(checkshots: dict) -> dict:
    st10010_index = build_index(ST10010_SEGY)
    results = {}
    blind_summary = json.loads(ST10010_BLIND_SUMMARY.read_text(encoding="utf-8")) if ST10010_BLIND_SUMMARY.exists() else None
    regional_summary = json.loads(REGIONAL_PRIOR_ST10010_SUMMARY.read_text(encoding="utf-8")) if REGIONAL_PRIOR_ST10010_SUMMARY.exists() else None
    with ST10010Volume(segy_path=ST10010_SEGY, index=st10010_index) as vol:
        for well in WELLS:
            try:
                out = run_coarse_to_fine(well, st10010_index, vol)
            except ValueError as exc:
                results[well] = {"status": "out_of_grid", "error": str(exc)}
                print(f"{well}: OUT OF GRID on ST10010 -- {exc}")
                continue
            pred = out["pred"]
            m = score_against_checkshot(well, pred, checkshots)

            entry = {
                "status": "evaluated",
                "coarse_center_t0_ms": out["coarse"]["coarse_center_t0_ms"],
                "coarse_score": out["coarse"]["coarse_score"],
                "fine_search_window_ms": out["fine_search_window_ms"],
                "chosen_t0_ms": pred["coarse_t0_ms"],
                "fine_xcorr": pred["coarse_xcorr"],
                "fine_ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "metrics_multiscale": m,
            }
            if blind_summary is not None and blind_summary["results"].get(well, {}).get("status") == "evaluated":
                blind_r = blind_summary["results"][well]
                entry["blind_search_t0_delta_ms_st0202_vs_st10010"] = blind_r["t0_delta_ms"]
            if regional_summary is not None and regional_summary["results"].get(well, {}).get("status") == "evaluated":
                reg_r = regional_summary["results"][well]
                entry["regional_prior_t0_delta_ms_st0202_vs_st10010"] = reg_r["t0_delta_ms"]
            results[well] = entry
            print(f"{well} (ST10010): coarse_center={out['coarse']['coarse_center_t0_ms']:.0f}ms -> "
                  f"chosen_t0={pred['coarse_t0_ms']:.0f}ms MAE={m['mae_ms']:.1f}ms "
                  f"reject_ambiguous={pred['reject_ambiguous']}")
    return results, {
        "time_range_ms": [float(st10010_index["samples_ms"].min()), float(st10010_index["samples_ms"].max())],
        "sample_interval_ms": float(st10010_index["samples_ms"][1] - st10010_index["samples_ms"][0]),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    print("=== Stage A: ST0202, multiscale coarse-to-fine vs I0 blind search ===")
    st0202_results = run_st0202(checkshots)

    print("\n=== Diagnostic: is 2-15Hz a mistuned band, or does low-pass coarse "
          "search fail across wells regardless of the exact band? ===")
    index_for_diag = load_index()
    with SeismicVolume(index=index_for_diag) as vol_diag:
        band_sensitivity = band_sensitivity_diagnostic(index_for_diag, vol_diag, checkshots)
    print(f"  any single band lands all 3 wells within +/-{FINE_HALF_WIDTH_MS:.0f}ms of the checkshot "
          f"reference: {band_sensitivity['_verdict']['any_single_band_lands_all_3_wells_within_fine_half_width_ms']}")

    print("\n=== Stage B: ST10010, same multiscale method, cross-source comparison ===")
    st10010_results, st10010_survey = run_st10010(checkshots)

    # cross-survey t0 consistency for the multiscale method itself
    cross_source_deltas = {}
    for well in WELLS:
        st0202_t0 = st0202_results[well]["chosen_t0_ms"]
        st10010_r = st10010_results.get(well, {})
        if st10010_r.get("status") != "evaluated":
            continue
        delta = st10010_r["chosen_t0_ms"] - st0202_t0
        cross_source_deltas[well] = delta
        st10010_r["multiscale_t0_delta_ms_st0202_vs_st10010"] = delta
        print(f"{well}: multiscale t0_delta(ST0202->ST10010)={delta:+.0f}ms  "
              f"[I0 blind reference: {st10010_r.get('blind_search_t0_delta_ms_st0202_vs_st10010', 'n/a')}]  "
              f"[regional-prior reference: {st10010_r.get('regional_prior_t0_delta_ms_st0202_vs_st10010', 'n/a')}]")

    n_improved = sum(1 for r in st0202_results.values() if r["improved_vs_i0_blind"])
    n_avoided_cycle_jump = sum(1 for r in st0202_results.values() if r["avoided_i0s_cycle_jump"])
    n_kept_not_ambiguous_st0202 = sum(1 for r in st0202_results.values() if not r["reject_ambiguous"])
    n_evaluated_st10010 = sum(1 for r in st10010_results.values() if r.get("status") == "evaluated")
    n_kept_not_ambiguous_st10010 = sum(
        1 for r in st10010_results.values() if r.get("status") == "evaluated" and not r["reject_ambiguous"]
    )
    max_abs_multiscale_cross_source_delta = (
        max(abs(v) for v in cross_source_deltas.values()) if cross_source_deltas else float("nan")
    )
    blind_reference_deltas = [
        abs(r["blind_search_t0_delta_ms_st0202_vs_st10010"]) for r in st10010_results.values()
        if "blind_search_t0_delta_ms_st0202_vs_st10010" in r
    ]
    regional_reference_deltas = [
        abs(r["regional_prior_t0_delta_ms_st0202_vs_st10010"]) for r in st10010_results.values()
        if "regional_prior_t0_delta_ms_st0202_vs_st10010" in r
    ]

    summary = {
        "method": "coarse_to_fine_multiscale_search: stage1 band-passed (2-15Hz) whole-axis xcorr picks a region, "
                   "stage2 reruns I0's unmodified raw-amplitude xcorr+slope-constrained-DTW restricted to "
                   "+/-200ms around that region; no neighbor-well data used at any stage",
        "params": {
            "coarse_band_hz": [COARSE_LOW_HZ, COARSE_HIGH_HZ],
            "coarse_filter_order": COARSE_FILTER_ORDER,
            "coarse_step_ms": COARSE_STEP_MS,
            "coarse_nms_gap_ms": COARSE_NMS_GAP_MS,
            "fine_half_width_ms": FINE_HALF_WIDTH_MS,
            "fine_step_ms": FINE_STEP_MS,
        },
        "st0202_results": st0202_results,
        "st10010_survey": st10010_survey,
        "st10010_results": st10010_results,
        "band_sensitivity_diagnostic": band_sensitivity,
        "verdict": {
            "st0202_n_wells_improved_vs_i0_blind_search": n_improved,
            "st0202_n_wells_avoided_i0s_cycle_jump": n_avoided_cycle_jump,
            "st0202_n_wells_kept_not_ambiguous": n_kept_not_ambiguous_st0202,
            "st10010_n_wells_evaluated": n_evaluated_st10010,
            "st10010_n_wells_kept_not_ambiguous": n_kept_not_ambiguous_st10010,
            "max_abs_t0_delta_ms_multiscale_st0202_vs_st10010": (
                None if np.isnan(max_abs_multiscale_cross_source_delta) else max_abs_multiscale_cross_source_delta
            ),
            "max_abs_t0_delta_ms_blind_search_reference": max(blind_reference_deltas) if blind_reference_deltas else None,
            "max_abs_t0_delta_ms_regional_prior_reference": max(regional_reference_deltas) if regional_reference_deltas else None,
            "any_single_band_lands_all_3_wells_within_fine_half_width_ms": (
                band_sensitivity["_verdict"]["any_single_band_lands_all_3_wells_within_fine_half_width_ms"]
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nST0202: {n_improved}/3 wells improved vs I0 blind search "
          f"({n_avoided_cycle_jump}/3 explicitly avoided a >40ms-different cycle jump AND improved); "
          f"{n_kept_not_ambiguous_st0202}/3 pass the ambiguity gate")
    print(f"ST10010: {n_evaluated_st10010}/3 evaluated, {n_kept_not_ambiguous_st10010}/3 pass the ambiguity gate")
    print(f"Cross-source max|t0_delta|: multiscale={summary['verdict']['max_abs_t0_delta_ms_multiscale_st0202_vs_st10010']}ms "
          f"vs blind-search reference={summary['verdict']['max_abs_t0_delta_ms_blind_search_reference']}ms "
          f"vs regional-prior reference={summary['verdict']['max_abs_t0_delta_ms_regional_prior_reference']}ms")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
