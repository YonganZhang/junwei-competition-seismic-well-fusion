#!/usr/bin/env python3
"""P45 coarse-search fix candidate: envelope correlation instead of raw-
amplitude correlation for the t0 coarse search.

Root cause already pinned down (see p45_well_tie_st10010_robustness.py,
p45_regional_prior_st10010_robustness.py, p45_dtw_window_sensitivity.py):
I0's coarse search (normalized_xcorr_best_shift / the scoring loop inside
predict_well in p45_well_tie_physics_baseline.py) scores candidate t0 by
normalized cross-correlation of RAW amplitude waveforms (synthetic vs real
trace). Raw amplitude xcorr is sensitive to phase/polarity mismatch between
the Ricker-wavelet synthetic and the real seismic wavelet (which is neither
zero-phase nor exactly Ricker-shaped), so the score surface can lock onto a
spurious peak a full cycle away from the true tie point -- a classic
seismic-to-well-tie failure mode. This is a plausible cause of 19SR's
ambiguity and of the up-to-2716ms ST0202-vs-ST10010 t0 drift.

Fix candidate: score candidate t0 using the ENVELOPE (np.abs(hilbert(x))) of
both synthetic and real trace windows instead of raw amplitude. Envelope
(instantaneous amplitude) is insensitive to phase and polarity reversal --
if the seismic response energy for a given reflector arrives at the right
time, the envelope should peak there regardless of wavelet phase, so
envelope xcorr should be less prone to a false one-cycle-away peak.

Everything else is held IDENTICAL to I0: candidate t0 iteration
(T0_SEARCH_MS or a caller-supplied narrower window), the build_synthetic
physics, the top-3 non-maximum-suppression peak picking (peaks >40ms apart
count as distinct lobes), the <0.05 ambiguity gate, and the downstream
slope-constrained DTW refinement (dtw_refine, unchanged, still refines
around whichever t0 the coarse search now picks). Only the SCORE used
inside the coarse-search loop changes from raw-amplitude normalized xcorr
to envelope normalized xcorr.

Evaluation protocol (matches the project's existing P45 protocol, unchanged
so results are directly comparable):
  1. ST0202, all 3 wells (19A/19BT2/19SR), blind full-axis T0_SEARCH_MS
     search -- same candidate range as I0's own blind search -- compare
     MAE and ambiguity-gate outcome directly against
     p45_well_tie_physics_baseline.py's summary.json (I0, raw-amplitude).
  2. ST10010 cross-source robustness, all 3 wells, same blind full-axis
     search (via p45_well_tie_st10010_robustness.py's build_index /
     ST10010Volume) -- compare the ST0202-vs-ST10010 t0 drift against I0's
     already-measured 2716ms max drift (raw amplitude, blind) and the
     regional-prior fix's 316ms max drift (with a narrowed search window).

Honesty note: if envelope correlation does not reduce t0 drift or MAE, or
makes some wells worse, this script reports the real numbers -- it does not
cherry-pick a favorable subset or loosen the ambiguity gate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, WELL_PICKS, PICK_NAME,
    T0_SEARCH_MS, DTW_WINDOW_MS,
    load_well_curve, parse_well_picks_full, md_to_tvdss,
    sonic_reflectivity_and_time, estimate_wavelet_frequency, build_synthetic,
    dtw_refine, metrics, parse_checkshots,
)
from p45_well_tie_st10010_robustness import ST10010_SEGY, build_index, ST10010Volume  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_coarsesearch_envelope_correlation"
I0_ST0202_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"
ST0202_BLIND_ST10010_SUMMARY = HERE / "_outputs/p45_well_tie_st10010_robustness/summary.json"
REGIONAL_PRIOR_ST10010_SUMMARY = HERE / "_outputs/p45_regional_prior_st10010_robustness/summary.json"


def envelope(x: np.ndarray) -> np.ndarray:
    """Instantaneous-amplitude envelope, phase/polarity-insensitive."""
    return np.abs(hilbert(np.asarray(x, dtype=np.float64)))


def predict_well_envelope(well: str, index: dict, get_trace, t0_candidates: np.ndarray | None = None) -> dict:
    """Identical to p45_well_tie_physics_baseline.predict_well EXCEPT the
    coarse-search score inside the t0 loop is normalized xcorr of the
    HILBERT ENVELOPE of synthetic vs real windows instead of raw amplitude.
    Candidate iteration, top-3 NMS peak picking (40ms lobe spacing), the
    ambiguity gate (<0.05 gap), and the downstream slope-constrained DTW
    refinement are unchanged copies of the I0 logic."""
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

    # coarse search: build synthetic at each candidate t0, score by
    # ENVELOPE normalized xcorr (phase/polarity-insensitive) instead of raw
    # amplitude. Keep ALL scores for top-K reporting and the ambiguity gate,
    # exactly like I0.
    scored: list[tuple[float, float]] = []
    synthetics: dict[float, np.ndarray] = {}
    for t0 in search_candidates:
        synthetic = build_synthetic(physics["reflectivity"], physics["relative_twt_ms"], t0, sample_ms, wavelet_freq)
        window_mask = (sample_ms >= t0 - 2) & (sample_ms <= t0 + span_ms + 2)
        idx = np.where(window_mask)[0]
        if len(idx) < 8:
            continue
        syn_win = synthetic[idx]
        real_win = real_trace[idx]
        syn_env = envelope(syn_win)
        real_env = envelope(real_win)
        s = syn_env - syn_env.mean()
        r = real_env - real_env.mean()
        denom = np.linalg.norm(s) * np.linalg.norm(r)
        score = float(np.dot(s, r) / denom) if denom > 0 else -1.0
        scored.append((float(t0), score))
        synthetics[float(t0)] = synthetic

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
    }


def _score_well(well: str, index: dict, vol, checkshots: dict, t0_candidates=None) -> tuple[dict, dict]:
    pred = predict_well_envelope(well, index, vol, t0_candidates=t0_candidates)
    depth_true, twt_true = checkshots[well]
    pred_at_checkshot_depth = np.interp(
        depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
        left=np.nan, right=np.nan,
    )
    valid = np.isfinite(pred_at_checkshot_depth)
    valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
    m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])
    return pred, m


def run_st0202(checkshots: dict) -> dict:
    sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))
    from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

    index = load_index()
    i0_summary = json.loads(I0_ST0202_SUMMARY.read_text(encoding="utf-8")) if I0_ST0202_SUMMARY.exists() else None

    results = {}
    with SeismicVolume(index=index) as vol:
        for well in WELLS:
            pred, m = _score_well(well, index, vol, checkshots)
            entry = {
                "coarse_t0_ms": pred["coarse_t0_ms"],
                "coarse_env_xcorr": pred["coarse_xcorr"],
                "top_k_peaks": pred["top_k_peaks"],
                "ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "dtw_distance": pred["dtw_distance"],
                "metrics_vs_own_checkshot": m,
            }
            if i0_summary is not None:
                i0_r = i0_summary["wells"][well]
                entry["i0_raw_amplitude_t0_ms"] = i0_r["coarse_t0_ms"]
                entry["i0_raw_amplitude_metrics"] = i0_r["metrics_vs_own_checkshot"]
                entry["i0_raw_amplitude_reject_ambiguous"] = i0_r["reject_ambiguous"]
                entry["mae_delta_vs_i0_ms"] = m["mae_ms"] - i0_r["metrics_vs_own_checkshot"]["mae_ms"]
                entry["improved_vs_i0"] = bool(m["mae_ms"] < i0_r["metrics_vs_own_checkshot"]["mae_ms"])
                entry["ambiguity_gate_improved"] = bool(
                    i0_r["reject_ambiguous"] and not pred["reject_ambiguous"]
                )
            results[well] = entry
            i0_note = ""
            if i0_summary is not None:
                i0_note = (f" | I0 raw-amp t0={entry['i0_raw_amplitude_t0_ms']:.0f}ms "
                           f"MAE={entry['i0_raw_amplitude_metrics']['mae_ms']:.1f}ms "
                           f"delta={entry['mae_delta_vs_i0_ms']:+.1f}ms "
                           f"{'IMPROVED' if entry['improved_vs_i0'] else 'NOT_IMPROVED'}")
            print(f"[ST0202] {well}: env_t0={pred['coarse_t0_ms']:.0f}ms env_xcorr={pred['coarse_xcorr']:.3f} "
                  f"gap={pred['ambiguity_gap']:.3f} reject={pred['reject_ambiguous']} "
                  f"MAE={m['mae_ms']:.2f}ms RMSE={m['rmse_ms']:.2f}ms{i0_note}")
    return results


def run_st10010(checkshots: dict) -> dict:
    st10010_index = build_index(ST10010_SEGY)
    blind_summary = (json.loads(ST0202_BLIND_ST10010_SUMMARY.read_text(encoding="utf-8"))
                      if ST0202_BLIND_ST10010_SUMMARY.exists() else None)
    prior_summary = (json.loads(REGIONAL_PRIOR_ST10010_SUMMARY.read_text(encoding="utf-8"))
                      if REGIONAL_PRIOR_ST10010_SUMMARY.exists() else None)

    results = {}
    with ST10010Volume(segy_path=ST10010_SEGY, index=st10010_index) as vol:
        for well in WELLS:
            try:
                pred_st10010, m_st10010 = _score_well(well, st10010_index, vol, checkshots)
            except ValueError as exc:
                results[well] = {"status": "out_of_grid", "error": str(exc)}
                print(f"[ST10010] {well}: OUT OF GRID -- {exc}")
                continue
            results[well] = {
                "status": "evaluated",
                "st10010_env_t0_ms": pred_st10010["coarse_t0_ms"],
                "st10010_env_xcorr": pred_st10010["coarse_xcorr"],
                "st10010_ambiguity_gap": pred_st10010["ambiguity_gap"],
                "st10010_reject_ambiguous": pred_st10010["reject_ambiguous"],
                "st10010_metrics": m_st10010,
            }
            print(f"[ST10010] {well}: env_t0={pred_st10010['coarse_t0_ms']:.0f}ms "
                  f"env_xcorr={pred_st10010['coarse_xcorr']:.3f} "
                  f"reject={pred_st10010['reject_ambiguous']} MAE={m_st10010['mae_ms']:.2f}ms")

    return results, blind_summary, prior_summary


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    print("=== ST0202: envelope-correlation coarse search vs I0 raw-amplitude (same blind T0_SEARCH_MS) ===")
    st0202_results = run_st0202(checkshots)

    print("\n=== ST10010: envelope-correlation coarse search, same blind T0_SEARCH_MS, cross-source drift ===")
    st10010_results, blind_summary, prior_summary = run_st10010(checkshots)

    cross_source = {}
    t0_deltas = []
    for well in WELLS:
        st0202_r = st0202_results.get(well, {})
        st10010_r = st10010_results.get(well, {})
        if st10010_r.get("status") != "evaluated" or "coarse_t0_ms" not in st0202_r:
            cross_source[well] = {"status": "incomplete"}
            continue
        t0_delta = st10010_r["st10010_env_t0_ms"] - st0202_r["coarse_t0_ms"]
        mae_delta = st10010_r["st10010_metrics"]["mae_ms"] - st0202_r["metrics_vs_own_checkshot"]["mae_ms"]
        gate_agrees = bool(st10010_r["st10010_reject_ambiguous"] == st0202_r["reject_ambiguous"])
        t0_deltas.append(abs(t0_delta))
        cross_source[well] = {
            "status": "evaluated",
            "st0202_env_t0_ms": st0202_r["coarse_t0_ms"],
            "st10010_env_t0_ms": st10010_r["st10010_env_t0_ms"],
            "t0_delta_ms": t0_delta,
            "st0202_mae_ms": st0202_r["metrics_vs_own_checkshot"]["mae_ms"],
            "st10010_mae_ms": st10010_r["st10010_metrics"]["mae_ms"],
            "mae_delta_ms": mae_delta,
            "gate_agrees": gate_agrees,
        }
        print(f"[cross-source] {well}: env t0_delta={t0_delta:+.0f}ms mae_delta={mae_delta:+.1f}ms "
              f"gate_agrees={gate_agrees}")

    max_abs_t0_delta_envelope = max(t0_deltas) if t0_deltas else float("nan")
    blind_raw_amp_max_t0_delta = (
        blind_summary["verdict"].get("max_abs_t0_delta_ms")
        if blind_summary is not None and "max_abs_t0_delta_ms" in blind_summary.get("verdict", {})
        else (max((abs(r["t0_delta_ms"]) for r in blind_summary["results"].values()
                    if r.get("status") == "evaluated"), default=None)
              if blind_summary is not None else None)
    )
    prior_max_t0_delta = (
        prior_summary["verdict"]["max_abs_t0_delta_ms_with_prior"] if prior_summary is not None else None
    )

    n_st0202_improved = sum(1 for r in st0202_results.values() if r.get("improved_vs_i0"))
    n_st0202_not_ambiguous = sum(1 for r in st0202_results.values() if not r.get("reject_ambiguous", True))
    n_gate_agrees = sum(1 for r in cross_source.values() if r.get("gate_agrees"))

    summary = {
        "method": "envelope_hilbert_normalized_xcorr_coarse_search_replacing_raw_amplitude_xcorr",
        "unchanged_from_i0": [
            "candidate_t0_iteration (T0_SEARCH_MS blind full-axis)",
            "top3_non_maximum_suppression (40ms lobe spacing)",
            "ambiguity_gate (<0.05 gap => reject_ambiguous)",
            "slope_constrained_dtw_refine (unchanged, operates on raw amplitude local windows)",
        ],
        "st0202_results": st0202_results,
        "st10010_results": st10010_results,
        "cross_source_st0202_vs_st10010": cross_source,
        "reference_numbers": {
            "i0_blind_raw_amplitude_max_abs_t0_delta_ms": blind_raw_amp_max_t0_delta,
            "regional_prior_max_abs_t0_delta_ms": prior_max_t0_delta,
        },
        "verdict": {
            "n_st0202_wells_improved_vs_i0_raw_amplitude": n_st0202_improved,
            "n_st0202_wells_not_ambiguous": n_st0202_not_ambiguous,
            "n_cross_source_wells_gate_agrees": n_gate_agrees,
            "max_abs_t0_delta_ms_envelope_blind": None if np.isnan(max_abs_t0_delta_envelope) else max_abs_t0_delta_envelope,
            "max_abs_t0_delta_ms_i0_blind_raw_amplitude_reference": blind_raw_amp_max_t0_delta,
            "max_abs_t0_delta_ms_regional_prior_reference": prior_max_t0_delta,
            "envelope_reduces_cross_source_drift_vs_i0_blind": (
                bool(max_abs_t0_delta_envelope < blind_raw_amp_max_t0_delta)
                if blind_raw_amp_max_t0_delta is not None and not np.isnan(max_abs_t0_delta_envelope)
                else None
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nST0202: {n_st0202_improved}/3 improved vs I0 raw-amplitude MAE; "
          f"{n_st0202_not_ambiguous}/3 not ambiguous")
    print(f"Cross-source: max|t0_delta| envelope={max_abs_t0_delta_envelope:.0f}ms "
          f"vs I0 blind raw-amplitude reference={blind_raw_amp_max_t0_delta}ms "
          f"vs regional-prior reference={prior_max_t0_delta}ms; "
          f"{n_gate_agrees}/3 wells' ambiguity gate agrees ST0202<->ST10010")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
