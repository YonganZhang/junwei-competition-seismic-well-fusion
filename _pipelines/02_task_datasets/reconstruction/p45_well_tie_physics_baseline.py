#!/usr/bin/env python3
"""P45 I0: physics-first well-seismic tie without checkshot/time-depth table.

Pipeline per well (uses ONLY sonic(DT) + density(RHOB) + trajectory + real
seismic; never reads the well's own checkshot as input):

  1. Load DT/RHOB on the well's own depth grid (MD), restrict to the
     contiguous non-NaN coverage interval.
  2. Convert MD -> TVDSS using the SAME official Well_picks_Volve_v1.dat file
     already used by step_04 weak tie (MD/TVDSS pairs at formation tops,
     linearly interpolated) -- this is a depth-type conversion, not a
     time/checkshot leak.
  3. Sonic integration: velocity_m_s = 1e6 * FT_TO_M / DT(us/ft); cumulative
     one-way time from the top of the valid interval gives a RELATIVE TWT
     curve with an unknown absolute static shift (t0) -- this is real
     physics, no checkshot used, and matches the well-known "sonic can't fix
     the absolute start time" limitation.
  4. Reflectivity: bruges.reflection.acoustic_reflectivity(vp, rho) in depth,
     resampled onto a regular two-way time grid at the survey sample
     interval using a candidate t0.
  5. Wavelet: Ricker wavelet whose dominant frequency is estimated from the
     REAL seismic trace's amplitude spectrum near the well (no checkshot).
  6. Synthetic = reflectivity (*) wavelet.
  7. Coarse search over candidate t0 via normalized cross-correlation against
     the real well-location seismic trace, then a slope-constrained DTW
     refinement in a local window around the best t0 (bounded stretch
     factor, so it cannot free-warp into an unphysical velocity -- see
     GitHub issue #1's own list of pitfalls).
  8. The DTW path maps depth samples -> TWT; report the predicted MD->TWT
     curve.

Ground truth for scoring is each well's own real VSP checkshot
(Volve_Seismic_VSP.zip), read only AFTER prediction, purely for evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

import h5py
import numpy as np
from bruges.filters import ricker
from bruges.reflection import acoustic_reflectivity
from dtaidistance import dtw

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
RAW_PROJECT_ROOT = Path(os.environ.get("VOLVE_RAW_PROJECT_ROOT", str(PROJECT_ROOT)))

WELL_LOGS_H5 = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/well_logs_clean.h5"
SEISMIC_INDEX = PROJECT_ROOT / "_pipelines/01_common_preprocess/outputs/seismic_index.npz"
WELL_PICKS = (
    RAW_PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_interp/Geophysical_Interpretations/"
    "Wells/Well_picks_Volve_v1.dat"
)
VSP_ZIP = RAW_PROJECT_ROOT / "_sandbox/volve_data/Volve_Seismic_VSP.zip"
SEGY_PATH = (
    RAW_PROJECT_ROOT
    / "_sandbox/volve_data/_extracted_seismic/ST0202/Stacks/"
    "ST0202R08_PZ_PSDM_FULL_OFFSET_PP_TIME.MIG_FIN.POST_STACK.3D.JS-017534.segy"
)
OUTPUT_DIR = HERE / "_outputs/p45_well_tie_physics_baseline"

FT_TO_M = 0.3048
WELLS = ("19A", "19BT2", "19SR")
LOG_KEY = {"19A": "15_9-19 A", "19BT2": "15_9-19 BT2", "19SR": "15_9-19 SR"}
PICK_NAME = {
    "19A": "NO 15/9-19 A",
    "19BT2": "NO 15/9-19 BT2",
    "19SR": "NO 15/9-19 SR",
}
CHECKSHOT_MEMBER = {
    "19A": "VSP/Checkshots/checkshot_15_9_19A.txt",
    "19BT2": "VSP/Checkshots/checkshot_15_9_19BT2.txt",
    "19SR": "VSP/Checkshots/checkshot_15_9_19_SR.txt",
}
# candidate static-shift search window (ms) and DTW local-slope bound.
T0_SEARCH_MS = np.arange(0.0, 3600.0, 4.0)
DTW_WINDOW_MS = 200.0  # +/- window around best coarse t0 refined by DTW
MAX_LOCAL_STRETCH = 1.20  # forbid >20% local velocity distortion (issue #1 pitfall)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WellPicks:
    md: np.ndarray
    tvdss: np.ndarray
    easting: np.ndarray
    northing: np.ndarray


def parse_well_picks_full(path: Path) -> dict[str, WellPicks]:
    """Same fixed-width parser as step_04, but also keeps TVDSS (col 6)."""
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    spans: list[tuple[int, int]] | None = None
    wells: dict[str, list[tuple[float, float, float, float]]] = {}
    for line in lines:
        if re.match(r"^\s*-{5,}", line):
            spans = [(m.start(), m.end()) for m in re.finditer(r"-+", line)]
            continue
        if spans is None or not line.strip() or line.startswith("Well NO") or line.lstrip().startswith("Well name"):
            continue
        cols = [line[s:e].strip() for s, e in spans]
        if len(cols) < 12:
            continue
        well_name = cols[0]
        try:
            md = float(cols[4])
            tvdss = abs(float(cols[6]))
            easting = float(cols[10])
            northing = float(cols[11])
        except ValueError:
            continue
        wells.setdefault(well_name, []).append((md, tvdss, easting, northing))
    out: dict[str, WellPicks] = {}
    for well, rows in wells.items():
        rows.sort(key=lambda r: r[0])
        md = np.array([r[0] for r in rows])
        _, idx = np.unique(md, return_index=True)
        rows_arr = np.array(rows)[idx]
        out[well] = WellPicks(
            md=rows_arr[:, 0], tvdss=rows_arr[:, 1],
            easting=rows_arr[:, 2], northing=rows_arr[:, 3],
        )
    return out


def parse_checkshots(path: Path, members: dict[str, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    number = re.compile(r"[-+]?\d+(?:\.\d+)?")
    with ZipFile(path) as archive:
        for well, member in members.items():
            lines = archive.read(member).decode("latin1", errors="replace").splitlines()
            depth: list[float] = []
            twt: list[float] = []
            for line in lines:
                values = [float(v) for v in number.findall(line)]
                if not line.lstrip().startswith("TIME-CKS") or len(values) < 4:
                    continue
                depth.append(abs(values[2]))
                twt.append(values[3])
            depth_arr = np.array(depth)
            twt_arr = np.array(twt)
            order = np.argsort(depth_arr)
            depth_arr, twt_arr = depth_arr[order], twt_arr[order]
            _, idx = np.unique(depth_arr, return_index=True)
            curves[well] = (depth_arr[idx], twt_arr[idx])
    return curves


def load_well_curve(well: str) -> dict:
    with h5py.File(WELL_LOGS_H5, "r") as f:
        g = f[LOG_KEY[well]]
        depth_md = g["depth_grid_m"][()].astype(np.float64)
        dt = g["clean"]["LFP_DT"][()].astype(np.float64)
        rhob = g["clean"]["LFP_RHOB"][()].astype(np.float64)
    valid = np.isfinite(dt) & np.isfinite(rhob) & (dt > 0) & (rhob > 0)
    # restrict to the single longest contiguous valid run
    runs = []
    start = None
    for i, v in enumerate(np.append(valid, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        raise RuntimeError(f"{well}: no contiguous valid DT+RHOB interval")
    best = max(runs, key=lambda r: r[1] - r[0])
    sl = slice(best[0], best[1])
    return {"depth_md": depth_md[sl], "dt": dt[sl], "rhob": rhob[sl]}


def md_to_tvdss(md: np.ndarray, picks: WellPicks) -> np.ndarray:
    order = np.argsort(picks.md)
    return np.interp(md, picks.md[order], picks.tvdss[order])


def sonic_reflectivity_and_time(depth_md: np.ndarray, dt: np.ndarray, rhob: np.ndarray) -> dict:
    velocity_m_s = 1.0e6 * FT_TO_M / dt  # DT in us/ft
    step_m = np.diff(depth_md)
    owt_increment_ms = 1000.0 * step_m / velocity_m_s[:-1]
    relative_owt_ms = np.concatenate([[0.0], np.cumsum(owt_increment_ms)])
    relative_twt_ms = 2.0 * relative_owt_ms
    reflectivity = acoustic_reflectivity(vp=velocity_m_s, rho=rhob, mode="same")
    return {"velocity_m_s": velocity_m_s, "relative_twt_ms": relative_twt_ms, "reflectivity": reflectivity}


def estimate_wavelet_frequency(trace: np.ndarray, dt_s: float) -> float:
    spectrum = np.abs(np.fft.rfft(trace * np.hanning(len(trace))))
    freqs = np.fft.rfftfreq(len(trace), d=dt_s)
    mask = (freqs > 5.0) & (freqs < 80.0)
    if not np.any(mask):
        return 30.0
    return float(freqs[mask][np.argmax(spectrum[mask])])


def build_synthetic(reflectivity_depth: np.ndarray, relative_twt_ms: np.ndarray,
                     t0_ms: float, sample_ms: np.ndarray, wavelet_freq_hz: float) -> np.ndarray:
    twt_ms = relative_twt_ms + t0_ms
    order = np.argsort(twt_ms)
    twt_ms_sorted = twt_ms[order]
    refl_sorted = reflectivity_depth[order]
    dt_s = float(sample_ms[1] - sample_ms[0]) / 1000.0
    refl_time = np.zeros_like(sample_ms, dtype=np.float64)
    valid = (sample_ms >= twt_ms_sorted[0]) & (sample_ms <= twt_ms_sorted[-1])
    refl_time[valid] = np.interp(sample_ms[valid], twt_ms_sorted, refl_sorted)
    wavelet, _ = ricker(duration=0.128, dt=dt_s, f=wavelet_freq_hz)
    synthetic = np.convolve(refl_time, wavelet, mode="same")
    return synthetic


def normalized_xcorr_best_shift(synthetic_window: np.ndarray, real_trace: np.ndarray,
                                 candidate_starts_idx: np.ndarray) -> tuple[int, float]:
    syn = synthetic_window - synthetic_window.mean()
    syn_norm = np.linalg.norm(syn)
    best_idx, best_score = candidate_starts_idx[0], -np.inf
    n = len(syn)
    for start in candidate_starts_idx:
        if start < 0 or start + n > len(real_trace):
            continue
        seg = real_trace[start:start + n]
        seg = seg - seg.mean()
        denom = syn_norm * np.linalg.norm(seg)
        score = float(np.dot(syn, seg) / denom) if denom > 0 else -1.0
        if score > best_score:
            best_score, best_idx = score, start
    return int(best_idx), float(best_score)


def dtw_refine(synthetic_local: np.ndarray, real_local: np.ndarray) -> tuple[np.ndarray, float]:
    """Slope-constrained DTW: window size bounds max local stretch/squeeze."""
    n = len(synthetic_local)
    window = max(2, int(n * (MAX_LOCAL_STRETCH - 1.0)))
    syn64 = np.ascontiguousarray(synthetic_local, dtype=np.float64)
    real64 = np.ascontiguousarray(real_local, dtype=np.float64)
    path = dtw.warping_path(syn64, real64, window=window, use_c=True)
    distance = dtw.distance(syn64, real64, window=window, use_c=True)
    return np.array(path), float(distance)


def predict_well(well: str, index: dict, get_trace, t0_candidates: np.ndarray | None = None) -> dict:
    """t0_candidates lets a caller narrow the coarse search (e.g. to a
    regional prior built from nearby wells' real checkshots) instead of
    the module-default blind full-range T0_SEARCH_MS."""
    search_candidates = T0_SEARCH_MS if t0_candidates is None else t0_candidates
    curve = load_well_curve(well)
    picks = parse_well_picks_full(WELL_PICKS)[PICK_NAME[well]]
    tvdss = md_to_tvdss(curve["depth_md"], picks)
    physics = sonic_reflectivity_and_time(curve["depth_md"], curve["dt"], curve["rhob"])

    # well XY at mid-depth via the same official picks (spatial, not time info)
    order = np.argsort(picks.md)
    x = float(np.interp(np.median(curve["depth_md"]), picks.md[order], picks.easting[order]))
    y = float(np.interp(np.median(curve["depth_md"]), picks.md[order], picks.northing[order]))
    il, xl = get_trace.utm_to_il_xl(x, y)
    real_trace = get_trace.get_trace(il, xl)
    sample_ms = index["samples_ms"]
    dt_s = float(sample_ms[1] - sample_ms[0]) / 1000.0

    wavelet_freq = estimate_wavelet_frequency(real_trace, dt_s)

    span_ms = physics["relative_twt_ms"][-1]
    n_samples_window = max(8, int(span_ms / (sample_ms[1] - sample_ms[0])) + 1)

    # coarse search: build synthetic at each candidate t0, score by xcorr.
    # Keep ALL scores so we can report top-K candidates and flag ambiguity
    # (issue #1 acceptance criterion: reject when the posterior is unclear
    # rather than silently trusting a single best correlation peak).
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
        s = syn_win - syn_win.mean()
        r = real_win - real_win.mean()
        denom = np.linalg.norm(s) * np.linalg.norm(r)
        score = float(np.dot(s, r) / denom) if denom > 0 else -1.0
        scored.append((float(t0), score))
        synthetics[float(t0)] = synthetic

    scored.sort(key=lambda p: p[1], reverse=True)
    # non-maximum suppression in t0 (peaks within 40ms count as the same lobe)
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

    # map each depth sample's relative_twt(+best_t0) through the DTW path onto real-trace time
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


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - target
    return {
        "rows": int(len(error)),
        "mae_ms": float(np.mean(np.abs(error))),
        "rmse_ms": float(np.sqrt(np.mean(error ** 2))),
        "bias_ms": float(np.mean(error)),
        "median_absolute_error_ms": float(np.median(np.abs(error))),
    }


def main() -> int:
    sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))
    from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

    index = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        for well in WELLS:
            pred = predict_well(well, index, vol)
            depth_true, twt_true = checkshots[well]
            pred_at_checkshot_depth = np.interp(
                depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
                left=np.nan, right=np.nan,
            )
            valid = np.isfinite(pred_at_checkshot_depth)
            valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
            m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])
            results[well] = {
                "coarse_t0_ms": pred["coarse_t0_ms"],
                "coarse_xcorr": pred["coarse_xcorr"],
                "top_k_peaks": pred["top_k_peaks"],
                "ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "dtw_distance": pred["dtw_distance"],
                "wavelet_freq_hz": pred["wavelet_freq_hz"],
                "inline": pred["inline"],
                "crossline": pred["crossline"],
                "own_log_depth_tvdss_range": [float(pred["depth_tvdss"].min()), float(pred["depth_tvdss"].max())],
                "n_checkshot_points_in_range": int(valid.sum()),
                "metrics_vs_own_checkshot": m,
            }
            reject_flag = " [REJECT: ambiguous]" if pred["reject_ambiguous"] else ""
            print(f"{well}: t0={pred['coarse_t0_ms']:.0f}ms xcorr={pred['coarse_xcorr']:.3f} "
                  f"gap={pred['ambiguity_gap']:.3f}{reject_flag} "
                  f"n_eval={m['rows']} MAE={m['mae_ms']:.2f}ms RMSE={m['rmse_ms']:.2f}ms bias={m['bias_ms']:.2f}ms")

    p23_reference = {
        "note": "P23 checkshot-based spatial interpolation on VALIDATION wells F11T2/F15A",
        "target_reservoir_mae_ms": 8.7389,
        "old_weak_tie_mae_ms": 633.1867,
    }
    summary = {
        "method": "sonic_integration + acoustic_reflectivity + ricker_wavelet + xcorr_coarse_search + slope_constrained_dtw",
        "wells": results,
        "p23_reference": p23_reference,
        "inputs": {
            "well_picks_sha256": sha256(WELL_PICKS),
            "checkshot_zip_sha256": sha256(VSP_ZIP),
        },
        "constraints": {
            "max_local_stretch": MAX_LOCAL_STRETCH,
            "dtw_window_ms": DTW_WINDOW_MS,
            "t0_search_range_ms": [float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1])],
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
