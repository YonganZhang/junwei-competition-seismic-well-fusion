#!/usr/bin/env python3
"""P45: swap the coarse-search t0 scoring function from raw-amplitude
normalized cross-correlation to self-supervised-encoder embedding cosine
similarity, and test whether that is more robust.

Motivation (per GitHub issue #1 + user direction, 2026-08-08): amplitude-
domain coarse search (I0's normalized xcorr) and its variants (envelope
correlation, band-pass, multi-scale search -- all tried and shown to not
help) are amplitude-sensitive. A self-supervised encoder pretrained on real
ST0202 traces (p45_seismic_trace_ssl_pretrain.py, masked-reconstruction,
never sees t0/checkshot labels) should, in principle, embed waveform SHAPE
into a space where amplitude-domain distortion matters less. This module
tests that hypothesis directly: same candidate range, same top-3 NMS, same
ambiguity-gate threshold as p45_well_tie_physics_baseline.predict_well --
the ONLY thing that changes is the per-candidate score, from normalized
xcorr(syn_window, real_window) to cosine(encoder(syn_window),
encoder(real_window)).

Honesty note: the encoder was pretrained on fixed 256-sample (1024ms)
windows with per-window z-score normalization. The coarse-search window
here spans the well's whole reflectivity interval (typically hundreds to
~2000 samples), which is longer and of variable length across wells/t0
candidates. The encoder's global-average-pool head accepts any length, so
this is architecturally valid, but it means embeddings here are evaluated
somewhat off-distribution from what the encoder was pretrained on. This is
reported as a risk, not hidden.

Evaluated on the exact same protocol as the existing physics baseline and
ST10010 robustness scripts:
  1. ST0202, 3 wells, blind full-range t0 search: embedding-cosine MAE vs
     I0's amplitude-xcorr MAE (both read fresh, not hardcoded).
  2. ST10010 (different seismic source/vintage, same 3 wells), blind
     full-range t0 search with the SAME embedding method: cross-source t0
     drift (st10010_t0 - st0202_t0) vs the existing xcorr-based drift
     already recorded in p45_well_tie_st10010_robustness/summary.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_seismic_trace_ssl_pretrain import TraceEncoder  # noqa: E402
from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, SEGY_PATH, T0_SEARCH_MS, DTW_WINDOW_MS,
    build_synthetic, dtw_refine, estimate_wavelet_frequency, load_well_curve,
    md_to_tvdss, metrics, parse_checkshots, parse_well_picks_full, sha256,
    sonic_reflectivity_and_time, WELL_PICKS, PICK_NAME,
)
from p45_well_tie_st10010_robustness import ST10010_SEGY, ST10010Volume, build_index  # noqa: E402
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_coarsesearch_ssl_embedding"
ENCODER_PATH = HERE / "_outputs/p45_seismic_trace_ssl_pretrain/encoder.pt"
I0_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"
ST10010_SUMMARY = HERE / "_outputs/p45_well_tie_st10010_robustness/summary.json"

NMS_GAP_MS = 40.0
AMBIGUITY_GAP_THRESHOLD = 0.05  # same value p45_well_tie_physics_baseline uses, unchanged


# ---------------------------------------------------------------------------
# Encoder loading + embedding scoring
# ---------------------------------------------------------------------------

def load_encoder(path: Path) -> TraceEncoder:
    if not path.exists():
        raise FileNotFoundError(
            f"pretrained encoder missing at {path} -- run "
            "p45_seismic_trace_ssl_pretrain.py first"
        )
    ckpt = torch.load(path, map_location="cpu")
    encoder = TraceEncoder(
        embed_dim=ckpt["config"]["embed_dim"],
        channels=tuple(ckpt["config"]["channels"]),
        kernel_size=ckpt["config"]["kernel_size"],
    )
    encoder.load_state_dict(ckpt["state_dict"])
    encoder.eval()
    return encoder


def encode_window(encoder: TraceEncoder, window: np.ndarray) -> np.ndarray:
    """Per-window z-score normalize (matches pretraining), then encode.
    Returns a (embed_dim,) numpy vector."""
    w = window.astype(np.float32)
    std = float(w.std())
    mean = float(w.mean())
    wn = (w - mean) / (std + 1e-8)
    x = torch.from_numpy(wn).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    with torch.no_grad():
        emb = encoder(x).squeeze(0).numpy()
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def embedding_score(encoder: TraceEncoder, syn_win: np.ndarray, real_win: np.ndarray) -> float:
    if len(syn_win) < 8 or np.std(syn_win) < 1e-8 or np.std(real_win) < 1e-8:
        return -1.0
    emb_syn = encode_window(encoder, syn_win)
    emb_real = encode_window(encoder, real_win)
    return cosine_similarity(emb_syn, emb_real)


# ---------------------------------------------------------------------------
# predict_well, embedding-scored coarse search (mirrors predict_well;
# ONLY the coarse-search score function changes -- candidate range, NMS,
# ambiguity gate threshold, and the DTW refine step are byte-for-byte the
# same logic as p45_well_tie_physics_baseline.predict_well)
# ---------------------------------------------------------------------------

def predict_well_embedding(well: str, index: dict, get_trace, encoder: TraceEncoder,
                            t0_candidates: np.ndarray | None = None) -> dict:
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
        score = embedding_score(encoder, syn_win, real_win)
        scored.append((float(t0), score))
        synthetics[float(t0)] = synthetic

    scored.sort(key=lambda p: p[1], reverse=True)
    peaks: list[tuple[float, float]] = []
    for t0, score in scored:
        if all(abs(t0 - p[0]) > NMS_GAP_MS for p in peaks):
            peaks.append((t0, score))
        if len(peaks) >= 3:
            break
    best_t0, best_score = peaks[0]
    second_score = peaks[1][1] if len(peaks) > 1 else float("nan")
    ambiguity_gap = best_score - second_score if len(peaks) > 1 else float("nan")
    reject_ambiguous = bool(len(peaks) > 1 and ambiguity_gap < AMBIGUITY_GAP_THRESHOLD)
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
        "coarse_score": best_score,
        "top_k_peaks": peaks,
        "ambiguity_gap": ambiguity_gap,
        "reject_ambiguous": reject_ambiguous,
        "dtw_distance": dtw_dist,
        "wavelet_freq_hz": wavelet_freq,
        "inline": int(il),
        "crossline": int(xl),
        "score_type": "embedding_cosine_similarity",
    }


def evaluate_survey(vol, index, wells, checkshots, encoder) -> dict:
    results = {}
    for well in wells:
        pred = predict_well_embedding(well, index, vol, encoder)
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
            "coarse_score": pred["coarse_score"],
            "top_k_peaks": pred["top_k_peaks"],
            "ambiguity_gap": pred["ambiguity_gap"],
            "reject_ambiguous": pred["reject_ambiguous"],
            "dtw_distance": pred["dtw_distance"],
            "wavelet_freq_hz": pred["wavelet_freq_hz"],
            "inline": pred["inline"],
            "crossline": pred["crossline"],
            "n_checkshot_points_in_range": int(valid.sum()),
            "metrics_vs_own_checkshot": m,
        }
        reject_flag = " [REJECT: ambiguous]" if pred["reject_ambiguous"] else ""
        print(f"  {well}: t0={pred['coarse_t0_ms']:.0f}ms score={pred['coarse_score']:.3f} "
              f"gap={pred['ambiguity_gap']:.3f}{reject_flag} "
              f"n_eval={m['rows']} MAE={m['mae_ms']:.2f}ms")
    return results


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    if not ENCODER_PATH.exists():
        summary = {"status": "FAILED_ENCODER_MISSING", "encoder_path": str(ENCODER_PATH)}
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"FAILED: pretrained encoder missing at {ENCODER_PATH}")
        return 1

    encoder = load_encoder(ENCODER_PATH)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"loaded encoder: {n_params} params, embed_dim={encoder.embed_dim}")

    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    t_start = time.time()

    # --- ST0202: embedding-cosine coarse search, same protocol as I0 ---
    print("\n=== ST0202 (primary survey), blind full-range t0 search ===")
    st0202_index = load_index()
    with SeismicVolume(segy_path=SEGY_PATH, index=st0202_index) as vol:
        st0202_results = evaluate_survey(vol, st0202_index, WELLS, checkshots, encoder)

    # --- ST10010: same embedding method, different seismic source/vintage ---
    print("\n=== ST10010 (different source/vintage), blind full-range t0 search ===")
    st10010_index = build_index(ST10010_SEGY)
    with ST10010Volume(segy_path=ST10010_SEGY, index=st10010_index) as vol:
        st10010_results = {}
        for well in WELLS:
            try:
                pred = predict_well_embedding(well, st10010_index, vol, encoder)
            except ValueError as exc:
                st10010_results[well] = {"status": "out_of_grid", "error": str(exc)}
                print(f"  {well}: OUT OF GRID on ST10010 -- {exc}")
                continue
            depth_true, twt_true = checkshots[well]
            pred_at_checkshot_depth = np.interp(
                depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
                left=np.nan, right=np.nan,
            )
            valid = np.isfinite(pred_at_checkshot_depth)
            valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
            m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])
            st10010_results[well] = {
                "status": "evaluated",
                "coarse_t0_ms": pred["coarse_t0_ms"],
                "coarse_score": pred["coarse_score"],
                "ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "metrics_vs_own_checkshot": m,
            }
            print(f"  {well}: t0={pred['coarse_t0_ms']:.0f}ms score={pred['coarse_score']:.3f} "
                  f"MAE={m['mae_ms']:.2f}ms reject={pred['reject_ambiguous']}")

    elapsed_s = time.time() - t_start

    # --- load reference numbers fresh (never hardcoded) for honest comparison ---
    i0_summary = json.loads(I0_SUMMARY.read_text(encoding="utf-8")) if I0_SUMMARY.exists() else None
    st10010_xcorr_summary = json.loads(ST10010_SUMMARY.read_text(encoding="utf-8")) if ST10010_SUMMARY.exists() else None

    # --- ST0202: embedding MAE vs xcorr MAE, per well ---
    st0202_comparison = {}
    n_st0202_improved = 0
    for well in WELLS:
        emb_mae = st0202_results[well]["metrics_vs_own_checkshot"]["mae_ms"]
        xcorr_mae = i0_summary["wells"][well]["metrics_vs_own_checkshot"]["mae_ms"] if i0_summary else None
        improved = bool(xcorr_mae is not None and emb_mae < xcorr_mae)
        n_st0202_improved += int(improved)
        st0202_comparison[well] = {
            "embedding_mae_ms": emb_mae,
            "xcorr_baseline_mae_ms": xcorr_mae,
            "mae_delta_ms": (emb_mae - xcorr_mae) if xcorr_mae is not None else None,
            "embedding_improves_on_xcorr": improved,
            "embedding_reject_ambiguous": st0202_results[well]["reject_ambiguous"],
            "xcorr_reject_ambiguous": i0_summary["wells"][well]["reject_ambiguous"] if i0_summary else None,
        }

    # --- ST10010 cross-source t0 drift: embedding method vs existing xcorr drift ---
    st10010_comparison = {}
    n_st10010_drift_smaller = 0
    n_st10010_evaluated = 0
    for well in WELLS:
        r10 = st10010_results.get(well, {})
        if r10.get("status") != "evaluated":
            st10010_comparison[well] = {"status": r10.get("status", "missing")}
            continue
        n_st10010_evaluated += 1
        emb_t0_delta = r10["coarse_t0_ms"] - st0202_results[well]["coarse_t0_ms"]
        xcorr_t0_delta = (
            st10010_xcorr_summary["results"][well]["t0_delta_ms"]
            if st10010_xcorr_summary and st10010_xcorr_summary["results"].get(well, {}).get("status") == "evaluated"
            else None
        )
        drift_smaller = bool(xcorr_t0_delta is not None and abs(emb_t0_delta) < abs(xcorr_t0_delta))
        n_st10010_drift_smaller += int(drift_smaller)
        st10010_comparison[well] = {
            "embedding_t0_delta_ms": emb_t0_delta,
            "xcorr_baseline_t0_delta_ms": xcorr_t0_delta,
            "embedding_drift_smaller_than_xcorr": drift_smaller,
            "embedding_mae_ms": r10["metrics_vs_own_checkshot"]["mae_ms"],
            "xcorr_baseline_mae_ms": (
                st10010_xcorr_summary["results"][well]["st10010_metrics"]["mae_ms"]
                if st10010_xcorr_summary and st10010_xcorr_summary["results"].get(well, {}).get("status") == "evaluated"
                else None
            ),
        }

    verdict = {
        "st0202_n_wells_embedding_beats_xcorr_mae": n_st0202_improved,
        "st0202_n_wells_total": len(WELLS),
        "st10010_n_wells_embedding_drift_smaller_than_xcorr": n_st10010_drift_smaller,
        "st10010_n_wells_evaluated": n_st10010_evaluated,
        "honest_summary": (
            "Embedding-cosine scoring is compared per-well against xcorr on both "
            "accuracy (ST0202 MAE) and cross-source robustness (ST10010 t0 drift). "
            "See per-well comparison dicts above for the actual counts; no "
            "aggregate claim of 'better' is made beyond the literal counts."
        ),
    }

    summary = {
        "status": "OK",
        "purpose": (
            "Replace I0's amplitude-domain normalized-xcorr coarse-search score "
            "with cosine similarity between self-supervised-encoder embeddings "
            "of the synthetic and real trace windows. Candidate t0 range, top-3 "
            "NMS, and the ambiguity-gate threshold are unchanged from "
            "p45_well_tie_physics_baseline.predict_well."
        ),
        "encoder": {
            "path": str(ENCODER_PATH.relative_to(HERE)),
            "n_params": int(n_params),
            "embed_dim": int(encoder.embed_dim),
            "note": (
                "Encoder was pretrained on fixed 256-sample (1024ms) windows; "
                "here it scores variable-length windows spanning each well's "
                "whole reflectivity interval (hundreds to ~2000 samples), which "
                "is off-distribution from pretraining. Architecturally valid "
                "(global-average-pool head), but a real caveat, not swept under."
            ),
        },
        "config": {
            "nms_gap_ms": NMS_GAP_MS,
            "ambiguity_gap_threshold": AMBIGUITY_GAP_THRESHOLD,
            "t0_search_range_ms": [float(T0_SEARCH_MS[0]), float(T0_SEARCH_MS[-1])],
        },
        "st0202_results": st0202_results,
        "st0202_vs_xcorr_baseline": st0202_comparison,
        "st10010_results": st10010_results,
        "st10010_vs_xcorr_baseline": st10010_comparison,
        "verdict": verdict,
        "elapsed_seconds": elapsed_s,
        "inputs": {
            "well_picks_sha256": sha256(WELL_PICKS),
            "checkshot_zip_sha256": sha256(VSP_ZIP),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nST0202: embedding beats xcorr MAE on {n_st0202_improved}/{len(WELLS)} wells")
    print(f"ST10010: embedding cross-source t0 drift smaller than xcorr on "
          f"{n_st10010_drift_smaller}/{n_st10010_evaluated} evaluated wells")
    print(f"elapsed: {elapsed_s:.1f}s")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
