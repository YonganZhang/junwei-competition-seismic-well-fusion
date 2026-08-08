#!/usr/bin/env python3
"""P45 acceptance criterion (4): "synthetic-vs-real-trace match must beat a
sonic-integration + rigid-global-shift baseline".

This script builds exactly that simpler baseline and compares it against I0
(sonic integration + coarse xcorr search + slope-constrained DTW refinement)
on the same 3 wells, using the SAME best_t0 that I0's own coarse xcorr search
already picked (reused via p45_well_tie_physics_baseline.predict_well, no
re-implementation of the search).

Baseline ("global shift only"):
  1. Same sonic integration as I0 -> relative_twt_ms(depth) curve with an
     unknown absolute static shift.
  2. Take I0's coarse-search best_t0 (rigid, single global time shift; no
     local warping).
  3. Final predicted TWT = relative_twt_ms + best_t0. No DTW step at all.

I0 ("global shift + DTW"):
  Same as above, but then slope-constrained DTW locally re-warps the curve
  inside a +/-DTW_WINDOW_MS window around best_t0 to better match the real
  trace.

Both are scored against each well's own real VSP checkshot with the same
metrics() function, at the same checkshot depth samples, so the comparison
isolates exactly the effect of the DTW refinement step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from p45_well_tie_physics_baseline import (  # noqa: E402
    CHECKSHOT_MEMBER,
    PICK_NAME,
    SEGY_PATH,
    VSP_ZIP,
    WELL_PICKS,
    WELLS,
    load_well_curve,
    md_to_tvdss,
    metrics,
    parse_checkshots,
    parse_well_picks_full,
    predict_well,
    sonic_reflectivity_and_time,
)

OUTPUT_DIR = HERE / "_outputs/p45_global_shift_only_baseline"


def score_against_checkshot(depth_tvdss: np.ndarray, predicted_twt_ms: np.ndarray,
                             depth_true: np.ndarray, twt_true: np.ndarray) -> dict:
    pred_at_checkshot_depth = np.interp(
        depth_true, depth_tvdss, predicted_twt_ms, left=np.nan, right=np.nan,
    )
    valid = np.isfinite(pred_at_checkshot_depth)
    valid &= (depth_true >= depth_tvdss.min()) & (depth_true <= depth_tvdss.max())
    m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])
    m["n_checkshot_points_in_range"] = int(valid.sum())
    return m


def main() -> int:
    sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))
    from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

    index = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        for well in WELLS:
            # Reuse I0's full pipeline (blind full-range T0 search + DTW) to
            # get: (a) the coarse-search best_t0 without re-implementing the
            # xcorr search, and (b) I0's own DTW-refined final prediction for
            # a like-for-like comparison on this run.
            pred = predict_well(well, index, vol)
            best_t0 = pred["coarse_t0_ms"]

            # Rebuild the same relative_twt_ms(depth) curve I0 used, then
            # apply ONLY the rigid global shift (no DTW).
            curve = load_well_curve(well)
            picks = all_picks[PICK_NAME[well]]
            tvdss = md_to_tvdss(curve["depth_md"], picks)
            physics = sonic_reflectivity_and_time(curve["depth_md"], curve["dt"], curve["rhob"])
            shift_only_twt_ms = physics["relative_twt_ms"] + best_t0

            assert np.array_equal(tvdss, pred["depth_tvdss"]), f"{well}: depth grid mismatch vs I0"

            depth_true, twt_true = checkshots[well]
            m_shift_only = score_against_checkshot(tvdss, shift_only_twt_ms, depth_true, twt_true)
            m_i0_dtw = score_against_checkshot(pred["depth_tvdss"], pred["predicted_twt_ms"], depth_true, twt_true)

            dtw_improves = bool(m_i0_dtw["mae_ms"] < m_shift_only["mae_ms"])
            results[well] = {
                "coarse_t0_ms": best_t0,
                "coarse_xcorr": pred["coarse_xcorr"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "own_log_depth_tvdss_range": [float(tvdss.min()), float(tvdss.max())],
                "metrics_global_shift_only": m_shift_only,
                "metrics_i0_shift_plus_dtw": m_i0_dtw,
                "dtw_reduces_mae": dtw_improves,
                "mae_delta_ms_dtw_minus_shift_only": m_i0_dtw["mae_ms"] - m_shift_only["mae_ms"],
            }
            print(f"{well}: t0={best_t0:.0f}ms  "
                  f"shift_only MAE={m_shift_only['mae_ms']:.2f}ms  "
                  f"I0(shift+DTW) MAE={m_i0_dtw['mae_ms']:.2f}ms  "
                  f"DTW {'improves' if dtw_improves else 'does NOT improve'}")

    n_wells_dtw_better = sum(1 for r in results.values() if r["dtw_reduces_mae"])
    mean_mae_shift_only = float(np.mean([r["metrics_global_shift_only"]["mae_ms"] for r in results.values()]))
    mean_mae_i0_dtw = float(np.mean([r["metrics_i0_shift_plus_dtw"]["mae_ms"] for r in results.values()]))

    summary = {
        "method": "sonic_integration + reuse_I0_coarse_t0 + RIGID GLOBAL SHIFT ONLY (no DTW)",
        "acceptance_criterion": (
            "(4) synthetic-vs-real-trace match should be better than sonic "
            "integration + rigid global shift baseline"
        ),
        "wells": results,
        "aggregate": {
            "n_wells": len(WELLS),
            "n_wells_where_dtw_reduces_mae": n_wells_dtw_better,
            "mean_mae_ms_global_shift_only": mean_mae_shift_only,
            "mean_mae_ms_i0_shift_plus_dtw": mean_mae_i0_dtw,
            "dtw_reduces_mean_mae": bool(mean_mae_i0_dtw < mean_mae_shift_only),
        },
        "note": (
            "Both methods use the SAME best_t0 from I0's own blind coarse "
            "xcorr search over T0_SEARCH_MS (no checkshot used to pick it). "
            "The only difference is whether the DTW local-warp refinement "
            "step runs afterward. Ground truth (VSP checkshot) is read only "
            "for scoring, identically for both methods."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\naggregate: mean MAE shift_only=%.2fms  I0(shift+DTW)=%.2fms  dtw_reduces_mean_mae=%s"
          % (mean_mae_shift_only, mean_mae_i0_dtw, summary["aggregate"]["dtw_reduces_mean_mae"]))
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
