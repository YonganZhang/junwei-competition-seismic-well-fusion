#!/usr/bin/env python3
"""P45 traditional-method-informed search: narrow t0 using nearby wells'
real checkshot as a regional prior, instead of I0's blind full-axis search.

This is what a human geophysicist actually does without a checkshot for the
well being tied: borrow the field's already-established time-depth trend
from OTHER calibrated wells nearby (inverse-distance weighted, exactly
P23's spatial_prediction logic) to get a rough starting-time range, then
only fine-search within that range -- not blindly across the whole record.

Leave-one-well-out, honest: for the held-out well, ONLY the other two
wells' real checkshot is used to build the prior. The held-out well's own
checkshot is read only afterward, for scoring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, PICK_NAME, WELL_PICKS,
    predict_well, metrics, parse_checkshots, parse_well_picks_full,
    load_well_curve, md_to_tvdss,
)
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_well_tie_regional_prior"
I0_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"
PRIOR_HALF_WIDTH_MS = 300.0  # search only within +/- this window around the prior
PRIOR_SEARCH_STEP_MS = 4.0


def well_xy(well: str, curve: dict, picks) -> tuple[float, float]:
    order = np.argsort(picks.md)
    x = float(np.interp(np.median(curve["depth_md"]), picks.md[order], picks.easting[order]))
    y = float(np.interp(np.median(curve["depth_md"]), picks.md[order], picks.northing[order]))
    return x, y


def build_regional_prior_t0(held_out: str, checkshots: dict, all_picks: dict, held_curve: dict) -> float:
    """Inverse-distance-weighted prior for held_out's t0, using ONLY the
    other wells' real checkshot curves (never held_out's own)."""
    held_picks = all_picks[PICK_NAME[held_out]]
    hx, hy = well_xy(held_out, held_curve, held_picks)
    # target depth: midpoint of held_out's own valid DT/RHOB interval, in TVDSS
    target_tvdss = float(np.median(md_to_tvdss(held_curve["depth_md"], held_picks)))

    numerator, denominator = 0.0, 0.0
    for donor in WELLS:
        if donor == held_out:
            continue
        depth_true, twt_true = checkshots[donor]
        donor_picks = all_picks[PICK_NAME[donor]]
        donor_curve = load_well_curve(donor)
        dx, dy = well_xy(donor, donor_curve, donor_picks)
        distance2 = (hx - dx) ** 2 + (hy - dy) ** 2
        weight = 1.0 / (distance2 + 100.0 ** 2)
        # donor's checkshot TWT interpolated at the target TVDSS
        if target_tvdss < depth_true.min() or target_tvdss > depth_true.max():
            twt_at_target = float(np.interp(target_tvdss, depth_true, twt_true))  # extrapolated by np.interp clamp
        else:
            twt_at_target = float(np.interp(target_tvdss, depth_true, twt_true))
        numerator += weight * twt_at_target
        denominator += weight
    return numerator / denominator if denominator > 0 else 1500.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)
    st0202_summary = json.loads(I0_SUMMARY.read_text(encoding="utf-8"))

    results = {}
    with SeismicVolume(index=index) as vol:
        for well in WELLS:
            curve = load_well_curve(well)
            prior_t0 = build_regional_prior_t0(well, checkshots, all_picks, curve)
            candidates = np.arange(
                max(0.0, prior_t0 - PRIOR_HALF_WIDTH_MS),
                prior_t0 + PRIOR_HALF_WIDTH_MS,
                PRIOR_SEARCH_STEP_MS,
            )
            pred = predict_well(well, index, vol, t0_candidates=candidates)

            depth_true, twt_true = checkshots[well]
            pred_at_checkshot_depth = np.interp(
                depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
                left=np.nan, right=np.nan,
            )
            valid = np.isfinite(pred_at_checkshot_depth)
            valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
            m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])

            i0_r = st0202_summary["wells"][well]
            results[well] = {
                "regional_prior_t0_ms": prior_t0,
                "search_window_ms": [float(candidates[0]), float(candidates[-1])],
                "chosen_t0_ms": pred["coarse_t0_ms"],
                "coarse_xcorr": pred["coarse_xcorr"],
                "ambiguity_gap": pred["ambiguity_gap"],
                "reject_ambiguous": pred["reject_ambiguous"],
                "metrics_with_prior": m,
                "i0_blind_search_metrics": i0_r["metrics_vs_own_checkshot"],
                "mae_delta_vs_i0_ms": m["mae_ms"] - i0_r["metrics_vs_own_checkshot"]["mae_ms"],
                "improved_vs_i0": bool(m["mae_ms"] < i0_r["metrics_vs_own_checkshot"]["mae_ms"]),
            }
            print(f"{well}: prior_t0={prior_t0:.0f}ms window=[{candidates[0]:.0f},{candidates[-1]:.0f}]ms "
                  f"-> chosen={pred['coarse_t0_ms']:.0f}ms MAE={m['mae_ms']:.1f}ms "
                  f"(I0 blind MAE={i0_r['metrics_vs_own_checkshot']['mae_ms']:.1f}ms, "
                  f"delta={results[well]['mae_delta_vs_i0_ms']:+.1f}ms, "
                  f"{'IMPROVED' if results[well]['improved_vs_i0'] else 'NOT_IMPROVED'}, "
                  f"reject={pred['reject_ambiguous']})")

    n_improved = sum(1 for r in results.values() if r["improved_vs_i0"])
    n_kept = sum(1 for r in results.values() if not r["reject_ambiguous"])
    summary = {
        "method": "inverse_distance_weighted_checkshot_prior_from_other_wells_narrows_t0_search",
        "prior_half_width_ms": PRIOR_HALF_WIDTH_MS,
        "results": results,
        "verdict": {
            "n_wells_improved_vs_i0_blind_search": n_improved,
            "n_wells_kept_not_ambiguous": n_kept,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{n_improved}/3 wells improved vs I0 blind search; {n_kept}/3 pass the ambiguity gate")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
