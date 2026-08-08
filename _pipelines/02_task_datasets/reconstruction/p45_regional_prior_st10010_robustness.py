#!/usr/bin/env python3
"""P45: does the "regional prior narrows t0 search" fix (p45_well_tie_regional_prior.py)
also rescue acceptance criterion (c) -- ST0202 vs ST10010 robustness?

Prior result without any fix: I0's blind full-axis search drifted t0 by up
to 2716ms between ST0202 and ST10010, and flipped the ambiguity-gate
decision on 2/3 wells (p45_well_tie_st10010_robustness.py).

Prior result with the regional-prior fix but only tested on ST0202: 2/3
wells improved vs I0 blind search (p45_well_tie_regional_prior.py); the
ambiguity gate is still 0/3 "not ambiguous" (all three wells still get
flagged reject_ambiguous=True even after narrowing the window).

This script asks a NEW question: apply the SAME regional prior (same
inverse-distance-weighted checkshot value from the OTHER two wells -- the
prior itself is survey-independent, it is a physical two-way-time value,
not read off either seismic volume) to narrow the t0 search on ST10010
too, then compare the ST10010-with-prior result against the ST0202-with-
prior result (not against I0's blind search). If the prior-narrowing fix
genuinely generalizes, the two survey's "with prior" t0/MAE should now be
close to each other, since both start the coarse search from the same
+/-300ms window around the same physical prior.

Honesty note: if DTW refinement (dtaidistance, locally slope-constrained
to +/-20%) still walks the coarse-search winner to very different final
t0 on the two surveys because of trace-content differences inside the
window, that is exactly the "DTW is sensitive to small differences in the
starting point" failure mode diagnosed in the previous session for I0's
blind search -- this script reports that honestly rather than declaring a
fix that isn't real.
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
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, WELL_PICKS,
    predict_well, metrics, parse_checkshots, parse_well_picks_full, load_well_curve,
)
from p45_well_tie_regional_prior import (  # noqa: E402
    build_regional_prior_t0, PRIOR_HALF_WIDTH_MS, PRIOR_SEARCH_STEP_MS,
)
from p45_well_tie_st10010_robustness import ST10010_SEGY, build_index, ST10010Volume  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_regional_prior_st10010_robustness"
ST0202_REGIONAL_PRIOR_SUMMARY = HERE / "_outputs/p45_well_tie_regional_prior/summary.json"
# same tolerance as I0's own DTW local refinement window: if the two
# surveys' "with prior" t0 lands within this of each other, we call it
# "the same lobe" rather than a materially different tie.
T0_CONSISTENCY_TOLERANCE_MS = 200.0


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ST0202_REGIONAL_PRIOR_SUMMARY.exists():
        print(f"missing prerequisite: {ST0202_REGIONAL_PRIOR_SUMMARY} "
              "(run p45_well_tie_regional_prior.py main() first)")
        return 1
    st0202_regional = json.loads(ST0202_REGIONAL_PRIOR_SUMMARY.read_text(encoding="utf-8"))

    st10010_index = build_index(ST10010_SEGY)
    st10010_min_ms = float(st10010_index["samples_ms"].min())
    st10010_max_ms = float(st10010_index["samples_ms"].max())

    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)

    results = {}
    with ST10010Volume(segy_path=ST10010_SEGY, index=st10010_index) as vol:
        for well in WELLS:
            curve = load_well_curve(well)
            # same prior as the ST0202 regional-prior run: built ONLY from
            # the other two wells' real checkshot, survey-independent.
            prior_t0 = build_regional_prior_t0(well, checkshots, all_picks, curve)
            lo = max(st10010_min_ms, prior_t0 - PRIOR_HALF_WIDTH_MS)
            hi = min(st10010_max_ms, prior_t0 + PRIOR_HALF_WIDTH_MS)
            candidates = np.arange(lo, hi, PRIOR_SEARCH_STEP_MS)

            try:
                pred = predict_well(well, st10010_index, vol, t0_candidates=candidates)
            except ValueError as exc:
                results[well] = {"status": "out_of_grid", "error": str(exc)}
                print(f"{well}: OUT OF GRID on ST10010 -- {exc}")
                continue
            if len(candidates) == 0:
                results[well] = {"status": "empty_search_window",
                                  "regional_prior_t0_ms": prior_t0,
                                  "st10010_time_range_ms": [st10010_min_ms, st10010_max_ms]}
                print(f"{well}: prior {prior_t0:.0f}ms falls entirely outside ST10010's "
                      f"[{st10010_min_ms:.0f},{st10010_max_ms:.0f}]ms range -- no valid search window")
                continue

            depth_true, twt_true = checkshots[well]
            pred_at_checkshot_depth = np.interp(
                depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
                left=np.nan, right=np.nan,
            )
            valid = np.isfinite(pred_at_checkshot_depth)
            valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
            m = metrics(twt_true[valid], pred_at_checkshot_depth[valid])

            st0202_r = st0202_regional["results"][well]
            t0_delta_ms = pred["coarse_t0_ms"] - st0202_r["chosen_t0_ms"]
            mae_delta_ms = m["mae_ms"] - st0202_r["metrics_with_prior"]["mae_ms"]
            gate_agrees = bool(pred["reject_ambiguous"] == st0202_r["reject_ambiguous"])
            t0_consistent = bool(abs(t0_delta_ms) <= T0_CONSISTENCY_TOLERANCE_MS)

            results[well] = {
                "status": "evaluated",
                "regional_prior_t0_ms": prior_t0,
                "st10010_search_window_ms": [float(candidates[0]), float(candidates[-1])],
                "st10010_chosen_t0_ms": pred["coarse_t0_ms"],
                "st10010_xcorr": pred["coarse_xcorr"],
                "st10010_ambiguity_gap": pred["ambiguity_gap"],
                "st10010_reject_ambiguous": pred["reject_ambiguous"],
                "st10010_dtw_distance": pred["dtw_distance"],
                "st10010_metrics_with_prior": m,
                "st0202_search_window_ms": st0202_r["search_window_ms"],
                "st0202_chosen_t0_ms": st0202_r["chosen_t0_ms"],
                "st0202_reject_ambiguous": st0202_r["reject_ambiguous"],
                "st0202_metrics_with_prior": st0202_r["metrics_with_prior"],
                "t0_delta_ms": t0_delta_ms,
                "mae_delta_ms": mae_delta_ms,
                "both_rejected_or_both_kept": gate_agrees,
                "t0_consistent_within_tolerance": t0_consistent,
            }
            print(f"{well}: prior_t0={prior_t0:.0f}ms | ST0202-w/prior t0={st0202_r['chosen_t0_ms']:.0f}ms "
                  f"MAE={st0202_r['metrics_with_prior']['mae_ms']:.1f}ms | "
                  f"ST10010-w/prior t0={pred['coarse_t0_ms']:.0f}ms MAE={m['mae_ms']:.1f}ms | "
                  f"t0_delta={t0_delta_ms:+.0f}ms mae_delta={mae_delta_ms:+.1f}ms "
                  f"gate_agrees={gate_agrees} t0_consistent={t0_consistent}")

    n_evaluated = sum(1 for r in results.values() if r.get("status") == "evaluated")
    n_gate_agrees = sum(1 for r in results.values() if r.get("both_rejected_or_both_kept"))
    n_t0_consistent = sum(1 for r in results.values() if r.get("t0_consistent_within_tolerance"))
    max_abs_t0_delta = max(
        (abs(r["t0_delta_ms"]) for r in results.values() if r.get("status") == "evaluated"),
        default=float("nan"),
    )

    # compare against the un-fixed blind-search robustness numbers (from
    # the previous P45(c) test) purely for reporting, without re-deriving
    # them here.
    blind_summary_path = HERE / "_outputs/p45_well_tie_st10010_robustness/summary.json"
    blind_max_abs_t0_delta = None
    if blind_summary_path.exists():
        blind_summary = json.loads(blind_summary_path.read_text(encoding="utf-8"))
        blind_deltas = [abs(r["t0_delta_ms"]) for r in blind_summary["results"].values()
                         if r.get("status") == "evaluated"]
        blind_max_abs_t0_delta = max(blind_deltas) if blind_deltas else None

    passes_acceptance_criterion_c = bool(
        n_evaluated == len(WELLS) and n_t0_consistent == len(WELLS) and n_gate_agrees == len(WELLS)
    )

    summary = {
        "method": "same_regional_prior_narrows_t0_search_on_BOTH_st0202_and_st10010_then_compare_with_prior_vs_with_prior",
        "prior_half_width_ms": PRIOR_HALF_WIDTH_MS,
        "t0_consistency_tolerance_ms": T0_CONSISTENCY_TOLERANCE_MS,
        "st10010_time_range_ms": [st10010_min_ms, st10010_max_ms],
        "results": results,
        "verdict": {
            "n_wells_evaluated_on_both": n_evaluated,
            "n_wells_ambiguity_gate_agrees": n_gate_agrees,
            "n_wells_t0_consistent_within_tolerance": n_t0_consistent,
            "max_abs_t0_delta_ms_with_prior": None if np.isnan(max_abs_t0_delta) else max_abs_t0_delta,
            "max_abs_t0_delta_ms_blind_search_reference": blind_max_abs_t0_delta,
            "passes_acceptance_criterion_c": passes_acceptance_criterion_c,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{n_t0_consistent}/{len(WELLS)} wells t0-consistent within {T0_CONSISTENCY_TOLERANCE_MS:.0f}ms; "
          f"{n_gate_agrees}/{len(WELLS)} wells ambiguity-gate agrees; "
          f"max|t0_delta|={max_abs_t0_delta:.0f}ms (blind-search reference was "
          f"{blind_max_abs_t0_delta}ms); "
          f"passes_acceptance_criterion_c={passes_acceptance_criterion_c}")
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
