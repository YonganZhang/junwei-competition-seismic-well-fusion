#!/usr/bin/env python3
"""P45: diagnose/fix DTW local-slope window sensitivity to tiny t0 differences.

Diagnosed problem (see p45_well_tie_regional_prior.py 19BT2 result): the I0
blind search picks coarse_t0=2560.00ms (own-checkshot MAE=183.8ms); the
regional-prior-narrowed search picks coarse_t0=2555.57ms -- only 4.44ms
different -- yet after dtw_refine() the final MAE jumps to 312.8ms (+129ms).
Hypothesis under test: MAX_LOCAL_STRETCH=1.20 makes the DTW local-slope
window (window = max(2, int(n * (MAX_LOCAL_STRETCH - 1.0)))) too narrow, so
the DTW path cannot absorb a few ms of coarse-search jitter and instead
locks onto a bad warp, amplifying it into a large MAE swing.

Two experiments, both leave-one-well-out honest (never touch a well's own
checkshot for anything except final scoring):

  1. CORE: for well 19BT2, force predict_well() to start from EXACTLY the
     two real coarse t0's that were actually chosen in prior runs
     (t0_candidates=[t0], single value -> no re-search, isolates the DTW
     refinement step only) at a sweep of MAX_LOCAL_STRETCH values. Report
     the MAE at each start point and the delta between them -- a smaller
     delta means the DTW step is less sensitive to the ~4.4ms start jitter.

  2. SIDE-EFFECT CHECK: for all three wells, run the ORIGINAL full blind t0
     search (t0_candidates=None) at each MAX_LOCAL_STRETCH value. The
     coarse t0 selection itself never depends on MAX_LOCAL_STRETCH (it only
     affects DTW refinement), so this isolates whether widening/narrowing
     the DTW window helps or hurts the wells that already work today, so a
     "fix" for 19BT2 is not credited if it quietly breaks 19A/19SR.

No threshold is loosened to manufacture a pass; this script only measures
and reports what MAX_LOCAL_STRETCH actually does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

import p45_well_tie_physics_baseline as pb  # noqa: E402
from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, PICK_NAME, WELL_PICKS,
    predict_well, metrics, parse_checkshots, parse_well_picks_full,
)
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_dtw_window_sensitivity"
REGIONAL_PRIOR_SUMMARY = HERE / "_outputs/p45_well_tie_regional_prior/summary.json"
I0_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"

BASELINE_STRETCH = pb.MAX_LOCAL_STRETCH  # 1.20, the value currently shipped
STRETCH_VALUES = [1.05, 1.10, BASELINE_STRETCH, 1.30, 1.50, 2.00]
CORE_WELL = "19BT2"


def score_against_checkshot(well: str, pred: dict, checkshots: dict) -> dict:
    depth_true, twt_true = checkshots[well]
    pred_at_checkshot_depth = np.interp(
        depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"],
        left=np.nan, right=np.nan,
    )
    valid = np.isfinite(pred_at_checkshot_depth)
    valid &= (depth_true >= pred["depth_tvdss"].min()) & (depth_true <= pred["depth_tvdss"].max())
    return metrics(twt_true[valid], pred_at_checkshot_depth[valid])


def run_core_experiment(index, vol, checkshots) -> dict:
    """Fix the coarse t0 to the two REAL values already produced by earlier
    scripts (I0 blind search vs regional-prior-narrowed search for 19BT2),
    then vary only MAX_LOCAL_STRETCH, to isolate the DTW refinement step's
    sensitivity to the ~4.4ms start-point jitter."""
    i0_r = json.loads(I0_SUMMARY.read_text(encoding="utf-8"))["wells"][CORE_WELL]
    prior_r = json.loads(REGIONAL_PRIOR_SUMMARY.read_text(encoding="utf-8"))["results"][CORE_WELL]
    t0_i0 = float(i0_r["coarse_t0_ms"])
    t0_prior = float(prior_r["chosen_t0_ms"])

    by_stretch = {}
    for stretch in STRETCH_VALUES:
        pb.MAX_LOCAL_STRETCH = stretch
        pred_i0 = predict_well(CORE_WELL, index, vol, t0_candidates=np.array([t0_i0]))
        pred_prior = predict_well(CORE_WELL, index, vol, t0_candidates=np.array([t0_prior]))
        m_i0 = score_against_checkshot(CORE_WELL, pred_i0, checkshots)
        m_prior = score_against_checkshot(CORE_WELL, pred_prior, checkshots)
        delta_mae = abs(m_prior["mae_ms"] - m_i0["mae_ms"])
        by_stretch[f"{stretch:.2f}"] = {
            "max_local_stretch": stretch,
            "is_current_default": bool(abs(stretch - BASELINE_STRETCH) < 1e-9),
            "t0_i0_blind_ms": t0_i0,
            "t0_regional_prior_ms": t0_prior,
            "start_point_diff_ms": abs(t0_i0 - t0_prior),
            "mae_at_t0_i0_ms": m_i0["mae_ms"],
            "mae_at_t0_prior_ms": m_prior["mae_ms"],
            "delta_mae_ms": delta_mae,
            "dtw_distance_at_t0_i0": pred_i0["dtw_distance"],
            "dtw_distance_at_t0_prior": pred_prior["dtw_distance"],
        }
    pb.MAX_LOCAL_STRETCH = BASELINE_STRETCH  # restore before side-effect run
    return by_stretch


def run_side_effect_check(index, vol, checkshots) -> dict:
    """For ALL wells, run the ORIGINAL full blind t0 search at each stretch
    value. Coarse t0 selection does not depend on MAX_LOCAL_STRETCH (only
    the DTW refinement step reads it), so any MAE change here is purely the
    DTW window's effect on wells that currently work fine."""
    out: dict = {well: {} for well in WELLS}
    for stretch in STRETCH_VALUES:
        pb.MAX_LOCAL_STRETCH = stretch
        for well in WELLS:
            pred = predict_well(well, index, vol)  # full blind search, unchanged
            m = score_against_checkshot(well, pred, checkshots)
            out[well][f"{stretch:.2f}"] = {
                "max_local_stretch": stretch,
                "coarse_t0_ms": pred["coarse_t0_ms"],
                "mae_ms": m["mae_ms"],
                "dtw_distance": pred["dtw_distance"],
                "reject_ambiguous": pred["reject_ambiguous"],
            }
    pb.MAX_LOCAL_STRETCH = BASELINE_STRETCH
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    parse_well_picks_full(WELL_PICKS)  # sanity: file readable, not otherwise used here

    with SeismicVolume(index=index) as vol:
        core = run_core_experiment(index, vol, checkshots)
        side_effect = run_side_effect_check(index, vol, checkshots)

    assert abs(pb.MAX_LOCAL_STRETCH - BASELINE_STRETCH) < 1e-9, "module global not restored"

    baseline_key = f"{BASELINE_STRETCH:.2f}"
    baseline_delta = core[baseline_key]["delta_mae_ms"]
    print(f"CORE ({CORE_WELL}): start points {core[baseline_key]['t0_i0_blind_ms']:.2f}ms vs "
          f"{core[baseline_key]['t0_regional_prior_ms']:.2f}ms "
          f"(diff={core[baseline_key]['start_point_diff_ms']:.2f}ms)")
    for key, row in core.items():
        flag = " <- current default" if row["is_current_default"] else ""
        print(f"  stretch={row['max_local_stretch']:.2f}: MAE@i0={row['mae_at_t0_i0_ms']:.1f}ms "
              f"MAE@prior={row['mae_at_t0_prior_ms']:.1f}ms delta={row['delta_mae_ms']:.1f}ms{flag}")

    best_key = min(core, key=lambda k: core[k]["delta_mae_ms"])
    best_delta = core[best_key]["delta_mae_ms"]
    print(f"\nMost stable (smallest delta_mae) stretch = {core[best_key]['max_local_stretch']:.2f} "
          f"(delta={best_delta:.1f}ms) vs default delta={baseline_delta:.1f}ms")

    print("\nSIDE-EFFECT CHECK (full blind search, all wells):")
    for well in WELLS:
        for key, row in side_effect[well].items():
            print(f"  {well} stretch={row['max_local_stretch']:.2f}: "
                  f"MAE={row['mae_ms']:.1f}ms reject_ambiguous={row['reject_ambiguous']}")

    # honest verdict: does ANY stretch value reduce the 19BT2 sensitivity
    # (delta_mae) WITHOUT making any other well's blind-search MAE worse
    # than at the current default?
    baseline_side = {well: side_effect[well][baseline_key]["mae_ms"] for well in WELLS}
    candidate_verdicts = {}
    for key, row in core.items():
        stretch = row["max_local_stretch"]
        if row["is_current_default"]:
            continue
        improves_core = row["delta_mae_ms"] < baseline_delta
        no_regression = all(
            side_effect[well][key]["mae_ms"] <= baseline_side[well] + 1e-6
            for well in WELLS
        )
        regressed_wells = [
            well for well in WELLS
            if side_effect[well][key]["mae_ms"] > baseline_side[well] + 1e-6
        ]
        candidate_verdicts[key] = {
            "max_local_stretch": stretch,
            "improves_core_sensitivity": bool(improves_core),
            "no_regression_on_other_wells": bool(no_regression),
            "regressed_wells": regressed_wells,
            "net_recommended": bool(improves_core and no_regression),
        }

    any_net_win = any(v["net_recommended"] for v in candidate_verdicts.values())

    summary = {
        "method": "sweep_MAX_LOCAL_STRETCH_isolating_dtw_refinement_sensitivity_to_coarse_t0_jitter",
        "baseline_max_local_stretch": BASELINE_STRETCH,
        "stretch_values_tested": STRETCH_VALUES,
        "core_well": CORE_WELL,
        "core_experiment_fixed_t0": core,
        "side_effect_check_all_wells_blind_search": side_effect,
        "candidate_verdicts": candidate_verdicts,
        "verdict": {
            "baseline_delta_mae_ms": baseline_delta,
            "most_stable_stretch": core[best_key]["max_local_stretch"],
            "most_stable_delta_mae_ms": best_delta,
            "any_stretch_improves_core_without_regressing_other_wells": bool(any_net_win),
            "honest_conclusion": (
                "at least one MAX_LOCAL_STRETCH value reduces 19BT2's sensitivity to the "
                "coarse-t0 jitter without hurting 19A/19SR"
                if any_net_win else
                "no MAX_LOCAL_STRETCH value tested both reduces 19BT2's sensitivity to the "
                "coarse-t0 jitter AND avoids regressing at least one other well; window width "
                "alone does not fix the underlying instability"
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUTPUT_DIR / "summary.json")
    print(summary["verdict"]["honest_conclusion"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
