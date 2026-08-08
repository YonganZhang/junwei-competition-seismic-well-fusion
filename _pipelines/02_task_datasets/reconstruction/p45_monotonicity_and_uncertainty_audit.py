#!/usr/bin/env python3
"""P45 audit: acceptance criteria (2) monotonicity/velocity sanity and
(6) does reported uncertainty (ambiguity_gap / coarse_xcorr) actually track
real error (MAE against the well's own checkshot)?

(2) Re-runs predict_well() for I0 (blind full-axis search) and the regional
prior method on all 3 wells (predicted_twt_ms is NOT stored in the existing
summary.json files, only aggregate metrics -- so this needs a live call,
not a re-fit of anything). For each (well, method) predicted MD->TWT curve:
  - checks predicted_twt_ms is strictly increasing with depth
  - converts adjacent-sample TWT slope into an implied interval velocity
    and flags points outside a plausible sedimentary-rock range.

(6) Reads ambiguity_gap / coarse_xcorr and the matching MAE from the THREE
already-computed summary.json files (I0 blind on ST0202, regional prior,
I0 blind on ST10010) -- no re-running of experiments -- and reports the
Spearman correlation, with an explicit small-sample caveat.

Honesty note: this script does not change any judgment thresholds to make
either baseline look better; it reports what the data says.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import WELLS, predict_well  # noqa: E402
from p45_well_tie_regional_prior import (  # noqa: E402
    PRIOR_HALF_WIDTH_MS, PRIOR_SEARCH_STEP_MS, build_regional_prior_t0,
)
from p45_well_tie_physics_baseline import (  # noqa: E402
    VSP_ZIP, CHECKSHOT_MEMBER, WELL_PICKS, parse_checkshots,
    parse_well_picks_full, load_well_curve,
)
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_monotonicity_and_uncertainty_audit"
I0_SUMMARY = HERE / "_outputs/p45_well_tie_physics_baseline/summary.json"
PRIOR_SUMMARY = HERE / "_outputs/p45_well_tie_regional_prior/summary.json"
ST10010_SUMMARY = HERE / "_outputs/p45_well_tie_st10010_robustness/summary.json"

# plausible interval-velocity range for clastic/carbonate sedimentary rocks
# (mudstone/shale ~1500-2500, sand/limestone up to ~6000 m/s)
VELOCITY_MIN_M_S = 1500.0
VELOCITY_MAX_M_S = 6000.0


# ---------------------------------------------------------------------------
# (2) monotonicity + implied velocity, evaluated on the actual predicted
# MD/TVDSS -> TWT curve returned by predict_well() (not stored in existing
# summary.json files, so this re-derives it live -- deterministic given the
# same inputs, no re-fitting/training involved).
# ---------------------------------------------------------------------------
def audit_curve(well: str, method: str, depth_tvdss: np.ndarray, predicted_twt_ms: np.ndarray) -> dict:
    order = np.argsort(depth_tvdss)
    depth = depth_tvdss[order]
    twt = predicted_twt_ms[order]

    depth_diff = np.diff(depth)
    twt_diff = np.diff(twt)
    n_steps = int(len(twt_diff))

    non_monotonic_mask = twt_diff <= 0.0
    n_non_monotonic = int(non_monotonic_mask.sum())
    non_monotonic_fraction = float(n_non_monotonic / n_steps) if n_steps else float("nan")

    # implied one-way interval velocity from TWT slope; only defined where
    # twt_diff > 0 (non-monotonic steps have no physical velocity -- an
    # infinite or negative value, so they are reported separately, not
    # silently included in the velocity range check).
    valid = twt_diff > 0.0
    velocity_m_s = np.full(n_steps, np.nan)
    velocity_m_s[valid] = 2000.0 * depth_diff[valid] / twt_diff[valid]

    in_range = valid & (velocity_m_s >= VELOCITY_MIN_M_S) & (velocity_m_s <= VELOCITY_MAX_M_S)
    n_velocity_checked = int(valid.sum())
    n_velocity_out_of_range = int(valid.sum() - in_range.sum())
    velocity_out_of_range_fraction = (
        float(n_velocity_out_of_range / n_velocity_checked) if n_velocity_checked else float("nan")
    )

    examples_non_monotonic = [
        {
            "depth_tvdss_m": float(depth[i]),
            "twt_ms_at_depth": float(twt[i]),
            "twt_ms_at_next_depth": float(twt[i + 1]),
            "twt_delta_ms": float(twt_diff[i]),
        }
        for i in np.where(non_monotonic_mask)[0][:5]
    ]
    examples_velocity_out_of_range = [
        {
            "depth_tvdss_m": float(depth[i]),
            "implied_velocity_m_s": float(velocity_m_s[i]),
        }
        for i in np.where(valid & ~in_range)[0][:5]
    ]

    return {
        "well": well,
        "method": method,
        "n_depth_samples": int(len(depth)),
        "n_steps": n_steps,
        "strictly_monotonic_increasing": bool(n_non_monotonic == 0),
        "n_non_monotonic_steps": n_non_monotonic,
        "non_monotonic_fraction": non_monotonic_fraction,
        "examples_non_monotonic": examples_non_monotonic,
        "velocity_range_checked_m_s": [VELOCITY_MIN_M_S, VELOCITY_MAX_M_S],
        "n_velocity_steps_checked": n_velocity_checked,
        "n_velocity_out_of_range": n_velocity_out_of_range,
        "velocity_out_of_range_fraction": velocity_out_of_range_fraction,
        "examples_velocity_out_of_range": examples_velocity_out_of_range,
        "velocity_m_s_min_observed": float(np.nanmin(velocity_m_s)) if n_velocity_checked else None,
        "velocity_m_s_max_observed": float(np.nanmax(velocity_m_s)) if n_velocity_checked else None,
        "velocity_m_s_median_observed": float(np.nanmedian(velocity_m_s)) if n_velocity_checked else None,
    }


def run_monotonicity_audit(index: dict) -> list[dict]:
    checkshots = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)
    all_picks = parse_well_picks_full(WELL_PICKS)

    curve_audits: list[dict] = []
    with SeismicVolume(index=index) as vol:
        for well in WELLS:
            # I0: blind full-axis search (module default T0_SEARCH_MS)
            pred_i0 = predict_well(well, index, vol)
            curve_audits.append(
                audit_curve(well, "I0_blind_ST0202", pred_i0["depth_tvdss"], pred_i0["predicted_twt_ms"])
            )

            # regional prior: same construction as p45_well_tie_regional_prior.py
            curve = load_well_curve(well)
            prior_t0 = build_regional_prior_t0(well, checkshots, all_picks, curve)
            candidates = np.arange(
                max(0.0, prior_t0 - PRIOR_HALF_WIDTH_MS),
                prior_t0 + PRIOR_HALF_WIDTH_MS,
                PRIOR_SEARCH_STEP_MS,
            )
            pred_prior = predict_well(well, index, vol, t0_candidates=candidates)
            curve_audits.append(
                audit_curve(well, "regional_prior", pred_prior["depth_tvdss"], pred_prior["predicted_twt_ms"])
            )
    return curve_audits


# ---------------------------------------------------------------------------
# (6) uncertainty (ambiguity_gap / coarse_xcorr) vs realized error (MAE),
# read from the already-computed summary.json files -- no re-running.
# ---------------------------------------------------------------------------
def collect_uncertainty_points() -> list[dict]:
    points: list[dict] = []

    i0 = json.loads(I0_SUMMARY.read_text(encoding="utf-8"))
    for well, r in i0["wells"].items():
        points.append({
            "well": well,
            "method": "I0_blind_ST0202",
            "ambiguity_gap": r["ambiguity_gap"],
            "coarse_xcorr": r["coarse_xcorr"],
            "mae_ms": r["metrics_vs_own_checkshot"]["mae_ms"],
        })

    prior = json.loads(PRIOR_SUMMARY.read_text(encoding="utf-8"))
    for well, r in prior["results"].items():
        points.append({
            "well": well,
            "method": "regional_prior",
            "ambiguity_gap": r["ambiguity_gap"],
            "coarse_xcorr": r["coarse_xcorr"],
            "mae_ms": r["metrics_with_prior"]["mae_ms"],
        })

    st10010 = json.loads(ST10010_SUMMARY.read_text(encoding="utf-8"))
    for well, r in st10010["results"].items():
        if r.get("status") != "evaluated":
            continue
        points.append({
            "well": well,
            "method": "I0_blind_ST10010",
            "ambiguity_gap": r["st10010_ambiguity_gap"],
            "coarse_xcorr": r["st10010_xcorr"],
            "mae_ms": r["st10010_metrics"]["mae_ms"],
        })

    return points


def spearman_report(x: list[float], y: list[float], x_name: str, y_name: str) -> dict:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    n = int(finite.sum())
    if n < 3:
        return {
            "x": x_name, "y": y_name, "n": n,
            "spearman_r": None, "p_value": None,
            "note": "n<3, correlation not computable",
        }
    rho, p = spearmanr(x_arr[finite], y_arr[finite])
    return {
        "x": x_name,
        "y": y_name,
        "n": n,
        "spearman_r": float(rho),
        "p_value": float(p),
        "note": (
            "n<10: correlation estimate is LOW CONFIDENCE, reported as-is, "
            "not treated as a validated relationship"
        ),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()

    print("=== (2) monotonicity + implied-velocity audit ===")
    curve_audits = run_monotonicity_audit(index)
    for a in curve_audits:
        flag = "OK" if a["strictly_monotonic_increasing"] else "NON-MONOTONIC"
        print(
            f"{a['well']:6s} {a['method']:18s} monotonic={flag} "
            f"non_mono={a['n_non_monotonic_steps']}/{a['n_steps']} "
            f"velocity_out_of_range={a['n_velocity_out_of_range']}/{a['n_velocity_steps_checked']} "
            f"v_range_observed=[{a['velocity_m_s_min_observed']:.0f},{a['velocity_m_s_max_observed']:.0f}]m/s"
            if a["velocity_m_s_min_observed"] is not None else
            f"{a['well']:6s} {a['method']:18s} monotonic={flag} no valid velocity steps"
        )

    total_steps = sum(a["n_steps"] for a in curve_audits)
    total_non_monotonic = sum(a["n_non_monotonic_steps"] for a in curve_audits)
    n_curves_non_monotonic = sum(1 for a in curve_audits if not a["strictly_monotonic_increasing"])
    total_velocity_checked = sum(a["n_velocity_steps_checked"] for a in curve_audits)
    total_velocity_out = sum(a["n_velocity_out_of_range"] for a in curve_audits)

    print("\n=== (6) uncertainty (gap/xcorr) vs realized MAE ===")
    points = collect_uncertainty_points()
    for p in points:
        print(f"{p['well']:6s} {p['method']:18s} gap={p['ambiguity_gap']:.4f} "
              f"xcorr={p['coarse_xcorr']:.3f} MAE={p['mae_ms']:.1f}ms")

    gap = [p["ambiguity_gap"] for p in points]
    xcorr = [p["coarse_xcorr"] for p in points]
    mae = [p["mae_ms"] for p in points]
    corr_gap = spearman_report(gap, mae, "ambiguity_gap", "mae_ms")
    corr_xcorr = spearman_report(xcorr, mae, "coarse_xcorr", "mae_ms")
    print(f"\nspearman(ambiguity_gap, MAE) = {corr_gap['spearman_r']} (p={corr_gap['p_value']}, n={corr_gap['n']})")
    print(f"spearman(coarse_xcorr, MAE)  = {corr_xcorr['spearman_r']} (p={corr_xcorr['p_value']}, n={corr_xcorr['n']})")

    expected_gap_sign = "negative (higher gap = less ambiguous = expected lower MAE)"
    gap_matches_expectation = (
        corr_gap["spearman_r"] is not None and corr_gap["spearman_r"] < 0
    )
    expected_xcorr_sign = "negative (higher xcorr = better fit = expected lower MAE)"
    xcorr_matches_expectation = (
        corr_xcorr["spearman_r"] is not None and corr_xcorr["spearman_r"] < 0
    )

    summary = {
        "criterion_2_monotonicity_and_velocity": {
            "velocity_range_checked_m_s": [VELOCITY_MIN_M_S, VELOCITY_MAX_M_S],
            "curves": curve_audits,
            "aggregate": {
                "n_curves": len(curve_audits),
                "n_curves_strictly_monotonic": len(curve_audits) - n_curves_non_monotonic,
                "n_curves_non_monotonic": n_curves_non_monotonic,
                "total_depth_steps": total_steps,
                "total_non_monotonic_steps": total_non_monotonic,
                "overall_non_monotonic_step_fraction": (
                    float(total_non_monotonic / total_steps) if total_steps else None
                ),
                "total_velocity_steps_checked": total_velocity_checked,
                "total_velocity_out_of_range": total_velocity_out,
                "overall_velocity_out_of_range_fraction": (
                    float(total_velocity_out / total_velocity_checked) if total_velocity_checked else None
                ),
            },
        },
        "criterion_6_uncertainty_vs_error": {
            "points": points,
            "n_points": len(points),
            "spearman_ambiguity_gap_vs_mae": corr_gap,
            "spearman_coarse_xcorr_vs_mae": corr_xcorr,
            "expected_relationship": {
                "ambiguity_gap": expected_gap_sign,
                "coarse_xcorr": expected_xcorr_sign,
            },
            "observed_matches_expectation": {
                "ambiguity_gap": gap_matches_expectation,
                "coarse_xcorr": xcorr_matches_expectation,
            },
            "caveat": (
                f"Only {len(points)} (well,method) points from 3 wells across 3 already-run "
                "experiments -- not independent samples in the statistical sense (same 3 wells "
                "reused 3 times), and far below any n needed for a reliable correlation estimate. "
                "Any correlation value here is a weak, low-confidence signal, not evidence that the "
                "uncertainty measures are calibrated."
            ),
        },
        "verdict": {
            "criterion_2_passed": n_curves_non_monotonic == 0 and total_velocity_out == 0,
            "criterion_6_uncertainty_tracks_error": bool(
                gap_matches_expectation and xcorr_matches_expectation
            ),
            "note": (
                "criterion_6 verdict is a directional read of a low-confidence correlation "
                "(n<=9), not a statistically validated claim."
            ),
        },
    }

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
