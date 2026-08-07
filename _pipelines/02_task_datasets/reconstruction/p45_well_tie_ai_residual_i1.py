#!/usr/bin/env python3
"""P45 I1: does a small learned residual model beat the I0 physics baseline?

Honest framing of the sample-size constraint (see finding.md): only 3 Volve
wells have real DT+RHOB coverage, so there is no statistically defensible way
to train a well-level classifier (n=3). What IS defensible is a much smaller,
regularized residual-correction model trained on individual checkshot-depth
points (hundreds per well) pooled across two wells and evaluated, out-of-well,
on the third (leave-one-well-out, 3 folds) -- generalization is tested at the
well level, which is the level that actually matters for an unknown work
area.

Model: Ridge regression only (small, linear, hard to overfit with 2 training
wells) predicting residual = I0_predicted_twt - true_checkshot_twt from local
physical features (velocity, density, local reflectivity energy, normalized
depth position, and the well's own coarse ambiguity_gap/xcorr as constant
context features). If Ridge cannot beat "no correction" out-of-well, that is
reported as the honest result -- this stage must not force a positive
result (per the goal's own gate: "打不赢就诚实停在物理方法").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

from p45_well_tie_physics_baseline import (  # noqa: E402
    WELLS, CHECKSHOT_MEMBER, VSP_ZIP, SEGY_PATH,
    load_well_curve, parse_checkshots, predict_well, metrics,
)
from step_01_load_seismic import SeismicVolume, load_index  # noqa: E402

OUTPUT_DIR = HERE / "_outputs/p45_well_tie_ai_residual_i1"


def build_point_features(well: str, pred: dict, curve: dict) -> dict:
    """Per checkshot-depth-point features, aligned via interpolation onto
    the well's own predicted_twt_ms / depth_tvdss grid."""
    depth_true, twt_true = CHECKSHOTS[well]
    lo, hi = pred["depth_tvdss"].min(), pred["depth_tvdss"].max()
    in_range = (depth_true >= lo) & (depth_true <= hi)
    depth_true = depth_true[in_range]
    twt_true = twt_true[in_range]
    if len(depth_true) == 0:
        return {"n": 0}

    pred_twt = np.interp(depth_true, pred["depth_tvdss"], pred["predicted_twt_ms"])
    velocity = 1.0e6 * 0.3048 / curve["dt"]
    # reflectivity local energy proxy: rolling std of density*velocity (impedance)
    impedance = velocity * curve["rhob"]
    local_energy = np.zeros_like(impedance)
    win = 11
    for i in range(len(impedance)):
        lo_i, hi_i = max(0, i - win // 2), min(len(impedance), i + win // 2 + 1)
        local_energy[i] = float(np.std(impedance[lo_i:hi_i]))

    velocity_at = np.interp(depth_true, pred["depth_tvdss"], velocity)
    rhob_at = np.interp(depth_true, pred["depth_tvdss"], curve["rhob"])
    energy_at = np.interp(depth_true, pred["depth_tvdss"], local_energy)
    norm_depth = (depth_true - lo) / max(hi - lo, 1e-6)

    n = len(depth_true)
    # NOTE: deliberately excludes well-level constants (coarse_xcorr,
    # ambiguity_gap) as per-point features. With only 3 wells, a feature
    # that is CONSTANT within each well is statistically indistinguishable
    # from a well-identity indicator; a linear model trained on 2 such
    # constants can assign it a large, unstable coefficient and then
    # catastrophically extrapolate on the third well's very different
    # constant. (First run without this fix produced a 32-SECOND error on
    # 19BT2 -- kept as a documented dead end, not silently dropped.)
    features = np.column_stack([
        velocity_at,
        rhob_at,
        energy_at,
        norm_depth,
    ])
    residual = pred_twt - twt_true
    return {
        "n": n, "features": features, "residual": residual,
        "pred_twt": pred_twt, "true_twt": twt_true, "depth": depth_true,
    }


def main() -> int:
    global CHECKSHOTS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    CHECKSHOTS = parse_checkshots(VSP_ZIP, CHECKSHOT_MEMBER)

    per_well = {}
    with SeismicVolume(segy_path=SEGY_PATH, index=index) as vol:
        for well in WELLS:
            curve = load_well_curve(well)
            pred = predict_well(well, index, vol)
            per_well[well] = build_point_features(well, pred, curve)

    all_features = np.concatenate([per_well[w]["features"] for w in WELLS if per_well[w]["n"] > 0])
    feature_names = ["velocity_m_s", "rhob_g_cc", "local_impedance_std", "norm_depth"]

    results = {}
    groups_lookup = {w: i for i, w in enumerate(WELLS)}
    for held_out in WELLS:
        train_wells = [w for w in WELLS if w != held_out]
        if per_well[held_out]["n"] == 0 or any(per_well[w]["n"] == 0 for w in train_wells):
            results[held_out] = {"status": "skipped_no_points"}
            continue

        X_train = np.concatenate([per_well[w]["features"] for w in train_wells])
        y_train = np.concatenate([per_well[w]["residual"] for w in train_wells])
        groups_train = np.concatenate([
            np.full(per_well[w]["n"], groups_lookup[w]) for w in train_wells
        ])

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)

        # inner leave-one-well-out CV across the 2 training wells to pick alpha
        logo = LeaveOneGroupOut()
        best_alpha, best_cv_mae = None, np.inf
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            errs = []
            for tr_idx, va_idx in logo.split(X_train_s, y_train, groups_train):
                if len(np.unique(groups_train[tr_idx])) == 0:
                    continue
                model = Ridge(alpha=alpha).fit(X_train_s[tr_idx], y_train[tr_idx])
                pred_resid = model.predict(X_train_s[va_idx])
                errs.append(np.mean(np.abs(pred_resid - y_train[va_idx])))
            cv_mae = float(np.mean(errs)) if errs else np.inf
            if cv_mae < best_cv_mae:
                best_cv_mae, best_alpha = cv_mae, alpha

        model = Ridge(alpha=best_alpha).fit(X_train_s, y_train)

        X_test = scaler.transform(per_well[held_out]["features"])
        predicted_residual = model.predict(X_test)
        corrected_twt = per_well[held_out]["pred_twt"] - predicted_residual

        baseline_metrics = metrics(per_well[held_out]["true_twt"], per_well[held_out]["pred_twt"])
        corrected_metrics = metrics(per_well[held_out]["true_twt"], corrected_twt)

        results[held_out] = {
            "status": "evaluated",
            "n_train_points": int(len(y_train)),
            "n_test_points": int(per_well[held_out]["n"]),
            "selected_alpha": best_alpha,
            "inner_cv_mae_ms": best_cv_mae,
            "ridge_coefficients": dict(zip(feature_names, model.coef_.tolist())),
            "i0_baseline_metrics": baseline_metrics,
            "i1_corrected_metrics": corrected_metrics,
            "mae_delta_ms": corrected_metrics["mae_ms"] - baseline_metrics["mae_ms"],
            "improved": bool(corrected_metrics["mae_ms"] < baseline_metrics["mae_ms"]),
        }
        status = "IMPROVED" if results[held_out]["improved"] else "NOT_IMPROVED"
        print(f"{held_out}: I0 MAE={baseline_metrics['mae_ms']:.2f}ms -> "
              f"I1 MAE={corrected_metrics['mae_ms']:.2f}ms  delta={results[held_out]['mae_delta_ms']:+.2f}ms  [{status}]")

    n_evaluated = sum(1 for r in results.values() if r.get("status") == "evaluated")
    n_improved = sum(1 for r in results.values() if r.get("improved"))
    verdict = {
        "n_wells_evaluated": n_evaluated,
        "n_wells_improved": n_improved,
        "decision": (
            "I1_NOT_PROMOTED_INSUFFICIENT_EVIDENCE"
            if n_evaluated < 3 or n_improved < n_evaluated
            else "I1_CANDIDATE_ALL_WELLS_IMPROVED_STILL_N3_CAUTION"
        ),
        "caveat": (
            "Only 3 wells total; leave-one-well-out with 2 training wells per "
            "fold is the minimum defensible generalization test, not a "
            "statistically powered claim. Any 'improved' result here is a "
            "weak signal, not a promotion-grade result."
        ),
    }
    summary = {
        "feature_names": feature_names,
        "per_well_point_counts": {w: per_well[w]["n"] for w in WELLS},
        "results": results,
        "verdict": verdict,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nverdict:", verdict["decision"])
    print("wrote", OUTPUT_DIR / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
