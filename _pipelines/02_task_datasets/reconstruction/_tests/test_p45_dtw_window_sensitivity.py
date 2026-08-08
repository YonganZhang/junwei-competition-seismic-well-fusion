#!/usr/bin/env python3
"""Black-box tests for p45_dtw_window_sensitivity.py.

Runs the real sweep against real well/seismic data (no mocking of physics)
and checks structural + honesty invariants of the produced summary.json:
the module-global MAX_LOCAL_STRETCH must be restored after the sweep, the
reported deltas must match a fresh independent recomputation, and the
verdict flags must be internally consistent with the numbers they claim to
summarize.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "01_common_preprocess"))

import p45_dtw_window_sensitivity as mod  # noqa: E402
import p45_well_tie_physics_baseline as pb  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    mod.main()
    path = mod.OUTPUT_DIR / "summary.json"
    assert path.exists()
    return json.loads(path.read_text(encoding="utf-8"))


def test_module_global_restored_after_sweep(summary):
    # main() must leave the shared physics-baseline module state untouched
    # for any other script importing it afterward in the same interpreter.
    assert pb.MAX_LOCAL_STRETCH == mod.BASELINE_STRETCH == 1.20


def test_all_stretch_values_present(summary):
    keys = set(summary["core_experiment_fixed_t0"].keys())
    expected = {f"{v:.2f}" for v in mod.STRETCH_VALUES}
    assert keys == expected
    for well in ("19A", "19BT2", "19SR"):
        assert set(summary["side_effect_check_all_wells_blind_search"][well].keys()) == expected


def test_exactly_one_default_flagged(summary):
    flags = [row["is_current_default"] for row in summary["core_experiment_fixed_t0"].values()]
    assert sum(flags) == 1
    default_row = next(r for r in summary["core_experiment_fixed_t0"].values() if r["is_current_default"])
    assert default_row["max_local_stretch"] == pytest.approx(1.20)


def test_start_points_match_known_prior_run(summary):
    row = summary["core_experiment_fixed_t0"]["1.20"]
    # These are the real, previously-computed values from
    # p45_well_tie_physics_baseline (I0 blind) and p45_well_tie_regional_prior
    # for 19BT2 -- the exact 4.4ms jitter the task asked us to investigate.
    assert row["t0_i0_blind_ms"] == pytest.approx(2560.0, abs=1e-6)
    assert row["t0_regional_prior_ms"] == pytest.approx(2555.5650016284512, abs=1e-6)
    assert row["start_point_diff_ms"] == pytest.approx(4.4349983715488, abs=1e-3)


def test_baseline_delta_reproduces_diagnosed_129ms_gap(summary):
    # Sanity: at the CURRENT shipped default (1.20) this script must
    # reproduce the already-diagnosed ~129ms MAE swing, or the experiment
    # setup itself is wrong.
    row = summary["core_experiment_fixed_t0"]["1.20"]
    assert row["delta_mae_ms"] == pytest.approx(129.0, abs=2.0)


def test_delta_mae_matches_independent_recomputation(summary):
    for key, row in summary["core_experiment_fixed_t0"].items():
        recomputed = abs(row["mae_at_t0_prior_ms"] - row["mae_at_t0_i0_ms"])
        assert row["delta_mae_ms"] == pytest.approx(recomputed, abs=1e-6)


def test_verdict_most_stable_is_actual_minimum(summary):
    deltas = {k: v["delta_mae_ms"] for k, v in summary["core_experiment_fixed_t0"].items()}
    min_key = min(deltas, key=lambda k: deltas[k])
    min_stretch = summary["core_experiment_fixed_t0"][min_key]["max_local_stretch"]
    assert summary["verdict"]["most_stable_stretch"] == pytest.approx(min_stretch)
    assert summary["verdict"]["most_stable_delta_mae_ms"] == pytest.approx(deltas[min_key], abs=1e-6)


def test_candidate_verdict_flags_are_internally_consistent(summary):
    baseline_delta = summary["verdict"]["baseline_delta_mae_ms"]
    baseline_side = {
        well: summary["side_effect_check_all_wells_blind_search"][well]["1.20"]["mae_ms"]
        for well in ("19A", "19BT2", "19SR")
    }
    for key, cand in summary["candidate_verdicts"].items():
        core_row = summary["core_experiment_fixed_t0"][key]
        assert cand["improves_core_sensitivity"] == (core_row["delta_mae_ms"] < baseline_delta)
        regressed = [
            well for well in ("19A", "19BT2", "19SR")
            if summary["side_effect_check_all_wells_blind_search"][well][key]["mae_ms"] > baseline_side[well] + 1e-6
        ]
        assert set(cand["regressed_wells"]) == set(regressed)
        assert cand["no_regression_on_other_wells"] == (len(regressed) == 0)
        assert cand["net_recommended"] == (cand["improves_core_sensitivity"] and cand["no_regression_on_other_wells"])


def test_honest_no_net_win_verdict_matches_this_run(summary):
    # This is the real, honestly-measured outcome for this dataset: no
    # tested MAX_LOCAL_STRETCH both reduces the 19BT2 sensitivity AND
    # avoids regressing another well. If a future run of the sweep finds a
    # genuine net win, this assertion (and the task's framing) should be
    # revisited -- it is not a hardcoded "must always fail" gate, it is a
    # regression check against the specific dataset in this repo.
    assert summary["verdict"]["any_stretch_improves_core_without_regressing_other_wells"] is False
    # narrowing the window (below default) reduces the delta_mae (isolated
    # DTW sensitivity), directly contradicting the "widening helps" prior --
    # but it does so by making the good (I0) start point worse, not for free.
    narrower = summary["core_experiment_fixed_t0"]["1.05"]
    default = summary["core_experiment_fixed_t0"]["1.20"]
    assert narrower["delta_mae_ms"] < default["delta_mae_ms"]
    assert narrower["mae_at_t0_i0_ms"] > default["mae_at_t0_i0_ms"]


def test_no_nan_or_negative_mae_in_outputs(summary):
    for row in summary["core_experiment_fixed_t0"].values():
        for key in ("mae_at_t0_i0_ms", "mae_at_t0_prior_ms", "delta_mae_ms"):
            assert np.isfinite(row[key]) and row[key] >= 0
    for well_rows in summary["side_effect_check_all_wells_blind_search"].values():
        for row in well_rows.values():
            assert np.isfinite(row["mae_ms"]) and row["mae_ms"] >= 0
