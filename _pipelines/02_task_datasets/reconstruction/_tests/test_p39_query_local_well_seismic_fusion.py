from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))

import p38_pilot_core as p38_core
import p38_real_well_phif_direct_seismic as p38
import p39_query_local_well_seismic_fusion as p39


def _synthetic_rows() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2693)
    sections = rng.normal(size=(3, 3, 400, 160)).astype(np.float32)
    return {
        "native_sections": sections,
        "section_id": np.asarray([0, 0, 1, 2], dtype=np.int32),
        "trace_token_id": np.asarray([80, 80, 70, 90], dtype=np.int16),
        "local_time_index": np.asarray([120, 140, 180, 220], dtype=np.int16),
        "time_position": np.asarray([120, 140, 180, 220], dtype=np.float32) / 399.0,
    }


def _small_learning_problem() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(2693)
    parent = np.repeat(np.arange(3), 8)
    well = rng.normal(size=(24, 12)).astype(np.float32)
    seismic = rng.normal(size=(24, 15)).astype(np.float32)
    base = np.clip(0.2 + 0.02 * well[:, 0], 0.0, 1.0).astype(np.float32)
    target = np.clip(base + 0.01 * seismic[:, 0], 0.0, 1.0).astype(np.float32)
    return well, seismic, base, target, parent


def test_zero_gate_fallback_is_exact_array_equality() -> None:
    base = np.asarray([0.0, 0.1, 0.5, 1.0], dtype=np.float32)
    correction = np.asarray([-0.5, 0.2, 0.5, -0.4], dtype=np.float32)
    result = p39.anchored_prediction(base, np.zeros_like(base), correction)
    np.testing.assert_array_equal(result, base)


def test_anchored_head_matches_p38_parameter_ceiling_and_shapes() -> None:
    import torch

    assert p39.anchored_parameter_count() == p38_core.parameter_count() == 6305
    model = p39.AnchoredResidualHead(maximum_gate=0.1, correction_bound=0.5)
    delta, gate, correction = model(
        torch.zeros((5, 32), dtype=torch.float32),
        torch.zeros((5, 32), dtype=torch.float32),
    )
    assert delta.shape == gate.shape == correction.shape == (5,)
    assert torch.all(gate > 0.0)
    assert torch.all(gate < 0.1)
    assert torch.all(torch.abs(correction) <= 0.5)


def test_local_waveform_features_move_the_native_query() -> None:
    arrays = _synthetic_rows()
    local, audit = p39.local_waveform_features(arrays, shift_samples=0)
    shifted, shifted_audit = p39.local_waveform_features(arrays, shift_samples=40)
    assert local.shape == shifted.shape == (4, 60)
    assert audit["clipped_rows"] == 0
    assert shifted_audit["shift_samples"] == 40
    assert np.all(np.any(local != shifted, axis=1))
    # Two depths on the same section/trace must receive distinct local evidence.
    assert not np.array_equal(local[0], local[1])


def test_combined_representation_retains_global_and_replaces_all_position_terms() -> None:
    arrays = _synthetic_rows()
    local, _ = p39.local_waveform_features(arrays, shift_samples=0)
    gfm = np.arange(4 * 99, dtype=np.float32).reshape(4, 99)
    override = np.full(4, 0.75, dtype=np.float32)
    combined = p39.combine_local_global(gfm, local, position_override=override)
    assert combined.shape == (4, 159)
    np.testing.assert_array_equal(combined[:, :96], gfm[:, :96])
    np.testing.assert_allclose(combined[:, 96:99], p39._position_features(override))  # noqa: SLF001
    np.testing.assert_array_equal(combined[:, 99:], local)


def test_cyclic_map_replaces_complete_seismic_evidence() -> None:
    parent = np.repeat(np.arange(3), [4, 5, 6])
    md = np.concatenate([np.arange(4), np.arange(5), np.arange(6)]).astype(float)
    evidence = np.arange(len(parent) * 159, dtype=np.float32).reshape(len(parent), 159)
    mapping = p38._cyclic_row_map(parent, md)  # noqa: SLF001
    replaced = evidence[mapping]
    np.testing.assert_array_equal(replaced, evidence[mapping])
    assert np.all(parent[mapping] != parent)
    assert np.all(np.any(replaced != evidence, axis=1))


def test_outer_held_target_can_be_nonfinite_without_being_read_for_fit() -> None:
    well, seismic, base, target, parent = _small_learning_problem()
    held = parent == 0
    target_with_firewall = target.copy()
    target_with_firewall[held] = np.nan
    train = np.flatnonzero(~held)
    config = {
        "action_id": "default",
        "early_stopping_patience": 20,
        "gate_strength": 0.1,
        "weight_decay": 1e-4,
    }
    bundle, predicted = p39.fit_anchored_head(
        well_features=well,
        seismic_features=seismic,
        base_prediction=base,
        target=target_with_firewall,
        parent_index=parent,
        train_indices=train,
        validation_indices=None,
        config=config,
        seed=2693,
        device="cpu",
        fixed_steps=2,
    )
    assert predicted is None
    assert bundle.audit["held_parent_target_used"] is False
    held_prediction, _, _ = bundle.predict(
        well_features=well,
        seismic_features=seismic,
        base_prediction=base,
        indices=np.flatnonzero(held),
    )
    assert np.isfinite(held_prediction).all()


def test_cross_fitted_residual_inner_logo_is_deterministic_and_outer_blind() -> None:
    well, seismic, base, target, parent = _small_learning_problem()
    target[parent == 0] = np.nan
    config = {
        "action_id": "default",
        "early_stopping_patience": 5,
        "gate_strength": 0.1,
        "weight_decay": 1e-4,
    }
    kwargs = dict(
        well_features=well,
        seismic_features=seismic,
        base_prediction=base,
        target=target,
        parent_index=parent,
        outer_train_parents=[1, 2],
        config=config,
        seed=2693,
        device="cpu",
        stream=9,
        fixed_steps=2,
    )
    first = p39.inner_logo_anchored(**kwargs)
    second = p39.inner_logo_anchored(**kwargs)
    selected = first["selected_indices"]
    np.testing.assert_array_equal(first["predictions"][selected], second["predictions"][selected])
    assert set(np.unique(parent[selected])) == {1, 2}
    assert all(fold["selected_steps"] == 2 for fold in first["folds"])
    assert all(
        fold["training_audit"]["base_targets_are_outer_train_cross_fitted"]
        for fold in first["folds"]
    )


def test_weight_control_routing_is_complete_and_non_aliasing() -> None:
    assert p39.WEIGHT_CONTROLS == (
        "moment_random_gfm_random",
        "moment_pretrained_gfm_random",
        "moment_random_gfm_pretrained",
        "moment_pretrained_gfm_pretrained",
    )
    assert len(set(p39.WEIGHT_CONTROLS)) == 4
    assert p39.FINAL_CANDIDATE == "moment_pretrained_gfm_pretrained"


def test_all_weight_controls_share_exact_pretrained_locked_base_per_fold() -> None:
    parent = np.repeat(np.arange(3), 4)
    shared = {
        fold: np.linspace(0.01 + fold, 0.12 + fold, len(parent), dtype=np.float32)
        for fold in range(3)
    }
    control_bases = {name: shared for name in p39.WEIGHT_CONTROLS}
    audit = p39.fixed_base_attribution_audit(
        control_bases=control_bases,
        parent_index=parent,
    )
    assert audit["all_controls_all_folds_exact_shared_base"] is True
    for name in p39.WEIGHT_CONTROLS:
        for fold in range(3):
            row = audit["controls"][name]["folds"][str(fold)]
            assert row["base_sha256"] == row["shared_base_sha256"]
            assert row["max_abs_diff_vs_shared"] == 0.0
            assert row["outer_train_max_abs_diff_vs_shared"] == 0.0
            assert row["held_max_abs_diff_vs_shared"] == 0.0


def test_fixed_base_audit_fails_closed_on_random_moment_base_substitution() -> None:
    parent = np.repeat(np.arange(3), 4)
    shared = {
        fold: np.linspace(0.01 + fold, 0.12 + fold, len(parent), dtype=np.float32)
        for fold in range(3)
    }
    control_bases = {
        name: {fold: values.copy() for fold, values in shared.items()}
        for name in p39.WEIGHT_CONTROLS
    }
    control_bases["moment_random_gfm_random"][1][5] += 1e-3
    with pytest.raises(RuntimeError, match="exact locked P38 well base"):
        p39.fixed_base_attribution_audit(
            control_bases=control_bases,
            parent_index=parent,
        )


def test_promotion_logic_requires_every_boolean_gate_but_not_count_field() -> None:
    acceptance = {
        "macro_below_locked_well_only": True,
        "wins_at_least_two_parents": True,
        "parent_wins": 2,
        "bootstrap_ci95_upper_below_zero": True,
        "both_pretrained_below_both_random": True,
        "both_pretrained_below_moment_pretrained_gfm_random": True,
        "both_pretrained_below_moment_random_gfm_pretrained": True,
        "cyclic_mismatch_degrades_macro_and_two_parents": True,
        "full_twt_mismatch_degrades_macro_and_two_parents": True,
        "train_only_calibration_finite_nonvacuous": True,
        "zero_gate_exact_fallback": True,
        "firewall_and_provenance_pass": True,
    }
    assert p39.promotion_state(acceptance) == (True, "PROMOTABLE_PILOT_SIGNAL")
    acceptance["full_twt_mismatch_degrades_macro_and_two_parents"] = False
    assert p39.promotion_state(acceptance) == (False, "FEASIBLE_NO_PROMOTION")


def test_phase0_freeze_and_verification_reproduce() -> None:
    output = TRACK / "_outputs/p39_query_local_well_seismic_fusion"
    result = p39._phase0_checks(output)  # noqa: SLF001
    assert result["status"] == "PASS_PHASE0"
    assert all(result["checks"].values())


def test_final_artifacts_verify_independently_when_present() -> None:
    output = TRACK / "_outputs/p39_query_local_well_seismic_fusion"
    summary = output / "summary.json"
    fixed_base = output / "fixed_base_attribution_audit.json"
    if not summary.is_file() or not fixed_base.is_file():
        pytest.skip("P39.1 durable final evidence has not been generated yet")
    result = p39._recompute_final(output)  # noqa: SLF001
    assert result["status"] == "PASS"
    assert all(result["checks"].values())


def test_path_firewall_rejects_forbidden_and_out_of_scope_paths(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        p39._assert_legal_paths([tmp_path / "test.h5"])  # noqa: SLF001
    with pytest.raises(ValueError, match="scratch"):
        p39._validate_scratch(tmp_path / "cache")  # noqa: SLF001
    with pytest.raises(ValueError, match="reconstruction"):
        p39._validate_output(tmp_path / "output")  # noqa: SLF001
