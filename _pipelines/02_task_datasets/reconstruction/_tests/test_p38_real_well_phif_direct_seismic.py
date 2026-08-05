from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK))

import p38_real_well_phif_direct_seismic as p38
import p38_foundation_feature_worker as worker
import p38_pilot_core as core
from _models.reconstruction import moment_well


def test_nearest_curve_uses_explicit_mask_instead_of_silent_zero_fill() -> None:
    depth = np.asarray([100.0, 100.1, 100.2, 100.3])
    value = np.asarray([1.0, -999.25, 3.0, 4.0])
    valid = value != -999.25
    matched, observed, spacing = p38._nearest_curve(  # noqa: SLF001
        np.asarray([100.0, 100.1, 100.2, 100.5]), depth, value, valid
    )
    assert spacing == pytest.approx(0.15)
    np.testing.assert_array_equal(observed, [True, False, True, False])
    assert matched[0] == 1.0
    assert np.isnan(matched[1])
    assert matched[2] == 3.0
    assert np.isnan(matched[3])


def test_centered_window_is_native_and_rejects_oversized_span() -> None:
    start = p38._centered_start(  # noqa: SLF001
        np.asarray([2202, 2205]),
        window_size=160,
        lower_bound=1932,
        upper_bound=2536,
    )
    assert start <= 2202 <= 2205 <= start + 159
    with pytest.raises(ValueError, match="cannot fit"):
        p38._centered_start(  # noqa: SLF001
            np.asarray([2000, 2200]),
            window_size=160,
            lower_bound=1932,
            upper_bound=2536,
        )


def test_phase0_verification_fails_closed_below_parent_coverage_gate() -> None:
    target = {
        "parents": {parent: {"target": {"zero_rows": 0}} for parent in p38.PARENT_ORDER},
        "semantic_gate_passed": True,
        "phif_is_distinct_from_phie": True,
    }
    alignment = {
        "alignment_gate_passed": True,
        "interpolation_applied": False,
        "padding_applied": False,
        "parents": {
            parent: {
                "retained_fraction": 0.89 if index == 1 else 1.0,
                "native_window": {"finite": True},
            }
            for index, parent in enumerate(p38.PARENT_ORDER)
        },
    }
    split = {
        "state": "FROZEN_ACTIVE",
        "folds": [{}, {}, {}],
        "split_before": ["encoder_call", "normalization"],
    }
    freeze = {
        "state": "PHASE0_PASSED",
        "firewall": {
            "phase0_encoder_calls": 0,
            "phase0_training_runs": 0,
            "test_h5_opened": False,
            "frozen_holdout_opened": False,
            "train_h5_opened": False,
        },
    }
    verification = p38._phase0_verification(  # noqa: SLF001
        target, alignment, split, freeze
    )
    assert verification["status"] == "FAIL_PHASE0"
    assert verification["checks"]["all_parent_coverages_at_least_90_percent"] is False


def test_path_firewall_rejects_forbidden_and_out_of_scope_paths(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        p38._assert_no_forbidden_paths([tmp_path / "test.h5"])  # noqa: SLF001
    with pytest.raises(ValueError, match="owned path"):
        p38._validate_output_path(tmp_path / "p38-output")  # noqa: SLF001
    with pytest.raises(ValueError, match="owned path"):
        p38._validate_output_path(tmp_path / "scratch", scratch=True)  # noqa: SLF001


def test_split_and_acceptance_constants_are_immutable_contract_values() -> None:
    assert p38.PARENT_ORDER == ("15/9-19", "15/9-F-11", "15/9-F-15")
    assert p38.MIN_PARENT_COVERAGE == 0.90
    assert p38.WELL_WINDOW == 33
    assert p38.SEISMIC_TIME_SAMPLES == 400
    assert p38.SEISMIC_TRACE_COUNT == 160
    assert p38.MAX_SCIENTIFIC_ITERATIONS == 3
    assert p38.HEAD_MAX_UPDATES == 120
    assert p38.TWT_PERTURBATION_SAMPLES == 40


def test_foundation_projection_is_deterministic_and_dimension_locked() -> None:
    values = np.arange(2 * 3 * 20, dtype=np.float32).reshape(2, 3, 20)
    first = worker._project(values, stream=7)  # noqa: SLF001
    second = worker._project(values, stream=7)  # noqa: SLF001
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 3, worker.PROJECTION_DIM)
    assert np.isfinite(first).all()


def test_reconstruction_moment_adapter_contract_is_six_by_33() -> None:
    contract = moment_well.capabilities()
    assert contract["task_types"] == ["reconstruction"]
    assert contract["input_shape"] == "[B,6,33]"
    assert contract["output_shape"] == "[B,6,4,768]"
    assert contract["supports_same_architecture_random_init"] is True


def test_parent_balanced_scaler_does_not_row_weight_large_parent() -> None:
    values = np.concatenate(
        [np.zeros((2, 1), dtype=np.float32), np.full((20, 1), 10.0, dtype=np.float32)]
    )
    parent = np.asarray([0, 0] + [1] * 20)
    mean, scale = core._balanced_location_scale(  # noqa: SLF001
        values, np.arange(len(values)), parent
    )
    assert mean[0] == pytest.approx(5.0)
    assert scale[0] == pytest.approx(5.0)


def test_common_head_parameter_budget_is_identical_across_modality_masks() -> None:
    model = core.BoundedFusionHead(maximum_gate=0.1)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count == core.parameter_count()
    x = np.zeros((3, core.FEATURE_DIM), dtype=np.float32)
    import torch

    for presence in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        output, gate = model(torch.from_numpy(x), torch.from_numpy(x), *presence)
        assert output.shape == (3,)
        assert gate.shape == (3, core.HIDDEN_DIM)


def test_paired_depth_block_bootstrap_detects_uniform_candidate_gain() -> None:
    target = np.linspace(0.05, 0.30, 120)
    parent = np.repeat(np.arange(3), 40)
    md = np.tile(np.arange(40, dtype=float), 3)
    control = target + 0.04
    candidate = target + 0.01
    result = core.paired_depth_block_bootstrap(
        target=target,
        candidate=candidate,
        control=control,
        parent_index=parent,
        md_m=md,
        block_m=10.0,
        draws=500,
        seed=2693,
    )
    assert result["ci95"][1] < 0.0
    assert result["probability_candidate_better"] == 1.0


def test_agent_action_uses_inner_metric_only_and_default_tie_break() -> None:
    allowlist = [
        {"action_id": "default"},
        {"action_id": "stronger_regularization"},
        {"action_id": "shorter_patience"},
    ]
    tied = {
        "default": {"macro_rmse": 0.1},
        "stronger_regularization": {"macro_rmse": 0.1},
        "shorter_patience": {"macro_rmse": 0.2},
    }
    assert p38.select_agent_action(allowlist, tied)["action_id"] == "default"
    improved = dict(tied)
    improved["stronger_regularization"] = {"macro_rmse": 0.09}
    assert (
        p38.select_agent_action(allowlist, improved)["action_id"]
        == "stronger_regularization"
    )
    with pytest.raises(ValueError, match="missing"):
        p38.select_agent_action(allowlist, {"default": {"macro_rmse": 0.1}})


def test_foundation_interpreter_contract_preserves_venv_symlink_text(tmp_path: Path) -> None:
    # Path.resolve() on venv/bin/python commonly points at /usr/bin/python and
    # silently drops the venv site-packages.  The command builder must retain
    # the caller-owned interpreter path; source inspection locks that bug fix.
    source = Path(p38.__file__).read_text(encoding="utf-8")
    assert "str(interpreter)" in source
    assert "Do not resolve this symlink" in source
    assert "Path(foundation_python).expanduser().resolve()" not in source
