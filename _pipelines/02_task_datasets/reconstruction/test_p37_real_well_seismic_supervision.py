from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import p37_real_well_seismic_supervision as p37


def test_actual_survey_parser_recovers_datum_coordinates_and_monotonic_rows() -> None:
    text = """
RT #2 @ 25.00mTVD Reference:
6,478,563.53
435,050.02
109.00 0.00 0.00 109.00 6.89 7.02 0.00 0.000 0.000 0.000
200.00 1.00 2.00 199.50 8.00 9.00 1.00 0.100 0.200 0.300
"""
    survey = p37.parse_survey_text(text)
    np.testing.assert_array_equal(survey["md_m"], [109.0, 200.0])
    np.testing.assert_array_equal(survey["tvd_m_from_reference"], [109.0, 199.5])
    np.testing.assert_allclose(survey["northing_m"], [6478570.42, 6478571.53])
    np.testing.assert_allclose(survey["easting_m"], [435057.04, 435059.02])
    assert survey["reference_elevation_m_above_msl"] == 25.0


def test_curve_equivalence_metrics_detects_non_alias_even_when_correlated() -> None:
    depth = np.asarray([100.0, 101.0, 102.0, 103.0])
    phie = np.asarray([0.10, 0.20, 0.30, 0.40])
    phif = np.asarray([0.08, 0.18, 0.28, 0.38])
    result = p37.curve_equivalence_metrics(depth, phie, depth, phif)
    assert result["overlap_rows"] == 4
    assert result["mae_fraction"] == pytest.approx(0.02)
    assert result["rmse_fraction"] == pytest.approx(0.02)
    assert result["correlation"] == pytest.approx(1.0)
    assert result["elementwise_equal"] is False


def test_target_token_inventory_counts_parent_wells_not_branches() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "Well_logs/06.LFP/15_9-19 A/a.las",
            "~Curve\nLFP_PHIE.v/v\n",
        )
        archive.writestr(
            "Well_logs/06.LFP/15_9-19 BT2/b.las",
            "~Curve\nLFP_PHIE.v/v\n",
        )
        archive.writestr(
            "Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-F-11 T2/c.las",
            "~Curve\nPHIF.v/v\n",
        )
    buffer.seek(0)
    with ZipFile(buffer) as archive:
        result = p37.scan_target_tokens(archive)
    assert result["parent_count_with_literal_phie"] == 1
    assert result["literal_phie_parents"] == ["15/9-19"]
    assert result["parent_count_with_phif"] == 1
    assert sorted(result["parents"]) == ["15/9-19", "15/9-F-11"]


def test_development_path_firewall_rejects_test_and_holdout_names(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="forbidden"):
        p37._assert_development_path(tmp_path / "test.h5")  # noqa: SLF001
    with pytest.raises(RuntimeError, match="forbidden"):
        p37._assert_development_path(tmp_path / "frozen_holdout" / "train.h5")  # noqa: SLF001


def test_verification_locks_fail_closed_supervision_decision() -> None:
    summary = {
        "decision": {
            "state": p37.DECISION,
            "pilot_run": False,
            "p21_remains_default": True,
            "eclipse_grid_proxy_substituted": False,
        },
        "supervision_gate": {
            "independent_parent_wells_audited": 3,
            "literal_native_phie_parent_wells_available": 1,
            "parents_with_legal_train_kji": ["15/9-19", "15/9-F-15"],
        },
        "phie_phif_same_well_audit": {"elementwise_equal": False},
        "well_profiles": {
            "F11T2": {"alignment_join": {"legal_train_kji_rows": 0}}
        },
        "baseline_preservation": {
            "p21_rmse": p37.P21_RMSE,
            "p30_decision": "FEASIBLE_NO_PROMOTION",
        },
        "firewall": {
            "development_hdf5_opened": ["train.h5"],
            "hdf5_label_datasets_read": [],
            "frozen_holdout_opened": False,
        },
    }
    verification = p37._verification(  # noqa: SLF001
        summary, {"raw_archives_modified": False}
    )
    assert verification["status"] == "PASS_BLOCKED_EVIDENCE"
    assert all(verification["checks"].values())


def test_depth_unit_conversion_for_dlis_tenth_inch() -> None:
    values = np.asarray([0.0, 10.0, 100.0])
    np.testing.assert_allclose(
        p37._depth_to_m(values, "0.1 in"),  # noqa: SLF001
        [0.0, 0.0254, 0.254],
    )
