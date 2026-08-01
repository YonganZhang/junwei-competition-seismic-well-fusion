from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(PROJECT_ROOT / "_code") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "_code"))


import p28_agentic_optimization as p28  # noqa: E402


def test_split_is_disjoint_and_family_scoped() -> None:
    records = p28.load_records()
    split = p28.split_records(records)
    train_families = {record.family_id for record in split["train"]}
    selection_families = {record.family_id for record in split["selection_dev"]}
    promotion_families = {record.family_id for record in split["promotion_dev"]}
    assert train_families == {"15/9-F-11", "15/9-F-1"}
    assert selection_families == {"15/9-19"}
    assert promotion_families == {"15/9-19"}
    assert {record.well_id for record in split["selection_dev"]} == {"15/9-19 A"}
    assert {record.well_id for record in split["promotion_dev"]} == {"15/9-19 BT2", "15/9-19 SR"}
    assert not ({record.sample_id for record in split["selection_dev"]} & {record.sample_id for record in split["promotion_dev"]})


def test_anonymous_deepseek_prompt_hides_route_names_and_is_strict_json(tmp_path: Path, monkeypatch) -> None:
    fake_pilots = [
        {"route_id": route.route_id, "blocked": False, "feedback": "flat"}
        for route in p28.ROUTES
    ]
    prompt = p28.strict_anonymous_prompt(fake_pilots, fake_pilots, "abc123")
    prompt_text = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
    assert "tiny_mlp" not in prompt_text
    assert "reservoir_linear" not in prompt_text
    assert "reservoir_ridge" not in prompt_text
    assert prompt["schema"]["action"] == "select|stop"
    assert len(prompt["routes"]) == 4
    monkeypatch.setattr(p28, "call_deepseek", lambda payload: {"action": "select", "route_id": "route_2", "reason": "ok"})
    decision = p28.build_deepseek_decision(fake_pilots, "abc123", tmp_path / "prompt.json")
    assert decision["route_id"] == "route_2"
    assert (tmp_path / "prompt.json").is_file()


def test_selection_helpers_choose_expected_route() -> None:
    pilots = [
        {"route_id": "tiny_mlp_default", "blocked": False, "selection": {"physical_MAE_macro": 3.0}},
        {"route_id": "tiny_mlp_l2", "blocked": False, "selection": {"physical_MAE_macro": 2.0}},
        {"route_id": "reservoir_ridge", "blocked": False, "selection": {"physical_MAE_macro": 4.0}},
        {"route_id": "reservoir_linear", "blocked": False, "selection": {"physical_MAE_macro": 5.0}},
    ]
    assert p28.choose_a2d_route(pilots) == "tiny_mlp_l2"
    primary_pilots = [
        {"route_id": "tiny_mlp_default", "blocked": False, "selection": {"composite_mean_train_std_normalized_RMSE": 3.0}},
        {"route_id": "tiny_mlp_l2", "blocked": False, "selection": {"composite_mean_train_std_normalized_RMSE": 2.0}},
        {"route_id": "reservoir_ridge", "blocked": False, "selection": {"composite_mean_train_std_normalized_RMSE": 4.0}},
        {"route_id": "reservoir_linear", "blocked": False, "selection": {"composite_mean_train_std_normalized_RMSE": 5.0}},
    ]
    assert p28.choose_a4_route(primary_pilots) == "tiny_mlp_l2"
    assert p28.choose_a3_route(pilots, seed=2693) in {pilot["route_id"] for pilot in pilots}


def test_cig_gate_records_expected_blocker() -> None:
    gate = p28.gather_cig_gate()
    assert gate["status"] == "blocked"
    assert "404" in gate["reason"] or "mismatch" in gate["reason"].lower()
    assert "CIG-Bench-Property.pth" in gate["blocked_checkpoints"][0]


def test_identity_replay_and_primary_metric_alignment(monkeypatch) -> None:
    class FakeModel:
        pass

    fake_prediction = np.array([[0.1, 0.2, 0.3]], dtype=float)
    fake_metrics = {
        "physical_MAE_macro": 1.0,
        "composite_mean_train_std_normalized_RMSE": 0.9,
        "worst_group_RMSE": {"PHIF": 0.1, "KLOGH": 0.2, "SW": 0.3},
    }

    monkeypatch.setattr(p28, "load_a0_checkpoint", lambda n_features: FakeModel())
    monkeypatch.setattr(p28, "infer", lambda model, features, stats: fake_prediction.copy())
    monkeypatch.setattr(p28, "evaluate_predictions", lambda actual, predicted, stats: dict(fake_metrics))
    monkeypatch.setattr(p28, "prediction_hash", lambda array: "hash-a0" if array is fake_prediction or np.array_equal(array, fake_prediction) else "hash-b")

    replay = p28.identity_replay(
        train_features=np.zeros((1, 2)),
        selection_features=np.zeros((1, 2)),
        selection_targets=np.zeros((1, 3)),
        promotion_features=np.zeros((1, 2)),
        promotion_targets=np.zeros((1, 3)),
        stats={},
        a0_selection_hash="hash-a0",
        a0_promotion_hash="hash-a0",
    )
    assert replay["kind"] == "identity_replay"
    assert replay["selection_hash_matches_a0"] is True
    assert replay["promotion_hash_matches_a0"] is True
    assert replay["executor"] == "a0_frozen_tiny_mlp"
    assert replay["action"] == "identity_replay"

    assert p28.compare_primary_to_baseline(fake_metrics, {"composite_mean_train_std_normalized_RMSE": 1.0}) == "improved"
    assert p28.compare_primary_to_baseline(fake_metrics, {"composite_mean_train_std_normalized_RMSE": 0.9}) == "flat"
    assert p28.compare_primary_to_baseline(fake_metrics, {"composite_mean_train_std_normalized_RMSE": 0.8}) == "worse"


def test_build_protocol_registers_a1_and_a4_and_primary_metric() -> None:
    records = p28.load_records()
    split = p28.split_records(records)
    route_pilots = [
        {"route_id": route.route_id, "blocked": False, "selection": {"composite_mean_train_std_normalized_RMSE": float(index + 1)}}
        for index, route in enumerate(p28.ROUTES)
    ]
    protocol = p28.build_protocol(
        split=split,
        a0_selection_hash="sel-hash",
        a0_promotion_hash="prom-hash",
        gate=p28.gather_cig_gate(),
        route_pilots=route_pilots,
        strategy_choice={
            "A1": {"selected_by": "identity_replay", "chosen_route_id": "A1_identity"},
            "A2L": {"selected_by": "blocked", "chosen_route_id": None, "deepseek": {"action": "stop", "reason": "x", "route_id": None}},
            "A2D": {"selected_by": "deterministic_pilot_rank", "chosen_route_id": "tiny_mlp_default"},
            "A3": {"selected_by": "pcg64_no_replacement", "chosen_route_id": "tiny_mlp_default"},
            "A4": {"selected_by": "deterministic_primary_metric_search", "chosen_route_id": "tiny_mlp_default"},
        },
        budget_steps=32,
        seed_pool=(2693, 2694, 2695),
    )
    assert protocol["primary_metric"]["documented"] == "composite_mean_train_std_normalized_RMSE"
    assert protocol["primary_metric"]["implemented_causal_metric"] == "physical_MAE_macro"
    assert protocol["arms"]["A1"]["kind"] == "identity_replay"
    assert protocol["arms"]["A4"]["kind"] == "deterministic_primary_metric_search"
    assert protocol["promotion_gate"]["primary_metric"] == "composite_mean_train_std_normalized_RMSE"
