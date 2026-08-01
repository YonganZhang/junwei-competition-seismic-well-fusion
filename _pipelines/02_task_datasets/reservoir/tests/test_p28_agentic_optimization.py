from __future__ import annotations

import json
import sys
from pathlib import Path


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
    assert p28.choose_a3_route(pilots, seed=2693) in {pilot["route_id"] for pilot in pilots}


def test_cig_gate_records_expected_blocker() -> None:
    gate = p28.gather_cig_gate()
    assert gate["status"] == "blocked"
    assert "404" in gate["reason"] or "mismatch" in gate["reason"].lower()
    assert "CIG-Bench-Property.pth" in gate["blocked_checkpoints"][0]
