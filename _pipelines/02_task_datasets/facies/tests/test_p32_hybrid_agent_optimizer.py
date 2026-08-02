from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import p32_hybrid_agent_optimizer as p32  # noqa: E402


def _candidate(scale: float, lr: float, dice: float, frozen: bool) -> dict:
    return {
        "fusion_scale_initial": scale,
        "fusion_lr": lr,
        "dice_weight": dice,
        "sam2_frozen": frozen,
        "rationale": "bounded joint configuration",
    }


def test_candidate_contract_accepts_four_unique_joint_configs() -> None:
    payload = {
        "candidates": [
            _candidate(0.35, 1e-4, 0.25, False),
            _candidate(0.50, 2e-4, 0.50, False),
            _candidate(0.20, 1e-4, 0.50, True),
            _candidate(0.65, 3e-4, 0.35, False),
        ]
    }
    candidates = p32.validate_candidates(payload)
    assert len(candidates) == 4
    assert candidates[2].sam2_frozen is True


def test_candidate_contract_rejects_bounds_and_duplicates() -> None:
    with pytest.raises(ValueError, match="outside frozen bounds"):
        p32.validate_candidates(
            {"candidates": [_candidate(0.9, 1e-4, 0.25, False)] * 4}
        )
    duplicate = _candidate(0.35, 1e-4, 0.25, False)
    with pytest.raises(ValueError, match="duplicate"):
        p32.validate_candidates({"candidates": [duplicate] * 4})


def test_prompt_contains_only_safe_categorical_diagnostics() -> None:
    prompt = p32.build_prompt(
        {
            "F3": {"loss_level": "moderate", "fusion_scale_state": "stuck"},
            "Penobscot": {"loss_level": "high", "fusion_scale_state": "stuck"},
        }
    )
    text = json.dumps(prompt, sort_keys=True).lower()
    assert "0.229" not in text
    assert "sample_id" not in text
    assert "/mnt/" not in text
    assert "test data" in text


def test_gate_requires_mean_improvement_and_task_nondegradation() -> None:
    deterministic = {
        "promotion": {
            "equal_mean": 0.25,
            "tasks": {"F3": {"miou": 0.30}, "Penobscot": {"miou": 0.20}},
        }
    }
    agent = {
        "promotion": {
            "equal_mean": 0.26,
            "tasks": {"F3": {"miou": 0.305}, "Penobscot": {"miou": 0.215}},
        }
    }
    assert p32.promotion_gate(agent, deterministic)["decision"] == "RETAIN_HYBRID"
    agent["promotion"]["tasks"]["F3"]["miou"] = 0.29
    assert p32.promotion_gate(agent, deterministic)["decision"] == "KEEP_DETERMINISTIC"


def test_executable_config_ignores_prose_and_identifiers() -> None:
    first = {
        "action_id": "agent_1",
        "changed_factor": "joint",
        "description": "first wording",
        "fusion_scale_initial": 0.8,
        "fusion_lr": 5e-4,
        "dice_weight": 0.75,
        "sam2_frozen": False,
    }
    second = {**first, "action_id": "agent_4", "description": "other wording"}
    assert p32.executable_config(first) == p32.executable_config(second)
