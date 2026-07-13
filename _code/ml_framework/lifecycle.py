"""Experiment state machine and single-use frozen-test firewall."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class ExperimentState(str, Enum):
    DRAFT = "DRAFT"
    SPLIT_LOCKED = "SPLIT_LOCKED"
    SMOKE_PASSED = "SMOKE_PASSED"
    CV_COMPLETE = "CV_COMPLETE"
    CONFIG_FROZEN = "CONFIG_FROZEN"
    REFIT_COMPLETE = "REFIT_COMPLETE"
    TEST_CONSUMED = "TEST_CONSUMED"
    VERIFIED = "VERIFIED"


_NEXT = {
    ExperimentState.DRAFT: ExperimentState.SPLIT_LOCKED,
    ExperimentState.SPLIT_LOCKED: ExperimentState.SMOKE_PASSED,
    ExperimentState.SMOKE_PASSED: ExperimentState.CV_COMPLETE,
    ExperimentState.CV_COMPLETE: ExperimentState.CONFIG_FROZEN,
    ExperimentState.CONFIG_FROZEN: ExperimentState.REFIT_COMPLETE,
    ExperimentState.REFIT_COMPLETE: ExperimentState.TEST_CONSUMED,
    ExperimentState.TEST_CONSUMED: ExperimentState.VERIFIED,
}


@dataclass
class ExperimentLifecycle:
    experiment_id: str
    state: ExperimentState = ExperimentState.DRAFT
    evidence: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    test_consumed_at: str | None = None

    def advance(self, next_state: ExperimentState, evidence: Mapping[str, Any]) -> None:
        expected = _NEXT.get(self.state)
        if next_state != expected:
            raise RuntimeError(f"invalid lifecycle transition {self.state.value} -> {next_state.value}; expected {expected}")
        if not evidence:
            raise ValueError(f"{next_state.value} requires durable evidence")
        if next_state == ExperimentState.TEST_CONSUMED:
            required = {"config_hash", "checkpoint_hash", "split_hash"}
            missing = sorted(required - set(evidence))
            if missing:
                raise ValueError(f"test consumption evidence missing {missing}")
            if self.test_consumed_at is not None:
                raise RuntimeError("frozen test has already been consumed")
            self.test_consumed_at = datetime.now(timezone.utc).isoformat()
        self.state = next_state
        self.evidence[next_state.value] = dict(evidence)

    def require_development_access(self) -> None:
        if self.state in {ExperimentState.TEST_CONSUMED, ExperimentState.VERIFIED}:
            raise RuntimeError("this experiment already consumed frozen test; open a new experiment version")

    def require_test_access(self, *, config_hash: str, checkpoint_hash: str, split_hash: str) -> None:
        if self.state != ExperimentState.REFIT_COMPLETE:
            raise RuntimeError("frozen test is accessible only after CONFIG_FROZEN and REFIT_COMPLETE")
        frozen = self.evidence.get(ExperimentState.CONFIG_FROZEN.value, {})
        refit = self.evidence.get(ExperimentState.REFIT_COMPLETE.value, {})
        locked = self.evidence.get(ExperimentState.SPLIT_LOCKED.value, {})
        if frozen.get("config_hash") != config_hash:
            raise RuntimeError("config hash does not match frozen configuration")
        if refit.get("checkpoint_hash") != checkpoint_hash:
            raise RuntimeError("checkpoint hash does not match completed refit")
        if locked.get("split_hash") != split_hash:
            raise RuntimeError("split hash does not match locked split")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload
