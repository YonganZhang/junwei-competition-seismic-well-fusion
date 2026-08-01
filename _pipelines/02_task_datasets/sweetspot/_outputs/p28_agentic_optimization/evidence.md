# P28 T3-only hybrid execution-agent pilot

Verdict: `BLOCKED_EXECUTOR`

## What was actually done

- Archived A0 reference for T3 XGBoost was frozen from immutable stage-3 / stage-4 evidence.
- A2L DeepSeek selection was archived under a strict no-raw-metric, no-label, no-path prompt contract when available.
- A2D deterministic route choice and A3 random-policy choice were computed from the same allowlist.
- The private P28 worktree does not contain a portable executor entrypoint, so no new online evidence was replayed.

## Why it is blocked

BLOCKED_EXECUTOR: no portable P28 executor exists in the current sweetspot worktree; immutable sibling results are only references and cannot be replayed as new online evidence.

## Honest retain / hybrid / reject verdict

- Retain: A0 archived baseline reference only.
- Hybrid: A2L decision record is usable as a policy artifact, but not as executed science.
- Reject: A2D/A3/A4 execution until a legal portable executor is ported into P28 private code.

## T5–T7 gate states

- T5: `not_feasible`
- T6: `blocked`
- T7: `blocked`
