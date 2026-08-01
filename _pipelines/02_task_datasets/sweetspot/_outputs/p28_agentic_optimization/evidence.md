# P28 T3-only hybrid execution-agent pilot

Verdict: `reject`

## What was actually done

- Archived A0 reference for T3 XGBoost was frozen from immutable stage-3 / stage-4 evidence.
- A2L DeepSeek selection was evaluated under a strict no-raw-metric, no-label, no-path prompt contract.
- A2D deterministic choice used fold-train aggregates only.
- A3 random policy used the same route/action registry and a randomly chosen trial budget.
- Execution ran on development-only folds only; no test or holdout access was used.

## Why it is rejected or blocked

not blocked

## Honest retain / hybrid / reject verdict

- Retain: A0 archived baseline reference only.
- Hybrid: A2L decision record is usable as a policy artifact, but not as executed science.
- Reject: A2L stop, or A2D/A3/A4 promotion failure against the archived baseline.

## T5–T7 gate states

- T5: `not_feasible`
- T6: `blocked`
- T7: `blocked`
