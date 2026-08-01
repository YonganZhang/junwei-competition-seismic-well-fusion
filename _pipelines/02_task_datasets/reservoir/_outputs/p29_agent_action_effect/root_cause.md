# P29 root cause

| stage | connected | evidence |
|---|---|---|
| prompt | yes | route semantics + safe normalized deltas only |
| action | yes | A2L/A2D/A3 route selected from pilot evidence; A1 is identity no-op |
| executor | yes | shared NumPy/MLP training loop on train-only statistics |
| prediction | yes | every trial records prediction hash |
| metric | yes | documented composite primary metric used for selection/promotion |
| promotion | yes | matched-budget A0 causal comparator; historical checkpoint is non-causal reference only |
| gate | yes | per-strategy gates; A2L retain verdict is independent of A3 failure |
| endpoint | yes | outputs written to `_outputs/p29_agent_action_effect/` |

- A0 selection hash: `b0e108749d880bd55a2ddf0649ad3f7d2d4a95c5e8da9a119474db772a417fb6`
- A1 replay matches A0: `True`
- oracle ceiling route: `reservoir_linear`
- promotion gate passed: `False`
