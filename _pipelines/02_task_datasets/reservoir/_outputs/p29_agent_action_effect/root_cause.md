# P29 root cause

| stage | connected | evidence |
|---|---|---|
| prompt | yes | route semantics + safe normalized deltas only |
| action | yes | A2L/A2D/A3 route selected from pilot evidence; A1 is identity no-op |
| executor | yes | shared NumPy/MLP training loop on train-only statistics |
| prediction | yes | every trial records prediction hash |
| metric | yes | documented composite primary metric used for selection/promotion |
| promotion | yes | candidate-only gate excludes A0/A1 |
| endpoint | yes | outputs written to `_outputs/p29_agent_action_effect/` |

- A0 selection hash: `43110d0d819b8f067a4a98771ea6aeb53492f0200fa6de8b10054ad71281f2ea`
- A1 replay matches A0: `True`
- oracle ceiling route: `reservoir_linear`
- promotion gate passed: `False`
