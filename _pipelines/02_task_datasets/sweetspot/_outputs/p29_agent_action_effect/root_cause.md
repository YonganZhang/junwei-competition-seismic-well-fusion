# P29 root cause

The P28 prompt compared candidate selection MAE against the historical Stage3 aggregate.
That made the feedback labels `worse` regardless of the same-fold same-executor A0 baseline.

Fixed in P29:
- baseline is now `same_fold_same_executor_a0`
- prompt exposes only signed normalized deltas and remaining budget
- selection and promotion stay disjoint
- A2D and A3 are independent controls

Honest outcome:
- A2L status: `STOPPED`
- verdict: `REJECT_AGENT`
- retained LLM decision: `False`
