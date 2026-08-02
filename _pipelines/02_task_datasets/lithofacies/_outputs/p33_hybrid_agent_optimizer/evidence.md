# Lithofacies P33 hybrid-agent evidence

Decision: **KEEP_CURRENT_DEFAULT**.

- Agent selected: `{'max_depth': 3, 'eta': 0.2, 'rounds': 60, 'subsample': 1.0, 'colsample_bytree': 1.0}`.
- Deterministic selected: `{'max_depth': 5, 'eta': 0.05, 'rounds': 60, 'subsample': 0.75, 'colsample_bytree': 0.8}`.
- Promotion Macro-F1 delta: `+0.027372974127`.
- Promotion delta versus current A0: `-0.013749663974`.
- Paired seed wins versus deterministic/A0: `3/3` and `0/3`.
- Matched selection budget: `2160` boosting rounds per strategy.
- Selection uses LOGO folds 0--2; promotion uses fold 3 only.
- Frozen test and known holdout were not read.
- Attribution is hybrid: LLM candidate proposal plus deterministic scheduling and promotion gate.
