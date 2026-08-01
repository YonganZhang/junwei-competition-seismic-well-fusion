# P28 agentic optimization

## A0 freeze
- baseline model: `tiny_mlp`
- baseline metrics path: `_outputs/metrics.json`
- baseline run_manifest hash: `9b34d916f75860294673a4061401a7fcfce462b0c8b3d85aeb5c43bf8fafeef9`
- A1 prediction hash (selection-dev): `8294ee83dea95acc30f03099ab1bf52a6b920982badc89352f36d08b249d08d9`

## Nested split
- train: families=['15/9-F-1', '15/9-F-11'] / wells=['15/9-F-1', '15/9-F-1 A', '15/9-F-1 B', '15/9-F-1 C', '15/9-F-11 A', '15/9-F-11 B', '15/9-F-11 T2'] / n=988
- selection_dev: families=['15/9-19'] / wells=['15/9-19 A'] / n=59
- promotion_dev: families=['15/9-19'] / wells=['15/9-19 BT2', '15/9-19 SR'] / n=88

## CIG route gate
- status: `blocked`
- reason: ModelScope default checkpoint 404 and input contract mismatch
- evidence: `_outputs/p18_cigbench_property/evidence.md`

## Pilot feedback
- tiny_mlp_default: feedback=worse selection_MAE_macro=152.49056758227624
- tiny_mlp_l2: feedback=worse selection_MAE_macro=152.68780490008007
- reservoir_ridge: feedback=worse selection_MAE_macro=195.4851990412726
- reservoir_linear: feedback=worse selection_MAE_macro=195.48518355864812

## Strategy outcomes
- A2L: chosen_route=None selected_by=blocked promotion_gate=False
  - selection_median_MAE_macro=blocked
  - promotion_median_MAE_macro=blocked
- A2D: chosen_route=tiny_mlp_default selected_by=deterministic_pilot_rank promotion_gate=False
  - selection_median_MAE_macro=171.233811
  - promotion_median_MAE_macro=224.939900
- A3: chosen_route=tiny_mlp_default selected_by=pcg64_no_replacement promotion_gate=False
  - selection_median_MAE_macro=171.233811
  - promotion_median_MAE_macro=224.939900

## Promotion gate
- passed: `False`
- primary threshold: seed-median physical_MAE_macro improvement >= 1% vs A0 on promotion-dev
- worst-group threshold: per-target worst_group_RMSE worsening < 2% vs A0 on promotion-dev

## Commands
- `python3 -m py_compile _pipelines/02_task_datasets/reservoir/p28_agentic_optimization.py _pipelines/02_task_datasets/reservoir/tests/test_p28_agentic_optimization.py`
- `pytest -q _pipelines/02_task_datasets/reservoir/tests/test_p28_agentic_optimization.py`
- `python3 _pipelines/02_task_datasets/reservoir/p28_agentic_optimization.py --budget-steps 32 --pilot-steps 8`
