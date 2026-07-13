# Test Run Ledger - 军伟的比赛

> Historical run evidence. Do not rewrite old command contexts to look cleaner; append new runs with cwd and exact command shape.

## 2026-07-13 P4 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p4-training-integration`
- branch: `p4-training-integration`
- root seed: `2693`
- portable suites: 按 `_gates.yml` 的 `p4-shared` 至 `p4-reconstruction` 逐一执行；可选真实产物缺失时为可审计 skip，非静默通过。
- explicit real smoke: fault coordinate/index preflight、F3/Penobscot 各 1 CPU epoch、property P4 integration、lithofacies HDF5、reconstruction HDF5 split/preprocess 均 exit 0。
- facies campaigns: `p4_workflow.py` 在 `_tmp/p4-acceptance/facies_f3` 与 `facies_penobscot` 完成 `prepare -> smoke -> cv -> freeze -> refit -> test -> visualize`。
- reconstruction campaigns: `p4_reconstruction.py` 在 `_tmp/p4-acceptance/reconstruction_strict_v2` 与 `reconstruction_conditional_v2` 完成 `prepare -> cv -> refit -> test`；`p4_visualize.py` 从归档 predictions/metrics 生成 diagnostics。
- lithofacies campaign: `p4_runner.py freeze/refit/test --run-root _pipelines/02_task_datasets/lithofacies/_outputs/p4_runs/integration_cv_smoke --dataset-root <track-lithofacies real HDF5>`；`visualize_p4.py` 读归档预测生成分类/校准图。
- artifact audit: 7 manifests / 144 manifest artifacts verified；seven-target registry 7 cases / 50 registry artifacts verified；`all_targets_independent=true`。
- 结果与 blocker: `P4_acceptance_evidence.md`。

## 2026-07-14 P5 Stage-1 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-model-benchmark-integration`
- branch: `p5-model-benchmark-integration`
- root seed: `2693`
- source branches: six clean P5 track commits, integrated in fault/facies/property/lithofacies/sweetspot/reconstruction order.
- `torch-common` full combined gate: 53 passed, 6 skipped, 77 subtests passed.
- `tabular-cpu` combined gate: 31 passed, 2 skipped, 20 subtests passed.
- live development inputs: used where task labels/contracts were approved; fault formal comparison and all sweetspot real task cells stopped fail-closed at their scientific gates.
- frozen test: not accessed by Stage-1 runners.
- results, structured skips and SSDO downgrade: `P5_stage1_acceptance_evidence.md`.
