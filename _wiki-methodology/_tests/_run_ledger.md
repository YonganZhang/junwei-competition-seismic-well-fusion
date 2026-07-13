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

## 2026-07-14 P5 Stage-2 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-model-benchmark-integration`
- branch: `p5-model-benchmark-integration`
- integrated HEAD: `d46a7b5`
- root seed: `2693`
- source commits: fault `53db563`, facies `3c6c3b8`, property `cab82e5`, lithofacies `5302954`, sweetspot `ad5fde8`, reconstruction `15c4ae1`; cherry-picked in fixed track order.
- Stage-2 matrix: 140 preregistered cells; 53 development pilots, 87 structured skip/blocked, 0 failed, 0 timeout.
- `torch-common` Stage-2 combined gate: 59 passed, 22 subtests passed.
- `tabular-cpu` applicable four-track gate: 40 passed. Reconstruction is intentionally excluded from this environment because it lacks `h5py`; its 12 Stage-2 tests passed in `torch-common`.
- Stage-1 regression after integration: 53 passed, 6 skipped, 77 subtests passed.
- frozen test: not accessed by any Stage-2 runner; all leaderboards are fixed-fold development evidence only.
- results and scientific stop lines: `P5_stage2_acceptance_evidence.md`.

## 2026-07-14 P5 Stage-3 integration acceptance

- cwd: `/mnt/data/yongan-admin-2/projects/师弟-军伟的比赛-2693e5/.claude/worktrees/p5-model-benchmark-integration`
- branch: `p5-model-benchmark-integration`; integrated result HEAD before documentation: `9e5f501`.
- protocol commit: `16bebd1`; root seed `2693`; repeat seeds `1867973658`, `2137841944`, `3902865753`.
- development matrix: five runnable tracks, 441 attempted cells; 437 pass, 3 fail, 1 timeout. Fault had zero scientifically legal training cells and remained `not_rankable`.
- independent full regressions: fault `41 passed` plus `15 passed/7 skipped/2 subtests`; facies `47 passed/1 skipped/43 subtests`; property `37 passed/5 skipped`; lithofacies P5 `25 passed/1 skipped/26 subtests`, data contract `8 passed/4 skipped`, mixed legacy `5 passed/6 skipped/3 subtests`; sweetspot data `21 passed`, models `51 passed/1 skipped`; reconstruction `55 passed/10 skipped/53 subtests`.
- dependency policy: `torch-common` and `tabular-cpu` were not modified. The legacy lithofacies module that imports Torch plus HDF5/LAS/SEG-Y readers was rerun in a one-shot `uv run --isolated` group.
- frozen test: no Stage-3 runner opened frozen arrays, labels, predictions or historical metrics.
- results, winners, figures and boundaries: `P5_stage3_acceptance_evidence.md`.
