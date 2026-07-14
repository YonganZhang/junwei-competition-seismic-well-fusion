# P5 direct benchmark report integration evidence

Date: 2026-07-14

## Scope and precondition

- Worktree branch: `p5-model-benchmark-integration`
- Required base: `c22c9ebd788cdf759d1e7b93c5ac445f49521651`
- Pre-integration `HEAD`: `c22c9ebd788cdf759d1e7b93c5ac445f49521651`
- Pre-integration status: clean; `git status --short --untracked-files=all` produced no entries.
- Allowed payload: the six track-private reports listed below, followed by this evidence file only.
- No conflict occurred. No report text, source code, data, split, result, output, model, label, or shared-framework file was manually changed during the cherry-picks.

## Ordered cherry-pick mapping

The order below is the executed order. Stable patch IDs match between each source and integrated commit, so no conflict-resolution edit altered a source patch.

| Order | Track | Source commit | Integrated commit | Stable patch ID |
|---:|---|---|---|---|
| 1 | fault | `24c56638643bcd2dc791434ed09616c75a796dfe` | `cf1bcbcb7fe191fedbf7da30c9481e09075461ea` | `86a24c617d5b0f042c05242c13a179552e83242c` |
| 2 | facies | `95fa5fbd8ac547286b8863c7eb5095bb6c308c0b` | `b0d4e37c826ac075843e5a612602b13cefe1cf16` | `0a7e9c514bd8af31327b9d5d7503c4fa07342934` |
| 3 | property initial report | `14c7dcbf8157bf18ab5cebc9826de936e05c690d` | `f4b670ff31ae7328d5f3d50bc2828285ec31443d` | `feecb87fe3d746d1ba85a6ec8620e1ba0e7c8715` |
| 4 | property source correction | `66f3037f8df050bf869d0338662fa06e61a6eb75` | `831403e3e7d699ce2e5451c490b4dfef488d58da` | `e63efcf3adb38ec7daf5488368beb546612a7976` |
| 5 | lithofacies | `d8dc570327c15e5881323911a691a20b13c56a4e` | `fe35f572b813aaaa794cbd5aada50f761acfc57a` | `b974c847e379b70d3a6d0e14d7e021a290448f00` |
| 6 | sweetspot | `50600042fddab8816258c03ea8a3f8b48ff8486a` | `97e5d65c81bdd735b15bf8eb4e91b0ca9115d3b8` | `2f0c59601dcf95797d0375e7598501875c5efde1` |
| 7 | reconstruction | `46d6c294e38ea41a1d4e547eb2b6012457603c6e` | `83c0b684259a53f09356b3d7dea1dc12e4a33704` | `5e3d925d9f81269fb83db85099d418f4c0a90f6d` |

Pre-evidence integration `HEAD`: `83c0b684259a53f09356b3d7dea1dc12e4a33704`.

## Integrated report inventory

Line counts use `wc -l`; hashes use `sha256sum` on the integrated working-tree bytes after all seven cherry-picks.

| Track | Path | Lines | SHA-256 |
|---|---|---:|---|
| fault | `_pipelines/02_task_datasets/fault/_reports/P5_direct_benchmark_research_20260714.md` | 397 | `0fc5efc721e4f19e54d4f659646c449ebc36c8bd2aee4f60065048e7505bc6ae` |
| facies | `_pipelines/02_task_datasets/facies/_reports/P5_direct_benchmark_research_20260714.md` | 241 | `e8292d5e7bafd03649443ab2f0daf28e31fd76b1de8ebe7f28f36aa3aa03de57` |
| property | `_pipelines/02_task_datasets/reservoir/_reports/P5_direct_benchmark_research_20260714.md` | 296 | `400b073ce9d8cd6fafce74195b959311946f3054cf2d87f637e0f370bd80b424` |
| lithofacies | `_pipelines/02_task_datasets/lithofacies/_reports/P5_direct_benchmark_research_20260714.md` | 337 | `d3e54da9050a2a8acf66c4d4b8158e911a1a052543949de1dec6948d95c93346` |
| sweetspot | `_pipelines/02_task_datasets/sweetspot/_reports/P5_direct_benchmark_research_20260714.md` | 343 | `aad66ac8710aa449055808dcb8990c6adfbe4f058aa220fdded807bfe45d7f1d` |
| reconstruction | `_pipelines/02_task_datasets/reconstruction/_reports/P5_direct_benchmark_research_20260714.md` | 325 | `a72454958c39bcbec59e48ac39b1cab9f6db98eced533dbda96ed699bfe76893` |

## Base-to-integration range evidence

Before adding this evidence file:

```text
$ git rev-list --count c22c9ebd788cdf759d1e7b93c5ac445f49521651..HEAD
7

$ git diff --name-status c22c9ebd788cdf759d1e7b93c5ac445f49521651..HEAD
A  _pipelines/02_task_datasets/facies/_reports/P5_direct_benchmark_research_20260714.md
A  _pipelines/02_task_datasets/fault/_reports/P5_direct_benchmark_research_20260714.md
A  _pipelines/02_task_datasets/lithofacies/_reports/P5_direct_benchmark_research_20260714.md
A  _pipelines/02_task_datasets/reconstruction/_reports/P5_direct_benchmark_research_20260714.md
A  _pipelines/02_task_datasets/reservoir/_reports/P5_direct_benchmark_research_20260714.md
A  _pipelines/02_task_datasets/sweetspot/_reports/P5_direct_benchmark_research_20260714.md
```

The final `c22c9eb..HEAD` acceptance set is exactly those six reports plus `_wiki-methodology/_tests/P5_direct_benchmark_report_integration_evidence.md`. The final evidence-commit SHA is intentionally verified after commit rather than embedded here, because embedding a commit's own SHA would be self-referential.

## Whitespace and TOP checks

```text
$ git diff --check c22c9ebd788cdf759d1e7b93c5ac445f49521651..HEAD
[no output]
exit=0

$ python3 ~/.claude/skills/share-top/scripts/topic-doctor.py . --links
[topic-doctor] project=<current p5-model-benchmark-integration worktree>
INFO PLAN_FOUND _wiki-methodology/_top/_task_plan.md kind=v4
INFO REGISTRY_SCHEMA _meta/_registry.yml entrypoints+code_domains+cross_domain_edges total=20
INFO DATA_REGISTRY _meta/_data_registry.yml entries=3
INFO WIKI_ENTRY _wiki-methodology/_wiki wiki entry exists
INFO TESTS_BRIEF _wiki-methodology/_tests tests brief exists
INFO SSDO_ENTRY _wiki-methodology/_tests audit-first/SSDO entry exists
INFO REF_SCHEMA . declared_refs=32
SUMMARY: block=0 warn=0 info=7
exit=0
```

The doctor project path is normalized above to avoid persisting a machine/worktree-specific absolute path. No TOP source file was modified by the doctor.

## Negative-scope proof

- Training code unchanged.
- Model definitions, labels, loss functions, feature contracts, and shared framework unchanged.
- Data files and data registry unchanged.
- Split manifests and split code unchanged.
- Metrics, results, predictions, figures, checkpoints, and output directories unchanged.
- No download, build, training, refit, frozen-test consumption, or visualization execution occurred.
- `master` was not merged or checked out; this work remained on `p5-model-benchmark-integration`.
- No push occurred.

These statements are enforced by the final base-range file allowlist: any path other than the six reports and this evidence file fails acceptance.
