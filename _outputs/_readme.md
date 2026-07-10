# _outputs/ — 实验结果(版本化)

## 用途

每次跑 `_pipelines/` 都在这里生成 `v<NN>/` 子目录。**108 次失败实验全留**(防 p-hacking)。

## 结构

```
_outputs/
├── _readme.md
├── _index.md            # 当前最佳 + 跑次摘要(由 _runs.yml 自动渲染)
├── v01/                 # 第 1 次跑
│   ├── _run_meta.yml    # 跑次元数据(合 _meta/_schema/_run_meta.yml)
│   ├── step_*.html      # 每 step 1 个 HTML(auto pubhtml)
│   ├── step_*.csv       # step 间中间态 CSV
│   └── _evidence/       # LLM evidence(议会 R1/R2 reasoning)
├── v02/ ...
├── v108/                # 第 108 次(假设这是 paper_final)
└── paper_final -> v108/ # 软链接到论文用版本
```

## 跑次清单

汇总在 `_meta/_runs.yml`。**任何时候追根溯源数字** → 先查 `_runs.yml` 找 run_id → 进 `_outputs/v<NN>/` 看细节。

## 论文写作

- 论文只引用 `paper_final/`(软链接)
- 失败实验**不写论文**,但 `_runs.yml` 留 trail(reviewer 问"复现性?" → 完整 108 次记录)

## Git 范围

| 进 Git | 不进 Git |
|---|---|
| `_run_meta.yml` 所有版本 | `step_*.csv` 大中间态(>10MB) |
| `paper_final` 软链接 | `_evidence/` 大 LLM logs |
| 小 HTML 报告 | 大 HTML(>1MB) |

**大文件原则**:跟 `_data/` 一样,只 commit 指纹 + 下载脚本(或 pubhtml URL)。

## 失败 vs 成功

`_run_meta.yml` 必填 `outcome: success / failed / partial` + `outcome_note`。

- **不删失败实验**:论文 reviewer 问"为什么不试 X?"你能拿出 trail "v53 试过 X,失败原因是 Y"
