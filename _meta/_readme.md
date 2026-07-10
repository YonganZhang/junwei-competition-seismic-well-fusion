# _meta/ — 元数据 + 系统账本

## 用途

项目所有元数据 + 系统账本的集中地。**只有 cc / 自动化工具会读写**,你日常不直接编辑。

## 文件清单

| 文件 | 干啥 | 谁维护 |
|---|---|---|
| `_readme.md` | 本文件 | 手动 |
| `_registry.yml` | ⭐ 代码/工具资产注册表(含 method/tests/phases 指针) | AI + 可选 helper |
| `_data_registry.yml` | 数据集、外部文件、重要资产登记 | AI + 人工复核 |
| `_runs.yml` | 跑次清单(每跑一次 +1 行,带 why + outcome) | AI 跑实验时自动追加 |
| `_naming.md` | 命名宪法(强制) | 手动(改时走 RFC) |
| `_changelog.md` | 重要架构/核心工具改动记录 | AI 或人工 |
| `_schema/` | frontmatter schema 定义 | 手动(改时走 RFC) |

## 关键设计

### 单一真源 (SSOT)
`_registry.yml` 是**代码/工具资产**的结构化注册表。数据资产走 `_data_registry.yml`; 当前状态走 `_wiki-methodology/_top/_task_plan.md`; 长知识正文走 `_wiki-methodology/_wiki/`。不同列只互相链接,不复制正文。

跨列引用使用 owner 原生 key,如 `code:<id>` / `data:<id>` / `test:<id>`;反向引用由 doctor 扫描派生,不手写边表。

### 指针联动
```
_registry.yml
    ├── method -> _wiki-methodology/_wiki/...
    ├── tests  -> _wiki-methodology/_tests/... 或项目测试目录
    └── phases -> _wiki-methodology/_top/...
```

`_code/sync_wiki.py` 若存在,只能在项目明确声明的生成区内投影这些指针; 未启用 hook 和测试前,不要把它写成默认自动化。

### 改这层文件的规则

| 文件 | 改的方式 |
|---|---|
| `_registry.yml` | 写主流程/复用工具时更新; register.py 只作为可选 helper |
| `_runs.yml` | AI 跑实验时自动追加 |
| `_naming.md` | 改要写 RFC,记 `_changelog.md` |
| `_schema/*` | 只允许加新 schema,不许改老的 |
| `_changelog.md` | 重要变更追加 1-3 行; 自动或手写都必须真实可追 |

## 跟六合一的关系

`_meta/` = 项目工程层账本(代码 + 实验);
`_wiki-methodology/` = 项目方法论层(算法 + 公式 + 知识 + 进展 + 测试覆盖);

**联动靠 `_registry.yml` 的 `method` / `tests` / `phases` 指针**; `_code/sync_wiki.py` 只是可选投影 helper。
