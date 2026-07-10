# Changelog

> 改 `_code/` 核心 / `_meta/_naming.md` / `_meta/_schema/` / 顶级架构时追加 1-3 行。可以由 helper 写,也可以人工写; 关键是事实可追,不要假装存在自动化。

## 格式

```
## YYYY-MM-DD HH:MM

- [scope] 改了什么(1 句话)
- [scope] ...
```

scope:
- `core` — 改 `_code/` 核心工具(register.py / sync_wiki.py / lint.py,如项目启用)
- `naming` — 改 `_meta/_naming.md` 命名宪法
- `schema` — 改 `_meta/_schema/` schema 定义
- `arch` — 改顶级架构(加/删 `_<dir>/`)

---

## 2026-05-17 01:00

- [arch] v4 架构落地: 建 11 个 `_<dir>/` 顶级目录
- [arch] 写 `_codemap.md` 项目根 COL4 代码地图 (current 只链接 `_wiki-methodology/_top/_task_plan.md`)
- [naming] 写 `_meta/_naming.md` 命名宪法 v1
- [schema] 写 `_meta/_schema/` 的 frontmatter / pipeline.yml / run_meta.yml schema
- [core] 写 `_code/sync_wiki.py` (registry 指针投影 helper,按项目启用)
- [core] 写 `_code/lint.py` (CI 校验)
- [core] 写 `_code/register.py` (AST 扫描 / registry helper)
