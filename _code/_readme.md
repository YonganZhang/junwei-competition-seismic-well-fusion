# _code/ - project code / optional helpers

## 用途

项目可复用代码、脚本或项目级 helper。若项目是现成软件 wrapper,活跃代码也可以保留在原生子仓中,由 `_codemap.md` 和 `_meta/_registry.yml` 登记边界。

## 子目录结构

```
_code/
├── _readme.md                # 本文件
│
├── sync_wiki.py              # legacy placeholder;默认 fail-loud,项目 opt-in 后才替换
├── lint.py                   # legacy placeholder;默认 fail-loud,用 top-lint 做结构校验
├── register.py               # legacy placeholder;默认 fail-loud,不再默认函数扫描
├── init_pipeline.py          # 加新 pipeline 骨架
├── init_domain.py            # 加新 domain 子目录骨架
│
└── <domain>/                 # cc 按需建子目录:
    ├── giri/                 # GIRI 算法(梯形 / 21 点 / R_Ph 下推)
    ├── llm/                  # LLM 议会(client / tavily / prompt builder)
    ├── viz/                  # 可视化工具
    └── ...
```

## 命名规则(强制)

- 函数:**snake_case + 6 个动词起头**(`compute_*` / `load_*` / `save_*` / `validate_*` / `build_*` / `run_*`)
- 文件:`<noun>.py` 或 `<noun>_<noun>.py`
- 类:**PascalCase + 单一名词**
- 详见 `_meta/_naming.md`

## 写新代码

- 主流程入口、复用工具、对外 API 或子仓代码域 → 手动登记到 `_meta/_registry.yml` 和 `_codemap.md`。
- 临时探查 → 放 `_tmp/` 或 `_sandbox/`,不进 registry。
- 函数级账本不是默认架构；确实需要时先写项目级说明、schema 和验收命令,再替换本目录的 legacy placeholder。

## 跟 _meta 的联动

- 主流程入口 / 复用工具 / 对外 API / 子仓代码域 → `_meta/_registry.yml`
- 数据、模型、样例、外部资产 → `_meta/_data_registry.yml`
- 改 `_code/` 核心 helper → 追加 `_meta/_changelog.md`
