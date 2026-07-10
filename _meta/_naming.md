# 命名宪法

> v4 架构强制命名规范。AI 写代码 / 建文件 / 改架构必须遵守。违反 = `_code/lint.py` block。

## 1. 顶级目录 / 文件

### 1.1 全 `_` 开头规则

**架构内的所有目录 / 系统文件,文件名第一个字符必须是 `_`**。

| 规则 | 含义 |
|---|---|
| `_xxx/` | 架构内目录,有规则约束 |
| `_xxx.md/.yml/.py` | 架构内文件 |
| `yyy` (无 `_`) | **临时 / 外来 / 还没归档** → 要么清理,要么归档到正式 `_<dir>/` |

**生态例外**:`AGENTS.md` / `CLAUDE.md` / `README.md` / `.git` / `.codex` / `.vscode` 等工具固定入口，不据此推翻 `_` 规则。

### 1.2 自检规则

- cc 每次任务后扫项目根:任何无 `_` 开头的文件 / 目录 = "可能 cc 违规创造"
- 处置:**临时文件 → 清理**,**外来文件 → 归档到合适 `_<dir>/`**
- 例外:`AGENTS.md` / `CLAUDE.md` / `README.md` / `.git/` / `.gitignore` / `.codex/` 等系统文件 / 用户上传未归档暂时容忍

## 2. 函数命名(`_code/` + `_pipelines/`)

### 2.1 格式
- **snake_case**
- **动词起头**
- 只允许 6 个主动词:`compute_*` / `load_*` / `save_*` / `validate_*` / `build_*` / `run_*`

### 2.2 例子

| ✅ 正确 | 🔴 错误 |
|---|---|
| `compute_f_factor` | `f_factor` (无动词) |
| `load_indicator_meta` | `do_meta` (动词模糊) |
| `validate_gciri` | `process()` (动词不在白名单) |
| `build_prompt` | `helper1` (无意义) |
| `run_r1_parliament` | `parliament_r1` (无动词起头) |

### 2.3 单一职责
函数名 = 它做什么的完整描述。如果函数名说不清,**说明函数职责太大,要拆分**。

## 3. 文件命名

### 3.1 step 文件(`_pipelines/01_<name>/step_*.py`)

格式:`step_<NN>_<verb>_<noun>.py`
- NN = 2 位数字(01, 02, ... 99),按 pipeline DAG 顺序
- verb = 6 个主动词之一
- noun = 名词(单个或 snake_case 多个)

例:
- `step_01_load_metadata.py`
- `step_03_run_r1_parliament.py`
- `step_06_compute_f_factor.py`
- `step_09_compute_gciri.py`

### 3.2 工具文件(`_code/`)

格式:`<noun>.py` 或 `<noun>_<noun>.py`(纯名词)

例:
- `llm_client.py`
- `trapezoid_giri.py`
- `tavily_cache.py`
- `r_ph_downscaler.py`

### 3.3 测试文件(`_tests/`)

格式:`test_<被测对象>.py`

例:
- `test_trapezoid_giri.py`(测 `_code/trapezoid_giri.py`)
- `test_step_06_compute_f_factor.py`

## 4. 类命名

### 4.1 格式
- **PascalCase**
- **单一名词**(不含动词)

### 4.2 例子

| ✅ 正确 | 🔴 错误 |
|---|---|
| `ParliamentRunner` | `RunParliament` (含动词) |
| `LLMClient` | `LLMManager` (名词太泛) |
| `GCIRIComputer` | `GCIRI` (太短没动词性) |
| `TavilyCache` | `Helper` (无意义) |

## 5. 外部对象命名(模型 / 数据集 / API)

### 5.1 LLM 评委(8 个议会评委)

格式:`<vendor>/<model_family>@<api_version>`

例:
- `openai/gpt-4o-mini@2026-04`
- `anthropic/claude-sonnet@2026-05`
- `google/gemini-flash-lite@2026-05`

议会内部代号:`m1, m2, ..., m8`(顺序固定,登记 `_meta/_parliament_models.yml`)。

### 5.2 数据集

格式:`<source>_<scope>_<version>`

例:
- `UCDB_R2024A`(GHS-UCDB R2024A)
- `GIRI_official_v2_2023`(Cardona 2023 官方 GIRI)
- `Tavily_search_2026-05-10`(Tavily 搜索 cache)

### 5.3 数据指纹

所有数据集进 `_meta/_data_sha256.yml`,记录:
- name / sha256 / download_script / size

**大文件不进 Git**,只 commit SHA256 + 下载脚本。

## 6. 变量命名

### 6.1 城市

格式:`city_<ISO3>_<Name>`

例:
- `city_CHN_HongKong`
- `city_USA_NewYork`
- `city_JPN_Tokyo`

### 6.2 指标

直接用 `alias_name`(从 `_registry.yml` / Excel Sheet 3):
- `Housing_habitability`
- `Urban_safety_and_social_cohesion`
- `Access_to_quality_education`

### 6.3 分数

统一 0-1 resilience form(1=最好,0=最差)。变量名加 `_resilience` 后缀:

```python
hk_education_resilience = 0.78  # ✅
hk_education_deficit = 0.22     # ✅ 显式标 deficit
hk_education = 0.78             # 🔴 不清楚方向
```

## 7. 版本号(关键!)

| 版本 | 哪里管 | 规则 |
|---|---|---|
| **代码版本** | Git(commit / tag) | **不在文件名加 `_v1/_v2/_final/_backup`** |
| **实验版本** | `_outputs/v<NN>/` | 数字,跑一次 +1 |
| **数据版本** | `_meta/_data_sha256.yml` | SHA256 |
| **模型版本** | `_meta/_parliament_models.yml` 的 `api_version` 字段 | vendor 标签 |

### 🔴 禁止

```
agent_v27_3_19_final_backup.py     ← 文件名版本号
giri_official_v2_2023_FINAL.py     ← 文件名版本号
pipeline_old.py                    ← 文件名标 old
step_06_v2.py                      ← step 加版本号
```

**正确做法**:改了就改了,Git 管历史。需要老版本 → `git checkout <commit>`。

## 8. 强制规则

### 8.1 写新函数
- 判断是否为主流程入口、复用工具、对外 API、MCP/tool/plugin 暴露点; 是则登记 `_meta/_registry.yml`
- 可用 `_code/register.py` 辅助扫描,但不得假设它自动运行
- 填三个指针字段:`method`(链方法论)/ `tests`(链测试)/ `phases`(链 plan/finding/decision)

### 8.2 临时/测试代码
- 加 `@transient` 装饰器
- `_code/lint.py` 跳过这类,不强制注册

### 8.3 新模型
- 必须先登记 `_meta/_parliament_models.yml`
- 字段:vendor / family / api_version / role(m1-m8 or consul)

### 8.4 新数据
- 必须先登记 `_meta/_data_registry.yml`
- 字段:name / sha256 或 provenance / download_script 或 source / size / license / used_in_pipeline

### 8.5 改 `_code/` 核心
- 追加 `_meta/_changelog.md` 一行说明
- **不阻塞,异步留 trail**
- 但破坏性改动(rename / 删函数)要走 RFC

### 8.6 改命名宪法(本文件)
- **要走 RFC**(写 `_meta/_schema/RFC-<NN>.md`)
- 用户审完才能改
- 改完 cc 自己以后遵守新规则
