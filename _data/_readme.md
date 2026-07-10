# _data/ — 数据

## 用途

项目所有**清洗后**数据(给 pipeline 用)。

## 跟其他目录的区别

| 目录 | 内容 |
|---|---|
| **`_data/`** | 清洗后数据,有 schema,版本化,给 pipeline 消费 |
| `_refs/` | 原始参考资料(PDF / 网页 / 任意杂物,不一定进 pipeline) |
| `_tmp/` | 临时下载,30 天没归档自动提示 |

## Git 范围

| 进 Git | 不进 Git |
|---|---|
| `_data/<name>/_meta.yml`(SHA256 + 下载脚本) | `_data/<name>/raw/*.csv` 大文件 |
| `_data/_data_sha256.yml`(指纹清单) | 实际数据 (gitignore) |

**大文件原则**:只 commit SHA256 + 下载脚本,不 commit 实际数据。

## 数据登记

每个数据集进 `_data/<name>/_meta.yml`:
```yaml
name: UCDB_R2024A
source: European Commission JRC
sha256: abc123...
size_mb: 900
download_script: scripts/download_ucdb.sh
license: CC BY 4.0
used_in_pipelines: [01_baseline_gciri]
```

同步登记到 `_meta/_data_registry.yml`(COL5 数据资产注册表)。

## 命名规则

`_data/<source>_<scope>_<version>/`
- `UCDB_R2024A/`
- `GIRI_official_v2_2023/`
- `Tavily_search_2026-05-10/`
