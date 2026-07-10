# _refs/ — 参考资料(混乱区)

## 用途

存你上传的**混乱原始资料**:论文 PDF / 网页截图 / PPT / Word / 标准 / 准则 / 任意格式。

**不强制分类**(用户偏好)。直接扔进去,有用了再让 cc 精读 + 归档到合适地方。

## 跟其他目录的区别

| 目录 | 内容 |
|---|---|
| `_refs/` | **混乱原始资料**(可能没用,可能要精读) |
| `_data/` | 清洗后数据,有 schema,给 pipeline 用 |
| `_tmp/` | 临时下载(自动 30 天提示归档) |

## 处理流程

1. 你上传文件 → 默认进 `_refs/`(或者 `_tmp/` 临时)
2. 让 cc 精读 → 提取信息 → 写到合适地方:
   - 方法论 → `_wiki-methodology/_wiki/_entities/`
   - 算法公式 → `_wiki-methodology/_wiki/_methods/`
   - 数据集 → 登记到 `_data/`
   - 论文引用 → 进 `_paper/bib/refs.bib`
3. 精读完的 `_refs/` 文件可保留或归档到 `_refs/_processed/`

## Git 范围

- ✅ Git:小 PDF / 网页存档(`<5MB`)
- ❌ 不进 Git:大 PDF / 视频 / 大数据集
- 通用原则:**`_refs/` 默认 `.gitignore`**(用户上传的乱东西不污染 Git),需要保留再 `git add -f` 单个文件

## 子目录(可选,不强制)

cc 精读多了可以按类归:
- `_refs/papers/`
- `_refs/standards/`
- `_refs/guidelines/`
- `_refs/_processed/`   ← 已经精读完归档了的

但**不强制**。一开始可以全扁平扔进 `_refs/`。
