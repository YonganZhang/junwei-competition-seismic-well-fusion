# _pipelines/ — 主流程(重灾区,命名宪法约束)

## 用途

项目所有"跑得通的"主流程。每个子目录 = 一个 pipeline = 一章。

## 子目录结构

```
_pipelines/
└── 01_<name>/
    ├── _readme.md         # 章节目录(每 step 一句话)
    ├── pipeline.yml       # ⭐ DAG 声明(合 _meta/_schema/_pipeline_yml.yml)
    ├── step_NN_verb_noun.py   # 每步 1 文件
    └── outputs/v01..v108/      # 实验输出版本化
        ├── _run_meta.yml  # 跑次元数据(合 _meta/_schema/_run_meta.yml)
        ├── step_*.html    # 每 step 1 个 HTML 报告(auto pubhtml)
        └── step_*.csv     # step 间中间态
```

## 命名规则(强制)

- pipeline 目录:`01_<verb_noun>/`(2 位编号 + verb_noun)
- step 文件:`step_<NN>_<verb>_<noun>.py`
- 6 个主动词:`compute_*` / `load_*` / `save_*` / `validate_*` / `build_*` / `run_*`
- 详见 `_meta/_naming.md`

## 加新 pipeline

```bash
python _code/init_pipeline.py --name 02_<your_name>
# 自动建好骨架 + pipeline.yml + _readme.md
```

## 跟六合一 / _meta 的联动

- 每条主 pipeline 或可复用入口注册到 `_meta/_registry.yml`;不默认登记每个函数。
- pipeline 的方法论和设计理由写 `_wiki-methodology/_wiki/`,从 `_codemap.md` 或 registry 链接即可。
- pipeline 的可复用测试入口写 `_wiki-methodology/_tests/_coverage.md`。
- 若项目确实需要 step/function 级账本,先写项目级 schema、验收命令和 fail-loud 边界,再启用对应 helper。

## 当前 active pipelines

- `01_common_preprocess/`：Layer1公共预处理（井震弱标定/地震体索引化/测井清洗），6条赛道共用，见 `_meta/_registry.yml` id=common_preprocess
- `02_task_datasets/{fault,facies,reservoir,lithofacies,sweetspot,reconstruction}/`：Layer2每赛道样本构建器，统一调用 `_code/dataset_io.py`，见 `_meta/_registry.yml` id=task_datasets
- 统一数据接口：`_code/dataset_io.py`（`save_split`/`load_dataset`/`dataset_stats`），设计说明见 `_wiki-methodology/_wiki/_methods/pipeline-skeleton.md`
