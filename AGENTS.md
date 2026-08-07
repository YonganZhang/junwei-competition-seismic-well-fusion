# 军伟的比赛 — 地震 + 测井多模态融合识别有利油气目标

六个赛道：①断层识别 ②地震相识别 ③储层物性 ④岩相识别 ⑤甜点评价 ⑥三维重建。
共用七段生命周期 `validate → prepare → baseline → optimize → promote → refit → verify`。

## 接手先读这几个（真源指针，本文件不复制状态）

| 想知道什么 | 去哪读 |
|---|---|
| 当前进度、阻塞、下一步 | `_wiki-methodology/_top/_task_plan.md` |
| 已有结论、哪些路已证伪 | `_wiki-methodology/_top/_findings/_active_digest.md`（含「不要重启的路线」）|
| 代码入口、六赛道接线、当前默认模型 | `_codemap.md` → `_codebook/six_track_pipelines.md` |
| 代码/数据登记 | `_meta/_registry.yml`、`_meta/_data_registry.yml` |
| ⑤甜点当前冠军 / 已否决路线 / 业主裁定 | `_pipelines/02_task_datasets/sweetspot/_outputs/incumbent/incumbent.json` |

```bash
python3 _code/six_track_pipeline/cli.py verify --track all --through verify   # 六赛道状态
python3 ~/.claude/skills/share-top/scripts/topic-brief.py .                   # 六合一冷启动
```

## 三条硬规矩

1. **分清「赛道」和「路线」，否则会得出完全错误的项目判断。**
   六条赛道**全部成功**：都跑通七段生命周期、都有默认模型、都相对基线有改善。
   而某些**具体增强路线**未通过晋级门（`REJECT_AGENT`、`NOT_PROMOTED`、`not_feasible`），
   意思是候选真跑过但没赢过当前默认，**不是赛道失败，更不是流水线失败**。
   看到这些状态不要重开同一条路——先读 `_active_digest.md` 的赛道层/路线层两张表，
   以及对应赛道的 rejected_routes。
2. **不要编辑冻结产物来反映新决策。** `label_mapping.v1.json`、各目标 `split_manifest.json` 的
   sha256 被归档 protocol/manifest 引用（分别有 5 和 7 处），改了就断证据链。新裁定写进当前真源。
3. **指标必须带标签溯源。** ⑤甜点的 T1/T2/T6/T7 标签是 CPI 解释产物（单条 `RHOB` 即可让 T6 达
   R² 0.9696），其指标不能当作预测能力对外引用。详见 finding P44。

## 共享机器纪律

GPU 是多人共享的 8 张 RTX 5080，上面常有 `localadmin` 等其他同学的训练进程。
用 GPU 前先看占用，只动自己账号且确认空闲的进程，不 kill 他人任务。

# 项目报告写作规则

- 撰写或改写技术报告前，必须加载 `share-sci-write` 并完整读取 `~/.codex/skills/share-sci-write/references/writing-styles/yongan-prose-kernel.md`。
- 面向低耐心的普通大学生，坚持一段一条主线、一句一个论点；删除 AI 腔、碎片支线与工程噪音，按“问题—方法—实验—结论”推进。
- 六个赛道固定使用“任务背景、方法原理、训练策略、实验设计、评估指标、实验结果”，不得把工程状态写进学术标题。
