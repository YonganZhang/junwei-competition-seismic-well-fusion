# _figures/ — 论文 / PPT 用的图

## 用途

每张论文 / PPT 用图 + 它的生成脚本同源。

## 结构

```
_figures/
├── _readme.md
├── _index.md              # 自动生成的图清单(数据源 / 脚本 / 出现在哪)
├── fig_01_world_map.png   # 图本体
├── fig_01_world_map.py    # 生成脚本(读 _outputs/.../*.csv → 出图)
├── fig_02_hk_radar.png
├── fig_02_hk_radar.py
└── ...
```

## 关键设计:图 + 脚本同源

**每张图必有对应 .py**,这样 reviewer 问"怎么画的?" → 直接给 .py。

`_index.md` 自动维护(由 `_code/sync_figures.py` 生成):

| 文件 | 数据源 | 生成脚本 | 出现在 |
|---|---|---|---|
| fig_01_world_map.png | _outputs/paper_final/step_09.csv | fig_01_world_map.py | _paper §3.1 |
| fig_02_hk_radar.png | _outputs/paper_final/step_06_hk.csv | fig_02_hk_radar.py | _ppt/iccs.pptx slide 12 |

## Git 范围

- ✅ Git:**所有 .py 脚本**(必进)+ **小 .png(<1MB)**
- ❌ 不进 Git:大 .png / .pdf(>1MB)→ 进 `_tmp/` 或外部存
- 大图原则:**Git 管脚本,脚本能重新生成图**

## 命名

- 论文用:`fig_<NN>_<short_desc>.png` / `.py`
- PPT 用:同上,可加后缀 `_ppt`(如 `fig_01_world_map_ppt.png`)

## 跟 _paper / _outputs 的联动

- 每张图的"数据源"指向 `_outputs/paper_final/<csv>`
- 每张图的"出现在"指向 `_paper/sections/<file>.tex` 或 `_ppt/<file>.pptx`
