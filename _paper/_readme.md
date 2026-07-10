# _paper/ — 论文(扁平结构)

## 用途

论文写作 + 审稿意见 + rebuttal。**扁平,不分 current/revisions**(用户偏好)。

## 结构

```
_paper/
├── _readme.md           # 本文件
├── main.tex             # ⭐ 论文主文件(英文,master 源头)
├── main_zh.md           # 中文版(给用户看,自动从 main.tex 生成或手写)
├── sections/            # 论文章节
│   ├── 01_intro.tex
│   ├── 02_methods.tex
│   └── ...
├── bib/                 # 参考文献
│   └── refs.bib
├── reviews/             # ⭐ 审稿意见 + rebuttal
│   ├── round_01_submission/
│   │   └── manuscript_2026-06-01.pdf
│   ├── round_02_review/
│   │   ├── reviewer_1.md      # reviewer 原意见
│   │   ├── reviewer_2.md
│   │   ├── reviewer_3.md
│   │   └── rebuttal.md        # 你的回复
│   └── round_03_revised/
│       └── manuscript_revised.pdf
└── _drafts/             # 历史 draft(不重要,定期清理)
```

## 多语言策略

- **英文是源头**(代码改动 / 追根溯源都在英文上)
- **中文是给用户看的**(可视化层 / 对话 / HTML 渲染)
- 改了英文 → cc 可自动生成中文翻译(给你 review)

## 跟 _outputs / _figures 的联动

- 论文里引用的数据 → 链到 `_outputs/paper_final/step_*.csv`
- 论文里引用的图 → 链到 `_figures/<name>/`
- **可追溯**:reviewer 问"这数字怎么来的?" → `_code/trace.py --city xx --indicator yy --version paper_final` 给完整 chain

## Git 范围

- ✅ Git:所有 .tex / .md / .bib / reviewer 意见 / rebuttal
- ❌ 不进 Git:编译产物(`*.aux` / `*.log` / `*.synctex.gz`)
- ⚠️ 大 PDF:< 5MB 可 commit,> 5MB 进 `_tmp/` 或外部存

## 命名

- 主文件:`main.tex`(英文 master)+ `main_zh.md`(中文)
- review 子目录:`round_<NN>_<stage>/`(submission / review / revised)
