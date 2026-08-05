# P5 R2 六赛道可视化

> **发布红线：本目录是协议覆盖率与模型性能汇总，不是六赛道领域可视化。**
> 禁止把这里的 `track_*.png` 当成断层、地震相、物性、岩相、甜点或重构的领域图进行
> 卡片渲染。真实领域图的唯一发布入口是
> `_pipelines/03_domain_visualization_delivery/`。

这组图只使用六个赛道各自最终 R2 工作结果中的真实汇总数据。源提交、源路径、源文件 SHA-256 和成图 SHA-256 均记录在 `visualization_manifest.json`。未纳入 Git 对象的岩相与重建汇总会原样保存到 `source_snapshots/`，避免工作树清理后无法复现。

## 图件

- `figure_01_protocol_readiness`：左侧是协议单元或证据覆盖率，右侧是数据、实验、模型、可排名性和测试防火墙门禁。
- `figure_02_r2_scientific_results`：六赛道科学结果总览。
  - a：断层赛道已有数据取得基线与最低解锁合同数量。
  - b：两个地震相任务的开发集 mIoU 学习曲线。
  - c：储层属性 PHIF/SW 的同目标归一化 RMSE；KLOGH 因绝对 RMSE 跨 `1e96–1e106`，使用独立的 `log10(RMSE)` 右轴。
  - d：三种岩相模型在 12 个折叠/种子单元上的固定九类 macro-F1。
  - e：甜点评价 T1–T3 相对预算 64 的有利方向变化。
  - f：条件重建四个模型/损失组合从 100 到 400 次更新的 B1 RMSE。
- `tracks/`：六个赛道各自的宽版单图，便于报告、PPT 和后续逐图精修。
- `LARGE_MODEL_ROADMAP.md`：六赛道按数据模态接入基础模型与文本大模型的执行顺序。

## 比较边界

- 六赛道任务、单位和指标不同，不能把 mIoU、macro-F1、RMSE、MAE、AP 放在同一数轴上比较高低。
- 图 1 的覆盖率分母来自各赛道自己的正式协议，表示执行或证据覆盖，不等同于模型精度。
- Property 的 KLOGH 当前数值发散，不能与 PHIF/SW 共用绝对 RMSE 轴，也不能把归一化后的下降误写成有效性能提升。
- Reconstruction 的图只表示 conditional 开发模式；strict 模式没有可用开发单元，不能据此声称现场泛化。
- 所有图均未读取冻结测试集指标。

## 复现

```bash
TMPDIR="$PWD/_tmp" MPLCONFIGDIR="$PWD/_tmp/mplconfig" \
python3 _code/visualization/p5_r2_six_track.py
```

服务器没有安装 Times New Roman 时，脚本会使用度量兼容的 TeX Gyre Termes，并在 manifest 中明确记录；不会静默回退到无关字体。
