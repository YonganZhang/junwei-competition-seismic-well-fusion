# 六赛道地学智能建模技术报告

这里是报告的长期真源。新的 AI 窗口应先读取项目根目录 `AGENTS.md`，再加载
`share-sci-write` 与永安写作内核，然后从本目录继续扩写。

## 固定目录

六个赛道各自独立成章，并固定使用：

1. 任务背景
2. 方法原理
3. 训练策略
4. 实验设计
5. 评估指标
6. 实验结果

不得把“证据边界、数据门、状态卡”等工程词放进学术标题；必要限制写入正文或复现附录。

## 文件说明

- `src/`：LaTeX 真源；每个赛道一个 section 文件。
- `figures/architecture/`：六张 Nano Banana 2 架构图。
- `figures/results/`：六个赛道的实验结果图。
- `prompts/nano_banana_v1/`：架构图首版提示词。
- `report_manifest.yml`：模型、状态和证据入口。
- `build_report.sh`：XeLaTeX 构建入口。
- `build/junwei_six_track_technical_report.pdf`：轻量交付版。
- `build/junwei_six_track_technical_report_print.pdf`：高分辨率印刷版。

## 扩写原则

新增模型时更新“方法原理”和“训练策略”；新增对照或数据时更新“实验设计”；新增结果
只进入“实验结果”。正文坚持问题—方法—实验—结论主线，复现路径集中放在附录。
