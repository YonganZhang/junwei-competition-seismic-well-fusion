# P26 主线合并与报告交付验收证据

## 验收结论

2026-08-01，P4--P24 累计研究线已从 `final-integration` 无损合并到
`master@8375b97`。合并前先将主工作树与集成分支重叠的文档保全为
`f4b6be8`，其他非重叠的用户改动保持未提交状态，未被覆盖。

赛道⑥的默认交付配置固定为 P21；P20 LoRA、Adapter 和分阶段解冻保留为
诊断证据，但不启用。P24 只支持同场区历史属性版本迁移，不扩展为
cross-field 或 fresh-blind 声明。

## Command gate

1. `python3 -m unittest discover -s _pipelines/03_domain_visualization_delivery/tests -p 'test_*.py' -v`
   结果：7/7 通过。
2. `cd _pipelines/02_task_datasets/reconstruction && PYTHONPATH=. python3 -m unittest -v
   _tests.test_p21_fixed_foundation_ensemble _tests.test_p24_historical_transfer`
   结果：11/11 通过。
3. `python3 -m unittest -v
   _pipelines.05_research_visualization_expansion.tests.test_research_visualization_expansion`
   结果：4/4 通过。
4. `sixone-cli.py doctor .`
   结果：`verdict=ok`，2 条 pipeline 均为 `fresh`。

## Live / user journey

- PDF 固定链接：
  `https://share.yongan.site/junwei-six-track-report/junwei_six_track_technical_report.pdf`
- PDF HTTP 200，`Content-Type=application/pdf`，`Content-Length=6187086`。
- PDF 本地 SHA-256：
  `41f7fa098b5b0ce1c3c903a5da819790a941220f9cc4dbf0ec6ffead4809a8b9`；45 页 A4。
- LaTeX 源码包固定链接：
  `https://share.yongan.site/junwei-six-track-latex-source/junwei_six_track_latex_source_latest.zip`
- 源码包 HTTP 200，140 个归档条目，`unzip -t` 无错误，SHA-256 为
  `5678a5c9a1f7ed913ed29f306a4d65b2f9a00bac00fb8cfcaf3acc8ab8a2ea0a`。

## Trace / SSDO audit

- PDF 已通过 XeLaTeX 构建、引用/交叉引用检查和重建章节逐页视觉复核。
- P21/P24 数值来自已归档 JSON，报告没有将 LoRA 的非零梯度写成指标贡献。
- 公网发布同时保留时间戳版本 URL 和稳定 latest URL；后续更新继续复用同一 topic。
- 本轮仅本地合并和发布文件，没有 push 远端仓库。
