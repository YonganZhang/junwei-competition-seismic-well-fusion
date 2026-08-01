# P27 技术报告 1.4 版终审与交付证据

## 验收结论

六赛道技术报告 1.4 版可交付。六个赛道均采用任务背景、方法原理、训练策略、实验设计、评估指标和实验结果六节结构。三维重建的技术总览、方法章节、结果章节和总结均采用 P21 固定基础核结论，并把证据限定为开发空间折与同场区历史版本迁移；当前不追加跨数据集测试。

## Command gate

1. `bash _paper/technical_report/build_report.sh`：成功生成 45 页 A4 PDF。
2. `python3 -m unittest -v _pipelines/05_research_visualization_expansion/tests/test_research_visualization_expansion.py`：4/4 通过。
3. LaTeX 日志检查：无未定义引用、无未定义文献、无编译错误、无越界盒子。复现附录中的长路径产生一项 `Underfull` 提示，不影响正文。
4. 六个赛道章节检查：每个章节文件恰有 6 个固定标题。
5. `unzip -tq _outputs/technical_report_delivery/junwei_six_track_latex_source_latest.zip`：143 项归档，无压缩错误。

## Live / user journey

- PDF 固定地址：`https://share.yongan.site/junwei-six-track-report/junwei_six_track_technical_report.pdf`
- PDF 版本地址：`https://share.yongan.site/junwei-six-track-report/junwei_six_track_technical_report.20260801-122750.pdf`
- PDF SHA-256：`bd793ea45c7687ee842e44d9a3c61af1e8978aaa08dd40e5f41c54aedf733b21`
- 源码包固定地址：`https://share.yongan.site/junwei-six-track-latex-source/junwei_six_track_latex_source_latest.zip`
- 源码包版本地址：`https://share.yongan.site/junwei-six-track-latex-source/junwei_six_track_latex_source_latest.20260801-123538.zip`
- 源码包 SHA-256：`753a0a41289888bb50e141b86d3bebb2ab61690bf43da89ea29e0e70a7837b7f`

## 视觉复核

人工复核封面、目录、插图目录、技术基础总览、三维重建、总结、复现附录与参考文献页面。总览表不再出现中文字符过度拉伸，科研总览图表头完整，P21 架构图与结果图未发生裁切。

## Trace / SSDO audit

`research_visualization_expansion` 在每次生成后重算输入和输出 SHA-256，并由 `validate_manifest` 拒绝来源哈希漂移。本轮合同措辞变更首次触发来源哈希不一致；重新执行渲染后，4 项测试全部通过，说明门禁能够阻止旧清单被误报为完成。PDF 发布由 `pubfile.sh` 同时写入固定指针和时间戳版本，并完成公网 HTTP 200 检查。
