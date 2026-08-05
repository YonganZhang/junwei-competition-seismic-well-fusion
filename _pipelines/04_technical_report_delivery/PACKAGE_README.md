# 军伟六赛道项目——可视化交付包

本压缩包由项目内的固定交付脚本生成，面向技术报告、阶段汇报和后续增量更新。

## 当前内容

- `visualizations/p12/`：已验收的 P12 可视化。当前包含赛道 1（断层）、赛道 3（储层物性）和赛道 5（甜点预测）。
- `technical_report/figures/`：技术报告插图。首版六赛道架构图完成后会自动进入此目录。
- `evidence/review_attestation.json`：P12 图件人工与哈希验收记录。
- `evidence/visual_style_guide.yml`：统一可视化样式合同。
- `PACKAGE_MANIFEST.json`：包内文件清单、大小和 SHA-256。

## 持续更新约定

下载链接使用固定文件名 `junwei_visualizations_latest.zip`。新增或修订图片后，重新运行：

```bash
python3 _pipelines/04_technical_report_delivery/build_visualization_package.py
bash /mnt/data/yongan-admin-2/.codex/skills/share-docs/scripts/pubfile.sh \
  _outputs/technical_report_delivery/junwei_visualizations_latest.zip \
  junwei-visualizations
```

最新指针 URL 保持不变；发布命令同时生成一个不可覆盖的时间戳版本，便于追溯历史交付。

