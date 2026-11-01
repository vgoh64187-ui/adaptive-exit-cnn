# 最终论文独立复现检查

- 检查时间：2026-11-01；项目副本由本地提交 `5903670` 的 `git archive` 导出至独立目录，不包含 `runs/` 与数据集。
- Python 环境：先前通过隔离安装核验的 Python 3.12.14 环境，未启用系统 site-packages；PyTorch 2.5.1+cu121，ReportLab 4.4.9。
- 在导出副本执行 `python -m unittest discover -s tests -v`：28 项通过。测试包含动态跳层、混合路由、协议冻结和结果核验。
- 依次执行 `paper/assemble_paper.py`、`paper/audit_report.py`、`paper/render_math.py` 与 `paper/build_pdf.py --draft`：成功生成 12 页检查 PDF。正文数字核验 41 项，引用的 7 张图均存在。源 Markdown SHA-256 为 `ac1ab0083351ed33fa80fd5d7ada00c65e41ae0424801c0426384c8bb50deb4b`。
- 上述独立副本生成的检查稿来自当时源文件。此后正式报告补充实验设备与仓库地址并重新渲染，共 12 页；各页已检查，正式版未显示检查稿标记。当前 PDF SHA-256 为 `e30f3863df5862b4c49641211648ae32a022974c3fec45f5c4107e38bb1d239d`。
- 此复现只验证仓库中的结果、报告生成和代码测试；不意味着在无数据与检查点的副本中重新训练了 15 个模型，也不声称重新执行了完整测试和计时实验。
