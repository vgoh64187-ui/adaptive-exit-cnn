# 论文生成说明

当前 `methods_draft.md` 与 `experiments_draft.md` 为内容草稿，正式实验结果齐全后合并为论文源稿。数学公式以 LaTeX 保留在 Markdown 中；生成 PDF 时使用 360 dpi 公式图片，并核验公式清单的源文件哈希。正文、标题、图表与参考文献仍是可提取文本。

在项目根目录运行，环境需先安装主依赖与 `requirements-report.txt`。当前版式使用 Windows 自带宋体、黑体和 Times New Roman；字体不复制到仓库。

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-report.txt
.venv/Scripts/python.exe paper/render_math.py paper/methods_draft.md --output tmp/report_math
.venv/Scripts/python.exe paper/build_pdf.py paper/methods_draft.md --math-dir tmp/report_math --output tmp/pdfs/methods_layout.pdf --draft
pdftoppm -scale-to 1500 -png tmp/pdfs/methods_layout.pdf tmp/pdfs/methods_page
```

PDF 为单栏、编号公式、三线表样式。支持 Markdown 标题、段落、粗体、行内与独立公式、表格和图片。图片路径相对于 Markdown 文件；图片的替代文本作为图注。生成后必须逐页检查字体、公式、图表、分页与数值。试排稿写入 `tmp/`，不作为最终报告；正式报告生成到 `output/pdf/`，保留源稿与生成脚本。

`paper/build_settings.py` 从实验清单、数据划分和已完成的基线/GAP 配置生成训练设置表及来源哈希；此表中的 15 次为预先声明的实验规模，实验未全部完成时不能表述为已完成。`paper/build_tables.py` 从模型统计和正式汇总生成其余表格。合并正式论文前须重新生成并核验表号及最新来源。

本机工作时，公式渲染复用了训练环境的 Matplotlib，PDF 排版使用 Codex 文档运行时中的 ReportLab；两者通过哈希核验的公式清单衔接，不向正在训练的环境安装包。
