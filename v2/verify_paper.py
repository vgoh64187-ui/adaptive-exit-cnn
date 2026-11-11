"""Audit Word table cells against paper source and frozen result summary."""
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from docx import Document
try:
    from v2.figure_provenance import figure_metadata
except ModuleNotFoundError:
    from figure_provenance import figure_metadata

root=Path.cwd()
if len(sys.argv)!=2:
    raise SystemExit("Usage: python v2/verify_paper.py WORD")
word=Path(sys.argv[1])
source=root/"paper/report_v2.md"
summary=root/"results/v2/final_comparison.json"
data=json.loads(summary.read_text(encoding="utf-8"))
text=source.read_text(encoding="utf-8")
doc=Document(word)
tables=re.findall(r"^表 (\d+) [^\n]+\n\n((?:\|[^\n]+\n)+)",text,re.M)
if [int(n) for n,_ in tables]!=list(range(1,15)) or len(doc.tables)!=14:
    raise RuntimeError("Expected 14 consecutively numbered tables")
for (number,md),table in zip(tables,doc.tables):
    rows=[[c.strip() for c in line.strip().strip("|").split("|")] for line in md.splitlines()]
    rows=[r for r in rows if not all(re.fullmatch(r"[-:\s]+",c) for c in r)]
    actual=[[c.text for c in row.cells] for row in table.rows]
    if rows!=actual:
        raise RuntimeError("Word table differs from paper source: "+number)
v1_audit_path=root/"results/v2/v1_paper_table_audit.json"
v1_audit=json.loads(v1_audit_path.read_text(encoding="utf-8"))
v1_source=root/"paper/report_logic_revision.md"
if hashlib.sha256(v1_source.read_bytes()).hexdigest()!=v1_audit["paper_sha256"]:
    raise RuntimeError("V1 paper source changed after table provenance audit")
v1_tables=re.findall(r"^表 (\d+) [^\n]+\n\n((?:\|[^\n]+\n)+)",
                     v1_source.read_text(encoding="utf-8"),re.M)
revision_audit_path=root/"results/v2/v1_revision_table_audit.json"
revision_audit=json.loads(revision_audit_path.read_text(encoding="utf-8"))
if (tables[:11]!=v1_tables[:11] or v1_audit["tables_verified"]!=[1,2,3,4]
        or revision_audit["tables_verified"]!=list(range(5,12))
        or revision_audit["paper_sha256"]!=v1_audit["paper_sha256"]):
    raise RuntimeError("V1 audited tables were changed during V2 assembly")
for name, expected in revision_audit["sources"].items():
    if hashlib.sha256((root / name.replace("\\", "/")).read_bytes()).hexdigest()!=expected:
        raise RuntimeError("V1 revision table data changed after audit: "+name)
figure_audit_path=root/"results/v2/v1_figure_data_audit.json"
figure_audit=json.loads(figure_audit_path.read_text(encoding="utf-8"))
if figure_audit["paper_sha256"]!=v1_audit["paper_sha256"] or len(figure_audit["checks"])!=9:
    raise RuntimeError("V1 figure-data audit does not match the source paper")
for index, entry in enumerate(figure_audit["checks"]):
    folder="print" if index<4 else "revision" if index<8 else "v2"
    path=root/"figures"/folder/(entry["name"]+".data.json")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=entry["data_sha256"]:
        raise RuntimeError("V1 figure data changed after audit: "+entry["name"])
figures=[int(n) for n in re.findall(r"^!\[图 (\d+) ",text,re.M)]
if figures!=list(range(1,20)):
    raise RuntimeError("Expected 19 consecutively numbered figures")
if len(doc.inline_shapes)<19:
    raise RuntimeError("Missing embedded figure or equation")
elements=doc.element.xpath('.//m:oMath')
math_ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
expected_equations=len(re.findall(r'\\\(.*?\\\)',text))+len(re.findall(r'^\\\[\s*$',text,re.M))
if len(elements)!=expected_equations:
    raise RuntimeError(f"Expected {expected_equations} editable equations, got {len(elements)}")
for element in elements:
    if any(len(e)==0 for e in element.xpath('.//m:nary/m:e',namespaces=math_ns)):
        raise RuntimeError('An editable n-ary equation has an empty operand')
    if any('\\' in (t.text or '') for t in element.xpath('.//m:t',namespaces=math_ns)):
        raise RuntimeError('An equation contains an unconverted command')
labels=[("V1 GAP","GAP 多出口"),("V1 local-global","原局部—全局融合"),
        ("V1 local-global + KD","原融合＋蒸馏"),("V2 residual","V2 残差融合"),
        ("V2 residual + selected KD","V2 残差融合＋选定蒸馏")]
def formatted(stats):
    return f"{100*stats['mean']:.2f} ± {100*stats['sample_sd']:.2f}"
expected12=[["验证目标","模型","测试准确率/%","测试实际计算比/%","预算达标种子"]]
for budget in ("0.65","0.75"):
    for key,label in labels:
        r=data["groups"][key][budget]
        expected12.append([f"{float(budget):.0%}",label,formatted(r["accuracy"]),
                           formatted(r["cost_ratio"]),f"{r['compliant']}/3"])
if [[c.text for c in r.cells] for r in doc.tables[11].rows]!=expected12:
    raise RuntimeError("Table 12 does not match audited V2 result summary")
expected13=[["模型","出口 1","出口 2","最终出口"]]
for key,label in labels:
    expected13.append([label]+[formatted(v) for v in data["groups"][key]["0.65"]["exit_accuracy"]])
if [[c.text for c in r.cells] for r in doc.tables[12].rows]!=expected13:
    raise RuntimeError("Table 13 does not match audited V2 result summary")
selection=json.loads((root/"runs_v2/screening_selection.json").read_text(encoding="utf-8"))
screen_names={"residual_seed42":"无蒸馏", "residual_kd_seed42":"即时蒸馏",
              "residual_delayed_kd_seed42":"延迟渐进蒸馏"}
expected14=[["方案","55% 准确率/%","65% 准确率/%","75% 准确率/%","准确率均值/%","计算比均值/%"]]
for record in [selection["control"],*selection["candidates"]]:
    expected14.append([screen_names[record["run"]]]+[f"{100*r['accuracy']:.2f}" for r in record["per_budget"]]+
                      [f"{100*record['mean_accuracy']:.2f}",f"{100*record['mean_cost']:.2f}"])
if [[c.text for c in r.cells] for r in doc.tables[13].rows]!=expected14:
    raise RuntimeError("Table 14 does not match validation screening record")
v2_delta=data["paired_differences"]["V2 residual minus V1 local-global"]["0.65"]
kd_delta=data["paired_differences"]["V2 residual + selected KD minus V2 residual"]["0.65"]
def pp(value):
    return f"{100*value['mean']:+.2f} ± {100*value['sample_sd']:.2f}"
expected_conclusion=(f"V2 残差融合相对原融合的同种子测试准确率差为 {pp(v2_delta['accuracy'])} 个百分点，"
                     f"实际计算比差为 {pp(v2_delta['cost_ratio'])} 个百分点；"
                     f"选定蒸馏方案相对 V2 无蒸馏组的对应差值为 {pp(kd_delta['accuracy'])} 和 "
                     f"{pp(kd_delta['cost_ratio'])} 个百分点。")
if expected_conclusion not in text:
    raise RuntimeError("V2 conclusion differs from the audited paired summary")
selected_rows=data["groups"]["V2 residual + selected KD"]["0.65"]["per_seed"]
exit_percent=[100*statistics.mean(r["exit_counts"][i]/r["n"] for r in selected_rows) for i in range(3)]
harm=100*statistics.mean(r["harmful_early_exit_fraction"] for r in selected_rows)
benefit=100*statistics.mean(r["beneficial_early_exit_fraction"] for r in selected_rows)
expected_exit=(f"测试样本分别有 {exit_percent[0]:.2f}%、{exit_percent[1]:.2f}% 和 {exit_percent[2]:.2f}% 在三个出口返回。"
               f"相对同一模型的最终出口预测，有害提前退出占全部测试样本的 {harm:.2f}%，"
               f"有益提前退出占 {benefit:.2f}%")
if expected_exit not in text:
    raise RuntimeError("Exit outcome narrative differs from the audited per-seed data")
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
figpaths=[(source.parent / rel).resolve() for rel in re.findall(r"^!\[图 \d+ [^\]]+\]\(([^)]+)\)",text,re.M)]
if any(not p.is_file() for p in figpaths):
    raise RuntimeError("Missing figure source")
if len(doc.inline_shapes)!=len(figpaths):
    raise RuntimeError("Word image count differs from figure count")
for shape,path in zip(doc.inline_shapes,figpaths):
    embedded=shape._inline.xpath('.//a:blip/@r:embed')
    if len(embedded)!=1 or hashlib.sha256(doc.part.related_parts[embedded[0]].blob).hexdigest()!=sha(path):
        raise RuntimeError("Embedded figure differs from cited source: "+str(path))
for name in ("v2_budget_accuracy","v2_paired_effects","v2_exit_risk"):
    image=root/"figures/v2"/(name+".png")
    metadata=root/"figures/v2"/(name+".data.json")
    if json.loads(metadata.read_text(encoding="utf-8"))!=figure_metadata(name,summary,image):
        raise RuntimeError("Comparison figure differs from audited summary: "+name)
report={"scope":"V2 Word table/source/summary and figure reference verification",
        "word_sha256":sha(word),"source_sha256":sha(source),"summary_sha256":sha(summary),
        "tables_verified":14,"v1_table_audit_sha256":sha(v1_audit_path),
        "v1_revision_audit_sha256":sha(revision_audit_path),
        "v1_figure_data_audit_sha256":sha(figure_audit_path),"figures_verified":19,
        "figures":{str(p.relative_to(root)):sha(p) for p in figpaths},
        "inline_shapes":len(doc.inline_shapes),"editable_equations":len(elements)}
out=root/"results/v2/paper_verification.json"
if out.exists():
    raise RuntimeError("Verification output exists; inspect before overwriting")
with out.open("x",encoding="utf-8") as f:
    json.dump(report,f,indent=2)
print(json.dumps(report,ensure_ascii=False))
