"""Summarize all prespecified validation routing variants and history diagnostics."""
import json
from pathlib import Path
import statistics
import hashlib

root=Path.cwd()
source=root/"results/v2/routing_validation.json"
data=json.loads(source.read_text(encoding="utf-8"))
variants=("confidence21","confidence101","margin005","margin015","safety005")
names={"confidence21":"21分位","confidence101":"101分位",
       "margin005":"101分位+margin0.05","margin015":"101分位+margin0.15",
       "safety005":"101分位+0.5百分点余量"}
lines=["# V2 第一阶段内部验证诊断",
       "",
       "本表来自已有模型的原策略选择子集。按类别分成 1,200 张策略拟合样本和 1,300 张内部留出样本，"
       "三个训练种子共享同一内部划分。温度、权重和检查点已经使用过原验证数据，故此结果仅作探索，"
       "不能作为独立测试表现，也不是三个独立数据划分。",
       "",
       "| 模型 | 预算 | 策略 | 可行种子 | 留出准确率/% | 留出成本/% | 留出超预算种子 |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for model in ("gap","lg_kd"):
    for budget in data["budgets"]:
        for v in variants:
            records=[x for run in data["runs"] if run["model"]==model for x in run["results"]]
            records=[x for x in records if x["variant"]==v and x["budget"]==budget]
            usable=[x for x in records if x["thresholds"] is not None]
            if usable:
                a=statistics.mean(x["held"]["accuracy"] for x in usable)*100
                c=statistics.mean(x["held"]["cost_ratio"] for x in usable)*100
                exceed=sum(x["held"]["cost_ratio"]>budget for x in usable)
                vals=f"{a:.2f} | {c:.2f} | {exceed}/{len(usable)}"
            else:
                vals="— | — | —"
            lines.append(f"| {model} | {budget:.0%} | {names[v]} | {len(usable)}/3 | {vals} |")
history=json.loads((root/"results/v2/history_diagnostics.json").read_text())
lines += ["", "## 检查点分析", "",
          "每组以训练时预先规定的平均出口准确率选择检查点。下表列出该检查点的最终出口验证准确率，"
          "并与同一训练记录中最终出口验证准确率最高的轮次比较。该比较不修改权重。", "",
          "| 模型 | 种子 | 选中轮次 | 选中时最终出口/% | 最终出口最佳轮次 | 最佳准确率/% |",
          "| --- | --- | --- | --- | --- | --- |"]
for x in history["records"]:
    lines.append(f"| {x['model']} | {x['seed']} | {x['selected_epoch']} | {100*x['selected_final']:.2f} | "
                 f"{x['final_best_epoch']} | {100*x['final_best_validation']:.2f} |")
lines += ["", "## 审核说明", "",
          "101 分位数并不保证留出样本的平均成本满足预算。55% 预算下 margin=0.15 的拟合问题对所考察模型均无可行解。"
          "数值属于按选模后的检查点和已有校准温度进行的内部路由诊断。", "",
          "计算源文件 SHA-256 为 "+hashlib.sha256(source.read_bytes()).hexdigest()+"。"]
(root/"results/v2/phase1_validation_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(root/"results/v2/phase1_validation_report.md")
