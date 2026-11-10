"""Produce a seed-42 validation-only structural screening summary."""
import json
from pathlib import Path

models=[("V1 GAP","runs/gap_seed42"),("V1 原融合","runs/lg_seed42"),
        ("V2 残差融合","runs_v2/residual_seed42")]
lines=["# V2 首组结构对照的验证分析","","结果使用种子 42。模型检查点依据包含这批样本的验证集选择，"
       "温度和策略也在原验证集拟合。所有数值属于探索性验证结果。V2 在第 78 轮后从检查点恢复，"
       "工作进程的数据增强随机序列可能与连续训练不同。","","## 各出口验证准确率",
       "","各出口准确率使用合并验证集的 5000 张图像，所有样本均执行到相应出口。","",
       "| 模型 | 出口 1/% | 出口 2/% | 最终出口/% | 选中轮次 |",
       "| --- | --- | --- | --- | --- |"]
policies={}
for label,path in models:
 p=Path(path)
 rows=[json.loads(x) for x in (p/"history.jsonl").read_text().splitlines() if x]
 best=max(rows,key=lambda x:x["selection_score"])
 a=best["validation"]["accuracy"]
 lines.append(f"| {label} | {100*a[0]:.2f} | {100*a[1]:.2f} | {100*a[2]:.2f} | {best['epoch']} |")
 policies[label]=json.loads((p/"policies.json").read_text())
lines += ["","## 动态路由策略","","下表使用 2500 张策略选择验证样本。每组在相同目标预算、"
          "原有 21 分位点候选网格下选择策略，再报告该样本上的准确率与实际平均 MACs 比。"
          "同一批数据用于搜索与报告，存在选择偏差。","","| 目标预算 | 模型 | 验证准确率/% | 验证计算比/% | 出口 1/2/3 样本数 |",
          "| --- | --- | --- | --- | --- |"]
for b in (.55,.65,.75,.85):
 for label,_ in models:
  rows=[r for r in policies[label]["policies"] if r.get("criterion")=="budget" and r.get("budget")==b
        and "policy" in r and not r["policy"]["consensus"]]
  assert len(rows)==1
  r=rows[0]["selection"]
  lines.append(f"| {b:.0%} | {label} | {100*r['accuracy']:.2f} | {100*r['cost_ratio']:.2f} | "
               +"/".join(str(v) for v in r["exit_counts"])+" |")
lines += ["","## 结果边界","",
          "只凭一个种子不能判断稳定收益。预算相同并不保证实际计算量相同。"
          "不得将此表当作官方测试结果或真实设备加速结果。下一步按预先登记的蒸馏方案和种子计划继续。"]
out=Path("results/v2/first_structure_validation.md")
out.write_text("\n".join(lines)+"\n",encoding="utf-8")
print(out)
