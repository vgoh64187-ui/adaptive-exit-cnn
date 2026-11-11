"""Regenerate the V1 reliability figure with explicit valid-seed counts."""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path.cwd()
source = root / "results/revision/routing_analysis.json"
data = json.loads(source.read_text(encoding="utf-8"))
runs = [r for r in data["runs"] if r["model"] == "lg_kd"]
if sorted(r["seed"] for r in runs) != [17, 42, 73]:
    raise RuntimeError("Expected three V1 fusion+distillation runs")
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "font.size": 9})
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), layout="constrained")
x = np.arange(3)
records = []
for offset, (field, label, color) in enumerate((("raw", "校准前", "#8899ad"),
                                                ("calibrated", "校准后", "#29846b"))):
    values = [[100*r["validation_reliability"][j][field]["ece"] for r in runs] for j in x]
    means = [mean(v) for v in values]
    axes[0].bar(x+(offset-.5)*.3, means, width=.29, color=color, label=label)
    axes[0].errorbar(x+(offset-.5)*.3, means, yerr=[stdev(v) for v in values],
                     fmt="none", ecolor="black", capsize=3, lw=1)
axes[0].set(ylabel="验证 ECE / %", title="(a) 温度校准的分出口影响")
axes[0].legend(frameon=False)
for offset, (field, label, color) in enumerate((("coverage", "出口覆盖率", "#397eb5"),
                                                ("error", "出口条件错误率", "#c77b23"))):
    values = [[100*r["risk65"]["exits"][j][field] for r in runs
               if r["risk65"]["exits"][j][field] is not None] for j in x]
    for j, group in enumerate(values):
        if not group:
            continue
        pos = j+(offset-.5)*.3
        axes[1].bar(pos, mean(group), width=.29, color=color, label=label if j == 0 else None)
        if len(group) >= 2:
            axes[1].errorbar(pos, mean(group), yerr=stdev(group), fmt="none",
                             ecolor="black", capsize=3, lw=1)
        else:
            axes[1].annotate("n=1", (pos, mean(group)), xytext=(0, 5),
                             textcoords="offset points", ha="center", fontsize=8)
        records.append({"exit": int(j+1), "metric": field, "n": len(group),
                        "values_percent": group, "mean_percent": mean(group),
                        "sample_sd_percent": stdev(group) if len(group) >= 2 else None})
axes[1].set(ylabel="测试覆盖率或条件错误率 / %", title="(b) 65% 策略的实际出口")
axes[1].set_ylim(top=max(80, axes[1].get_ylim()[1]))
axes[1].legend(frameon=False)
for ax in axes:
    ax.set_xticks(x, ["出口 1", "出口 2", "最终出口"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
output = root / "figures/v2"
output.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg", "pdf"):
    fig.savefig(output / f"v1_routing_reliability_valid_seeds.{ext}", dpi=300, bbox_inches="tight")
metadata = {"scope": "same V1 data, with missing-seed error bars omitted and n=1 annotated",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "records": records}
(output / "v1_routing_reliability_valid_seeds.data.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
print(output)
