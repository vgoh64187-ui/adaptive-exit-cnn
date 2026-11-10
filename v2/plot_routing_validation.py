"""Plot validation holdout routing costs and paired accuracies."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

data = json.loads(Path("results/v2/routing_validation.json").read_text(encoding="utf-8"))
target = Path("figures/v2")
target.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.1), constrained_layout=True, sharey="row")
colors = {"confidence21": "#367ab1", "confidence101": "#d57a23"}
markers = {17:"o", 42:"s", 73:"^"}
for row, model in enumerate(("gap", "lg_kd")):
    for col, budget in enumerate((0.55, 0.65)):
        ax = axes[row, col]
        ax.axvline(budget*100, color="#888888", linestyle="--", lw=1)
        for run in (r for r in data["runs"] if r["model"] == model):
            points = {}
            for record in run["results"]:
                if record["variant"] in colors and record["budget"] == budget and record["thresholds"] is not None:
                    points[record["variant"]] = record["held"]
            if len(points)==2:
                x = [points[v]["cost_ratio"]*100 for v in colors]
                y = [points[v]["accuracy"]*100 for v in colors]
                ax.plot(x, y, color="#bbc3ca", zorder=1, lw=.9)
                for variant in colors:
                    pt = points[variant]
                    ax.scatter(pt["cost_ratio"]*100, pt["accuracy"]*100,
                               c=colors[variant], marker=markers[run["seed"]],
                               s=44, edgecolors="white", linewidths=.5, zorder=2)
        ax.set_title(("GAP" if model == "gap" else "融合 + 蒸馏") + f" · 目标预算 {budget:.0%}")
        ax.set_xlabel("内部留出集平均计算比 / %")
        ax.grid(color="#e5ebef", linewidth=.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("内部留出集准确率 / %")
from matplotlib.lines import Line2D
legend = [Line2D([0],[0],marker="o",color="none",markerfacecolor=color,markeredgecolor="none",label=label)
          for color,label in [("#367ab1","21 分位点"),("#d57a23","101 分位点")]]
legend += [Line2D([0],[0],marker=marker,color="#555",linestyle="none",label=f"种子 {seed}")
           for seed,marker in markers.items()]
fig.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
for ext in ("png","pdf","svg"):
    fig.savefig(target/f"routing_validation_exploratory.{ext}", dpi=300, bbox_inches="tight")
print(target)
