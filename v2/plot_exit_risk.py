"""Plot exit allocation and early-exit outcome changes on the frozen test set."""
import json
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v2.figure_provenance import GROUPS, figure_metadata

source=Path("results/v2/final_comparison.json")
data=json.loads(source.read_text(encoding="utf-8"))
labels=("V1 GAP", "V1 融合", "V1 融合+蒸馏", "V2 残差", "V2 残差+蒸馏")
groups=[]
for name in GROUPS:
    rows=data["groups"][name]["0.65"]["per_seed"]
    if [r["seed"] for r in rows]!=[17,42,73] or any(sum(r["exit_counts"])!=r["n"] for r in rows):
        raise RuntimeError("Unexpected exit counts or seed order")
    groups.append(rows)
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
fig,axes=plt.subplots(1,2,figsize=(10.2,4.4),constrained_layout=True)
x=np.arange(len(groups)); bottom=np.zeros(len(groups))
for exit_index,(color,label) in enumerate((("#4d88b8","出口 1"),("#e39a45","出口 2"),("#6da985","最终出口"))):
    heights=np.array([mean(100*r["exit_counts"][exit_index]/r["n"] for r in group) for group in groups])
    axes[0].bar(x,heights,bottom=bottom,color=color,width=.65,label=label)
    bottom+=heights
axes[0].set(ylabel="测试样本占比 / %",title="(a) 实际退出分布（种子均值）",ylim=(0,100))
for index,(metric,label,color) in enumerate((("harmful_early_exit_fraction","有害提前退出","#b95e50"),
                                             ("beneficial_early_exit_fraction","有益提前退出","#3a9474"))):
    samples=[[100*r[metric] for r in group] for group in groups]
    axes[1].errorbar(x+(index-.5)*.18,[mean(v) for v in samples],yerr=[stdev(v) for v in samples],
                     fmt="o",color=color,capsize=3,markersize=4,lw=1,label=label)
axes[1].set(ylabel="占全部测试样本 / %",title="(b) 相对同模型最终出口的预测变化")
axes[1].legend(frameon=False,fontsize=8)
axes[1].grid(axis="y",color="#e4e9ee",lw=.7)
for ax in axes:
    ax.set_xticks(x,labels,rotation=25,ha="right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc="outside lower center",ncol=3,frameon=False,fontsize=8)
out=Path("figures/v2"); out.mkdir(parents=True,exist_ok=True)
name="v2_exit_risk"
for ext in ("png","pdf","svg"):
    fig.savefig(out/(name+"."+ext),dpi=300,bbox_inches="tight")
(out/(name+".data.json")).write_text(json.dumps(figure_metadata(name,source,out/(name+".png")),
                                                ensure_ascii=False,indent=2),encoding="utf-8")
print(out/name)
