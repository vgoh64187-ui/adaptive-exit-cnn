"""Publication figures generated exclusively from the audited V2 comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

data=json.loads(Path("results/v2/final_comparison.json").read_text(encoding="utf-8"))
out=Path("figures/v2")
out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
groups=[("V1 GAP","#367ab1","o"),("V1 local-global","#d57a23","s"),
        ("V1 local-global + KD","#7164a7","^"),("V2 residual","#238e72","D"),
        ("V2 residual + selected KD","#b93f55","P")]
budgets=("0.55","0.65","0.75","0.85")
fig,axes=plt.subplots(1,2,figsize=(10.4,4.2),constrained_layout=True)
for label,color,marker in groups:
    records=[data["groups"][label][b] for b in budgets]
    x=[100*r["cost_ratio"]["mean"] for r in records]
    y=[100*r["accuracy"]["mean"] for r in records]
    xerr=[100*r["cost_ratio"]["sample_sd"] for r in records]
    yerr=[100*r["accuracy"]["sample_sd"] for r in records]
    axes[0].errorbar(x,y,xerr=xerr,yerr=yerr,fmt=marker+"-",color=color,
                     capsize=2,lw=1,markersize=4,label=label)
    for seed_i in range(3):
        xs=[100*r["per_seed"][seed_i]["cost_ratio"] for r in records]
        ys=[100*r["per_seed"][seed_i]["accuracy"] for r in records]
        axes[0].scatter(xs,ys,s=7,color=color,alpha=.4)
    axes[1].plot([55,65,75,85],y,marker=marker,color=color,lw=1,label=label)
    axes[1].fill_between([55,65,75,85],
        np.array(y)-np.array(yerr),np.array(y)+np.array(yerr),color=color,alpha=.10)
axes[0].set_title("(a) 测试准确率与实际计算比")
axes[0].set_xlabel("测试平均计算比 / %")
axes[1].set_title("(b) 不同验证目标预算下的准确率")
axes[1].set_xlabel("验证目标预算 / %")
for ax in axes:
    ax.set_ylabel("测试准确率 / %")
    ax.grid(color="#e3e9ee",lw=.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
for ext in ("png","pdf","svg"):
    fig.savefig(out/f"v2_budget_accuracy.{ext}",dpi=300,bbox_inches="tight")

pairs=[("V2 residual minus V1 local-global","结构改动"),
       ("V2 residual + selected KD minus V2 residual","蒸馏训练"),
       ("V2 residual + selected KD minus V1 GAP","组合与 GAP"),
       ("V2 residual + selected KD minus V1 local-global + KD","组合与原融合蒸馏")]
fig,axes=plt.subplots(1,2,figsize=(9.4,3.9),constrained_layout=True,sharey=True)
colors_b={".55":"#367ab1",".65":"#238e72",".75":"#d57a23"}
for i,(key,label) in enumerate(pairs):
    for j,b in enumerate((".55",".65",".75")):
        r=data["paired_differences"][key]["0"+b]
        y=i+(j-1)*.19
        for ax,metric in zip(axes,("accuracy","cost_ratio")):
            ax.errorbar(100*r[metric]["mean"],y,xerr=100*r[metric]["sample_sd"],fmt="o",
                        color=colors_b[b],capsize=2,markersize=4,lw=.9)
for ax,title in zip(axes,("(a) 准确率差 / 百分点","(b) 实际计算比差 / 百分点")):
    ax.axvline(0,color="#888",ls="--",lw=.8)
    ax.set_title(title)
    ax.grid(axis="x",color="#e3e9ee",lw=.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_yticks(range(len(pairs)),[p[1] for p in pairs])
axes[0].invert_yaxis()
handles=[Line2D([0],[0],marker="o",color=c,linestyle="none",label=f"{int(float(b)*100)}%")
         for b,c in colors_b.items()]
fig.legend(handles=handles,loc="outside lower center",ncol=3,title="验证目标预算",frameon=False)
for ext in ("png","pdf","svg"):
    fig.savefig(out/f"v2_paired_effects.{ext}",dpi=300,bbox_inches="tight")
print(out)
