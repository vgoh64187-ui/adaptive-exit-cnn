"""Plot descriptive feature norms from the fixed validation diagnostic sample."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data=json.loads(Path("results/v2/features_validation.json").read_text())
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
fig,axes=plt.subplots(1,2,figsize=(8,3.3),constrained_layout=True,sharey=True)
colors={"global_norm":"#367ab1","local_norm":"#d57a23"}
for i,ax in enumerate(axes):
    for center,model in enumerate(("lg","lg_kd")):
        runs=[r for r in data["runs"] if r["model"]==model]
        for offset,run in zip((-.12,0,.12),runs):
            for feature,color in colors.items():
                xx=center+offset+(-.035 if feature=="global_norm" else .035)
                ax.scatter(xx,run["exit_means"][i][feature],color=color,s=28)
        ax.axvline(center+.5,color="#e9eef3",lw=.6)
    ax.set_yscale("log")
    ax.set_xticks([0,1],["无蒸馏融合","融合 + 蒸馏"])
    ax.set_title(f"出口 {i+1}")
    ax.grid(axis="y",color="#e2e8ed",lw=.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("特征向量 L2 范数的样本均值（对数刻度）")
handles=[Line2D([0],[0],marker="o",color="none",markerfacecolor=c,
                markeredgecolor="none",label=l) for c,l in [("#367ab1","全局分支"),("#d57a23","局部分支")]]
fig.legend(handles=handles,loc="outside lower center",ncol=2,frameon=False)
out=Path("figures/v2")
out.mkdir(parents=True,exist_ok=True)
for ext in ("png","pdf","svg"):
    fig.savefig(out/f"feature_norms_exploratory.{ext}",dpi=300,bbox_inches="tight")
print(out)
