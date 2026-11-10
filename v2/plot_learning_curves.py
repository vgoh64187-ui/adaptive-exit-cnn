"""Plot complete or in-progress learning curves from recorded validation history."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sources=[("GAP 原出口",Path("runs/gap_seed42/history.jsonl"),"#367ab1"),
         ("原局部—全局",Path("runs/lg_seed42/history.jsonl"),"#d57a23"),
         ("V2 残差融合",Path("runs_v2/residual_seed42/history.jsonl"),"#238e72")]
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
fig,axes=plt.subplots(1,3,figsize=(10.2,3.2),constrained_layout=True,sharey=True)
for label,path,color in sources:
    if not path.exists():
        continue
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for index,ax in enumerate(axes):
        ax.plot([x["epoch"] for x in rows],[100*x["validation"]["accuracy"][index] for x in rows],
                lw=1.25,color=color,label=label)
for i,ax in enumerate(axes):
    ax.set_title(f"出口 {i+1}")
    ax.set_xlabel("训练轮次")
    ax.set_xlim(0,160)
    ax.grid(color="#e7edf1",lw=.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("验证准确率 / %")
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="outside lower center",ncol=3,frameon=False)
output=Path("figures/v2")
output.mkdir(parents=True,exist_ok=True)
for ext in ("png","pdf","svg"):
    fig.savefig(output/f"learning_curves_seed42.{ext}",dpi=300,bbox_inches="tight")
