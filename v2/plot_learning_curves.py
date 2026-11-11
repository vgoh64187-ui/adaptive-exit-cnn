"""Plot observed seed-42 validation learning curves without smoothing."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

selection_path=Path("runs_v2/screening_selection.json")
if not selection_path.exists():
    raise RuntimeError("A fixed validation-selected distillation schedule is required")
selection=json.loads(selection_path.read_text(encoding="utf-8"))
chosen=selection["selected_kd"]
if chosen not in ("residual_kd_seed42","residual_delayed_kd_seed42"):
    raise RuntimeError("Unexpected distillation selection")
sources=[("V1 原融合",Path("runs/lg_seed42/history.jsonl"),"#d57a23","--"),
         ("V1 融合 + 蒸馏",Path("runs/lg_kd_seed42/history.jsonl"),"#7164a7","--"),
         ("V2 残差融合",Path("runs_v2/residual_seed42/history.jsonl"),"#238e72","-"),
         ("V2 残差融合 + 选定蒸馏",Path("runs_v2")/chosen/"history.jsonl","#b93f55","-")]
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
fig,axes=plt.subplots(1,3,figsize=(10.2,3.6),constrained_layout=True,sharey=True)
for label,path,color,style in sources:
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows)!=160 or [r["epoch"] for r in rows]!=list(range(1,161)):
        raise RuntimeError("Complete 160-epoch history required: "+str(path))
    for index,ax in enumerate(axes):
        ax.plot([x["epoch"] for x in rows],[100*x["validation"]["accuracy"][index] for x in rows],
                lw=1.05,color=color,linestyle=style,label=label)
for i,ax in enumerate(axes):
    ax.set_title(f"出口 {i+1}")
    ax.set_xlabel("训练轮次")
    ax.set_xlim(0,160)
    ax.grid(color="#e7edf1",lw=.6)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
axes[0].set_ylabel("验证准确率 / %")
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="outside lower center",ncol=2,frameon=False)
output=Path("figures/v2")
output.mkdir(parents=True,exist_ok=True)
for ext in ("png","pdf","svg"):
    fig.savefig(output/f"learning_curves_seed42.{ext}",dpi=300,bbox_inches="tight")
print(output)
