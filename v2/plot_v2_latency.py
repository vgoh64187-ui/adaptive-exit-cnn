"""Plot V2 seed-42 interleaved device timing with individual rounds."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

selection=json.loads(Path("runs_v2/screening_selection.json").read_text(encoding="utf-8"))
name=selection["selected_kd"]
data=json.loads((Path("results/v2")/(name+"_latency_interleaved.json")).read_text(encoding="utf-8"))
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
fig,axes=plt.subplots(2,2,figsize=(9.0,5.6),constrained_layout=True)
paths=[("full_same_weight","同权重完整骨干","#367ab1"),
       ("dynamic65","65% 动态退出","#d57a23")]
offsets=np.linspace(-.10,.10,5)
for row,(batch,metric,title,unit) in enumerate([(1,"p50","批量 1","p50 延迟 / ms"),
                                                (32,"throughput","批量 32","吞吐 / 张每秒")]):
 for col,device in enumerate(("cpu","cuda")):
    ax=axes[row,col]
    content=data["devices"][device][str(batch)]["rounds"]
    for index,(key,label,color) in enumerate(paths):
        values=[r[metric] for r in content[key]]
        if len(values)!=5: raise RuntimeError("Expected five interleaved rounds")
        ax.scatter(np.asarray(offsets)+index,values,color=color,s=24,alpha=.8)
        ax.hlines(np.mean(values),index-.18,index+.18,color="#222",lw=1.8)
    ax.set_xticks([0,1],[p[1] for p in paths])
    ax.set_title(("CPU" if device=="cpu" else "GPU")+" · "+title)
    ax.set_ylabel(unit)
    ax.grid(axis="y",color="#e3e9ee",lw=.7)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
out=Path("figures/v2")
out.mkdir(parents=True,exist_ok=True)
for ext in ("png","pdf","svg"):
    fig.savefig(out/f"v2_latency_interleaved.{ext}",dpi=300,bbox_inches="tight")
print(out)
