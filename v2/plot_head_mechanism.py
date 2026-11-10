"""Draw the exact V1 and V2 local-global exit mechanisms."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":9,"svg.fonttype":"none","pdf.fonttype":42})
fig,axes=plt.subplots(2,1,figsize=(9.1,6.6),constrained_layout=True)
colors={"input":"#e5edf5","global":"#e9f1fb","local":"#fff0df",
        "gate":"#f3effa","mix":"#e6f5ed","out":"#f1f5f8"}


def box(ax,xy,w,h,text,color):
    patch=FancyBboxPatch(xy,w,h,boxstyle="round,pad=0.02,rounding_size=0.055",
                         fc=color,ec="#526676",lw=.9)
    ax.add_patch(patch)
    ax.text(xy[0]+w/2,xy[1]+h/2,text,ha="center",va="center",linespacing=1.3)
    return patch


def line(ax,start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=11,
                                 color="#354a57",linewidth=1.1,connectionstyle="arc3"))


for j,ax in enumerate(axes):
    ax.set_xlim(0,12.8)
    ax.set_ylim(0,4.05)
    ax.axis("off")
    ax.text(.12,3.75,"(a) 原始局部—全局出口" if j==0 else "(b) V2 零初始化残差出口",
            fontsize=11,weight="bold",va="center")
    box(ax,(.20,1.68),1.05,.52,"阶段特征 F",colors["input"])
    box(ax,(1.95,2.65),2.0,.65,"全局平均池化\n全局特征 g",colors["global"])
    box(ax,(1.95,.52),2.0,.80,"深度卷积 + BN + ReLU\n空间加权汇聚 → ℓ",colors["local"])
    box(ax,(4.95,1.62),2.10,.68,"拼接 [g, ℓ] 与线性映射\nSigmoid 得门控 a",colors["gate"])
    formula="z = g + a ⊙ (ℓ − g)" if j==0 else "z = g + γ a ⊙ (ℓ − g)"
    box(ax,(8.0,1.65),2.45,.64,formula,colors["mix"])
    box(ax,(11.03,1.70),1.35,.55,"线性分类",colors["out"])
    line(ax,(1.25,2.0),(1.95,2.96))
    line(ax,(1.25,1.90),(1.95,.92))
    line(ax,(3.95,2.97),(4.95,2.20))
    line(ax,(3.95,.92),(4.95,1.72))
    line(ax,(7.05,1.96),(8.0,1.96))
    line(ax,(10.45,1.96),(11.03,1.96))
    # Direct g and ℓ paths into the fusion operator.
    line(ax,(3.95,3.11),(8.42,2.30))
    line(ax,(3.95,.76),(8.42,1.65))
    if j==1:
        ax.text(8.10,.88,"γ 为可学习标量，初始值为 0",color="#176b58",fontsize=9)
        ax.text(8.10,.49,"初始时 z = g，训练中允许正负变化",color="#176b58",fontsize=9)
out=Path("figures/v2")
out.mkdir(parents=True,exist_ok=True)
for ext in ("png","pdf","svg"):
    fig.savefig(out/f"v2_head_mechanism.{ext}",dpi=300,bbox_inches="tight")
print(out)
