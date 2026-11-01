"""Vector mechanism draft tied to src.models (not an experimental result)."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.font_manager import FontProperties

FONT = FontProperties(fname='C:/Windows/Fonts/simsun.ttc')
plt.rcParams.update({'svg.fonttype':'none', 'font.size':11, 'mathtext.fontset':'stix',
                     'axes.unicode_minus':False})
BLUE, ORANGE, GREEN = '#289ad7', '#eea348', '#57a675'


def label(ax, x, y, text, size=11, **kwargs):
    ax.text(x, y, text, ha=kwargs.pop('ha','center'), va='center',
            fontproperties=FONT, fontsize=size*1.3, **kwargs)


def arrow(ax, start, end, color='black', dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                lw=1.1, color=color, linestyle='--' if dashed else '-'))


def box(ax, x, y, w, h, text, color='white', size=10):
    ax.add_patch(Rectangle((x,y), w,h, facecolor=color, edgecolor='black', lw=.9))
    label(ax,x+w/2,y+h/2,text,size)


def tensor(ax, x, y, w, h, color, caption):
    for k in reversed(range(3)):
        dx,dy=.09*k,.08*k
        ax.add_patch(Rectangle((x+dx,y+dy),w,h,facecolor=color,alpha=.55,edgecolor='black',lw=.7))
    for i in range(1,4):
        ax.plot([x+w*i/4]*2,[y,y+h],color='white',lw=.7)
        ax.plot([x,x+w],[y+h*i/4]*2,color='white',lw=.7)
    label(ax,x+w/2,y-.3,caption,10)


def main():
    p=argparse.ArgumentParser();p.add_argument('--draft',action='store_true');args=p.parse_args()
    fig=plt.figure(figsize=(12,8),facecolor='white')
    ax=fig.add_axes([.025,.53,.95,.43]); ax.set_xlim(0,16); ax.set_ylim(0,4.4); ax.axis('off')
    label(ax,.05,4.15,'(a)  残差骨干与按样本提前退出',15,ha='left')
    box(ax,.2,2.1,1.15,.85,'输入图像\n3×32×32','#f5f5f5')
    box(ax,1.8,2.1,1.5,.85,'3×3 卷积\nBN + ReLU')
    arrow(ax,(1.35,2.52),(1.8,2.52))
    xs=[4.,6.8,9.6,12.4]
    dims=['64×32×32','128×16×16','256×8×8','512×4×4']
    previous=3.3
    for i,x in enumerate(xs):
        arrow(ax,(previous,2.52),(x-.05,2.52))
        tensor(ax,x,2.05,1.3,1.05,BLUE,dims[i])
        label(ax,x+.65,3.55,f'Stage {i+1}\n2 个残差块',11)
        previous=x+1.5
    for k,x in enumerate(xs[1:]):
        arrow(ax,(x+.65,1.55),(x+.65,1.18))
        box(ax,x-.05,.55,1.45,.62,'融合出口头' if k<2 else 'GAP + FC','#fff0d8' if k<2 else '#e1f1e7')
        label(ax,x+.65,.15,f'出口 {k+1}：100 类',10)
        if k<2:
            arrow(ax,(x+1.43,.87),(x+2.1,.87),GREEN)
            label(ax,x+2.,.42,'满足策略即返回',9)
    label(ax,2.8,.95,'实线：计算路径\n后续阶段只接收未退出样本',11)
    if args.draft:
        label(ax,15.6,4.1,'结构草图',9,ha='right',color='#666666')

    bx=fig.add_axes([.025,.04,.58,.43]); bx.set_xlim(0,10); bx.set_ylim(0,5); bx.axis('off')
    label(bx,.05,4.7,'(b)  局部—全局门控融合出口',15,ha='left')
    tensor(bx,.2,2.1,.8,1.,BLUE,'F：C×H×W')
    arrow(bx,(1.2,2.6),(1.65,3.6)); arrow(bx,(1.2,2.6),(1.65,1.55))
    box(bx,1.65,3.2,2,.8,'全局平均汇聚\n全局特征 g')
    box(bx,1.65,1.1,2,.9,'深度卷积 3×3\nBN + ReLU','#fff0d8')
    arrow(bx,(3.65,1.55),(4.05,1.55))
    box(bx,4.05,1.1,1.8,.9,'空间加权汇聚\n局部特征 l','#fff0d8')
    arrow(bx,(3.65,3.6),(6.15,3.6))
    bx.plot([5.85,5.98,5.98],[1.55,1.55,3.3],color='black',lw=1.1)
    arrow(bx,(5.98,3.3),(6.15,3.3))
    box(bx,6.15,3.15,2.6,.9,'拼接与线性映射\nSigmoid → 门控 a')
    arrow(bx,(7.45,3.15),(7.45,2.4))
    box(bx,6.15,1.5,2.6,.9,r'$z=g+a\odot(\ell-g)$','#e1f1e7')
    bx.plot([3.85,3.85,5.6],[3.6,2.65,2.65],color='black',lw=1.1)
    arrow(bx,(5.6,2.65),(6.15,2.2))
    arrow(bx,(5.85,1.55),(6.15,1.8))
    arrow(bx,(7.45,1.5),(7.45,.85))
    box(bx,6.15,.15,2.6,.7,'FC → logits q')
    label(bx,2.75,.35,'C = 128 或 256；融合输出维度为 C',10)

    cx=fig.add_axes([.64,.04,.335,.43]); cx.set_xlim(0,6); cx.set_ylim(0,5); cx.axis('off')
    label(cx,0,4.7,'(c)  因果退出与跳算',15,ha='left')
    box(cx,.3,3.45,2.2,.8,'温度缩放概率\n置信度与类别')
    box(cx,3.35,3.45,2.3,.8,'上一出口预测\n已保存类别')
    arrow(cx,(1.4,3.45),(2.95,2.95)); arrow(cx,(4.5,3.45),(2.95,2.95))
    box(cx,1.3,2.1,3.3,.85,'置信度达到验证阈值\n出口2检查类别一致')
    arrow(cx,(1.8,2.1),(1.35,1.1),GREEN); arrow(cx,(4.1,2.1),(4.65,1.1),BLUE)
    box(cx,.25,.35,2.2,.75,'返回当前预测','#e1f1e7')
    box(cx,3.55,.35,2.2,.75,'进入下一阶段','#e7f3fa')
    label(cx,.8,1.6,'满足',10,color=GREEN);label(cx,5.,1.6,'否则',10,color=BLUE)
    label(cx,3,-.1,'最终出口返回；决策不读取未来特征',10)
    target=Path('figures'); target.mkdir(exist_ok=True)
    for extension in ['svg','png']:
        stem='mechanism_draft' if args.draft else 'mechanism'
        fig.savefig(target/f'{stem}.{extension}',dpi=300,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
