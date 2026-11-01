"""Generate figures from actual data/results, preserving numerical sidecars.

SVG and PNG are produced together. Unfinished training requires --development
and is visibly marked. Final result curves require audited multi-seed summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import FancyArrowPatch, Rectangle

FONT_PATH=Path('C:/Windows/Fonts/simsun.ttc')
if FONT_PATH.exists():
    fontManager.addfont(str(FONT_PATH))
    FONT=FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT='Noto Serif CJK SC'
plt.rcParams.update({'font.family':[FONT,'DejaVu Sans'],'axes.unicode_minus':False,
                     'svg.fonttype':'none','font.size':10,'axes.labelsize':11,
                     'axes.titlesize':12,'legend.fontsize':9,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.7})
PALETTE={'baseline':'#343b47','gap':'#397eb5','mlp':'#a280b8',
         'local_global':'#e48d32','local_global_kd':'#39956d'}
NAMES={'baseline':'单出口基线','gap':'GAP 多出口','mlp':'近等参数 MLP',
       'local_global':'局部—全局融合','local_global_kd':'融合 + 蒸馏'}


def source(path):
    path=Path(path);content=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(content).hexdigest()),content


def save(fig,name,evidence,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for ext in ['svg','png']:
        fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight',facecolor='white')
    (out/f'{name}.data.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    plt.close(fig)


def data_figures(data,out):
    from torchvision.datasets import CIFAR100
    ds=CIFAR100(data,train=True)
    split_src,raw=source(Path(data)/'split_cifar100.json');split=json.loads(raw)
    classes=np.random.default_rng(17).choice(100,20,replace=False).tolist()
    labels=np.array(ds.targets);train=np.array(split['train'])
    indices=[int(train[np.flatnonzero(labels[train]==c)[0]]) for c in classes]
    fig,axes=plt.subplots(4,5,figsize=(9,7.7),layout='constrained')
    for ax,idx in zip(axes.flat,indices):
        im,c=ds[idx];ax.imshow(im,interpolation='nearest');ax.axis('off')
        ax.set_title(ds.classes[c],fontsize=10,pad=5)
    save(fig,'data_examples',dict(duty='展示百类小尺寸图像的对象与背景差异',
         sources=[split_src],source_partition='training',sample_indices=indices,
         rule='seed17 selects 20 classes; first fixed training index per class'),out)
    fig,axes=plt.subplots(1,2,figsize=(10,3.1),layout='constrained',gridspec_kw={'width_ratios':[1,1.3]})
    names=['训练','温度校准','策略选择','官方测试'];counts=[len(split[k]) for k in ['train','calibration','selection']]+[10000]
    colors=['#397eb5','#e48d32','#39956d','#838992']
    bars=axes[0].barh(names[::-1],counts[::-1],color=colors[::-1],height=.6)
    for bar,n in zip(bars,counts[::-1]):
        axes[0].text(n+600,bar.get_y()+bar.get_height()/2,f'{n:,}',va='center',fontsize=9)
    axes[0].set_xlim(0,55000);axes[0].set_xlabel('图像数量');axes[0].set_title('(a) 固定分层划分')
    ax=axes[1];ax.axis('off');ax.set_xlim(0,10);ax.set_ylim(0,4);ax.set_title('(b) 选模与最终评价的顺序')
    boxes=[(.3,2.2,'训练模型'),(3.6,2.2,'验证选模'),(6.9,2.2,'校准与选策略'),(6.9,.35,'冻结后测试')]
    for i,(x,y,t) in enumerate(boxes):
        ax.add_patch(Rectangle((x,y),2.5,1,fc='#f5f7fa',ec='#444444',lw=.8))
        ax.text(x+1.25,y+.5,t,ha='center',va='center',fontsize=10)
    for a,b in [((2.8,2.7),(3.6,2.7)),((6.1,2.7),(6.9,2.7)),((8.15,2.2),(8.15,1.35))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#444444',lw=1))
    ax.text(3.3,.85,'各对照使用相同划分\n测试标签不参与模型选择',ha='center',va='center',fontsize=10)
    save(fig,'data_protocol',dict(duty='明确数据职责与防止测试集参与选型',sources=[split_src],
         counts=dict(zip(names,counts)),official_test_count_source='CIFAR-100 official dataset specification'),out)


def training_figures(runs,out,development):
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    evidence=[]
    for name in runs:
        run=Path(name)
        config=json.loads((run/'config.json').read_text())
        if not development and (config['smoke_only'] or not (run/'completed.json').exists()):
            raise RuntimeError(f'{run}: unfinished/smoke run cannot produce final figure')
        src,raw=source(run/'history.jsonl')
        rows=[json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        epochs=[r['epoch'] for r in rows]
        if len(epochs)!=len(set(epochs)) or epochs!=sorted(epochs):
            raise RuntimeError(f'{run}: repeated/out-of-order epoch logs')
        key=config['head']+('_kd' if config['kd'] else '')
        label=NAMES[key]+f'（种子{config["seed"]}）'
        loss=[r['train_loss'] for r in rows]
        val=[100*r['validation']['accuracy'][-1] for r in rows]
        axes[0].plot(epochs,loss,label=label,color=PALETTE[key],lw=1.25)
        axes[1].plot(epochs,val,label=label,color=PALETTE[key],lw=1.25)
        evidence.append(dict(source=src,run=str(run),epochs=epochs,train_loss=loss,
                             validation_final_accuracy_percent=val,development=development))
    for ax in axes:
        ax.set_xlabel('训练轮次');ax.grid(alpha=.18,lw=.5)
    axes[0].set_ylabel('训练目标');axes[1].set_ylabel('最终出口验证准确率（%）')
    axes[0].set_title('(a) 优化过程');axes[1].set_title('(b) 验证集表现')
    axes[1].legend(loc='lower right')
    if development:
        fig.suptitle('训练进行中 · 开发检查图',fontsize=13,color='#8a563b')
    save(fig,'training_development' if development else 'training_curves',
         dict(duty='验证优化收敛与训练完整性，不以训练目标跨方法高低判断优劣',runs=evidence),out)


def result_figures(summary_path,out):
    from summary_guard import verify_summary
    src,_=source(summary_path);summary=verify_summary(summary_path)
    groups=summary['groups']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    records=[]
    for key in ['gap','mlp','local_global','local_global_kd']:
        head='local_global' if key.endswith('_kd') else key
        kd=.5 if key.endswith('_kd') else 0.
        selected=sorted([g for g in groups if g['head']==head and g['kd']==kd and
                         g['criterion']=='budget' and not g['consensus']],key=lambda g:g['budget'])
        if not selected:
            continue
        x=[g['cost_ratio_mean'] for g in selected];y=[100*g['accuracy_mean'] for g in selected]
        err=[100*g['accuracy_std'] for g in selected]
        axes[0].errorbar(x,y,yerr=err,lw=1.1,marker='o',ms=3,capsize=2,color=PALETTE[key],label=NAMES[key])
        records.extend(selected)
    for consensus in [False,True]:
        selected=sorted([g for g in groups if g['head']=='local_global' and g['kd']==.5
                         and g['criterion']=='budget' and g['consensus']==consensus],key=lambda g:g['budget'])
        if selected:
            axes[1].errorbar([g['cost_ratio_mean'] for g in selected],
                             [100*g['accuracy_mean'] for g in selected],
                             yerr=[100*g['accuracy_std'] for g in selected],lw=1.1,marker='s',ms=3,capsize=2,
                             color='#39956d' if consensus else '#e48d32',label='置信度 + 一致性' if consensus else '置信度')
            records.extend(selected)
    # Same trained weights and sequential head costs; these are not separately
    # trained truncated networks. Show fixed depths next to the dynamic policy.
    fixed=[]
    for criterion in ['fixed_exit_1','fixed_exit_2','full_depth']:
        match=[g for g in groups if g['head']=='local_global' and g['kd']==.5
               and g['criterion']==criterion and not g['consensus']]
        if len(match)!=1:raise RuntimeError(f'Missing fixed-depth control: {criterion}')
        fixed.append(match[0])
    axes[1].errorbar([g['cost_ratio_mean'] for g in fixed],
                     [100*g['accuracy_mean'] for g in fixed],
                     yerr=[100*g['accuracy_std'] for g in fixed],fmt='D',ms=4,capsize=2,
                     linestyle='none',color='#343b47',label='同权重固定出口 1 / 2 / 3')
    for index,g in enumerate(fixed,1):
        axes[1].annotate(str(index),(g['cost_ratio_mean'],100*g['accuracy_mean']),
                         xytext=(4,4),textcoords='offset points',fontsize=8)
    records.extend(fixed)
    baseline=[g for g in groups if g['head']=='baseline' and g['criterion']=='full_depth']
    if len(baseline)!=1:raise RuntimeError('Missing unique single-exit baseline')
    records.extend(baseline)
    for ax in axes:
        b=baseline[0]
        ax.axhline(b['accuracy_mean']*100,color=PALETTE['baseline'],lw=1,ls='--',label='单出口准确率参考')
        ax.plot([b['cost_ratio_mean']],[b['accuracy_mean']*100],marker='x',color=PALETTE['baseline'],ms=5)
        ax.grid(alpha=.18,lw=.5);ax.set_xlabel('测试集实际平均 MACs / 基线 MACs')
        ax.set_ylabel('测试准确率（%）');ax.legend()
    axes[0].set_title('(a) 出口结构对比');axes[1].set_title('(b) 同权重动态与固定出口')
    save(fig,'budget_accuracy',dict(duty='同等计算约束下的识别质量与策略作用',sources=[src],
         records=records,errorbars='sample standard deviation over independent training seeds'),out)


def diagnostic_figures(runs,split,out,development):
    from inference import file_sha256
    for name in runs:
        run=Path(name)
        src,raw=source(run/f'{split}_diagnostics.json');stats=json.loads(raw)
        config=json.loads((run/'config.json').read_text())
        if not development and (stats['development'] or config['smoke_only'] or not (run/'completed.json').exists()):
            raise RuntimeError('Development diagnostics cannot produce final figures')
        if stats['weights_sha256']!=file_sha256(run/'best.pt'):
            raise RuntimeError('Diagnostics refer to another checkpoint')
        if stats['prediction_sha256']!=file_sha256(run/f'{split}_predictions.npz'):
            raise RuntimeError('Diagnostics refer to another prediction cache')
        exits=stats['exits']
        if len(exits)==1:
            fig,row=plt.subplots(1,2,figsize=(8.5,3.7),layout='constrained')
            axes=np.asarray(row).reshape(2,1)
        else:
            fig,axes=plt.subplots(2,len(exits),figsize=(4*len(exits),6.1),squeeze=False,layout='constrained')
        risk_max=max(r['risk'] for e in exits for kind in ['raw','calibrated'] for r in e[kind]['risk_coverage'])
        risk_ceiling=min(1.,max(.1,np.ceil(risk_max*10)/10))
        for col,record in enumerate(exits):
            exit_label='最终出口' if len(exits)==1 else f'出口 {record["exit"]}'
            for kind,label,color in [('raw','原始概率','#397eb5'),('calibrated','温度校准','#e48d32')]:
                rows=[b for b in record[kind]['bins'] if b['n']]
                axes[0,col].plot([b['confidence'] for b in rows],[b['accuracy'] for b in rows],
                                 color=color,lw=1,marker='o',ms=3,label=label)
                curve=record[kind]['risk_coverage']
                axes[1,col].plot([r['coverage'] for r in curve],[r['risk'] for r in curve],
                                 color=color,lw=1.1,label=label)
            ax=axes[0,col];ax.plot([0,1],[0,1],ls='--',color='#777777',lw=.8)
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='分箱平均置信度',ylabel='分箱实际准确率',
                   title=f'{exit_label} · 可靠性')
            ax.text(.04,.96,f'ECE：{record["raw"]["ece"]:.4f} → {record["calibrated"]["ece"]:.4f}',
                    transform=ax.transAxes,va='top',fontsize=9)
            axes[1,col].set(xlim=(0,1),ylim=(0,risk_ceiling),xlabel='按置信度保留的样本比例',
                            ylabel='保留样本错误率',title=f'{exit_label} · 风险—覆盖率')
        for ax in axes.flat:ax.grid(alpha=.18,lw=.5)
        axes[0,0].legend(loc='lower right');axes[1,0].legend(loc='upper right')
        scope='验证策略选择集' if split=='validation' else '官方测试集'
        fig.suptitle(f'{run.name} · {scope}'+(' · 开发检查图' if development else ''),
                     color='#8a563b' if development else '#222222',fontsize=12)
        save(fig,f'{run.name}_{split}_reliability',dict(
             duty='分别检验概率可靠性与置信度排序下的错误集中情况；不将校准改善等同于退出收益',
             sources=[src],development=development,scope=stats['scope'],
             exits=exits,binning='15 equal-width confidence bins; empty bins omitted',
             risk_order='Descending confidence, stable ties; each exit considered separately'),out)


def routing_figures(summary_path,out,head,kd,consensus):
    from summary_guard import verify_summary
    src,_=source(summary_path);summary=verify_summary(summary_path)
    records=sorted([g for g in summary['groups'] if g['head']==head and g['kd']==kd
                    and g['criterion']=='budget' and g['consensus']==consensus],key=lambda g:g['budget'])
    if not records:raise RuntimeError('Requested routing comparison is absent')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    pos=np.arange(len(records));bottom=np.zeros(len(records))
    for i,color in enumerate(['#397eb5','#e48d32','#39956d']):
        values=np.array([g[f'exit{i+1}_fraction_mean']*100 for g in records])
        axes[0].bar(pos,values,bottom=bottom,color=color,width=.62,label=f'出口 {i+1}')
        bottom+=values
    axes[0].set_xticks(pos,[str(g['budget']) for g in records])
    axes[0].set(xlabel='验证选择的目标预算比',ylabel='测试集退出比例（%）',ylim=(0,100),title='(a) 按预算分配网络深度')
    axes[0].legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.1),frameon=False)
    x=[g['cost_ratio_mean'] for g in records]
    for metric,label,color in [('harmful_early_exit_fraction','过早退出导致错误','#c35b54'),
                               ('beneficial_early_exit_fraction','提前退出保留正确预测','#397eb5')]:
        axes[1].errorbar(x,[g[metric+'_mean']*100 for g in records],
                         yerr=[g[metric+'_std']*100 for g in records],
                         marker='o',ms=3,lw=1.1,capsize=2,label=label,color=color)
    axes[1].set(xlabel='测试集实际平均 MACs / 基线 MACs',ylabel='占全部测试样本的比例（%）',title='(b) 相对同网络最终出口的预测变化')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.18,lw=.5)
    key=head+('_kd' if kd else '')
    fig.suptitle(NAMES[key]+(' · 置信度 + 一致性' if consensus else ' · 置信度'),fontsize=12)
    save(fig,f'routing_{key}_{"consensus" if consensus else "confidence"}',dict(
         duty='解释计算如何分配，以及相对同权重最终出口的正确/错误转移',sources=[src],records=records,
         definitions=dict(harmful='early wrong AND final correct',beneficial='early correct AND final wrong'),
         denominator='all 10000 official test samples per seed',
         identity='dynamic_accuracy - final_accuracy = beneficial_fraction - harmful_fraction'),out)


def architecture_figures(profile_path,out):
    src,raw=source(profile_path);data=json.loads(raw);profiles=data['profiles']
    model_src,_=source('src/models.py')
    if model_src['sha256']!=data['model_source_sha256']:
        raise RuntimeError('Architecture counts do not match current model code')
    heads=['gap','mlp','local_global']
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    x=np.arange(3)
    for i,head in enumerate(heads):
        values=np.array(profiles[head]['exit_macs'])/1e6
        axes[0].bar(x+(i-1)*.23,values,width=.22,color=PALETTE[head],label=NAMES[head])
    reference=profiles['baseline']['backbone_macs']/1e6
    axes[0].axhline(reference,color=PALETTE['baseline'],ls='--',lw=1,label='完整单出口骨干')
    axes[0].set_xticks(x,['出口 1','出口 2','最终出口'])
    axes[0].set(ylabel='累计执行成本（百万 MACs）',ylim=(0,reference*1.27),title='(a) 不同退出位置的计算路径')
    axes[0].legend(fontsize=8,ncol=2,loc='upper left');axes[0].grid(axis='y',alpha=.15)
    values=[profiles[h]['auxiliary_parameters']/1000 for h in heads]
    bars=axes[1].bar(x,values,color=[PALETTE[h] for h in heads],width=.55)
    axes[1].set_xticks(x,['GAP','MLP','局部—全局'])
    axes[1].set(ylabel='两个辅助出口的参数量（千）',ylim=(0,max(values)*1.4),title='(b) 出口容量与匹配控制')
    for bar,head in zip(bars,heads):
        n=profiles[head]['auxiliary_parameters']
        axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+4,f'{n/1000:.3f}',ha='center',fontsize=9)
    axes[1].text(.5,.96,f'MLP 与融合头的参数差：{data["matched_capacity_difference_parameters"]}',
                 transform=axes[1].transAxes,ha='center',va='top',fontsize=9)
    axes[1].grid(axis='y',alpha=.15)
    save(fig,'architecture_cost',dict(duty='核验出口额外成本与容量对照的匹配程度',sources=[src,model_src],
             profiles=profiles,scope=data['scope']),out)


def exit_figures(summary_path,out):
    from summary_guard import verify_summary
    src,_=source(summary_path);summary=verify_summary(summary_path)
    groups=summary['groups'];records=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    metrics=['exit1_accuracy','exit2_accuracy','final_exit_accuracy']
    for ax,methods in zip(axes,[['gap','mlp','local_global'],['local_global','local_global_kd']]):
        for key in methods:
            head='local_global' if key.endswith('_kd') else key
            kd=.5 if key.endswith('_kd') else 0.
            selected=[g for g in groups if g['head']==head and g['kd']==kd and g['criterion']=='full_depth']
            if len(selected)!=1:raise RuntimeError(f'Missing or duplicate complete-depth record: {key}')
            g=selected[0];records.append(g)
            ax.errorbar([1,2,3],[100*g[m+'_mean'] for m in metrics],
                         yerr=[100*g[m+'_std'] for m in metrics],color=PALETTE[key],
                         marker='o',ms=4,lw=1.2,capsize=3,label=NAMES[key])
        ax.set_xticks([1,2,3],['出口 1','出口 2','最终出口']);ax.set_ylabel('测试准确率（%）')
        ax.grid(alpha=.18);ax.legend(loc='lower right',fontsize=9)
    axes[0].set_title('(a) 无蒸馏条件下的出口结构对比')
    axes[1].set_title('(b) 同一融合结构的蒸馏训练对比')
    save(fig,'exit_ablations',dict(duty='把出口结构与训练辅助的作用分开检验',sources=[src],records=records,
                                 errorbars='sample standard deviation over three independent seeds'),out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('kind',choices=['data','training','results','diagnostics','routing','architecture','exits'])
    p.add_argument('--runs',nargs='+',default=['runs/baseline_seed42'])
    p.add_argument('--data',default='data')
    p.add_argument('--summary',default='results/final/summary.json')
    p.add_argument('--output',default='figures')
    p.add_argument('--development',action='store_true')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--head',choices=['gap','mlp','local_global'],default='local_global')
    p.add_argument('--kd',type=float,default=.5)
    p.add_argument('--consensus',action='store_true')
    p.add_argument('--profiles',default='results/architecture/profiles.json')
    args=p.parse_args()
    if args.kind=='data':data_figures(args.data,args.output)
    elif args.kind=='training':training_figures(args.runs,args.output,args.development)
    elif args.kind=='diagnostics':diagnostic_figures(args.runs,args.split,args.output,args.development)
    elif args.kind=='routing':routing_figures(args.summary,args.output,args.head,args.kd,args.consensus)
    elif args.kind=='architecture':architecture_figures(args.profiles,args.output)
    elif args.kind=='exits':exit_figures(args.summary,args.output)
    else:result_figures(args.summary,args.output)


if __name__=='__main__':main()
