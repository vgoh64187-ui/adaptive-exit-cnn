"""Extend the existing self-contained Chinese explainer after V2 paper verification."""
import base64
import hashlib
import html
import json
import re
from pathlib import Path
from statistics import mean
from urllib.parse import quote

root=Path.cwd()
old=root/"output/提前退出CNN论文导读.html"
output=root/"output/提前退出CNN论文导读-V2.html"
metadata_output=root/"output/提前退出CNN论文导读-V2.metadata.json"
summary_path=root/"results/v2/final_comparison.json"
paper_check=root/"results/v2/paper_verification.json"
word=root/"output/基于ResNet18的提前退出CNN课程论文-V2实验版.docx"
if output.exists() or metadata_output.exists():
    raise RuntimeError("V2 explainer exists; inspect it before replacing")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
check=json.loads(paper_check.read_text(encoding="utf-8"))
if check["summary_sha256"]!=sha(summary_path) or check["word_sha256"]!=sha(word):
    raise RuntimeError("Paper or results changed since verification")
if check["source_sha256"]!=sha(root/"paper/report_v2.md"):
    raise RuntimeError("Paper source changed since verification")
summary=json.loads(summary_path.read_text(encoding="utf-8"))
selection=json.loads((root/"runs_v2/screening_selection.json").read_text(encoding="utf-8"))
if summary["selection"]!=selection["selected_kd"]:
    raise RuntimeError("Summary and validation selection differ")
names=(("V1 GAP","V1 GAP 多出口"),("V1 local-global","V1 局部—全局"),
       ("V1 local-global + KD","V1 融合＋蒸馏"),("V2 residual","V2 残差融合"),
       ("V2 residual + selected KD","V2 残差融合＋选定蒸馏"))
def pair(v): return f"{100*v['mean']:.2f} ± {100*v['sample_sd']:.2f}"
rows=[]
for key,label in names:
    r=summary["groups"][key]["0.65"]
    rows.append(f"<tr><th scope='row'>{html.escape(label)}</th><td>{pair(r['accuracy'])}%</td>"
                f"<td>{pair(r['cost_ratio'])}%</td><td>{r['compliant']}/3</td></tr>")
screen_names={"residual_seed42":"无蒸馏","residual_kd_seed42":"即时蒸馏",
              "residual_delayed_kd_seed42":"延迟渐进蒸馏"}
screen_rows=[]
for r in [selection["control"],*selection["candidates"]]:
    screen_rows.append("<tr><th scope='row'>"+screen_names[r["run"]]+"</th>"+
                       "".join(f"<td>{100*v['accuracy']:.2f}%</td>" for v in r["per_budget"])+
                       f"<td>{100*r['mean_accuracy']:.2f}%</td><td>{100*r['mean_cost']:.2f}%</td></tr>")
figures=((12,"v2_head_mechanism.png","先看结构：V2 在原融合表达式中加入可学习系数 γ，两个中间出口各有一个。γ 初始化为 0，因此初始融合输出等于全局特征；训练后 γ 可以为正也可以为负。"),
         (13,"v2_budget_accuracy.png","横轴是真实平均计算比或验证目标预算，纵轴是测试准确率。看同一预算时要先比较实际成本，不能把目标相同当成实际成本相同。"),
         (14,"v2_paired_effects.png","每一行是同一种子下两个方案的差值；0 线表示两者相同。准确率和实际成本应一起看，误差线表示三个种子的样本标准差。"),
         (15,"v2_latency_interleaved.png","计时在固定验证图像上交错执行五轮。CPU/GPU 和批量 1/32 分开看：MACs 的减少不自动等于设备延迟减少。"),
         (16,"learning_curves_seed42.png","曲线来自训练日志，每条线是一个出口的验证准确率。它帮助理解收敛与选模，但只展示种子 42，不代表全部种子的波动。"),
         (17,"v2_feature_correction.png","该图对固定 100 张验证图像观察局部特征和融合修正的尺度，相对全局特征给出比值。它是描述性诊断，不单独证明准确率变化的原因。"),
         (18,"routing_validation_exploratory.png","图中同一形状代表同一个训练种子，灰线连接两种阈值网格；竖虚线是目标预算。留出集用于内部诊断，权重和温度此前已用过原验证数据。"),
         (19,"v2_exit_risk.png","左侧是样本实际在哪个出口返回；右侧比较提前退出与同模型最终出口的预测。有害表示早退错而最终出口对，有益则相反，分母均为全部测试样本。"))
figure_html=[]
for number,name,explanation in figures:
    path=root/"figures/v2"/name
    relative=path.relative_to(root).as_posix()
    recorded={k.replace('\\','/'):v for k,v in check["figures"].items()}
    if relative not in recorded or recorded[relative]!=sha(path):
        raise RuntimeError("Figure changed since Word verification: "+name)
    image=base64.b64encode(path.read_bytes()).decode("ascii")
    caption=html.escape(f"图 {number} · {explanation}")
    figure_html.append(f'<figure class="figure" id="v2-fig-{number}"><figcaption>{caption}</figcaption>'
                       f'<a class="img-link" href="#v2-fig-{number}" aria-label="放大图 {number}">'
                       f'<img src="data:image/png;base64,{image}" alt="{caption}"></a></figure>')
chosen="即时蒸馏" if summary["selection"]=="residual_kd_seed42" else "延迟渐进蒸馏"
if summary["selection"] not in ("residual_kd_seed42","residual_delayed_kd_seed42"):
    raise RuntimeError("Unexpected screening selection")
section=f'''<section id="v2-study">
<div class="tag">11 · V2 结构、蒸馏与实验解释</div>
<h2>从原融合出口到可学习的残差修正</h2>
<p class="lead">原版先建立 ResNet-18 基线，再加入提前退出。V2 在保留原实验的基础上，进一步问：浅层出口的局部与全局信息应如何融合，蒸馏什么时候介入，以及预算阈值如何影响结果？</p>
<h3>1 · 经典网络的问题与结构改进</h3>
<p>完整 ResNet-18 对所有图像执行相同深度。多出口让容易判断的图像提前返回，计算预算由此分配给更难的图像。原辅助头融合全局特征 g 与局部特征 ℓ，表达式为 z = g + a ⊙ (ℓ − g)。V2 在修正项前加入可学习标量：<strong>z = g + γa ⊙ (ℓ − g)</strong>。这里 a 是通道门控，⊙ 表示逐元素乘法。</p>
<details><summary>为什么从 γ = 0 开始？</summary><p>初始化时融合结果等于 g；训练会根据数据调整 γ，局部分支的实际作用还由 a 共同决定。γ 并不被限制在正值或固定区间，因此不能把它简单理解为一个始终介于 0 和 1 之间的比例。是否改善预测，要看对照实验。</p></details>
{figure_html[0]}
<h3>2 · 蒸馏方案怎样选择？</h3>
<p>在相同 V2 结构上比较无蒸馏、从首轮启用的即时自蒸馏，以及前 60 轮关闭、接下来 20 轮逐步升至 0.5 的延迟蒸馏。先用种子 42 的验证策略选择子集，在 55%、65%、75% 三个目标预算上计算准确率，然后对三个预算等权平均；依据预登记规则选定<strong>{chosen}</strong>，再扩展其他种子。无蒸馏组保留为结构对照。</p>
<div class="table-card"><div class="table-title">表 14 · 种子 42 验证初筛</div><div class="table-wrap"><table><thead><tr><th scope="col">训练方式</th><th scope="col">55% 准确率</th><th scope="col">65% 准确率</th><th scope="col">75% 准确率</th><th scope="col">准确率均值</th><th scope="col">计算比均值</th></tr></thead><tbody>{''.join(screen_rows)}</tbody></table></div></div>
<p>这是单种子探索，且模型检查点使用过同一验证数据；表中小幅差异用于选择后续实验，不能单独证明跨种子稳定性。</p>
<h3>3 · 准确率和计算预算一起读</h3>
<p>表 12 的目标预算用于验证集选阈值，测试集实际平均计算比由测试样本的退出位置决定。下表保留两项与达标种子数，均值及误差均来自三个训练种子。</p>
<div class="table-card"><div class="table-title">表 12 · 65% 验证目标预算</div><div class="table-wrap"><table><thead><tr><th scope="col">模型</th><th scope="col">测试准确率</th><th scope="col">测试实际计算比</th><th scope="col">达标种子</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
{figure_html[1]}{figure_html[2]}
<h3>4 · 实际退出与预测变化</h3>
<p>出口分布解释平均计算量：更多样本在浅层返回，通常意味着后续阶段执行得更少。图 19 再对同模型的最终出口做逐样本参照，分辨提前退出带来的错误与纠正；两种比例不是相互独立模型之间的精度差。</p>
{figure_html[7]}
<h3>5 · 训练、特征和路由诊断</h3>
{figure_html[4]}{figure_html[5]}{figure_html[6]}
<p>图 18 的内部划分对不同训练种子是同一份样本，而不是三个独立测试集。V2 的正式多种子对照仍使用初筛前登记的 21 分位点规则。</p>
<h3>6 · 计算量与设备时间</h3>
{figure_html[3]}
<p>计时使用种子 42、固定验证样本和当前设备。批量大小、路由判断与硬件调度都会影响结果；不要用平均 MACs 直接替代实测延迟。</p>
<h3>7 · 如何理解证据范围</h3>
<p>V2 在原实验结果已经被查看后设计，测试时复用了同一个 CIFAR-100 官方测试集。因此这里是探索性补充。模型训练、阈值选择、测试成本和训练恢复事件均应结合正文与复现记录理解。</p>
<p><a href="{quote(word.name)}">打开 V2 Word 论文</a> · <a href="#overview">回到经典网络主线</a></p>
</section>'''
original=old.read_text(encoding="utf-8")
if original.count('</nav>')!=1 or original.count('<footer>')!=1 or original.count('<section id="overview">')!=1:
    raise RuntimeError("Original explainer layout changed")
original=original.replace('<title>提前退出 CNN · 论文导读</title>',
                          '<title>提前退出 CNN · V2 论文导读</title>',1)
original=original.replace('</nav>', '<a href="#v2-study">11 · V2 补充实验</a></nav>',1)
banner='<p class="lead">本页前十节保留原实验导读；新版结构、蒸馏和补充实验见 <a href="#v2-study">第 11 节 V2 说明</a>。</p>\n'
original=original.replace('<section id="overview">',banner+'<section id="overview">',1)
corrected=root/"figures/v2/v1_routing_reliability_valid_seeds.png"
corrected_key=corrected.relative_to(root).as_posix()
if recorded.get(corrected_key)!=sha(corrected):
    raise RuntimeError("Revised V1 reliability figure changed since Word verification")
corrected_image=base64.b64encode(corrected.read_bytes()).decode("ascii")
caption9="图 9 分出口 ECE、覆盖率和条件错误率；n=1 表示仅一个有效种子，因而不绘制样本标准差。"
corrected_figure=(f'<figure class="figure" id="fig-9"><figcaption id="fig-label-9">{caption9}</figcaption>'
                  f'<a class="img-link" href="#fig-9" aria-label="放大图 9">'
                  f'<img src="data:image/png;base64,{corrected_image}" alt="{caption9}" loading="lazy"></a>'
                  '<p class="hint">点击图像查看大图</p></figure>')
original,count=re.subn(r'<figure class="figure" id="fig-9">.*?</figure>',lambda _:corrected_figure,
                       original,count=1,flags=re.S)
if count!=1:
    raise RuntimeError("Original reliability figure anchor changed")
original=original.replace('<footer>',section+'\n<footer>',1)
original=original.replace('对应终稿《提前退出CNN课程论文-含GAP蒸馏消融.docx》。',
                          '前十节对应原实验，V2 内容对应新版论文。',1)
output.write_text(original,encoding="utf-8")
metadata={"output":output.name,"sha256":sha(output),"summary_sha256":sha(summary_path),
          "word_sha256":sha(word),"original_explainer_sha256":sha(old),
          "paper_verification_sha256":sha(paper_check),"embedded_v2_figures":len(figures),
          "figure_sha256":{f"figures/v2/{name}":sha(root/"figures/v2"/name) for _,name,_ in figures}}
metadata["figure_sha256"][corrected_key]=sha(corrected)
metadata_output.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(metadata,ensure_ascii=False))
