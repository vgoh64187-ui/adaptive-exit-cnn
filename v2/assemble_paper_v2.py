"""Assemble a data-driven V2 paper after all predeclared audits and benchmarks."""
import hashlib
import json
import re
import statistics
from pathlib import Path
from v2.figure_provenance import figure_metadata

root=Path.cwd()
source=root/"paper/report_logic_revision.md"
comparison=root/"results/v2/final_comparison.json"
audits=root/"results/v2/v2_full_audit.json"
selection=root/"runs_v2/screening_selection.json"
resume=root/"results/v2/run_resume_events.json"
routing_path=root/"results/v2/routing_validation.json"
routing_report=root/"results/v2/phase1_validation_report.md"
output=root/"paper/report_v2.md"
if output.exists():
    raise RuntimeError("Paper output exists; refusing to overwrite")
data=json.loads(comparison.read_text(encoding="utf-8"))
for name in ("v2_budget_accuracy","v2_paired_effects","v2_exit_risk"):
    figure=root/"figures/v2"/(name+".png")
    metadata_path=root/"figures/v2"/(name+".data.json")
    if json.loads(metadata_path.read_text(encoding="utf-8"))!=figure_metadata(name,comparison,figure):
        raise RuntimeError("V2 comparison figure provenance mismatch: "+name)
routing=json.loads(routing_path.read_text(encoding="utf-8"))
routing_reference=re.search(r"计算源文件 SHA-256 为 ([0-9a-f]{64})",routing_report.read_text(encoding="utf-8"))
if not routing_reference or routing_reference.group(1)!=hashlib.sha256(routing_path.read_bytes()).hexdigest():
    raise RuntimeError("Routing-validation report is not bound to its data")
routing_figure=root/"figures/v2/routing_validation_exploratory.png"
routing_figure_data=root/"figures/v2/routing_validation_exploratory.data.json"
figure_metadata=json.loads(routing_figure_data.read_text(encoding="utf-8"))
if (figure_metadata["source_sha256"]!=hashlib.sha256(routing_path.read_bytes()).hexdigest()
        or figure_metadata["image_sha256"]["png"]!=hashlib.sha256(routing_figure.read_bytes()).hexdigest()
        or len(figure_metadata["points"])!=24):
    raise RuntimeError("Routing-validation figure is not bound to its source")
expected_points=[]
for run in routing["runs"]:
    for record in run["results"]:
        if (run["model"] in ("gap","lg_kd") and record["budget"] in (.55,.65)
                and record["variant"] in ("confidence21","confidence101") and record["thresholds"] is not None):
            held=record["held"]
            expected_points.append(dict(model=run["model"],budget=record["budget"],seed=run["seed"],
                                        variant=record["variant"],held_n=held["n"],
                                        accuracy=held["accuracy"],cost_ratio=held["cost_ratio"]))
point_key=lambda row:(row["model"],row["budget"],row["seed"],row["variant"])
if sorted(figure_metadata["points"],key=point_key)!=sorted(expected_points,key=point_key):
    raise RuntimeError("Routing figure points differ from validation records")
def routing_stats(model,budget,variant):
    runs=[run for run in routing["runs"] if run["model"]==model]
    if sorted(run["seed"] for run in runs)!=[17,42,73]:
        raise RuntimeError("Expected three routing-diagnostic seeds")
    rows=[]
    for run in runs:
        records=[r for r in run["results"] if r["budget"]==budget and r["variant"]==variant]
        if len(records)!=1 or records[0]["thresholds"] is None:
            raise RuntimeError("Expected a feasible routing-diagnostic policy")
        if records[0]["fit"]["n"]!=1200 or records[0]["held"]["n"]!=1300:
            raise RuntimeError("Unexpected routing-diagnostic sample counts")
        rows.append(records[0]["held"])
    return dict(accuracy=statistics.mean(r["accuracy"] for r in rows),
                cost=statistics.mean(r["cost_ratio"] for r in rows),
                violations=sum(r["cost_ratio"]>budget for r in rows))
route_21=routing_stats("gap",.55,"confidence21")
route_101=routing_stats("gap",.55,"confidence101")
audit=json.loads(audits.read_text(encoding="utf-8"))
selected=json.loads(selection.read_text(encoding="utf-8"))
resume_events=json.loads(resume.read_text(encoding="utf-8")) if resume.exists() else []
if not isinstance(resume_events,list):
    raise RuntimeError("Training interruption records must be a list")
resume_detail="；".join(f"{e['run']} 在第 {e['resume_after_epoch']} 轮后恢复" for e in resume_events) or "未记录恢复事件"
if data["selection"]!=selected["selected_kd"]:
    raise RuntimeError("Summary and frozen selection differ")
lock=root/"runs_v2/v2_screening_lock.json"
if data["locked_v2_protocol_sha256"]!=hashlib.sha256(lock.read_bytes()).hexdigest():
    raise RuntimeError("Summary does not match lock")
locked_sources=json.loads(lock.read_text(encoding="utf-8"))["sources"]
resume_key="results/v2/run_resume_events.json"
if resume.exists():
    if locked_sources.get(resume_key)!=hashlib.sha256(resume.read_bytes()).hexdigest():
        raise RuntimeError("Resume record is not bound to the frozen protocol")
elif resume_key in locked_sources:
    raise RuntimeError("Frozen resume record is missing")
if len(audit["runs"])!=7:
    raise RuntimeError("Full predeclared V2 audit not found")
selected_name=selected["selected_kd"]
benchmark=root/"results/v2"/(selected_name+"_latency_interleaved.json")
if not benchmark.exists():
    raise RuntimeError("Benchmark required before paper assembly")
b=json.loads(benchmark.read_text(encoding="utf-8"))
# Five rotated rounds per path, as in the preregistered timing protocol.
def timing(device,batch,name,key):
    records=b["devices"][device][str(batch)]["rounds"][name]
    if len(records)!=5: raise RuntimeError("Expected five timing rounds")
    return sum(x[key] for x in records)/len(records)
def pair(mean):
    return f"{100*mean['mean']:.2f} ± {100*mean['sample_sd']:.2f}"
def pct(value):
    return f"{100*value:.2f}%"
labels=[("V1 GAP","GAP 多出口"),
        ("V1 local-global","原局部—全局融合"),
        ("V1 local-global + KD","原融合＋蒸馏"),
        ("V2 residual","V2 残差融合"),
        ("V2 residual + selected KD","V2 残差融合＋选定蒸馏")]
method="即时蒸馏" if selected_name=="residual_kd_seed42" else "延迟渐进蒸馏"
if selected_name not in ("residual_kd_seed42","residual_delayed_kd_seed42"):
    raise RuntimeError("Unexpected selection")
screen_rows=["表 14 种子 42 蒸馏时序的验证集初筛结果","",
             "| 方案 | 55% 准确率/% | 65% 准确率/% | 75% 准确率/% | 准确率均值/% | 计算比均值/% |",
             "| --- | --- | --- | --- | --- | --- |"]
screen_candidates=[selected["control"],*selected["candidates"]]
screen_labels={"residual_seed42":"无蒸馏", "residual_kd_seed42":"即时蒸馏",
               "residual_delayed_kd_seed42":"延迟渐进蒸馏"}
if {r["run"] for r in screen_candidates}!=set(screen_labels):
    raise RuntimeError("Unexpected screening candidates")
for record in screen_candidates:
    values=record["per_budget"]
    if len(values)!=3 or any(row["n"]!=2500 for row in values):
        raise RuntimeError("Unexpected screening policy results")
    if (abs(statistics.mean(row["accuracy"] for row in values)-record["mean_accuracy"])>1e-12
            or abs(statistics.mean(row["cost_ratio"] for row in values)-record["mean_cost"])>1e-12):
        raise RuntimeError("Screening averages differ from policy results")
    screen_rows.append("| "+screen_labels[record["run"]]+" | "+" | ".join(
        [f"{100*row['accuracy']:.2f}" for row in values]+
        [f"{100*record['mean_accuracy']:.2f}",f"{100*record['mean_cost']:.2f}"])+" |")
s=source.read_text(encoding="utf-8")
old_figure="![图 9 三个种子的分出口 ECE、覆盖率和条件错误率。误差线为至少两个有效种子间的样本标准差。](../figures/revision/routing_reliability.png)"
new_figure="![图 9 分出口 ECE、覆盖率和条件错误率。误差线为有效种子的样本标准差；n=1 表示仅一个有效种子。](../figures/v2/v1_routing_reliability_valid_seeds.png)"
if s.count(old_figure)!=1: raise RuntimeError("V1 reliability figure anchor mismatch")
s=s.replace(old_figure,new_figure,1)
anchor="### 2.3 多出口训练"
paragraph=("为检验辅助出口的融合形式，V2 在相同局部、全局分支和通道门控上引入一个零初始化的可学习标量 γ，"
           "将融合式改为 z = g + γ a ⊙ (ℓ − g)。训练开始时该出口以全局特征 g 为输入，γ 随训练学习，"
           "可以取正值或负值。每个辅助出口各有一个独立标量。对照保留原结构、数据和训练设置，"
           "用于分析融合形式的影响。两种出口的具体路径见图 12。\n\n")
if s.count(anchor)!=1: raise RuntimeError("Method insertion anchor mismatch")
s=s.replace(anchor,paragraph+anchor,1)
anchor="### 2.4 因果一致性决策与动态执行"
paragraph=("进一步在 V2 残差融合头上比较无蒸馏、即时自蒸馏和延迟渐进自蒸馏。"
           "即时方案从第一轮使用原始蒸馏系数；延迟方案在前 60 轮不使用蒸馏，随后用 20 轮将系数线性升至 0.5。"
           "种子 42 的验证策略选择子集用于预先确定两个蒸馏时序中的一种，比较目标为 55%、65%、75% 预算下"
           "置信度策略的验证准确率均值。选定"+method+"后，与无蒸馏 V2 对照分别补充种子 17 和 73。"
           "该选择使用已参与模型选取的验证数据，属于探索性筛选。\n\n")
if s.count(anchor)!=1: raise RuntimeError("Training insertion anchor mismatch")
s=s.replace(anchor,paragraph+anchor,1)
rows=["表 12 V2 对照在不同目标预算下的准确率与实际计算比（均值 ± 样本标准差）","",
      "| 验证目标 | 模型 | 测试准确率/% | 测试实际计算比/% | 预算达标种子 |",
      "| --- | --- | --- | --- | --- |"]
for budget in ("0.65","0.75"):
    for key,label in labels:
        r=data["groups"][key][budget]
        rows.append(f"| {float(budget):.0%} | {label} | {pair(r['accuracy'])} | {pair(r['cost_ratio'])} | {r['compliant']}/3 |")
rows += ["","表 13 各模型在完整测试集上的固定出口准确率（%，均值 ± 样本标准差）","",
         "| 模型 | 出口 1 | 出口 2 | 最终出口 |","| --- | --- | --- | --- |"]
for key,label in labels:
    r=data["groups"][key]["0.65"]
    rows.append("| "+label+" | "+" | ".join(pair(v) for v in r["exit_accuracy"])+" |")
v2=data["groups"]["V2 residual"]["0.65"]
v1=data["groups"]["V1 local-global"]["0.65"]
kd=data["groups"]["V2 residual + selected KD"]["0.65"]
kd_exit_rows=kd["per_seed"]
if len(kd_exit_rows)!=3 or any(sum(row["exit_counts"])!=row["n"] for row in kd_exit_rows):
    raise RuntimeError("Unexpected selected-model exit counts")
kd_exit_fraction=[statistics.mean(row["exit_counts"][i]/row["n"] for row in kd_exit_rows) for i in range(3)]
kd_harmful=statistics.mean(row["harmful_early_exit_fraction"] for row in kd_exit_rows)
kd_beneficial=statistics.mean(row["beneficial_early_exit_fraction"] for row in kd_exit_rows)
gap=data["groups"]["V1 GAP"]["0.65"]
delta=data["paired_differences"]["V2 residual minus V1 local-global"]["0.65"]
kd_delta=data["paired_differences"]["V2 residual + selected KD minus V2 residual"]["0.65"]
v1_profiles=data["groups"]["V1 local-global"]["0.65"]["per_seed"]
v2_profiles=data["groups"]["V2 residual"]["0.65"]["per_seed"]
if any(a["path_macs"]!=b["path_macs"] or b["parameter_count"]-a["parameter_count"]!=2
       for a,b in zip(v1_profiles,v2_profiles)):
    raise RuntimeError("V1/V2 profiles do not support the paper's count comparison")
v2_parameters=v2_profiles[0]["parameter_count"]
cpu_dynamic=timing("cpu",1,"dynamic65","p50")
cpu_full=timing("cpu",1,"full_same_weight","p50")
gpu_dynamic=timing("cuda",1,"dynamic65","p50")
gpu_full=timing("cuda",1,"full_same_weight","p50")
cpu_dynamic_p95=timing("cpu",1,"dynamic65","p95")
cpu_full_p95=timing("cpu",1,"full_same_weight","p95")
gpu_dynamic_p95=timing("cuda",1,"dynamic65","p95")
gpu_full_p95=timing("cuda",1,"full_same_weight","p95")
cpu_dynamic_tput=timing("cpu",32,"dynamic65","throughput")
cpu_full_tput=timing("cpu",32,"full_same_weight","throughput")
gpu_dynamic_tput=timing("cuda",32,"dynamic65","throughput")
gpu_full_tput=timing("cuda",32,"full_same_weight","throughput")
gpu_timing_cost=b["devices"]["cuda"]["1"]["dynamic_mean_cost_ratio"]
abstract=("摘要：针对 ResNet-18 对不同图像执行固定深度计算的问题，本文在 CIFAR-100 上建立单出口基线，"
          "并在中间阶段增加局部—全局分类出口与基于置信度的动态退出。为研究浅层融合，"
          "进一步设置零初始化可学习残差系数，并比较即时与延迟自蒸馏。"
          "原单出口基线测试准确率为 77.19% ± 0.05%。"
          f"在 65% 验证目标预算下，V2 无蒸馏残差融合的测试准确率为 {pair(v2['accuracy'])}%，"
          f"实际平均计算比为 {pair(v2['cost_ratio'])}%；选定蒸馏方案分别为 "
          f"{pair(kd['accuracy'])}% 和 {pair(kd['cost_ratio'])}%。"
          f"五轮交错计时下，所选 V2 模型在 GPU 批量 1 的动态路径 p50 均值为 {gpu_dynamic:.3f} ms，"
          f"同权重完整骨干为 {gpu_full:.3f} ms。"
          "V2 在已公开原实验结果的同一测试集上评估，结果属于探索性补充。")
s,count=re.subn(r"^摘要：.*$",lambda match:abstract,s,count=1,flags=re.M)
if count!=1: raise RuntimeError("Abstract anchor mismatch")
text=[
"## 4.9 V2 融合结构与蒸馏时序补充实验",
"",
"V2 研究在保留原实验权重和结果的基础上进行。原融合头和 V2 残差头使用相同骨干、"
"出口位置和多出口监督。蒸馏时序在种子 42 的验证集上选择，随后补充种子 17 和 73。"
"三种子均使用相同 CIFAR-100 划分、160 轮训练和一致的置信度策略搜索口径。"
"原测试集结果在 V2 设计前已经公开，本节为该测试集上的探索性补充，不作为全新独立测试结论。"
f"恢复记录：{resume_detail}。恢复后数据加载工作进程的随机增强序列可能与连续运行不同。",
"",
"![图 12 原局部—全局出口与 V2 零初始化残差出口。两个出口各设置一个可学习标量，初始值为零。](../figures/v2/v2_head_mechanism.png)",
"",
"表 12 同时列出不同目标预算下的准确率、测试实际成本和达标种子。预算用于验证策略选择，"
"测试集实际平均计算比可能不同；55% 等其他预算结果见图 13。表 13 对全体测试样本在每个出口分别计算固定出口准确率，"
"用于分析浅层表征与动态路由的关系。",
"",
*rows,
"",
f"在 65% 目标预算下，V2 无蒸馏残差融合的测试准确率为 {pair(v2['accuracy'])}%，"
f"实际平均计算比为 {pair(v2['cost_ratio'])}%；原融合模型分别为 {pair(v1['accuracy'])}% 和 "
f"{pair(v1['cost_ratio'])}%。配对种子准确率差为 {100*delta['accuracy']['mean']:+.2f} ± "
f"{100*delta['accuracy']['sample_sd']:.2f} 个百分点，实际计算比差为 "
f"{100*delta['cost_ratio']['mean']:+.2f} ± {100*delta['cost_ratio']['sample_sd']:.2f} 个百分点。",
"",
f"在相同目标预算下，V2 残差融合＋{method}得到 {pair(kd['accuracy'])}% 的准确率和 "
f"{pair(kd['cost_ratio'])}% 的实际计算比；GAP 多出口对应 {pair(gap['accuracy'])}% 和 "
f"{pair(gap['cost_ratio'])}%。相对 V2 无蒸馏组，蒸馏组的配对准确率差为 "
f"{100*kd_delta['accuracy']['mean']:+.2f} ± {100*kd_delta['accuracy']['sample_sd']:.2f} 个百分点，"
f"实际计算比差为 {100*kd_delta['cost_ratio']['mean']:+.2f} ± "
f"{100*kd_delta['cost_ratio']['sample_sd']:.2f} 个百分点。比较同时涉及训练方式和动态路由落点。",
"",
"![图 13 V1 与 V2 的测试准确率和计算预算对照。(a) 横轴为实际平均计算比；(b) 横轴为验证目标预算。均值与误差线基于三个种子。](../figures/v2/v2_budget_accuracy.png)",
"",
"![图 14 V2 结构和蒸馏对照的同种子差值。横轴分别为准确率差与测试实际计算比差，点为均值，横线为样本标准差。](../figures/v2/v2_paired_effects.png)",
"",
f"V2 两个辅助出口各增加一个标量参数，模型参数量为 {v2_parameters:,}。按本文卷积、线性层和空间加权汇聚"
"统计口径，三个累计路径 MACs 与原融合头相同。该口径未包含新增的逐元素标量乘法，"
"实际延迟还受门控、归一化、批次压缩和设备同步等影响。",
"",
f"在固定的 256 张验证样本、五轮交错计时中，选定 V2 蒸馏模型的批量 1 动态路径与"
f"同权重完整骨干在 CPU 上的 p50 均值分别为 {cpu_dynamic:.3f} ms 和 {cpu_full:.3f} ms，"
f"在 GPU 上分别为 {gpu_dynamic:.3f} ms 和 {gpu_full:.3f} ms；"
f"GPU 计时样本的实际平均计算比为 {100*gpu_timing_cost:.2f}%。"
f"两种路径在 CPU 的 p95 均值分别为 {cpu_dynamic_p95:.3f} ms 和 {cpu_full_p95:.3f} ms，"
f"GPU 的 p95 均值分别为 {gpu_dynamic_p95:.3f} ms 和 {gpu_full_p95:.3f} ms。"
f"批量 32 下，CPU 动态和完整路径吞吐分别为 {cpu_dynamic_tput:.0f} 和 {cpu_full_tput:.0f} 张每秒，"
f"GPU 分别为 {gpu_dynamic_tput:.0f} 和 {gpu_full_tput:.0f} 张每秒。"
"这些测量对应种子 42 和指定设备，其含义需与测试集平均 MACs 分开理解。"
"图 15 同时展示批量 1 延迟与批量 32 吞吐的五轮结果。",
"",
"![图 15 选定 V2 蒸馏模型的 CPU 和 GPU 交错计时。圆点表示各轮结果，黑线表示五轮均值。](../figures/v2/v2_latency_interleaved.png)",
"",
"图 16 对照种子 42 的四组模型训练轨迹，分别绘出三个出口的合并验证集准确率。"
"这些验证结果参与检查点选择，曲线用于展示训练过程，跨种子和预算的结果见表 12 与表 13。"
"训练恢复事件及其对增强随机序列的影响见本节开头。",
"",
"![图 16 种子 42 各出口的验证学习曲线。曲线使用训练日志原始记录，未进行平滑。](../figures/v2/learning_curves_seed42.png)",
"",
"为观察融合分支尺度，从策略选择验证子集的每个类别中固定取一张图像，"
"分别测量各训练种子两个出口的全局特征、局部特征和融合修正向量 L2 范数。"
"图 17 将 100 张图像上的平均局部特征范数与平均修正范数分别除以平均全局特征范数。"
"该描述用于检查分支计算状态，无法单独证明尺度与准确率变化之间的因果关系。",
"",
"![图 17 各模型辅助出口的分支和残差修正幅度。每个点为一个训练种子在固定 100 张验证图像上的均值之比。纵轴为对称对数刻度。](../figures/v2/v2_feature_correction.png)",
"",
"为检验路由阈值选择对预算落点的影响，将原验证策略选择子集按类别分为 1,200 张拟合样本与 1,300 张内部留出样本，"
"对 GAP 与原融合蒸馏模型的三个种子比较 21 和 101 分位点网格。模型权重、检查点及温度此前已使用原验证数据，"
"三个种子共享同一内部划分，因此该实验属于内部探索，不视作独立测试。",
"",
f"以 GAP 和 55% 目标为例，21 分位点的内部留出准确率均值为 {pct(route_21['accuracy'])}，"
f"平均计算比为 {pct(route_21['cost'])}，{route_21['violations']}/3 个种子超出目标；"
f"101 分位点分别为 {pct(route_101['accuracy'])}、{pct(route_101['cost'])} 和 "
f"{route_101['violations']}/3。图 18 展示两个模型在 55% 和 65% 目标下的逐种子实际成本及准确率。"
"该结果用于说明网格密度与预算落点的关系，V2 正式对照仍使用初筛前登记的 21 分位点规则。",
"",
"![图 18 21 与 101 分位点阈值网格的内部留出比较。相同符号表示同一训练种子，灰线连接同一种子的两种策略；虚线为验证目标预算。](../figures/v2/routing_validation_exploratory.png)",
"",
"表 14 给出预登记筛选规则下三种训练方式在种子 42 的验证策略选择子集上的结果。"
"各预算的准确率与实际计算比由同一候选策略得到；均值对三个预算点等权计算。"
"筛选只在即时蒸馏与延迟渐进蒸馏之间进行，无蒸馏组作为结构对照始终保留。"
"该数据已参与检查点选择，初筛差异用于确定扩展实验，不单独推断跨种子稳定性。",
"",
*screen_rows,
"",
f"在 65% 验证目标下，V2 选定蒸馏模型的测试样本分别有 {pct(kd_exit_fraction[0])}、"
f"{pct(kd_exit_fraction[1])} 和 {pct(kd_exit_fraction[2])} 在三个出口返回。"
f"相对同一模型的最终出口预测，有害提前退出占全部测试样本的 {pct(kd_harmful)}，"
f"有益提前退出占 {pct(kd_beneficial)}，均为三个种子的平均值。"
"前者表示提前退出预测错误而最终出口正确，后者表示提前退出正确而最终出口错误；"
"两个比例均以全部测试样本为分母。图 19 同时展示出口分布与预测变化，"
"用于解释计算分配及其识别代价。",
"",
"![图 19 65% 验证目标下的测试出口分布及预测变化。(a) 各出口比例为三个种子的均值；(b) 点和误差线分别为三个种子的均值与样本标准差。](../figures/v2/v2_exit_risk.png)",
"",
]
anchor="## 5 结论"
if s.count(anchor)!=1: raise RuntimeError("Result insertion anchor mismatch")
s=s.replace(anchor,"\n".join(text)+"\n"+anchor,1)
# Preserve the original conclusion and add a measured V2 comparison without assuming direction.
anchor="## 代码与复现"
provenance=(f"在 65% 验证目标预算下，V2 残差融合相对原融合的同种子测试准确率差为 "
            f"{100*delta['accuracy']['mean']:+.2f} ± {100*delta['accuracy']['sample_sd']:.2f} 个百分点，"
            f"实际计算比差为 {100*delta['cost_ratio']['mean']:+.2f} ± "
            f"{100*delta['cost_ratio']['sample_sd']:.2f} 个百分点；"
            f"选定蒸馏方案相对 V2 无蒸馏组的对应差值为 "
            f"{100*kd_delta['accuracy']['mean']:+.2f} ± {100*kd_delta['accuracy']['sample_sd']:.2f} 和 "
            f"{100*kd_delta['cost_ratio']['mean']:+.2f} ± {100*kd_delta['cost_ratio']['sample_sd']:.2f} 个百分点。"
            "这些数值为三个种子的均值与样本标准差，应结合测试实际成本和设备计时解释。"
            "该补充实验采用与原工作相同的数据划分，测试集已在 V2 方案设计前使用，"
            "所报告结果属于探索性分析。\n\n")
if s.count(anchor)!=1: raise RuntimeError("Conclusion insertion anchor mismatch")
s=s.replace(anchor,provenance+anchor,1)
anchor="## 参考文献"
detail=("V2 的实验计划、训练和评估代码位于 v2/，已冻结协议和逐运行文件位于 runs_v2/，"
        "核验和汇总记录见 results/v2/，图表生成源文件与结果见 v2/ 和 figures/v2/。"
        "新旧实验的模型权重和数据集按运行说明保存或重新训练；测试汇总仅在确认冻结记录后生成。\n\n")
if s.count(anchor)!=1: raise RuntimeError("Reproduction insertion anchor mismatch")
s=s.replace(anchor,detail+anchor,1)
output.write_text(s,encoding="utf-8")
print(output)
