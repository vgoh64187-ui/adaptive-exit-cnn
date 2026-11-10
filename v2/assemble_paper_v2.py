"""Assemble a data-driven V2 paper after all predeclared audits and benchmarks."""
import hashlib
import json
from pathlib import Path

root=Path.cwd()
source=root/"paper/report_logic_revision.md"
comparison=root/"results/v2/final_comparison.json"
audits=root/"results/v2/v2_full_audit.json"
selection=root/"runs_v2/screening_selection.json"
resume=root/"results/v2/run_resume_events.json"
output=root/"paper/report_v2.md"
if output.exists():
    raise RuntimeError("Paper output exists; refusing to overwrite")
data=json.loads(comparison.read_text(encoding="utf-8"))
audit=json.loads(audits.read_text(encoding="utf-8"))
selected=json.loads(selection.read_text(encoding="utf-8"))
if data["selection"]!=selected["selected_kd"]:
    raise RuntimeError("Summary and frozen selection differ")
lock=root/"runs_v2/v2_screening_lock.json"
if data["locked_v2_protocol_sha256"]!=hashlib.sha256(lock.read_bytes()).hexdigest():
    raise RuntimeError("Summary does not match lock")
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
s=source.read_text(encoding="utf-8")
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
import re
s,count=re.subn(r"^摘要：.*$",lambda match:abstract,s,count=1,flags=re.M)
if count!=1: raise RuntimeError("Abstract anchor mismatch")
text=[
"## 4.9 V2 融合结构与蒸馏时序补充实验",
"",
"V2 研究在保留原实验权重和结果的基础上进行。原融合头和 V2 残差头使用相同骨干、"
"出口位置和多出口监督。蒸馏时序在种子 42 的验证集上选择，随后补充种子 17 和 73。"
"三种子均使用相同 CIFAR-100 划分、160 轮训练和一致的置信度策略搜索口径。"
"原测试集结果在 V2 设计前已经公开，本节为该测试集上的探索性补充，不作为全新独立测试结论。"
"V2 无蒸馏模型的种子 42 在第 78 轮后从轮次边界恢复，数据加载工作进程的随机增强序列可能与连续运行不同。",
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
"这些测量对应种子 42 和指定设备，其含义需与测试集平均 MACs 分开理解。"
"图 15 同时展示批量 1 延迟与批量 32 吞吐的五轮结果。",
"",
"![图 15 选定 V2 蒸馏模型的 CPU 和 GPU 交错计时。圆点表示各轮结果，黑线表示五轮均值。](../figures/v2/v2_latency_interleaved.png)",
"",
]
anchor="## 5 结论"
if s.count(anchor)!=1: raise RuntimeError("Result insertion anchor mismatch")
s=s.replace(anchor,"\n".join(text)+"\n"+anchor,1)
# Preserve the original conclusion and add a measured V2 comparison without assuming direction.
anchor="## 代码与复现"
provenance=("V2 对照在 65% 目标预算下得到的结构和蒸馏差异，需同时结合三个种子的准确率、"
            "测试实际成本以及设备计时解释。该补充实验采用与原工作相同的数据划分，"
            "测试集已在 V2 方案设计前使用，所报告结果属于探索性分析。\n\n")
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
