"""Build revision figures and manuscript sections directly from measured records."""
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from make_figures import plt, save, source

RESULTS = Path("results/revision")
FIGURES = Path("figures/revision")
LABELS = {"gap": "GAP", "mlp": "近等参数 MLP", "lg": "局部—全局", "lg_kd": "局部—全局＋蒸馏"}
COLORS = dict(zip(LABELS, ("#397eb5", "#9169a9", "#c77b23", "#29846b")))
TIMING_NAMES = {"single_exit_baseline": "单出口基线", "full_same_weight": "同权重完整骨干",
                "fixed_exit_1": "固定出口 1", "fixed_exit_2": "固定出口 2", "dynamic65": "动态 65%"}


def table(columns, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in [columns, ["---"] * len(columns), *rows])


def fig(number, caption, name):
    return f"![图 {number} {caption}](../figures/revision/{name}.png)"


def pct(value):
    return f"{100 * value:.2f}"


def budget_tables(data, runs):
    checks = data["budget_compliance"]
    rows = []
    for budget in (.55, .65, .75, .85):
        for model, label in LABELS.items():
            group = [r for r in checks if r["budget"] == budget and not r["consensus"]
                     and r["run"].rsplit("_seed", 1)[0] == model]
            rows.append([f"{budget:.0%}", label, pct(mean(r["accuracy"] for r in group)),
                         pct(mean(r["actual"] for r in group)), f"{sum(r['met'] for r in group)}/3",
                         pct(max(0., max(r["actual"] for r in group) - budget))])
    table_text = table(["目标", "模型", "准确率/%", "计算比/%", "达标", "最大超额/百分点"], rows)
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), layout="constrained")
    for model, label in LABELS.items():
        group = [r for r in checks if not r["consensus"] and r["run"].rsplit("_seed", 1)[0] == model]
        for budget in (.55, .65, .75, .85):
            samples = [r for r in group if r["budget"] == budget]
            axes[0].scatter([100*r["actual"] for r in samples], [100*r["accuracy"] for r in samples],
                            color=COLORS[model], alpha=.7, s=18, label=label if budget == .55 else None)
            axes[1].scatter([100*budget]*3, [100*(r["actual"]-budget) for r in samples],
                            color=COLORS[model], alpha=.7, s=18)
    axes[0].set(xlabel="测试平均计算比 (%)", ylabel="测试准确率 (%)", title="A 逐种子成本与准确率")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=.8)
    axes[1].set(xlabel="验证目标预算 (%)", ylabel="测试计算比减预算 (百分点)", title="B 预算偏差")
    save(figure, "budget_compliance", {"source": str(RESULTS/"routing_analysis.json"), "rows": rows}, FIGURES)
    matched = []
    for cap in (.65, .75):
        for model, label in LABELS.items():
            group = [m for r in runs[model] for m in r["matched"] if m["target"] == cap]
            matched.append([f"{cap:.0%}", label, f"{len(group)}/3",
                            pct(mean(m["validation_cost"] for m in group)) if group else "—",
                            pct(mean(m["test"]["cost_ratio"] for m in group)) if group else "—",
                            pct(mean(m["test"]["accuracy"] for m in group)) if group else "—"])
    return "\n\n".join([
        "## 4.5 预算达标与计算成本对照",
        f"按种子核对全部预算策略，共有 {sum(not r['met'] for r in checks)}/{len(checks)} 个组合在测试集超出各自验证目标。"
        "表 5 单列置信度策略，其中最大超额按三个种子中的最大正偏差计算。55% 目标下，GAP 的准确率高于融合蒸馏组。"
        "两者在 65% 目标下的排序反转，说明组合收益取决于预算。",
        "表 5 置信度策略的测试预算达标情况", table_text,
        fig(8, "逐种子的实际计算成本与预算偏差。虚线表示测试平均成本等于验证目标。", "budget_compliance"),
        "进一步在验证策略选择子集上，将候选策略的实际计算比限制在目标下方 0.5 个百分点内，再选择验证准确率最高的阈值。"
        "表 6 记录各组能找到候选的种子数，并给出测试端实际落点。少于三个种子的组按可用种子汇总，跨组比较时应核对种子构成。"
        "计算带宽约束施加在验证端，测试端成本可随样本分布变化。",
        "表 6 验证成本对齐后的测试结果",
        table(["目标", "模型", "可用种子", "验证成本/%", "测试成本/%", "准确率/%"], matched),
        "65% 附近的成本对齐比较支持融合蒸馏组合在这一预算邻域的准确率收益。"
        "75% 附近无蒸馏融合组也高于 GAP，说明结构效果随预算变化。"
        "这些点与原表 3 的策略选择目标不同，允许对齐策略为接近指定成本放弃一部分验证集准确率。",

    ])


def nonempty_errors(exits):
    return [e["error"] for e in exits if e["error"] is not None]


def reliability_section(runs):
    subset = runs["lg_kd"]
    rows = []
    risk = []
    for i in range(3):
        diagnostics = [r["validation_reliability"][i] for r in subset]
        exits = [r["risk65"]["exits"][i] for r in subset]
        risk.append(exits)
        rows.append([i+1, pct(mean(d["raw"]["ece"] for d in diagnostics)),
                     pct(mean(d["calibrated"]["ece"] for d in diagnostics)),
                     f'{mean(d["raw"]["nll"] for d in diagnostics):.3f}',
                     f'{mean(d["calibrated"]["nll"] for d in diagnostics):.3f}',
                     pct(mean(e["coverage"] for e in exits)), pct(mean(nonempty_errors(exits))) if nonempty_errors(exits) else "—",
                     f"{len(nonempty_errors(exits))}/3"])
    fig_obj, axes = plt.subplots(1, 2, figsize=(8.4, 3.25), layout="constrained")
    x = np.arange(3)
    for i, (field, label) in enumerate([("raw", "校准前"), ("calibrated", "校准后")]):
        values = [[100*r["validation_reliability"][j][field]["ece"] for r in subset] for j in x]
        axes[0].bar(x + (i-.5)*.3, [mean(v) for v in values], width=.29,
                    yerr=[stdev(v) for v in values], capsize=3, label=label, color=("#8899ad", "#29846b")[i])
    axes[0].legend(frameon=False)
    axes[0].set(ylabel="验证 ECE (%)", title="A 温度校准的分出口影响")
    for i, (field, label) in enumerate([("coverage", "出口覆盖率"), ("error", "出口条件错误率")]):
        values = [[100*e[field] for e in group if e[field] is not None] for group in risk]
        axes[1].bar(x+(i-.5)*.3, [mean(v) if v else np.nan for v in values], width=.29,
                    yerr=[stdev(v) if len(v)>1 else 0 for v in values], capsize=3, label=label, color=("#397eb5", "#c77b23")[i])
    axes[1].set(ylabel="测试样本比例或错误率 (%)", title="B 65% 策略的实际出口")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, ["出口 1", "出口 2", "最终出口"])
    save(fig_obj, "routing_reliability", {"source": str(RESULTS/"routing_analysis.json"), "rows": rows}, FIGURES)
    agreement = []
    for i in range(0, 10, 2):
        values = [[r["agreement65"][i+j] for r in subset] for j in (0, 1)]
        label = f'{values[0][0]["lower"]:.2f}–{min(1., values[0][0]["upper"]):.2f}'
        counts = [sum(v["count"] for v in group) for group in values]
        rates = [pct(sum(v["correct"] for v in group)/n) if n else "—" for group,n in zip(values,counts)]
        agreement.append([label, counts[0], rates[0], counts[1], rates[1]])
    comparison = []
    for b in (.55, .65, .75):
        for model in ("gap", "lg_kd"):
            outcomes = [next(m for m in r["budget_metrics"] if m["budget"] == b) for r in runs[model]]
            comparison.append([f"{b:.0%}", LABELS[model], *[
                pct(mean(m[mode][metric] for m in outcomes)) for mode in ("raw", "calibrated")
                for metric in ("accuracy", "cost_ratio")]])
    random_accuracy = mean(r["risk65"]["random_same_exit_counts_expectation"] for r in subset)
    selected_accuracy = mean(r["risk65"]["accuracy"] for r in subset)
    transitions = ", ".join(f'{mean(e["harmful"] for e in group):.1f}/{mean(e["beneficial"] for e in group):.1f}' for group in risk[:2])
    return "\n\n".join([
        "## 4.6 校准与退出可靠性",
        "在相同权重上，将未校准置信度重新用验证策略选择子集搜索阈值，与原温度校准策略比较。"
        "表 7 同时报告准确率和实际成本，校准收益随出口和预算而变。表 8 的 ECE 和 NLL 在验证策略选择子集计算，"
        "覆盖率与条件错误率则在冻结策略的测试样本上计算。条件错误率只对实际在对应出口返回的样本取平均，未在该出口返回样本的种子不参与条件错误率汇总。",
        "表 7 校准前后重新选择置信度阈值的测试结果",
        table(["目标","模型","校准前准确率/%","校准前成本/%","校准后准确率/%","校准后成本/%"], comparison),
        "表 8 融合蒸馏模型的分出口可靠性与实际路由",
        table(["出口","原始 ECE/%","校准 ECE/%","原始 NLL","校准 NLL","覆盖率/%","错误率/%","错误率种子"], rows),
        fig(9, "三个种子的分出口 ECE、覆盖率和条件错误率。误差线为至少两个有效种子间的样本标准差。", "routing_reliability"),
        f"在 65% 主策略的测试退出计数下，随机分配相同数量样本到各出口的期望准确率为 {pct(random_accuracy)}%，"
        f"置信度路由实际为 {pct(selected_accuracy)}%。随机对照使用测试退出计数固定算力，只用于回顾性机制诊断。"
        f"相对同权重最终出口，第一、第二出口每万张样本的有害与有益早退平均计数依次为 {transitions}。",
        "为检验一致性的附加信息，表 9 对实际到达第二出口的样本按第二出口置信度分箱，并分开汇总预测一致与不一致时的正确率。"
        "同一样本在多个种子上重复出现，表内合计仅作描述。分箱后的关联可解释一致性判据的行为，其策略收益仍需结合预算和覆盖率判断。",
        "表 9 到达第二出口的样本按置信度和一致性分组",
        table(["置信度区间","不一致计数","不一致准确率/%","一致计数","一致准确率/%"], agreement),
    ])


def timing_section():
    path = RESULTS/"latency_interleaved.json"
    timing = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), layout="constrained")
    for ax, (device, batch) in zip(axes.flat, [("cpu","1"),("cuda","1"),("cpu","32"),("cuda","32")]):
        series = timing["devices"][device][batch]
        for index, (name, label) in enumerate(TIMING_NAMES.items()):
            values = series["rounds"][name]
            points = [r["p50"] if batch=="1" else r["throughput"] for r in values]
            ax.scatter(points, [index]*len(points), color="#29846b" if name=="dynamic65" else "#397eb5", s=19, alpha=.75)
            ax.scatter([mean(points)], [index], color="#222222", s=20, marker="|")
            rows.append([device.upper(), batch, label,
                         f'{mean(r["p50"] for r in values):.3f}', f'{mean(r["p95"] for r in values):.3f}',
                         f'{mean(r["throughput"] for r in values):.0f}'])
        ax.set_yticks(range(5), list(TIMING_NAMES.values()))
        ax.invert_yaxis()
        ax.set(title=f"{device.upper()}   批量 {batch}", xlabel="每轮 p50 / ms" if batch=="1" else "每轮吞吐 / 张每秒")
    save(figure, "latency_interleaved", {"source": str(path), "rows": rows}, FIGURES)
    gpu = timing["devices"]["cuda"]["1"]
    dyn, full = (mean(r["p50"] for r in gpu["rounds"][key]) for key in ("dynamic65", "full_same_weight"))
    return "\n\n".join([
        "## 4.7 65% 主策略的交错计时",
        "补测固定种子 42 的融合蒸馏权重和验证选择的 256 张图像，在 CPU 与 GPU、批量 1 与 32 下执行。"
        "每种路径预热 32 批，随后用轮换顺序交错测量五轮。固定浅层直接执行至指定出口，"
        "完整骨干绕过辅助头，动态路径则包含置信度判断、索引和实际退出。所有动态预测及退出位置与同输入全前向模拟一致。",
        f"GPU 批量 1 时，65% 动态策略和同权重完整骨干的每轮 p50 均值分别为 {dyn:.3f} 和 {full:.3f} ms，"
        f"该计时子集上动态路径的实际平均计算比为 {pct(gpu['dynamic_mean_cost_ratio'])}%。"
        "各轮测量使用同一模型和输入，描述当前机器上的计时波动。固定路径对照有助于观察实际执行差异，路由开销的精确归因还需分段分析。",
        "表 10 交错计时的五轮均值",
        table(["设备","批量","路径","p50/ms","p95/ms","吞吐/张每秒"], rows),
        fig(10, "批量 1 每轮延迟与批量 32 每轮吞吐。每个圆点为一轮，黑色短线为五轮均值。", "latency_interleaved"),
    ])


def factorial_section(runs):
    from audit_results import audit_run
    import torch

    torch.set_num_threads(2)
    lock = Path("runs/revision/protocol_lock.json")
    controls = [Path(f"runs/revision/gap_kd_seed{seed}") for seed in (17, 42, 73)]
    if not all((run / "test_metrics.json").exists() for run in controls):
        return ("## 4.8 结构与蒸馏的补充对照\n\n"
                "现有结构对照可描述融合模型的分出口差异，现有蒸馏对照可估计蒸馏在融合模型上的效果。"
                "GAP 加蒸馏按同样训练配方补充三个种子，完成后将与现有三组形成结构和蒸馏的四组对照。"
                "本稿的收益归因以已完成的三组比较为限。")
    groups = {name: {r["seed"]: next(m["calibrated"] for m in r["budget_metrics"] if m["budget"] == .65)
                     for r in runs[name]} for name in ("gap", "lg", "lg_kd")}
    records = []
    audits = []
    for path in controls:
        audits.append(audit_run(path, lock))
        result = json.loads((path / "test_metrics.json").read_text())
        policy = next(r for r in result["policies"] if r["criterion"] == "budget" and r["budget"] == .65
                      and not r["policy"]["consensus"])
        records.append(dict(seed=result["seed"], exit_accuracy=result["exit_accuracy"], **policy["test"]))
    groups["gap_kd"] = {r["seed"]: r for r in records}
    evidence = {"scope": "Post-review control on the previously reported CIFAR-100 split",
                "lock": str(lock), "audits": audits, "groups": groups}
    (RESULTS/"gap_kd_factorial.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    names = {"gap":"GAP", "gap_kd":"GAP＋蒸馏", "lg":"局部—全局", "lg_kd":"局部—全局＋蒸馏"}
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), layout="constrained")
    for index, (name, label) in enumerate(names.items()):
        values = list(groups[name].values())
        accuracy = [100*r["accuracy"] for r in values]
        cost = [100*r["cost_ratio"] for r in values]
        rows.append([label, f"{mean(accuracy):.2f} ± {stdev(accuracy):.2f}",
                     f"{mean(cost):.2f} ± {stdev(cost):.2f}", f"{sum(v<=65 for v in cost)}/3"])
        for ax, samples in zip(axes, (accuracy, cost)):
            ax.scatter(samples, [index]*3, s=24, color="#397eb5" if "kd" not in name else "#29846b")
            ax.scatter([mean(samples)], [index], s=25, color="#222222", marker="|")
    for ax, label in zip(axes, ("测试准确率 (%)", "测试平均计算比 (%)")):
        ax.set_yticks(range(4), list(names.values()))
        ax.invert_yaxis()
        ax.set_xlabel(label)
    save(figure, "factorial_gap_kd", {"source": str(RESULTS/"gap_kd_factorial.json"), "rows": rows}, FIGURES)
    effects = {name: [100*(groups[with_kd][seed]["accuracy"]-groups[plain][seed]["accuracy"]) for seed in (17,42,73)]
               for name, with_kd, plain in (("GAP","gap_kd","gap"), ("融合","lg_kd","lg"))}
    interaction = [a-b for a,b in zip(effects["融合"], effects["GAP"])]
    return "\n\n".join([
        "## 4.8 结构与蒸馏的补充对照",
        "GAP 加蒸馏沿用三个原种子、同样训练轮次和蒸馏超参数，训练后在验证集选阈值，并在独立的补充协议锁下测试。"
        "该对照设计晚于原测试结果公开，复用同一 CIFAR-100 测试集。"
        "种子 42 和 73 在轮次边界恢复检查点后继续训练，种子 17 连续训练。",
        "表 11 结构和蒸馏的四组对照，65% 目标预算",
        table(["模型","准确率/%","计算比/%","达标种子"], rows),
        fig(11, "三个种子在 65% 目标预算下的准确率与实际计算比。圆点为种子结果。", "factorial_gap_kd"),
        f"在 65% 目标下，蒸馏对 GAP 和融合的配对准确率差均值分别为 {mean(effects['GAP']):+.2f} 和 "
        f"{mean(effects['融合']):+.2f} 个百分点。两种结构的蒸馏收益之差为 {mean(interaction):+.2f} 个百分点，"
        "应结合各组测试实际成本和达标情况解读。此对照区分结构与蒸馏，门控、空间加权和局部卷积的独立贡献仍需更细消融。",
    ])


def main():
    routing = json.loads((RESULTS/"routing_analysis.json").read_text(encoding="utf-8"))
    runs = {model: [r for r in routing["runs"] if r["model"] == model] for model in LABELS}
    sections = [budget_tables(routing, runs), reliability_section(runs), timing_section(), factorial_section(runs)]
    paper = Path("paper/report_draft.md").read_text(encoding="utf-8")
    paper = paper.replace("## 5 结论", "\n\n".join(sections) + "\n\n## 5 结论", 1)
    note = ("补充分析在原始实验结果公开后开展，使用相同的测试预测缓存和已冻结权重。"
            "新的阈值只在验证策略选择子集拟合，所得测试比较属于同一数据集上的补充诊断。"
            "训练期间合并使用两个验证子集选择检查点，之后分别进行温度拟合和阈值选择。")
    paper = paper.replace("### 3.2 结构与决策对照", note + "\n\n### 3.2 结构与决策对照", 1)
    text = ("候选阈值由验证置信度的 21 个分位点与两个边界值构成。这里的最优策略定义在有限候选网格与验证样本上。"
            "路径平均成本可以等价写成各阶段增量成本乘以到达该阶段的频率再求和。"
            "提前退出比较的是继续计算可能带来的纠错收益和新增成本，"
            "置信度与相邻预测一致性是推理时序上可获得的经验判据。")
    paper = paper.replace("## 3 实验设置", text + "\n\n## 3 实验设置", 1)
    paper = paper.replace("这些结果说明蒸馏对浅层出口的帮助最突出", "无蒸馏融合第一出口为 64.40%，低于 GAP 的 70.63%。这些结果说明蒸馏对融合模型的浅层出口帮助最突出", 1)
    paper = paper.replace(
        "后续可在隔离负载下开展交错重复计时，并在其他图像分布和边缘硬件上验证策略的适配性。",
        "交错计时补充了完整骨干、固定浅层和真实动态路径的比较。"
        "测试成本对齐与分出口诊断进一步表明，组合收益随预算变化。"
        "其他图像分布与边缘设备上的适配性仍需独立验证。",
        1,
    )
    paper = paper.replace(
        "代码、结果与运行说明保存在 GitHub 私有仓库：https://github.com/vgoh64187-ui/adaptive-exit-cnn 。仓库 README 包含环境安装、数据准备、训练、推理与报告生成命令；模型权重副本和策略校验方式见仓库中的权重说明。",
        "原始实验的代码与运行说明见 GitHub 私有仓库 https://github.com/vgoh64187-ui/adaptive-exit-cnn 。"
        "本次补充实验的工作区入口为 revision_gap_kd.py，训练清单为 configs/revision_gap_kd.json。"
        "补充协议锁、各轮训练记录和测试预测缓存在 runs/revision/，四组逐种子结果和复算记录见 results/revision/gap_kd_factorial.json。"
        "复核并生成表图可运行 assemble_revision.py。",
        1,
    )
    Path("paper/report_revision.md").write_text(paper, encoding="utf-8")
    print("Revision manuscript and figures generated.")


if __name__ == "__main__":
    main()
