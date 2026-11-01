"""Assemble the editable report from evidence-linked sections and tables."""
from pathlib import Path
import json

root=Path(__file__).resolve().parent
def read(name): return (root/name).read_text(encoding='utf-8')

methods=read('methods_draft.md')
body,refs=methods.split('## 参考文献（待最终格式核验）')
body=body.replace('> 方法部分初稿。结构与公式对应当前实现；摘要和实验结论在正式结果生成后撰写。','')
title,body=body.split('\n',1)
body=body.replace('## 0 引言', '## 引言')
abstract=('摘要：针对固定深度卷积网络在计算资源受限场景下难以按样本分配计算的问题，'
          '本文以 CIFAR ResNet-18 为骨干，在中间阶段设计局部—全局门控分类出口，结合置信度、'
          '相邻出口一致性与自蒸馏，研究多出口图像分类的准确率与计算成本。'
          '在 CIFAR-100 上，以三个随机种子训练五组模型，并固定验证集确定的退出策略后进行测试。'
          '65% 目标预算下，融合加蒸馏模型的测试准确率为 75.32% ± 0.19%，实际平均计算比为 '
          '64.36% ± 0.69%；完整单出口基线准确率为 77.19% ± 0.05%。'
          '在本文测量条件下，GPU 动态推理的单张延迟未随计算量降低而下降，表明部署时需要联合考虑路由开销与硬件特性。\n\n'
          '关键词：卷积神经网络；提前退出；计算预算；特征融合；知识蒸馏\n')
body=body.replace('## 2 局部—全局协同提前退出网络',
    '![图 1 网络结构与局部—全局出口机理。中间出口基于已执行的残差阶段做出分类判断。](../figures/mechanism.png)\n\n## 2 局部—全局协同提前退出网络')
exp=read('experiments_draft.md').split('## 3 实验设置',1)[1].split('## 4 结果组织',1)[0]
analysis=read('results_analysis_draft.md').split('\n',1)[1]
analysis='## 4 实验结果与分析\n'+analysis
settings=read('generated/training_settings.md').replace('表 4 ', '表 1 ', 1)
architecture=read('generated/architecture_table.md').replace('表 1 ', '表 2 ', 1)
exit_table=read('generated/exit_accuracy_table.md').replace('表 2 ', '表 4 ', 1)
budget_table=read('generated/budget_accuracy_table.md')
analysis=analysis.replace('表 2、表 3', '表 3、表 4').replace('表 2 显示', '表 4 显示')
summary=json.loads((root.parent/'results/final/summary.json').read_text(encoding='utf-8'))
routing65=next(g for g in summary['groups'] if g['head']=='local_global' and g['kd']==.5 and g['criterion']=='budget' and g['budget']==.65 and not g['consensus'])
paired=json.loads((root.parent/'results/final/paired_effects.json').read_text(encoding='utf-8'))
fusion75=next(g['summary'] for g in paired['comparisons'] if g['contrast']=='local_features_vs_gap' and g['target_validation_budget']==.75)
paired_text=(f"在 75% 目标预算下，融合头相对 GAP 头的同种子准确率差为 {fusion75['accuracy_delta_pp_mean']:+.2f} ± {fusion75['accuracy_delta_pp_std']:.2f} 个百分点，"
             f"实际计算比差为 {fusion75['cost_ratio_delta_pp_mean']:+.2f} ± {fusion75['cost_ratio_delta_pp_std']:.2f} 个百分点（图 4）；"
             '这一对照同时呈现识别与计算差异，并不将目标预算相同等同于实际成本相同。')
routing_text=(f"在 65% 目标预算下，三个出口平均接收测试样本的 {100*routing65['exit1_fraction_mean']:.2f}%、"
              f"{100*routing65['exit2_fraction_mean']:.2f}% 和 {100*routing65['exit3_fraction_mean']:.2f}%。"
              f"相对同权重最终出口，提前退出使 {100*routing65['harmful_early_exit_fraction_mean']:.2f}% 的全部样本由正确变为错误，"
              f"同时使 {100*routing65['beneficial_early_exit_fraction_mean']:.2f}% 的全部样本由错误变为正确；"
              '两者差额对应动态准确率相对最终出口的变化。这些数值为三个种子的均值，分母均为各自全部测试样本。')
analysis=analysis.replace('## 4.2 出口特征与蒸馏',paired_text+'\n\n## 4.2 出口特征与蒸馏')
analysis=analysis.replace('相应的退出分布展示于图 5。','相应的退出分布展示于图 5。\n\n'+routing_text)
analysis=analysis.replace('## 4.1 按预算分配计算', budget_table+'\n\n## 4.1 按预算分配计算').replace('## 4.2 出口特征与蒸馏', exit_table+'\n\n## 4.2 出口特征与蒸馏')
fig2='![图 2 测试集准确率与实际计算比。横轴为测试样本的实际平均成本；点为三个独立种子的均值，误差线为样本标准差。](../figures/print/budget_accuracy.png)'
fig3='![图 3 不同出口结构的分类准确率对照。](../figures/print/exit_ablations.png)'
fig4='![图 4 相同种子配对的准确率与实际计算比差值，横线表示样本标准差。](../figures/print/paired_ablations.png)'
fig5='![图 5 融合蒸馏模型的退出分布与计算分配。](../figures/print/routing_local_global_kd_confidence.png)'
fig6='![图 6 固定验证样本上的实际延迟和吞吐。CPU 基线存在明显波动，见正文复测说明。](../figures/print/deployment_tradeoffs.png)'
fig7='![图 7 验证样本的动态执行实例。按出口和分类正误分层选取每层原始索引最小的样本，蓝色为实际执行阶段。](../figures/print/lg_kd_seed42_dynamic_examples.png)'
analysis=analysis.replace(exit_table, fig2+'\n\n'+exit_table)
analysis=analysis.replace('## 4.3 退出策略与实际执行', fig3+'\n\n'+fig4+'\n\n## 4.3 退出策略与实际执行')
analysis=analysis.replace('图中部署计时采用', fig5+'\n\n图中部署计时采用')
analysis=analysis+'\n'+fig6+'\n'
conclusion=('## 5 结论\n\n本文构建了基于残差网络的多出口图像分类系统，并通过近等参数出口、自蒸馏和退出策略对照，分析了识别能力与计算预算的关系。'
            '测试结果显示，融合加蒸馏在低预算点提供了可量化的计算—准确率权衡；浅层出口的判别能力得到提升。'
            '实际计时表明，动态路由的收益取决于设备与执行方式，MACs 不能直接替代延迟。后续可在隔离负载下开展交错重复计时，'
            '并在其他图像分布和边缘硬件上验证策略的适配性。\n')
application=('## 4.4 预算约束下的分类应用示例\n\n'
             '图 7 展示种子 42 融合加蒸馏模型在 75% 目标预算下采用置信度策略处理验证图像的路径。'
             '样本按出口位置和预测正误分层，每个非空层选取原始索引最小的图像；这组示例用于说明路径，不用于估计整体准确率。'
             '程序以真实动态前向记录各残差阶段访问次数，并逐例核对缓存预测。'
             '第一、第二和最终出口的累计成本分别为 287.388 M、421.943 M 和 556.212 M MACs；最终出口包含前两个辅助头的开销。\n')
out=(title+'\n\n'+abstract+'\n'+body+'\n## 3 实验设置\n'+exp+'\n'+settings+'\n\n'+architecture+'\n\n'+analysis+'\n'+application+'\n'+fig7+'\n'+conclusion+'\n## 参考文献\n'+refs)
(root/'report_draft.md').write_text(out,encoding='utf-8')
print('paper/report_draft.md assembled')
