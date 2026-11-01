"""Assemble the editable report from evidence-linked sections and tables."""
from pathlib import Path

root=Path(__file__).resolve().parent
def read(name): return (root/name).read_text(encoding='utf-8')

methods=read('methods_draft.md')
body,refs=methods.split('## 参考文献（待最终格式核验）')
body=body.replace('> 方法部分初稿。结构与公式对应当前实现；摘要和实验结论在正式结果生成后撰写。','')
title,body=body.split('\n',1)
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
tables='\n\n'.join([read('generated/architecture_table.md'),read('generated/exit_accuracy_table.md'),read('generated/budget_accuracy_table.md')])
settings=read('generated/training_settings.md')
figs=('![图 2 测试集准确率与实际计算比。横轴为测试样本的实际平均成本；三条独立种子的汇总见图中标注。](../figures/final/budget_accuracy.png)\n\n'
      '![图 3 不同出口结构的分类准确率对照。](../figures/final/exit_ablations.png)\n\n'
      '![图 4 相同种子配对的准确率与实际计算比差值，横线表示样本标准差。](../figures/final/paired_ablations.png)\n\n'
      '![图 5 融合蒸馏模型的退出分布与计算分配。](../figures/final/routing_local_global_kd_confidence.png)\n\n'
      '![图 6 固定验证样本上的实际延迟和吞吐。CPU 基线存在明显波动，见正文复测说明。](../figures/final/deployment_tradeoffs.png)\n')
conclusion=('## 5 结论\n\n本文构建了基于残差网络的多出口图像分类系统，并通过近等参数出口、自蒸馏和退出策略对照，分析了识别能力与计算预算的关系。'
            '测试结果显示，融合加蒸馏在低预算点提供了可量化的计算—准确率权衡；浅层出口的判别能力得到提升。'
            '实际计时表明，动态路由的收益取决于设备与执行方式，MACs 不能直接替代延迟。后续可在隔离负载下开展交错重复计时，'
            '并在其他图像分布和边缘硬件上验证策略的适配性。\n')
out=(title+'\n\n'+abstract+'\n'+body+'\n## 3 实验设置\n'+exp+'\n'+settings+'\n\n'+tables+'\n\n'+analysis+'\n'+figs+'\n'+conclusion+'\n## 参考文献\n'+refs)
(root/'report_draft.md').write_text(out,encoding='utf-8')
print('paper/report_draft.md assembled')
