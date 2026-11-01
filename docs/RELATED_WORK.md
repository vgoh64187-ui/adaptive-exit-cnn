# 相关工作与创新边界（2026-10-31 初步检索）

| 原始来源 | 已有贡献 | 本项目如何使用 |
|---|---|---|
| He et al., Deep Residual Learning for Image Recognition, CVPR 2016, https://arxiv.org/abs/1512.03385 | 残差学习与经典骨干 | ResNet-18 基线；明确小图输入层适配 |
| Teerapittayanon et al., BranchyNet, ICPR 2016, https://arxiv.org/abs/1709.01686 | 辅助出口和提前终止推理 | GAP 多出口与置信度基线，不能把早退本身称为创新 |
| Huang et al., Multi-Scale Dense Networks for Resource Efficient Image Classification, ICLR 2018, https://arxiv.org/abs/1703.09844 | 多尺度表征与资源自适应分类 | 解释中间表征质量与预算评价的重要性 |
| Kaya et al., Shallow-Deep Networks: Understanding and Mitigating Network Overthinking, ICML 2019, https://proceedings.mlr.press/v97/kaya19a.html | 过度思考、内部分类器、置信度早退 | 评价有害早退与有益早退，避免默认最终出口永远更正确 |
| Phuong and Lampert, Distillation-Based Training for Multi-Exit Architectures, ICCV 2019, https://openaccess.thecvf.com/content_ICCV_2019/html/Phuong_Distillation-Based_Training_for_Multi-Exit_Architectures_ICCV_2019_paper.html | 多出口知识蒸馏 | 蒸馏是已有训练手段，需独立消融，不能列作独立原创 |
| Xu et al., LGViT: Dynamic Early Exiting for Accelerating Vision Transformer, ACM MM 2023, https://arxiv.org/abs/2308.00255 | 局部感知/全局聚合异构出口，冻结骨干的两阶段自蒸馏 | 是重要近邻工作；不能泛称“首次局部全局早退”。本项目为 CNN 同一出口内门控融合，需继续检查具体重合 |

候选课程贡献：以轻量深度卷积和空间加权汇聚补充 GAP，在 CNN 同一出口头内做通道门控融合；结合已计算出口的类别一致性进行因果决策；以近等参数 MLP、结构/策略/蒸馏分离消融验证。论文贡献以最终证据为准，不声称国际首创或 SOTA。

## 基线参考口径

作者公开实验仓库 https://github.com/weiaicunzai/pytorch-cifar100 列出 ResNet-18 约 11.2M 参数、200 轮、Top-1 error 24.39%（准确率 75.61%）。这是该仓库的实验值，不是 ResNet 原论文在本协议下的保证。
本项目保留 5,000 张验证图像，候选训练 160 轮+cosine，与该仓库不同；正式报告中不能直接把数字并表为相同训练口径。
阶段性质量目标：基线在验证集进入约 73–76% 的合理区间并呈收敛趋势；若不满足，先在验证集检查实现与训练配方，冻结后再解封测试。区间是项目验收参考，非实测结果和老师指定门槛。

## 尚需核验

完整原文中的方法细节、BibTeX、置信度校准原始来源、预算策略的一致性近邻方法。最终参考文献需逐条核对，避免只据摘要形成新颖性结论。

## 原文细节复核补充

已读取 LGViT 原文 HTML 第3.3节，https://arxiv.org/html/2308.00255v1 ：LPH 使用 1×1 扩展、随出口位置变化的深度卷积、1×1 投影和平均池化，并与 CLS token 相加；因此它的局部头内已经存在局部与全局信息融合。不能把“同一头内融合”本身作为与 LGViT 的唯一差异。本项目具体候选是 CNN 上的空间加权局部汇聚与通道门控插值，实验中用等容量 MLP 判断是否产生可验证收益。

置信度与一致性都有先例：

- Zhou et al., BERT Loses Patience: Fast and Robust Inference with Early Exit, NeurIPS 2020。原始来源 https://arxiv.org/abs/2006.04152 和 https://proceedings.neurips.cc/paper_files/paper/2020/hash/d4dd111a4fd973394238aca5c05bebe3-Abstract.html 。PABEE 在连续若干内部分类器预测不变时退出。
- Zhang et al., PCEE-BERT: Accelerating BERT Inference via Patient and Confident Early Exiting, Findings of NAACL 2022, pp.327–338。https://aclanthology.org/2022.findings-naacl.25/ 。该方法要求连续若干中间层预测足够自信；不要与2024年的同缩写 Performance Control Early Exiting 混淆。
- 本项目相邻类别一致+当前置信度是这些思路的场景化组合，不作为独立首创；策略对比应保留纯置信度、纯一致性及组合条件的职责区分。

校准来源：Guo, Pleiss, Sun and Weinberger, On Calibration of Modern Neural Networks, ICML 2017, PMLR70:1321–1330，https://proceedings.mlr.press/v70/guo17a.html 。温度缩放在独立校准子集拟合；本项目必须实测其 NLL/ECE 与路由影响，不能假定概率校准必然提高早退效果。

蒸馏原文已核验：Phuong and Lampert, ICCV2019, pp.1355–1364，官方PDF https://openaccess.thecvf.com/content_ICCV_2019/papers/Phuong_Distillation-Based_Training_for_Multi-Exit_Architectures_ICCV_2019_paper.pdf 。其目标为晚出口概率指导早出口，本文使用最终出口 detach 的 KL 项为可复现训练扩展，标清与原文完整策略的差别。
