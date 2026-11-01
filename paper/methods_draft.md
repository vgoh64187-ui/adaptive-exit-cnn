# 面向计算预算的局部—全局协同提前退出卷积神经网络

> 方法部分初稿。结构与公式对应当前实现；摘要和实验结论在正式结果生成后撰写。

## 0 引言

图像分类网络通常为所有输入执行相同深度的特征提取，而图像的判别难度并不一致。计算预算受限时，依据中间预测决定是否继续执行网络，可以将计算资源分配给需要进一步辨别的样本。提前退出网络通过在骨干中间阶段增加分类分支实现这种按需推理，其性能取决于中间表征的判别能力与退出决策的可靠性。

残差网络利用恒等映射与残差分支构建深层特征提取路径[1]，为研究不同深度的表征提供了清晰的结构基础。BranchyNet 在网络内部设置辅助分类器，通过提前终止推理减少计算[2]；MSDNet 将多尺度表征与多出口组织相结合，研究资源约束下的识别[3]。Shallow-Deep Networks 进一步指出，更深层预测并非总能修正浅层结果，部分样本会发生正确预测被后续层改变的现象[4]。因此，提前退出需要同时考虑正确分类、计算成本与错误早退之间的关系。

本文围绕残差 CNN 的中间出口构建局部—全局协同分类结构。出口在保留全局平均表征的同时，通过深度卷积和空间加权汇聚提取局部响应，并以通道门控形成融合特征。推理阶段依据当前置信度与已执行出口的类别一致性作出决策，已退出样本不再进入后续阶段。实验围绕同等计算预算下的识别质量展开，分别检验出口结构、因果决策与蒸馏训练的作用，并将理论计算量与实际执行延迟分开评价。

## 1 相关工作

提前退出的基础机制、中间表征设计与训练辅助分别构成三条相关研究线索。BranchyNet 和 Shallow-Deep Networks 提供多出口与置信度决策的基础[2,4]，MSDNet 体现了中间表征与预算适配的协同设计[3]。蒸馏训练能够利用不同深度分类器之间的预测关系改善多出口训练[5]。LGViT 面向视觉 Transformer 设计异构出口，其局部感知头通过卷积提取局部表征并与 CLS 特征相加[6]。本文以经典残差 CNN 为研究对象，通过空间加权汇聚与通道门控组织出口特征，并以近等参数容量对照分析具体结构的作用。

退出决策方面，PABEE 根据连续内部预测的一致性决定是否终止计算[8]；PCEE-BERT 将置信度与连续层判据结合[9]。本文采用当前置信度与相邻已执行出口类别一致的组合条件，通过固定权重的策略消融评价计算分配。置信度使用验证集温度缩放进行校准[7]，校准效果与退出收益分别检验。

## 2 局部—全局协同提前退出网络

### 2.1 残差骨干与多出口结构

骨干采用适配 CIFAR 图像尺寸的 ResNet-18。输入为三通道 32×32 图像，起始层使用步长为 1 的 3×3 卷积，并移除原始大图结构的初始最大池化。四个阶段均包含两个基本残差块，输出张量依次为 64×32×32、128×16×16、256×8×8 和 512×4×4。基本残差块计算为

\[
\boldsymbol{x}_{l+1}=\operatorname{ReLU}\big(\mathcal{F}(\boldsymbol{x}_l)+\mathcal{S}(\boldsymbol{x}_l)\big),
\]

其中，\(\mathcal{F}\) 由两层 3×3 卷积及批归一化构成，第一层卷积后使用 ReLU；\(\mathcal{S}\) 在尺寸一致时为恒等映射，在下采样或通道数变化时采用 1×1 卷积及批归一化。第二、第三阶段后设置辅助出口，第四阶段后保留全局平均池化与线性分类器，形成三个按深度排序的预测位置。

### 2.2 局部—全局门控出口

设某辅助出口输入特征为 \(F\in\mathbb{R}^{C\times H\times W}\)。全局分支采用平均汇聚：

\[
g_c=\frac{1}{HW}\sum_{u,v}F_{cuv}.
\]

局部分支先进行 3×3 深度卷积、批归一化和 ReLU，得到 \(L\)。对 \(L\) 使用 1×1 卷积生成空间评分 \(s_{uv}\)，沿空间维进行 softmax，并汇聚各位置响应：

\[
w_{uv}=\frac{\exp(s_{uv})}{\sum_{u',v'}\exp(s_{u'v'})},\qquad
\ell_c=\sum_{u,v}w_{uv}L_{cuv}.
\]

融合门控由两个分支的拼接特征生成：

\[
\boldsymbol{a}=\sigma\big(W_a[\boldsymbol{g};\boldsymbol{\ell}]+\boldsymbol{b}_a\big),\qquad
\boldsymbol{z}=\boldsymbol{g}+\boldsymbol{a}\odot(\boldsymbol{\ell}-\boldsymbol{g}).
\]

其中 \(\boldsymbol{a}\in(0,1)^C\)，\(\odot\) 表示逐元素乘法。融合特征经线性分类器得到类别 logits。该出口结构的实验评价采用 GAP-linear 与近等参数 MLP 对照，以分别检验局部信息引入和容量变化的作用。

### 2.3 多出口训练

令三个出口的 logits 为 \(q^{(j)}\)，真实标签为 \(y\)。基础训练目标为各出口交叉熵的平均值。蒸馏扩展以最终出口的停止梯度预测作为教师，仅向前两个出口传递软标签监督：

\[
\mathcal{L}=\frac{1}{3}\sum_{j=1}^{3}\operatorname{CE}(q^{(j)},y)
+\lambda\frac{T^2}{2}\sum_{j=1}^{2}\operatorname{KL}\left(
\operatorname{softmax}(\operatorname{sg}(q^{(3)})/T)
\;\|\;\operatorname{softmax}(q^{(j)}/T)\right).
\]

\(T\) 为蒸馏温度，\(\operatorname{sg}\) 为停止梯度运算。不使用蒸馏的结构对照设置 \(\lambda=0\)。蒸馏属于独立训练辅助，结构贡献由无蒸馏对照判断。

### 2.4 因果一致性决策与动态执行

第 \(j\) 个出口使用验证校准集拟合的温度 \(\tau_j\)，得到概率 \(p^{(j)}=\operatorname{softmax}(q^{(j)}/\tau_j)\)。定义置信度 \(c_j=\max_kp_k^{(j)}\) 和预测类别 \(\hat y_j=\arg\max_kp_k^{(j)}\)。置信度基线在 \(c_j\geq\theta_j\) 时退出；一致性策略在第二出口增加 \(\hat y_2=\hat y_1\) 的条件，第一出口仍使用置信度条件。未满足条件的样本继续进入下一阶段，最终出口直接返回预测。

一致性条件只使用当前及此前已执行出口，不需要深层预测。实现中维护活动样本索引，在每次决策后压缩特征批次，后续残差阶段只处理尚未退出的样本。计算量统计计入路径上已经执行的全部出口头；实际延迟同时包含活动批次管理与分支判断开销。

### 2.5 预算约束下的策略选择

对第 \(i\) 个样本记退出位置为 \(e_i\)，执行至该出口的累计成本为 \(M_{e_i}\)。平均相对计算成本定义为

\[
\rho=\frac{1}{N M_{\mathrm{base}}}\sum_{i=1}^{N}M_{e_i},
\]

其中 \(M_{\mathrm{base}}\) 为单出口骨干完整前向的 MACs。验证策略选择集上，在预设 \(\rho\leq B\) 的候选策略中选择准确率最高者，准确率相同时选择计算成本较低者。测试阶段固定模型、温度和阈值，报告测试集实际准确率与实际平均成本。该约束对应样本分布上的平均预算。

## 参考文献（待最终格式核验）

[1] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR, 2016.

[2] Teerapittayanon S, McDanel B, Kung H T. BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks. ICPR, 2016.

[3] Huang G, Chen D, Li T, Wu F, van der Maaten L, Weinberger K Q. Multi-Scale Dense Networks for Resource Efficient Image Classification. ICLR, 2018.

[4] Kaya Y, Hong S, Dumitras T. Shallow-Deep Networks: Understanding and Mitigating Network Overthinking. ICML, PMLR 97, 2019: 3301–3310.

[5] Phuong M, Lampert C H. Distillation-Based Training for Multi-Exit Architectures. ICCV, 2019: 1355–1364.

[6] Xu G, Hao J, Shen L, et al. LGViT: Dynamic Early Exiting for Accelerating Vision Transformer. ACM MM, 2023.

[7] Guo C, Pleiss G, Sun Y, Weinberger K Q. On Calibration of Modern Neural Networks. ICML, PMLR 70, 2017: 1321–1330.

[8] Zhou W, Xu C, Ge T, et al. BERT Loses Patience: Fast and Robust Inference with Early Exit. NeurIPS, 2020.

[9] Zhang Z, Zhu W, Zhang J, et al. PCEE-BERT: Accelerating BERT Inference via Patient and Confident Early Exiting. Findings of NAACL, 2022: 327–338.
