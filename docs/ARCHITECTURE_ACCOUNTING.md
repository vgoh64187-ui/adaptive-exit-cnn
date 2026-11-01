# 出口结构的独立计算核对

对应代码：scripts/architecture_analysis.py，输出：results/architecture/profiles.json。

令出口输入为 C×H×W，类别数为 K。只统计可训练参数；MACs沿用卷积、线性层、注意力加权求和口径，偏置加法、BN、激活、softmax、普通池化和路由不计入MACs。实际延迟另测。

- GAP：参数 CK+K，MACs CK。
- MLP：隐藏维度 D，参数 (C+K+1)D+K，MACs (C+K)D。
- 局部—全局头：参数 2C²+(K+13)C+K+1；其中深度卷积9C、BN 2C、空间评分C+1、门控2C²+C、分类CK+K。
- 融合头MACs：11CHW+2C²+CK；空间部分由深度卷积9CHW、评分CHW及加权汇聚CHW构成。

脚本分别核验各辅助头的公式值、PyTorch参数计数、profile模块MACs统计。累计出口成本包含此前执行的辅助头；完整路径也包括两个辅助头。结构计算量不能直接解释为相同比例的真实加速。

公式核对用于确保论文复杂度解释与真实实现一致，不依赖训练精度。数值表由paper/build_tables.py读取同一架构输出生成，避免手工抄写。
