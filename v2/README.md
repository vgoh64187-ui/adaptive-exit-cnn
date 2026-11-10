# V2 实验记录与运行入口

本目录存放相对原课程实验的探索性补充代码。V1 的 runs、docs/protocol_lock.json 和旧模型权重保留。V2 使用独立的 runs_v2、results/v2 与 figures/v2。

## 实验计划

- EXPERIMENT_PLAN.md 记录阶段、验证边界和数据集口径。
- screening_manifest.json 是启动种子 42 单因素初筛前登记的三个训练配置。
- routing_validation.py 将已有模型原策略选择集按类别分为内部拟合和留出子集。温度与权重已使用原验证数据，结果仅用于探索。
- history_diagnostics.py 比较原检查点选择和最终出口最佳验证轮次。
- features_validation.py 在验证集每类一张图上描述原融合头的门控、特征和空间权重，诊断结果不构成因果证明。

## 运行环境与命令

在项目根目录，使用已安装 PyTorch 2.5.1、torchvision 和 CUDA 12.1 的 Python 环境。下列命令用 python 表示该环境的解释器。

    python -m unittest v2.test_v2 -v
    python v2/routing_validation.py
    python v2/history_diagnostics.py
    python v2/features_validation.py
    python v2/summarise_phase1.py
    python v2/plot_routing_validation.py

训练需要使用空的输出目录。恢复训练时指定相同参数、同一代码版本及 --resume。V2 恢复继承原训练脚本的限制，进程重启后数据加载 worker 的增强随机状态可能与连续运行不同。

    python -m v2.train_v2 --head residual_local_global --epochs 160 --batch-size 128 --workers 2 --seed 42 --out runs_v2/residual_seed42
    python -m v2.train_v2 --head residual_local_global --epochs 160 --batch-size 128 --workers 2 --seed 42 --out runs_v2/residual_seed42 --resume

每轮写入 history.jsonl，并先写临时文件再替换 last.pt。完整结束后生成 completed.json。--limit 是冒烟测试，不用于正式指标。

完整训练后，v2.evaluate_v2 可在验证集拟合温度与阈值。通过 v2.freeze_v2 将完整训练、验证预测、阈值和评估代码的哈希写入新的只创建锁文件，再评估测试集。原 CIFAR-100 测试集的旧结果已经公开，因此测试结果属于探索性补充。

    python -m v2.evaluate_v2 runs_v2/residual_seed42 --split validation
    python -m v2.freeze_v2 --output runs_v2/v2_lock.json runs_v2/residual_seed42
    python -m v2.evaluate_v2 runs_v2/residual_seed42 --split test --lock runs_v2/v2_lock.json

此示例仅说明协议入口。正式对照应先按实验清单完成相应模型和种子，并一起冻结，再开展各组测试和结果汇总。评估脚本拒绝覆盖已有输出。
