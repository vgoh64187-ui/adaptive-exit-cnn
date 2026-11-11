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

## 单因素筛选和恢复

screening_manifest.json 固定了种子 42 的三组对照：残差融合无蒸馏、从第一轮开始的自蒸馏、前 60 轮关闭并在之后 20 轮线性增至 0.5 的自蒸馏。先在验证策略选择集上按 55%、65%、75% 三个目标的平均准确率选择蒸馏时序，再为无蒸馏和选定时序分别补种子 17、73。该筛选使用已参与检查点选取的验证数据。

    python -m v2.train_v2 --head residual_local_global --epochs 160 --batch-size 128 --workers 2 --seed 42 --kd 0.5 --kd-start 0 --kd-ramp 0 --out runs_v2/residual_kd_seed42
    python -m v2.train_v2 --head residual_local_global --epochs 160 --batch-size 128 --workers 2 --seed 42 --kd 0.5 --kd-start 60 --kd-ramp 20 --out runs_v2/residual_delayed_kd_seed42

上述命令适用于对应输出目录没有已有配置的情况。恢复已有实验前，应确认该目录没有正在运行的训练进程、last.pt 可读取且最后一条训练日志与检查点轮次一致；随后使用同一参数并增加 --resume。恢复事件与工作进程随机增强序列的限制应记录在 results/v2/run_resume_events.json。

v2.freeze_v2 在测试前检验完整轮次、训练源文件哈希、验证缓存及预先规定的筛选结果。测试完成后，v2.audit_v2 从逐图 logits 重新计算各出口准确率和所有策略，v2.summarise_v2 在全量审核通过后生成配对汇总。v2.assemble_paper_v2 根据汇总与设备计时结果生成论文源；v2.build_word_omml 使用 python-docx、lxml、Pillow、latex2mathml 3.81.1 和本机 Microsoft Office 的 MML2OMML.XSL 生成正文、表格与可编辑公式。其他环境可使用 paper/render_math.py 和 v2/build_word.py 生成公式为图片的备用版本。论文、图表和仓库需在实验全部结束后另行核验和同步。

## 完整 V2 流程与数据依赖

以下命令均在项目根目录运行。完整复现需要原 CIFAR-100 数据与固定划分、V1 各组的权重和测试预测缓存，以及 V2 完整训练产物；这些大文件不随普通 Git 提交。若只有代码和汇总图，无法仅靠汇总图复算逐样本结果，需要按根目录说明重新训练或取得相应原始产物。仓库内的结果索引和散列只能帮助核对，不能替代权重及预测缓存。

1. 按前述命令训练三组种子 42，并分别执行 `python -m v2.evaluate_v2 runs_v2/<运行名> --split validation`。已有完整输出不得覆盖。随后执行 `python -m v2.select_screening`；该命令按清单中预先登记的验证集规则选择蒸馏时序。
2. 保留选中的蒸馏配置，分别用种子 17、73 训练无蒸馏残差组和选中蒸馏组，并生成四组验证预测。`python v2/continue_screening.py` 可在三组种子 42 完成后执行这一串行流程，随后冻结七次运行并逐一开展探索性测试；恢复已有实验前仍须确认没有其他同目录训练进程。不要在冻结前手动执行 V2 测试。
3. 运行 `python -m v2.audit_after_screening`，在所有测试结果齐全后复算逐图结果。运行 `python v2/benchmark_after_screening.py` 在审核完成后执行计时；该计时脚本需要 CUDA GPU、CPU 及原单出口基线权重，测量应避开其他训练负载。
4. 运行 `python v2/summarise_after_screening.py`，等待审核和计时完成后生成三种子汇总、图表和论文源。最终测试为此前已经使用过的 CIFAR-100 测试集，属于探索性结果；验证预算目标和测试实际成本需同时报告。重新运行前应保存或移走旧输出，脚本对关键结果使用不覆盖策略。
5. 在安装上述 Word 依赖的 Windows 环境中，使用 `python v2/build_word_omml.py paper/report_v2.md <输出.docx>`，再运行 `python v2/verify_paper.py <输出.docx>` 核对表格、图引用和公式数量。运行需要相应论文源、汇总 JSON 和图像文件；视觉排版仍需逐页检查。示例命令不能绕过训练、冻结、测试与审核，也不能将模拟试排数据作为正式结果。

`screen_queue.py`、`continue_screening.py` 和各 `*_after_screening.py` 是等待并推进流程的辅助程序，使用前应先检查是否已有同一程序或同一训练目录的进程，避免重复运行。`v2/word_after_screening.py` 使用本机 Codex 文档工具与临时依赖目录，主要服务于当前工作环境；跨机器复现应采用上面的 Word 生成器命令，并自行安装依赖。

完成全部实验、逐图审核和 Word 核验后，可运行 `python -m v2.package_evidence` 生成 `release/v2-evidence/`。脚本先验证冻结协议、测试缓存、论文源与 Word 校验散列，再打包轻量配置、固定数据划分索引、训练日志、策略和结果，并生成文件散列清单。该包不包括原始图像、模型权重和逐图预测缓存，不能单独替代原始文件做逐图复算；目标路径或暂存路径已存在时会拒绝覆盖。

已有原版自包含导读时，完成 Word 校验后可运行 `python -m v2.build_explainer`，保留原导读并在独立 HTML 中补充 V2 实验说明及其图表。生成前会检查论文、汇总和图像散列。HTML 将 V2 图片嵌入文件，打开时不需要网络；导读的具体数字仅在真实审核结果存在时生成。
