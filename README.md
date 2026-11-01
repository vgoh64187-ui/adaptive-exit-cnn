# Adaptive Exit CNN

面向计算预算的局部—全局协同提前退出 CNN 课程研究项目。

五类模型、三个随机种子的 15 组训练和冻结测试已完成；测试汇总见 `results/final/summary.json`，实验结果与计算预算详见 `paper/generated/`。论文仍在排版与核验阶段。完整计划见 PROJECT_PLAN.md。
经典骨干采用 CIFAR 版 ResNet-18；现有教材未明确课程实验三任务，相关差异仍待课程材料核对。

## 结果真实性

仅运行成功的结果可进入论文。数据划分、模型训练、阈值选择和最终测试分离；主结果多种子重复。
动态推理必须真正跳过后续层，离线预测缓存仅用于策略评估，不作为实测推理延迟。

## 环境

Python 3.12，支持 CUDA 12.1 的 NVIDIA 显卡驱动。安装示例：

```powershell
uv venv --python 3.12 --seed .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

本机初始实验复用了相邻课程目录已安装的 `../六个神经网络实验/.runenv/Scripts/python.exe`；每个 run 保存真实环境信息与 pip freeze。该路径不是复现要求，独立环境安装后的命令如下。

此依赖文件同时使用 PyPI 和 PyTorch 官方 CUDA 下载源，安装步骤使用 pip 的多索引解析。`uv pip install -r requirements.txt` 的默认首索引策略会把部分普通依赖限定到 PyTorch 源，因此不作为本项目的安装命令。也可使用 `python -m venv .venv` 创建自带 pip 的环境。

Windows / Python 3.12 的独立环境已验证安装、22项行为测试、一轮CPU小规模训练及论文PDF生成，记录见 `docs/REPRODUCTION_CHECK.json`。需要锁定全部间接依赖时，使用 `python -m pip install -r requirements-lock.win-cuda.txt`；该锁定清单包含报告生成依赖。小规模训练仅验证流程，不作为性能结果。

2026-11-01 在原实验环境再次运行测试套件，28 项通过。模型检查点和数据集未纳入 Git：克隆仓库后需按下方命令重新下载数据并训练，或者使用与 `docs/protocol_lock.json` 中 SHA-256 一致的已有检查点。现有冻结协议绑定原始运行文件，重新训练的权重即使配置相同也应重新生成协议锁和结果，不能直接冒用原锁。

## 已完成的测试结果

CIFAR-100 官方测试集上，单出口 ResNet-18 的三种子准确率为 77.19% ± 0.05%；验证目标预算为 65% 时，融合加蒸馏模型以 64.36% ± 0.69% 的实际平均计算比取得 75.32% ± 0.19% 的准确率。误差为三个独立训练种子的样本标准差，详细结果、策略和成本口径见 `paper/generated/budget_accuracy_table.md`。设备实测和一次 CPU 基线复测显示计时波动，不能直接由 MACs 换算延迟收益，见 `paper/results_analysis_draft.md`。

## 运行

```powershell
# 下载官方 CIFAR-100，支持分段续传，最终校验官方 MD5
.venv/Scripts/python.exe scripts/download_cifar.py

# 网络行为检查（不需要数据）
.venv/Scripts/python.exe -m unittest discover -s tests -v

# 冒烟运行，仅检查流程，不能用于最终论文性能表
.venv/Scripts/python.exe train.py --head local_global --epochs 2 --limit 1024 --out runs/smoke

# 基线；完整训练配方见 docs/EXPERIMENT_PROTOCOL.md
.venv/Scripts/python.exe train.py --head baseline --epochs 160 --seed 42 --out runs/baseline_seed42

# 多出口结构候选
.venv/Scripts/python.exe train.py --head gap --epochs 160 --seed 42 --out runs/gap_seed42
.venv/Scripts/python.exe train.py --head mlp --epochs 160 --seed 42 --out runs/mlp_seed42
.venv/Scripts/python.exe train.py --head local_global --epochs 160 --seed 42 --out runs/lg_seed42
.venv/Scripts/python.exe train.py --head local_global --kd 0.5 --epochs 160 --seed 42 --out runs/lg_kd_seed42

# 用验证集校准概率并选择预算策略
.venv/Scripts/python.exe evaluate.py runs/lg_seed42 --split validation

# 方法与协议冻结之后才能进行最终测试，锁文件须真实记录冻结内容
.venv/Scripts/python.exe evaluate.py runs/lg_seed42 --split test --protocol-lock docs/protocol_lock.json

# 当前机理图草稿
.venv/Scripts/python.exe scripts/draw_mechanism.py

# 串行执行 seed42 结构对照，自动接续当前活跃训练，不重复启动
.venv/Scripts/python.exe run_queue.py configs/seed42_queue.json

# 三种子完整实验队列（已有同名完整结果会被复用；同一时间只运行一个队列）
.venv/Scripts/python.exe run_queue.py configs/main_queue.json

# 使用验证集选择的预算策略进行实际动态分类
.venv/Scripts/python.exe demo.py runs/lg_seed42 --sample-index 3 --budget 0.75
.venv/Scripts/python.exe demo.py runs/lg_seed42 --image example.jpg --budget 0.85 --consensus

# 本地交互应用，启动后在浏览器打开输出的 127.0.0.1 地址
.venv/Scripts/python.exe serve_demo.py runs/lg_seed42 --device cpu

# 完成训练、停止其他训练负载后，测量 CPU/GPU 延迟
.venv/Scripts/python.exe benchmark.py runs/lg_seed42 --device cpu --batch-size 1 --output results/latency/lg_cpu_b1.json
.venv/Scripts/python.exe benchmark.py runs/lg_seed42 --device cuda --batch-size 1 --output results/latency/lg_cuda_b1.json
# 正式冻结测试完成后，按预先固定的36项矩阵执行：
.venv/Scripts/python.exe run_benchmarks.py --plan-only
.venv/Scripts/python.exe run_benchmarks.py
# 全部正式计时完成后，生成成本、延迟与吞吐对照图：
.venv/Scripts/python.exe plot_deployment.py
```

中断后使用原命令加 `--resume` 从完整轮次检查点继续。代码哈希和关键配置必须一致；恢复后的 worker 数据增强随机状态可能与不间断运行不同，运行记录中需披露。正式主要实验优先使用不间断运行。

队列状态保存在 `runs/queue_state.json`。创建 `runs/QUEUE_PAUSE` 可以禁止后续任务启动，但不会中断当前训练。若需立即暂停，应先停止队列，再停止正在运行的训练，核验最近完整轮次检查点并记录恢复位置。恢复队列之前，只有在用户明确恢复后才移除该暂停文件。队列检测到外部训练异常终止会停下，不根据观测超时自动重启。

演示支持本地图像或固定验证集样本；输出类别、置信度、真实出口、累计 MACs。预算约束针对验证集平均计算量，单张图像的成本可高于预算比。单次演示延迟包含首次调用开销，不能替代预热后的 benchmark。
benchmark 同时记录阶段访问次数和缓存路由的一致性；精度验证不在计时段内。默认采用固定验证样本子集，分类精度主表仍使用完整官方测试集。

## 冻结、图表与结果汇总

`configs/main_queue.json` 预先列出 5 组模型 × 3 个种子的 15 个运行。全部训练和验证选择完成后运行下列命令；未完成时冻结程序会报出缺失项，不会创建空锁文件。

```powershell
.venv/Scripts/python.exe freeze_protocol.py configs/main_queue.json
# 对清单中每个 run 分别执行最终测试，例：
.venv/Scripts/python.exe evaluate.py runs/lg_kd_seed42 --split test --protocol-lock docs/protocol_lock.json
# 从逐张预测缓存复算并核验单个运行：
.venv/Scripts/python.exe audit_results.py runs/lg_kd_seed42
# 所有 run 的最终测试齐全后汇总：
.venv/Scripts/python.exe summarise.py configs/main_queue.json
.venv/Scripts/python.exe paired_effects.py configs/main_queue.json
.venv/Scripts/python.exe plot_ablation.py --budget .75
.venv/Scripts/python.exe make_figures.py results
.venv/Scripts/python.exe make_figures.py routing --consensus

# 验证集上的可靠性诊断与数据展示：
.venv/Scripts/python.exe diagnostics.py runs/lg_kd_seed42 --split validation
.venv/Scripts/python.exe make_figures.py diagnostics --runs runs/lg_kd_seed42
.venv/Scripts/python.exe make_figures.py data
# 验证集案例：各出口按正确/错误分组，固定选取最小索引，实际核验跳算
.venv/Scripts/python.exe make_examples.py runs/lg_kd_seed42 --budget .75 --consensus
```

完整训练队列确认 `queue_complete` 后，也可用 `.venv/Scripts/python.exe run_final_tests.py` 按清单自动冻结、逐个测试、逐样本核验和汇总。该命令在队列未完成时拒绝访问测试集；中断后可重跑，已有测试结果必须通过缓存与冻结协议核验才会复用。

锁文件绑定已完成训练的权重、配置、验证策略、日志、缓存元数据和评估代码哈希。冻结后不重新选择阈值。最终汇总拒绝权重或策略错配、冒烟结果、测试样本不足以及缺少三种子重复的比较组。图表同时输出 PNG、SVG 和 `.data.json` 数值来源记录。

汇总会自动调用逐样本核验：检查测试缓存哈希、完整样本索引、冻结策略与模型结构，重新统计准确样本数、退出分布、平均MACs和有害/有益早退比例。汇总表中的指标必须与缓存复算一致；核验程序及汇总程序也纳入测试前的代码冻结。

可靠性图的 ECE/NLL/Brier 使用未参与温度拟合的策略选择子集；风险—覆盖率图逐出口展示，不将单出口排序直接等同于多出口实际路由。固定出口对照在共享训练权重下运行，并计入此前已执行的辅助头成本。

交互应用支持固定验证样本与本地图像上传，显示预测概率、实际退出位置、累计 MACs 和阶段跳算路径，并导出带权重标识的 JSON 记录。所有推理在本机执行。`--development` 可用于冒烟权重检查界面，并显示用途标记；正式展示使用完整训练权重。界面的单次延迟不进入部署性能表。

## 文件组织

- `src/models.py`：手写残差骨干、四类出口、因果路由与 MACs 统计。
- `src/data.py`：分层固定划分与数据管线。
- `train.py`：从零训练、验证选模、检查点恢复。
- `evaluate.py`：预测缓存、温度校准、预算策略及最终评价。
- `run_queue.py`：串行对照实验与验证队列，可接续已有进程。
- `benchmark.py`：真实动态延迟、吞吐与跳算审计。
- `demo.py`：本地预算可调图像分类应用。
- `serve_demo.py`、`app/index.html`：交互界面与本地 HTTP 服务，无额外 Web 框架依赖。
- `summarise.py`：要求完整结果、校验权重哈希，按独立训练种子汇总。
- `paired_effects.py`：结构、容量、蒸馏及策略的同种子配对差值，保留测试实际计算差异。
- `tests/`：真实跳层、混合批次一致性与梯度检查。
- `docs/`：实验协议、版式记录、相关工作与核验记录。
- `paper/`：论文源稿，结果生成前不写性能结论。
- `figures/`：可编辑矢量图与预览。
- `runs/`：本地产生，不直接提交大权重和逐样本缓存。

每次运行保存 `config.json`、`environment.json`、`profile.json`、`history.jsonl`、`best.pt`、`last.pt` 和 `completed.json`。最终复现包将提供经过审计的指标、图表、运行索引及权重获取说明。

## 参考

- ResNet: https://arxiv.org/abs/1512.03385
- BranchyNet: https://arxiv.org/abs/1709.01686
- MSDNet: https://arxiv.org/abs/1703.09844
- Shallow-Deep Networks: https://proceedings.mlr.press/v97/kaya19a.html
- CIFAR: https://www.cs.toronto.edu/~kriz/cifar.html
