# V2 当前工作状态

目标：在保留原实验的条件下，完成验证路由诊断、结构和蒸馏对照，按冻结结果更新论文、图表、运行说明及 GitHub。

已完成：
- 原实验 15 个主模型和 3 个 GAP 蒸馏模型逐一通过权重、预测缓存、协议哈希及 10000 张测试样本的重算审核，汇总位于 results/v2/v1_full_audit.json。
- 原 selection 验证集按类分为 1200 张拟合和 1300 张留出，完成三种子 GAP、融合+蒸馏的阈值密度、margin 和安全余量实验。结果在 results/v2/routing_validation.json、phase1_validation_report.md 和 figures/v2/routing_validation_exploratory.png。该留出不是独立于模型选模的数据。
- 用按类各一张的 100 张验证图进行特征范数等描述诊断。结果在 results/v2/features_validation.json 和 figures/v2/feature_norms_exploratory.png。
- 初步实现 V2 零初始化残差融合头，冒烟训练及动态路由一致性、梯度和阈值搜索测试通过。

正在运行：
- runs_v2/residual_seed42，完整 160 轮，训练源文件为 v2/train_v2.py 和 v2/src/*.py，运行中不要修改这些文件。
- v2/screen_queue.py 等待该组完成，再顺序运行 residual_kd_seed42 和 residual_delayed_kd_seed42。
- v2/continue_screening.py 等待三组种子 42 完成，依次计算验证集结果，按 screening_manifest.json 的预先登记规则选蒸馏版本，再给无蒸馏和选定蒸馏方案分别训练种子 17、73。完成后进行验证、冻结和探索性测试。
- v2/benchmark_after_screening.py 等待全部锁定测试完成，计时选定的蒸馏模型。
- v2/monitor_checkpoint.py 不干预训练，只从已保存最佳检查点观察融合系数。

重要后续任务：
- 监测训练与所有队列，任一进程失败需阅读各运行目录的 queue.log 或 pipeline.log，确认原因再安全恢复。不要同时启动同一个输出目录的两份训练进程。
- 筛选依据是已有验证集的策略准确率，在官方测试结果已公开后开展，不能称为独立确认。需要报告实际成本与预算违反情况。
- 评估三种子结果及实际延迟，绘制最终图表，并将结论和边界写入论文源、重建 Word，保证表图引用和数字一致。
- 将用户此前授权的代码、配置、日志汇总、图表、论文源码、说明同步到私有 GitHub。检查大文件、私有资料和文件哈希；模型权重及 CIFAR 数据不直接纳入普通 Git。
- 任务目标完成前保持 active，不标记 complete。

## 暂停与恢复记录

- 用户恢复后检查到原训练及队列均已停止；residual_seed42 检查点和日志均为第 78 轮，训练源码哈希匹配。已从第 79 轮恢复，详情见 results/v2/run_resume_events.json。根据原训练记录说明，重启后数据加载工作进程的增强随机状态可能与连续运行不同。
- 已恢复原单种子训练、screen_queue、continue_screening、audit_after_screening、benchmark_after_screening 和融合系数采样进程。
- 已新增汇总脚本 v2/summarise_v2.py 与 v2/plot_final_comparison.py，由 v2/summarise_after_screening.py 等待审核和计时结束后执行。
- V1 审稿补充及当前静态 V2 阶段代码和诊断结果已上传至 GitHub 私有仓库 main，提交 6120feade6e3ee12a6299ca964540d8f0692025f。60 个新增文件 Git blob 与本地一致，原有 234 个 blob 未变化。README 入口后续更新提交 b3f48849fa9763b3f533f7c1650436227e98951a。训练后还需同步新结果、后续脚本及最终论文，不能把阶段性提交视为完成。

## 当前阶段补充

- residual_seed42 已完成 160 轮，恢复发生于第 78 轮后；已生成 5000 张验证预测，并逐一重算全部 21 个验证策略。results/v2/first_structure_validation.md 汇总其与 V1 同种子验证对照。此处结果用于探索，不作为独立测试结论。
- 新旧结构计数与选中检查点参数记录在 results/v2/structure_profile_seed42.json；仅按现行统计口径，两者累计 MACs 相同，该统计未纳入逐元素门控、残差等开销。
- 第二组 residual_kd_seed42 正在 screen_queue 中训练。其后是延迟蒸馏，接着由 continue_screening 完成验证选择及另外两个种子的训练、冻结与测试。冻结和审计脚本通过第一组完整训练产物的预检。
- 用户可见的原 Word 逻辑修订版已经同步回 paper/report_logic_revision.md。基于该源重新生成的试排版与原 Word 的 111 段文字、11 张表以及图和公式数量一致。
- 机理图 figures/v2/v2_head_mechanism.* 和可复现 Word 生成器 v2/build_word.py 已准备。V2 测试结果尚未产生，暂不修改正式结论。
- 上述静态源与机理图已同步 GitHub main，提交 fc4a874be6ee6d420a944626c819df4ba4867cd5。为保证冻结协议使用的原始源码字节，.gitattributes 的 V2 Python 文件规则调整已上传，提交 bb620541cf552216955e3196d5315fe4b8319733。

## 最新阶段：蒸馏初筛与论文流水线

- residual_kd_seed42 已完成 160 轮并评估 5000 张验证图；residual_delayed_kd_seed42 正在训练。对即时蒸馏组与无蒸馏组，已从验证缓存重新计算各 21 条策略的准确率、成本与出口人数，详见 results/v2/immediate_kd_validation_audit.json。这些是同一验证选择集上的单种子探索，不能作为独立测试结论。
- 审核、计时、汇总、绘图和 Word 等待程序已运行。最终多种子实验完成之前不要生成正式结论。
- 新版 Word 生成器 v2/build_word_omml.py 使用可编辑 Office 公式。完整 17 图、13 表的隔离模拟试排已审核 31 处公式，模拟文件只在 tmp/v2_assembly_fixture4 内，绝不可作为真实实验结果。正式输出后仍需逐页检查及复核最终结果。
- Word 等待程序于本阶段仅为切换可编辑公式生成器而重启；训练进程未中断。其依赖 latex2mathml 3.81.1 在 tmp/word_math_deps，和本机 Microsoft Office 的 MML2OMML.XSL。

## 用户主动暂停：2026-11-10

- 用户要求暂停。已停止后续训练、审计、计时、论文汇总及 Word 等待队列，并确认没有 V2 训练或上述队列的 Python 进程。
- residual_delayed_kd_seed42 的 last.pt 可正常读取，检查点 epoch 为 34，与 history.jsonl 最后一条 epoch 34 一致，未发现 .tmp 临时文件。last.pt SHA-256：3e3171d9bc55b1dcd2fc632e624f5f2f564b384733a5679f97a02a6485934940；训练尚未完成。恢复时从第 35 轮继续，并在 results/v2/run_resume_events.json 登记本次中断与 worker 增强随机序列限制。
- 已完成 residual_seed42 和 residual_kd_seed42 的 160 轮训练及验证缓存；两组各 21 条策略已复核，结果见 results/v2/immediate_kd_validation_audit.json。现有 V1 及 V2 结果均保留。
- 恢复前先复核检查点和训练源哈希、确认没有同目录训练进程，再依次启动 screen_queue.py、continue_screening.py、audit_after_screening.py、benchmark_after_screening.py、summarise_after_screening.py 与 word_after_screening.py；不要并行重启相同实验。最后生成的真实论文须重新检查。
- 工作树仍有尚未同步的代码与阶段性文件，GitHub main 最近已核验提交 dc86500f562876671149a77b8054ed7c31f72079；暂停不代表最终结果或论文已上传。

## 恢复：2026-11-11

- 按用户重新激活的目标恢复实验：停止状态下检查 last.pt 与日志同为第 34 轮，检查点 SHA-256 与暂停记录一致；通过 v2.train_v2.source_hash 核验训练源码与配置一致。
- 已登记 residual_delayed_kd_seed42 的恢复事件，并从第 35 轮继续。screen_queue、continue_screening、audit_after_screening、benchmark_after_screening、summarise_after_screening 和 word_after_screening 已重新启动。审计程序需使用 python -m v2.audit_after_screening，以正确加载模块。
- 论文生成器现在读取 run_resume_events.json，在补充实验段落列出所有已记录的恢复事件。已在 tmp 的模拟装配中验证两次事件均被引用；最终结果和正式论文尚未生成。
- 本次会话中的 GitHub connector 当前调用返回“not available to the model”，需要之后重新检查；既有远程提交不受影响，本地实验继续。

## 2026-11-11 继续执行与复现补充

- 已确认延迟蒸馏在第 60 轮的系数为 0，第 61 轮为 0.025，第 80 轮为 0.5，符合预登记渐进设置。训练和下游队列仍在运行。
- v2/README.md 补充完整筛选、训练、冻结、审核、计时、图表与 Word 命令，并说明 CIFAR 数据、V1 权重和预测缓存依赖。
- 尚未开始正式冻结；v2/freeze_v2.py 已添加恢复记录的 SHA-256 绑定。v2/assemble_paper_v2.py 核验绑定后引用恢复记录；针对完整模拟输入和篡改记录的拒绝检查已通过，模拟结果仅留在 tmp。
- 正式冻结前用已完成的 residual_seed42 做真实预检，发现 Windows 路径分隔符会使新加的恢复记录键与论文校验键不一致。已在 freeze_v2.py 中将恢复记录和筛选记录路径规范化为斜杠，重新预检并逐项校验全部 21 个源文件散列。preflight_resume_record_lock3.json 仅是单运行预检，不能用于正式测试；旧预检记录保留。
- 新增 v2/audit_v1_paper_tables.py，对原论文表 1–4、生成表、数据源散列和 119 项数值来源进行核对，结果为 results/v2/v1_paper_table_audit.json；范围不包含正文和其余图表。最终 Word 校验已要求前四表与该已审计原稿一致，包含 17 图及 31 个可编辑公式的模拟稿已通过该新增检查。
- 新增 v2/audit_v1_revision_tables.py：依据路由、交错计时和因子对照记录重新计算原论文表 5–11 的 62 行，均与论文一致，结果见 results/v2/v1_revision_table_audit.json。最终 Word 校验同时检查原论文 11 张表与其审计记录，并核对修订数据源散列；隔离模拟稿已通过。该审核不代表正文与图表逐项核验已经完成。
- 原图 9 的最终出口条件错误率仅有 1 个有效种子；原图以零长度误差线表示该项，容易被误读。已保留原图，并新增 figures/v2/v1_routing_reliability_valid_seeds.*：对 n=1 标注有效数量，不画样本标准差。覆盖率与条件错误率的均值、有效种子数逐项与原图数据及表 8 一致；V2 论文源会使用修订图，隔离装配已验证替换路径。
- 三组种子 42 初筛全部完成；延迟蒸馏的验证缓存、权重和 21 条策略已逐一重新计算，记录于 results/v2/delayed_kd_validation_audit.json。按预登记的 55%、65%、75% 验证准确率均值选择即时蒸馏（75.00%；延迟 74.92%），不使用 V2 测试结果。continue_screening 已开始 residual_seed17，并将补齐对应蒸馏组与种子 73。
- 图 2–5 与图 8–11 的数据来源及新版图 9 的有效种子计数通过检查，记录于 results/v2/v1_figure_data_audit.json。此检查不自动核验图像像素；最终页面仍须视觉检查。
- 即时蒸馏选定后，四组扩展训练按单 GPU 顺序推进。新增 v2/package_evidence.py：正式结果及 Word 审核完成后，先核验冻结源、逐图缓存、结果、模型统计及 Word 散列，再生成不含权重和逐图缓存的轻量记录包 release/v2-evidence。已用完全隔离的模拟输入验证 76 个文件的复制与散列；正式包尚未生成。
- 已将原验证子集内部路由比较纳入 V2 论文装配：新增图 18 与自动计算的 21/101 分位点结果，明确训练种子共用划分且已参与模型选择，不能称为独立测试。含 18 图、13 表的隔离 Word 模拟稿通过核验；正式稿仍须逐页检查。轻量记录包已纳入此路由数据与诊断报告。
- 图 18 新增逐点元数据与图像、数据源散列；论文装配会核验其 24 个绘图点数量及文件绑定。隔离测试确认篡改图像会使装配失败。记录包已包含两个图表元数据文件，模拟打包现核对 80 个文件。
- 论文生成器新增表 14，按筛选记录展示无蒸馏、即时蒸馏和延迟蒸馏在三个验证预算点的准确率及等权均值；保留单种子、验证集探索边界。Word 校验将表 14 与 screening_selection.json 逐格比对。隔离模拟稿已通过 14 表、18 图及可编辑公式检查，且图 18 和表 14 所在页面已目视检查。
- V2 预算曲线和配对差值图新增绑定最终汇总、逐点数据与 PNG 散列的元数据；论文装配和 Word 核验会检查这些绑定。Word 核验还逐幅比较文档内嵌图片与其源文件的二进制散列，模拟稿通过，替换源图的隔离测试被正确拒绝。
- 为解释提前退出的实际作用，新增 v2/plot_exit_risk.py：图 19 对照各模型测试出口分布以及相对同模型最终出口的有害/有益提前退出比例，均值和样本标准差按三个种子计算；正文会明确比例分母为全部测试样本。图、逐点数据和最终汇总散列绑定。19 图、14 表的隔离模拟 Word 已通过嵌入图片与数据核验；模拟值不用于正式论文。
- 图 19 的 19 图、14 表隔离模拟 Word 已渲染并检查其图文相邻页面，未发现截断或表格拆分；正式稿仍需根据最终数值重新逐页检查。
- 论文结论改为由审核汇总生成 65% 预算下的结构及蒸馏同种子准确率、实际成本差值；新增自动核验结论与出口分布叙述。19 图、14 表的隔离模拟 Word 已通过新检查；示例数值仅用于测试。
- 原自包含 HTML 导读对应 V1；新增 v2/build_explainer.py 在正式 Word 和汇总审核后保留原文、增加 V2 结构、筛选、表 12/14 与图 12–19 的解释，嵌入图片并校验原图散列。已用隔离模拟稿测试 8 张新增 PNG、所有页内锚点和输出散列，正式 HTML 尚未生成；浏览器视觉检查待最终产物。v2.explainer_after_screening 等待已启动，用户暂停时需一并停止，恢复时检查是否已有该等待进程。
- V2 导读现在也替换原可靠性图为标注 n=1 的版本，并生成独立元数据记录导读、原稿、Word、汇总与 9 张相关图片的 SHA-256。隔离测试通过锚点与嵌图散列；不将模拟页面作为交付。
- residual_seed17 已完成 160 轮并写入 completed.json；自动队列已启动 residual_kd_seed17（相同种子、即时蒸馏系数 0.5）。扩展训练尚未结束，三种子对照结果待其余训练、验证和冻结后产生。
- 最终汇总增加数学一致性检查：出口人数之和必须等于样本数，由累计路径 MACs 和出口人数重算实际计算比，且动态准确率减最终出口准确率必须等于有益早退比例减有害早退比例。V1 的 36 个模型/预算记录已通过；对人数、成本、早退指标的三种隔离篡改均被拒绝。
- 用户完成 Git Credential Manager 登录后，已以 vgoh64187-ui 读取私有仓库 main（dc86500f562876671149a77b8054ed7c31f72079）；针对未存在分支的 git push --dry-run 成功，随后 ls-remote 确认测试分支未创建。说明读写检查通过，正式同步尚未完成。隔离克隆 tmp/github_sync 正在进行，网络较慢；不影响训练。
- 已通过 Git Credential Manager 以指定账号读取与试推送；经精简隔离副本提交并推送阶段性代码、诊断记录、修订图及复现说明。main 现核验为 945e372aa1ddd14bbc78d6af5b00955bf4dc98f0（前一提交 f20a52fb581f85e19acc6a5bf553fe686ebe35c3），无原仓库文件删除。最终实验、论文、导读和轻量记录包尚未产生，完成并核验后仍需再同步。
- 轻量记录包将包含 data/split_cifar100.json 的固定划分索引，不包含原始图像。该改动已通过隔离打包测试，并已同步到仓库。
- residual_kd_seed17 已完整完成 160 轮并写入 completed.json；队列已启动 residual_seed73（无蒸馏，seed 73）。其后还需训练 residual_kd_seed73，并完成验证、冻结与测试审核。

## 用户暂停：2026-11-11 12:10（本地时间）

- 应用户要求停止 continue_screening 及审计、计时、汇总、Word、导读等待进程，再停止 residual_seed73 训练及其数据加载子进程。未检测到上述 V2 队列仍在运行。
- residual_seed73 的 history.jsonl 最后一条为第 23 轮；last.pt 内 epoch=22 为从 0 开始计数，与第 23 轮一致。检查点 SHA-256：903b421c844818ad489fa89957c292f8e151239fa62b82ff9811198d1b1a30a0；没有 last.tmp。继续时从第 24 轮恢复，并在真正恢复时记录恢复事件及 DataLoader worker 随机性说明。
- residual_seed17 与 residual_kd_seed17 已完成；residual_kd_seed73 尚未开始。恢复时先核对源码和检查点散列、确认无重复进程，再恢复训练队列和各下游等待进程。正式冻结、测试、论文、导读与最终同步尚待完成。
- GitHub main 最近核验为 3f9d48cd2fceb583dfef641faac602d848479bdc；已同步阶段性代码与说明，未同步最终结果。

## 恢复训练

- 用户重新激活目标后，确认没有旧的 V2 训练或等待进程，核对 residual_seed73 的第 23 轮日志、检查点 SHA-256 与训练源码散列。已登记恢复事件并以 --resume 从第 24 轮继续；日志确认设备为 CUDA。
- continue_screening 及审计、计时、汇总、Word、导读等待进程均已重新启动。后续仍需完成 seed73 蒸馏对照、冻结、审计、可视检查和最终同步。

## 冻结检查与历史记录缺口

- 七组训练及验证已完成。冻结检查发现 residual_delayed_kd_seed42 的 history.jsonl 缺第 35 轮；该次暂停时日志记录第 34 轮，恢复后从第 36 轮开始，符合检查点替换后、写日志前中断的情况。该轮实际数值不可恢复，保留缺口，不凭空补行。恢复事件已按日志证据修订，保留更正说明。
- freeze_v2.py 只对这一个实验的精确缺行情况允许冻结，其他缺行或重复继续拒绝；论文生成器明确披露该问题。该模型仅参与单种子验证筛选，并非三种子主要蒸馏对照。冻结后测试仍属于已在 V1 使用过的测试集上的探索性比较。
- 修正后重新执行队列，已生成冻结文件并启动测试；所有原始训练历史保留。
