# 实验日志索引

总览：[实验总报告](../experiment_report.md)。背景：[实验动机](experiment_motivation.md)。

## 维护规则

每个实验阶段新增 `NNN_阶段名.md`，使用 [阶段模板](_template.md)，并同步更新总报告与本索引。失败、中止、没有收益和设计修订都应记录；不能只记录最终成功路径。已有日志的事实错误应追加勘误，注明证据，不静默覆盖原结论。

运行日期与记录日期分开。001–005 在 2026-09-10 回顾补录；后续阶段见各自记录日期。除非有运行时间证据，不将记录日期当作 GPU 实验日期。

| 阶段 | 日志 | 状态与性质 |
|---|---|---|
| 001 | [早期 7 条探索 baseline](001_exploratory_baseline.md) | 已运行；指标不完整，暴露重复调用和规划问题 |
| 002 | [Benchmark / Evaluator v2 构建](002_benchmark_evaluator_v2.md) | 已实现；oracle 通过不代表评估规则完备 |
| 003 | [360 条 Qwen3 baseline](003_qwen3_baseline_v2.md) | 已运行；保留原协议 142/360 |
| 004 | [v2.1 回顾性审计](004_retrospective_audit_v21.md) | 审计完成；206 条答案待审、8 条需新推理 |
| 005 | [v2.1 正式接线与复核支持](005_scoring_integration_v21.md) | 已实现并测试；不是新模型实验 |
| 006 | [8 条修订问题补跑](006_corrected_prompt_rerun.md) | 8/8 执行通过、作者 AI 自审通过；非独立人工评审 |
| 007 | [206 条答案首轮自审](007_answer_review_v21.md) | 191通过、10失败、5存疑；5条待第二轮，协议未改 |
| 008 | [5 条争议用户参与裁决](008_user_adjudication_v21.md) | 1通过、4失败；旧352条当前口径192通过；非全量独立评审 |
| 009 | [v2.2恢复过程约束](009_recovery_process_v22.md) | 修复时序遗漏；188/352，107项测试通过，非模型变化 |
| 010 | [任务契约清查](010_contract_inventory.md) | 360条清单、4组模拟反例；明确剩余风险，未改评分 |
| 011 | [LoRA训练准备](011_training_readiness.md) | 2.3公开契约、320/80/80独立组合数据、CPU小模型训练通过；待4B GPU |
| 012 | [启动器失败与修复](012_launcher_failure.md) | H800已分配，脚本多余参数阻塞；修订版2已修复，待用户重启 |
| 013 | [pilot Base/SFT真实对比](013_pilot_base_sft_comparison.md) | 两个split均执行3/80→80/80；3427调用重放通过，答案与训练历史待核 |
| 014 | [完整训练记录核验](014_training_metadata_audit.md) | 1 epoch、495步、训练阶段27分6秒；配置/来源一致，答案待审 |
| 015 | [pilot答案证据复核](015_pilot_answer_review.md) | 作者AI审166条：SFT160通过，Base6错报零失败；非独立人工复核 |
| 016 | [客观证据审计](016_objective_evidence_audit.md) | 320条审计，不自动判答案通过；泛化挑战草案未运行 |
| 017 | [泛化挑战准备](017_challenge_v1_readiness.md) | 120题/18组已冻结，oracle通过；待Base/SFT成对GPU推理 |
| 018 | [挑战实测结果](018_challenge_v1_results.md) | 执行18/120→82/120，定位日志时机与导航失败；无新训练 |
| 019 | [挑战答案复核](019_challenge_answer_review.md) | 作者AI任务6/120→59/120；发现只读观测盲点，保留历史规则 |
| 020 | [观测证据审计原型](020_observation_audit_prototype.md) | 8条漏检可识别，等价工具可通过；未接入旧评分 |
| 021 | [2.4与控制实验准备](021_protocol24_control_readiness.md) | 76题、本地验证通过；新模型推理等待用户GPU决策 |
| 022 | [控制实验实测](022_control_v1_results.md) | Base/SFT各76；完整重放，保留只读清单退化 |
| 023 | [控制答案复核](023_control_answer_review.md) | 作者AI审94条；SFT62/76任务通过、Base9通过1未决 |
| 024 | [只读保护与事实摘要](024_readonly_guard_and_facts.md) | 本地验证完成，32条新推理待GPU |
| 025 | [保护实测结果](025_guard_pilot_results.md) | 32题核验；副作用减少，严格任务率未改善 |
| 026 | [证据子能力训练准备](026_evidence_subskill_readiness.md) | 288条数据及运行入口准备，待GPU预检/小规模新LoRA |
| 027 | [报告子能力实测](027_evidence_subskill_results.md) | 24步新LoRA两split48/48；旧LoRA报告形式退化 |
| 028 | [报告压力测试准备](028_evidence_stress_readiness.md) | 96题/24上下文，三种扰动；待GPU，不训练 |

当前待办：报告压力包本地准备完成，需要用户申请SuperPOD做两模型各96次推理；不训练、不直接替换执行Agent。完整Agent提升尚未验证，1条Base未决仍待独立复审。
