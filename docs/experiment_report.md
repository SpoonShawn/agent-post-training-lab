# Agent Post-Training Lab：实验总报告

阶段056：repaired targeted SFT在32条ID和32条OOD上均任务成功，执行与答案均32/32、违规0；这是修复后模型的绝对结果，尚未与同一repaired benchmark上的原SFT配对，暂不宣称提升。下一步只补跑64条原SFT baseline即可完成配对比较。[阶段日志](experiment_logs/056_repaired_sft_first_results.md)。

阶段055：发现旧数据train没有`revoke_before_commit`而OOD全部包含该机制，新增repaired benchmark并固定共享机制、独立group和五类知识面。计划从原SFT做128步targeted SFT，评估32条repaired ID与32条repaired OOD；这是新协议，旧结果不改写。尚未运行GPU。[设计与指令](experiment_logs/055_repaired_benchmark_and_targeted_sft.md)。

阶段054：GRPO pilot的12个独立评测文件已完整上传并审计。SFT ID为128/128，6个GRPO checkpoint均128/128；SFT OOD为258/400，t1.2/t1.8均257/400，t1.5均256/400，0条改善、1–2条退化。执行成功率均400/400、违规调用均244，差异集中在tool_errors最终报告计数。当前证据不支持GRPO泛化提升。[阶段结果](experiment_logs/054_grpo_independent_eval_results.md)。

阶段053：GRPO独立评测作业598060运行3:43:51时完成873/3168条，按当前平均速度预计超过8小时上限。已修复评估器：按顺序/metadata/重放校验现有记录后续接，完整文件跳过，不覆盖部分结果。[续跑记录](experiment_logs/053_grpo_eval_resume.md)。尚无完整ID/OOD结论。

阶段052已准备6个GRPO checkpoint的独立ID/OOD评测入口，固定128条ID与400条OOD，不回灌训练、不覆盖SFT结果。当前等待SuperPOD后台评测；训练reward不能替代泛化指标。[评测准备](experiment_logs/052_grpo_independent_eval_readiness.md)。

阶段051已上传完整GRPO pilot：3个独立SFT初始化arm、temperature 1.2/1.5/1.8，各2个update，共288条训练rollout。t=1.2从0.9792升至1.0并饱和；t=1.5为0.6667→0.625；t=1.8为0.0208→0。所有arm参数均变化，但这仍是训练侧统计，尚无ID/OOD提升结论。下一步运行独立confirmation评测。[阶段日志](experiment_logs/051_grpo_pilot_uploaded_results.md)。

阶段050：复查595207后发现v1不仅有显存峰值问题，还有temperature顺序混杂、behavior logprob一致性未验证和tool validity固定值等接线风险。已重写为v2 runner：分块目标logits、共享基座双adapter、三档temperature独立初始化、8192上下文memory/gradient gate。旧pilot不算完成，需先通过GPU gate。[重设计记录](experiment_logs/050_grpo_runner_redesign.md)。

阶段049：作业595207完成temperature 1.2的两个update（第一批reward饱和，第二批32条中30条成功并出现优势），随后在下一档temperature的完整词表teacher-forced forward因显存耗尽失败；没有完整GRPO成绩，也未进行ID/OOD评测。已加入use_cache关闭、gradient checkpointing和逐turn内存释放，保留原始OOM记录。[故障与中间结果](experiment_logs/049_grpo_pilot_oom.md)。

阶段048：作业595203在模型加载前因LoRA修复后的代码SHA与旧pilot协议不一致而停止；无训练结果。已保留旧协议并升级为v2协议重新冻结，数据、reward和实验边界不变。[故障记录](experiment_logs/048_grpo_pilot_protocol_drift.md)。

阶段047：GRPO pilot 作业595095在加载双模型后因LoRA adapter默认冻结，优化器收到空参数而失败；无模型更新、无checkpoint、无成绩。已修复policy/reference的冻结状态并增加启动检查，原始失败日志保留。[故障记录](experiment_logs/047_grpo_pilot_launcher_failure.md)。

阶段046已冻结正式GRPO pilot设计：原SFT初始化，24个train结构组、group size 4、temperature 1.2/1.5/1.8，各2次更新，共192条训练episodes；ID/OOD只做独立更新后评测，32条variance probe不回灌。当前仅完成代码与协议验证，GRPO尚未训练，等待SuperPOD运行。[设计与边界](experiment_logs/046_grpo_pilot_design.md)。

最新进度（阶段044/045，2026-09-17）：正式80步DPO及784题评测已完成，三臂2352条轨迹重放。开发256/256、ID128/128保持不变，OOD从SFT258/400降至DPO256/400，0改善、2退化；偏好开发loss约3.72e-9并未转化为任务收益，16条权限失败未解决。[完整负结果与限制](experiment_logs/044_full_dpo_results.md)。已冻结原SFT、仅train的8题×4次GRPO奖励方差/行为概率采样，需[SuperPOD操作](experiment_logs/045_grpo_rollout_readiness.md)。该采样没有优化器更新，不是GRPO训练；多seed、对照及独立后端仍未完成。以下均为历史快照。

最新进度（阶段042/043，2026-09-17）：两步DPO实测已核验，98对固定参考完整；初始loss=log2，梯度范数36.49/29.73，数值/mask检查通过，H800总窗口236.67秒。四探针偏好差距改善主要来自压低rejected，两个chosen概率略降；没有闭环任务提升证据。[结果与局限](experiment_logs/042_dpo_gate_results.md)。已冻结从原SFT重启的80对/80步正式小规模DPO，随后同784题比较，支持checkpoint/逐题续跑；需要[后台GPU操作](experiment_logs/043_full_dpo_readiness.md)。正式DPO尚未运行，GRPO、多seed和独立后端未完成。以下为历史快照。

最新进度（阶段040/041，2026-09-17）：98条Base训练侧候选已重放，Base完整任务0/98，对应SFT候选98/98，形成98对跨策略偏好、0弃权。按组冻结80训练/18偏好开发；后者在SFT阶段见过，不称独立确认。[配对与局限](experiment_logs/040_cross_policy_pairs.md)。已实现多轮assistant-only DPO、固定SFT参考缓存、两遍精确链式梯度及数值/mask检查；本地无torch，真实反向传播待[两步GPU工程检查](experiment_logs/041_dpo_engineering_gate.md)。正式DPO未训练、无新任务成绩；低负例概率不等于DPO初始梯度消失的说明已补充。以下为历史快照。

最新进度（阶段038/039，2026-09-17）：训练侧偏好probe392/392轨迹核验，全成功；98题各4候选完全相同，0偏好对、98题全部弃权。这是偏好供给负结果，不是新泛化成绩，也没有进行DPO。保留原结果，不伪造负例、不回灌确认题。[完整复盘](experiment_logs/038_preference_probe_zero_pairs.md)。下一步已冻结同98个train任务的Base真实轨迹采集，与固定SFT候选形成跨策略离线候选；需要[短GPU作业](experiment_logs/039_cross_policy_preferences_ready.md)。负例可能过易，配对后仍须检查DPO梯度信号，不能保证收益。以下为历史快照。

最新进度（阶段036/037，2026-09-17）：完整SFT首seed已运行并核验1568条轨迹。完整任务Base均0；SFT开发256/256、确认ID128/128、OOD258/400（64.5%）。OOD126条只错错误计数，16条撤权后循环至预算耗尽，共244次违规请求；执行条件400/400不能当完整成功。训练3943步/10268秒；两次评测中断已续完，原始失败926条保留索引。[实测与故障复盘](experiment_logs/036_full_sft_results.md)。已准备仅train的98题×4候选采样检查，下一步需[后台GPU采样](experiment_logs/037_preference_probe_readiness.md)；不更新权重，不是DPO训练，若无好坏对会如实记录。DPO/GRPO、多seed与独立后端仍未完成。以下为历史快照。

最新进度（阶段035，2026-09-16）：上传工程结果28/28重放通过；Base/两步adapter各执行2/14、完整任务0/14，失败全部保留，未改评分。两步只监督最长报告前缀，不代表完整多轮SFT。H800峰值allocated13.26GiB；真实训练输入52593783tokens、监督663355tokens。已冻结fresh完整SFT：2058轨迹/31542预测例子、1epoch/3943步，Base/SFT各784题分开发/确认ID/OOD评价；正式训练尚未运行。需要[SuperPOD操作](transaction_full_sft_v1.md)，支持检查点续跑。[实测、具体失败与设计取舍](experiment_logs/035_transaction_smoke_and_full_sft.md)。DPO/GRPO、多seed、独立后端仍待完成。以下保留历史快照。

最新进度（阶段034，2026-09-16）：新完整Agent事务数据与模型接线已冻结，2058训练/256开发/128确认ID/400确认OOD，98结构组；训练展开31542个下一轮预测例子。2842条oracle通过，非模型成绩。新确认模型推理未开始，现需[SuperPOD工程检查](transaction_v1.md)：真实分词、14题Base、两步LoRA、14题重载检查。拿到显存/token/吞吐后才冻结完整SFT预算；DPO/GRPO仍未运行。[数据边界与问题记录](experiment_logs/034_transaction_dataset_gpu_gate.md)。

项目重新立项（2026-09-16，阶段033）：用户明确要求完成完整后训练路线，阶段031/032收尾仅属于pilot，不再表示整个项目结束。[新路线](full_experiment_plan.md)包括完整Agent SFT、DPO、在线GRPO、统一多seed对照和独立工具后端迁移。已新增隔离事务环境，支持幂等、提交后超时、revision冲突、异步检查、回滚和撤权；50开发oracle重放通过，非模型成绩。正式数据/训练尚未准备完，不需GPU；[过程与问题](experiment_logs/033_full_agent_restart.md)。旧成果和未决项原样保留。

学习交接（2026-09-16，阶段032）：新增[初学者学习路线/讲述/追问](project_learning_guide.md)及[完整证据退化案例](learning_case_walkthrough.md)，对应原始工具定义、完整输入和三个模型回答。本阶段仅本地文档，不新增训练、推理或成绩，不需要SuperPOD；[记录](experiment_logs/032_learning_handoff.md)。

收尾状态（2026-09-16，阶段031）：本轮预定范围已完成模型运行与审计，不再申请GPU。最终确认集48上下文×3布局：Base21/144，fixed56/144（38.89%），mixed83/144（57.64%）；同192训练样本、24步、15960监督tokens，mixed提高18.75个百分点，但配对7条退化，主要收益集中表格。全部864条输出合法JSON，524条完整报告失败保留；评分未改。见[实测与失败](experiment_logs/031_evidence_ablation_results.md)、[收尾/复现/面试事实清单](experiment_closeout.md)。这是单seed合成报告子任务，不是完整Agent泛化或生产成绩。DPO/GRPO、独立复审及真实设备列为未来工作，不算本次已完成。以下均为历史快照。

最新进度（2026-09-16，阶段030）：固定呈现/混合呈现的成对LoRA实验包已冻结：相同192训练上下文、答案与24优化步，验证/确认各48上下文×3布局。新结构组不跨split，旧题不回灌。Base及两个新adapter各288次推理尚待SuperPOD；没有新成绩。需用户申请GPU，见[具体指令](evidence_ablation_v1.md)、[阶段记录](experiment_logs/030_evidence_ablation_readiness.md)。本轮正负结果均可收尾，预计剩余2–3工作日（排队/操作延迟顺延），不包含DPO/GRPO，不保证涨分。

最新进度（2026-09-16，阶段029）：压力实测192条已核验。原文/换写/历史增长/权限翻转各24题，完整报告Base分别15/8/0/4，新报告LoRA24/13/0/7；所有输出合法JSON，121条失败保留。新LoRA历史组20条最新验证字段错误、15条总失败数错误，权限组17条决策错误，不能归因于评分太死板。原文各24条逐字复现旧回答；模板内满分未泛化，不修改旧评分。见[压力结果、失败与下一步](experiment_logs/029_evidence_stress_results.md)。当前不需GPU；下一阶段先准备独立数据与等预算消融，尚未训练。以下为历史快照。

当前进度（阶段028）：报告压力测试96题/24上下文已本地准备，四组为原文、证据换写、增加历史、当前权限翻转；模型成绩尚未产生。需SuperPOD固定Base与报告LoRA各推理96次，不训练。旧满分仅作小模板任务结果，不外推。见[准备与工程问题](experiment_logs/028_evidence_stress_readiness.md)、[GPU指令](evidence_stress_v1.md)。

最新进度（阶段027）：报告子能力真实LoRA完成192样本、1epoch/24步。验证完整报告Base32/48→新LoRA48/48，确认29/48→48/48；旧事务LoRA两组均0/48，全部输出工具调用而非JSON报告。不是完整Agent成绩，也不是继续旧adapter后的修复。数据/训练/三模型288条输出已核验，131条失败完整保留。见[实测与适用边界](experiment_logs/027_evidence_subskill_results.md)。

当前进度（阶段026）：新工具历史审计子能力数据288条（192训练/48验证/48确认，48组合组）及独立训练入口已准备；只监督最终结构化报告，失败历史不作为正向动作目标。尚未真实分词/推理/训练，需要SuperPOD完成预检和小规模新LoRA实验。不是第二版完整Agent，不修改旧成绩。见[数据、问题与边界](experiment_logs/026_evidence_subskill_readiness.md)、[运行指令](evidence_pilot_v1.md)。

最新进度（阶段025）：32条保护推理已核验，107次受保护调用重放一致。SFT只读清单成功副作用24→0，新轨迹15次违规请求全部拦截，但严格执行仍4/8；3条耗尽预算，1条取证后错报零失败。对应16题任务仍SFT3/16，Base4通过1未决，非独立作者AI口径；不能与完整76题成绩混比。结论：系统安全改善，模型任务率未提升。见[实测与负结果](experiment_logs/025_guard_pilot_results.md)。无新训练。

当前进度（阶段024）：只读执行保护与工具事实摘要已独立实现；原模型回答、违规尝试和历史指标不覆盖。152条离线事实一致性及4条首次违规拦截验证通过，不代表新模型成功。已冻结16题×两模型的32次保护反馈推理，需要用户申请GPU观察真实后续行为；不重训。见[实现与局限](experiment_logs/024_readonly_guard_and_facts.md)、[SuperPOD指令](guard_pilot.md)。

最新进度（2026-09-15，阶段022/023）：控制实验Base/SFT各76条已上传并核验，1558次工具返回及152终态重放一致。五条件执行分别Base 7/20、2/20、6/20、2/8、6/8；SFT 20/20、19/20、20/20、8/8、4/8。清单让SFT只读4条退化，不能当通用修复。94条作者AI复核后，SFT任务62/76（81.58%）；Base9通过、66失败、1未决，任务率待定。非独立盲审、非新训练、非独立held-out。详见[执行对照](experiment_logs/022_control_v1_results.md)与[答案/问题复盘](experiment_logs/023_control_answer_review.md)。以下状态均为历史快照。

当前进度（阶段021）：新协议2.4与76题控制实验已完成本地准备，旧2.3及两套冻结包不变；76条oracle通过。真实Base/SFT各76次推理尚未运行，现在等待用户决定是否申请SuperPOD。新实验不训练、不发布模型成绩；[操作与对照解释](control_v1.md) · [阶段日志](experiment_logs/021_protocol24_control_readiness.md)。下文均保留历史快照。

当前完成（阶段020）：状态观测证据检查原型已实现，定位Base8条缺观测并接受等价状态返回；未接入2.3、不改历史分数。当前任务成绩以阶段019作者AI复核5%/49.17%为准；独立评审、新协议接线与新控制实验尚未完成。见[原型与边界](experiment_logs/020_observation_audit_prototype.md)。

最新进度（2026-09-15，阶段019）：100条挑战答案完成程序辅助的作者AI复核，Base6通过12失败，SFT59通过23失败；任务率分别5%和49.17%，非独立盲审。原执行18/120、82/120不变。新发现8条Base只读题未取得状态观测却被执行层放行，缺陷公开保留，尚未修改冻结协议。详见[答案复核与评估盲点](experiment_logs/019_challenge_answer_review.md)和[学习复盘](learning_review.md)。

最新进度（2026-09-15，阶段018）：挑战实测Base执行18/120（15%）、SFT82/120（68.33%）。SFT原文20/20、重排0/20，但重排19条终态正确，全部缺本次操作日志证据；另有真实导航与动作遗漏问题。2366次工具返回重放通过。100条执行通过答案待审，最终任务率尚未确定。未训练、未改评分，详见[挑战结果与负例](experiment_logs/018_challenge_v1_results.md)。

当前进度（阶段017）：120条/18组训练后挑战集已冻结，120条oracle通过；尚无新模型结果。固定原Base/SFT各推理120题，不重训。见[GPU操作指南](challenge_v1.md)和[构造/失败日志](experiment_logs/017_challenge_v1_readiness.md)。下方各阶段进度保留为历史快照。

最新进度（阶段016）：新增不改评分的客观证据审计，覆盖320条，Base发现35条明确数字矛盾候选、89条未解析待审；SFT160条数字匹配但不自动批准答案。136项测试通过。新泛化挑战目前仅有[构造方案草案](generalization_challenge_plan.md)，尚未生成或推理。详见[审计日志](experiment_logs/016_objective_evidence_audit.md)。

最新进度（阶段015）：166条执行通过答案完成程序辅助的协议作者AI复核，SFT160通过、Base6失败（都错报零失败）。验证与确认各自任务成功Base0/80、SFT80/80；原执行成绩3/80→80/80不变。非独立人工/盲审，有限模板内结果，不证明复杂业务泛化。详见[答案复核日志](experiment_logs/015_pilot_answer_review.md)。以下阶段013/014的待审状态为当时快照。

最后更新：2026-09-14。状态：pilot_v1真实Base/SFT四份推理结果和完整训练元数据已核验。验证与确认分别从Base执行3/80提升到SFT执行80/80；答案尚待复核，不发布最终任务成功率。H800完成1 epoch、495步，训练过程平均loss0.03510224、最终验证loss0.00002065187；Trainer训练阶段约27分6秒。DPO/GRPO未开始。旧352条2.2成绩仍188/352=53.41%，不与新数据混比。

本报告随每个实验阶段更新；成功、失败、设计修改和未决事项均保留。细节与原始证据见 [阶段日志索引](experiment_logs/README.md)。历史阶段是基于仓库产物回顾补录，不将补录日期当作运行日期。

最新现场更新（阶段013）：启动修复后用户完成运行，原始结果提交ec581f4。全部3427条工具调用可精确重放，比较配置一致；本次未改评分标准。见[Base/SFT比较](experiment_logs/013_pilot_base_sft_comparison.md)。阶段012启动失败记录继续保留。

## 1. 目标与研究边界

最新补充（阶段014）：小体积训练记录已提交a5c1c92，manifest、基础权重和Base来源哈希一致；训练run总耗时28分19秒，不等于全流程耗时。极低验证loss只反映正确历史前缀下的模板化目标预测，不代表真实业务泛化。见[训练记录核验](experiment_logs/014_training_metadata_audit.md)。

研究独立构造的中文 BugOps/Tool-Use Agent 是否可通过后训练提高规划、状态跟踪、参数生成、证据使用和异常恢复能力。路线为 instruction model baseline → LoRA SFT → DPO → 条件允许时 GRPO；后几项只是计划，不是已完成结果。

[实验动机](experiment_logs/experiment_motivation.md) 保留用户原文。环境、工具、知识库及故障都是合成模拟，不含前雇主内部数据，也不等于真实 Android 应用测试。Base 在本报告指尚未经过本项目后训练的 instruction model，不是纯预训练权重。

## 2. 阶段进度与证据

| 阶段 | 目标 | 状态/结论 | 详细记录 |
|---|---|---|---|
| 031 呈现对照实测与收尾 | 验证收益及退化，完成证据闭环 | 确认fixed56/144→mixed83/144；改善34、退化7；保留524失败 | [日志](experiment_logs/031_evidence_ablation_results.md) |
| 030 呈现多样性对照准备 | 同历史同答案、等样本/步数比较 | 192训练/48验证/48确认上下文；待GPU，旧评分不改 | [日志](experiment_logs/030_evidence_ablation_readiness.md) |
| 029 报告压力实测 | 固定模型检验三类扰动 | 新LoRA24/13/0/7（各24）；暴露证据时序及权限判断问题 | [日志](experiment_logs/029_evidence_stress_results.md) |
| 001 探索 baseline | 检查基础工具使用 | 7 条运行完成；14 次工具失败，匹配指标仅 6 条适用 | [日志](experiment_logs/001_exploratory_baseline.md) |
| 002 v2 构建 | 可验证状态、可控故障、扩集 | 360 条 oracle 通过；后续发现设计盲点 | [日志](experiment_logs/002_benchmark_evaluator_v2.md) |
| 003 模型 baseline | 测量现有能力 | v2 任务成功 142/360；存在真实错误与误判 | [日志](experiment_logs/003_qwen3_baseline_v2.md) |
| 004 v2.1 审计 | 分离测量缺陷与模型失败 | 352 条可重评分，206 条执行通过；答案待审 | [日志](experiment_logs/004_retrospective_audit_v21.md) |
| 005 正式接线 | 版本分流、复核、分母保护 | 已实现，95 项测试通过，无新模型推理 | [日志](experiment_logs/005_scoring_integration_v21.md) |
| 006 修订问题补跑 | 验证补充事件编号后的 8 条输入 | 覆盖 8/8，执行和作者 AI 自审均 8/8 通过 | [日志](experiment_logs/006_corrected_prompt_rerun.md) |
| 007 旧轨迹答案自审 | 核对 206 条执行成功后的答案 | 首轮覆盖206/206；191通过、10失败、5存疑；规则未改 | [日志](experiment_logs/007_answer_review_v21.md) |
| 008 用户参与裁决 | 逐条审阅 5 条争议 | 1通过、4失败；352条当前口径192通过；过程约束问题待修订 | [日志](experiment_logs/008_user_adjudication_v21.md) |
| 009 恢复过程修复 | 显式失败后检查时序 | v2.2：188/352；7条新增执行失败，4条新增任务失败 | [日志](experiment_logs/009_recovery_process_v22.md) |
| 010 任务契约清查 | 题面与隐藏条件一致性 | 360条清单、4组模拟反例；111项测试；未改成绩 | [日志](experiment_logs/010_contract_inventory.md) |
| 011 训练准备 | 公开契约、独立数据、LoRA入口 | 320/80/80按组隔离；120测试；CPU小模型训练通过，待4B上机 | [日志](experiment_logs/011_training_readiness.md) |
| 012 启动失败修复 | 修正Shell参数与入口测试 | H800检查成功，argparse失败；bundle revision2，122测试，待上机重试 | [日志](experiment_logs/012_launcher_failure.md) |
| 013 pilot真实对比 | 固定协议下比较Base/SFT | 两个split均3/80→80/80执行通过；语义与训练历史待补核 | [日志](experiment_logs/013_pilot_base_sft_comparison.md) |
| 014 训练历史核验 | 补齐真实训练配置与曲线 | 完成1 epoch、495步；哈希和统计一致，语义仍待审 | [日志](experiment_logs/014_training_metadata_audit.md) |
| 015 pilot答案复核 | 检查实际工具证据与最终声明 | 作者AI审166条：160通过6失败；任务两split各0/80→80/80 | [日志](experiment_logs/015_pilot_answer_review.md) |
| 016 客观证据审计 | 检查失败次数矛盾并明确弃权 | 320条审计；不改评分，挑战方案仍为草案 | [日志](experiment_logs/016_objective_evidence_audit.md) |
| 017 泛化挑战准备 | 固定模型测四类扰动 | 120题/18组oracle通过，待GPU推理；无训练 | [日志](experiment_logs/017_challenge_v1_readiness.md) |
| 018 挑战实测 | 固定模型检查迁移边界 | 执行18/120→82/120；表达重排全部缺日志证据，答案待审 | [日志](experiment_logs/018_challenge_v1_results.md) |
| 019 挑战答案复核 | 证据、自洽和完整性 | 作者AI任务6/120→59/120；8条只读观测盲点未改冻结分数 | [日志](experiment_logs/019_challenge_answer_review.md) |
| 020 观测证据原型 | 区分后台真值与可见证据 | 定位8条漏检；候选审计，不改冻结2.3 | [日志](experiment_logs/020_observation_audit_prototype.md) |
| 021 新协议与控制实验 | 前瞻观测/日志条件、提示清单对照 | 2.4独立入口、76题已准备；等待GPU决策，无新训练 | [日志](experiment_logs/021_protocol24_control_readiness.md) |
| 022 控制实验实测 | 固定模型的三组配对比较 | 152题核验；清单改善部分任务，却导致SFT只读4条退化 | [日志](experiment_logs/022_control_v1_results.md) |
| 023 控制答案复核 | 检查完整性与证据来源 | 作者AI审94条；SFT62/76任务通过，Base9通过1未决 | [日志](experiment_logs/023_control_answer_review.md) |
| 024 工程保护 | 只读工具门禁、证据摘要 | 本地验证完成；32次新反馈推理待GPU，不改模型成绩 | [日志](experiment_logs/024_readonly_guard_and_facts.md) |
| 025 保护实测 | 区分安全效果和模型成功 | 32题核验；副作用降为0，严格任务成功未提升 | [日志](experiment_logs/025_guard_pilot_results.md) |
| 026 子能力训练准备 | 证据报告、权限与重试分类 | 288条按组划分；本地测试，待SuperPOD真实预检/训练 | [日志](experiment_logs/026_evidence_subskill_readiness.md) |
| 027 报告子能力实测 | 固定报告任务比较三模型 | 新LoRA两split48/48，旧LoRA输出形式退化；非完整Agent提升 | [日志](experiment_logs/027_evidence_subskill_results.md) |
| 028 报告压力准备 | 呈现/历史/权限三类扰动 | 96题本地验证，待Base/报告LoRA各96次推理 | [日志](experiment_logs/028_evidence_stress_readiness.md) |

代码/产物节点：`287ffee` v2 实现、`c6a8005` 真实 baseline、`0fecffa` 初版审计、`4a2bf12` 正式接线。阶段 001 产物节点 `2ff581a`。

## 3. 配置与数据集

已验证的 360 条实测配置：Qwen3-4B-Instruct-2507；greedy（do_sample=false）；每轮 max_new_tokens=512；无统一 max_steps override，默认 16，具体任务按 case 预算执行；Python 3.11.16 / torch 2.9.1 / transformers 5.16.1。权重、代码与运行指纹完整值见 [阶段 003](experiment_logs/003_qwen3_baseline_v2.md)。GPU 型号、作业起止时间、总耗时、token 总量和随机种子未在已查阅元数据中记录，不猜测。

[v2 数据](../data/eval/bugops_eval_v2.jsonl)：360 条，12 族 × 15 场景 ID × 2 表述。类别分布 60 knowledge、60 single_tool、30 navigation、120 long_horizon、60 recovery、30 no_tool。成对改写不能按独立样本计置信区间；180 个 ID 也不自动代表独立场景抽样。

该数据最初按 held-out 构造，但已用于协议调整，现是开发／审计集。不能将 case、答案、轨迹或改写回灌训练；最终确认性结果需要独立的分组隔离数据。

阶段010补充规模边界：知识库仅8条（历史事件2、构建记录3、UI指南3）；360题对应166种严格environment+required_calls签名，不等于166个独立语义场景。该项目应定位为合成微型基准，不是360个真实业务缺陷。

阶段011新增独立配置事务pilot：24个完整动作组合先分16/4/4组，再展开为320训练、80验证、80确认；3956个训练assistant样本、97356目标tokens。新生成器不导入旧题/答案，构造后精确污染检查零重叠，不声称已证明所有语义独立性。共享工具原语、确认仅4组，结论限组合泛化小实验；本轮不训练知识检索。见[冻结manifest](../data/pilot_v1/manifest.json)。

## 4. 指标与实测结果

### pilot_v1：固定2.3协议的新模型比较（阶段013）

| 数据 | Base执行通过 | SFT执行通过 | Base恢复通过 | SFT恢复通过 |
|---|---:|---:|---:|---:|
| 验证80条 | 3/80（3.75%） | 80/80（100%） | 0/20 | 20/20 |
| 确认80条 | 3/80（3.75%） | 80/80（100%） | 0/20 | 20/20 |

两组分别77条失败转通过、3条保持通过、零执行退化；无故障长链路分别3/60→60/60。确认集非预期工具失败212→0，但SFT仍有20次预设故障并完成恢复。答案未全量复核，最终任务率为null。每个split仅4个组合组，不能按80个独立场景推断泛化或宣称生产能力；本轮没有知识检索测试。步骤均值确认11.375→12.55，补齐日志/状态取证可能增加步骤，不声称效率提升。详见[机器可复核汇总](../results/analysis/pilot_v1/comparison.json)。

### 原 v2：保留历史任务成绩

以下是同一个 360 条运行、原 2.0 规则的指标。工具类比率多为 case 宏平均；参数准确率适用 330 条，不与总体分母混用。

| 指标 | 结果 |
|---|---:|
| 运行覆盖 | 360/360 |
| 任务成功 | 142/360 = 39.4% |
| 长链路任务成功 | 0/120 |
| 恢复任务成功 | 1/60 = 1.7% |
| 工具 precision / recall | 0.827 / 0.780 |
| ordered / set match | 0.406 / 0.406 |
| 参数准确率 | 0.495 |
| 有效调用率 | 0.999 |
| 原始 / 调整后调用执行成功 | 0.812 / 0.843 |
| 未知工具率 / 重复率 | 0.001 / 0.012 |
| 平均 Agent steps | 4.972 |

原始 [结果文件](../results/baseline/qwen3_4b_baseline_v2.jsonl) 保持不变。v1 的 3/6 轨迹匹配不是任务率，不与本表比较模型提升。

### v2.1：回顾性执行审计，不是模型提升

| 类别 | 原 v2 任务通过 | v2.1 可重评分 | v2.1 执行通过 |
|---|---:|---:|---:|
| no_tool | 30/30 | 30 | 30 |
| single_tool | 59/60 | 60 | 60 |
| knowledge | 52/60 | 52 | 50 |
| navigation | 0/30 | 30 | 23 |
| long_horizon | 0/120 | 120 | 3 |
| recovery | 1/60 | 60 | 40 |

两列“通过”口径不同。初次审计执行通过 206/352=58.5%，当时206条答案全部待审，146条执行失败；8条因输入修订排除，现已另行补跑。该历史待审快照保留在 [原审计汇总](../results/audit/v21/summary.json)，不能将它当作最终任务成功率或模型提升。

### 206 条首轮作者 AI 自审（阶段 007）

逐条核对问题、实际工具证据和答案后：191 pass、10 fail、5 uncertain。属于协议作者自审，已接触模型身份及旧结果，非独立人工/盲审。未改规则，也未重新推理。

| 类别 | 可重评分分母 | 任务通过 | 任务失败（含执行失败） | 未决 |
|---|---:|---:|---:|---:|
| no_tool | 30 | 30 | 0 | 0 |
| single_tool | 60 | 59 | 1 | 0 |
| knowledge | 52 | 40 | 8 | 4 |
| navigation | 30 | 23 | 7 | 0 |
| long_horizon | 120 | 2 | 118 | 0 |
| recovery | 60 | 37 | 22 | 1 |
| 合计 | 352 | 191 | 156 | 5 |

完整任务率仍为 null，逻辑界限为 **191/352–196/352，即54.26%–55.68%**，不是置信区间。156 个失败由原146个执行失败加10个答案失败组成。不能把191/206当整体分数，也不拼接另外8条发布统一运行成绩。

见 [首轮判定](../results/reviews/v21_author_reviews.jsonl)、[复核后汇总](../results/audit/v21_reviewed/summary.json)、[5条复审证据包](../results/reviews/v21_second_review_queue.jsonl)。失败包括无最终答案、平台/版本范围扩张、编造修复结论或故障原因、把全过程说成无错误。通过项中的截断/泛泛保证等非致命问题也保留在理由中；主观边界需后续抽查。

### 8 条修订输入的新推理：单独报告

用户提交 `4de417e`，本地于 2026-09-14 核查：覆盖 8/8、执行达成 8/8、工具 precision/recall 均 1.000、平均 2 steps。逐条读取问题、实际工具返回和答案后，作者 AI 自审 8 pass，0 fail，0 uncertain；本子集任务成功 8/8，不是独立人工评审。

模型权重及主要生成配置与原 baseline 相同，输入和 runtime 指纹不同，详见 [阶段 006](experiment_logs/006_corrected_prompt_rerun.md)。[自审文件](../results/reviews/rerun_v21_author_reviews.jsonl) 已绑定来源哈希；原始结果和队列不改写。不把这 8 条与旧 352 条直接拼接作统一运行，也不外推为长链路能力提升。

新观察：检索同时返回两个事件且同分，模型本次均选对；旧检索参数精确匹配仅 0.250，结构化参数为 n/a。本轮只记录局限，不为提高分数修改规则。

### 5 条争议的用户参与裁决（阶段 008）

用户逐条查看完整问题、调用、返回与答案，并在术语解释后给出意见。助手整理为 1 pass、4 fail，保留用户原话及首轮判定；不是独立盲审。其余 201 条首轮标签逐字段不变。

当前 206 条执行通过答案合计 192 pass、14 fail、0 uncertain；加上 146 条执行失败，旧轨迹可重评分分母 352 的任务成功为 **192/352=54.55%**，失败 160，未决 0。分类通过数依次为 no_tool 30/30、single_tool 59/60、knowledge 41/52、navigation 23/30、long_horizon 2/120、recovery 37/60。另 8 条补跑继续单独报告。

见 [合并裁决](../results/reviews/v21_adjudicated_reviews.jsonl)、[新汇总](../results/audit/v21_adjudicated/summary.json)。上文阶段 007 的 null 与区间是保留的历史快照。本次未改变评分代码，未重新推理，不是模型提升。恢复案例暴露的“失败后检查再重试”约束遗漏尚未修复；当前成绩不是冻结协议下的确认性结果。

### v2.2：恢复过程修复（阶段009）

在相同352条上，执行通过206→199、任务通过192→188，任务成功率为188/352=53.41%，164失败、0未决。60条恢复任务执行通过40→33、任务通过37→33，其余类别不变。原因是7条遗漏必要的失败后状态检查，其中3条此前已有答案错误。属于测量修正，不是模型性能变化。答案复核只复用原问题/结果的已验证来源标签；8条补跑仍单报。

新增过程证据和负例测试，107项测试通过；旧2.1代码行为/结果保留。[原理与复现](protocol_v22.md) · [新汇总](../results/audit/v22/summary.json)。并未宣称所有隐藏条件和语义边界已解决。

## 5. Bad case、问题与原因分析

| 问题类型 | 证据/用例 | 原因与处理 |
|---|---|---|
| 等价路径误判 | navigate_graphics_001 | 动作返回已有状态，却强制 inspect/verify；放开工具身份，保留终态 |
| 同义/否定误判 | return_home_then_dungeon_002、negative_control_ios_002 | 少量关键词不能表达语义；改显式复核，但仍检查日志证据 |
| 检索参数过严 | knowledge 类参数旧分数为零 | 自由文本 query 精确匹配失真；结构化参数另计，检索看证据 |
| 输入缺信息 | knowledge_black_screen_011 等 8 条 | 题面未给事件编号；补充并重推理，不复用旧答案 |
| 真实规划失败 | reproduce_black_screen_001 | 未启动 battle、缺日志却得出未复现；不能通过放宽规则消除 |
| 恢复被低估 | recovery_navigation_001 | 故障后确实重试成功；取消唯一验证工具要求，保留跨轮恢复条件 |
| 汇总/来源风险 | pending 与 false 混合、旧复核套新答案 | 未决不得丢弃，复核绑定来源哈希与署名 |
| 运行指导缺陷 | 曾未先进入目录、未先申请 GPU | 分开登录节点审计与计算节点推理，明确释放资源 |

详细定位、反例和设计取舍见 [实验问题复盘](experiment_lessons_v21.md)。仍需审查部分隐藏日志条件与题面要求是否一致。没有证据证明所有低分来自 evaluator，也没有完成独立语义评审。

## 6. 解决方案与验证

保留原始结果；版本化修改而非追改旧成绩；统一评分入口；拒绝混合/未知协议和 query 错配；新代码进入运行指纹；禁止无意覆盖输出；复核记录来源、逐条证据、判定、理由、审阅者类型和时间；待审明确展示；8 条修改问题单独导出。

工程验证：阶段 005 的 95 项自动化测试通过，含 360 条 oracle、等价路径、缺日志、错误终态、输入隔离、复核导入和未决分母。真实 baseline 集成测试核对 352/206/8 与源哈希。测试中的合成 pass 仅存在临时目录，不能视为已完成答案复核。

局限：oracle 通过不代表模型成功；测试通过不证明所有题目无歧义；人工/AI 复核亦需质量控制；该模拟环境结果不等于生产 GUI 能力。

## 7. 下一步与完成标准

阶段023当前待办：不需要重跑已完成推理。先基于本轮证据设计本地可测的任务条件清单/轨迹事实摘要，任何运行链路改变另冻结版本；若随后需要GPU再由用户决定。Base1条验证解释待独立复审，其余AI标签亦需抽检。不要用通用清单覆盖旧策略，或立即增加训练轮数；本轮诊断数据不入训练。下列为旧阶段计划，完成情况以最新日志为准。

阶段015更新：下列第4项首轮答案审查已完成；独立复核仍未完成。下一阶段设计新的冻结挑战集，固定现有adapter测表达/任务结构/约束/故障位置变化，先核实模拟器支持，不直接增加训练轮数。尚未构造新集或安排GPU推理，不宣称已解决过拟合疑虑。

1. 阶段010列出的日志/顺序/禁止多余操作/只读/工具等价问题已在2.3公开约定版处理并测试；旧360题均修改题面，必须新推理，不能继承旧成绩。旧答案一致性抽查仍未完成，但本轮主要执行指标和新数据不依赖重判旧标签。
2. 8 条补跑及作者 AI 自审已完成，当前无需重复申请 GPU。若之后需完整重跑，应先确定协议/输入并记录作业、硬件和时间；不要直接拼接旧 352 条与新 8 条作同一原始运行。
3. pilot_v1四份Base/SFT推理及完整训练run.json、trainer_state.json已完成核验，无需重跑或重新申请GPU。阶段014补齐实际训练统计；保存时adapter缺内容哈希的来源局限仍保留。
4. 对本轮执行通过答案逐条开展绑定来源的语义复核，再发布最终任务成绩。主指标仍为预注册确认集执行成功率；不据当前确认成绩调参，后续基于其反馈开发应准备新确认集。先完成证据闭环，再讨论扩场景或DPO/GRPO。

操作命令见 [v2.1 指南](protocol_v21.md)。除明确标为已完成的补跑外，其余仍是待办。

## 8. 持续维护与纠错

### 阶段 057：Repaired benchmark 配对基线与定向 SFT（2026-09-21）

为避免把“新测试集容易”误判成模型提升，在同一份 frozen `transaction_repaired_v1` ID/OOD（各32条）上补跑原始 full-SFT baseline：

| Adapter | ID task success | OOD task success | 平均调用 | policy violations |
|---|---:|---:|---:|---:|
| Original full SFT | 0/32 | 0/32 | 36.0 | 497 / 494 |
| Repaired targeted SFT | 32/32 | 32/32 | 5.0 | 0 / 0 |

两者 execution success 均为 32/32，因此差异不是工具执行器可用性，而是模型是否在 `permission_denied` 后停止、读取最终状态并报告证据。原始 SFT 会重复 `inspect_workspace`/`stage_config` 直到 36-call 上限；repaired targeted SFT 在 5 次调用内正确报告 `blocked`。这是目前第一组严格配对的明显正结果：ID/OOD 均提升 100 个百分点，平均调用减少31次。

该结果的边界已冻结：只有64条 repaired 确认评测，且 targeted SFT 使用了 repaired train，因此不能外推为所有旧任务的泛化提升。后续必须检查旧 v1 confirmation 的回归，再扩展 repaired challenge set。完整证据见 [阶段057](experiment_logs/057_repaired_sft_paired_baseline.md)。

仓库 [AGENTS.md](../AGENTS.md) 固化阶段日志与总报告同步要求；新阶段按 [模板](experiment_logs/_template.md) 记录，保持失败与设计变更理由。已有用户动机文档不改写。

2026-09-10 补录：助手曾在本轮口头误报 v1 匹配为 4/6，已核对为 3/6；见 [阶段 001 勘误](experiment_logs/001_exploratory_baseline.md)。这属于记录过程错误，不更改原始模型数据。
