# 034 — 结构隔离多轮数据与GPU工程检查交接

## 状态与证据

2026-09-16，本地正式数据/接线完成，到达首次新任务GPU协助边界。非回顾补录，无本阶段模型结果。用户要求不等逐步确认、持续推进，并希望实验有质量；本阶段保持既定可复现/不筛负结果原则。

源提交cd91f8f。新增training/transaction_data.py、transaction_sft.py、agent/transaction_model.py、准备/工程检查/启动脚本、tests/test_transaction_data.py，数据data/transaction_v1/。manifest记录文件和执行代码SHA、基础模型与依赖版本、运行预算；第一份未发布manifest保留在results/preflight/transaction_v1_manifest_draft.json。

## 目标与配置

从单轮报告回到自主多轮工具交互，先固定分组与契约，再确定真实分词/显存/吞吐。八事务工具与观测型teacher沿用阶段033，新模型backend显式使用新TOOLS而非旧页面工具；不修改旧冻结运行代码。

正式数据seed20260917（固定整数种子，不表示实际运行日期）；max_turns40、max_calls36、上下文8192、输出512、greedy。基础权重和torch/transformers/peft/accelerate版本沿用已验证来源。当前本地确无这三项模型依赖（torch/transformers/peft）及4B权重，GPU型号、token总量和真实耗时尚待上机。

首次GPU仅14题Base→2步LoRA→14题smoke-adapter；r16/alpha32/dropout0.05/all-linear、lr1e-4、batch1/accumulation8。最长16个train预测前缀用于压力smoke，8个最长dev前缀用于loss；不是完整训练，也不能将该adapter继续用于正式SFT。正式训练的epoch/optimizer预算需在工程检查后、正式模型比较前冻结。

## 数据集与构造

96个可写故障结构（暂时失败0/1/2×超时0/1×并发0/1×检查成败×延迟0/1/2/3），加初始只读和提交前撤权2组，共98。超时且并发的24组全部留给OOD；其余72组按检查成败分层哈希划为48train/16dev/8确认ID。初始只读放train，提交前撤权放OOD。

每组从6初态配置×6目标配置（排除相同）×revision7/19共60配置变体中确定性取样。train每组42，其他每组16；四种公开请求措辞在组内分配。每条history由工具实际执行，teacher只根据返回决定恢复，不读取隐藏故障flag。

| 分区 | 任务/组 | assistant预测例子 | oracle结局 | 工具调用范围 |
|---|---|---:|---|---|
| train | 2058/49 | 31542 | applied1008 / rolled_back1008 / blocked42 | 2–23 |
| dev | 256/16 | 3936 | applied128 / rolled_back128 | 9–21 |
| confirmation_id | 128/8 | 2144，仅审计不导出训练 | applied64 / rolled_back64 | 11–19 |
| confirmation_ood | 400/25 | 7392，仅审计不导出训练 | applied192 / rolled_back192 / blocked16 | 5–24 |

所有2842条oracle通过公开任务契约和重放，组、规范化实际工具/错误/状态序列、scenario+goal均不跨split。额外对98组各一条进行序列化assistant→runner回放，保证训练工具消息与执行路径一致。以上均是程序测试，不是模型成绩。

明确限制：仅24种公开请求在不同环境下重复，不是2842独立语言任务；工具集与基本业务流程仍窄。ID是同机制范围下的结构组合留出，不意味着原始独立业务采样。OOD的撤权只有1组，dev没有permission，不得用整体率掩盖不均衡。旧评测题没有读取进新生成器；阶段033工程fixture用于开发环境，不作为声称未见过的新模型结果。

## 指标与产物用途

主指标仍为execution_success、policy_violations、answer_correct以及三者约束的task_success，附实际调用数/副作用数/终止原因/生成tokens与时间。正式确认ID和OOD分开，applied/rolled_back/blocked分开；blocked不是业务配置应用成功。

smoke14题来自12个dev组各1例及2个train初始只读探针，具有工程选择偏差，不作为正式baseline或独立held-out发布。确认题此轮只在本地/分词预检中走规则oracle，不执行模型推理，不能用于训练/选模型。

## Bad case、问题与修复

1. 多轮模型可能比oracle绕更长历史，不能只验证teacher长度。新增运行时ContextBudgetExceeded明确记录context_budget终止，保留已有事件、不截断证据；其他真实实现错误仍抛出。
2. 新schema与旧SFT encoder绑定不兼容。新增独立编码器，显式验证新system/tools，只监督assistant消息目标+EOS，工具返回是输入不是loss；拒绝confirmation分区编码。假tokenizer仅证明遮罩逻辑，真实分词待GPU。
3. 约2000轨迹会展开成31542个预测样本，GPU工作量不能按“2000条短文本”估计。采用流式分词统计，只保留最长16/8例，避免预检将所有长序列堆内存；需实测后再确定完整训练预算。
4. 仅存smoke样本hash不方便后续核验。冻结前补充逐样本case_id、assistant_index和长度引用，同时记录训练峰值显存和每5回合进度，保留初稿manifest。数据/标签/切分未改，无模型运行或分数受影响。
5. 续跑检查采用保留JSON类型的比较，拒绝把task_success=true改成数字1，防止Python宽松比较掩盖篡改；检查case顺序、元数据、事件重放、指标和模型回合/最终回答一致。
6. “漂亮实验”不能通过丢弃失败或把相关变体称成独立样本实现。规模、公开请求重复、组数、类别不平衡、smoke目的均明确记录；暂不写任何模型提升结论。

## 验证

新增12项测试覆盖冻结规模/结构审计、OOD留出、确定性和去重、98组序列化回放与隐私输入、smoke不读取确认、完整多轮目标mask、确认编码拒绝、上下文安全终止、新backend工具/schema长度检查、续跑重放与类型漂移、写保护及Shell失败停止。当前完整测试结果提交前追加。没有在本地运行真实4B、真实分词或训练。

提交前271项unittest全部通过（新增12项），最终manifest重建验证、Shell语法检查与git diff --check通过。正式数据2842条规则轨迹审计通过，98组串行模型接口替身回放通过；均非真实模型能力结果。

## 下一步与用户操作

现在需要一次SuperPOD工程检查，按[完整指令](../transaction_v1.md)进入实验目录、更新、申请1小时GPU，再运行scripts/superpod_transaction_smoke.sh。20–60分钟仅规划估计，不含排队，不保证；真实成本由本次结果决定。

拿到结果后继续：核对分词/显存/吞吐/运行正确性，冻结完整SFT预算、补恢复示范消融计划，跑正式Base与SFT；随后训练侧偏好采样/DPO，再在线GRPO与独立后端。没有把工程检查当成正式训练完成，也没有将其作为整个项目收尾。
