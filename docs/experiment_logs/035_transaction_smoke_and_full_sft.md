# 035 — 工程结果审计与首轮完整SFT冻结

## 状态与证据

2026-09-16；收到用户上传931bc5d（运行代码fc11520）。当日记录，非事后补写目标。原始四文件在results/transaction_v1_smoke；新增分析summary.json/bad_cases.json位于results/analysis/transaction_v1_smoke，记录逐文件SHA。完整SFT尚未运行，需要SuperPOD。新代码/计划与此日志同提交，旧冻结代码不修改。

## 目标、配置与数据

验证阶段034新工具链真实GPU分词、训练、adapter重载；工程样本14题=12个dev组各1题+2个train只读探针，无确认模型推理。最长16个train下一轮前缀、8个dev前缀loss；全部来自同一train结构组/同一dev结构组，且都是最终报告回合。因此两步训练**没有覆盖多轮动作监督**，不能当正式SFT失败或成功。源代码是按长度挑选，引用证明这一偏差；无需改smoke选样，它的目的是压力检查，正式训练用全量。

Qwen3-4B-Instruct-2507，基础权重SHA8a3508ed…f68c，数据manifest4978ff7e…6ea，完整哈希见机器产物。H800，torch2.9.1/transformers5.16.1/peft0.20.0/accelerate1.14.0。r16/alpha32/dropout.05/all-linear、lr1e-4、batch1累积8、两步、seed20260917、bf16。原始训练开始06:12:45UTC，结束06:13:15UTC；Trainer训练17.2601秒，不能把30秒整个窗口全称训练耗时。

## 指标与结果

28/28记录逐事件重放、工具回合、最终答案、指标和指纹一致。Base与两步adapter各：执行2/14、答案0/14、完整任务0/14、权限违规0；执行通过仅两个只读探针，不意味着配置成功。Base116次工具调用、两步128次；完整失败28条全部保留，非独立正式benchmark。

训练loss1.16126、8例dev loss0.50264；无训练前同例loss，不能声称loss改善。峰值allocated13.25765GiB/reserved14.59961GiB；两步正常、adapter重载输出正常，只证明工程可用。Base生成计时合计69.19秒，两步134.15秒，样本太小且路径不同，不作为正式加速/变慢结论。

真实分词：train31542例、总输入52593783tokens、监督663355、最长2743；dev3936例、总输入6476996、监督82872、最长2620。oracle评测最长prompt2765+512<8192；真实模型仍可能绕路超长，保留context_budget失败，不截断。分词/显存来自上传GPU测量，本地没有重新运行tokenizer或权重；未上传adapter及私有token缓存只核验关联哈希，不能宣称本地验过内容。

## Bad case与原因

Base txn1_44ec49d9e1fb0aff_00：inspect初始revision19/config standard+cache true；stage/validate/commit成功到revision20/high+cache false；start_check、两次poll最后failed。然后直接报告rolled_back、revision19和旧配置。**原始7次调用没有rollback或末次inspect**，实际配置仍为新配置；还把失败数写成数组。执行错误与报告捏造同时存在，不是只因严格JSON不通过。

Base txn1_25e90230abc8030f_00只读时不写入，执行正确，但只答blocked而非契约的完整报告；这是执行与答案分层的正例，不应改成完整任务成功。其他案例含暂时故障后提前停止、未完成异步检查等；原轨迹全部在bad_cases中。

已证实：工程无OOM、28题无预算截断；动作/证据/报告失败是真实。待验证：完整多轮SFT能否改善；不把两步负结果解释成训练数据太少导致过拟合，证据不足。

## 解决方案与设计变更

保持环境/评分/数据不变，新增正式脚本与独立协议，不把smoke改成正式成绩。根据实测冻结1epoch/3943步，fresh Base初始化、不接smoke adapter；全部2058轨迹，31,542预测例子。Base/SFT各开发256、确认ID128、OOD400，固定最终checkpoint，不看确认调参。与两步工程检查不是等预算对照，不发布二者增益。

输入缓存用标准SQLite逐例落盘，避免约5259万输入tokens以三套Python列表占用大量内存；GPU重分词汇总必须与预检一致。训练每250步保存模型/优化器/调度器/RNG，可续跑；保留中断尝试，异常停止。完整配置、来源指纹见data/transaction_v1_full_sft_plan.json。没有引入DPO/GRPO实现，更没有把普通SFT称成它们。

## 验证

本地新增5项工程重放、缓存读写/边界/错误统计拒绝发布、完整/不完整checkpoint选择、预算及分区测试。完整276项unittest通过（17.4秒），正式计划重建核验、Shell语法检查及git diff --check通过。真实GPU完整训练、真实恢复、正式模型成绩未验证；无评测标准修改、无丢弃旧失败。

## 下一步

按[完整操作指令](../transaction_full_sft_v1.md)申请12小时单GPU完成Base→fresh完整SFT→SFT评测。预计训练5–10小时，其他1–3小时只是规划区间，排队另算；最长前缀两步直接线性外推约9.45小时不能当准确ETA。用户上传后完整审计，再推进训练侧偏好采样和DPO。全部研究路线仍按3–6周规划，不保证性能上涨。单seed、窄域、公开请求重复、旧独立复审未完成等限制继续保留。

## 追加记录／勘误

无旧成绩勘误。本阶段证明的是工程就绪，不是模型提升。
