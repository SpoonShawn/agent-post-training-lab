# 完整事务 Agent：第一轮正式 SFT

这是完整路线的首个单seed SFT，不是最终实验，也不是继续两步smoke训练。协议见 `data/transaction_v1_full_sft_plan.json`；数据、评分、工具环境沿用冻结transaction_v1，不修改旧结果。

## 已冻结的比较

Base → 全部2058训练轨迹SFT → 相同784任务推理。784=开发256+确认ID128+确认OOD400，分别报告，不混成一个泛化结论。确认只评价，不参与训练或checkpoint选择；过程中不根据确认分数改参数。训练1epoch，31542个assistant预测样本，batch1/累积8，预期3943优化步；r16/alpha32/dropout0.05/all-linear、lr1e-4、AdamW、恒定学习率、无warmup、seed20260917、bf16/梯度检查点。固定最后checkpoint，不按确认成绩选优。输入52593783 tokens（包括重复历史），监督663355 tokens；不是5259万独立训练tokens。

上下文8192/生成512/40回合/36工具调用/greedy。无静默截断，只对assistant目标计算loss；全量dev仅在训练后计算loss。磁盘SQLite缓存避免所有前缀以Python整数列表常驻内存。每250步保存完整恢复状态，不自动删除历史checkpoint；预留至少30GB可用磁盘较稳妥。

工程检查H800两步17.26秒，线性外推3943步约9.45小时，但最长前缀、初始化、正式平均长度不同，**不能当准确ETA**。规划训练5–10小时，分词/两轮推理/保存另预留1–3小时；先申请12小时，排队另算。超时可重新申请后重跑同脚本：已有完整推理记录核验后跳过，训练恢复最近完整checkpoint（最多重做249步）。第一次checkpoint之前中断、损坏JSON行或未完成缓存发布将停止并要求诊断，不删除文件。CUDA/Trainer真实恢复仍待本次运行验证，不能承诺bitwise完全一致。

恢复接口依据[Hugging Face Trainer文档](https://huggingface.co/docs/transformers/main_classes/trainer)，依赖仍锁定已实测版本，不自动升级。训练中断尝试和错误保留在training_run.json；末次Trainer耗时可能只覆盖恢复后的部分，不能直接称总GPU耗时。

## SuperPOD指令

登录节点：

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python -m scripts.analyze_transaction_smoke
df -h .
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=12:00:00 --pty bash
```

成功进入GPU节点后：

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
bash scripts/superpod_transaction_full_sft.sh
```

阶段顺序：`base`为784题推理，`train`才是完整LoRA训练，`sft`为新模型784题推理。出现最终完成提示后退出GPU并上传；不要训练运行中输入exit。若资源限时不允许12小时，保留报错交回，不自行改变训练步数。

```bash
exit
cd /home/zshaoaj/agent-post-training-lab
git add results/transaction_v1_full_sft/
git commit -m "Upload transaction full SFT first-seed results"
git push origin main
```

不上传checkpoint/token缓存，不删除它们；后续DPO/GRPO还要用正式SFT。若报错，上传已产生的结果并提供末尾错误。不要运行`--freeze`来绕过指纹不一致。

## 后续与边界

第一轮结束后先完整重放审计，记录成功和退化；随后训练侧真实偏好采样/DPO、在线GRPO、关键设置多seed与独立后端。恢复示范消融是单独的未来实验臂，必须先冻结训练侧干预及等预算定义，不用当前单臂证明恢复示范的因果贡献。新任务仍是窄域合成事务环境，仅24个公开请求模板/目标组合；98结构组不是2842个独立业务场景。
