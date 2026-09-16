# Transaction v1：正式数据与首次GPU工程检查

这次不是重复旧报告实验，而是让模型实际执行八种事务工具，依据每轮返回处理故障、检查和回滚。当前交接只验证新链路能否运行，不启动完整SFT、DPO或GRPO。

## 已冻结数据

| 分区 | 任务数 | 结构组 | 用途 |
|---|---:|---:|---|
| train | 2058 | 49 | 完整多轮SFT候选数据、后续训练侧采样 |
| dev | 256 | 16 | 开发诊断与超参数选择 |
| confirmation_id | 128 | 8 | 同一机制范围内未见结构组合 |
| confirmation_ood | 400 | 25 | 未训练的超时+并发组合，及提交前撤权 |

合计2842任务、98组；配置值/revision/措辞变体不跨结构组切分。训练展开31542个assistant下一轮预测例子，不是31542个独立任务。OOD的384条是24个复合故障组，另16条来自1个撤权组，必须分开解释。

公开请求只有24种（6种目标配置×4种表述），同一请求对应不同隐藏状态/故障环境。这是受控交互实验，不能宣传成2842种不同业务需求。组隔离检查比较实际oracle工具名及返回状态/错误序列，避免仅通过改ID/配置实现假隔离；仍共享小工具集和流程，不证明广泛语义独立。

初始只读42条只出现在train，提交前撤权16条只出现在OOD；dev没有permission类别。后续指标必须按任务类别/ID/OOD分别报告，不能以多数类掩盖权限问题。正式确认集此轮不做模型推理；预检只检查其规则轨迹能否容纳在上下文中。

## 为什么先跑工程检查

新环境、schema、system、数据编码和模型接口都不同于旧pilot。本地已完成2842条oracle与结构审计，但没有torch/transformers/peft、4B权重或GPU，不能声称真实分词/训练已通过。

运行顺序：

1. 校验数据/代码/权重/依赖，真实分词全部train/dev预测样本，并检查dev/确认oracle提示长度。8192上下文，512输出预算，不截断。
2. Base执行14个工程任务：12个dev组各1例，另2个train初始只读探针。不是正式baseline，更不是独立测试成绩。
3. 从train选择16个最长assistant前缀，做两步LoRA；从dev选择8个最长前缀检查loss。记录样本引用、训练耗时和峰值显存。此选择仅为长度/内存压力测试，不用于估计正常训练效果。
4. 保存并重新加载smoke adapter，再跑同14题，验证整个保存/推理流程。

正式训练仍应重新从Base开始，不能把这个两步smoke adapter继续训练成正式SFT。完整预算待真实token与吞吐出来后冻结；确认集不参与预算或模型选择。两步适配器即使分数低也不代表正式SFT失败。

## 登录节点

```bash
cd /home/zshaoaj/agent-post-training-lab
git pull --ff-only
conda activate agent-post-training
python -m scripts.prepare_transaction_v1
srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=01:00:00 --pty bash
```

## 获得GPU后

```bash
cd /home/zshaoaj/agent-post-training-lab
conda activate agent-post-training
nvidia-smi
bash scripts/superpod_transaction_smoke.sh
```

这是1小时申请上限，不是耗时保证。规划估计20–60分钟，主要看真实分词和模型是否循环；尚无本环境实测吞吐。每200训练轨迹输出分词进度，每100评估任务输出长度检查进度；推理每题及每5回合输出进度。不要因读取/校验权重暂时无输出就直接终止。

确认当前python仍是原环境，版本不符时脚本会停止；不要现场升级。动态模型历史超长时记录context_budget而非截断或伪造答案。其他加载/训练异常会停止整个流程并保留错误，不作为模型任务失败悄悄吞掉。

## 完成后上传

看到Complete再退出GPU：

```bash
exit
cd /home/zshaoaj/agent-post-training-lab
git add results/transaction_v1_smoke/
git commit -m "Add transaction Agent GPU readiness results"
git push origin main
```

目录应含preflight.json、base.jsonl（14行）、training_run.json、sft_smoke.jsonl（14行）。本地接着核对实际token/峰值显存/吞吐/错误，为正式SFT给下一份预算和指令。保留checkpoints/transaction_v1/及logs/transaction_v1_smoke/，不要上传大权重。

若失败或时限中断：保留输出和已生成结果，先反馈错误。完整推理记录可续跑；中断训练、损坏行、不同指纹会拒绝覆盖，不能删除重跑掩盖失败。没有全部文件时不宣称工程检查完成。
