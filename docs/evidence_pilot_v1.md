# 工具历史审计子能力：数据与SuperPOD流程

本实验不是第二版完整Agent，也不是旧测试题改写训练。模型读取新构造的工具历史，输出结构化JSON报告和后续处理类别，不实际执行后续动作。目标是隔离失败计数、状态/验证证据与重试判断能力。题面公开输出字段和分类规则，所以主要测规则执行与证据提取，不测自主发现重试策略。

288条、48个故障类型×验证字段组合组：192训练/32组、48验证/8组、48确认/8组。每组两平台×三画质。按完整组隔离，表述模板和工具共享；不是288个独立业务场景。旧Agent题及其答案不用于生成；精确旧query重叠为0，不能据此宣称语义完全独立。

模型只监督最终JSON，历史失败调用放在user证据中，不训练其错误动作。沿用旧SYSTEM_PROMPT、TOOLS和assistant-only编码器；确认数据不导出训练轨迹。

三种模型使用同一组96条验证/确认问题：原Base、旧pilot LoRA、新子能力LoRA，共288次单轮推理。新LoRA从Base初始化，不继续旧adapter；仅192条报告数据，1epoch、预计24优化步，r16/alpha32/dropout.05、lr1e-4、batch1/累积8、seed20260915，不packing、不截断。两步GPU smoke独立保存，不作为正式训练成绩。

主指标：完整JSON字段、数值、类型全部正确（键顺序无关，拒绝重复键和额外文字）。辅指标：有效JSON、decision正确。不是旧Agent task_success，不能直接比较；自动标签来自实际工具返回的确定性字段，不冒充独立人工语义复核。

## 登录节点

    cd /home/zshaoaj/agent-post-training-lab
    git pull --ff-only
    conda activate agent-post-training
    python -m scripts.prepare_evidence_pilot --verify
    srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=02:00:00 --pty bash

## GPU节点

    cd /home/zshaoaj/agent-post-training-lab
    conda activate agent-post-training
    nvidia-smi
    bash scripts/superpod_evidence_pilot.sh

顺序：实际权重/版本/分词预检→Base96题→旧LoRA96题→独立2步smoke→新LoRA1epoch→新LoRA96题。正式训练前要求前两模型完整证据，不根据确认分数改超参数。2小时是申请上限，不是耗时保证。

分词预检不得超过4096上下文，评测预留512生成token；失败立即停。新adapter在checkpoints/evidence_pilot_v1/adapter，旧adapter不改。推理可继续完整行结果；末行截断会明确报错，不自动修复。训练中断保留目录并停止，先诊断，不自动覆盖重训。真实运行入口尚待本次上机验证。

## 所有阶段完成后

    exit
    cd /home/zshaoaj/agent-post-training-lab
    git add results/evidence_pilot_v1/
    git commit -m "Add evidence subskill pilot training and evaluation results"
    git push origin main

结果目录包含三模型JSONL、分词预检、smoke及正式训练元数据（含trainer_state和adapter保存哈希），不用上传大权重。报错请提供完整输出，不绕过指纹或删除旧记录。
