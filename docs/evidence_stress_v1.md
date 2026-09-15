# 报告模型压力测试

本轮只推理，不训练。固定原Base与已训练的报告LoRA，各96条，共192次单轮报告生成。旧事务LoRA此前96条均未输出报告，本轮不重复测它；这不是将其旧失败删除。

24个已看过的android确认上下文，分别形成四组：

| 条件 | 条数 | 检查什么 |
|---|---:|---|
| 原题对照 | 24 | 同运行流程的参考成绩，亦可检查与旧结果是否复现 |
| 证据呈现变化 | 24 | 逐条输入/返回展示，要求移到证据后；事实和目标不变 |
| 增加历史步骤 | 24 | 额外验证和日志；可写场景返回首页，最终状态与最后验证不能照抄旧值 |
| 当前权限翻转 | 24 | 历史不变、当前权限更新；只改变decision，不抹掉历史失败 |

四组共享24个上下文，来源8个组合组，不是96个独立业务场景，也不是新的独立确认集。只做训练后诊断，不回灌训练。主比较是每模型内原题→三个扰动的成对变化，以及同条件Base→报告LoRA。分别报告格式、字段和decision错误，不只看总体率。

输出字段和评分沿用冻结报告任务；分类规则仍在题面公开，不把decision当成完整自主恢复能力。所有额外历史由模拟器真实执行，模型输入只含题面及工具证据，不含target或后台环境。实际分词+512生成预算不得超过4096，否则停止而非截断。

## 登录节点

    cd /home/zshaoaj/agent-post-training-lab
    git pull --ff-only
    conda activate agent-post-training
    python -m scripts.evidence_stress_v1 --verify
    srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=01:00:00 --pty bash

## 获得GPU后

    cd /home/zshaoaj/agent-post-training-lab
    conda activate agent-post-training
    nvidia-smi
    bash scripts/superpod_evidence_stress.sh

1小时是申请上限，不是耗时保证。每个模型加载前校验权重、报告adapter、数据/代码指纹和实际分词长度。支持按完整记录续跑；完整结果不会重复加载模型。末行截断/指纹错误会停止，请保留输出交由诊断，不删文件或绕过校验。

## 两个模型都完成后

    exit
    cd /home/zshaoaj/agent-post-training-lab
    git add results/evidence_stress_v1/
    git commit -m "Add evidence report stress-test inference results"
    git push origin main

结果含原始答案、逐题指标、运行指纹和时间。旧训练/评测不覆盖，新数据位于benchmarks/evidence_stress_v1，独立于训练候选数据目录。
