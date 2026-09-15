# 只读执行保护：GPU验证

不训练、不改提示/schema/步数/evaluator。固定原16条只读题，Base与原LoRA各16条，共32次新推理。与已有control对应题比较，是诊断复用而非独立测试集。

唯一在线变化：可信应用设置read_only=true时，每个工具调用在执行前经过只读白名单检查，禁止工具返回policy_blocked给模型。策略不是模型设置，也不读取隐藏成功答案。原模型回答保留；系统事实摘要在运行结束后生成，不反馈模型、不替换原回答评分。

分别观察禁用调用尝试、拦截次数、成功副作用、状态/日志取证、正常终止和答案忠实性。冻结2.4仍惩罚禁用操作尝试，即使拦截也不改判。安全保护不等于模型能力提升。

manifest固定旧control包、权重和新增代码。不要用旧control汇总器读取本实验，它会拒绝选择/指纹错配。上传后使用受保护执行器重放，不能用未保护executor重放。

## 登录节点

    cd /home/zshaoaj/agent-post-training-lab
    git pull --ff-only
    conda activate agent-post-training
    python -m scripts.run_guard_pilot --verify-only
    srun --account=mscaiesuperpod --partition=normal --gres=gpu:1 --time=01:00:00 --pty bash

## 分配到GPU后

    cd /home/zshaoaj/agent-post-training-lab
    conda activate agent-post-training
    nvidia-smi
    bash scripts/superpod_guard_pilot.sh

1小时为申请上限，不是耗时承诺。哈希/加载可能暂时无输出。支持续跑，Base失败立即停止，不覆盖旧结果。不要删除旧结果或重训。

## 两模型完成后

    exit
    cd /home/zshaoaj/agent-post-training-lab
    git add results/baseline/guard_pilot_v1_base.jsonl results/baseline/guard_pilot_v1_sft.jsonl
    git commit -m "Add read-only guard pilot inference results"
    git push origin main

如报错，保留完整输出，不绕过指纹/版本校验。
