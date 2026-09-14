# 012 — 首次SuperPOD启动失败与命令修复

## 状态与证据

2026-09-14现场记录。用户在dgx-09提供NVIDIA H800、81559MiB显存、驱动570.158.01/CUDA12.8的nvidia-smi输出；启动脚本先显示bundle verified，随后显示NVIDIA H800及run_baseline.py: error: unrecognized arguments: + + +。这是首次4B管线启动失败，不是模型或训练失败。

## 实验目标与配置

运行阶段011的pilot_v1。此次定位并修复启动器，不改2.3、训练超参数、基础权重或数据。GPU已分配且torch CUDA探测成功；实际导入等待时间未记录，不猜测CUDA卡死。模型推理与训练尚未开始。

## 数据集、指标与结果

320/80/80与全部数据SHA不变；无新推理结果、loss或任务率。旧模拟smoke仍仅是CPU小模型验证。新增启动参数测试后共122项测试通过，尚需用户在当前GPU节点重新启动确认。

## Bad case 与原因分析

生成Shell多行命令时，JavaScript模板字符串里的续行反斜线吞掉换行，patch行首的加号进入了命令正文。六条命令均含独立“+”参数。Shell允许这种普通参数，所以bash -n通过；Python argparse才拒绝。之前120项测试和独立CPU训练未实际执行这个Shell入口，导致交付遗漏。责任属于助手生成与测试，不是用户操作。

早先“verified之后无输出”处于torch导入/检查阶段；后续用户输出证明GPU检查成功。多余参数是已证实失败原因，不能把等待阶段推测为GPU故障。

## 解决方案与设计变更

改为单行完整命令，避免续行转义；从脚本路径定位仓库；加入7阶段提示、torch导入/CUDA检查无缓冲提示，日志统一追加并保留set -euo pipefail。无权重或环境安装变更。

保持数据版本pilot_v1，显式发布bundle_revision=2启动器工程修订，记录revision_reason和previous_manifest_sha256；原manifest保存在results/training_preflight/manifest_before_launcher_fix.json。只更新启动器/打包脚本代码哈希，数据、训练器、评估及模型运行指纹均未变。此前CPUsmoke继续绑定旧manifest，不伪称新训练测试；未绕过校验或重写旧实验结果。

## 验证

新增tests/test_pilot_launcher.py：在临时仓库实际执行Shell，以替身python记录argv，四条评测命令交给真实argparse解析；检查训练参数、阶段输出和8次调用顺序。故意令Base命令返回2，验证管线立即停止、不会启动训练。替身不加载模型，不冒充GPU执行。

数据准备可重复、manifest验证成功、122项测试通过、git diff --check通过。

## 下一步与恢复

用户仍在已分配GPU节点时，git pull --ff-only后再次运行bash scripts/superpod_train_pilot.sh即可；不需要重新申请GPU或重新安装依赖。若已经退出GPU allocation则需重新申请。旧错误发生在argparse，没有成功的Base记录/训练checkpoint需要恢复；日志保留。若出现其他错误，保留输出继续诊断，不删除历史文件。

## 追加勘误

阶段011“可开始训练”的交付包含未测到的启动器缺陷；小模型训练成功和Shell语法通过不能证明完整入口可用。后续必须包括实际argv/入口检查。
