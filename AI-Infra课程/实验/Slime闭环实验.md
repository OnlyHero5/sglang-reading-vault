---
title: "Slime 闭环实验"
type: exercise
framework: slime
topic: "RL 后训练"
learning_role: practice
source_baseline: "22cdc6e1"
difficulty: intermediate
estimated_time: "90 到 180 分钟"
prerequisites:
  - "[[RL训练闭环主线]]"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---

# Slime 闭环实验

## 读者任务

把 rollout、训练重放和权重同步拆成三个可独立验收的边界，证明同一批 Sample 能先被落盘、再在不启动 SGLang 的情况下训练，并能区分训练 step、`Sample.rollout_id` 与 `weight_versions`。

## 环境分级

| 模式 | 环境 | 能证明什么 |
|------|------|------------|
| 静态契约 | 当前知识库 + Python 标准库 | debug flag、保存/加载路径和版本字段仍存在 |
| 轻量测试 | Slime 基础依赖 | 参数约束与部分 server argument 规则 |
| 两阶段 E2E | 隔离 Linux GPU 节点、Ray、Megatron、SGLang、HF 下载能力 | rollout-only 落盘和 train-only 重放真的闭环 |

当前 upstream 的两阶段测试 `tests/test_qwen2.5_0.5B_debug_rollout_then_train.py` 实际设置 `NUM_GPUS = 8`；文件开头“2 GPUs”的说明已经漂移，不能据此准备资源。该测试还会停止 Ray，并按进程名清理 sglang/slime/redis，只能在隔离节点运行。

## 静态模式

```powershell
rg -n 'debug_rollout_only|debug_train_only|save_debug_rollout_data|load_debug_rollout_data|weight_version' slime/train.py slime/slime
```

预期：能找到 debug 分支、Sample 保存/加载和权重版本透传位置。

再用标准库核对 canonical E2E 测试的真实资源和两阶段参数，不导入 Slime：

```powershell
@'
import ast
from pathlib import Path

p = Path("slime/tests/test_qwen2.5_0.5B_debug_rollout_then_train.py")
tree = ast.parse(p.read_text(encoding="utf-8"))
constants = {
    n.targets[0].id: ast.literal_eval(n.value)
    for n in tree.body
    if isinstance(n, ast.Assign)
    and len(n.targets) == 1
    and isinstance(n.targets[0], ast.Name)
    and isinstance(n.value, ast.Constant)
}
text = p.read_text(encoding="utf-8")
print({k: constants.get(k) for k in ("MODEL_NAME", "NUM_GPUS", "NUM_ROLLOUT")})
print("rollout_only=", "--debug-rollout-only" in text)
print("save=", "--save-debug-rollout-data" in text)
print("load=", "--load-debug-rollout-data" in text)
'@ | python -
```

预期：输出模型 `Qwen2.5-0.5B-Instruct`、`NUM_GPUS=8`、`NUM_ROLLOUT=2`，并且三个布尔值均为 `True`。

## 轻量参数验收

从 `slime/` upstream 目录执行：

```powershell
python -m pytest tests/test_megatron_argument_validation.py tests/utils/test_megatron_server_arguments.py -q
```

预期：目标参数测试全部通过。记录实际 collected/passed 数，不把当前数量写成永久契约。它只证明参数约束，不证明 Ray placement、GPU rollout 或 Megatron 初始化；缺 `ray` 时不要追加 `tests/test_placement_group.py` 并把收集失败误判成代码错误。

## 两阶段 E2E

前提：隔离 Linux 节点、8 张可用 CUDA GPU、可工作的 Ray/Megatron/SGLang 环境，以及下载模型和 GSM8K 的权限。从 `slime/` 目录执行：

```bash
python tests/test_qwen2.5_0.5B_debug_rollout_then_train.py
```

脚本会自动完成：

```text
下载 Qwen2.5-0.5B-Instruct 与 GSM8K
→ Phase 1: --debug-rollout-only
→ 保存 rollout_{rollout_id}.pt
→ Phase 2: --load-debug-rollout-data
→ 不启动 SGLang，重放两轮训练
```

Phase 1 记录 Sample 的：

- tokens / response length
- loss mask
- reward
- rollout logprob
- `rollout_id`
- `weight_versions`

预期：生成 `rollout_0.pt`、`rollout_1.pt` 等可检查文件，但 actor 不进入 optimizer step。文件顶层保存 `rollout_id` 和序列化后的 `samples`。

## Train-only 重放

canonical 脚本的第二阶段通过 `--load-debug-rollout-data <path>/{rollout_id}.pt` 自动令 `debug_train_only=True` 并跳过 SGLang。固定脚本与 checkpoint 后，记录 loss、KL、advantage 统计和 gradient norm。

预期：不调用 SGLang 生成也能重现训练侧问题；Sample 缺少必要字段时应在明确边界失败。

## DP split 检查

打印或断点检查每个 DP rank 获得的 rollout ids、micro-batch indices 和 global batch size。

预期：需要 group baseline 的 response 保持正确分组；rank-local 数据总和能还原全局 batch。

## Weight version 检查

完整正常闭环可启用 `--check-weight-update-equal` 检查 initial weight push 前后的 snapshot/compare；当前同步和异步入口只在首次 push 后调用 compare，不能把这个 flag 说成每轮自动证明。每轮更新还要记录 updater/engine version、CI version checker 或额外 instrumentation，并对照下一轮 Sample 的 `weight_versions`。不要在两阶段 train-only 重放中期待 SGLang version 更新，因为该阶段根本不会实例化 rollout engine。

故意遗漏某个 engine 的故障演练只允许在隔离环境进行，并应同时记录：该 engine 是否在 updater 的目标集合中、更新调用是否返回、下一轮终态 metadata 的版本列表。

预期：系统应明确报告 engine 被跳过或版本不一致，而不是静默继续产生旧版本样本。

## 通过标准

- [ ] Rollout-only 与 train-only 可独立解释。
- [ ] 能从 Sample 追到 rank-local RolloutBatch。
- [ ] 能手工检查一组 reward 到 advantage 的方向。
- [ ] 能证明下一轮 rollout 使用了更新后的权重版本。
- [ ] 能解释 canonical 测试为何要求隔离 Linux 节点，以及为什么其“2 GPUs”注释不能作为当前资源依据。
