---
title: "RolloutManager · 学习检查"
type: exercise
framework: slime
topic: "RolloutManager"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# RolloutManager · 学习检查

## 读者能做什么

- [ ] 能画出 `train.py → RolloutManager.generate → rollout_fn → Sample → train_data → build_dp_schedule → Box(ray.put) → async_train`。
- [ ] 能解释 RolloutManager 为什么是 `num_gpus=0` 的 Ray Actor。
- [ ] 能区分 `generate(rollout_id)` 参数和 `Sample.rollout_id` 字段。
- [ ] 能说出 `tokens/rewards/loss_masks/rollout_ids/rollout_mask_sums/partition/micro_batch_indices` 的来源和用途。
- [ ] 能说明 `raw_reward/total_lengths` 为什么保留全局列。
- [ ] 能解释 `get_updatable_engines_and_lock` 为什么排除 frozen 模型。
- [ ] 能说明可变 fanout 默认 normalization、尾部 rollout 取整和 `balance_by_flops` token cap 三个边界。

## 源码定位验收

| 问题 | 入口 |
|------|------|
| 训练主循环如何调用 rollout | `train.py::train` |
| RolloutManager 如何创建 | `create_rollout_manager` |
| 初始化加载哪些插件 | `RolloutManager.__init__` |
| generate 四阶段 | `RolloutManager.generate` |
| 兼容旧 rollout 返回 | `call_rollout_fn` |
| debug 复放 | `_get_rollout_data` |
| compact rollout id 校验 | `_validate_rollout_id_annotated` |
| reward 后处理 | `_post_process_rewards` |
| Sample 转 train_data | `_convert_samples_to_train_data` |
| DP schedule | `build_dp_schedule` |
| per-rank ObjectRef | `_split_train_data_by_dp` |
| 权重更新 engines + lock | `get_updatable_engines_and_lock` |

## 失败模式验收

- [ ] compact/subagent 输出多个 sibling 但没设 `rollout_id`：能指出校验函数和修法。
- [ ] unique rollout 数小于 `global_batch_size`：能解释为什么 sample 数多也会失败。
- [ ] static micro batch 对齐失败：能指出要调 `global_batch_size/micro_batch_size/dp_size/mb_group`。
- [ ] `debug_rollout_only` 下训练没有数据：能解释这是预期提前返回。
- [ ] 权重更新拿不到 engine：能检查是否存在 `update_weights=True` 的 server。
- [ ] 训练侧字段 dtype 异常：能回到 `_tensorize_rollout_data_for_training` 检查字段是否列在 dtype map 中。
- [ ] custom converter 缺 `tokens/rollout_ids` 或混合可选字段：能解释为何当前错误会延迟到 split/训练侧。

## 运行或观测验收

任选一种完成：

- [ ] 开 `debug_rollout_only` 跑一轮，确认有 Sample 日志或 debug dump，但不产生训练 ObjectRef。
- [ ] 用 `load_debug_rollout_data` 复放一份样本，确认跳过 SGLang 请求但仍进入 convert/split。
- [ ] 构造一个最小 compact nested output，验证 sibling `rollout_id` 缺失时会失败，共享 id 后通过。
- [ ] 跑 `tests/test_dp_schedule.py`，确认 partitions 和 micro-batch indices 满足 invariants。
- [ ] 构造 5 个 rollout、`global_batch_size=2`，确认最后一个 rollout 不进入 partition；再解释这不是“最后一步自动缩 batch”。

## 最小静态与单元检查

完成学习后执行：

```powershell
rg -n 'def generate\(|_get_rollout_data|_validate_rollout_id_annotated|_convert_samples_to_train_data|_split_train_data_by_dp|get_updatable_engines_and_lock' slime/slime/ray/rollout.py
Push-Location slime
python -m pytest tests/test_dp_schedule.py -q
Pop-Location
```

通过标准：

- 静态结果能串出 generate、debug replay、rollout id 校验、train_data 转换与 DP split。
- DP schedule 单测通过；缺 Ray/Megatron 不影响该纯调度测试时，才可把失败归入实现。
- 当前基线预期为 `9 passed`；测试覆盖静态/动态/VPP/oversize/rollout grouping/trailing trim，但不覆盖 `balance_by_flops` 超 token cap 的 OOM 风险。
- 当前环境实测 `9 passed`。插件 runtime-hook contracts 因缺 `httpx` 在 collection 阶段失败，并伴随 Torch/NumPy ABI 警告；这属于环境覆盖限制。
- 当前 reward 方法 AST 实测确认：固定 fanout 按 reshape 后的行中心化；可变 fanout fallback 只做整批中心化。GPU fanout E2E 仍需完整 SGLang/Megatron 环境。
- 能用源码入口和运行现象证明关键判断。
- 读者能沿一个 `rollout_id` 复述样本生产线，并能定位至少 4 类失败模式。
