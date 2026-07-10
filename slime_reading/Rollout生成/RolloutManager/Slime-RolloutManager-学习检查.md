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
updated: 2026-07-10
---
# RolloutManager · 学习检查

## 读者能做什么

- [ ] 能画出 `train.py → RolloutManager.generate → rollout_fn → Sample → train_data → build_dp_schedule → Box(ray.put) → async_train`。
- [ ] 能解释 RolloutManager 为什么是 `num_gpus=0` 的 Ray Actor。
- [ ] 能区分 `generate(rollout_id)` 参数和 `Sample.rollout_id` 字段。
- [ ] 能说出 `tokens/rewards/loss_masks/rollout_ids/rollout_mask_sums/partition/micro_batch_indices` 的来源和用途。
- [ ] 能说明 `raw_reward/total_lengths` 为什么保留全局列。
- [ ] 能解释 `get_updatable_engines_and_lock` 为什么排除 frozen 模型。

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

## 运行或观测验收

任选一种完成：

- [ ] 开 `debug_rollout_only` 跑一轮，确认有 Sample 日志或 debug dump，但不产生训练 ObjectRef。
- [ ] 用 `load_debug_rollout_data` 复放一份样本，确认跳过 SGLang 请求但仍进入 convert/split。
- [ ] 构造一个最小 compact nested output，验证 sibling `rollout_id` 缺失时会失败，共享 id 后通过。
- [ ] 跑 `tests/test_dp_schedule.py`，确认 partitions 和 micro-batch indices 满足 invariants。

## 维护检查

完成学习后执行：

```powershell
node maintenance\audit_source_evidence.mjs --note 'slime_reading\Rollout生成\RolloutManager\Slime-RolloutManager-源码走读.md'
node maintenance\audit_source_evidence.mjs
node maintenance\audit_wikilinks.mjs
git diff --check
```

通过标准：

- 源码引用文件存在，行号在当前 upstream 范围内。
- 双链无断链。
- 能用源码入口和运行现象证明关键判断。
- 读者能沿一个 `rollout_id` 复述样本生产线，并能定位至少 4 类失败模式。
