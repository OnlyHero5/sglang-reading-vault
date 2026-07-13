---
title: "Megatron-Actor初始化 · 学习检查"
type: exercise
framework: slime
topic: "Megatron-Actor初始化"
learning_role: practice
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/exercise
  - source-reading
updated: 2026-07-13
---
# Megatron-Actor初始化 · 学习检查

## 读者能做什么

- [ ] 能画出 `RayTrainGroup -> TrainRayActor.init -> initialize.init -> initialize_model_and_optimizer -> weight_updater` 的主线。
- [ ] 能解释 Ray actor rank、PyTorch world rank、Megatron DP/TP/PP/CP/EP rank 的区别。
- [ ] 能说明 `debug_rollout_only` 为什么不会加载模型，以及后续 `train/save/update_weights` 为什么要有 guard。
- [ ] 能说出 `start_rollout_id = loaded_rollout_id + 1` 的来源，并解释为什么所选 role 的 rank 返回值必须一致。
- [ ] 能说明使用 critic 时只检查 critic ranks，显式 `args.start_rollout_id` 又会覆盖候选返回值。
- [ ] 能说出 actor 与 critic init 的差异：critic 有模型但没有 `weights_backuper` 和 `weight_updater`。
- [ ] 能说明 `weight_updater` 在 init 里只是选型，真正连接 rollout engines 在 `update_weights()`。
- [ ] 能画出 offload 的 `init sleep -> train wake -> train sleep` 状态机。
- [ ] 能指出这只是成功路径；train/save/update 与辅助 checkpoint load 都没有 finally 回滚。
- [ ] 能用症状把问题分到 Ray 创建、distributed 初始化、HF cache、checkpoint、weight sync 或 offload。

## 可执行检查

静态入口检查：

```powershell
rg -n 'class TrainRayActor|def init\(|initialize_model_and_optimizer|start_rollout_id|weights_backuper|weight_updater' slime/slime/ray slime/slime/backends
rg -n 'debug_rollout_only|offload_train|sleep|wake_up' slime/slime/ray slime/slime/backends
```

预期：第一条能串出 Ray actor 到 backend actor 的初始化与 checkpoint 进度返回；第二条能定位不建模与 offload 状态分叉。

## 运行验证设计

完整验证需要 Ray、CUDA、Megatron、checkpoint 和可用 GPU。可以按三组现象判断：

| 配置 | 预期现象 |
|------|----------|
| 正常训练 | 所选 role 的各 rank 返回同一个 `start_rollout_id`，driver 不触发一致性 assert |
| `debug_rollout_only` | init 快速返回 `0`，不出现 Megatron checkpoint load 或模型构建日志 |
| `offload_train` | init 末尾出现 sleep 相关 memory 日志，第一次 train 前出现 wake 相关 memory 日志 |

## 排障演练

- [ ] 若 actor 创建时报 `torch_memory_saver` 动态库缺失，能定位到 `slime/ray/actor_group.py` 的 runtime env 注入。
- [ ] 若卡在 distributed 初始化，能先检查 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`WORLD_SIZE`、backend 和 timeout。
- [ ] 若卡在 HF tokenizer 读取，能想到 node 内串行读与 gloo barrier。
- [ ] 能解释“节点内串行、节点间并发”，并检查 global rank 是否按固定节点块排列。
- [ ] NumPy 2.x 断言失败后，知道必须重建 actor 而不是原地再次 init。
- [ ] 若 colocate + delta 报 assert，能解释为什么必须改 full 或关闭 colocate。
- [ ] 若 init 成功但推权失败，能转到 [[Slime-分布式权重同步]] 查 rollout engines 连接和 updater 通道。

## 复盘问题

- [ ] 如果要新增一种训练后端，哪些逻辑应留在 `TrainRayActor`，哪些应留给后端 actor？
- [ ] 如果要新增一个辅助权重 tag，应接在 `weights_backuper` 的哪段生命周期？
- [ ] 如果要让 offload 更细粒度，应同时考虑哪些 process group、CUDA tensor 引用和 updater 连接？
- [ ] 如果 checkpoint 恢复后 rollout id 错了，应从哪些源码入口确认加载步数？
