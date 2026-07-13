---
title: "训练步骤"
type: map
framework: slime
topic: "训练步骤"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/map
  - source-reading
updated: 2026-07-13
---
# 训练步骤

> **Slime 训练后端**
> **源码范围：** `train.py`、`train_async.py`、`slime/ray/actor_group.py`、`slime/backends/megatron_utils/actor.py`、`model.py`、`data.py`、`loss.py`

## 读者为什么要读

RolloutManager 已经把一次 rollout 的样本放进 Ray Object Store。Train Step 回答下一个问题：Megatron actor 如何把这包样本变成一次参数更新，并把 critic values、log-prob、advantage、loss、optimizer step 串成闭环。

读完本专题，应该能排查：

- `async_train` 卡住、返回值不符合预期，或者误以为它是 Megatron 异步训练。
- PPO + Critic 路径里 actor 没拿到 `values`，导致 advantage 不对。
- 动态 batch 下 `num_microbatches` 和 `global_batch_sizes` 对不上。
- rollout log-prob、ref log-prob、actor train log-prob 口径混淆。
- offload、routing replay、pipeline last stage 造成的训练分支误判。
- actor/critic 使用不同 TP/PP/CP/DP 配置时，为什么 values 可能送到错误 worker，甚至 rollout 分片配置被 critic 覆盖。
- 训练异常后为什么不能默认原地重试，以及 async 模式的 policy lag 到底有几层。

## 一句话模型

Train Step 是一个 **分阶段训练事务**：先恢复 rank-local 数据，再按配置执行零到多次 aux forward，补齐 value/log-prob/advantage，最后进入一个或多个 Megatron optimizer step。它没有统一回滚；任何阶段失败，都可能留下 iterator、模型 mode、DDP hook、梯度、offload 或环境变量状态。

```mermaid
flowchart LR
  A["Box(ObjectRef)<br/>rollout_data_ref"]
  B["rank rollout_data<br/>tokens / masks / rewards"]
  C["aux forward<br/>values / log_probs"]
  D["advantages<br/>returns / kl"]
  E["Megatron train<br/>forward-backward"]
  F["optimizer step<br/>backup actor"]
  G["update_weights<br/>rollout engines"]

  A --> B --> C --> D --> E --> F --> G
```

## 首次阅读路径

| 文件 | 读它解决什么 |
| ------ | -------------- |
| [[Slime-训练步骤-核心概念]] | 建立“训练转换器”模型，分清 Ray、Actor、Megatron 三层 |
| [[Slime-训练步骤-源码走读]] | 沿 PPO + Critic 主线追踪一次真实训练更新 |
| [[Slime-训练步骤-数据流]] | 看 `rollout_data`、`DataIterator`、`external_data` 如何变形 |
| [[Slime-训练步骤-排障指南]] | 按症状定位 offload、log-prob 复用、critic-only、PP last stage 等问题 |
| [[Slime-训练步骤-学习检查]] | 用图、问题和命令验收自己是否真的读通 |

## 主线位置

```mermaid
sequenceDiagram
  participant RM as RolloutManager
  participant T as train.py
  participant C as Critic group
  participant A as Actor group
  participant M as Megatron
  participant R as SGLang rollout

  RM-->>T: rollout_data_ref
  T->>C: async_train(rollout_id, ref)
  C->>M: value forward + value_loss train
  C-->>T: value_refs
  T->>A: async_train(rollout_id, ref, external_data=value_refs)
  A->>M: logprob forward + policy train
  A-->>T: done
  T->>R: sync 模式每轮；async 模式按 interval update_weights
```

源码入口：来源：train.py L63-L89

这条主线的关键不是“调用了几个函数”，而是四个边界：

- Ray 边界：主进程只拿 ObjectRef，不直接训练。
- DP 边界：每个 Megatron DP rank 只取自己的 `Box`。
- PP 边界：只有 pipeline last stage 产出 `values/log_probs/advantages`。
- 闭环边界：同步主循环在 actor train 后立即推权重；async 主循环按 interval 推送，不能把 optimizer 完成等同于 rollout engine 已更新。
- 拓扑边界：critic refs 按 worker 序号交给 actor，RolloutManager 的训练并行配置又会被最后调用 `set_rollout_manager` 的 critic rank 0 覆盖；启用 critic 时不能只保证 GPU 总数相同，还要核对 TP/PP/CP/DP 与 PP-last-stage rank 映射。

## 与上下游的关系

| 方向 | 模块 | 关系 |
|------|------|------|
| 上游 | [[Slime-RolloutManager]] | 生产 `list[Box]`，并按 DP rank 切好训练数据 |
| 上游 | [[Slime-Megatron-Actor初始化]] | 初始化 Ray train actor、model、optimizer、backup tags |
| 并行 | [[Slime-训练数据]] | 解释 `process_rollout_data`、`get_data_iterator`、`get_batch` |
| 并行 | [[Slime-Advantage计算]] | 解释 KL、advantage、return 的算法分支 |
| 并行 | [[Slime-Policy-Loss]] | 解释 policy/value/SFT/custom loss 细节 |
| 下游 | [[Slime-分布式权重同步]] | actor 参数更新后推送到 SGLang engines |

## 验证抓手

- PPO + Critic：看 `tests/test_qwen3_4B_ppo.py`，关注 `--use-critic`、`--num-critic-only-steps`、`--advantage-estimator ppo`。
- Debug 复放：用 rollout debug 数据验证 `_get_rollout_data → train_actor`，不用重新跑 SGLang。
- 日志：关注 `train/step`、`train/*global_batch_size`、`train/ppo_kl`、`train/kl_loss`、`train/train_rollout_logprob_abs_diff`。
- 状态：异常前后记录 iterator offset、`model.training`、`config.no_sync_func/param_sync_func`、GC enabled、grad buffer 和 offload 状态；当前训练路径没有事务式恢复。
- async policy lag：记录 rollout 生成时的权重版本、训练完成版本和实际同步版本；`update_weights_interval=N` 表示 rollout engine 最多跨多个 actor step 才更新，且 interval 边界会先等待已启动的下一轮生成，再同步权重。
- 审计：本专题的源码引用应通过 `node maintenance\audit_source_evidence.mjs --note ...`。
