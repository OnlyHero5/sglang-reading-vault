---
title: "Slime 专题看板"
type: dashboard
framework: slime
topic: "Slime"
learning_role: reference
tags:
  - framework/slime
  - content/dashboard
  - source-reading
updated: 2026-07-13
---

# Slime 专题看板

## 阅读路径

首次阅读只走 [[RL训练闭环主线]]，再按对象进入专题：

- Ray 与资源布局：[[Slime-Ray编排]]
- Sample 与 rollout：[[Slime-Rollout生成]]
- Megatron 训练：[[Slime-训练后端]]
- 权重一致性：[[Slime-权重同步]]
- Agent 与扩展：[[Slime-高级特性]] · [[Slime-扩展与生态]]

## 动态内容

![[Slime内容.base]]

## 使用建议

排查数据错误时沿 `Sample -> train_data -> ObjectRef -> RolloutBatch`；排查训练错误时沿 `logprob -> advantage -> loss`；排查旧权重时按实际 updater 核对 `writer/reader -> lock/quiescence -> payload -> cache/commit -> engine version`，不要套用统一 pause/flush 顺序。

看板只提供入口。同步、one-step async、fully async 与 external buffer 的 staleness/恢复语义不同，应先确认当前主循环和 rollout contract。
