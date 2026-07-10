---
title: "Slime 补充主题"
type: reference
framework: slime
topic: "总结复盘"
learning_role: reference
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/reference
  - source-reading
updated: 2026-07-10
---
# Slime 补充主题

有些 upstream 子系统没有独立专题，不代表它们不重要；它们可能不是默认 RL 主线的第一层入口，或者已经被相邻专题覆盖。本篇告诉你什么时候需要补读它们。

## 1. 速查表

| Upstream 主题 | 为什么暂不独立成专题 | 阅读入口 |
|---------------|----------------------|---------------------|
| `megatron_server.py` | 辅助 Megatron 服务入口，不是默认 RL 主循环 | [[Slime-可观测性与CI]]、[[Slime-Megatron-Actor初始化]] |
| CI / 生产可观测 | 测试和观测是工程支撑，不改变训练闭环语义 | [[Slime-可观测性与CI]]、[[Slime-学习路径]] |
| `slime_plugins/rollout_buffer` | 可选 external buffer 插件，非默认 DataSource | [[Slime-插件与示例]]、[[Slime-数据源]] |
| Critic-only 阶段 | 与 actor 共用主循环和 train actor 基础设施 | [[Slime-训练主循环]]、[[Slime-训练步骤]] |
| `train_async.py` | 同步主循环的异步变体 | [[Slime-其他Rollout路径]]、[[Slime-训练主循环-排障指南]] |
| FSDP 后端 | Megatron 是当前主路径，FSDP 只做边界提及 | [[Slime-Megatron-Actor初始化-核心概念]] |

## 2. 什么时候补读 `megatron_server.py`

只有当你在做 Megatron forward-only、服务化调试、或非 SGLang rollout 方案时才需要打开它。默认训练入口仍是 `train.py`，默认 rollout 仍是 SGLang engine。

阅读顺序：

1. [[Slime-可观测性与CI]]
2. [[Slime-Megatron-Actor初始化-核心概念]]
3. upstream `slime/backends/megatron_utils/server/megatron_server.py`

## 3. 什么时候补读 CI 和 contract tests

当你改 customization hook、example、agent adapter 或参数校验时，优先看 CPU contract tests；当你改 Megatron/SGLang 联动、checkpoint、PD、async 或 GPU path 时，再看 GPU e2e。

阅读顺序：

1. [[Slime-自定义扩展-学习检查]]
2. [[Slime-插件与示例-学习检查]]
3. [[Slime-可观测性与CI]]

## 4. 什么时候补读 rollout_buffer

如果你的轨迹生成跑在外部服务或另一组机器上，需要 HTTP buffer 聚合 `instance_id` group，再由训练侧拉回 OpenAI messages 转 `Sample`，就读 rollout_buffer。

阅读顺序：

1. [[Slime-插件与示例-源码走读]]
2. [[Slime-插件与示例-数据流]]
3. [[Slime-数据源-排障指南]]

## 5. 与 SGLang 的交叉补课

| 主题 | Slime 侧 | SGLang 侧 |
|------|----------|-----------|
| Rollout 推理 | [[Slime-SGLang-Engine]] | [[SGLang-Scheduler]] |
| 权重热更新 | [[Slime-分布式权重同步]] | [[SGLang-CheckpointEngine]] |
| Agent tool parse | [[Slime-自定义扩展-源码走读]] | [[SGLang-OpenAI-API-源码走读]] |
| 外部服务和网关 | [[Slime-插件与示例]] | [[SGLang-model-gateway]] |

跨库对照见 [[knowledge_maps/三框架知识地图]]。

## 导航

- [[Slime-导读与总览]]
- [[Slime-综合学习检查]]
- [[SGLang-补充主题]]
