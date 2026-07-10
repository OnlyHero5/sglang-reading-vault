---
title: "阅读方法 · 排障指南"
type: troubleshooting
framework: slime
topic: "阅读方法"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 阅读方法 · 排障指南

## 你为什么要读

本篇按读者最常见的误解排障。方法论专题的目标不是让你记住宣传语，而是避免后续读源码时走错坐标系。

## 1. 推荐阅读顺序是什么

先建立三角闭环，再读主循环和参数系统：

| 阶段 | 专题 | 目的 |
|------|------|------|
| 0 | [[Slime-阅读方法]] | 建立 Training / Rollout / Data Buffer 坐标 |
| I | [[Slime-训练主循环]] | 跟 `generate → train → update_weights` |
| II | [[Slime-Ray参数]]、[[Slime-训练与Rollout参数]] | 搞清 CLI 到 `args` 的事实编译 |
| III | [[Slime-PlacementGroup]]、[[Slime-RayTrainGroup]] | 理解 Ray 资源和 rank actor |
| IV | [[Slime-RolloutManager]] 到 [[Slime-外部推理引擎]] | 读 rollout 生产线 |
| V | [[Slime-Megatron-Actor初始化]] 到 [[Slime-上下文并行与路由重放]] | 读训练后端 |
| VI | [[Slime-分布式权重同步]] 到 [[Slime-插件与示例]] | 读权重同步、agent、扩展 |

## 2. 读 Slime 前要先懂多少 SGLang

至少要懂三件事：

- SGLang server mode 如何接受 `/generate`。
- router 为什么能给多 engine 提供单入口。
- 权重热更新为什么不同于普通 serving。

博文强调 RL workload 有大量在线采样，因此 inference performance 是关键，Slime 才选择深度集成 SGLang。来源：docs/en/blogs/introducing_slime.md L53-L59

不用先读完 SGLang 全库；先读 [[SGLang-HTTP-Server]]、[[SGLang-Scheduler]]、[[SGLang-分布式]] 中和 serving 主线相关的部分即可。

## 3. Slime 和 veRL 的本质区别是什么

一句话：Slime 把 Megatron 和 SGLang 的原生控制面留给用户，自己专注 RL loop；多 backend 框架通常会做统一抽象层，再把不同后端映射进去。

| 问题 | Slime | 多 backend RL 框架常见取舍 |
|------|-------|----------------------------|
| 训练后端 | Megatron 原生参数和并行 | 自有 worker / trainer 抽象 |
| Rollout | SGLang deep integration | 多推理后端公共接口 |
| 扩展方式 | `--*-path` hook | 子类、worker、pipeline 配置 |
| 主循环 | 暴露 `train.py` | trainer class 封装 |

Slime 的优势是大规模 Megatron + SGLang 生产路径更直接；代价是你要理解这两个上游系统本身。

## 4. 为什么只选 SGLang 一个 rollout backend

因为 RL rollout 需要用到 SGLang 的服务、路由、缓存、disaggregation、MoE、权重更新等特有能力。README 明确说，选择单一 rollout backend 可以直接发挥 SGLang-specific 能力，而不是被迫抽象成多个推理引擎的公共能力子集。来源：README.md L22-L26

vime 这类项目说明 rollout backend 可以被替换，但那属于基于 Slime substrate 的外部扩展，不是 Slime 主线设计。

## 5. Data Buffer 是独立服务吗

不是。README 的 Data Buffer 是架构角色名：它管理 prompt 初始化、自定义数据和 rollout 生成方法。来源：README.md L90-L92

实现上会分散在 DataSource、RolloutManager、Sample group、train_data conversion 和 Ray ObjectRef 交付里。不要一看到 Data Buffer 就去找单独 daemon。

## 6. Agent RL 要不要换框架

不需要。Slime 的方法论是让 agent workflow 通过 data generation 或 reward workflow 接进同一闭环。多数情况优先看 [[Slime-自定义扩展]]，从 `custom_generate + custom_rm` 开始；只有默认 rollout 外循环不够时，才替换完整 `rollout_function`。

## 7. 为什么有 rollout-only 和 train-only debug

RL bug 往往不会立刻报错，必须能拆开验证。README 把 rollout-only、train-only 调试列为 correctness-first infrastructure 的一部分。来源：README.md L18-L26

`load_debug_rollout_data` 会切到 train-only，避免实例化 SGLang servers。来源：slime/utils/arguments.py L1844-L1849

## 8. 怎么判断自己读懂了本专题

不看正文，能回答以下问题就够：

- Slime 的两大能力是什么。
- Training / Rollout / Data Buffer 三角各自负责什么。
- `generate → train → update_weights` 为什么构成闭环。
- `--sglang-*` 为什么不是普通配置糖。
- 自定义 agent workflow 为什么不应该 fork training kernel。

答不出来时，回到 [[Slime-阅读方法-源码走读]] 的 README 和博文证据。

## 9. 怎么用命令验证自己没有读偏

用三个只读命令检查坐标系：

```powershell
rg -n "High-Performance Training|Flexible Data Generation|Correctness-first infrastructure" slime/README.md
rg -n "server-based mode|SGLang-only debug mode|Megatron-only debug mode|train.py" slime/docs/en/blogs/introducing_slime.md
rg -n "debug_rollout_only|load_debug_rollout_data|debug_train_only" slime/slime/utils/arguments.py
```

预期：

- README 同时命中训练、数据生成和正确性基础设施，说明 Slime 不是单纯 loss 库。
- 博文同时命中 server-based rollout、两个 debug mode 和暴露 `train.py`，说明少封装是为了让同步点和调试路径可见。
- `arguments.py` 命中 rollout-only 与 train-only 的互斥和改写逻辑，说明调试模式不是文档口号，而是会改变资源编排和后端启动的真实分支。
