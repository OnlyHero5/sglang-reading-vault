---
title: "插件与示例 · 核心概念"
type: concept
framework: slime
topic: "插件与示例"
learning_role: core
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/concept
  - source-reading
updated: 2026-07-10
---
# 插件与示例 · 核心概念

## 你为什么要读

这个专题的关键不是“有哪些目录”，而是“哪些代码会被 Slime 主循环调用，哪些代码只是外部协作者”。先把边界分清，拷贝 example 时才不会把服务端、生成函数、reward 函数和模型插件混在一起。

## 1. examples 与 plugins 的区别

| 目录 | 定位 | 被谁调用 |
|------|------|----------|
| `examples/` | 可运行工作流样板 | 训练脚本通过 `--*-path` 指向其中函数 |
| `slime_plugins/` | 可 import 扩展库 | 核心、example 或启动 side effect import |
| 外部服务 | 检索、agent 环境、rollout buffer | example 通过 HTTP 或 CLI 调用 |

源码依据：`examples/README.md` L3-L20 列出 fully_async、multi_agent、search-r1、tau-bench 等 workflow；`slime_plugins/rollout_buffer/README.md` L3-L18 说明 Rollout Buffer 独立于训练进程。

## 2. Search-R1 是单样本多轮工具调用

Search-R1 的接入方式是 `custom_generate + custom_rm`。启动脚本直接指向 `generate_with_search.generate` 和 `reward_func`。来源：`examples/search-r1/run_qwen2.5_3B.sh` L115-L120

它保留默认 RolloutManager，只替换每个 `Sample` 内部如何生成：

- 模型输出 `<search>` 或 `<answer>`。
- search 动作访问本地或 Google 检索服务。
- 检索结果作为 observation token 追加到 response。
- 模型生成 token 的 loss mask 为 1，observation token 的 loss mask 为 0。
- reward 函数用 EM 与格式分给样本打分。

源码依据：`examples/search-r1/generate_with_search.py` L14-L41 定义搜索配置；L179-L244 展示 generate 主循环；L277-L293 展示 reward 函数。

## 3. multi_agent 也是 custom_generate，不是 rollout_function

旧读法容易把 multi_agent 误认为完整 `rollout_function` 替换；实际脚本使用的是 `--custom-generate-function-path`。来源：`examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh` L38-L45

`generate_with_multi_agents` 在单样本 generate 边界里加载 `custom_multi_agent_function_path`，让 agent system 返回 `list[Sample]`。来源：`examples/multi_agent/rollout_with_multi_agents.py` L8-L33

这说明 multi_agent 的核心是 fan-out：一个输入 prompt 触发多个子 agent，最终仍交回默认 rollout 外循环。

## 4. rollout_buffer 是外部轨迹队列

rollout_buffer 和前两个样板不同：它不是在 Slime 进程内直接生成 sample，而是启动一个独立 FastAPI 服务。

服务端 `buffer.py` 负责：

- 自动发现 generator。
- 接收 `/buffer/write` 写入。
- 按 `instance_id` 和 group size 攒够样本。
- 通过 `/get_rollout_data` 让训练侧拉取成组数据。

训练侧 `rollout_buffer_example.py` 负责：

- 通过 `/start_rollout` 通知外部服务启动生成。
- 轮询 `/get_rollout_data`。
- 校验每条记录包含 `uid/instance_id/messages/reward/extra_info`。
- 用 `MultiTurnLossMaskGenerator` 把 OpenAI messages 转成 `Sample`。

源码依据：`slime_plugins/rollout_buffer/buffer.py` L54-L109、L259-L329；`slime_plugins/rollout_buffer/rollout_buffer_example.py` L138-L170、L215-L307。

## 5. GLM5 是模型结构插件

GLM5 不属于可直接运行的 rollout example。它展示 `slime_plugins/` 也可以放模型结构扩展，例如 DSA/MLA attention、cross-layer index sharing 和 Megatron spec provider。

源码依据：`slime_plugins/models/glm5/glm5.py` L37-L52 判断 skip top-k 层，L145-L198 在 forward 中处理 index sharing holder。

这个插件的风险点和 rollout example 不同：它更靠近 Megatron 并行、checkpoint、converter 和模型结构，不应拿 Search-R1 的调试方式套用。

## 6. 迁移 example 到自己项目的原则

先拷贝最窄的可调用函数，不要复制整个目录：

1. 只改单样本生成：拷贝 `generate` 与必要的 helper。
2. 需要 reward：拷贝 `reward_func`，再接 `--custom-rm-path`。
3. 需要外部队列：拷贝 rollout buffer 的服务端与 `generate_rollout` 包装器。
4. 需要模型结构：把插件作为 package import，按模型初始化和 checkpoint 路径验证。

每一步都回到 [[Slime-自定义扩展-学习检查]] 跑 contract 或小规模 smoke。
