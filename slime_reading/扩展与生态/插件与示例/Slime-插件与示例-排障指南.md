---
title: "插件与示例 · 排障指南"
type: troubleshooting
framework: slime
topic: "插件与示例"
learning_role: debug
source_baseline: "22cdc6e1"
tags:
  - framework/slime
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# 插件与示例 · 排障指南

## 你为什么要读

示例最容易被误用成整套复制粘贴：Search-R1、multi-agent、rollout buffer 和模型插件替换的是不同边界。本文先确认扩展点和返回契约，再检查路径加载、字段对齐与副作用，避免用错误的示例验收标准排查另一类插件。

本篇按迁移 example 时最容易踩的边界排障。

## 1. Search-R1 为什么用 custom_generate 而不是 rollout_function

因为它只改变单个 sample 内部的多轮 search 逻辑，仍然可以复用默认 RolloutManager 的 batch、filter、RM、debug dump 和训练数据交付。

启动脚本也证明它只注册 `--custom-generate-function-path` 与 `--custom-rm-path`。源码入口：`examples/search-r1/run_qwen2.5_3B.sh` L115-L120。

## 2. multi_agent 到底是不是 rollout_function

不是。当前 multi_agent 脚本使用 `--custom-generate-function-path examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents`。源码入口：`examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh` L38-L45。

它返回 `list[Sample]`，所以更准确的模型是“单样本 generate fan-out”，而不是完整替换 rollout 外循环。

## 3. Search-R1 的 logprob 为什么不能重 tokenize

当 `return_logprob=True` 时，token id 和 logprob 来自推理引擎。如果你先裁剪字符串再重新 tokenize，token 数和 logprob 数可能不一致。Search-R1 的修复是把 `</search>`、`</answer>` 加到 stop 参数，让引擎在边界停止。源码入口：`examples/search-r1/generate_with_search.py` L93-L98、L145-L177。

症状通常是：

- `rollout_log_probs` 长度和 response token 长度不同。
- TIS 权重异常。
- answer 后 trailing text 被训练。

## 4. observation token 为什么不能参与 loss

检索结果和工具 observation 是环境返回，不是 policy 采样。如果它们参与 policy loss，模型会被训练去“生成检索服务返回的文本”。Search-R1 用 `trainable=False` 把 observation 加入上下文但排除出 loss。源码入口：`examples/search-r1/generate_with_search.py` L179-L244。

## 5. rollout_buffer 何时值得使用

适合这几类场景：

- 轨迹生成跑在另一组机器或服务里。
- agent rollout 很慢，训练进程不应直接等待每条轨迹。
- 需要按 `instance_id` 攒够 group 后再训练。
- 多种 `TASK_TYPE` generator 共用一套 HTTP buffer 服务。

如果只是单机 search/RAG，优先用 `custom_generate`。

## 6. generator 模块必须实现什么

`buffer.py` 自动发现 `generator/*.py`。每个 generator 必须提供：

| 符号 | 必需性 | 用途 |
|------|--------|------|
| `TASK_TYPE` | 必需 | 作为 generator 路由 key |
| `run_rollout(data)` | 必需 | 启动外部轨迹生成 |
| `transform_group` | 可选 | 读取时转换 group |
| `is_valid_group` | 可选 | 判断 group 是否可消费 |
| `get_group_data_meta_info` | 可选 | 统计 meta 信息 |

源码入口：`slime_plugins/rollout_buffer/buffer.py` L54-L109。

## 7. `/get_rollout_data` 一直返回空

按顺序检查：

1. `/start_rollout` 是否真的启动 generator。
2. generator 是否向 `/buffer/write` 写入。
3. 写入 item 是否带 `instance_id`。
4. 同一 `instance_id` 的样本数是否达到 `num_repeat_per_sample`。
5. 自定义 `is_valid_group` 是否过于严格。

`BufferQueue.__len__` 只统计有效 group，未攒够 group 时训练侧会继续等待。

## 8. rollout_buffer 拉到数据后 Sample shape 不对

训练侧 wrapper 要求每条 record 至少有 `uid`、`instance_id`、`messages`、`reward`、`extra_info`。源码入口：`slime_plugins/rollout_buffer/rollout_buffer_example.py` L138-L170。

随后 `MultiTurnLossMaskGenerator` 会把 messages 转成 token 与 loss mask，再构造 `Sample`。如果 messages 不是预期 OpenAI 格式，通常会在 loss mask 生成或 response length 计算时暴露。

## 9. GLM5 插件和 examples 有什么关系

GLM5 是模型结构插件，不是 rollout workflow。它扩展的是 Megatron attention spec、DSA index sharing、pipeline split 和 checkpoint 参数集合。源码入口：`slime_plugins/models/glm5/glm5.py` L37-L52、L145-L198。

排障时先看模型初始化、pipeline stage 是否从 computing layer 开始、checkpoint 是否包含/跳过对应 indexer 权重，不要从 rollout path 入手。

## 10. 从 example 拷贝到自己项目的最小步骤

1. 只拷贝你要注册的函数及其 helper。
2. 把 CLI path 改成你自己的 package path。
3. 用 [[Slime-自定义扩展-学习检查]] 的 contract tests 检查签名和返回结构。
4. 用小 batch 跑一次，观察 `Sample` 字段、reward、loss mask 和日志。
5. 再扩大到完整训练脚本。

不要先整目录复制再改 import；这样很容易把 README、脚本路径、外部服务地址和任务依赖混在一起。
