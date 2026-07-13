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
updated: 2026-07-13
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

它返回 `list[Sample]`，所以更准确的模型是“单样本 generate fan-out”，而不是完整替换 rollout 外循环。`run_agent_system` 已经为 solver/rewriter/selector 分阶段调用 RM 并写 reward，外层 RM 通常不会再次打分。

若训练指标异常，检查的不只是共同 `rollout_id`：返回数量随失败分支变化时，默认 reward normalization 会把拍平后的整批 reward 当作一组。示例脚本默认没有 `--disable-rewards-normalization`，需要显式关闭或提供按 rollout id 分组的 reward postprocess。源码入口：`examples/multi_agent/agent_system.py` L198-L296、`slime/ray/rollout.py` L686-L711。

## 3. Search-R1 的 logprob 为什么不能重 tokenize

当 `return_logprob=True` 时，token id 和 logprob 来自推理引擎。如果你先裁剪字符串再重新 tokenize，token 数和 logprob 数可能不一致。Search-R1 的修复是把 `</search>`、`</answer>` 加到 stop 参数，让引擎在边界停止。源码入口：`examples/search-r1/generate_with_search.py` L93-L98、L145-L177。

症状通常是：

- `rollout_log_probs` 长度和 response token 长度不同。
- TIS 权重异常。
- answer 后 trailing text 被训练。

## 4. observation token 为什么不能参与 loss

检索结果和工具 observation 是环境返回，不是 policy 采样。如果它们参与 policy loss，模型会被训练去“生成检索服务返回的文本”。Search-R1 用 `trainable=False` 把 observation 加入上下文但排除出 loss。源码入口：`examples/search-r1/generate_with_search.py` L179-L244。

还要确认 `output_token_logprobs` 存在；示例在 `return_logprob=True` 时把它当硬依赖。未知 finish reason 不会被映射成完成态，partial rollout 会直接 assert。eval 也不会自动获得独立配置，因为函数签名没有 `evaluation` 参数。

## 5. rollout_buffer 何时值得使用

适合这几类场景：

- 轨迹生成跑在另一组机器或服务里。
- agent rollout 很慢，训练进程不应直接等待每条轨迹。
- 需要按 `instance_id` 攒够 group 后再训练。
- 多种 `TASK_TYPE` generator 共用一套 HTTP buffer 服务。

如果只是单机 search/RAG，优先用 `custom_generate`。如果需要生产队列语义，当前 rollout_buffer 只能作为接口草图：它没有持久化、ack/lease、去重、重启恢复或多租户隔离。

## 6. generator 模块必须实现什么

`buffer.py` 实际自动发现 `generator/*.py`，并不执行 README 所说的 `_generator.py` 后缀约束。每个 generator 必须提供：

| 符号 | 必需性 | 用途 |
|------|--------|------|
| `TASK_TYPE` | 必需 | 作为 generator 路由 key |
| `run_rollout(data)` | 必需 | 启动外部轨迹生成 |
| `transform_group` | 可选 | 读取时转换 group |
| `is_valid_group` | 可选 | 判断 group 是否可消费 |
| `get_group_data_meta_info` | 可选 | 统计 meta 信息 |

源码入口：`slime_plugins/rollout_buffer/buffer.py` L54-L109。

模板中的 `normalize_group_data` 不会自动生效，因为发现器寻找的名字是 `transform_group`。若希望读取时归一化，必须改名或提供同名 wrapper。重复 `TASK_TYPE` 也不会报冲突，后扫描模块会覆盖先前映射。源码入口：`buffer.py` L87-L103、`generator/base_generator.py` L300-L351。

## 7. `/get_rollout_data` 一直返回空

按顺序检查：

1. `/start_rollout` 是否真的启动 generator。
2. generator 是否向 `/buffer/write` 写入。
3. 写入 item 是否带 `instance_id`。
4. 同一 `instance_id` 的样本数是否达到 `num_repeat_per_sample`。
5. 自定义 `is_valid_group` 是否过于严格。

`BufferQueue.__len__` 只统计有效 group，未攒够 group 时训练侧会继续等待。函数名 `_get_valid_groups_with_timeout` 具有误导性：当前 `timed_out_groups` 和 `finished_groups` 都不会被填充，不存在真正的超时放行或过期清理。

如果 HTTP 服务根本不可达，训练侧 `start_rollout` 会同步、无 sleep 地无限重试；数据拉取也没有总超时。先用 `Invoke-RestMethod` 单独验证三个 endpoint，不要直接等待训练日志。

## 8. rollout_buffer 拉到数据后 Sample shape 不对

训练侧 wrapper 要求每条 record 至少有 `uid`、`instance_id`、`messages`、`reward`、`extra_info`。源码入口：`slime_plugins/rollout_buffer/rollout_buffer_example.py` L138-L170。

随后 `MultiTurnLossMaskGenerator` 会把 messages 转成 token 与 loss mask，再构造 `Sample`。如果 messages 不是预期 OpenAI 格式，通常会在 loss mask 生成或 response length 计算时暴露。

再检查 group 是否**恰好**等于 `n_samples_per_prompt`。服务端默认只要求 `len(group) >= group_size`，读取会返回完整超额 group，而 `RolloutDataSourceWithBuffer.add_samples` 要求严格等长。wrapper 的 `select_rollout_data` 只按 group 数截取，不会裁剪组内多余 sample。

还要检查 `group_index`、`index` 与 prompt/response 字段：示例把 `index` 设为 `instance_id`，没有显式设置 `group_index`，并把 `prompt` 写成 uid。训练主体可能仍能依赖 tokens/loss mask 工作，但 group metrics 和日志语义会漂移。

## 9. GLM5 插件和 examples 有什么关系

GLM5 是模型结构插件，不是 rollout workflow。它扩展的是 Megatron attention spec、DSA index sharing、pipeline split 和 checkpoint 参数集合。源码入口：`slime_plugins/models/glm5/glm5.py` L37-L52、L145-L198。

排障时先看 `--spec` 是否指向 `get_glm5_spec`、输入是否 packed sequence、每个 pipeline stage 是否从 computing layer 开始、checkpoint 是否包含/跳过对应 indexer 权重，以及 TileLang/Apex/Transformer Engine 依赖是否齐全，不要从 rollout path 入手。

若初始化中途异常后其他模型也出现 normalization 配置异常，检查 `DSAMLASelfAttention.__init__` 临时把 `config.normalization` 从 RMSNorm 改成 LayerNorm 的区段；该切换没有 `try/finally`，`build_module(k_norm)` 抛错时共享 config 可能留在 LayerNorm。

## 10. 从 example 拷贝到自己项目的最小步骤

1. 只拷贝你要注册的函数及其 helper。
2. 把 CLI path 改成你自己的 package path。
3. 先用 AST/`py_compile` 检查真实 example；通用 contract test 的 reference args 不足以直接执行需要模型、tokenizer 和外部服务的示例函数。
4. 用小 batch 跑一次，观察 `Sample` 字段、reward、loss mask 和日志。
5. 再扩大到完整训练脚本。

不要先整目录复制再改 import；这样很容易把 README、脚本路径、外部服务地址和任务依赖混在一起。也不要直接运行示例 shell 做 smoke：这些脚本开头会 `pkill -9` SGLang、Ray 和 Python 进程，属于专用容器内的全量启动脚本。
