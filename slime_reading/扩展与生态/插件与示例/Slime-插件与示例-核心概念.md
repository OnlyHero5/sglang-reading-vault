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
updated: 2026-07-13
---
# 插件与示例 · 核心概念

## 你为什么要读

这个专题的关键不是“有哪些目录”，而是“哪些代码会被 Slime 主循环调用，哪些代码只是外部协作者”。先把边界分清，拷贝 example 时才不会把服务端、生成函数、reward 函数和模型插件混在一起。

## 1. examples 与 plugins 的区别

| 目录 | 定位 | 被谁调用 |
|------|------|----------|
| `examples/` | 可运行工作流样板 | 训练脚本通过 `--*-path` 指向其中函数 |
| `slime_plugins/` | 可 import 扩展库 | CLI path、`--spec`、核心或 example 显式 import |
| 外部服务 | 检索、agent 环境、rollout buffer | example 通过 HTTP 或 CLI 调用 |

源码依据：`examples/README.md` L3-L20 列出 fully_async、multi_agent、search-r1、tau-bench 等 workflow；`slime_plugins/rollout_buffer/README.md` L3-L18 说明 Rollout Buffer 独立于训练进程。

## 2. Search-R1 是单样本多轮工具调用

Search-R1 的接入方式是 `custom_generate + custom_rm`。启动脚本把示例目录加入 Ray `PYTHONPATH`，再用短 path `generate_with_search.generate` 和 `reward_func`；它不是可直接 import 的 `examples.search_r1.*` 包。来源：`examples/search-r1/run_qwen2.5_3B.sh` L115-L130

它保留默认 RolloutManager，只替换每个 `Sample` 内部如何生成：

- 模型输出 `<search>` 或 `<answer>`。
- search 动作访问本地或 Google 检索服务。
- 检索结果作为 observation token 追加到 response。
- 模型生成 token 的 loss mask 为 1，observation token 的 loss mask 为 0。
- reward 函数用 EM 与格式分给样本打分。

实现边界也要记住：`generate` 断言不支持 partial rollout，不显式接收 `evaluation`，全局配置与 semaphore 在进程内共享；reward 假设 `sample.label["ground_truth"]` 存在。它是任务样板，不是通用 search agent SDK。

源码依据：`examples/search-r1/generate_with_search.py` L14-L41 定义搜索配置；L179-L244 展示 generate 主循环；L277-L293 展示 reward 函数。

## 3. multi_agent 也是 custom_generate，不是 rollout_function

旧读法容易把 multi_agent 误认为完整 `rollout_function` 替换；实际脚本使用的是 `--custom-generate-function-path`。来源：`examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh` L38-L45

`generate_with_multi_agents` 在单样本 generate 边界里加载 `custom_multi_agent_function_path`，让 agent system 返回 `list[Sample]`。wrapper 每个 sample 都重新加载 tokenizer，并把 sampling 参数、tokenizer 和配置写入 `args`；agent system 随后 deepcopy 这份 args。来源：`examples/multi_agent/rollout_with_multi_agents.py` L8-L33

这说明 multi_agent 的核心是 fan-out：一个输入 prompt 触发 solver、rewriter、selector 多阶段调用，最终交回数量可变的 sibling。`agent_system.py` 会在各阶段直接调用 batched RM、按正确/错误系数缩放 reward，并把所有 sibling 的 `rollout_id` 写成输入 `sample.index`。来源：`examples/multi_agent/agent_system.py` L198-L296。

这并不自动保留 GRPO 分组。默认 reward normalization 只在扁平样本数恰好等于 `rollout_batch_size * n_samples_per_prompt` 时按固定宽度 reshape；变量 fan-out 走 fallback 后会把整批 reward 当成一组。示例脚本没有关闭 rewards normalization，因此迁移时必须显式决定是否禁用或自定义 reward postprocess。

## 4. rollout_buffer 是外部轨迹队列

rollout_buffer 和前两个样板不同：它不是在 Slime 进程内直接生成 sample，而是启动一个独立 FastAPI 服务。

服务端 `buffer.py` 负责：

- 自动发现 generator。
- 接收 `/buffer/write` 写入。
- 按 `instance_id` 和最小 group size 判断可消费样本。
- 通过 `/get_rollout_data` 让训练侧拉取成组数据。

训练侧 `rollout_buffer_example.py` 负责：

- 通过 `/start_rollout` 通知外部服务启动生成。
- 轮询 `/get_rollout_data`。
- 校验每条记录包含 `uid/instance_id/messages/reward/extra_info`。
- 用 `MultiTurnLossMaskGenerator` 把 OpenAI messages 转成 `Sample`。

源码依据：`slime_plugins/rollout_buffer/buffer.py` L54-L109、L259-L329；`slime_plugins/rollout_buffer/rollout_buffer_example.py` L138-L170、L215-L307。

名称容易让人高估它的可靠性：队列只在内存中，`_get_valid_groups_with_timeout` 当前没有真正的 timeout/finished-group 逻辑；`/start_rollout` 会重建进程级全局 buffer；wrapper 用同步 `requests.post`、`time.sleep` 和无总截止时间的轮询。它演示的是服务边界，不提供持久化、ack、租约、去重或崩溃恢复。

README 与代码还有两处漂移：README 要求 generator 文件以 `_generator.py` 结尾，但发现器实际扫描所有 `*.py`；模板定义 `normalize_group_data`，发现器寻找的却是 `transform_group`，所以该 normalization 函数不会自动挂载。来源：`slime_plugins/rollout_buffer/README.md` L23-L38、`buffer.py` L54-L102、`generator/base_generator.py` L300-L351。

## 5. GLM5 是模型结构插件

GLM5 不属于可直接运行的 rollout example。模型脚本通过 `--spec slime_plugins.models.glm5.glm5 get_glm5_spec` 显式选择 provider；它展示 `slime_plugins/` 可以放 DSA/MLA attention、cross-layer index sharing 和 Megatron layer spec。来源：`scripts/models/glm5-744B-A40B.sh` L13-L15、`scripts/models/glm5.2-744B-A40B.sh` L19-L21。

源码依据：`slime_plugins/models/glm5/glm5.py` L37-L52 判断 skip top-k 层，L145-L198 在 forward 中处理 index sharing holder。

这个插件的风险点和 rollout example 不同：forward 强制 packed sequence，index top-k 硬编码为 2048；跨层 holder 不跨 PP 边界，所以每个 pipeline stage 必须从 computing layer 开始；skip layer 删除 indexer 子模块以匹配 checkpoint 参数集合。它更靠近 Megatron 并行、checkpoint、converter 和自定义 CUDA/TileLang op，不应拿 Search-R1 的调试方式套用。

## 6. 迁移 example 到自己项目的原则

先拷贝最窄的可调用函数，不要复制整个目录：

1. 只改单样本生成：拷贝 `generate` 与必要 helper，并列出 partial/eval/logprob 前提。
2. 需要 reward：拷贝 `reward_func`，再接 `--custom-rm-path`，主动校验 batch 长度与 label schema。
3. 需要 fan-out：先决定 sibling 的 rollout id、reward 口径、normalization 与失败时空列表语义。
4. 需要外部队列：把 rollout buffer 当接口原型，补持久化、幂等、超时、并发隔离和可观测性。
5. 需要模型结构：用 `--spec` 显式加载 provider，按 PP layout、packed sequence、checkpoint 和 kernel 环境验证。

每一步都回到 [[Slime-自定义扩展-学习检查]] 明确契约；真实 example 先跑本专题的静态 smoke，需要 contract test 时为 tokenizer、args 与外部服务提供受控 fixture，最后再做小规模 workflow。
