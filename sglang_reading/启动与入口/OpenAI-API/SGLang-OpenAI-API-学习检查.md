---
title: "OpenAI-API · 学习检查"
type: exercise
framework: sglang
topic: "OpenAI-API"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# OpenAI-API · 学习检查

## 读完应能回答

- [ ] 为什么 `http_server.py` 的 `/v1/chat/completions` route 不是主要业务逻辑入口。
- [ ] `OpenAIServingBase.handle_request` 的四步模板是什么。
- [ ] Completion 的 `prompt` 如何变成 `GenerateReqInput.text` 或 `GenerateReqInput.input_ids`。
- [ ] Chat 为什么必须先经过 `_process_messages`，以及 tool/reasoning 约束在哪里进入 sampling params。
- [ ] TokenizerManager 从哪一行开始承担 normalize、priority、pause、LoRA 校验、tokenize 和 send scheduler。
- [ ] 为什么内部 chunk 是累积态时，OpenAI SSE 必须维护 `stream_offsets`。
- [ ] 为什么 Chat stream 第一包可能只有 role，没有正文。
- [ ] `n>1` 时 usage 为什么不能直接把每个 choice 的 prompt tokens 相加。
- [ ] Ollama adapter 为什么不继承 `OpenAIServingBase`，但仍复用 `GenerateReqInput`。

## 打开源码后的定位题

| 问题 | 应定位到 |
|------|----------|
| 路由进错 handler | `http_server.py` 的 route 和 `lifespan` handler 初始化 |
| `stream=true` 先返回 JSON error | `OpenAIServingChat._handle_streaming_request` 的 generator kick-start |
| 文本重复或漏字 | Chat/Completion stream 的 offset 更新 |
| `model:adapter` 覆盖 `lora_path` | `OpenAIServingBase._resolve_lora_path` |
| `X-Data-Parallel-Rank` 不符合预期 | `extract_routed_dp_rank_from_header` |
| tool call arguments 断裂 | `_process_tool_call_stream` 的 parser 状态 |
| reasoning 不在 content 里 | `_process_reasoning_stream` 和 `reasoning_content` 字段 |
| usage 少算 prompt tokens | `UsageProcessor.calculate_streaming_usage` |
| Ollama 输出过长 | `_convert_options_to_sampling_params` 的默认 `max_new_tokens` |

## 可观测验证

**操作：** 在可运行服务中逐项构造请求，并记录请求参数、关键中间字段和返回 chunk 顺序；一次只改变一个条件。

**预期：** 每个现象都能回指到兼容层中的明确状态或优先级，而不是用“OpenAI 协议大概如此”来解释。

1. 普通 Chat stream：确认第一包 role-only，后续正文按 delta 增长，末尾有结束事件。
2. Completion stream：打印 `index`、offset、`len(content["text"])`，确认多 choice 不互相污染。
3. Tool call stream：打印 parser 类型、`normal_text`、`calls`，确认 required tool choice 是否走 `JsonArrayParser`。
4. LoRA：构造 `model="base:adapter"` 与显式 `lora_path` 同时存在的请求，确认最终 adapter 来自 `model`。
5. DP rank：同时传 body `routed_dp_rank` 和 header `X-Data-Parallel-Rank`，确认 header 优先。
6. Usage：用 `n=2` 请求确认 prompt tokens 只按每组 choice 的第一个 index 统计。
7. Ollama：不传 `num_predict` 时确认内部 `max_new_tokens` 为 2048。
8. Embedding：传空字符串、混合类型 list、负 token id，确认错误在 embedding serving 层返回。

## 迁移结论

OpenAI API 兼容层的主线是“外部协议形状到内部生成契约，再到外部流式协议”。读者不需要背 endpoint 表，而要能沿 `request -> GenerateReqInput -> TokenizerManager -> content chunk -> SSE/NDJSON` 这条线定位问题。
