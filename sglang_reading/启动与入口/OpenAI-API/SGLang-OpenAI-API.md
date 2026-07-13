---
title: "OpenAI-API"
type: map
framework: sglang
topic: "OpenAI-API"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-11
---
# OpenAI-API

## 你为什么要读

读这个专题，不是为了记住 SGLang 支持多少个 `/v1/*` endpoint，而是为了回答一个更实用的问题：当业务用 OpenAI SDK 调 SGLang 时，外部 JSON 是在哪里变成内部生成请求的，内部生成结果又在哪里变回 SDK 期待的流式 chunk。

读完后你应该能解决三类问题：

1. OpenAI Chat、Completion、Embedding、Ollama 请求分别在哪一层转成 `GenerateReqInput` 或 `EmbeddingReqInput`。
2. 流式输出重复、空 delta、tool call 断裂、reasoning 字段缺失时，应该先看哪个状态表。
3. LoRA、DP rank、routing key、usage、abort 这些 serving 语义为什么不属于 Scheduler，而属于兼容层和 TokenizerManager 的交界处。

## 源码主线

把兼容层想成“协议海关”：外面进来的是 OpenAI/Ollama 各自的护照，里面只认 SGLang 的内部通行证。

```text
HTTP route
  -> OpenAI/Ollama serving handler
  -> GenerateReqInput / EmbeddingReqInput
  -> TokenizerManager.generate_request
  -> internal content chunk
  -> OpenAI SSE / Ollama NDJSON
```

OpenAI 路由本身很薄，真正的转换发生在 handler。`lifespan` 中先把 handler 挂到 `app.state`。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L291-L323
    # Initialize OpenAI serving handlers
    fast_api_app.state.openai_serving_completion = OpenAIServingCompletion(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_chat = (
        _global_state.tokenizer_manager.serving_chat_class(
            _global_state.tokenizer_manager, _global_state.template_manager
        )
    )
    fast_api_app.state.openai_serving_embedding = OpenAIServingEmbedding(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_classify = OpenAIServingClassify(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_score = OpenAIServingScore(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_rerank = OpenAIServingRerank(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_tokenize = OpenAIServingTokenize(
        _global_state.tokenizer_manager, _global_state.template_manager
    )
    fast_api_app.state.openai_serving_detokenize = OpenAIServingDetokenize(
        _global_state.tokenizer_manager
    )
    fast_api_app.state.openai_serving_transcription = OpenAIServingTranscription(
        _global_state.tokenizer_manager
    )

    # Initialize Ollama-compatible serving handler
    fast_api_app.state.ollama_serving = OllamaServing(_global_state.tokenizer_manager)
```

路由函数只取 handler 并委托，说明业务逻辑不在 `http_server.py` 的 endpoint 函数里。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1598-L1613
@app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_completions(request: CompletionRequest, raw_request: Request):
    """OpenAI-compatible text completion endpoint."""
    return await raw_request.app.state.openai_serving_completion.handle_request(
        request, raw_request
    )


@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

## 阅读路径

| 顺序 | 文档 | 读完要拿到什么 |
|------|------|----------------|
| 1 | [[SGLang-OpenAI-API-核心概念]] | 兼容层的心理模型：协议海关、内部通行证、流式状态机 |
| 2 | [[SGLang-OpenAI-API-源码走读]] | 一个 Chat 请求从 route 到 SSE chunk 的主线 |
| 3 | [[SGLang-OpenAI-API-数据流]] | 请求、响应、usage、abort、Ollama 的状态边界 |
| 4 | [[SGLang-OpenAI-API-排障指南]] | 用症状反查源码入口 |
| 5 | [[SGLang-OpenAI-API-学习检查]] | 不打开源码时能否复述主线，打开源码时能否验证行号 |

## 首次阅读建议

第一次读只抓四个对象：

| 对象 | 作用 |
|------|------|
| `OpenAIServingBase.handle_request` | OpenAI endpoint 的模板方法 |
| `OpenAIServingChat._convert_to_internal_request` | 把 `messages/tools/reasoning` 压成 `GenerateReqInput` |
| `TokenizerManager.generate_request` | 真正进入 tokenization、scheduler、响应等待的边界 |
| `stream_offsets` + `incremental_streaming_output` | 区分内部累积快照与分段 chunk，并决定外部 delta 是否还需要切片 |

如果你在排查业务接入问题，优先读 Chat stream；如果你在排查采样参数、结构化输出或 logprobs，先读 Completion；如果你在排查客户端兼容性，单独对照 Ollama。
