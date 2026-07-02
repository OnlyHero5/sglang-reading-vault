---
type: module-moc
module: 04-OpenAI-API
batch: "04"
doc_type: moc
title: "OpenAI API 兼容层"
tags:
 - sglang/batch/04
 - sglang/module/openai-api
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# OpenAI API 兼容层

> 阶段 I · 地基 | 源码范围：`srt/entrypoints/openai/`、`srt/entrypoints/ollama/`

## 本模块目标

读完本目录下全部文档后，你应能**不打开 `sglang/` 源码目录**，回答：

1. SGLang 如何把 OpenAI 格式的 HTTP 请求转成内部 `GenerateReqInput` / `EmbeddingReqInput`？
2. `OpenAIServingBase.handle_request` 的统一模板是什么？各 endpoint 如何扩展？
3. Chat / Completion / Embedding / Ollama 四条路径的差异与共同点？
4. 流式响应（SSE vs NDJSON）在哪一层组装？

## 文档导航

| 文件 | 内容 |
|------|------|
| [[04-OpenAI-API-01-核心概念]] | 协议层、Serving 层、Ollama 适配 |
| [[04-OpenAI-API-02-源码走读]] | 按调用顺序精读核心类与函数 |
| [[04-OpenAI-API-03-数据流与交互]] | HTTP → Serving → TokenizerManager 数据流 |
| [[04-OpenAI-API-04-关键问题]] | FAQ、易错点、与原生 `/generate` 对比 |
| [[04-OpenAI-API-05-checkpoint]] | 验收清单 |

## 源码范围一览

| 目录 | 职责 |
|------|------|
| `openai/protocol.py` | OpenAI 兼容 Pydantic 模型（请求/响应/错误） |
| `openai/serving_base.py` | 所有 OpenAI handler 的抽象基类 |
| `openai/serving_*.py` | 各 endpoint 具体实现（chat、completion、embedding 等） |
| `openai/sse_utils.py` | Chat 流式 SSE chunk 构建 |
| `openai/usage_processor.py` | Token 用量统计 |
| `ollama/protocol.py` | Ollama API Pydantic 模型 |
| `ollama/serving.py` | Ollama → SGLang 转换与响应封装 |

路由注册在HTTP Server 的 `http_server.py` 中；本模块聚焦 **协议转换与 Serving 逻辑**。

## 最关键的一段入口代码

**Explain：** HTTP Server 启动时在 `init_app_state` 中实例化各 OpenAI Serving handler，并挂到 FastAPI `app.state`。路由函数只做一件事：把请求委托给对应 handler 的 `handle_request`。

**Code：**

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

**Comment：**

- 所有 handler 共享同一个 `TokenizerManager`——真正发 ZMQ 请求、收 detokenized 输出的是它（TokenizerManager 详述）。
- `serving_chat_class` 允许模型定制 Chat handler（如 DeepSeek 专用 parser），默认是 `OpenAIServingChat`。
- Ollama 不走 `OpenAIServingBase`，但同样调用 `tokenizer_manager.generate_request`。

## 典型请求路径（一句话）

```
POST /v1/chat/completions
 → openai_v1_chat_completions()
 → OpenAIServingChat.handle_request()
 → _convert_to_internal_request() → GenerateReqInput
 → tokenizer_manager.generate_request()
 → SSE / JSON 响应
```

## 阅读路径

← [[03-HTTP-Server-00-MOC|HTTP Server]]（路由挂载、Engine 初始化）

→ [[05-gRPC-Proto-00-MOC|gRPC/Proto：gRPC 与 Proto]]
