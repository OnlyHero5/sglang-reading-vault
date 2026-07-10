---
title: "TokenizerManager"
type: map
framework: sglang
topic: "TokenizerManager"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# TokenizerManager

## 你为什么要读

TokenizerManager 是 SGLang 请求链路里的前台调度台：它不做 GPU forward，也不做 continuous batching；它负责把 API 请求变成 Scheduler 能消费的 tokenized IPC 对象，再把 Detokenizer 或 Scheduler 回来的批输出拆回每个 HTTP 请求。

读完本专题，读者应该能解决三类问题：

| 读者任务 | 能回答的问题 |
|----------|--------------|
| 首次读源码 | 一个 `/generate` 请求如何从 FastAPI 进入 Scheduler，又如何回到 SSE/JSON |
| 排查线上问题 | 为什么请求在 pause/weight update 时卡住、为什么流式中间包可能没有完整 text、为什么 skip-tokenizer 不能传 text |
| 准备改代码 | 哪些状态属于 `ReqState`，哪些控制面操作走 `FanOutCommunicator`，多 HTTP worker 如何保证回包不串线 |

## 模块位置

```mermaid
flowchart LR
    HTTP["HTTP / OpenAI API / Engine"]
    TM["TokenizerManager<br/>前台请求状态机"]
    SCH["Scheduler<br/>GPU batch 与 KV 预算"]
    DT["DetokenizerManager<br/>token ids 到文本"]

    HTTP -->|"GenerateReqInput / EmbeddingReqInput"| TM
    TM -->|"TokenizedGenerateReqInput<br/>TokenizedEmbeddingReqInput"| SCH
    SCH -->|"BatchTokenIDOutput"| DT
    DT -->|"BatchStrOutput"| TM
    SCH -->|"BatchTokenIDOutput<br/>skip_tokenizer_init=True"| TM
    TM -->|"dict / SSE chunk"| HTTP
```

核心主线是：

```text
GenerateReqInput
  -> ReqState
  -> TokenizedGenerateReqInput
  -> Scheduler
  -> BatchStrOutput 或 BatchTokenIDOutput
  -> ReqState.out_list + event
  -> HTTP yield
```

TokenizerManager 的关键不是“会分词”，而是同时维护三条边界：

| 边界 | TokenizerManager 做什么 | 不做什么 |
|------|--------------------------|----------|
| API 到 Scheduler | normalize、分词、多模态处理、采样参数校验、LoRA 解析、IPC 发送 | 不决定 GPU batch 准入 |
| 后端输出到 HTTP | 按 `rid` 找 `ReqState`、累加文本/token ids、设置 event、yield chunk | 不负责 token id 到字符串的增量 decode，除非 skip tokenizer bypass |
| 数据面到控制面 | 数据面按 `rid` 多路复用；控制面用 communicator 等待 Scheduler rank 回复 | 不把权重更新、flush cache 当作普通 generate 请求 |

## 阅读顺序

| 顺序 | 文档 | 读者目标 |
|------|------|----------|
| 1 | [[SGLang-TokenizerManager-核心概念]] | 建立“双协程调度台”和 `ReqState` 心理模型 |
| 2 | [[SGLang-TokenizerManager-源码走读]] | 沿一条 generate 请求读源码证据 |
| 3 | [[SGLang-TokenizerManager-数据流]] | 看清对象形态、IPC 路由和多 worker 分叉 |
| 4 | [[SGLang-TokenizerManager-排障指南]] | 用症状表定位 pause、streaming、skip-tokenizer、abort 等问题 |
| 5 | [[SGLang-TokenizerManager-学习检查]] | 自检是否能画图、追生命周期、设计验证实验 |

## 源码范围

| upstream 文件 | 读法 |
|---------------|------|
| `sglang/python/sglang/srt/managers/tokenizer_manager.py` | 主线文件：`generate_request`、分词、发送、等待、收包、`ReqState` |
| `sglang/python/sglang/srt/managers/io_struct.py` | API 对象和 IPC 对象形态 |
| `sglang/python/sglang/srt/managers/tokenizer_control_mixin.py` | 权重、cache、profile 等控制面 fan-out |
| `sglang/python/sglang/srt/managers/tokenizer_manager_score_mixin.py` | score API 如何复用 generate/embedding 数据面 |
| `sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py` | 多 HTTP worker 的 router、回包拆分和 pause broadcast |
| `sglang/python/sglang/srt/entrypoints/http_server.py` | FastAPI 如何消费 `generate_request` async generator |
| `sglang/python/sglang/srt/server_args.py` | `skip_tokenizer_init`、`batch_notify_size`、`incremental_streaming_output`、IPC port |

## 和相邻专题的关系

| 上一跳 | 本专题 | 下一跳 |
|--------|--------|--------|
| [[SGLang-OpenAI-API]] 把 OpenAI/Ollama/HTTP 请求转成内部 input object | TokenizerManager 注册状态、分词、发送、等待输出 | [[SGLang-ScheduleBatch数据结构]] 和 [[SGLang-Detokenizer]] 解释请求进入 Scheduler 后的对象形态和回程 |

如果只想先跑通请求链路，读 `01 -> 02 -> 05` 即可；如果在排查生产流式输出或多 worker 问题，直接读 `03 -> 04`。
