---
title: "gRPC-Proto · 核心概念"
type: concept
framework: sglang
topic: "gRPC-Proto"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# gRPC-Proto · 核心概念

gRPC/Proto 的核心不是“多了一套 API”，而是仓库同时存在两种不同成熟度的 gRPC 入口：当前 `--grpc-mode` 委托外部 `smg-grpc-servicer`；Native 实现则用 Proto/Tonic、PyO3 bridge 和 Python `RuntimeHandle` 组成跨语言闸门，但尚未接入默认 HTTP 启动链。下面的 Rust channel、tokenizer fast path 和 abort guard 都只描述 Native 实现。

## 读者任务

读本篇是为了建立四个判断：

1. Proto 只定义边界契约，不决定推理算法。
2. Rust gRPC crate 是 PyO3 扩展，运行时仍要拿 Python `RuntimeHandle`。
3. `--grpc-mode` 是当前可见启动路径，但它是 legacy wrapper，不等同于 Native Rust gRPC 已接入默认 HTTP 分支。
4. backpressure、abort、sidecar 是 gRPC 入口真正容易出错的地方。

## Native 实现的三层协议闸门

| 层 | 负责什么 | 不负责什么 |
|----|----------|------------|
| Proto | RPC 名、请求/响应字段、stream/unary 形态 | 不决定 Scheduler 如何调度 |
| Rust Tonic | 连接、stream、message size、callback channel、Rust tokenizer fast path | 不直接运行模型 |
| Python RuntimeHandle | 构造 `GenerateReqInput`/`EmbeddingReqInput`，调用 `TokenizerManager`，处理 OpenAI pass-through 和 admin 操作 | 不暴露 gRPC wire 细节给 Scheduler |

## Proto 分三类 RPC

Proto 文件把接口分成三类，读者应该先按用途分层，而不是按行号背 RPC 名：

```proto
# 来源：proto/sglang/runtime/v1/sglang.proto L4-L35
service SglangService {
  // SGLang-native RPCs (typed proto)
  rpc TextGenerate(TextGenerateRequest) returns (stream TextGenerateResponse);
  rpc Generate(GenerateRequest) returns (stream GenerateResponse);
  rpc TextEmbed(TextEmbedRequest) returns (TextEmbedResponse);
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc Classify(ClassifyRequest) returns (ClassifyResponse);
  rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);
  rpc Detokenize(DetokenizeRequest) returns (DetokenizeResponse);
  rpc HealthCheck(HealthCheckRequest) returns (HealthCheckResponse);
  rpc GetModelInfo(GetModelInfoRequest) returns (GetModelInfoResponse);
  rpc GetServerInfo(GetServerInfoRequest) returns (GetServerInfoResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
  rpc GetLoad(GetLoadRequest) returns (GetLoadResponse);
  rpc Abort(AbortRequest) returns (AbortResponse);
  rpc FlushCache(FlushCacheRequest) returns (FlushCacheResponse);
  rpc PauseGeneration(PauseGenerationRequest) returns (PauseGenerationResponse);
  rpc ContinueGeneration(ContinueGenerationRequest) returns (ContinueGenerationResponse);

  // OpenAI-compatible RPCs (JSON pass-through)
  rpc ChatComplete(OpenAIRequest) returns (stream OpenAIStreamChunk);
  rpc Complete(OpenAIRequest) returns (stream OpenAIStreamChunk);
  rpc OpenAIEmbed(OpenAIRequest) returns (OpenAIResponse);
  rpc OpenAIClassify(OpenAIRequest) returns (OpenAIResponse);
  rpc Score(OpenAIRequest) returns (OpenAIResponse);
  rpc Rerank(OpenAIRequest) returns (OpenAIResponse);

  // Admin/Ops RPCs
  rpc StartProfile(StartProfileRequest) returns (StartProfileResponse);
  rpc StopProfile(StopProfileRequest) returns (StopProfileResponse);
  rpc UpdateWeightsFromDisk(UpdateWeightsRequest) returns (UpdateWeightsResponse);
}
```

这段证明三件事：

- typed RPC 用结构化 proto message 表达生成、embedding、tokenize、admin。
- OpenAI-compatible RPC 不重复定义 Chat/Completion 的复杂 schema，而是 JSON pass-through。
- Admin/Ops 和业务请求在同一个 gRPC service 里，但 Python 侧会分发到不同 runtime 方法。

## TextGenerate 与 Generate 是输入编码差异

`TextGenerate` 是文本进、文本出；`Generate` 是 token ids 进、token ids 出。它们不是两套模型执行路径。

```proto
# 来源：proto/sglang/runtime/v1/sglang.proto L58-L78
message TextGenerateRequest {
  string text = 1;
  optional SamplingParams sampling_params = 2;
  optional bool stream = 3;
  optional bool return_logprob = 4;
  optional int32 top_logprobs_num = 5;
  optional int32 logprob_start_len = 6;
  optional bool return_text_in_logprobs = 7;
  optional string rid = 8;
  optional string lora_path = 9;
  optional string routing_key = 10;
  optional int32 routed_dp_rank = 11;
  map<string, string> trace_headers = 12;
  optional string session_id = 13;
}

message TextGenerateResponse {
  string text = 1;
  map<string, string> meta_info = 2;
  bool finished = 3;
}
```

```proto
# 来源：proto/sglang/runtime/v1/sglang.proto L82-L101
message GenerateRequest {
  repeated int32 input_ids = 1;
  optional SamplingParams sampling_params = 2;
  optional bool stream = 3;
  optional bool return_logprob = 4;
  optional int32 top_logprobs_num = 5;
  optional int32 logprob_start_len = 6;
  optional string rid = 7;
  optional string lora_path = 8;
  optional string routing_key = 9;
  optional int32 routed_dp_rank = 10;
  map<string, string> trace_headers = 11;
  optional string session_id = 12;
}

message GenerateResponse {
  repeated int32 output_ids = 1;
  map<string, string> meta_info = 2;
  bool finished = 3;
}
```

两者都带 `sampling_params`、`rid`、`routing_key`、`routed_dp_rank`、`trace_headers`、`session_id`，说明 gRPC typed generate 仍然要对接 SGLang 内部的请求追踪、DP 路由、LoRA 和 session 语义。

## Rust gRPC 是 PyO3 扩展

Python package 构建时把 Rust crate 编译成 `sglang.srt.grpc._core`：

```toml
# 来源：python/pyproject.toml L218-L221
[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
```

这说明 Native gRPC 的实现形态是 Python 进程内扩展，而不是另起一个完全独立的 Rust 服务。Rust 负责 Tonic 服务器和 channel；Python 仍提供 runtime handle。这里证明的是“扩展可以怎样启动”，不证明默认 HTTP 分支或 `--grpc-mode` 已经调用 `start_server`。

## 当前启动模式要分清

`ServerArgs` 里有两个相关概念：CLI 字段 `grpc_mode`，以及 `__post_init__` 里从环境变量读取的 Native gRPC 预留字段。

```python
# 来源：python/sglang/srt/server_args.py L527-L533
    # -------------------------------------------------------------------------
    # HTTP server
    # -------------------------------------------------------------------------
    host: A[str, "The host of the HTTP server."] = "127.0.0.1"
    port: A[int, "The port of the HTTP server."] = 30000
    fastapi_root_path: A[str, "App is behind a path based routing proxy."] = ""
    grpc_mode: A[bool, "If set, use gRPC server instead of HTTP server."] = False
```

```python
# 来源：python/sglang/srt/server_args.py L2910-L2923
        # Native gRPC flags — env-only for now, not exposed as CLI args.
        # Set as instance attributes (not dataclass fields) to avoid
        # argparse namespace lookup in from_cli_args.
        self.enable_grpc = envs.SGLANG_ENABLE_GRPC.get()

        grpc_port_env = envs.SGLANG_GRPC_PORT.get()
        self.grpc_port = (
            grpc_port_env if grpc_port_env is not None else self.port + 10000
        )

        if not (1 <= self.grpc_port <= 65535):
            raise ValueError(
                f"SGLANG_GRPC_PORT ({self.grpc_port}) must be between 1 and 65535"
            )
```

`grpc_mode` 是当前启动分支会消费的布尔开关；`enable_grpc/grpc_port` 是环境变量驱动的 Native gRPC 配置事实，目前从 `ServerArgs` 生成，但默认 HTTP server 分支尚未在 `launch_server.py` 里消费它。

端口冲突也只在启用 Native gRPC 时检查：

```python
# 来源：python/sglang/srt/server_args.py L7157-L7164
        if (
            self.enable_grpc
            and self.grpc_port is not None
            and self.grpc_port == self.port
        ):
            raise ValueError(
                f"SGLANG_GRPC_PORT ({self.grpc_port}) must differ from --port ({self.port})"
            )
```

## Backpressure 是双边契约

gRPC stream 的稳定性靠两个方向共同维护：

| 方向 | 源码对象 | 语义 |
|------|----------|------|
| Rust -> Python | `ChunkSendStatus.Ready/Pending/Closed` | 告诉 Python 当前 chunk 是否已经写进 channel |
| Python -> Rust | `chunk_callback(payload, finished=flag)` | 把 `TokenizerManager` chunk 交回 Rust |
| Rust channel | `response_channel_capacity` | 限制未消费响应堆积 |
| Python wait | `ready_event` | 当 Rust 返回 `Pending` 时等待 on-ready 唤醒 |

```rust
# 来源：rust/sglang-grpc/src/bridge.rs L69-L92
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkSendStatus {
    Ready,
    Pending,
    Closed,
}

fn lock_or_recover<'a, T>(mutex: &'a Mutex<T>, name: &'static str) -> MutexGuard<'a, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        tracing::warn!(mutex = name, "Recovering from poisoned gRPC bridge mutex");
        poisoned.into_inner()
    })
}

/// Holds a reference to the Python RuntimeHandle and manages per-request channels.
pub struct PyBridge {
    runtime_handle: PyObject,
    state: BridgeStateRef,
    rust_tokenizer: Option<RustTokenizer>,
    context_len: i32,
    response_channel_capacity: usize,
    tokio_handle: Handle,
}
```

如果读者只记住一个模型，就是：Python 是 producer，Rust channel 是缓冲池，gRPC client 是 consumer。channel 满时，bridge 允许一个 chunk 被 parked，并用 `Pending` 要求 Python 等待 on-ready；若 producer 不遵守契约、在 parked chunk 排空前再次发送，bridge 会以 `ChannelFull` 关闭 stream 并向 Python 传播 abort。它不是“队列一满就立即取消”，也不是无限缓存。

## terminal 与 abort 不是同一个动作

| 事件 | Native Rust 侧动作 | 是否还要 abort Python 请求 |
|------|--------------------|----------------------------|
| 收到 `ResponseChunk::Finished` | disarm guard，发送 `finished=true` | 不需要，请求已正常终止 |
| 收到 Python error chunk | disarm guard，返回 gRPC internal error | Python producer 已进入错误终态 |
| client drop / receiver closed | guard drop 或 `ClientDisconnected` | 需要，按 `rid` 传播 abort |
| 等待 chunk 超时 | stream 返回 timeout | typed generate 会触发 abort；OpenAI backpressure timeout 还依赖 guard/channel 生命周期收尾 |
| 显式 `Abort` RPC | 直接按请求参数调用 bridge/runtime | 与 client drop 是不同入口 |

因此“stream 结束”和“取消 Scheduler 请求”不能混为一谈。正常 terminal 必须 disarm，异常丢弃才应保留或主动触发 abort。

## 安全边界

Native `run_grpc_server` 当前没有接入 HTTP 的 API-key/admin-key 校验，源码仍把认证列为待接入事项。这个结论只适用于本仓库的 Native Tonic listener；legacy `smg-grpc-servicer` 的完整认证行为需要按其实际版本另行审计，不能由本文推断。

## 复盘

本专题后续所有源码都围绕这些不变量展开：

- Proto 是稳定外壳，内部仍映射到 `GenerateReqInput` 或 `EmbeddingReqInput`。
- Native Rust server 和 legacy `--grpc-mode` 都服务 gRPC，但生命周期和部署依赖不同。
- gRPC stream 的关键不是“能返回 token”，而是慢客户端、断连、超时和 terminal chunk 是否被正确治理。
- 读源码时优先追踪 `rid`，因为 channel、abort、callback、Scheduler 请求都围绕它关联。
