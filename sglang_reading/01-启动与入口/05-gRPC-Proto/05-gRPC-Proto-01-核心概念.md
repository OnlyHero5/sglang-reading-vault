---
type: batch-doc
module: 05-gRPC-Proto
batch: "05"
doc_type: concept
title: "gRPC/Proto · 核心概念"
tags:
 - sglang/batch/05
 - sglang/module/grpc-proto
 - sglang/doc/concept
aliases:
 - "01-核心概念"
updated: 2026-07-02
---
# gRPC/Proto · 核心概念

---

## 用户故事：网关工程师对接 `TextGenerate` 流式

### Persona

**陈工**，sgl-model-gateway 维护者，需要让 Rust 网关通过 gRPC 调 SGLang 原生 `TextGenerate` RPC，把 token 流式转发给上游编排系统，但客户端只收到首包后 stream 挂起。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | 网关 stub 调 `SglangService.TextGenerate`，Proto 字段与 `SamplingParams` 对齐 |
| T1 | Rust `SglangServiceImpl` 经 PyBridge 调 `RuntimeHandle.submit_*`，首 chunk 正常 |
| T2 | 后续 chunk 阻塞——排查 `ChunkCallback` 与 mpsc backpressure |
| T3 | 确认 Python 每产出一 chunk 即 `__call__(chunk_dict, finished, error)`，网关侧按 `finished` 关 stream |

**Explain：** gRPC 层是 **Proto 契约 + Rust wire + Python RuntimeHandle** 三段式。`TextGenerate` 属于 **SGLang-native typed RPC**（非 OpenAI pass-through）：Rust `request_utils` 把 Proto 转 JSON dict，Python 构造 `GenerateReqInput` 并驱动与 HTTP 相同的 TM/Scheduler 链路。流式语义由 **server-side stream + ChunkCallback** 驱动，与 HTTP SSE 封装层不同但下游 IO 结构一致。

**Code：**

```protobuf
# 来源：proto/sglang/runtime/v1/sglang.proto L84-L86
  optional SamplingParams sampling_params = 2;
  optional bool stream = 3;
  optional bool return_logprob = 4;
```

**Comment：** `--grpc-mode` 时整站走 gRPC 主端口；Native 伴生模式见 `SGLANG_ENABLE_GRPC`（gRPC/Proto §3.2）。OpenAI 兼容可走 `ChatComplete` pass-through 复用OpenAI API。

### 如果…会怎样（调试）

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| Stream 首包后无数据 | ChunkCallback 未持续 invoke 或 Rust 未 poll channel | 查 `grpc_bridge.py` 与 PyO3 callback 日志 |
| 连错端口 | `--grpc-mode` 下 `--port` 是 gRPC 非 HTTP | metrics 走 sidecar `port+1` |
| 字段映射缺失 | Proto optional 未 set 不进 Python dict | 对照 `sampling_params_to_map` 输出 |

---

## 1. 架构位置

SGLang 对外暴露请求的入口层有 **HTTP（FastAPI）** 与 **gRPC（Tonic）** 两条通道。gRPC 层位于 **入口层与调度层之间**：

- **上方**：gRPC 客户端、sgl-model-gateway（Rust 网关，model-gateway）、外部推理编排系统。
- **下方**：`TokenizerManager` → `Scheduler`（TokenizerManager–Scheduler），与 HTTP 路径共享同一套 IO 结构（`GenerateReqInput` 等）。

Proto 文件是 **跨语言契约**；Rust 服务器负责 wire 协议与流控；Python `RuntimeHandle` 负责把 Proto 字段映射为内部 dataclass 并驱动异步生成。

---

## 2. 核心术语

| 术语 | 含义 |
|------|------|
| **SglangService** | `sglang.proto` 中定义的 gRPC 服务，含 Generate/Embed/Health 等 20+ RPC |
| **Tonic** | Rust gRPC 框架；`SglangServiceImpl` 实现 generated trait |
| **PyBridge** | Rust 结构体，持有 Python `RuntimeHandle`，管理 per-request `mpsc` channel |
| **RuntimeHandle** | Python 类（`grpc_bridge.py`），Rust 通过 PyO3 同步调用其 `submit_*` 方法 |
| **ChunkCallback** | PyO3 回调对象；Python 每产出一个 chunk 就 `__call__(chunk_dict, finished, error)` |
| **smg-grpc-servicer** | 外部 PyPI 包；`--grpc-mode` 时负责启动调度器 + Rust gRPC + 生命周期 |
| **HTTP sidecar** | gRPC 模式下伴生的 aiohttp 小服务，暴露 `/metrics`、`/start_profile` |
| **OpenAI pass-through** | Proto 中 `OpenAIRequest { bytes json_body }` 模式，Rust 不解析 JSON，原样交给 Python OpenAI serving |

---

## 3. 两类 gRPC 部署模式

### 3.1 `--grpc-mode`（Legacy 独立模式）

**Explain：** CLI 布尔标志 `grpc_mode=True` 时，`launch_server` **不启动 HTTP**，整站以 gRPC 对外。Python 侧仅 `grpc_server.serve_grpc` 薄封装，核心逻辑在外部 `smg-grpc-servicer` 包。

**Code：**

```python
# 来源：python/sglang/srt/server_args.py L530-L533
    host: A[str, "The host of the HTTP server."] = "127.0.0.1"
    port: A[int, "The port of the HTTP server."] = 30000
    fastapi_root_path: A[str, "App is behind a path based routing proxy."] = ""
    grpc_mode: A[bool, "If set, use gRPC server instead of HTTP server."] = False
```

**Comment：** 此模式下 `--port` 语义变为 gRPC 监听端口（由 servicer 解释）；HTTP sidecar 默认 `--port + 1`。

### 3.2 `SGLANG_ENABLE_GRPC`（Native 伴生模式，演进中）

**Explain：** 通过环境变量启用进程内 Rust gRPC，默认端口 `HTTP port + 10000`。目前 **尚未在 `launch_server` 默认 HTTP 分支接线**，但 `server_args` 已预留校验逻辑；Rust `start_server` + `RuntimeHandle` 是为该路径准备的。

**Code：**

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

**Comment：** 与 `--grpc-mode` 互斥的设计意图是：最终 HTTP + gRPC 双栈并存，而非二选一。

---

## 4. Proto 的三层 RPC 设计

**Explain：** `sglang.proto` 将 RPC 分为 **SGLang 原生 typed**、**OpenAI JSON pass-through**、**Admin/Ops** 三层，避免为每个 OpenAI 端点单独维护 message 类型。

**Code：**

```protobuf
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

**Comment：**
- **Typed RPC**：Rust `request_utils` 把 Proto 字段转为 JSON dict，Python 构造 `GenerateReqInput`。
- **Pass-through**：`json_body` 字节流直接进 `submit_openai_chat` 等，复用OpenAI API 的 OpenAI 层。
- **Admin**：Profile、权重热更新等运维操作，走 JSON 回调通道。

---

## 5. Rust PyO3 扩展的打包方式

**Explain：** gRPC 核心不在纯 Python，而是 `rust/sglang-grpc` crate，构建时由 `tonic-build` 编译 proto，运行时作为 `sglang.srt.grpc._core` 导入。

**Code：**

```toml
# 来源：python/pyproject.toml（setuptools-rust 段，与阅读方法论 一致）
[[tool.setuptools-rust.ext-modules]]
target = "sglang.srt.grpc._core"
path = "../rust/sglang-grpc/Cargo.toml"
binding = "PyO3"
```

**Comment：** Python 调用 `from sglang.srt.grpc._core import start_server, GrpcServerHandle` 即可在独立线程启动 Tonic 服务器；`RuntimeHandle` 作为构造参数传入。

---

## 6. 共享采样参数 `SamplingParams`

**Explain：** Generate 类 RPC 共用 `SamplingParams` message，Rust 侧 `sampling_params_to_map` 将其转为 Python dict 的 `sampling_params` 字段。

**Code：**

```protobuf
# 来源：proto/sglang/runtime/v1/sglang.proto L37-L54
// Sampling parameters shared across text and tokenized RPCs.
message SamplingParams {
  optional float temperature = 1;
  optional float top_p = 2;
  optional int32 top_k = 3;
  optional float min_p = 4;
  optional float frequency_penalty = 5;
  optional float presence_penalty = 6;
  optional float repetition_penalty = 7;
  optional int32 max_new_tokens = 8;
  optional int32 min_new_tokens = 9;
  repeated string stop = 10;
  repeated int32 stop_token_ids = 11;
  optional bool ignore_eos = 12;
  optional int32 n = 13;
  optional string json_schema = 14;
  optional string regex = 15;
}
```

**Comment：** `optional` /proto3 需要 `--experimental_allow_proto3_optional`（见 `build.rs`）；未设置的字段不会出现在 Python dict 中。

---

## 7. 设计动机小结

1. **性能**：Rust 处理 wire 协议、流式 backpressure、Tokenize/Detokenize（可选无 GIL），Python 专注推理调度。
2. **契约稳定**：Proto 作为 gateway、多语言 client、Python server 的共同 schema。
3. **复用 HTTP 逻辑**：OpenAI pass-through 与 `RuntimeHandle.submit_openai_*` 避免重复实现采样/模板/流式 SSE 转换。
4. **运维友好**：gRPC 主端口 + HTTP sidecar 分离 metrics/profile，避免在 gRPC 上嵌 HTTP/2 混用。
