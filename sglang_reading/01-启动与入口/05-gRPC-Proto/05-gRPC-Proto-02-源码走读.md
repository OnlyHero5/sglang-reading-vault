---
type: batch-doc
module: 05-gRPC-Proto
batch: "05"
doc_type: walkthrough
title: "gRPC/Proto · 源码走读"
tags:
 - sglang/batch/05
 - sglang/module/grpc-proto
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# gRPC/Proto · 源码走读

> 走读顺序：`sglang.proto` → `build.rs` → `lib.rs` → `bridge.rs` → `server.rs` → `request_utils.rs` → `grpc_bridge.py` → `grpc_server.py`

---

## 1. sglang.proto — 服务契约

### 1.1 流式 Generate 消息对

**Explain：** `TextGenerate` 面向文本输入输出；`Generate` 面向 token id 输入输出。二者在 Rust 侧都映射为 `req_type="generate"`，区别仅在 `build_*_dict` 构造的字段。

**Code：**

```protobuf
# 来源：proto/sglang/runtime/v1/sglang.proto L58-L101
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

// ---- Tokenized generate (input_ids in, token_ids out) ----

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

**Comment：** `rid` 可选；Rust handler 未提供时自动生成 UUID。`trace_headers` 用于分布式追踪注入 TokenizerManager。

---

## 2. build.rs — Proto 编译

**Explain：** 构建脚本调用 `tonic-build`，从 monorepo 根 `proto/` 生成 Rust 类型与 `SglangService` server trait；同时输出 descriptor bin 供反射（若启用）。

**Code：**

```rust
// 来源：rust/sglang-grpc/build.rs L1-L16
fn main() -> Result<(), Box<dyn std::error::Error>> {
 let proto_path = "../../proto/sglang/runtime/v1/sglang.proto";

 tonic_build::configure()
 .build_server(true)
 .build_client(false)
 .protoc_arg("--experimental_allow_proto3_optional")
 .file_descriptor_set_path(
 std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
 .join("sglang_descriptor.bin"),
 )
 .compile_protos(&[proto_path], &["../../proto"])?;

 println!("cargo:rerun-if-changed={}", proto_path);
 Ok(())
}
```

**Comment：** 只 build server、不 build client——Python/Rust 进程内仅需 servicer 侧代码；gateway 使用独立的 `smg_grpc_client`。

---

## 3. lib.rs — Python 入口 `start_server`

### 3.1 模块导出与 Proto include

**Code：**

```rust
// 来源：rust/sglang-grpc/src/lib.rs L1-L8
pub mod bridge;
pub mod server;
pub mod tokenizers;
pub(crate) mod utils;

pub mod proto {
 tonic::include_proto!("sglang.runtime.v1");
}
```

### 3.2 启动流程：绑定、Tokio、PyBridge、后台线程

**Explain：** `start_server` 是 PyO3 暴露给 Python 的唯一启动函数：解析地址、提取 tokenizer 信息、创建 `PyBridge`、在名为 `sglang-grpc` 的 std thread 中 `block_on` Tonic。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/lib.rs L141-L257（节选）
#[pyfunction]
#[pyo3(signature = (host, port, runtime_handle, worker_threads=4, response_channel_capacity=64, response_timeout_secs=300))]
fn start_server(
 host: String,
 port: u16,
 runtime_handle: PyObject,
 worker_threads: usize,
 response_channel_capacity: usize,
 response_timeout_secs: u64,
) -> PyResult<GrpcServerHandle> {
 let addr: SocketAddr = format!("{}:{}", host, port).parse()...;
 let listener = TcpListener::bind(addr)...;

 let tokenizer_info = extract_tokenizer_info(&runtime_handle)?;
 let rust_tokenizer = tokenizer_info.tokenizer_path.as_deref().and_then(|p| {
 RustTokenizer::from_tokenizer_path(p, tokenizer_info.tokenizer_mode.as_deref(), ...)
 });

 let rt = tokio::runtime::Builder::new_multi_thread()
 .worker_threads(worker_threads)
 .enable_all()
 .thread_name("sglang-grpc-tokio")
 .build()...;

 let bridge = Arc::new(PyBridge::new(
 runtime_handle,
 rust_tokenizer,
 tokenizer_info.context_len,
 response_channel_capacity,
 rt.handle().clone(),
 ));

 let join_handle = std::thread::Builder::new()
 .name("sglang-grpc".to_string())
 .spawn(move || {
 rt.block_on(server::run_grpc_server(
 listener, bridge_clone, shutdown_clone, response_timeout,
 ))
 })...;

 Ok(GrpcServerHandle { shutdown, join_handle: Some(join_handle) })
}
```

**Comment：**
- `extract_tokenizer_info` 一次性拿 GIL 读 `runtime_handle.tokenizer_manager`，供 Rust 原生 Tokenize。
- `GrpcServerHandle.shutdown()` 通过 `Notify` 优雅停止 Tonic。
- gRPC 与 Python asyncio **不同事件循环**：Rust Tokio + Python TM loop 通过 callback + `spawn_blocking` 协作。

---

## 4. bridge.rs — PyBridge 与 ChunkCallback

### 4.1 响应 chunk 类型

**Code：**

```rust
// 来源：rust/sglang-grpc/src/bridge.rs L13-L33
#[derive(Debug, Clone)]
pub enum ResponseChunk {
 Data(ResponseData),
 Finished(ResponseData),
 Error(String),
}

#[derive(Debug, Clone)]
pub struct ResponseData {
 pub text: Option<String>,
 pub output_ids: Option<Vec<i32>>,
 pub embedding: Option<Vec<f32>>,
 pub json_bytes: Option<Vec<u8>>,
 pub meta_info: HashMap<String, String>,
}
```

### 4.2 submit_request — Proto dict → Python

**Explain：** 每个 gRPC 请求创建独立 `mpsc` channel；Rust 持 `Receiver`，Python 通过 `ChunkCallback` 往 `Sender` 推 chunk。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/bridge.rs L177-L207
 pub fn submit_request(
 &self,
 rid: &str,
 req_type: &str,
 req_dict: HashMap<String, serde_json::Value>,
 ) -> PyResult<Receiver<ResponseChunk>> {
 let receiver = self.create_channel(rid)?;
 let rid_owned = rid.to_string();

 let result = Python::with_gil(|py| -> PyResult<()> {
 let py_req_dict = json_map_to_pydict(py, &req_dict)?;
 let callback = self.make_chunk_callback(py, rid_owned)?;

 let kwargs = PyDict::new(py);
 kwargs.set_item("req_type", req_type)?;
 kwargs.set_item("req_dict", py_req_dict)?;
 kwargs.set_item("chunk_callback", callback)?;

 self.runtime_handle
 .call_method(py, "submit_request", (), Some(&kwargs))?;
 Ok(())
 });
 ...
 }
```

### 4.3 ChunkCallback.__call__ — Python → Rust 背压

**Explain：** 当 channel 满时返回 `ChunkSendStatus::Pending`，Python 侧 `_send_with_backpressure` 等待 `on_ready` 再发下一块——防止 Rust 无限缓冲拖垮内存。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/bridge.rs L630-L694（节选）
 #[pyo3(signature = (chunk, finished=false, error=None))]
 fn __call__(
 &self,
 chunk: &Bound<'_, PyDict>,
 finished: bool,
 error: Option<String>,
 ) -> PyResult<ChunkSendStatus> {
 ...
 let data = ResponseData {
 text,
 output_ids,
 embedding,
 json_bytes: None,
 meta_info,
 };
 let msg = if finished {
 ResponseChunk::Finished(data)
 } else {
 ResponseChunk::Data(data)
 };
 try_send_chunk(py, &self.rid, &self.state, ..., &sender, msg)
 }
```

---

## 5. server.rs — Tonic Service 实现

### 5.1 text_generate 流式 handler

**Explain：** 典型流式 RPC：提交请求 → async_stream 循环 recv channel → 映射为 `TextGenerateResponse`；客户端断开或超时触发 `RequestAbortGuard` 调用 Python `abort`。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/server.rs L222-L287（节选）
 async fn text_generate(
 &self,
 request: Request<proto::TextGenerateRequest>,
 ) -> Result<Response<Self::TextGenerateStream>, Status> {
 let req = request.into_inner();
 let rid = req.rid.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
 let req_dict = build_text_generate_dict(&rid, &req);

 let mut receiver = self.bridge
 .submit_request(&rid, "generate", req_dict)
 .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;

 let stream = async_stream::stream! {
 let mut abort_guard = RequestAbortGuard::new(bridge.clone(), rid_clone.clone());
 loop {
 match recv_chunk_with_timeout(&mut receiver, response_timeout, ...).await {
 Ok(Some(ResponseChunk::Data(data))) => {
 yield Ok(proto::TextGenerateResponse {
 text: data.text.unwrap_or_default(),
 meta_info: data.meta_info,
 finished: false,
 });
 }
 Ok(Some(ResponseChunk::Finished(data))) => {
 abort_guard.disarm();
 yield Ok(proto::TextGenerateResponse { ..., finished: true });
 break;
 }
 ...
 }
 }
 };
 Ok(Response::new(Box::pin(stream)))
 }
```

### 5.2 run_grpc_server — 消息大小与优雅关闭

**Code：**

```rust
// 来源：rust/sglang-grpc/src/server.rs L978-L1006
pub async fn run_grpc_server(
 listener: std::net::TcpListener,
 bridge: Arc<PyBridge>,
 shutdown: Arc<Notify>,
 response_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
 let max_message_size = resolve_max_message_size(); // 默认 64 MiB，可用 SGLANG_TONIC_PAYLOAD 覆盖
 let svc = proto::sglang_service_server::SglangServiceServer::new(service)
 .max_decoding_message_size(max_message_size)
 .max_encoding_message_size(max_message_size);

 tonic::transport::Server::builder()
 .add_service(svc)
 .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
 shutdown.notified().await;
 })
 .await?;
 Ok(())
}
```

**Comment：** 默认 64 MiB 是为多模态与 OpenAI JSON body 留余量；Tonic 默认仅 4 MiB。

### 5.3 tokenize — Rust 优先、Python 回退

**Code：**

```rust
// 来源：rust/sglang-grpc/src/server.rs L474-L519（节选）
 async fn tokenize(&self, request: Request<proto::TokenizeRequest>) -> ... {
 let req = request.into_inner();
 let add_special = req.add_special_tokens.unwrap_or(true);

 if let Some(tok) = self.bridge.rust_tokenizer() {
 let tokens = tok.encode(&req.text, add_special).map_err(Status::internal)?;
 return Ok(Response::new(proto::TokenizeResponse { tokens: ..., count, ... }));
 }

 // Fallback to Python
 let json_str = tokio::task::spawn_blocking({
 let bridge = self.bridge.clone();
 move || bridge.tokenize_py(&text, add_special)
 }).await...;
 ...
 }
```

---

## 6. request_utils.rs — Proto → Python dict

**Explain：** `build_text_generate_dict` 把 Proto 字段转为与 `GenerateReqInput` 兼容的键值，并附加 `received_time` 供延迟统计。

**Code：**

```rust
// 来源：rust/sglang-grpc/src/utils/request_utils.rs L90-L138
pub(crate) fn build_text_generate_dict(
 rid: &str,
 req: &proto::TextGenerateRequest,
) -> HashMap<String, serde_json::Value> {
 let mut d = HashMap::new();
 d.insert("rid".into(), serde_json::json!(rid));
 d.insert("text".into(), serde_json::json!(req.text));
 d.insert("sampling_params".into(), sampling_params_to_map(&req.sampling_params));
 d.insert("stream".into(), serde_json::json!(req.stream.unwrap_or(false)));
 d.insert("return_logprob".into(), serde_json::json!(req.return_logprob.unwrap_or(false)));
 ...
 if let Some(trace) = trace_headers_to_json(&req.trace_headers) {
 d.insert("external_trace_header".into(), trace);
 }
 d.insert("received_time".into(), serde_json::json!(now_timestamp()));
 d
}
```

---

## 7. grpc_bridge.py — RuntimeHandle

### 7.1 submit_request 分发

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_bridge.py L262-L291
    def submit_request(
        self,
        *,
        req_type: str,
        req_dict: dict,
        chunk_callback,
        is_disconnected_fn: Optional[Callable[[], bool]] = None,
    ):
        mock_request = (
            _GrpcRequest(is_disconnected_fn=is_disconnected_fn)
            if is_disconnected_fn is not None
            else None
        )
        if req_type == "generate":
            from sglang.srt.managers.io_struct import GenerateReqInput

            obj = GenerateReqInput(**req_dict)
            stream = req_dict.get("stream", False)
            self._submit_on_tm_loop(
                self._run_generate(obj, chunk_callback, stream, mock_request)
            )
        elif req_type == "embed":
            from sglang.srt.managers.io_struct import EmbeddingReqInput

            obj = EmbeddingReqInput(**req_dict)
            self._submit_on_tm_loop(self._run_embed(obj, chunk_callback, mock_request))
        else:
            raise ValueError(
                f"Unknown req_type: {req_type!r} (expected 'generate' or 'embed')"
            )
```

### 7.2 流式 generate 与 backpressure

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_bridge.py L293-L324
    async def _run_generate(self, obj, chunk_callback, stream: bool, request):
        ready_event = None
        try:
            ready_event = self._install_on_ready(chunk_callback) if stream else None
            gen = self.tokenizer_manager.generate_request(obj, request=request)
            if stream:
                async for chunk in gen:
                    finished = (
                        chunk.get("meta_info", {}).get("finish_reason") is not None
                    )
                    keep_going = await self._send_with_backpressure(
                        chunk_callback,
                        ready_event,
                        chunk,
                        finished=finished,
                        timeout_abort_rid=obj.rid,
                    )
                    if finished or not keep_going:
                        return
                # Defensive: generator exited without a finish_reason chunk.
                self._safe_callback(chunk_callback, {}, finished=True)
            else:
                result = await gen.__anext__()
                self._safe_callback(chunk_callback, result, finished=True)
        except StopAsyncIteration:
            self._safe_callback(chunk_callback, {}, finished=True)
        except Exception as e:
            logger.error("gRPC generate error for rid=%s: %s", obj.rid, e)
            self._send_native_error(chunk_callback, str(e))
        finally:
            if stream:
                self._uninstall_on_ready(chunk_callback)
```

**Comment：** `_submit_on_tm_loop` 把 coroutine 投递到 TokenizerManager 的 asyncio loop，避免 Rust 线程阻塞等待 GPU。

---

## 8. grpc_server.py — Legacy 封装与 Sidecar

### 8.1 serve_grpc 主函数

**Explain：** 导入 `smg_grpc_servicer`，注册 `on_request_manager_ready` 回调以启动 HTTP sidecar；旧版 servicer 无此 hook 时会禁用 sidecar 或抛错。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L156-L254（节选）
async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler."""
    try:
        from smg_grpc_servicer.sglang.server import serve_grpc as _serve_grpc
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e

    sidecar_app = web.Application()
    sidecar_runner = None
    sidecar_port = (
        server_args.grpc_http_sidecar_port
        if server_args.grpc_http_sidecar_port is not None
        else server_args.port + 1
    )

    # Metrics setup: must set PROMETHEUS_MULTIPROC_DIR before scheduler
    # processes import prometheus_client, since the env var is inherited
    # at fork time.
    if server_args.enable_metrics:
        try:
            from sglang.srt.observability.func_timer import enable_func_timer
            from sglang.srt.utils import set_prometheus_multiproc_dir

            set_prometheus_multiproc_dir()
            enable_func_timer()
            _add_metrics_routes(sidecar_app)
        except Exception as e:
            logger.error(
                "Failed to set up metrics: %s. Continuing without metrics.",
                e,
                exc_info=True,
            )

    async def _on_request_manager_ready(request_manager, srv_args, sched_info):
        nonlocal sidecar_runner
        try:
            _add_admin_routes(sidecar_app, request_manager)
        except Exception as e:
            logger.error(
                "Failed to set up admin routes: %s. "
                "Continuing without admin endpoints.",
                e,
                exc_info=True,
            )
        try:
            sidecar_runner = await _start_sidecar_server(
                server_args.host, sidecar_port, sidecar_app
            )
        except OSError as e:
            logger.error(
                "Failed to start HTTP sidecar server: %s. "
                "Continuing without metrics/profile endpoints.",
                e,
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                "Unexpected error starting HTTP sidecar server: %s. "
                "Continuing without metrics/profile endpoints.",
                e,
                exc_info=True,
            )

    # Older smg-grpc-servicer releases (≤ 0.5.2) accept only (server_args,
    # model_info) and reject the on_request_manager_ready hook. The hook is
    # what calls _start_sidecar_server, so dropping the kwarg disables the
    # entire HTTP sidecar (Prometheus /metrics and /start_profile +
    # /stop_profile). Core gRPC serving still works without it.
    serve_kwargs: dict = {}
    sidecar_supported = (
        "on_request_manager_ready" in inspect.signature(_serve_grpc).parameters
    )
    if sidecar_supported:
        serve_kwargs["on_request_manager_ready"] = _on_request_manager_ready
    elif server_args.enable_metrics:
        # User explicitly asked for metrics but the installed servicer can't
        # start the sidecar that serves them — fail loud rather than silently
        # produce a server with no /metrics endpoint.
        raise RuntimeError(
            "--enable-metrics requires smg-grpc-servicer ≥ 0.5.3 (the version "
            "that accepts 'on_request_manager_ready'); installed version "
            "lacks the hook so the HTTP sidecar would never start. Upgrade "
            "smg-grpc-servicer or remove --enable-metrics."
        )
    else:
        logger.warning(
            "Installed smg-grpc-servicer does not accept "
            "'on_request_manager_ready'; HTTP sidecar disabled "
            "(no /metrics, /start_profile, /stop_profile). "
            "Upgrade smg-grpc-servicer to ≥ 0.5.3 to enable it."
        )

    try:
        await _serve_grpc(server_args, model_info, **serve_kwargs)
```

### 8.2 Profile 路由（sidecar 复用 HTTP 语义）

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L107-L127
            req = ProfileReq(
                req_type=ProfileReqType.START_PROFILE,
                output_dir=body.get("output_dir"),
                start_step=body.get("start_step"),
                num_steps=body.get("num_steps"),
                activities=body.get("activities"),
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=body.get("profile_by_stage", False),
                profile_id=str(time.time()),
                merge_profiles=body.get("merge_profiles", False),
                profile_prefix=body.get("profile_prefix"),
                profile_stages=body.get("profile_stages"),
            )
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Start Profile")
            if err:
                return err
            return web.Response(text="Start profiling.\n")
```

**Comment：** gRPC 主端口不提供 REST profile API；运维通过 sidecar HTTP POST `/start_profile` 触发，与 HTTP 模式行为对齐。

---

## 走读小结

| 步骤 | 组件 | 关键动作 |
|------|------|----------|
| 1 | `sglang.proto` | 定义 RPC 与 message |
| 2 | `build.rs` | tonic-build 生成 Rust 类型 |
| 3 | `server.rs` | Tonic handler 收请求、开 stream |
| 4 | `request_utils.rs` | Proto → JSON dict |
| 5 | `bridge.rs` | dict + callback → Python |
| 6 | `grpc_bridge.py` | GenerateReqInput → TokenizerManager |
| 7 | `bridge.rs` | chunk 回写 mpsc → gRPC stream |
| 8 | `grpc_server.py` | （legacy）servicer 生命周期 + sidecar |
