---
title: "gRPC-Proto · 排障指南"
type: troubleshooting
framework: sglang
topic: "gRPC-Proto"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# gRPC-Proto · 排障指南

## 你为什么要读

gRPC 问题常跨越 Proto、Rust tonic、PyO3 bridge 和 Python RuntimeHandle。本文从“模式是否真的启用、请求在哪一侧反序列化、chunk 在哪里背压”三类证据入手，先判断故障属于协议、桥接还是共用的 TokenizerManager 以下主线。

本篇按症状排障，不按组件百科展开。每个问题都给出源码入口和验证抓手。

## Q1：传了 `--grpc-mode` 后为什么没走 HTTP？

`--grpc-mode` 当前是互斥启动模式。`run_server` 在 `server_args.grpc_mode` 为真时直接进入 `serve_grpc`，不会落到默认 HTTP `launch_server`。

```python
# 来源：python/sglang/launch_server.py L29-L35
    elif server_args.grpc_mode:
        # TODO: Once the native Rust gRPC server starts alongside HTTP in the
        # default path below (controlled by SGLANG_ENABLE_GRPC / SGLANG_GRPC_PORT),
        # remove this legacy SMG path and the grpc_mode flag.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
```

验证抓手：如果你同时期待 HTTP OpenAI endpoint 和 gRPC 主端口，当前 `--grpc-mode` 不是这个形态。它是 gRPC 独立服务加 sidecar 运维端点。

## Q2：`SGLANG_ENABLE_GRPC=1` 为什么没有看到 gRPC listener？

`SGLANG_ENABLE_GRPC` 目前只进入 `ServerArgs.__post_init__`，生成 `enable_grpc` 和 `grpc_port` 实例属性。`launch_server.py` 的默认 HTTP 分支还没有消费它。

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

判断方式：看到 `SGLANG_ENABLE_GRPC` 配置存在，不等于 Native Rust gRPC 已经随 HTTP 启动。要确认启动链路，需要在 HTTP server 分支里找到 `start_server` 调用；当前主线里还没有这个接线。

## Q3：`--grpc-mode` 报缺 `smg-grpc-servicer` 怎么办？

当前 `--grpc-mode` wrapper 依赖外部包。缺包或版本不匹配会在 import 阶段 fail-fast。

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L156-L166
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
```

依赖也出现在 Python package metadata 里：

```toml
# 来源：python/pyproject.toml L66-L70
  "sentencepiece",
  "setproctitle",
  "sgl-deep-gemm==0.1.4",
  "sglang-kernel==0.4.4",
  "smg-grpc-servicer>=0.5.0",
```

排查顺序：先确认安装环境里有 `smg-grpc-servicer[sglang]`，再看 ImportError 的 chained exception 是否指向二进制或版本漂移。

## Q4：启用 `--enable-metrics` 后为什么要求升级 servicer？

metrics 在 `--grpc-mode` 下不是 gRPC 主服务直接暴露，而是 sidecar HTTP 暴露 `/metrics`。旧版 servicer 不支持 `on_request_manager_ready` hook，sidecar 没法启动；如果用户显式启用 metrics，源码选择直接报错。

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L224-L251
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
```

验证抓手：如果核心 gRPC 请求正常但 `/metrics` 访问失败，先检查 servicer signature 和 sidecar 启动日志。

## Q5：`SGLANG_GRPC_PORT` 和 `--port` 相同为什么直接报错？

Native gRPC 预留路径要求 gRPC 端口和 HTTP 端口不同。源码只在 `enable_grpc` 为真时检查这个冲突。

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

这不是业务错误，而是端口绑定前的配置保护。

## Q6：大请求被 gRPC 拒绝，应该调哪里？

Native Rust Tonic server 的消息大小默认是 64 MiB，可用 `SGLANG_TONIC_PAYLOAD` 临时覆盖。源码注释说明它还没有提升为正式 CLI 参数。

```rust
# 来源：rust/sglang-grpc/src/server.rs L29-L58
/// 64 MiB — leaves headroom for multimodal inputs and OpenAI JSON pass-through bodies,
/// well above tonic's 4 MiB decode default.
pub const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Resolve the per-message size cap (bytes) applied to the Tonic encoder/decoder.
//
// TODO(grpc-args): promote SGLANG_TONIC_PAYLOAD to a proper `--grpc-max-message-size`
// server argument once the launcher PR (3/4) wires server args through.
fn resolve_max_message_size() -> usize {
    match std::env::var("SGLANG_TONIC_PAYLOAD") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(n) if n > 0 => {
                tracing::info!(
                    bytes = n,
                    "Using SGLANG_TONIC_PAYLOAD override for gRPC max message size"
                );
                n
            }
            _ => {
                tracing::warn!(
                    value = %raw,
                    default = DEFAULT_GRPC_MAX_MESSAGE_SIZE,
                    "Ignoring invalid SGLANG_TONIC_PAYLOAD; using default"
                );
                DEFAULT_GRPC_MAX_MESSAGE_SIZE
            }
        },
        Err(_) => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
    }
}
```

验证抓手：日志里应能看到使用 override 或忽略无效 override 的记录。多模态或大 JSON pass-through 请求优先检查这个上限。

## Q7：客户端断开后 Scheduler 一定立刻停止吗？

设计目标是沿同一 `rid` 尽快停止，但不能只凭“客户端断开”就断言已经取消成功。Rust stream drop 会触发 `RequestAbortGuard`，它调用 `bridge.abort`；Python 再把 `TokenizerManager.abort_request` 投递回其 event loop。没有 Tokio runtime、event loop 卡住或 5 秒 abort wait 超时都会留下日志，必须核验传播链。

```rust
# 来源：rust/sglang-grpc/src/server.rs L115-L139
impl Drop for RequestAbortGuard {
    fn drop(&mut self) {
        if self.armed {
            // Dropping a response stream means the client stopped consuming; propagate
            // cancellation to Python without blocking the Tokio worker.
            spawn_abort(self.bridge.clone(), self.rid.clone());
        }
    }
}

fn spawn_abort(bridge: Arc<PyBridge>, rid: String) {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            let _ = handle.spawn_blocking(move || {
                let _ = bridge.abort(&rid, false);
            });
        }
        Err(_) => {
            tracing::warn!(
                rid,
                "Skipping gRPC request abort because no Tokio runtime is available"
            );
        }
    }
}
```

如果断连后 KV 或 running request 不释放，要沿 `rid` 查四处：Rust stream 是否 drop、guard/receiver-closed 是否触发 abort、`RuntimeHandle.abort` 是否超时、`TokenizerManager.abort_request` 是否收到。

## Q8：Classify 为什么提交为 `embed`？

Proto 有独立的 `ClassifyRequest`，但 Rust handler 把它转成 embedding 类请求，并用 `req_type="embed"` 进入 Python。

```rust
# 来源：rust/sglang-grpc/src/server.rs L432-L451
    async fn classify(
        &self,
        request: Request<proto::ClassifyRequest>,
    ) -> Result<Response<proto::ClassifyResponse>, Status> {
        let req = request.into_inner();
        if req.text.is_empty() && req.input_ids.is_empty() {
            return Err(Status::invalid_argument(
                "Classify requires either text or input_ids",
            ));
        }
        let rid = req
            .rid
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        let req_dict = build_classify_dict(&rid, &req);

        let mut receiver = self
            .bridge
            .submit_request(&rid, "embed", req_dict)
            .map_err(|e| pyerr_to_status(e, "Failed to submit request"))?;
```

原因是 Python bridge 用 `EmbeddingReqInput` 承载这条内部请求，没有单独的 `req_type="classify"` 通道；外部 RPC 名仍是 Classify，内部提交类型才是 `embed`。

## Q9：Tokenize 很慢，应该先看 Rust 还是 Python？

先看 Rust tokenizer 是否加载成功。只有 Rust native tokenizer 不可用时才 fallback 到 Python。

```rust
# 来源：rust/sglang-grpc/src/server.rs L474-L493
    async fn tokenize(
        &self,
        request: Request<proto::TokenizeRequest>,
    ) -> Result<Response<proto::TokenizeResponse>, Status> {
        let req = request.into_inner();
        let add_special = req.add_special_tokens.unwrap_or(true);

        // Try Rust-native tokenizer first (no GIL)
        if let Some(tok) = self.bridge.rust_tokenizer() {
            let tokens = tok
                .encode(&req.text, add_special)
                .map_err(Status::internal)?;
            let count = tokens.len() as i32;
            return Ok(Response::new(proto::TokenizeResponse {
                tokens: tokens.iter().map(|&t| t as i32).collect(),
                count,
                max_model_len: self.bridge.context_len(),
                input_text: req.text,
            }));
        }
```

验证抓手：查日志中是否出现 Rust tokenizer loaded、disabled 或 fallback 相关信息。

## Q10：Native gRPC 可以直接公网暴露吗？

当前 Native Rust listener 源码里把认证标成尚未接入的能力。默认暴露前需要补齐 HTTP API-key/admin-key 等价检查。

```rust
# 来源：rust/sglang-grpc/src/server.rs L973-L1007
/// Start the Tonic gRPC server on the given address.
//
// TODO(grpc-auth): this listener is currently unauthenticated. Before exposing
// it in any default deploy path, gate it with the same API-key / admin-key
// checks the HTTP server applies (see issue tracking gRPC auth parity).
pub async fn run_grpc_server(
    listener: std::net::TcpListener,
    bridge: Arc<PyBridge>,
    shutdown: Arc<Notify>,
    response_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = listener.local_addr()?;
    let listener = tokio::net::TcpListener::from_std(listener)?;
    let service = SglangServiceImpl {
        bridge,
        response_timeout,
    };

    let max_message_size = resolve_max_message_size();
    let svc = proto::sglang_service_server::SglangServiceServer::new(service)
        .max_decoding_message_size(max_message_size)
        .max_encoding_message_size(max_message_size);

    tracing::info!("gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(svc)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
            shutdown.notified().await;
            tracing::info!("gRPC server shutting down");
        })
        .await?;

    Ok(())
}
```

生产判断：Native listener 更适合绑定受控内网地址，或放在带认证、TLS 和访问控制的 gateway 后。这个结论不能替代对 legacy `smg-grpc-servicer` 的版本级安全审计。

## Q11：channel 一满为什么有时只是 `Pending`，有时直接关闭？

Native bridge 允许一个 parked chunk。第一次 `try_send` 遇到 full 时，它注册 pending send 并返回 `Pending`；Python producer 应等待 on-ready。若 pending 尚未排空就再次调用 callback，说明 producer 违反了单飞契约，bridge 才记录 `ChannelFull`、关闭 stream 并触发 abort。

验证顺序：

1. 看 Python `_send_with_backpressure` 是否真的等待 `ready_event`。
2. 看 Rust 是否只存在一个 pending send。
3. 对齐 `Pending`、on-ready、`ChannelFull` 与同一 `rid` 的时间顺序。

因此不要把“队列满”直接等价成 OOM，也不要盲目把 channel capacity 调大掩盖 producer 没等待的问题。

## Q12：显式 Abort、客户端断连和 timeout 是同一条路径吗？

不是。显式 `Abort` RPC 是管理请求主动指定目标；客户端断连依赖 Tonic stream drop 或 receiver closed；stream chunk timeout 由 handler 调用 `abort_now`；Python producer 的 backpressure timeout 则只有 typed generate 直接携带 `timeout_abort_rid`。排障时必须先判断是谁发起终止，再看 abort 是否回到 TokenizerManager。

## 排障速查

| 症状 | 优先看 | 预期判断 |
|------|--------|----------|
| `--grpc-mode` 启动 import error | `grpc_server.py serve_grpc` | 外部 servicer 缺失或版本不匹配 |
| `/metrics` 不存在 | sidecar hook | servicer 版本不支持 hook 或 sidecar bind 失败 |
| `SGLANG_ENABLE_GRPC` 无 listener | `launch_server.py` | Native gRPC 尚未接入默认 HTTP 分支 |
| 大请求失败 | `SGLANG_TONIC_PAYLOAD` | message size 上限不够或 env 值无效 |
| stream 中途挂死 | `try_send_chunk` + `_send_with_backpressure` | client 慢、channel 满、on-ready 未触发 |
| 断连后请求不释放 | `RequestAbortGuard` + `RuntimeHandle.abort` | abort 没有沿 `rid` 传播 |
| channel full 后立即关闭 | parked send + producer 等待 | producer 在 `Pending` 后仍继续发送，或 on-ready 没有唤醒 |
| OpenAI pass-through 背压超时 | `_send_with_backpressure` + stream guard | Python 先停止发送，最终 abort 依赖 Rust stream/channel 收尾 |
