---
type: batch-doc
module: 27-model-gateway
batch: "27"
doc_type: faq
title: "model-gateway：关键问题"
tags:
 - sglang/batch/27
 - sglang/module/model-gateway
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# model-gateway：关键问题

---

## 1. Gateway 与 srt 内置 HTTP server 如何选型？

**Explain：** 单模型、单副本部署可直接 `python -m sglang.launch_server`，无需 gateway。多副本、PD 分离、多模型统一入口、熔断重试、K8s service discovery 场景应前置 `smg`。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/factory.rs L44-L45
 ConnectionMode::Http => match &ctx.router_config.mode {
 RoutingMode::Regular { .. } => Self::create_regular_router(ctx).await,
```

**Comment：**

- Gateway 增加 ~1ms 代理延迟，换取运维能力。
- 也可 srt 与 gateway 同 pod：gateway 监听 30000，srt 监听 8000。

---

## 2. 为什么 PD readiness 要求两种 worker？

**Explain：** PD 路由假设 prefill 与 decode 是不同进程/池。若只有 prefill healthy，请求会卡在无法 decode；只有 decode 则无法 bootstrap KV。

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L110-L117
 let has_prefill = healthy_workers
 .iter()
 .any(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }));
 let has_decode = healthy_workers
 .iter()
 .any(|w| matches!(w.worker_type(), WorkerType::Decode));
 has_prefill && has_decode
```

**Comment：**

- K8s readiness probe 应指向 gateway `/readiness`，而非单个 worker。
- 滚动升级时先起 decode 再起 prefill，或并行起避免 503 窗口过长。

---

## 3. IGW 与单 Router 模式区别

**Explain：** `enable_igw=false` 时 `RouterFactory::create_router` 只创建一个 router 实例。`enable_igw=true` 时 RouterManager 注册多套 router，适合同一 gateway 同时代理 HTTP Regular 与 gRPC PD 等 heterogeneous 拓扑。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/router_manager.rs L91-L92
 if config.router_config.enable_igw {
 info!("Initializing RouterManager in multi-router mode (IGW)");
```

**Comment：**

- IGW 自动创建 PD router（即使主 mode 为 Regular）。
- 内存与启动时间随 router 数量线性增加。

---

## 4. Worker 选不中时的错误路径

**Explain：** `select_worker_for_model` 返回 `None` 时，router 返回 503 `service_unavailable`，不会 fallback 到随机 worker。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L129-L130
 Err(e) => error::service_unavailable("no_workers", e),
 }
```

**Comment：**

- 常见原因：model_id 无注册 worker、全部熔断、PD 角色缺失。
- 检查 `GET /workers` 与 worker health endpoint。

---

## 5. 流式请求如何穿透 Gateway？

**Explain：** Gateway 不缓冲完整响应；用 `BreakerTrackedStream` 将 worker SSE 流式转发给客户端，熔断器监控 upstream 断开。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L41（imports）
 streaming_utils::BreakerTrackedStream,
```

**Comment：**

- 客户端 disconnect 时应 cancel upstream（Axum drop guard）。
- 重试对流式请求通常仅首次 connection 失败有效。

---

## 6. gRPC vs HTTP ConnectionMode

**Explain：** `ConnectionMode::Grpc` 使用 tonic 与 srt gRPC Engine 通信，延迟更低、支持 binary protobuf；HTTP 兼容性更好。OpenAI 模式仅 HTTP。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/factory.rs L40-L42
 RoutingMode::OpenAI { .. } => {
 Err("OpenAI mode requires HTTP connection_mode".to_string())
 }
```

**Comment：**

- gRPC router 使用 pipeline stages（preparation → request_building → response_processing）。
- 同一 gateway 进程 IGW 可同时持有 HTTP 与 gRPC router。

---

## 7. Consistent Hash 与 Session 粘性

**Explain：** Prefix-hash / cache-aware policy 利用 `HashRing` 使相同 prompt prefix 落到同一 worker，提高 RadixAttention cache 命中。纯 round-robin 无粘性。

**Code：**

```rust
// 来源：sgl-model-gateway/src/core/worker_registry.rs L33-L39
/// Consistent hash ring for O(log n) worker selection.
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index).
```

**Comment：**

- routing key 通常来自 request text hash 或 header `x-routing-key`。
- worker 扩缩容时仅 ~1/N 请求 remap。
