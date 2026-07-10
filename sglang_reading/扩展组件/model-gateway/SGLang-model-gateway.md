---
title: "model-gateway"
type: map
framework: sglang
topic: "model-gateway"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# model-gateway

> **源码范围：** `sgl-model-gateway/src/` — `server.rs`、`routers/`、`core/worker*.rs`、`policies/` 
> **Git 基线：** `70df09b` 
> **前置专题：** [[SGLang-sgl-kernel]] · **下一专题：** [[SGLang-前端语言]]

---

## 1. 本模块目标

专题读法：`sgl-model-gateway`（Rust 二进制 `smg`）是 SGLang 的**独立 API 网关层**，位于客户端与 srt worker 之间。它用 Axum 暴露 OpenAI 兼容 HTTP/gRPC 端点，通过 `WorkerRegistry` 管理后端 worker 池，按负载均衡策略（round-robin、consistent hash、cache-aware 等）路由请求，并支持 PD disaggregation（prefill/decode 分离）、IGW 多 router 模式、熔断与重试。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/server.rs L70-L78
#[derive(Clone)]
pub struct AppState {
 pub router: Arc<dyn RouterTrait>,
 pub context: Arc<AppContext>,
 pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
 pub router_manager: Option<Arc<RouterManager>>,
 pub mesh_handler: Option<Arc<MeshServerHandler>>,
 pub mesh_sync_manager: Option<Arc<MeshSyncManager>>,
}
```

读法：

- `AppState` 注入所有 Axum handler：路由决策走 `router`，worker 管理走 `context.worker_registry`。
- IGW 模式下 `router_manager` 协调 HTTP/gRPC × Regular/PD 四套 router。

---

## 2. 在全局架构中的位置

```
Client (OpenAI SDK / curl)
 │ HTTP/gRPC
 ▼
sgl-model-gateway (smg) ← 本模块
 │ 反向代理 + 选 worker
 ▼
srt worker(s) — /v1/chat/completions, /generate, gRPC Engine
```

| 组件 | 职责 |
|------|------|
| `server.rs` | Axum 路由表、health/readiness、handler 薄层 |
| `routers/http/router.rs` | Regular 模式 HTTP 反向代理 |
| `routers/http/pd_router.rs` | Prefill+Decode 分离路由 |
| `core/worker_registry.rs` | worker 注册、consistent hash ring |
| `routers/router_manager.rs` | IGW 多 router 编排 |
| `policies/` | worker 选择策略 |

---

## 3. 自测与验收标准

- [ ] 能说明 gateway 与 srt 的职责边界（gateway 不做 forward，只做路由/代理）
- [ ] 能追踪 `/v1/chat/completions` 从 Axum handler 到 worker HTTP 的路径
- [ ] 能解释 PD 模式下 readiness 为何要求 prefill+decode 各至少一个 healthy worker
- [ ] 能用一条真实或静态请求轨迹证明 handler、router、policy、worker client 的交接，并记录失败时由哪一层重试或熔断

→ [[SGLang-model-gateway-核心概念]] · [[SGLang-model-gateway-源码走读]] · [[SGLang-model-gateway-数据流]]
