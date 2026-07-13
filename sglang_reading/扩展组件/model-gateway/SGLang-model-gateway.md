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
updated: 2026-07-12
---
# model-gateway

> **源码范围：** `sgl-model-gateway/src/` — `server.rs`、`routers/`、`core/worker*.rs`、`policies/` 
> **Git 基线：** `70df09b` 
> **前置专题：** [[SGLang-sgl-kernel]] · **下一专题：** [[SGLang-前端语言]]

---

## 1. 本模块目标

专题读法：`sgl-model-gateway`（Rust 二进制 `smg`）是 SGLang 的独立 API 网关层，位于客户端与推理 worker 之间。它用 Axum 暴露 HTTP API，可通过 HTTP 或 gRPC 连接 worker；`RouterManager` 先决定使用 Regular、PD、OpenAI 等哪类 router，具体 router 再从 `WorkerRegistry` 的可用候选中按 policy 选 worker。

读本专题要守住三个边界：

1. Gateway 代理请求和响应，但不执行模型 forward，也不搬运 PD 的 KV tensor。
2. `healthy`、`available`、`ready` 是三层状态：worker 健康、健康且 breaker 可执行、Gateway 达到当前模式最低服务条件。
3. retry 只能发生在 response 交给客户端之前；流式 2xx 之后的 body error 只能归因和中断，不能透明换 worker 续传。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/server.rs L70-L78
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
- IGW 模式下 `router_manager` 最多协调 HTTP Regular、HTTP PD、HTTP OpenAI、gRPC Regular、gRPC PD 五类 router。

---

## 2. 在全局架构中的位置

```
Client (OpenAI SDK / curl)
 │ HTTP
 ▼
sgl-model-gateway (smg) ← 本模块
 │ 选 router + 选 worker + 反向代理
 ▼
SGLang / vLLM / external worker(s)
```

| 组件 | 职责 |
|------|------|
| `server.rs` | Axum 路由表、health/readiness、handler 薄层 |
| `routers/http/router.rs` | Regular 模式 HTTP 反向代理 |
| `routers/http/pd_router.rs` | Prefill+Decode 分离路由 |
| `core/worker_registry.rs` | worker 注册、consistent hash ring |
| `routers/router_manager.rs` | IGW 多 router 编排 |
| `core/retry.rs` | 可重试状态、退避与 attempt 生命周期 |
| `routers/streaming_utils.rs` | 流结束、流错误、客户端取消的 breaker 归因 |
| `policies/` | worker 选择策略 |

---

## 3. 自测与验收标准

- [ ] 能说明 gateway 与 srt 的职责边界（gateway 不做 forward，只做路由/代理）
- [ ] 能追踪 `/v1/chat/completions` 从 Axum handler 到 worker HTTP 的路径
- [ ] 能解释单一 PD 模式下 readiness 为何要求 prefill+decode 各至少一个 healthy worker
- [ ] 能说明 IGW readiness 200 为什么不保证某个 PD 模型已经凑齐一对 worker
- [ ] 能沿一次 HTTP PD attempt 解释 pair、bootstrap room、并行双发、Prefill body 对响应交付的门控、KV 直传，以及 breaker 分侧归因已经覆盖和仍未覆盖的路径
- [ ] 能用一条真实或静态请求轨迹证明 handler、router、policy、worker client 的交接，并指出流式 2xx 前后不同的重试边界

→ [[SGLang-model-gateway-核心概念]] · [[SGLang-model-gateway-源码走读]] · [[SGLang-model-gateway-数据流]]
