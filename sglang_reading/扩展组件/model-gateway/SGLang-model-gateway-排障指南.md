---
title: "model-gateway · 排障指南"
type: troubleshooting
framework: sglang
topic: "model-gateway"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# model-gateway · 排障指南

---

## 你为什么要读

Gateway 返回的 5xx 可能来自自身路由，也可能只是转发了 worker 失败；流式请求还可能在 HTTP 200 之后才出错。本文按 readiness、worker registry、policy、retry、circuit breaker 和下游响应逐层取证，先确定错误的真正所有者。

## 1. Gateway 与 srt 内置 HTTP server 如何选型？

**读法：** 单模型、单副本部署可直接 `python -m sglang.launch_server`，无需 gateway。多副本、PD 分离、多模型统一入口、熔断重试、K8s service discovery 场景应前置 `smg`。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/factory.rs L44-L45
 ConnectionMode::Http => match &ctx.router_config.mode {
 RoutingMode::Regular { .. } => Self::create_regular_router(ctx).await,
```

**要点：**

- Gateway 增加一次代理和选路开销；实际延迟必须在当前网络、协议和连接复用条件下测量，再与运维收益权衡。
- 也可 srt 与 gateway 同 pod：gateway 监听 30000，srt 监听 8000。

---

## 2. 为什么 PD readiness 要求两种 worker？

**读法：** PD 路由假设 prefill 与 decode 是不同进程/池。若只有 prefill healthy，请求会卡在无法 decode；只有 decode 则无法 bootstrap KV。

**源码锚点：**

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

**要点：**

- K8s readiness probe 应指向 gateway `/readiness`，而非单个 worker。
- 滚动升级时先起 decode 再起 prefill，或并行起避免 503 窗口过长。

---

## 3. IGW 与单 Router 模式区别

**读法：** `enable_igw=false` 时 `RouterFactory::create_router` 只创建一个 router 实例。`enable_igw=true` 时 RouterManager 注册多套 router，适合同一 gateway 同时代理 HTTP Regular 与 gRPC PD 等 heterogeneous 拓扑。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/router_manager.rs L91-L92
 if config.router_config.enable_igw {
 info!("Initializing RouterManager in multi-router mode (IGW)");
```

**要点：**

- IGW 自动创建 PD router（即使主 mode 为 Regular）。
- 内存与启动时间随 router 数量线性增加。

---

## 4. Worker 选不中时的错误路径

**读法：** `select_worker_for_model` 返回 `None` 时，router 返回 503 `service_unavailable`，不会 fallback 到随机 worker。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L129-L130
 Err(e) => error::service_unavailable("no_workers", e),
 }
```

**要点：**

- 常见原因：model_id 无注册 worker、全部熔断、PD 角色缺失。
- 检查 `GET /workers` 与 worker health endpoint。

---

## 5. 流式请求如何穿透 Gateway？

**读法：** Gateway 不缓冲完整响应；用 `BreakerTrackedStream` 将 worker SSE 流式转发给客户端，熔断器监控 upstream 断开。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L41（imports）
 streaming_utils::BreakerTrackedStream,
```

**要点：**

- 客户端 disconnect 时应 cancel upstream（Axum drop guard）。
- 重试对流式请求通常仅首次 connection 失败有效。

---

## 6. gRPC vs HTTP ConnectionMode

**读法：** `ConnectionMode::Grpc` 使用 tonic 与 srt gRPC Engine 通信，延迟更低、支持 binary protobuf；HTTP 兼容性更好。OpenAI 模式仅 HTTP。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/factory.rs L40-L42
 RoutingMode::OpenAI { .. } => {
 Err("OpenAI mode requires HTTP connection_mode".to_string())
 }
```

**要点：**

- gRPC router 使用 pipeline stages（preparation → request_building → response_processing）。
- 同一 gateway 进程 IGW 可同时持有 HTTP 与 gRPC router。

---

## 7. Consistent Hash 与 Session 粘性

**读法：** Prefix-hash / cache-aware policy 利用 `HashRing` 使相同 prompt prefix 落到同一 worker，提高 RadixAttention cache 命中。纯 round-robin 无粘性。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/core/worker_registry.rs L33-L39
/// Consistent hash ring for O(log n) worker selection.
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index).
```

**要点：**

- routing key 通常来自 request text hash 或 header `x-routing-key`。
- worker 扩缩容时仅 ~1/N 请求 remap。

---

## 验证抓手

Gateway 的问题最容易混在一起：readiness、worker selection、router mode、streaming 和 hash policy 不是同一层。先用静态检索确认每个问题的源码入口。

```powershell
rg -n "readiness|has_prefill|has_decode|enable_igw|OpenAI mode requires HTTP|BreakerTrackedStream|HashRing|service_unavailable|select_worker_for_model" sglang/sgl-model-gateway/src
```

预期现象：

- `server.rs` 命中 `/readiness` 以及 `has_prefill`、`has_decode`，证明 PD readiness 不是单 worker 健康检查。
- `router_manager.rs` 和配置文件命中 `enable_igw`，证明 IGW 是 router 管理模式，不是单个 worker 开关。
- `factory.rs` 命中 `OpenAI mode requires HTTP connection_mode`，证明 OpenAI 路径不能切到 gRPC。
- `http/router.rs` 命中 `select_worker_for_model` 和 `service_unavailable`，证明选不中 worker 会走 503，而不是随机 fallback。
- `streaming_utils.rs` 或各 router 命中 `BreakerTrackedStream`，证明 streaming 转发有熔断器跟踪。
- `worker_registry.rs`、policy 文件命中 `HashRing`，证明 consistent hash 是 policy/registry 层能力。

如果线上只有 503，不要只看 HTTP handler。先把错误码与这些入口对齐：`no_workers` 查 worker registry 和 health；PD readiness 查 prefill/decode 两类 worker；stream 中断查 `BreakerTrackedStream` 的完成或错误路径。
