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
updated: 2026-07-12
---
# model-gateway · 排障指南

---

## 你为什么要读

Gateway 返回的 5xx 可能来自自身路由，也可能只是转发了 worker 失败；流式请求还可能在 HTTP 200 之后才出错。本文按 readiness、worker registry、policy、retry、circuit breaker 和下游响应逐层取证，先确定错误的真正所有者。

## 1. Gateway 与 srt 内置 HTTP server 如何选型？

**读法：** 单模型、单副本且不需要统一鉴权、协议适配、独立重试/熔断或服务发现时，可以直接使用 `python -m sglang.launch_server`。多副本、PD 分离、多模型统一入口、独立熔断重试或 K8s service discovery 是前置 `smg` 的典型理由，但最终仍要用当前部署复杂度与代理开销权衡。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/factory.rs L44-L45
 ConnectionMode::Http => match &ctx.router_config.mode {
 RoutingMode::Regular { .. } => Self::create_regular_router(ctx).await,
```

**要点：**

- Gateway 增加一次代理和选路开销；实际延迟必须在当前网络、协议和连接复用条件下测量，再与运维收益权衡。
- srt 与 Gateway 可以同 pod 部署，但监听地址必须以各自实际配置为准，不能把示例端口当成协议固定值。

---

## 2. 为什么 PD readiness 要求两种 worker？

**读法：** PD 路由假设 prefill 与 decode 是不同进程/池。若只有 prefill healthy，请求会卡在无法 decode；只有 decode 则无法 bootstrap KV。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/server.rs L111-L117
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
- 启动先后顺序本身不构成正确性保证；只有两类 healthy worker 同时存在，PD Gateway 才 ready。滚动升级应以两池最小可用容量和 readiness 结果为准。

---

## 3. IGW 与单 Router 模式区别

**读法：** `enable_igw=false` 时 `RouterFactory::create_router` 只创建一个 router 实例。`enable_igw=true` 时 RouterManager 注册多套 router，适合同一 gateway 同时代理 HTTP Regular 与 gRPC PD 等 heterogeneous 拓扑。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/router_manager.rs L91-L92
 if config.router_config.enable_igw {
 info!("Initializing RouterManager in multi-router mode (IGW)");
```

**要点：**

- IGW 会尝试创建 HTTP/gRPC 的 Regular、PD router，以及 OpenAI router；某个 router 创建失败只记录 warning，但若最终一个 router 都没有则启动失败。
- router 数量增加会增加对象与连接状态，但不能在没有测量的情况下断言内存或启动时间严格线性增长。

---

## 4. Worker 选不中时的错误路径

**读法：** `select_worker_for_model` 返回 `None` 时，router 返回 503 `service_unavailable`，不会 fallback 到随机 worker。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/router.rs L129-L130
 Err(e) => error::service_unavailable("no_workers", e),
 }
```

**要点：**

- 常见原因：model_id 无注册 worker、全部熔断、PD 角色缺失。
- 检查 `GET /workers` 与 worker health endpoint。

---

## 5. 为什么 HTTP 200 之后流断不能自动重试？

**症状：** 客户端已经收到若干 SSE chunk，随后报 EOF、连接重置或 stream error；Gateway 日志可能已经记录本次 router request 为初始成功。

**可能原因：** Decode worker 在 2xx response 建立后发生 transport error；或者客户端主动取消，导致 Gateway drop upstream body。两者对 breaker 的归因不同。

**源码入口：**

```rust
// 来源：sgl-model-gateway/src/routers/streaming_utils.rs L108-L128
impl<E: Display> Stream for BreakerTrackedStream<E> {
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(b))) => Poll::Ready(Some(Ok(b))),
            Poll::Ready(Some(Err(e))) => {
                error!("Upstream stream error from worker {}: {}", self.log_url, e);
                self.terminal = Terminal::Errored;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                if self.terminal == Terminal::Active {
                    self.terminal = Terminal::Completed;
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
```

**操作：**

1. 先确认客户端是否已经收到任意 token；收到后就不要期待 Gateway 透明重试。
2. 查 `Upstream stream error from worker` 日志及对应 decode URL。
3. 将客户端取消与 upstream error 分开统计：前者只表示调用方不再消费，不能证明 worker 坏了。
4. 如果业务要求恢复，客户端必须按应用语义发起新请求，并自行决定是否允许重复 token、重新采样或幂等重放。

**预期：**

- 通用 HTTP 路径的底层 stream clean end：对应 worker breaker 记一次 success；它不解析 `[DONE]`。
- HTTP PD 路径检测到包含 `data: [DONE]` 的 chunk：先标记 success，再发送 chunk 并停止继续 poll；因此 `[DONE]` 后的 trailing error 不会被观察，客户端在该 chunk 发送时断开也仍可能记 success。
- upstream stream error：decode breaker 记一次 failure。
- 客户端在结果未知时断开：breaker 不记 success，也不记 failure。

**边界：** RetryExecutor 只看到 upstream 的初始 HTTP response。2xx response 一旦交给客户端，后续 body error 不会重新进入 worker 选择流程；否则 Gateway 无法证明新 worker 的 token 与已发送前缀一致。

---

## 6. gRPC vs HTTP ConnectionMode

**读法：** `ConnectionMode::Grpc` 使用 tonic 与 srt gRPC Engine 通信，HTTP 路径使用 reqwest/Axum 代理。二者的序列化、连接和 streaming 实现不同；哪一种延迟更低必须在当前 payload、连接复用、TLS、并发和 worker 部署条件下测量。OpenAI 模式仅 HTTP。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/factory.rs L40-L42
 RoutingMode::OpenAI { .. } => {
 Err("OpenAI mode requires HTTP connection_mode".to_string())
 }
```

**要点：**

- gRPC router 使用 pipeline stages（preparation → request_building → response_processing）。
- 同一 gateway 进程 IGW 可同时持有 HTTP 与 gRPC router。

---

## 7. PD 请求为什么会“双边都收到，却只有一边报错”？

**症状：** Prefill 日志显示请求成功，客户端却收到 decode 502；或者 decode 已建立连接，但最终报缺 KV/等待超时。

**可能原因：** HTTP PD 使用 `tokio::join!` 并行双发。Gateway 不会等 Prefill HTTP response 再发送 Decode 请求；Decode 先收到请求是正常的，它随后依赖相同 `bootstrap_room` 对应的 KV 到达。

**源码入口：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/pd_router.rs L681-L706
let prefill_request = self.build_post_with_headers(
    &self.client,
    &prepared_prefill.endpoint_url,
    &prepared_prefill.body,
    headers,
    false,
);
let decode_request = self.build_post_with_headers(
    &self.client,
    &prepared_decode.endpoint_url,
    &prepared_decode.body,
    headers,
    false,
);

let (prefill_result, decode_result) =
    tokio::join!(prefill_request.send(), decode_request.send());
```

**操作：**

1. 用同一个请求关联 `prefill_url`、`decode_url` 和 `bootstrap_room`，不要按日志先后猜调用顺序。
2. 分别核对两侧 HTTP status/transport result；最终客户端 5xx 不能直接代表两台 worker 都失败。
3. 若 decode 等不到 KV，继续进入 [[SGLang-PD分离-排障指南]]，检查 room、bootstrap host/port、transfer backend 和 metadata gate。
4. 若启用了 DP-aware worker，确认 decode body 同时包含自身 `data_parallel_rank` 与 prefill 的 `disagg_prefill_dp_rank`。

**预期：** 两侧 `send()` 并行推进；Prefill 成功时其 breaker 不应因为 Decode transport failure 被错误惩罚；Decode 只有在正确 room 的 KV 可见后才能继续生成。还要注意客户端未必立刻看到已建立的 Decode SSE：Gateway 会先消费完整 Prefill response body，Prefill body 变慢同样可能抬高 TTFT。

**归因边界：** 上述精确分侧主要成立于 Decode transport/non-2xx 等 decode 驱动错误。若 Decode 已成功取得 response、随后 Prefill body 读取或解析失败，当前实现可能按最终 response 粗粒度归因：非流式可能连带惩罚 Decode，流式可能没有给 Decode 记录 terminal。不要用“最终 5xx”或“文档说分侧”替代逐分支检查。

---

## 8. PD 重试为什么可能换掉整对 worker？

**症状：** 同一个外部请求的日志中出现多个 prefill/decode URL 或多个 `bootstrap_room`；第一次只有 Decode 失败，第二次 Prefill 也被重新执行。

**可能原因：** 每个 retry attempt 都从 `select_pd_pair` 开始，并重新向原始请求注入 bootstrap metadata。PD retry 不是“只补发失败的一侧”，而是整对重放。

**操作：**

1. 检查最终 status 是否属于 `408/429/500/502/503/504`。
2. 按 attempt 观察选中的 PF/DC URL；round-robin 可能换 worker，一致性策略也可能因 routing key 稳定而选回原 worker。
3. 对有外部副作用或非确定性采样的请求，不要假设重放与第一次尝试完全等价。
4. 先确认配置语义：当前 `max_retries` 实际是总 attempt 数，`3` 表示首次加两次重试；再评估重复 prefill 成本、额外 KV 传输和尾延迟，而不是只追求更高成功率。

**预期：** 可重试状态进入退避并执行新 attempt；4xx（除 408/429）直接返回；流式 2xx 后的 body error 不会触发新的 attempt。

---

## 9. Consistent Hash 与 Session 粘性

**读法：** Prefix-hash / cache-aware policy 利用 `HashRing` 使相同 prompt prefix 落到同一 worker，提高 RadixAttention cache 命中。纯 round-robin 无粘性。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/core/worker_registry.rs L33-L39
/// Consistent hash ring for O(log n) worker selection.
/// Each worker is placed at multiple positions (virtual nodes) on the ring
/// based on hash(worker_url + vnode_index).
```

**要点：**

- routing key 可来自 request text 派生值或 header `x-smg-routing-key`；手工目标 worker 使用另一条 `x-smg-target-worker` 控制头。
- 一致性哈希的理论目标是扩缩容时只重映射约 `1/N` 的 key；150 个 virtual nodes、实际拓扑、ring 分布和健康跳过都会影响观测值，不能把 `~1/N` 当作无条件 SLA。

---

## 10. 为什么重复注册后出现重复 worker 或旧模型残影？

**症状：** 同一 URL 更新注册信息后，某模型候选数异常增大、hash ring 出现重复权重，或 worker 已改 model/type/connection 但旧索引仍能查到。

**可能原因：** `WorkerRegistry::register` 对同 URL 复用 `WorkerId` 并覆盖主表，却仍向 model/type/connection 索引追加，且不先清理旧索引；这不是事务式 upsert。删除模型最后一个 worker 后，内部还可能保留 empty snapshot/ring key，虽然 `get_models()` 会过滤空模型。

**操作：**

1. 同时比对 URL 主表、model snapshot、type/connection index 与 hash ring，不要只看 `GET /workers` 的一层视图。
2. 控制面更新 worker 属性时优先显式 remove 后 register，并观察候选数是否回到预期。
3. 若无法改变控制面，避免对同 URL 重复推送注册事件，并为 registry upsert 增加去重/迁移测试。

**预期：** 干净注册时每个逻辑 worker 在对应模型 snapshot 中只出现一次；remove 最后一台 worker 后，外部模型列表不再展示该模型。内部空 key 是否清理，要以修复后的实现或定向测试为准。

---

## 11. 为什么 PD `/server_info` 看起来来自错误的池？

当前基线的注释写“first decode server”，实际代码却调用 `proxy_to_first_prefill_worker("server_info", None)`。排障时以调用链为准：现状代理到 Prefill helper。若需要 Decode 池信息，应直接查询 Decode worker 或先修正/澄清上游实现，不能依据注释推断。

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
