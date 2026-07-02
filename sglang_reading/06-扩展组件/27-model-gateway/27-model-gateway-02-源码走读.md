---
type: batch-doc
module: 27-model-gateway
batch: "27"
doc_type: walkthrough
title: "model-gateway · 源码走读"
tags:
 - sglang/batch/27
 - sglang/module/model-gateway
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# model-gateway · 源码走读

> 走读顺序：`server.rs` 路由表 → handler 委托 → `RouterFactory` → `http/router.rs` 选 worker → `worker_registry.rs` → `router_manager.rs` → PD readiness

---

## 1. Server 启动与路由表

### 1.1 `build_app` — Axum 路由注册

**Explain：** `build_app` 组装 protected/public 路由。推理 API（chat/completions/embeddings 等）走 protected 层并过 auth middleware；health/metrics 走 public。

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L544-L551
 let protected_routes = Router::new()
 .route("/generate", post(generate))
 .route("/v1/chat/completions", post(v1_chat_completions))
 .route("/v1/completions", post(v1_completions))
 .route("/v1/rerank", post(v1_rerank))
 .route("/v1/responses", post(v1_responses))
 .route("/v1/embeddings", post(v1_embeddings))
 .route("/v1/classify", post(v1_classify))
```

**Comment：**

- 路由路径与 OpenAI API 对齐，便于 SDK 零改动接入。
- `/generate` 保留 SGLang 原生 JSON 接口兼容。

### 1.2 `generate` handler — 委托 RouterTrait

**Explain：** 所有生成类 handler 模式一致：提取 model_id → 调用 `router.route_*`。

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L172-L182
async fn generate(
 State(state): State<Arc<AppState>>,
 headers: http::HeaderMap,
 Json(body): Json<GenerateRequest>,
) -> Response {
 let model_id = body.model.as_deref();
 state
 .router
 .route_generate(Some(&headers), &body, model_id)
 .await
}
```

**Comment：**

- headers 透传给 router 用于 routing key、auth、trace context 注入。
- `RouterTrait` 统一 HTTP/gRPC/PD/OpenAI 实现。

### 1.3 Worker 管理 REST API

**Explain：** Gateway 暴露 control plane API 动态增删 worker，无需重启 gateway。

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L413-L420
async fn create_worker(
 State(state): State<Arc<AppState>>,
 Json(config): Json<WorkerConfigRequest>,
) -> Response {
 match state.context.worker_service.create_worker(config).await {
 Ok(result) => result.into_response(),
 Err(err) => err.into_response(),
 }
}
```

**Comment：**

- `WorkerService` 封装注册、健康检查启动、metadata 拉取。
- 配合 service discovery（K8s DNS / 静态配置）自动注册 worker。

---

## 2. Router 工厂

### 2.1 `RouterFactory::create_router`

**Explain：** 根据 `connection_mode` × `routing_mode` 二维配置实例化唯一主 router（非 IGW）或供 RouterManager 批量创建（IGW）。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/factory.rs L23-L42
 pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
 match ctx.router_config.connection_mode {
 ConnectionMode::Grpc { .. } => match &ctx.router_config.mode {
 RoutingMode::Regular { .. } => Self::create_grpc_router(ctx).await,
 RoutingMode::PrefillDecode { .. } => {
 Self::create_grpc_pd_router(
 prefill_policy.as_ref(),
 decode_policy.as_ref(),
 &ctx.router_config.policy,
 ctx,
 )
 .await
 }
 RoutingMode::OpenAI { .. } => {
 Err("OpenAI mode requires HTTP connection_mode".to_string())
 }
 },
```

**Comment：**

- OpenAI 模式强制 HTTP（代理外部 HTTPS API）。
- gRPC PD router 使用 pipeline stage 架构（见 `routers/grpc/regular/stages/`）。

### 2.2 `Router::new` — HTTP Regular Router

**Explain：** Regular HTTP router 注入 `WorkerRegistry`、`PolicyRegistry`、共享 `reqwest::Client`。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L69-L79
 pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
 Ok(Router {
 worker_registry: ctx.worker_registry.clone(),
 policy_registry: ctx.policy_registry.clone(),
 client: ctx.client.clone(),
 dp_aware: ctx.router_config.dp_aware,
 enable_igw: ctx.router_config.enable_igw,
 retry_config: ctx.router_config.effective_retry_config(),
 })
 }
```

**Comment：**

- `dp_aware` 启用 data-parallel 路由感知（按 DP rank 选 worker）。
- `retry_config` 控制 upstream 5xx/429 重试与 backoff。

---

## 3. Worker 选择

### 3.1 `select_worker_for_model`

**Explain：** HTTP Regular 路由核心：按 model 过滤 worker → 过滤 available（健康+未熔断）→ policy 选 index → 返回 `Arc<dyn Worker>`。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L140-L191
 let workers = self.worker_registry.get_workers_filtered(
 effective_model_id,
 Some(WorkerType::Regular),
 Some(ConnectionMode::Http),
 None,
 false,
 );

 let available: Vec<Arc<dyn Worker>> = workers
 .iter()
 .filter(|w| w.is_available())
 .cloned()
 .collect();
 if available.is_empty() {
 return None;
 }

 let policy = match model_id {
 Some(model) => self.policy_registry.get_policy_or_default(model),
 None => self.policy_registry.get_default_policy(),
 };

 let hash_ring = self
 .worker_registry
 .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

 let idx = policy
 .select_worker(
 &available,
 &SelectWorkerInfo {
 request_text: text,
 tokens: None,
 headers,
 hash_ring,
 },
 )
 .await?;

 Some(available[idx].clone())
```

**Comment：**

- `request_text` 供 cache-aware / prefix-hash policy 使用。
- gRPC 路径可传 `tokens` 做更精确的 prefix routing。

### 3.2 Consistent Hash Ring

**Explain：** 每个 model 维护预计算 hash ring，150 virtual nodes/worker，O(log n) 查找，worker 增减时约 1/N key 迁移。

**Code：**

```rust
// 来源：sgl-model-gateway/src/core/worker_registry.rs L50-L70
impl HashRing {
 pub fn new(workers: &[Arc<dyn Worker>]) -> Self {
 let mut entries: Vec<(u64, Arc<str>)> =
 Vec::with_capacity(workers.len() * VIRTUAL_NODES_PER_WORKER);

 for worker in workers {
 let url: Arc<str> = Arc::from(worker.url());
 let url_bytes = url.as_bytes();

 for vnode in 0..VIRTUAL_NODES_PER_WORKER {
 let mut hasher = blake3::Hasher::new();
 hasher.update(url_bytes);
 hasher.update(b"#");
 hasher.update(&(vnode as u64).to_le_bytes());
 let hash = hasher.finalize();
 let pos = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
 entries.push((pos, Arc::clone(&url)));
 }
 }
```

**Comment：**

- blake3 保证跨 Rust 版本 hash 稳定。
- ring 仅在 worker 增删时 rebuild，请求路径无写锁。

---

## 4. 请求代理与重试

### 4.1 `route_typed_request` — 带重试的转发

**Explain：** 统一入口记录 metrics → `RetryExecutor` 包装单次转发 → 按 status 判断是否重试。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L194-L223
 pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
 &self,
 headers: Option<&HeaderMap>,
 typed_req: &T,
 route: &'static str,
 model_id: Option<&str>,
 ) -> Response {
 let start = Instant::now();
 let is_stream = typed_req.is_stream();
 let text = typed_req.extract_text_for_routing();
 let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
 let endpoint = route_to_endpoint(route);

 let response = RetryExecutor::execute_response_with_retry(
 &self.retry_config,
 |_: u32| async {
 let res = self
 .route_typed_request_once(headers, typed_req, route, model_id, is_stream, &text)
 .await;
 res
 },
 |res, _attempt| is_retryable_status(res.status()),
 |delay, attempt| { /* metrics */ },
 )
 .await;
```

**Comment：**

- 流式请求走 `BreakerTrackedStream`，熔断器跟踪 upstream 断开。
- 每次重试可能选不同 worker（若 policy 非 sticky）。

### 4.2 `proxy_get_request` — 简单 GET 代理

**Explain：** `/get_model_info` 等管理端点选第一个 healthy worker 转发。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L82-L97
 fn select_first_worker(&self) -> Result<String, String> {
 let workers = self.worker_registry.get_all();
 let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
 if healthy_workers.is_empty() {
 Err("No workers are available".to_string())
 } else {
 Ok(healthy_workers[0].url().to_string())
 }
 }

 async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
 match self.select_first_worker() {
 Ok(worker_url) => {
 let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
```

**Comment：**

- 管理 GET 不做 load balance，任意 healthy worker 即可（信息应一致）。
- header 过滤由 `header_utils::should_forward_request_header` 控制。

---

## 5. RouterManager（IGW）

### 5.1 结构与注册

**Explain：** IGW 模式下多个 `RouterTrait` 实例存入 `DashMap<RouterId, ...>`，snapshot 供无锁迭代。

**Code：**

```rust
// 来源：sgl-model-gateway/src/routers/router_manager.rs L62-L78
pub struct RouterManager {
 worker_registry: Arc<WorkerRegistry>,
 routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
 routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
 default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
 enable_igw: bool,
}

impl RouterManager {
 pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
 Self {
 worker_registry,
 routers: Arc::new(DashMap::new()),
 routers_snapshot: ArcSwap::from_pointee(Vec::new()),
 default_router: Arc::new(std::sync::RwLock::new(None)),
 enable_igw: false,
 }
 }
```

**Comment：**

- `router_ids` 模块定义静态常量：`HTTP_REGULAR`、`HTTP_PD`、`GRPC_REGULAR`、`GRPC_PD`。
- register 后 refresh snapshot，读路径零锁。

---

## 6. 健康与管理

### 6.1 `liveness` vs `readiness`

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L98-L100
async fn liveness() -> Response {
 (StatusCode::OK, "OK").into_response()
}
```

**Comment：** liveness 不查 worker；readiness 见 01-核心概念 §5。

### 6.2 `flush_cache` — 广播所有 worker

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L401-L405
async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
 WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
 .await
 .into_response()
}
```

**Comment：**

- 并行 POST 各 worker `/flush_cache`。
- 与 srt 单实例 flush 语义一致，gateway 做 fan-out。

---

## 7. 解析辅助 API

### 7.1 Function call / reasoning 解析

**Explain：** Gateway 提供独立于推理的解析端点，复用 worker 或本地 parser 配置。

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L80-L92
async fn parse_function_call(
 State(state): State<Arc<AppState>>,
 Json(req): Json<ParseFunctionCallRequest>,
) -> Response {
 parse::parse_function_call(&state.context, &req).await
}

async fn parse_reasoning(
 State(state): State<Arc<AppState>>,
 Json(req): Json<SeparateReasoningRequest>,
) -> Response {
 parse::parse_reasoning(&state.context, &req).await
}
```

**Comment：**

- 用于 OpenAI Responses API 的 tool call 后处理。
- 不经过 worker 选路，直接在 gateway 进程内解析 JSON。

---

## 8. Service Discovery

### 8.1 启动时注册 worker

**Explain：** `startup` 中可选启动 K8s/DNS service discovery，周期性 poll 并更新 registry。

**Code：**

```rust
// 来源：sgl-model-gateway/src/server.rs L66（imports）
 service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
```

**Comment：**

- 静态配置 worker URL 与 discovery 可并存。
- 新 worker 加入触发 hash ring rebuild 与 health check 协程。

---

## 9. Round Robin Policy 示例

**Explain：** 最简单 policy：`select_worker` 轮询 index，无状态。

**Code：**

```rust
// 来源：sgl-model-gateway/src/policies/round_robin.rs L31-L46
 async fn select_worker(
 &self,
 workers: &[Arc<dyn Worker>],
 _info: &SelectWorkerInfo<'_>,
 ) -> Option<usize> {
 let healthy_indices = get_healthy_worker_indices(workers);

 if healthy_indices.is_empty() {
 return None;
 }

 // Get and increment counter atomically
 let count = self.counter.fetch_add(1, Ordering::Relaxed);
 let selected_idx = count % healthy_indices.len();

 Some(healthy_indices[selected_idx])
 }
```

**Comment：**

- 只对 **healthy** worker 轮询，避免把流量打到未就绪实例。
- 生产常用 cache-aware 或 prefix-hash 提升 KV cache 命中。
- policy 按 model_id 注册，不同模型可用不同策略。

---

## 10. Worker HTTP Client

**Explain：** 全局共享 `reqwest::Client`，30s 默认超时，避免每请求建连。

**Code：**

```rust
// 来源：sgl-model-gateway/src/core/worker.rs L35-L40
static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
 reqwest::Client::builder()
 .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
 .build()
 .expect("Failed to create worker HTTP client")
});
```

**Comment：**

- 流式响应使用同一 client 的 `bytes_stream()`。
- gRPC worker 使用独立 `GrpcClient`（见 `routers/grpc/client/`）。
