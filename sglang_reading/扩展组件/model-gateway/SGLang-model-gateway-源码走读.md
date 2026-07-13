---
title: "model-gateway · 源码走读"
type: walkthrough
framework: sglang
topic: "model-gateway"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# model-gateway · 源码走读

> 走读顺序：`server.rs` 路由表 → handler 委托 → `RouterManager` / `RouterFactory` → Regular HTTP 单 worker 路由 → HTTP PD 双 worker 路由 → `WorkerRegistry` / `Worker` 的健康、熔断与负载生命周期。

---

## 长文读法

这篇按 gateway 的数据面和控制面边界读：`server.rs` 固定 Axum 路由和 handler 委托，`RouterFactory` 把 connection mode 与 routing mode 编译成具体 router，HTTP router 在热路径里完成 worker 过滤、policy 选择和请求代理，`WorkerRegistry` / `RouterManager` 维护运行时拓扑、IGW 快照、健康和服务发现。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 首次建立 gateway 主线 | 1 到 2 | endpoint 是协议门面，真正选 worker 的逻辑从 `RouterTrait` 后面开始 |
| 排查请求为什么没到目标 worker | 3 到 4 | 先看 model id、healthy 候选、policy，再看代理重试、headers 和 streaming 生命周期 |
| 理解 HTTP PD 为什么不是串行两跳 | 5 | 每次 attempt 重新选一对 worker、生成 room、并行双发；KV 不经过 Gateway |
| 判断当前 router 形态 | 2、5 | `connection_mode × routing_mode` 决定 HTTP/gRPC、Regular/PD/OpenAI/IGW 的实现组合 |
| 排查动态 worker 注册或摘除 | 1.3、3.2、6 | 控制面要同时更新 registry、hash ring 和 RouterManager 可见拓扑 |
| 排查健康、flush、管理端点 | 7 | liveness 是进程级，worker health/load/cache 操作属于集群或 worker 级 |

读的时候保持两条线分开：数据面是“请求选 worker 并代理”，控制面是“worker 拓扑和健康状态如何更新”。两条线共享 `AppContext`，但职责不要混读。

## 1. Server 入口只做协议边界

### 1.1 `build_app`：把外部 API 固定成 Axum 路由

**问题与约束：** Gateway 要同时暴露 SGLang 原生 `/generate`、OpenAI 风格 `/v1/*`、Responses/Conversation 等多组 API；这些入口需要统一鉴权、payload 限制、request id 注入，但不能把 HTTP 细节扩散到每种 router 实现里。

**设计选择：** `build_app` 先把推理类路由集中放进 `protected_routes`，再由 middleware 统一包裹；具体怎么选 worker、怎么转发、是否走 PD 或 gRPC，都留给 `RouterTrait` 后面的实现。

**读法：** 这段代码是 gateway 的最外层协议表。它定义了哪些路径属于“受保护推理 API”，也解释了为什么后续 handler 看起来很薄：server 层只负责从 HTTP 世界进入内部 trait 世界。

**源码锚点：**

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

**代码逻辑：** `Router::new()` 创建 Axum 子路由表，连续 `.route()` 把不同 endpoint 绑定到对应 async handler。这里没有创建 worker，也没有读取 policy。

**为什么这样写：** 路由表稳定，转发策略可变。把 API 入口和后端路由分开后，新增 policy、IGW、多 worker、OpenAI 外部 provider 时不需要改 HTTP endpoint 的公共契约。

**不变量与失败模式：** 入口路径必须与 handler 请求类型匹配；如果这里漏挂 endpoint，后面的 router 再完整也无法被访问。反过来，如果把 worker 选择塞进 server 层，会让每个 API handler 都重复 policy / retry / metrics 逻辑。

**要点：** 读这段时要抓住边界：`server.rs` 是协议门面，不是负载均衡器。真正的 gateway 设计从 handler 委托开始。

### 1.2 `generate` handler：把请求委托给 `RouterTrait`

**问题与约束：** 每个 API handler 都要抽取 headers、body、model id，再调用对应路由能力；如果每个 handler 自己做 worker 选择，OpenAI / gRPC / PD / IGW 会变成多套分叉。

**设计选择：** handler 只把 `HeaderMap`、typed request 和 `model_id` 交给 `state.router`。`state.router` 的静态类型是 `Arc<dyn RouterTrait>`，运行时可以是 `RouterManager`、HTTP router、gRPC router 或 PD router。

**读法：** `/generate` 是最简单的委托样例。它不判断路由模式，只抽取 `body.model.as_deref()`，然后调用统一 trait 方法。

**源码锚点：**

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

**代码逻辑：** Axum extractor 拿到共享 `AppState`、请求头和 JSON body；handler 从 body 中取可选模型名，再把所有上下文传给 `route_generate`。

**为什么这样写：** model-gateway 的中心抽象不是 HTTP handler，而是“同一个请求可以被不同 router 后端处理”。trait 委托让 server 层不必知道当前是单 router、IGW 多 router、HTTP、gRPC 还是 PD。

**不变量与失败模式：** handler 必须保留 headers，因为路由键、trace、鉴权透传都可能依赖 headers；如果丢 headers，cache-aware/manual policy 与 upstream trace 都会失效。

**要点：** 读者可以把 server handler 看成 adapter：把 Axum 请求变成 `RouterTrait` 调用。

### 1.3 Worker 管理 REST API：控制面不重启 gateway

**问题与约束：** serving 集群的 worker 会动态上下线；如果 gateway 只能启动时读取静态配置，扩缩容、外部 provider 注册和故障摘除都要重启入口服务。

**设计选择：** `server.rs` 暴露 worker control plane API，但具体注册、metadata 拉取、健康检查协调交给 `worker_service`。server 不直接操作 registry 内部索引。

**读法：** `create_worker` 是动态注册入口。它把 JSON 配置交给 `worker_service.create_worker`，成功或失败都转换成 HTTP response。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/server.rs L413-L421
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

**代码逻辑：** handler 从 `AppContext` 拿 `worker_service`，调用异步创建流程；返回值统一走 `IntoResponse`。

**为什么这样写：** 控制面和数据面共享 `AppContext`，但控制面通过 service 层封装副作用，避免 REST handler 直接更新 `WorkerRegistry`、hash ring、health checker 和 mesh sync。

**不变量与失败模式：** worker 注册必须同时更新 URL 映射、模型索引、连接类型索引和 hash ring；只改其中一处会导致“列表能看到但请求选不到”或“一致性哈希仍指向旧 worker”。

**要点：** 这不是简单 CRUD。创建 worker 是把运行时 topology 写进 gateway 的路由数据结构。

---

## 2. Router 构造把模式组合集中起来

### 2.1 `RouterFactory::create_router`：二维配置决定 router 类型

**问题与约束：** Gateway 同时支持 HTTP/gRPC 连接方式、Regular/PD/OpenAI 路由模式。连接方式和路由模式不是任意组合，例如 OpenAI 外部 provider 必须走 HTTP。

**设计选择：** `RouterFactory` 以 `connection_mode × routing_mode` 做集中分派；不合法组合在工厂层报错，合法组合构造对应 router 并注入共享 `AppContext`。

**读法：** 这段 match 是 gateway 路由形态的权威入口。它把配置空间收敛成少数实现类：HTTP Regular、HTTP PD、gRPC Regular、gRPC PD、OpenAI。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/factory.rs L23-L37
pub async fn create_router(ctx: &Arc<AppContext>) -> Result<Box<dyn RouterTrait>, String> {
    match ctx.router_config.connection_mode {
        ConnectionMode::Grpc { .. } => match &ctx.router_config.mode {
            RoutingMode::Regular { .. } => Self::create_grpc_router(ctx).await,
            RoutingMode::PrefillDecode {
                prefill_policy,
                decode_policy,
                ..
            } => {
                Self::create_grpc_pd_router(
                    prefill_policy.as_ref(),
                    decode_policy.as_ref(),
                    &ctx.router_config.policy,
                    ctx,
                )
```

**代码逻辑：** 外层 match 看连接模式，内层 match 看路由模式；PD 分支把 prefill/decode policy 配置传入 PD router 构造路径。

**为什么这样写：** 模式组合如果散落在各 handler，会产生大量“某 endpoint 支持某模式吗”的局部判断。工厂集中分派后，运行期只面对 `RouterTrait`，错误组合也能在启动或配置加载阶段暴露。

**不变量与失败模式：** OpenAI mode 不能走 gRPC；PD router 需要 prefill/decode 两类 policy。新增路由模式时必须同时扩展工厂和 `RouterManager` 的 router id 选择，否则 IGW 可能无法选中新模式。

**要点：** 工厂层回答的是“创建哪种路由器”，不是“请求去哪个 worker”。后者在 router 的热路径里发生。

### 2.2 `Router::new`：HTTP Regular Router 注入共享依赖

**问题与约束：** HTTP Regular router 的热路径需要 registry、policy、HTTP client、DP-aware 标志和 retry 配置；这些依赖如果每次请求临时创建，会把连接池、策略状态和指标上下文打散。

**设计选择：** `Router::new` 从 `AppContext` 克隆共享 `Arc` 与 client，把 router 做成轻量 facade；真正可变状态放在 registry/policy/client 内部。

**读法：** 这段构造函数说明 HTTP Regular router 是“依赖注入对象”，不是资源拥有者。它拿到的是共享引用。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/routers/http/router.rs L71-L80
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

**代码逻辑：** 构造函数拷贝配置位并克隆共享组件，返回一个 `Router` 值。

**为什么这样写：** router 是高并发请求共享对象。把 registry/policy/client 都变成 `Arc` 共享，可以让 `RouterManager` 同时持有多个 router，而不复制 worker 状态或 HTTP 连接池。

**不变量与失败模式：** `retry_config` 是构造时生效的配置快照；如果需要动态更新，必须通过上层配置刷新机制，而不是假设 router 会自动重读 config。

**要点：** 这里的设计哲学是“路由器持有能力引用，不持有拓扑真相”。拓扑真相在 `WorkerRegistry`。

---

## 3. Worker 选择是热路径的核心

### 3.1 `select_worker_for_model`：过滤、可用性、policy 三段式

**问题与约束：** 一次请求要在多模型、多 worker、多连接模式中选一个可用实例；选择逻辑既要排除 unhealthy / circuit-open worker，又要给 cache-aware、manual、round-robin 等 policy 留输入。

**设计选择：** 先由 `WorkerRegistry` 按模型、worker type、连接模式筛候选，再用 `is_available()` 做健康与熔断过滤，最后调用 policy 选 index。

**读法：** 这段是 HTTP Regular 路由的决策核心。`enable_igw=false` 时忽略 `model_id`，单 router 模式按全局 worker 池选；IGW 模式才按模型索引筛。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/router.rs L140-L191
let workers = self.worker_registry.get_workers_filtered(
    effective_model_id,
    Some(WorkerType::Regular),
    Some(ConnectionMode::Http),
    None,  // any runtime type
    false, // get all workers, we'll filter by is_available() next
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
```

**代码逻辑：** `get_workers_filtered` 返回候选；`is_available` 叠加 health 与 circuit breaker；policy registry 决定当前模型使用哪个负载策略。

**为什么这样写：** 过滤和策略分离后，policy 不需要知道 worker 注册表、连接类型、模型索引这些全局结构，只对“当前可选 worker 列表”决策。这让 policy 可插拔。

**不变量与失败模式：** policy 返回的是 `available` 的 index，不是 registry 的全局 worker id；如果候选为空，必须返回 `None` 并让上层转成 503。HTTP 路径没有 tokens，只能给 prefix policy 提供文本或 hash ring。

**要点：** 这段把“谁有资格接请求”和“谁最适合接请求”拆开，是 gateway 能支持多策略的关键。

### 3.2 `HashRing::new`：一致性哈希只在拓扑变更时构建

**问题与约束：** cache-aware / prefix-hash 类策略希望相似 routing key 稳定落到同一 worker，以提高 KV cache 命中；但 worker 增删时又不能让所有 key 大范围迁移。

**设计选择：** `WorkerRegistry` 为每个 model 维护预计算 `HashRing`，每个 worker 放 150 个 virtual nodes，用 blake3 生成稳定 ring position。

**读法：** 这段构建逻辑说明 hash ring 是拓扑数据，不是请求临时数据。请求路径只做查找，注册/删除 worker 时才 rebuild。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/core/worker_registry.rs L50-L70
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
```

**代码逻辑：** 每个 worker URL 只分配一个 `Arc<str>`，然后为每个 vnode 计算 hash position，把 `(position, url)` 推入 entries；后续会按 position 排序。

**为什么这样写：** virtual nodes 缓解少量 worker 时的分布不均；`Arc<str>` 避免 150 次复制 URL；blake3 保证跨进程、跨 Rust 默认 hasher 变化时仍稳定。

**不变量与失败模式：** hash ring 必须和 model worker snapshot 同步重建；如果 worker 删除后 ring 未更新，policy 可能选到已不存在或已摘除的 URL。

**要点：** 一致性哈希的设计目标不是“平均轮询”，而是“稳定地把相似请求黏在 cache 更可能命中的位置”。

### 3.3 Registry 更新并不是事务式 upsert

**问题与约束：** 动态注册看起来像“按 URL 更新 worker”，但主表、模型 snapshot、类型索引、连接索引和 hash ring 必须一起迁移，才能真正满足 upsert 语义。

**当前实现：** 同 URL 注册会复用 existing `WorkerId` 并覆盖 `workers` 主表，随后仍把 worker/id 追加到模型、类型和连接索引；它没有先从旧索引移除旧 worker。因此重复注册或改变 model/type/connection 的更新可能留下重复项、旧项和重复 virtual nodes。`remove()` 会过滤 snapshot 并重建 ring，但最后一个 worker 被删除后仍可能留下 empty snapshot/ring key，外部 `get_models()` 因过滤空项而看不到这些内部残留。

**读法：** 这里要区分“代码想维护的多索引不变量”与“当前基线实际保证的原子性”。生产控制面应优先 remove 后 register，或在上游修复为先清旧索引再追加；排障时则同时检查主表、model snapshot 与 hash ring，不能只看 URL 是否存在。

---

## 4. 请求代理要同时处理重试、指标与流式生命周期

### 4.1 `route_typed_request`：重试包裹一次转发

**问题与约束：** 上游 worker 可能返回 5xx/429 或网络错误；gateway 既要重试，又要记录 router/worker 维度指标，还要避免每个 API endpoint 复制一套 retry 代码。

**设计选择：** 所有 typed generation request 走 `route_typed_request`；它提取 stream/text/model/endpoint 信息，用 `RetryExecutor` 包裹 `route_typed_request_once`。

**读法：** 这段是 HTTP typed 请求的通用模板。真正的一次 worker 选择和发送发生在闭包里，因此每次 retry 都可以重新执行一次路由。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/router.rs L194-L247
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
                .route_typed_request_once(
                    headers, typed_req, route, model_id, is_stream, &text,
                )
                .await;
            Metrics::record_router_upstream_response(
                metrics_labels::ROUTER_HTTP,
                res.status().as_u16(),
                extract_error_code_from_response(&res),
            );
            res
        },
        |res, _attempt| is_retryable_status(res.status()),
```

**代码逻辑：** 函数先记录请求上下文；retry executor 每次尝试调用 `route_typed_request_once`，所以 worker 过滤与 policy 选择也会重新执行；外层再按 response status 记录成功、非重试错误或耗尽指标。

**为什么这样写：** retry 必须包住“选 worker + send”而不是只包住 “send”，否则第一次选中的坏 worker 会在所有 retry 中重复被打。这里把选择放入 operation 闭包，给非 sticky policy 换 worker 的机会。

**不变量与失败模式：** retry 只根据 attempt 返回的初始 HTTP status 决策。`RetryConfig.max_retries` 在实现中实际是总 attempt 上限：`3` 表示最多执行三次 operation。stream 请求若已经返回 2xx，后续 body error 不会重新进入这里；否则 Gateway 无法保证新 worker 的输出与已经发送的 token 前缀一致。

**要点：** 这段体现 gateway 的生产取舍：请求代理不是裸转发，而是 metrics、retry、policy 重新选择的组合。

### 4.2 `proxy_get_request`：管理 GET 走第一个 healthy worker

**问题与约束：** `/model_info`、`/server_info` 这类 GET 管理信息通常不需要负载均衡策略，但仍要把请求头安全转发给 worker，并在没有健康 worker 时返回明确错误。

**设计选择：** `select_first_worker` 只过滤 `is_healthy()`，选第一个 healthy URL；`proxy_get_request` 用共享 client 发 GET 并按 header 白名单转发。

**读法：** 这段与 generation 热路径形成对照：管理 GET 不走 policy，不用 hash ring，也不触发 typed request retry 模板。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/router.rs L82-L97
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

**代码逻辑：** GET 代理先找一个 healthy worker，构造 `GET {worker_url}/{endpoint}`，再复制允许转发的 request headers。

**为什么这样写：** 管理 GET 的语义是“问集群中任一可用 worker 的状态/信息”，不需要引入复杂 policy。这样减少热路径依赖，也避免某些 policy 因缺少 routing text 而行为不明确。

**不变量与失败模式：** `select_first_worker` 只看 health，不看 circuit breaker；如果 GET endpoint 实际依赖某个特定模型或 worker，本实现会过于粗粒度，需要改为模型感知路由。

**要点：** Gateway 并不是所有 endpoint 都均衡转发。不同 endpoint 的一致性要求不同，代码用不同代理路径表达这一点。

### 4.3 `send_typed_request`：DP-aware 请求改写

**问题与约束：** Data parallel aware 路由需要把逻辑 worker URL 中的 rank 信息传给 backend，但请求的外部 API 仍是普通 JSON；rank 不能依赖 worker 从 URL 里猜。

**设计选择：** 当 `dp_aware` 开启时，从 worker URL 解析 `base_url@rank`，把 `data_parallel_rank` 写入 JSON body，然后发到 base URL；非 DP 路径直接原样 JSON 转发。

**读法：** 这段代码说明 DP rank 是 gateway 在代理层注入的请求字段，而不是用户显式传入的 API 参数。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/router.rs L486-L548
async fn send_typed_request<T: serde::Serialize>(
    &self,
    headers: Option<&HeaderMap>,
    typed_req: &T,
    route: &'static str,
    worker: &Arc<dyn Worker>,
    is_stream: bool,
    load_guard: Option<WorkerLoadGuard>,
) -> Response {
    let worker_url = worker.url();
    let api_key = worker.api_key().clone();
    const DP_RANK_KEY: &str = "data_parallel_rank";

    let mut request_builder = if self.dp_aware {
        let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
            Ok(tup) => tup,
            Err(e) => {
                return error::internal_error(
                    "dp_rank_extraction_failed",
                    format!("Failed to extract dp_rank: {}", e),
                );
            }
        };
        let mut json_val = match serde_json::to_value(typed_req) {
            Ok(j) => j,
            Err(e) => {
                return error::bad_request(
                    "serialization_failed",
                    format!("Convert into serde_json::Value failed: {}", e),
                );
            }
        };
        if let Some(map) = json_val.as_object_mut() {
            map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
        }
```

**代码逻辑：** `send_typed_request` 拿到已选中的 worker；若 `dp_aware` 为真，先解析 worker URL 中的 DP rank，再把 rank 注入 JSON body，并把请求发往不带 `@rank` 后缀的 base URL。

**为什么这样写：** 让 DP-aware 仍复用 typed request/retry/metrics 主路径，避免单独开一套 endpoint。DP 只是 worker URL 和 JSON body 的代理细节。

**不变量与失败模式：** DP-aware 要求 worker URL 形如 `base@rank`；如果 rank 解析失败或 body 不是 JSON object，代理层必须返回错误，不能把错误请求发到 backend。

**要点：** 这部分读源码时要顺着函数调用继续看 `send_typed_request`，不要把 `route_typed_request` 误认为只做普通 HTTP 转发。

### 4.4 流式响应：熔断结果要等 stream 结束

**问题与约束：** SSE/streaming 请求可能先收到 HTTP 200，随后上游连接中途断开；如果收到 200 就记录 worker success，会掩盖“200 后流断”的坏 worker。

**设计选择：** 非流式请求在 response status 返回后立即 `record_outcome`；流式请求跳过 eager 记录，交给 `BreakerTrackedStream` 在 body 生命周期结束时判断。

**读法：** 这段注释非常关键：streaming 的成功条件不是 initial status，而是流干净结束。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/router.rs L613-L648
} else {
    let mut response_headers = header_utils::preserve_response_headers(res.headers());
    response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

    let mut tracked = BreakerTrackedStream::new(
        res.bytes_stream(),
        worker.clone(),
        worker_url.to_string(),
    );
    if !status.is_success() {
        tracked.mark_errored();
    }
    let body = Body::from_stream(tracked);

    let mut response = Response::new(body);
    *response.status_mut() = status;
    *response.headers_mut() = response_headers;

    if let Some(guard) = load_guard {
        response = AttachedBody::wrap_response(response, guard);
    }
    response
}
```

**代码逻辑：** 流式分支保留响应头并强制 SSE content-type；上游 `bytes_stream()` 被包进 `BreakerTrackedStream`，再转换成 Axum body；如果存在 load guard，则绑定到 response body 生命周期。

**为什么这样写：** 对 LLM serving 来说，流式 token 传输是主路径之一。把 stream 生命周期纳入熔断器判断，能更准确地区分“worker 接受请求”与“worker 完整服务请求”。

**不变量与失败模式：** 流式请求若 `send()` 在建立 response 前失败，必须立即记录失败；若 response 已建立，则由 body wrapper 负责最终状态。两者不能重复计数，也不能漏计。

**要点：** 这里是生产 gateway 和玩具 proxy 的差异：生命周期不是函数返回就结束，尤其是 streaming response。

---

## 5. HTTP PD：一次 attempt 同时管理两台 worker

Regular 路由的原子选择单位是一台 worker；HTTP PD 的原子选择单位是一对 `(prefill, decode)`。真正理解这段代码，需要同时追踪四本账：pair、room、两侧 HTTP result、decode stream terminal。

### 5.1 每次 attempt 都重选 pair，并生成新的 room

**问题与约束：** 选中时 worker 已通过 `is_available()` 过滤，但它仍可能在发送阶段发生 transport failure、返回可重试状态，或在选择后状态变化；下一次尝试不能沿用旧 pair 和旧 bootstrap room。

**设计选择：** `execute_dual_dispatch` 把 `select_pd_pair`、请求序列化、bootstrap 注入和双发都放进 retry operation。原始 typed request 用 `Arc` 共享，但 room 是每个 attempt 新生成的。

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/pd_router.rs L389-L430
let response = RetryExecutor::execute_response_with_retry(
    &self.retry_config,
    {
        move |attempt: u32| {
            let shared_request = Arc::clone(&shared_request);
            let context = context.clone();
            async move {
                let (prefill, decode) = match self
                    .select_pd_pair(
                        context.request_text.as_deref(),
                        context.model_id,
                        context.headers.as_ref(),
                    )
                    .await
                {
                    Ok(pair) => pair,
                    Err(e) => return Self::handle_server_selection_error(e),
                };

                let mut json_request =
                    match serde_json::to_value(shared_request.as_ref()) {
                        Ok(v) => v,
                        Err(e) => return Self::handle_serialization_error(e),
                    };
                json_request = match Self::inject_bootstrap_into_value(
                    json_request,
                    prefill.as_ref(),
                    context.batch_size,
                ) {
                    Ok(v) => v,
                    Err(e) => return Self::handle_serialization_error(e),
                };
```

**不变量：** retry 是整对重放，不是只补发失败的一侧。调整 retry 次数会同时增加 prefill 计算、decode 占位和 KV transfer 成本。

### 5.2 两侧 body 共享 room，但 DP rank 含义不同

**问题与约束：** Prefill 和 Decode 必须对同一个 bootstrap room 达成一致；DP-aware 部署还要分别告诉两侧“自己在哪个 DP rank”，并告诉 Decode“Prefill 来自哪个 rank”。

**设计选择：** 先向共享 JSON 注入 bootstrap host/port/room；`prepare_worker_request` 再调用各自 worker 的 `prepare_request` 注入自身 `data_parallel_rank`；最后只给 decode body 增加 `disagg_prefill_dp_rank`。

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/pd_router.rs L349-L363
async fn prepare_pd_worker_requests<'a>(
    route: &'static str,
    json_request: &'a Value,
    prefill: &dyn Worker,
    decode: &dyn Worker,
) -> Result<(PreparedWorkerRequest<'a>, PreparedWorkerRequest<'a>), String> {
    let prefill_request =
        Self::prepare_worker_request(route, prefill, Cow::Borrowed(json_request)).await?;
    let decode_json_request =
        Self::inject_prefill_dp_rank_for_decode(Cow::Borrowed(json_request), prefill)?;
    let decode_request =
        Self::prepare_worker_request(route, decode, decode_json_request).await?;
    Ok((prefill_request, decode_request))
}
```

**不变量：** `data_parallel_rank` 是目标 worker 自身 rank；`disagg_prefill_dp_rank` 是 Decode 用来识别 Prefill 来源的 rank。二者不能互换。

### 5.3 HTTP 双发并行，KV 不经过 Gateway

**问题与约束：** Decode 要尽早接收 bootstrap metadata 并准备接收 KV；如果 Gateway 等 Prefill HTTP 完成后才发 Decode，会把两池重新串行化。

**设计选择：** 两个 request builder 先完成，然后用 `tokio::join!` 同时 poll 两个 `send()` future。Gateway 只接收两侧 HTTP response；KV tensor 在 worker 间按 transfer backend 传输。

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

**边界：** `join!` 只证明两次 HTTP send 并发，不证明两侧 GPU 计算互不依赖。Decode 仍要等待相同 room 的 KV/metadata 可用；而 Gateway 在拿到两侧 response 后，还会先完整消费 Prefill body，再构造或返回 Decode response，所以 Decode SSE 对客户端的可见时间仍受 Prefill body 门控。

### 5.4 最终 response status 不能替代分侧 breaker 归因

**问题与约束：** Decode transport failure 可以让客户端看到 502，但 Prefill 可能已经正常返回；若用最终 502 同时惩罚两台 worker，会错误打开健康 Prefill 的 breaker。

**设计选择：** Decode transport failure、Decode non-2xx 等 decode 驱动分支会在内层分别 `record_outcome`，再给 response 插入 `BreakerOutcomesRecorded` marker，让外层跳过按最终 status 的统一归因。streaming Decode 则交给 `BreakerTrackedStream` 在 body 结束时记录。

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/http/pd_router.rs L840-L872
Err(e) => {
    decode.record_outcome(false);
    let prefill_ok = match &prefill_result {
        Ok(res) => {
            let s = res.status();
            s.is_success() || s.is_client_error()
        }
        Err(_) => false,
    };
    prefill.record_outcome(prefill_ok);

    let mut response = error::bad_gateway(
        "decode_server_error",
        format!("Decode server error: {}", e),
    );
    response.extensions_mut().insert(BreakerOutcomesRecorded);
    response
}
```

**已覆盖边界与缺口：** 客户端看到的是一次请求的最终响应；breaker 理想上应记录每台 worker 的真实责任，不能用同一个布尔值粗暴替代。当前闭环主要覆盖 decode 驱动错误。若 Decode 已取得成功 response，而 `process_prefill_response()` 随后失败，内层没有插 marker：外层可能按最终 Prefill error 粗粒度归因，非流式会把两侧都记失败，流式则可能只记 Prefill 且不再给 Decode 建 wrapper。pair 选中后的 request preparation 错误也可能走统一归因。因此本文不能把“分侧 breaker”写成所有组合都成立的保证。

### 5.5 流式 2xx 是自动重试的终点

`create_streaming_response` 在 Decode byte stream 外包 `BreakerTrackedStream`。HTTP PD 会检查 chunk 中的 `data: [DONE]`：先 `mark_completed()`，再把该 chunk 发给客户端，随后 break。因此它不会继续观察 `[DONE]` 后的 trailing error；客户端恰在该 chunk 上断开时也仍记 success。通用 HTTP router 不识别 `[DONE]`，只有底层 stream 返回 `None` 才算 completed。wrapper 的 `mark_completed()` 也不是吸收态：若调用后继续 poll 到 error，仍会升级为 Errored。客户端在没有 clean terminal 时主动断开则保持 Active，不记结果。

这条边界解释了两个生产现象：

- 日志里“router 初始成功”与“decode breaker 最终失败”可以同时成立。
- 客户端恢复必须重新发起应用级请求；Gateway 不能证明另一台 worker 会生成完全相同的剩余 token。

### 5.6 `get_server_info` 的注释与调用对象冲突

`PDRouter::get_server_info` 的注释写“first decode server”，但实际调用 `proxy_to_first_prefill_worker("server_info", None)`。在这条基线上只能陈述“代码实际代理到 Prefill helper”，不能依据注释断言结果来自 Decode。若两池 server info 不同，这会直接影响运维判断，也值得作为上游待澄清点。

---

## 6. RouterManager 把单路由和 IGW 统一成同一个 trait

### 6.1 `RouterManager` 结构：DashMap 注册，ArcSwap 快照读取

**问题与约束：** IGW 模式下同一进程可能同时存在 HTTP Regular、HTTP PD、gRPC Regular、gRPC PD、OpenAI router；请求热路径要快速遍历可用 router，但注册阶段又需要动态添加。

**设计选择：** `routers` 用 `DashMap` 管理注册，`routers_snapshot` 用 `ArcSwap<Vec<Arc<dyn RouterTrait>>>` 给热路径做无锁快照遍历，`default_router` 兜底单路由模式。

**读法：** 这段结构定义了 RouterManager 的核心职责：它不是一个具体转发器，而是多个 `RouterTrait` 实例的选择器。

**源码锚点：**

```rust
// 定位骨架（非逐行摘录）：来源 sgl-model-gateway/src/routers/router_manager.rs L62-L78
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
```

**代码逻辑：** manager 持有 worker registry、router map、读路径快照、默认 router id 和 IGW 开关。构造时先空注册，`from_config` 再填充。

**为什么这样写：** 多 router 注册是低频事件，请求选择是高频事件。DashMap 适合注册/查找，ArcSwap 快照适合高频遍历；二者组合避免每个请求都持有 map shard lock。

**不变量与失败模式：** register router 后必须刷新 snapshot；否则新 router 已在 DashMap 中存在，但无 model 请求或无明确 model 的热路径可能看不到它。

**要点：** RouterManager 是 gateway 的“路由路由器”：先选哪类 router，再由具体 router 选 worker。

### 6.2 IGW 选的是下游能力，不是客户端协议

客户端请求始终先进入 Axum HTTP handler。给定 model 时，`get_router_for_model` 遍历该模型的 worker，并按 external > gRPC PD > HTTP PD > gRPC Regular > HTTP Regular 评分；这里的 HTTP/gRPC 描述 Gateway 到 worker 的连接。没有 model 时才遍历 `routers_snapshot`，`x-prefer-pd` 只改变 PD/Regular 偏好，并结合全局 worker distribution 排除没有对应 worker 的模式。这个选择仍是粗粒度的 router 分类，具体 router 随后必须再做 model、type、connection、health 与 breaker 过滤。

---

## 7. 健康与管理端点区分存活、就绪和集群操作

### 7.1 `liveness`：进程活着即可

**问题与约束：** Kubernetes 等编排系统会区分 liveness 和 readiness。liveness 如果依赖 worker 状态，worker 故障可能导致 gateway 进程被误杀，放大故障。

**设计选择：** `liveness` 只返回 OK，不查 registry、不查 worker、不触发网络请求。

**读法：** 这段很短，但语义明确：liveness 只回答“gateway 进程是否还能响应”。

**源码锚点：** 来源：`sgl-model-gateway/src/server.rs` L98-L100。`liveness` 直接构造 HTTP 200 响应，body 为 `OK`。

**代码逻辑：** 直接构造 `(200, "OK")` response。

**为什么这样写：** liveness 是进程健康，不是服务容量。把 worker readiness 放到 readiness/health_generate，能避免 backend 暂时不可用时入口服务被重启。

**不变量与失败模式：** 这个 endpoint 不能被扩展成 worker 检查；否则 readiness 与 liveness 语义会混淆，部署系统可能在下游故障时杀掉上游 gateway。

**要点：** 读运维相关代码时要先看语义边界。这里的“少做事”是设计选择。

### 7.2 `flush_cache`：集群级操作由 WorkerManager fan-out

**问题与约束：** Gateway 后面可能有多个 worker。对外一个 `/flush_cache` 请求需要变成对所有 backend 的清理操作，而不是只清理某个被选中的 worker。

**设计选择：** handler 调用 `WorkerManager::flush_cache_all`，把 registry 和共享 client 交给管理层，由它并发 fan-out 到 worker。

**读法：** `flush_cache` 是控制面广播型操作，不走普通请求选 worker policy。

**源码锚点：**

```rust
// 来源：sgl-model-gateway/src/server.rs L401-L405
async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    WorkerManager::flush_cache_all(&state.context.worker_registry, &state.context.client)
        .await
        .into_response()
}
```

**代码逻辑：** 从 `AppContext` 取 registry 和 client，调用 `flush_cache_all`，再转 HTTP response。

**为什么这样写：** cache flush 是全局副作用，不能用负载均衡挑一个 worker。把它放在 WorkerManager 可以复用 worker 列表、认证、并发请求和错误聚合逻辑。

**不变量与失败模式：** 如果有 worker flush 失败，返回结果必须能反映失败，否则用户会误以为整个集群 cache 都已清空。

**要点：** 管理 endpoint 的核心不是“请求路由”，而是“对拓扑执行一致操作”。

---

## 延伸阅读：不混入请求主链的控制面

以下能力重要，但不应塞进一次 generation request 的 walkthrough：

| 能力 | 入口 | 应去哪里读 |
|------|------|------------|
| function-call / reasoning 本地解析 | `server.rs::parse_function_call`、`parse_reasoning` | 协议与 parser 专题 |
| service discovery | `start_service_discovery` | Gateway 部署与 worker 生命周期文档 |
| round-robin、cache-aware、consistent hash | `policies/` | [[SGLang-model-gateway-核心概念]] |
| HTTP/gRPC 健康阈值、DP-aware worker、load guard | `core/worker.rs` | [[SGLang-model-gateway-数据流]]、[[SGLang-model-gateway-排障指南]] |

把它们移出主线不是降低重要性，而是保持 walkthrough 的对象单一：本篇只追一次请求如何从 endpoint 进入、选 router/worker、完成 Regular 或 PD 转发，并在 response 生命周期结束时归还状态。

---

## 运行验证

维护本文时，先用下面的命令确认 gateway 主线还在原位：

```powershell
rg -n "create_router|route_typed_request_once|execute_dual_dispatch|tokio::join|BreakerOutcomesRecorded|BreakerTrackedStream|RouterManager|WorkerRegistry" sglang/sgl-model-gateway/src
```

预期信号：

- `routers/factory.rs` 仍能找到 router 创建入口。
- `routers/http/pd_router.rs` 仍能找到每次 attempt 选 pair、bootstrap 注入、并行双发和分侧 breaker 归因。
- `routers/router_manager.rs` 仍能找到单路由与 IGW 统一管理入口。
- `core/worker_registry.rs` 仍承载 model/type/connection 索引与 hash ring。

如果其中某类命中消失，不要直接沿用本文结论；先回到对应模块确认职责是迁移、合并，还是被新的控制面替代。
