---
title: "model-gateway · 学习检查"
type: exercise
framework: sglang
topic: "model-gateway"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---

# model-gateway · 学习检查

## 你为什么要做这组检查

目标不是背文件名，而是证明你能把 Gateway 的 router 选择、worker 选择、PD 配对、retry 和 streaming terminal 分开。通过标准是：给出一个线上症状时，你能指出状态所有者、源码入口、可执行操作和预期，而不是把所有 5xx 都归咎于 SRT。

## 一、闭卷主线

- [ ] 能说明 model-gateway 是 Rust Axum 代理与路由层，不执行 GPU forward。
- [ ] 能画出 Regular 主链：Client → Axum handler →（IGW 时先经 `RouterManager`）→ concrete router → registry/filter → policy → worker → response body。
- [ ] 能画出 HTTP PD 主链：每个 attempt 选 PF/DC → 生成 room → 两侧 body 准备 → `tokio::join!` 双发 → worker 间传 KV → Decode response 经 Gateway 返回。
- [ ] 能解释 `is_healthy`、`is_available`、单一 PD readiness、IGW readiness 四者为何不能互换。
- [ ] 能解释 `data_parallel_rank` 与 `disagg_prefill_dp_rank` 分别属于谁。
- [ ] 能说明为什么客户端收到 token 后，Gateway 不能换 worker 继续同一条 SSE。
- [ ] 能说明 `max_retries=3` 为什么只产生三次总 attempt。
- [ ] 能区分通用 HTTP 的 clean EOF 与 HTTP PD 的 `[DONE]` terminal，并说明 `[DONE]` 后为何不再观察 trailing error。
- [ ] 能指出重复 URL 注册不是事务式 upsert，以及它可能污染哪些索引。

## 二、静态证据链

在仓库根目录运行：

```powershell
rg -n 'async fn v1_chat_completions|route_chat' sglang/sgl-model-gateway/src/server.rs
rg -n 'select_worker_for_model|route_typed_request_once|execute_response_with_retry' sglang/sgl-model-gateway/src/routers/http/router.rs
rg -n 'select_pd_pair|inject_bootstrap_into_value|prepare_pd_worker_requests|tokio::join|BreakerOutcomesRecorded' sglang/sgl-model-gateway/src/routers/http/pd_router.rs
rg -n 'is_retryable_status|REQUEST_TIMEOUT|TOO_MANY_REQUESTS|GATEWAY_TIMEOUT' sglang/sgl-model-gateway/src/core/retry.rs
rg -n 'Terminal::Completed|Terminal::Errored|Terminal::Active' sglang/sgl-model-gateway/src/routers/streaming_utils.rs
rg -n 'let max = config.max_retries.max\(1\)|attempt \+ 1 >= max' sglang/sgl-model-gateway/src/core/retry.rs
rg -n 'pub fn register|existing_id|model_index|worker_type_index|connection_mode_index|pub fn remove' sglang/sgl-model-gateway/src/core/worker_registry.rs
rg -n 'get_server_info|proxy_to_first_prefill_worker\("server_info"' sglang/sgl-model-gateway/src/routers/http/pd_router.rs
```

预期：

- 第一组只看到 handler 委托，不看到 GPU forward。
- Regular 路径的 retry operation 内包含 `route_typed_request_once`，说明每次 attempt 可以重新选 worker。
- PD 路径的 `tokio::join!` 同时持有 prefill/decode send future，不存在“Prefill HTTP 返回后 Gateway 才发送 Decode”的串行边；但 Decode response 交给客户端前仍先消费 Prefill body，两者不能混为一谈。
- retryable status 精确为 `408/429/500/502/503/504`。
- stream clean end、upstream error、客户端中途 drop 分别落到 Completed、Errored、Active。
- retry 上限用 `max_retries.max(1)` 和 `attempt + 1 >= max` 表达，证明配置值是总 attempt 数。
- registry 搜索应同时暴露 existing-id 复用与多索引追加；`get_server_info` 搜索应暴露注释/Prefill helper 的基线冲突。

任一预期不成立，都不能继续沿用本文结论；先确认源码基线是否迁移。

## 三、故障推演

### 推演 A：readiness 200，但目标 PD 模型请求 503

写出两种不同模式的判断：

- 单一 `PrefillDecode`：readiness 200 应证明全局至少有一台 healthy PF 和一台 healthy DC。
- IGW：readiness 200 只证明存在任意 healthy worker；目标 model 仍可能缺 PF/DC 配对。

合格答案必须进入 `server.rs::readiness` 和 `PDRouter::select_pd_pair` 两个入口，不能只查 `/health`。

### 推演 B：Prefill 成功，Decode 建连失败

要求说明：

1. 为什么最终客户端可能看到 502。
2. 为什么 Prefill breaker 不应跟着记 failure。
3. 为什么下一 retry attempt 会重放两侧，并可能生成新 room。

### 推演 C：客户端已收到三个 token 后断流

先区分：

- upstream decode stream error：decode breaker failure；不能透明续传。
- 客户端主动取消：drop upstream，breaker 不记结果。

如果答案是“Gateway 自动重试并从第四个 token 继续”，则未通过。

### 推演 D：Decode 已返回 2xx，Prefill body 随后读取失败

要求说明：

1. 为什么并行双发不等于 Decode SSE 已经交付客户端。
2. 为什么这条路径不能套用“breaker 一定精确分侧”的结论。
3. 非流式与流式分别可能怎样漏记或误记 Decode 结果。

### 推演 E：同 URL 从模型 A 更新为模型 B

要求沿 `WorkerRegistry::register` 检查主表、model snapshot、type index、connection index 和 hash ring。若答案只说“主表覆盖，所以更新是幂等的”，则未通过。

## 四、可运行验证

若本机有 Rust toolchain：

```powershell
Set-Location sglang/sgl-model-gateway
cargo test routers::streaming_utils --lib
cargo test routers::http::pd_router --lib
cargo test core::worker_registry --lib
```

预期：stream terminal 与 PD request preparation 相关单测通过。若当前环境没有 `cargo`，静态替代是逐项核对以下测试仍存在：

```powershell
rg -n 'drop_while_active_records_nothing|clean_stream_records_one_success|stream_error_records_one_failure|test_prepare_pd_worker_requests_uses_dp_aware_rank|test_streaming_load_tracking' sglang/sgl-model-gateway/src
```

静态替代只能证明契约被测试文件表达，不能证明当前 crate 可编译或运行。

## 复盘

如果以上检查通过，你应能用一句话概括：**Gateway 的原子路由单位在 Regular 是一台 worker，在 HTTP PD 是一对 worker；双发并行不消除 Prefill body 对交付的门控，自动重试止于 response 交付，而 breaker 分侧归因必须按具体失败分支审计。**

主链见 [[SGLang-model-gateway-源码走读]]，对象形态见 [[SGLang-model-gateway-数据流]]，线上入口见 [[SGLang-model-gateway-排障指南]]。
