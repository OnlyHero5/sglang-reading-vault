---
title: "可观测性 · 学习检查"
type: exercise
framework: sglang
topic: "可观测性"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-12
---
# 可观测性 · 学习检查

## 读者能做什么

- [ ] 能画出四本账：scrape 入口账、Scheduler 状态账、Tokenizer 请求账、旁路账。
- [ ] 能解释 `/metrics` 只是读取面，不是所有指标的生成点。
- [ ] 能分别沿 HTTP 的 `add_prometheus_middleware` 与 gRPC sidecar 的 `_add_metrics_routes` 复述 scrape lane。
- [ ] 能沿 `SchedulerMetricsReporter -> SchedulerStats -> SchedulerMetricsCollector.log_stats` 复述 Scheduler lane。
- [ ] 能沿 `TokenizerManager.collect_metrics -> TTFT / ITL / E2E` 复述 Tokenizer lane。
- [ ] 能说出 RequestLogger、OpenTelemetry trace、RequestMetricsExporter 为什么不等于 Prometheus metrics。
- [ ] 能解释默认只让 `attn_tp_rank == 0` 写 Scheduler metrics 的原因。
- [ ] 能说明 `enable_metrics_for_all_schedulers` 会展开 rank series，并区分普通 TP 副本与 DP-Attention scheduler 的聚合语义。
- [ ] 能解释 ITL 为什么是基于累计 token 增量的平均近似，而不是逐 token 时间戳或网络 chunk latency。
- [ ] 能解释 custom labels 必须先进入 allowed list 和 collector label schema。
- [ ] 能说明 `cache_hit_rate` 是 stats tick aggregate，不是单请求字段。

## 源码入口自检

| 你要解释的现象 | 应该能指出的源码入口 |
|----------------|----------------------|
| `/metrics` 如何挂载 | `python/sglang/srt/entrypoints/http_server.py` 的 `lifespan` |
| gRPC `/metrics` 如何暴露 | `python/sglang/srt/entrypoints/grpc_server.py` 的 `_add_metrics_routes` 与 sidecar ready hook |
| multiprocess registry 如何暴露 | `python/sglang/srt/utils/common.py` 的 `add_prometheus_middleware` |
| Scheduler rank 选择 | `SchedulerMetricsCollector.init_new` |
| `cache_hit_rate` 计算 | `SchedulerMetricsReporter` 的 prefill stats tick |
| Scheduler gauge 写入 | `SchedulerMetricsCollector.log_stats` |
| TTFT、ITL、E2E 写入时机 | `TokenizerManager.collect_metrics` |
| request finished 日志 | `RequestLogger.log_finished_request` |
| trace 初始化失败 | `process_tracing_init` |
| request metrics 文件导出 | `RequestMetricsExporterManager.write_record` |
| 热更新 duration | `observe_weight_load` 和 weight updater |

## 排障演练

不打开正文，试着回答这些问题：

- [ ] `/metrics` 404 时，你会如何先按 HTTP / gRPC 模式选择正确的源码函数与端口？
- [ ] `/metrics` 有 HTTP counter，但没有 `cache_hit_rate`，你会如何分层排查？
- [ ] 多 TP 部署里只有 `attn_tp_rank == 0` 有 Scheduler gauge，这是 bug 还是默认设计？
- [ ] gRPC 模式主端口没有 `/metrics` 时，为什么应该先找 sidecar 端口而不是 Scheduler collector？
- [ ] `priority="None"` 出现时，应该改请求、启动参数还是 Grafana 查询？
- [ ] custom label header 已发送但 series 没 label，可能缺哪两步？
- [ ] TTFT 有值但 E2E 没出现，和请求生命周期哪个阶段有关？
- [ ] request.finished 日志没有，但 Prometheus histogram 有值，说明哪条旁路没打开？
- [ ] trace 没有 span 时，为什么查 Prometheus 没用？
- [ ] 热更新后 cache hit 下降，为什么不一定是 RadixAttention 坏了？
- [ ] exporter 文件没写时，为什么要确认请求是否 finished？

## 最小运行验证

| 验证目标 | 操作 | 预期现象 |
|----------|------|----------|
| scrape 入口 | 启动时加 `--enable-metrics`；HTTP 模式访问主端口，gRPC 模式访问 sidecar 端口 | 返回 Prometheus 文本；关闭开关时不应有同样的 metrics route |
| Scheduler gauge | 在 `SchedulerMetricsCollector.log_stats` 打断点或加临时日志 | stats logging rank 进入；非默认 rank 只有 all-scheduler 时进入 |
| cache hit | 发送重复前缀请求，再观察 `cache_hit_rate` | 命中率随 workload 重复度变化，热更新后可能下降 |
| Tokenizer latency | 在 `TokenizerManager.collect_metrics` 打断点，记录前后累计 `completion_tokens` | 首次输出写 TTFT；后续新增 token 共享“事件间隔 / 新增数”的 ITL；finished 写 E2E |
| custom labels | 配置 allowed label 并发送 header | Tokenizer request metrics 出现对应 label value |
| RequestLogger | 关闭 metrics 但开启 request logging | 仍可写 request received 或 finished 日志 |
| exporter | 启用 file exporter 并发送完成请求 | 目标目录出现按小时命名的请求 metrics 文件 |
| trace | 配置 OpenTelemetry 和 OTLP endpoint | request stage 产生 span，失败时先看依赖和 endpoint |

## 学习复盘

如果你能完成以上自检，这个专题的核心模型就算建立起来：

1. 可观测性不是中心化收集器，而是多条写入 lane 共享一个 scrape 面。
2. Scheduler metrics 是系统窗口快照；Tokenizer metrics 是请求生命周期事件。
3. HTTP middleware、Gateway、RequestLogger、trace、exporter 各自回答不同问题。
4. label 和 rank 问题既受 collector 初始化与写入边界约束，也受查询聚合方式影响；普通 TP 与 DP-Attention 不能套用同一种 `sum` 规则。
5. 热更新和请求完成这类边沿事件，要找事件触发点，不要只盯周期性 stats tick。

下一步建议回到 [[SGLang-可观测性]]，把四本账模型和 [[SGLang-Scheduler]]、[[SGLang-TokenizerManager]]、[[SGLang-CheckpointEngine]] 串起来。
