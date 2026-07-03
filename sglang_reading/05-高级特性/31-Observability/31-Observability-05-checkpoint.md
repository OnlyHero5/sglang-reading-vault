---
type: batch-doc
module: 31-Observability
batch: "31"
doc_type: checkpoint
title: "可观测性 验收清单"
tags:
 - sglang/batch/31
 - sglang/module/observability
 - sglang/doc/checkpoint
updated: 2026-07-02
---
# 可观测性 验收清单

## 读者自测（不打开 sglang/）

- [ ] 仅读本模块 sglang_reading，能口头说明 Observability 的职责
- [ ] 能画出 Scheduler → MetricsReporter → /metrics 的完整链路
- [ ] 能说出 3 个核心类/函数及其职责（SchedulerMetricsCollector、MetricsReporter.log_stats、add_prometheus_middleware）
- [ ] 能追踪 metrics 从 record 到 `/metrics` 暴露的完整链路
- [ ] 能说明 weight_load metrics 与 32-CheckpointEngine 的关系
- [ ] 五篇正文 ≥ 15 段内嵌源码，每段后有中文讲解

## 核心结论（3 句话）

1. `--enable-metrics` 驱动 SchedulerMetricsCollector + TokenizerMetricsCollector，经 multiprocess 聚合暴露于 `/metrics`。
2. SchedulerStats → MetricsReporter.log_stats → Prometheus gauge；请求 finish → observe_one_finished_request → TTFT/ITL/e2e histogram。
3. RequestLogger 提供可选结构化请求日志，与 Gateway metrics 互补；权重热更新 metrics 见 32-CheckpointEngine 交叉引用。

## Observability 核心函数/类清单

| 符号 | 职责 |
|------|------|
| `SchedulerMetricsCollector.init_new` | 按 rank/args 创建 collector |
| `SchedulerMetricsCollector.log_stats` | SchedulerStats → Prometheus gauge |
| `SchedulerMetricsCollector.observe_weight_load` | 热更新 duration（edge-triggered） |
| `TokenizerMetricsCollector.observe_one_finished_request` | 请求 finish histogram/counter |
| `MetricsReporter` | Scheduler stats 组装与 periodic log_stats |
| `Scheduler.init_metrics_collector` | Scheduler 启动时绑定 collector |
| `add_prometheus_middleware` | HTTP `/metrics` ASGI 挂载 |
| `RequestLogger` | 可选结构化请求日志 |
| `ReqTimeStats.observe_per_stage_req_latency` | 单请求 stage latency |
| `RequestMetricsExporter` | 请求性能数据外部导出 |
| `resolve_collector_class` | stat_loggers DI 扩展点 |

## 遗留问题

- Ray Serve 嵌入场景需结合 `ray_wrappers.py` 单独成文
- checkpoint-engine 外部 ParameterServer 协议细节见 MoonshotAI/checkpoint-engine 仓库
