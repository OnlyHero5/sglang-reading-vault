---
type: module-moc
module: 31-Observability
batch: "31"
doc_type: moc
title: "可观测性（Observability）"
tags:
 - sglang/batch/31
 - sglang/module/observability
 - sglang/doc/moc
aliases:
 - "README"
updated: 2026-07-02
---
# 可观测性（Observability）

> **阶段 V · 高级特性** | Git：`70df09b` 
> **源码范围：** `srt/observability/`、`srt/utils/request_logger.py`、`entrypoints/http_server.py`（`/metrics`）、`managers/scheduler_components/metrics_reporter.py` 
> **前置专题：** [[07-Scheduler-00-MOC]]（SchedulerStats 采集）、[[27-model-gateway-00-MOC]]（Gateway 层 Prometheus 对比） 
> **关联专题：** [[12-ModelLoader-00-MOC]] §weight_sync · [[32-CheckpointEngine-00-MOC]]（权重热更新 metrics）

---

## 1. 本模块目标

**Explain：** SGLang 生产部署需要同时回答三类问题：当前有多少请求在跑/排队？prefix cache 命中率多少？单请求 TTFT/ITL/e2e 延迟分布如何？本模块覆盖 **Prometheus metrics**（`--enable-metrics`）、**Scheduler 周期 stats**（`SchedulerStats` → `MetricsReporter`）、**RequestLogger** 结构化日志，以及权重热更新相关的 `num_paused_reqs` / `weight_load_duration_seconds`。这些能力分散在 `observability/` 与 HTTP 中间件，而非单一文件。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L274-L276
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()
```

**Comment：**

- 多进程 Scheduler 子进程通过 `PROMETHEUS_MULTIPROC_DIR` 写入 gauge 文件，HTTP 层 `MultiProcessCollector` 聚合后暴露 `/metrics`。
- gRPC 模式 metrics 可能在 sidecar 端口（`grpc_http_sidecar_port`）；HTTP 默认路径 `/metrics` 在 auth 白名单，无需 API key。

---

## 2. 最关键入口：`/metrics` 挂载

**Explain：** HTTP 服务启动时在 ASGI app 上挂载 Prometheus multiprocess collector；`PROMETHEUS_MULTIPROC_DIR` 必须在 import `prometheus_client` 之前设置，否则 fork 出的 Scheduler 子进程 gauge 无法聚合。这是运维抓取 `sglang:*` 指标的唯一 HTTP 入口。

**Code：**

```python
# 来源：python/sglang/srt/utils/common.py L1589-L1599
def add_prometheus_middleware(app):
    # We need to import prometheus_client after setting the env variable `PROMETHEUS_MULTIPROC_DIR`
    from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
```

**Comment：**

- `MultiProcessCollector` 聚合 fork 出的 Scheduler worker 写入的 `.db` 文件。
- `/metrics` 路径在 auth 层白名单，便于 K8s probe 与 Prometheus scrape。
- 可通过 `--uvicorn-access-log-exclude-prefixes /metrics` 减少 access log 噪音。

---

## 3. 热更新 metrics 交叉引用

运行时权重热更新（checkpoint-engine / IPC）的 `weight_load_duration_seconds` 与 `num_paused_reqs` 见本模块 [[31-Observability-04-关键问题|04-关键问题]] Q4；完整 IPC 流程见 [[32-CheckpointEngine-00-MOC]] 与 [[12-ModelLoader-00-MOC]] §weight_sync。

---

## 4. 文档导航

| 文件 | 内容 |
|------|------|
| [[31-Observability-01-核心概念]] | Metrics 术语、SchedulerStats、与 Gateway Prometheus 区别 |
| [[31-Observability-02-源码走读]] | SchedulerMetricsCollector、log_stats、HTTP middleware |
| [[31-Observability-03-数据流与交互]] | Scheduler → MetricsReporter → /metrics scrape |
| [[31-Observability-04-关键问题]] | enable_metrics 开销、weight_load metrics、grammar 队列 |
| [[31-Observability-05-checkpoint]] | 验收清单 |

---

→ 下一专题：[[32-CheckpointEngine-00-MOC|32-CheckpointEngine]]
