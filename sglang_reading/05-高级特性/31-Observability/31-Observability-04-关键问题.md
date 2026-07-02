---
type: batch-doc
module: 31-Observability
batch: "31"
doc_type: faq
title: "可观测性：关键问题"
tags:
 - sglang/batch/31
 - sglang/module/observability
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# 可观测性：关键问题

## Q1：如何开启 metrics？

**Explain：** 启动 server 时加 `--enable-metrics`；多 Scheduler 子进程场景需确保 `PROMETHEUS_MULTIPROC_DIR` 已设置（launch_server 自动创建临时目录）。Grafana 抓取 `http://<host>:<port>/metrics`，无需 API key。可选 `--enable-mfu-metrics` 开启估算 MFU 相关指标。

**Code：**

```python
# 来源：python/sglang/srt/server_args.py L1070-L1071
    enable_metrics: A[bool, "Enable log prometheus metrics."] = False
    grpc_http_sidecar_port: A[
```

**Comment：**

- K8s PodMonitor 指向同一 /metrics 路径。
- extra_metric_labels 可追加自定义 label。
- 建议配合 `--uvicorn-access-log-exclude-prefixes /metrics` 降噪。

---

## Q2：srt metrics 与 Gateway Prometheus 有何区别？

**Explain：** Gateway metrics 反映**路由与负载均衡**（哪个 backend、排队深度、HTTP 状态码）；srt metrics 反映**推理引擎内部**（KV pool 使用率、spec 接受率、grammar 编译时间）。两者互补，不应混为同一 dashboard 的唯一数据源。

| 场景 | 看 Gateway | 看 srt |
|------|-----------|--------|
| 路由错误 / 502 | ✓ | |
| KV OOM / retract | | ✓ |
| TTFT P99 | 部分（HTTP 层） | ✓（engine 精确） |
| PD transfer 慢 | | ✓ kv_transfer_* |
| 权重热更新暂停 | | ✓ num_paused_reqs |

---

## Q3：性能开销多大？

**Explain：** metrics 开销主要来自：（1）每 stats_interval 一次 `log_stats` 遍历大量 Gauge.set（默认 interval 可配置）；（2）请求 finish 时 histogram observe（O(1) per request）；（3）multiprocess 文件 IO。`enable_metrics=False` 时 collector 为 None，热路径无 Prometheus 调用。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1051-L1052
        collector: Optional[SchedulerMetricsCollector] = None
        if enable_metrics:
```

**Comment：**

- RequestLogger 独立开关；高 QPS 下建议 log_requests_level=1 或关闭 JSON 大 payload 日志。
- enable_mfu_metrics 额外 increment FLOPs/bytes counter，开销略高。

---

## Q4：权重热更新相关 metrics

**Explain：** 运行时 IPC/disk 权重更新会 pause 请求并记录 duration；与 [[32-CheckpointEngine-00-MOC]] 及 [[12-ModelLoader-00-MOC]] §weight_sync 交叉阅读。监控热更新应同时看 `num_paused_reqs`、`weight_load_duration_seconds{source="ipc"}`、`cache_hit_rate`（flush 后骤降属预期）。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L166-L169
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
```

**Comment：**

- initial_weights_loaded 与 wait_weights_before_ready 见 CheckpointEngine。
- flush_cache 热更新后 prefix cache 失效（见 15-RadixAttention）。

---

## Q5：enable_metrics_for_all_schedulers 何时用？

**Explain：** 默认仅 attn_tp_rank==0 上报 Scheduler gauge，避免 TP 副本重复。调试多 Scheduler 或 dp_attention 场景需 per-rank 指标时开启 `--enable-metrics-for-all-schedulers`；Grafana 需按 tp_rank label 过滤，cardinality 随副本数线性增长。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1040-L1044
        enable_metrics = server_args.enable_metrics
        is_stats_logging_rank = ps.attn_tp_rank == 0
        current_scheduler_metrics_enabled = enable_metrics and (
            is_stats_logging_rank or server_args.enable_metrics_for_all_schedulers
        )
```

**Comment：**

- 生产默认关闭。
- dp_attention 开启时 TP0 指标可能不代表全局。

---

## Q6：Request logging 与 metrics 如何配合排障？

**Explain：** metrics 给 aggregate SLA（P99 TTFT、queue 深度）；RequestLogger JSON 给单请求 rid 级 replay。schedule_simulator 可 import request_logger JSON 做离线调度仿真。OpenTelemetry trace（observability/trace.py）可并存，提供分布式 trace。

**Code：**

```python
# 来源：python/sglang/srt/utils/request_logger.py L44-L55
class RequestLogger:
    def __init__(
        self,
        log_requests: bool,
        log_requests_level: int,
        log_requests_format: str,
        log_requests_target: Optional[List[str]],
    ):
        self.log_requests = log_requests
        self.log_requests_level = log_requests_level
        self.log_requests_format = log_requests_format
        self.log_requests_target = log_requests_target
```

**Comment：**

- log_requests_level: 0=off, 1=metadata, 2=+input, 3=+output。
- SGLANG_LOG_REQUEST_EXCEEDED_MS 过滤慢请求。

---

## Q7：cache_hit_rate 如何解读？

**Explain：** `cache_hit_rate` 反映 RadixAttention prefix cache 命中比例，由 Scheduler 在 stats tick 计算并写入 gauge。权重热更新 flush cache 后该值会骤降，属预期行为。长期偏低可能说明 workload 前缀重复度低或 cache 容量不足。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1265-L1266
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)
```

**Comment：**

- 与RadixAttention RadixAttention 联动。
- HiCache L3 另有 storage 相关 cached_tokens metrics。

---

## 附录：CheckpointEngine 交叉引用

完整 IPC 热更新流程、tensor_bucket 扁平化、`wait_weights_before_ready` 启动握手见 **[[32-CheckpointEngine-00-MOC|32-CheckpointEngine]]**；ModelLoader 侧 weight sync API 见 **12-ModelLoader §weight_sync**。
