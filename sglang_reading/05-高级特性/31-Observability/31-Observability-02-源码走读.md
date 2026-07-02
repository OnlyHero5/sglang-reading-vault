---
type: batch-doc
module: 31-Observability
batch: "31"
doc_type: walkthrough
title: "可观测性 · 源码走读"
tags:
 - sglang/batch/31
 - sglang/module/observability
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# 可观测性 · 源码走读

## 走读顺序

1. `server_args.py` — enable_metrics 开关
2. `metrics_collector.py` — SchedulerMetricsCollector 初始化与 log_stats
3. `scheduler.py` — init_metrics_collector
4. `metrics_reporter.py` — 周期 log_stats
5. `common.py` — add_prometheus_middleware
6. `request_logger.py` — 请求日志
7. `req_time_stats.py` — stage latency

---

## 1. SchedulerMetricsCollector.init_new

**Explain：** Scheduler 构造时调用 `init_new`：根据 server_args 决定是否创建 collector、是否启用 kv_cache_events、以及 Prometheus label（model_name、engine_type、tp_rank 等）。`resolve_collector_class` 支持 Ray Serve 等 embedded 场景替换 collector 实现。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1027-L1085
    @classmethod
    def init_new(
        cls,
        *,
        server_args: ServerArgs,
        ps: Any,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        enable_priority_scheduling: bool,
        enable_lora: bool,
        enable_hierarchical_cache: bool,
    ) -> SchedulerMetricsCollectorContext:
        enable_metrics = server_args.enable_metrics
        is_stats_logging_rank = ps.attn_tp_rank == 0
        current_scheduler_metrics_enabled = enable_metrics and (
            is_stats_logging_rank or server_args.enable_metrics_for_all_schedulers
        )
        enable_kv_cache_events = bool(
            server_args.kv_events_config
            and ps.pp_rank == 0
            and ps.attn_tp_rank == 0
            and ps.attn_cp_rank == 0
        )
        collector: Optional[SchedulerMetricsCollector] = None
        if enable_metrics:
            engine_type = DisaggregationMode.to_engine_type(
                server_args.disaggregation_mode
            )
            labels = {
                "model_name": server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "moe_ep_rank": ps.moe_ep_rank,
            }
            if enable_priority_scheduling:
                labels["priority"] = ""
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            if server_args.extra_metric_labels:
                labels.update(server_args.extra_metric_labels)
            scheduler_collector_cls = resolve_collector_class(
                server_args, STAT_LOGGER_ROLE_SCHEDULER, cls
            )
            collector = scheduler_collector_cls(
                labels=labels,
                enable_lora=enable_lora,
                enable_hierarchical_cache=enable_hierarchical_cache,
                enable_streaming_session=server_args.enable_streaming_session,
                server_args=server_args,
            )
        return SchedulerMetricsCollectorContext(
            enable_metrics=enable_metrics,
            is_stats_logging_rank=is_stats_logging_rank,
            current_scheduler_metrics_enabled=current_scheduler_metrics_enabled,
            enable_kv_cache_events=enable_kv_cache_events,
            collector=collector,
        )
```

**Comment：**

- `DisaggregationMode.to_engine_type` 写入 engine_type label。
- collector=None 时 metrics 代码路径短路。
- `enable_kv_cache_events` 仅在 pp/tp/cp rank 0 开启。

---

## 2. Scheduler.init_metrics_collector

**Explain：** Scheduler 在 parallel state 就绪后调用；将 context 存于 `metrics_collector_context`，collector 引用供 MetricsReporter 使用。与 `init_ipc_channels` 中 metrics_enabled 判断联动。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L590-L603
    def init_metrics_collector(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector
```

**Comment：**

- LoRA / HiCache 开关传入 collector 以条件创建子 gauge。
- DP 多副本时各 dp_rank 可有独立 label。

---

## 3. log_stats — Gauge 批量写入

**Explain：** `log_stats` 将 SchedulerStats 各字段映射到预创建的 Gauge/Counter；memory pool、PD queue、spec、LoRA pool 等分区写入，是 `/metrics` scrape 时看到的主要 Scheduler 侧数据。每 log interval 调用一次。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1260-L1295
    def log_stats(self, stats: SchedulerStats) -> None:
        # Basics
        self._log_gauge_queue_count(self.num_running_reqs, stats.num_running_reqs)
        self._log_gauge_queue_count(self.num_queue_reqs, stats.num_queue_reqs)
        self._log_gauge(self.num_grammar_queue_reqs, stats.num_grammar_queue_reqs)
        self._log_gauge(self.gen_throughput, stats.gen_throughput)
        self._log_gauge(self.cache_hit_rate, stats.cache_hit_rate)
        self._log_gauge(self.decode_sum_seq_lens, stats.decode_sum_seq_lens)

        # Memory pool usage ratios
        self._log_gauge(self.token_usage, stats.token_usage)
        self._log_gauge(self.full_token_usage, stats.full_token_usage)
        self._log_gauge(self.swa_token_usage, stats.swa_token_usage)
        self._log_gauge(self.mamba_usage, stats.mamba_usage)

        # Absolute token counts
        self._log_gauge(self.num_used_tokens, stats.num_used_tokens)
        self._log_gauge(self.kv_available_tokens, stats.kv_available_tokens)
        self._log_gauge(self.kv_evictable_tokens, stats.kv_evictable_tokens)
        self._log_gauge(self.kv_used_tokens, stats.kv_used_tokens)
        self._log_gauge(self.swa_available_tokens, stats.swa_available_tokens)
        self._log_gauge(self.swa_evictable_tokens, stats.swa_evictable_tokens)
        self._log_gauge(self.swa_used_tokens, stats.swa_used_tokens)
        self._log_gauge(self.mamba_available_tokens, stats.mamba_available_tokens)
        self._log_gauge(self.mamba_evictable_tokens, stats.mamba_evictable_tokens)
        self._log_gauge(self.mamba_used_tokens, stats.mamba_used_tokens)

        # Speculative decoding
        self._log_gauge(self.spec_accept_length, stats.spec_accept_length)
        self._log_gauge(self.spec_accept_rate, stats.spec_accept_rate)
        self._log_gauge(self.spec_num_steps, stats.spec_num_steps)
        self._log_gauge(self.spec_num_draft_tokens, stats.spec_num_draft_tokens)

        # Retract
        self._log_gauge(self.num_retracted_reqs, stats.num_retracted_reqs)
        self._log_gauge(self.num_paused_reqs, stats.num_paused_reqs)
```

**Comment：**

- routing_key 使用 GaugeHistogram 自定义类型。
- `last_log_time` 供 reporter 计算 gen_throughput。
- weight_load 在 update_weights 路径 inline observe（非 log_stats）。

---

## 4. observe_weight_load — 热更新 metrics

**Explain：** 权重更新期间 engine 暂停，`log_stats` tick 可能不触发；因此在 `update_weights_from_*` 完成时 inline 写入 `weight_load_duration_seconds`。`source` label 区分 disk/distributed/tensor/ipc 四种路径。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1143-L1149
    def observe_weight_load(self, duration_seconds: float, source: str) -> None:
        # Edge-triggered: engine is paused during the update, so log_stats
        # won't fire — write the gauge inline at end of update_weights_from_*.
        # `source` is "disk" | "distributed" | "tensor" | "ipc".
        self.weight_load_duration_seconds.labels(**self.labels, source=source).set(
            duration_seconds
        )
```

**Comment：**

- checkpoint-engine IPC 路径 source=ipc，见 [[32-CheckpointEngine-00-MOC]]。
- 与 `num_paused_reqs` 联合监控热更新影响。

---

## 5. MetricsReporter 触发点

**Explain：** `MetricsReporter` 在 Scheduler event loop 的 stats tick 调用 `metrics_collector.log_stats`；同时 increment realtime_tokens、forward_execution_seconds 等 counter。decode 路径每 iteration 还有轻量 realtime token 计数。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler_components/metrics_reporter.py L654-L660
            # Utilization / LoRA / HiCache
            self._calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self._update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self.scheduler.kv_events_publisher.emit_kv_metrics()
```

**Comment：**

- 与 `SchedulerStats` dataclass 字段一一对应。
- grammar stats 单独走 log_grammar_stats。
- `report_decode_stats` 每 decode iteration increment realtime tokens。

---

## 6. add_prometheus_middleware

**Explain：** http_server 启动 lifespan 中调用；创建 CollectorRegistry + MultiProcessCollector，Mount 到 `/metrics`。必须在 `set_prometheus_multiproc_dir` 之后 import prometheus_client。

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

- path_regex workaround 避免 307 redirect。
- gRPC sidecar 复用同一函数。

---

## 7. resolve_collector_class — DI 扩展

**Explain：** Embedded 场景（Ray Serve LLM）通过 ServerArgs.stat_loggers 替换五类 collector；role 键见 STAT_LOGGER_ROLE_* 常量。五处实例化点均调用此函数。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L199-L210
def resolve_collector_class(
    server_args: Optional[ServerArgs], role: str, default_cls: type
) -> type:
    """Return the subclass registered for `role` on `server_args.stat_loggers`,
    or `default_cls` if none is registered. Tolerates `server_args=None` and
    `stat_loggers=None`."""
    if server_args is None:
        return default_cls
    stat_loggers = getattr(server_args, "stat_loggers", None)
    if not stat_loggers:
        return default_cls
    return stat_loggers.get(role, default_cls)
```

**Comment：**

- 五 role：scheduler / tokenizer / storage / radix_cache / expert_dispatch。
- None server_args 时回退 default_cls。

---

## 8. emit_constants — 启动一次性 gauge

**Explain：** 模型加载完成后写入 max_total_num_tokens、page_size、engine_startup_time 等常量型 gauge；不会每 tick 更新。用于容量规划 dashboard。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1385-L1408
    def emit_constants(
        self,
        max_total_num_tokens: int,
        max_running_requests_under_SLO: Optional[int],
        engine_startup_time: float,
        engine_load_weights_time: float,
        page_size: int,
        num_pages: int,
        context_len: int,
        startup_available_gpu_memory_gb: float,
    ) -> None:
        self._log_gauge(self.max_total_num_tokens, max_total_num_tokens)
        if max_running_requests_under_SLO is not None:
            self._log_gauge(
                self.max_running_requests_under_SLO, max_running_requests_under_SLO
            )
        self._log_gauge(self.engine_startup_time, engine_startup_time)
        self._log_gauge(self.engine_load_weights_time, engine_load_weights_time)
        self._log_gauge(self.page_size, page_size)
        self._log_gauge(self.num_pages, num_pages)
        self._log_gauge(self.context_len, context_len)
        self._log_gauge(
            self.startup_available_gpu_memory_gb, startup_available_gpu_memory_gb
        )
```

**Comment：**

- 与冷启动 load 路径在 model_runner 完成后调用。
- `initial_weights_loaded` 与 wait_weights 见 CheckpointEngine。

---

## 9. RequestLogger.log_received_request

**Explain：** TokenizerManager 收到 GenerateReqInput 后调用；根据 log_requests_level 决定字段截断与是否 decode input_ids。JSON format 输出 structured log 事件 `request.received`。

**Code：**

```python
# 来源：python/sglang/srt/utils/request_logger.py L88-L111
    def log_received_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        tokenizer: Any = None,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        if not self.log_requests:
            return

        max_length, skip_names, _ = self.metadata
        headers = _extract_whitelisted_headers(request)
        if self.log_requests_format == "json":
            log_data = {
                "rid": obj.rid,
                "obj": _transform_data_for_logging(obj, max_length, skip_names),
            }
            if headers:
                log_data["headers"] = headers
            log_json(self.targets, "request.received", log_data)
        else:
            headers_str = f", headers={headers}" if headers else ""
            self._log(
                f"Receive: obj={_dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}{headers_str}"
            )
```

**Comment：**

- level≥3 记录 finish 时完整 output。
- schedule_simulator 可 load_from_request_logger 回放。

---

## 10. ReqTimeStats — stage latency

**Explain：** 单请求生命周期各 stage（queue、prefill、decode、disagg bootstrap 等）的 latency 追踪；finish 时 observe 到 per_stage_req_latency_seconds histogram。enable_metrics 与 tracing 可独立开启。

**Code：**

```python
# 来源：python/sglang/srt/observability/req_time_stats.py L260-L271
    def set_metrics_collector(
        self, collector: Union[SchedulerMetricsCollector, TokenizerMetricsCollector]
    ):
        if collector:
            self.enable_metrics = True
            self.metrics_collector = collector

    def observe_per_stage_req_latency(self, stage: RequestStageConfig, latency: float):
        if self.enable_metrics and stage.metrics_is_observed:
            self.metrics_collector.observe_per_stage_req_latency(
                stage.stage_name, latency
            )
```

**Comment：**

- stage 枚举与 PD bootstrap/transfer 对齐。
- bootstrap_done_time 用于 PD transfer latency。

---

## 11. enable_metrics 短路路径

**Explain：** `enable_metrics=False` 时 `init_new` 返回 collector=None；MetricsReporter 检查 `current_scheduler_metrics_enabled` 后跳过所有 Prometheus 调用，热路径零开销。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1051-L1052
        collector: Optional[SchedulerMetricsCollector] = None
        if enable_metrics:
```

**Comment：**

- RequestLogger 独立开关，不受 enable_metrics 影响。
- func_timer 仅在 enable_metrics 时开启（http_server）。

---

## 12. weight_updater 与 metrics 联动

**Explain：** Scheduler `weight_updater.update_weights_from_ipc` 使用 `_observe_weight_load("ipc")` context manager，在 IPC 热更新完成时记录 duration 并 flush cache。与 [[32-CheckpointEngine-00-MOC]] 直接相关。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L166-L178
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)
```

**Comment：**

- pause/resume 与 num_paused_reqs metrics 联动。
- 详见 12-ModelLoader §weight_sync。
