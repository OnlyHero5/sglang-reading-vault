---
title: "可观测性 · 排障指南"
type: troubleshooting
framework: sglang
topic: "可观测性"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# 可观测性 · 排障指南

## 读者任务

这篇按症状排障，不按文件顺序讲源码。每个问题都给出三件事：先判断它属于哪本账，再给源码入口，最后给验证动作。

如果只记一条规则：Prometheus 无数据先分清是 scrape 入口没挂、collector 没创建、rank 没写、请求没完成，还是你查错了旁路。

## 快速症状表

| 症状 | 优先归类 | 第一入口 | 直接验证 |
|------|----------|----------|----------|
| `/metrics` 404 | scrape 入口账 | `add_prometheus_middleware` / gRPC sidecar | 服务模式、开关与实际暴露端口 |
| `/metrics` 有输出但缺少 Scheduler gauge | Scheduler 状态账 | `SchedulerMetricsCollector.init_new` | 当前 rank 是否 `attn_tp_rank == 0` 或 all-scheduler |
| `cache_hit_rate` 长期为 0 | Scheduler 状态账 | `MetricsReporter` cache hit 计算 | workload 是否有重复前缀，stats tick 是否执行 |
| `priority="None"` 出现 | Scheduler 队列标签 | `QueueCount.from_reqs` | priority scheduling 是否设置默认 priority |
| custom label 不出现 | Tokenizer 请求账 | `extract_custom_labels`、collector labels | allowed labels 和请求 header 是否匹配 |
| TTFT 有、ITL 或 E2E 缺失 | Tokenizer 请求账 | `collect_metrics` | 请求是否继续生成或已 finished |
| HTTP status counter 有，KV 指标没有 | HTTP middleware 与 Scheduler 分离 | `track_http_status_code` | 不要把 HTTP 层当引擎层 |
| request.finished 没有 | RequestLogger 旁路 | `log_finished_request` | `--log-requests`、level、慢请求过滤 |
| trace 没有 span | trace 旁路 | `process_tracing_init` | OpenTelemetry 包和 OTLP endpoint |
| exporter 文件没写 | exporter 旁路 | `RequestMetricsExporterManager` | file exporter 是否启用，目录是否可写 |
| 热更新 duration 缺失 | weight update edge event | `observe_weight_load` | update 路径是否经过对应 source |

## Q1：`/metrics` 404 或空，先查什么？

先不要查 `SchedulerStats`。`/metrics` 是 HTTP-facing scrape 面，开关是 `server_args.enable_metrics`：HTTP 模式挂在主 FastAPI，gRPC 模式挂在独立 aiohttp sidecar。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L273-L276
    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()
```

gRPC sidecar 端口由独立参数决定，未配置时使用主 gRPC 端口加一。

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L170-L175
    sidecar_port = (
        server_args.grpc_http_sidecar_port
        if server_args.grpc_http_sidecar_port is not None
        else server_args.port + 1
    )
```

如果路由存在但内容异常，再查 multiprocess 目录是否在 import `prometheus_client` 前设置。SGLang 的挂载函数会创建 `CollectorRegistry` 并接入 `MultiProcessCollector`。

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

验证动作：

- 启动参数里确认 `--enable-metrics`。
- HTTP 模式访问主服务端口；gRPC 模式访问 `grpc_http_sidecar_port`，未配置时默认是 `port + 1`。
- `curl http://host:port/metrics`，确认不是 404、307 或 sidecar 连接失败。
- 查启动日志里的 `PROMETHEUS_MULTIPROC_DIR`，确认目录存在且 worker 可写。
- 若重启后出现陈旧 series，检查目录内文件与现存 PID；当前仓库没有显式 `mark_process_dead` 调用，不能仅凭 200 响应排除生命周期问题。

## Q2：为什么只有部分 Scheduler 指标有数据？

默认策略是只让 `attn_tp_rank == 0` 写 Scheduler aggregate stats，避免 TP 副本重复上报。需要每个 scheduler rank 都写时，才打开 `enable_metrics_for_all_schedulers`。

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1040-L1044
        enable_metrics = server_args.enable_metrics
        is_stats_logging_rank = ps.attn_tp_rank == 0
        current_scheduler_metrics_enabled = enable_metrics and (
            is_stats_logging_rank or server_args.enable_metrics_for_all_schedulers
        )
```

验证动作：

- 看 series label 里的 `tp_rank`、`pp_rank`、`dp_rank`。
- 如果是 dp_attention 或多 scheduler 排障，临时打开 all-scheduler，并在 Grafana 按 rank 过滤。
- 生产默认不建议打开全量 rank，因为 series 会按 rank 展开；普通 TP 下盲目求和还可能把副本状态放大，DP-Attention 下则可能需要保留 rank 维度。

## Q3：`cache_hit_rate` 怎么解读？

`cache_hit_rate` 不是单请求字段。它由 prefill stats tick 里的 hit tokens 和 input tokens 计算，再写入 `SchedulerStats`。

```python
# 来源：python/sglang/srt/managers/scheduler_components/metrics_reporter.py L606-L617
            priority_enabled = self.scheduler.enable_priority_scheduling
            effective_input_tokens = (
                prefill_stats.log_input_tokens
                - prefill_stats.reprocessed_log_input_tokens
            )
            effective_hit_tokens = (
                prefill_stats.log_hit_tokens - prefill_stats.reprocessed_log_hit_tokens
            )
            total_tokens = effective_input_tokens + effective_hit_tokens
            cache_hit_rate = (
                effective_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )
```

验证动作：

- workload 没有共享前缀时，长期低命中是合理现象。
- 热更新或 flush cache 后骤降也是合理现象。
- 如果完全没有 series，回到 Q2 查 rank 和 collector；如果 series 有但值低，再查 Radix cache workload。

## Q4：为什么出现 `priority="None"`？

Scheduler 队列指标用 `QueueCount` 记录 total 和 priority breakdown。源码注释说明，如果开启 priority scheduling 但请求 priority 是 `None`，Prometheus label 会出现字符串形式的 None。

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L44-L61
@dataclass
class QueueCount:
    """Holds both the total count and optional per-priority breakdown for a queue."""

    total: int = 0
    by_priority: Optional[Dict[int, int]] = None

    @classmethod
    def from_reqs(cls, reqs: List[Req], enable_priority_scheduling: bool = False):
        # NOTE: If requests have priority=None (no --default-priority-value set),
        # Counter will produce {None: N}, resulting in priority="None" Prometheus labels.
        # Set --default-priority-value when enabling priority scheduling to avoid this.
        by_priority = (
            dict(Counter(req.priority for req in reqs))
            if enable_priority_scheduling
            else None
        )
        return cls(total=len(reqs), by_priority=by_priority)
```

验证动作：

- 开启 priority scheduling 时配置默认 priority。
- Grafana 面板区分 total 队列和 per-priority 队列。
- 不要把 `priority=""`、具体 priority 和 `priority="None"` 混在同一条查询里。

## Q5：custom labels 为什么没出现在 Tokenizer metrics？

custom label 有两道门。第一道在 OpenAI serving 层：header 必须能解析成 dict，且 key 在 allowed labels 中。

```python
# 来源：python/sglang/srt/entrypoints/openai/serving_base.py L264-L270
        if isinstance(raw_labels, dict):
            custom_labels = {
                label: value
                for label, value in raw_labels.items()
                if label in self.allowed_custom_labels
            }
        return custom_labels
```

第二道在 Tokenizer collector 初始化：allowed labels 会先以空值加入 label schema。Prometheus client 的 labelnames 是初始化时固定的，后续请求只能填已有 label。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L538-L547
            if self.enable_priority_scheduling:
                labels["priority"] = ""
            if self.server_args.tokenizer_metrics_allowed_custom_labels:
                for label in self.server_args.tokenizer_metrics_allowed_custom_labels:
                    labels[label] = ""
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            tokenizer_collector_cls = resolve_collector_class(
                self.server_args,
                STAT_LOGGER_ROLE_TOKENIZER,
```

验证动作：

- 启动时配置 `--tokenizer-metrics-allowed-custom-labels tenant`。
- 请求 header 使用 `x-custom-labels`，除非你改了 `tokenizer_metrics_custom_labels_header`。
- header value 必须是 JSON object，且 key 在 allowed list 中。
- 只在 Tokenizer request metrics 里期待这些 label，不要在 Scheduler gauge 里找。

## Q6：TTFT、ITL、E2E 为什么不同时出现？

它们写入时机不同。第一次可观测输出写 TTFT；后续处理事件若累计 `completion_tokens` 增加，就用“事件间隔 / 新增 token 数”写一组 ITL 样本；请求 finished 才写 E2E 和 token counters。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L2411-L2431
        if (
            not state.ttft_observed
            and self.disaggregation_mode != DisaggregationMode.PREFILL
        ):
            state.ttft_observed = True
            state.last_completion_tokens = completion_tokens
            self.metrics_collector.observe_time_to_first_token(
                labels, state.time_stats.get_first_token_latency()
            )
        else:
            num_new_tokens = completion_tokens - state.last_completion_tokens
            if num_new_tokens:
                self.metrics_collector.observe_inter_token_latency(
                    labels,
                    state.time_stats.get_interval(),
                    num_new_tokens,
                )
                state.time_stats.set_last_time()
                state.last_completion_tokens = completion_tokens

        if state.finished:
```

验证动作：

- 流式请求正在生成时，可能先看到 TTFT 和 ITL，稍后才看到 E2E。
- PREFILL disaggregation 模式下 TTFT 分支被跳过，不能用普通 decode-only 直觉解读。
- ITL 依赖 completion tokens 增量，没有新增 token 的更新不会写。
- 一次更新新增多个 token 时，它们共享同一个平均 ITL 样本值；因此该 histogram 不是逐 token 时间戳，也不是网络 chunk latency。

## Q7：HTTP metrics、Gateway metrics 和 SRT metrics 怎么分？

SRT 进程内 HTTP middleware 只看 endpoint、method、status code、active request 和 routing key。它和 Scheduler 内部指标不是一个层级。

```python
# 来源：python/sglang/srt/utils/common.py L1662-L1678
        path, is_handled_path = _get_fastapi_request_path(request)
        method = request.method
        routing_key = request.headers.get("x-smg-routing-key")

        http_request_counter.labels(endpoint=path, method=method).inc()
        http_requests_active.labels(endpoint=path, method=method).inc()
        if routing_key:
            routing_keys_active.inc(routing_key)

        try:
            response = await call_next(request)

            http_response_counter.labels(
                endpoint=path,
                method=method,
                status_code=str(response.status_code),
            ).inc()
```

排障分工：

| 问题 | 优先看 |
|------|--------|
| HTTP 404、5xx、入口活跃请求 | HTTP middleware 或 Gateway |
| backend selection、限流、路由健康 | Gateway |
| KV pool、cache hit、retract、spec verify | SRT Scheduler metrics |
| TTFT、ITL、E2E、token counters | SRT Tokenizer metrics |

## Q8：RequestLogger 为什么没写日志？

RequestLogger 与 Prometheus 独立。它先看 `log_requests`，finish 日志还会受慢请求阈值过滤。

```python
# 来源：python/sglang/srt/utils/request_logger.py L159-L173
    def log_finished_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        out: Any,
        request: Optional[fastapi.Request] = None,
    ) -> None:
        if not self.log_requests:
            return

        e2e_latency_ms = out["meta_info"].get("e2e_latency", 0) * 1000
        if self.log_exceeded_ms > 0 and e2e_latency_ms < self.log_exceeded_ms:
            return

        max_length, skip_names, out_skip_names = self.metadata
        headers = _extract_whitelisted_headers(request)
```

log level 决定字段保留范围。level 0 会跳过 text、input ids、多模态、LoRA path 和 sampling params；level 1 仍跳过输入文本和多模态，但保留 sampling params；level 2 会截断到 2048；level 3 使用大上限。

```python
# 来源：python/sglang/srt/utils/request_logger.py L199-L230
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "mm_data_mooncake",
                    "lora_path",
                    "sampling_params",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = {
                    "text",
                    "input_ids",
                    "input_embeds",
                    "image_data",
                    "audio_data",
                    "video_data",
                    "mm_data_mooncake",
                    "lora_path",
                }
                out_skip_names = {"text", "output_ids", "embedding"}
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
```

验证动作：

- 确认 `--log-requests` 已开启。
- 检查 `SGLANG_LOG_REQUEST_EXCEEDED_MS` 是否过滤了快请求。
- 高 QPS 下优先用低 level，避免把 payload 日志当 metrics 后端。

## Q9：trace 开了但没有 span，先看什么？

trace 不是 Prometheus。它依赖 OpenTelemetry 包、OTLP endpoint 和 trace module 过滤。初始化失败会直接报错。

```python
# 来源：python/sglang/srt/observability/trace.py L177-L214
    if not opentelemetry_imported:
        opentelemetry_initialized = False
        raise RuntimeError(
            "opentelemetry package is not installed!!! Please not enable tracing or install opentelemetry"
        )

    try:
        resource = Resource.create(
            attributes={
                SERVICE_NAME: server_name,
            }
        )
        tracer_provider = TracerProvider(
            resource=resource, id_generator=TraceCustomIdGenerator()
        )

        schedule_delay_millis = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", 500
        )
        max_export_batch_size = get_int_env_var(
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", 64
        )

        processor = BatchSpanProcessor(
            span_exporter=get_otlp_span_exporter(otlp_endpoint),
            schedule_delay_millis=schedule_delay_millis,
            max_export_batch_size=max_export_batch_size,
        )
        tracer_provider.add_span_processor(processor)
        trace.set_tracer_provider(tracer_provider)
    except Exception as e:
        opentelemetry_initialized = False
        raise RuntimeError(
            f"initialize opentelemetry error:{e}. Please set correct otlp endpoint."
        )

    opentelemetry_initialized = True
    tracer = trace.get_tracer("sglang server")
```

验证动作：

- 确认 OpenTelemetry 相关包已安装。
- 确认 OTLP endpoint 协议和地址正确。
- 如果使用外部 trace header，只支持 `traceparent` 和 `tracestate`。
- 如果设置了 trace modules，确认 request 模块没有被过滤掉。

## Q10：RequestMetricsExporter 文件为什么没写？

exporter 只在配置启用时创建。finish 路径发现 manager 有 exporter，才异步写 record。

```python
# 来源：python/sglang/srt/observability/request_metrics_exporter.py L197-L220
    def exporter_enabled(self) -> bool:
        """Return true if at least one RequestMetricsExporter is enabled."""
        return len(self._exporters) > 0

    async def write_record(self, obj, out_dict: dict) -> None:
        """Write a record using all configured exporters."""
        for exporter in self._exporters:
            await exporter.write_record(obj, out_dict)


def create_request_metrics_exporters(
    server_args: ServerArgs,
    obj_skip_names: Optional[set[str]] = None,
    out_skip_names: Optional[set[str]] = None,
) -> List[RequestMetricsExporter]:
    """Create and configure `RequestMetricsExporter`s based on server args."""
    metrics_exporters = []

    if server_args.export_metrics_to_file:
        metrics_exporters.append(
            FileRequestMetricsExporter(server_args, obj_skip_names, out_skip_names)
        )

    return metrics_exporters
```

验证动作：

- 启动时启用 file exporter。
- 确认 `export_metrics_to_file_dir` 目录存在或可创建。
- 确认请求真的进入 finished 分支；未完成请求不会写完整 record。
- 健康检查请求可能被 exporter 逻辑过滤，排障时用真实 generate 请求验证。

## Q11：热更新期间指标怎么读？

热更新 duration 不是普通 stats tick 写入。因为 engine 更新时可能 pause，源码把 `observe_weight_load` 做成边沿触发，在更新结束处直接写。

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

IPC 路径的 source 来自 `update_weights_from_ipc`。

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L166-L169
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
```

验证动作：

- 根据更新方式过滤 `source="disk"`、`source="distributed"`、`source="tensor"` 或 `source="ipc"`。
- 同时观察 HTTP/控制面结果和真实请求；当前基线未找到 `num_paused_reqs` 的递增生产者，不要用它证明 IPC 热更新暂停了多少请求。
- 更新后 cache hit 下降可能是 cache flush 的结果，和 prefix cache workload 语义有关。

## Q12：`enable_metrics_for_all_schedulers` 该不该长期打开？

通常不该长期打开。它解决的是 per-rank 观测问题，不是默认生产指标问题。默认 `attn_tp_rank == 0` 写 stats，是为了避免普通 TP 副本重复上报并控制 series 数量。

适合打开的场景：

- dp_attention 排障，TP0 不能代表所有 scheduler。
- 某个 rank 的 queue 或 KV usage 异常，需要临时对比。
- 自定义 Grafana 面板已经按拓扑理解 rank label：普通 TP 副本通常过滤或去重，DP-Attention 的不同 scheduler 则可能需要分组保留。

不适合打开的场景：

- 只需要全局服务 SLA。
- Prometheus cardinality 已经紧张。
- Grafana 查询没有 rank 过滤，容易把普通 TP 副本状态求和成多倍；反过来，粗暴丢掉 DP-Attention rank 又会掩盖局部异常。

## 复盘

- 先把现象归类到 scrape、Scheduler、Tokenizer、HTTP middleware、日志、trace、exporter 或 weight update。
- 每类问题都有不同的触发时机：启动、stats tick、首次输出、累计 token 增量、finish、update 完成。
- label 问题要区分 collector 初始化时的 label schema 和请求执行时的 label value。
- Prometheus 适合 aggregate SLA；RequestLogger 和 exporter 适合单请求复盘。
- all-scheduler metrics 是排障工具，不是默认生产姿势。

下一篇 [[SGLang-可观测性-学习检查]] 用可执行清单检验是否真正读懂。
