---
title: "可观测性 · 核心概念"
type: concept
framework: sglang
topic: "可观测性"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-12
---
# 可观测性 · 核心概念

## 读者任务

这篇先建立模型：SGLang 的可观测性不是一个中心化收集器，而是多个进程在不同时间点写不同账本。读完后，你应该能把一个现象归类到 scrape 入口、Scheduler aggregate stats、Tokenizer request metrics、HTTP response middleware、RequestLogger、trace 或 exporter。

## 心理模型：四本账

| 账本 | 写入者 | 触发时机 | 解决什么问题 |
|------|--------|----------|--------------|
| scrape 入口账 | HTTP worker | lifespan / setup | Prometheus 能不能抓到 `/metrics` |
| Scheduler 状态账 | Scheduler | prefill/decode stats tick | 队列、KV pool、cache hit、spec、LoRA、PD、HiCache |
| Tokenizer 请求账 | TokenizerManager | 首次可观测输出、累计 completion token 增量、请求完成 | TTFT、近似 per-token ITL、E2E、prompt/generation tokens、cached tokens |
| 旁路账 | RequestLogger / trace / exporter | request received、stage 切片、finish | 单请求复盘、OpenTelemetry、离线请求记录 |

读 Observability 的核心，是不要把这四本账混在一起。`cache_hit_rate` 来自 Scheduler 状态账；TTFT 来自 Tokenizer 请求账；HTTP 2xx/5xx 计数来自 response middleware；请求原文只在 RequestLogger 或 exporter 里。

## 概念 1：开关决定是否创建 Prometheus 路径

`ServerArgs` 里 metrics 默认关闭。这里还定义了 gRPC sidecar 端口、MFU 估算、all-scheduler 上报、Tokenizer custom labels 和 histogram buckets。

```python
# 来源：python/sglang/srt/server_args.py L1070-L1081
    enable_metrics: A[bool, "Enable log prometheus metrics."] = False
    grpc_http_sidecar_port: A[
        Optional[int],
        "Port for the HTTP sidecar server in gRPC mode (--grpc-mode). Serves Prometheus metrics and profiling endpoints. Defaults to --port + 1. Not used in HTTP mode.",
    ] = None
    enable_mfu_metrics: A[bool, "Enable estimated MFU-related prometheus metrics."] = (
        False
    )
    enable_metrics_for_all_schedulers: A[
        bool,
        "Enable --enable-metrics-for-all-schedulers when you want schedulers on all TP ranks (not just TP 0) to record request metrics separately. This is especially useful when dp_attention is enabled, as otherwise all metrics appear to come from TP 0.",
    ] = False
```

custom label 配置只影响 Tokenizer 请求账。允许的 label 会先加入 collector 的 label schema，后续每次请求级 metrics 事件只能给这些既有 label 填具体值。

```python
# 来源：python/sglang/srt/server_args.py L1086-L1100
    tokenizer_metrics_custom_labels_header: A[
        str,
        "Specify the HTTP header for passing custom labels for tokenizer metrics.",
    ] = "x-custom-labels"
    tokenizer_metrics_allowed_custom_labels: A[
        Optional[List[str]],
        "The custom labels allowed for tokenizer metrics. The labels are specified via a dict in '--tokenizer-metrics-custom-labels-header' field in HTTP requests, e.g., {'label1': 'value1', 'label2': 'value2'} is allowed if '--tokenizer-metrics-allowed-custom-labels label1 label2' is set.",
    ] = None
    extra_metric_labels: A[
        Optional[Dict[str, str]],
        Arg(
            help='The custom labels for metrics. e.g. \'{"label1": "value1", "label2": "value2"}\'',
            type_parser=json.loads,
        ),
    ] = None
```

如果你看到 `/metrics` 404，先看 `enable_metrics`。如果你看到 label 没有出现，先看 label schema 是否在启动时被允许。

## 概念 2：Prometheus multiprocess 目录必须先设置

SGLang 使用 Prometheus multiprocess mode。源码注释强调必须在 import `prometheus_client` 前设置 `PROMETHEUS_MULTIPROC_DIR`。

```python
# 来源：python/sglang/srt/utils/common.py L1571-L1586
def set_prometheus_multiproc_dir():
    # Set prometheus multiprocess directory
    # sglang uses prometheus multiprocess mode
    # we need to set this before importing prometheus_client
    # https://prometheus.github.io/client_python/multiprocess/
    global prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        logger.debug("User set PROMETHEUS_MULTIPROC_DIR detected.")
        prometheus_multiproc_dir = tempfile.TemporaryDirectory(
            dir=os.environ["PROMETHEUS_MULTIPROC_DIR"]
        )
    else:
        prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = prometheus_multiproc_dir.name
    logger.debug(f"PROMETHEUS_MULTIPROC_DIR: {os.environ['PROMETHEUS_MULTIPROC_DIR']}")
```

所以 Observability 的第一层不在 collector 里，而在进程启动环境里。目录错、目录不可写、import 顺序错，都会让后面的指标写入或聚合失效。

这里还要区分“源码已经做了什么”和“不要擅自推断什么”。环境变量不存在时，函数创建临时目录并把它写回 `PROMETHEUS_MULTIPROC_DIR`；环境变量已存在时，它会在该路径下创建一个受 `TemporaryDirectory` 管理的子目录，但环境变量仍指向调用者给出的路径。当前仓库也没有显式的 `multiprocess.mark_process_dead` 调用。因此不能仅凭这段代码断言外部目录里的历史文件一定会被清空；看到进程重启后残留 series 时，应直接核对实际环境变量、目录内容和进程拓扑。

HTTP 与 gRPC 的暴露面也不同。HTTP 模式把 ASGI app mount 到主 FastAPI；gRPC 模式另起 aiohttp sidecar，在 handler 内新建 registry 并聚合同一 multiprocess 目录。

```python
# 来源：python/sglang/srt/entrypoints/grpc_server.py L40-L55
def _add_metrics_routes(app):
    """Add Prometheus /metrics endpoint to the aiohttp app."""
    from prometheus_client import (
        CollectorRegistry,
        multiprocess,
    )
    from prometheus_client.openmetrics.exposition import (
        CONTENT_TYPE_LATEST,
        generate_latest,
    )

    async def metrics_handler(request):
        try:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
```

因此“主端口 `/metrics` 404”在 HTTP 模式通常先查挂载开关；在 gRPC 模式还要查 sidecar 端口、sidecar 是否成功启动，以及 `smg-grpc-servicer` 是否支持 ready hook。

## 概念 3：Scheduler 只让特定 rank 写 aggregate stats

Scheduler metrics 初始化会计算四个事实：metrics 是否开启、当前 rank 是否是 stats logging rank、当前 scheduler 是否允许写 metrics、是否启用 KV cache events。

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1028-L1085
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

这个设计避免普通 TP 副本默认重复写同一类状态。打开 `enable_metrics_for_all_schedulers` 后，每个 scheduler 会以 `tp_rank`、`pp_rank`、可选 `dp_rank` 等标签形成独立 series。普通 TP 下这些状态可能是副本视角，直接 `sum` 会放大；DP-Attention 下不同 scheduler 又可能代表不同请求所有权。Grafana 必须先按拓扑决定过滤、分组还是聚合，不能只说“cardinality 变大”。

## 概念 4：`SchedulerStats` 是 aggregate 快照

Scheduler 状态账不是单请求日志。它把队列长度、cache hit、KV pool、spec、retract、PD、LoRA、HiCache 等字段集中到一个 dataclass，再由 collector 写成 gauge。

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L64-L100
@dataclass
class SchedulerStats:
    # Basics
    num_running_reqs: QueueCount = field(default_factory=QueueCount)
    num_queue_reqs: QueueCount = field(default_factory=QueueCount)
    num_grammar_queue_reqs: int = 0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    decode_sum_seq_lens: int = 0

    # Memory pool usage ratios (0.0–1.0).
    # Each pool tracks: used = total - available - evictable, usage = used / total.
    #
    # token_usage:      max(full, swa, mamba) — the bottleneck across all pools.
    #                   FIXME: misleadingly named "token_usage"; rename requires API deprecation.
    # full_token_usage: full-attention KV cache pool usage (always active).
    # swa_token_usage:  sliding-window attention KV cache pool usage (hybrid SWA models only, e.g. Gemma2).
    # mamba_usage:      Mamba SSM state pool usage (hybrid SSM models only, e.g. Jamba).
    token_usage: float = 0.0
    full_token_usage: float = 0.0
    swa_token_usage: float = 0.0
    mamba_usage: float = 0.0

    # Absolute token counts for the full-attention KV cache pool.
    # Invariant: kv_available_tokens + kv_evictable_tokens + kv_used_tokens <= max_total_num_tokens
    # (the gap accounts for protected/session-held tokens not exposed here).
    # max_total_num_tokens is emitted once at startup via emit_constants.
    #
    # kv_available_tokens:  free (unallocated) slots in the pool.
    # kv_evictable_tokens:  slots holding radix-cached KV data that can be evicted for new requests.
    # kv_used_tokens:       actively used slots (locked by running requests). Equals full_num_used.
    # num_used_tokens:      max(full_num_used, swa_num_used) for hybrid-SWA models, else full_num_used.
    #                       Does NOT include the mamba pool.
    num_used_tokens: int = 0
    kv_available_tokens: int = 0
    kv_evictable_tokens: int = 0
    kv_used_tokens: int = 0
```

`cache_hit_rate` 因此是一个 stats tick 内的 aggregate 比例，不是某个请求的命中结果。

## 概念 5：队列指标可能按 priority 拆 label

`QueueCount` 同时保存 total 和可选 priority breakdown。没有设置默认 priority 时，源码注释指出可能出现 `priority="None"` 标签。

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

Grafana 面板如果只过滤 `priority=""`，看到的是 total；如果过滤具体 priority，看到的是子队列。

## 概念 6：Tokenizer 请求账在请求输出路径写入

TokenizerManager 初始化 request metrics collector 时先建立 label schema。允许的 custom labels 会先以空值加入，后续 `collect_metrics` 处理每次请求级指标事件时再用请求值覆盖。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L527-L556
    def init_metric_collector_watchdog(self):
        # Metrics
        if self.enable_metrics:
            engine_type = DisaggregationMode.to_engine_type(
                self.server_args.disaggregation_mode
            )

            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
            }
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
                TokenizerMetricsCollector,
            )
            self.metrics_collector = tokenizer_collector_cls(
                server_args=self.server_args,
                labels=labels,
                bucket_time_to_first_token=self.server_args.bucket_time_to_first_token,
                bucket_e2e_request_latency=self.server_args.bucket_e2e_request_latency,
                bucket_inter_token_latency=self.server_args.bucket_inter_token_latency,
            )
```

请求路径里，第一次满足条件的输出事件 observe TTFT；后续事件先用累计 `completion_tokens` 求 token 增量，再 observe ITL；finish 时 observe E2E 和 token counters。这里的“事件”不能机械等同于网络 streaming chunk，更不能假定每次只带一个 token。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L2411-L2457
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
            # Get detailed cache breakdown if available
            cached_tokens_details = None
            if (
                hasattr(recv_obj, "cached_tokens_details")
                and recv_obj.cached_tokens_details
            ):
                cached_tokens_details = recv_obj.cached_tokens_details[i]

            spec_verify_ct = (
                recv_obj.spec_verify_ct[i]
                if hasattr(recv_obj, "spec_verify_ct")
                and recv_obj.spec_verify_ct
                and len(recv_obj.spec_verify_ct) > i
                else 0
            )

            self.metrics_collector.observe_one_finished_request(
                labels,
                recv_obj.prompt_tokens[i],
                completion_tokens,
                recv_obj.cached_tokens[i],
                state.time_stats.get_e2e_latency(),
                self._request_has_grammar(state.obj),
                cached_tokens_details,
                spec_verify_ct=spec_verify_ct,
            )
```

这里的系统压力是 per-request latency 不能等 stats tick。TTFT 在第一次可观测输出上写；ITL 使用两次处理之间的时间除以新增 token 数，把同一个平均值计入 `num_new_tokens` 个 histogram 样本；E2E 和 token counters 在 finish 事件写。它是低开销近似，不是逐 token 时间戳，也不是网络分块延迟。PREFILL disaggregation 模式还会跳过这里的 TTFT 分支。

## 概念 7：ReqTimeStats 同时服务 metrics 和 trace

`ReqTimeStatsBase` 持有 metrics collector 和 trace context。某个 stage 是否写入 per-stage histogram，由 `RequestStageConfig.metrics_is_observed` 决定；trace 则按 stage slice 生成 span。

```python
# 来源：python/sglang/srt/observability/req_time_stats.py L260-L305
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

    def init_trace_ctx(
        self,
        rid: str,
        bootstrap_room: Optional[int],
        external_trace_header: Optional[Dict[str, str]] = None,
    ):
        self.trace_ctx = TraceReqContext(
            rid=rid,
            bootstrap_room=bootstrap_room,
            role=self.disagg_mode_str(),
            module_name="request",
            external_trace_header=external_trace_header,
        )

        if not self.trace_ctx.tracing_enable:
            self.trace_ctx = TraceNullContext()

    def trace_slice(
        self,
        stage: RequestStageConfig,
        start_time: float,
        end_time: float,
        attrs: Optional[Dict] = None,
    ):
        if self.trace_ctx.tracing_enable:
            _slice = TraceSliceContext(
                slice_name=stage.stage_name,
                start_time_ns=convert_time_to_realtime_ns(start_time),
                end_time_ns=convert_time_to_realtime_ns(end_time),
                level=stage.level,
                attrs=attrs,
            )
            self.trace_ctx.trace_slice(_slice)
```

因此 per-stage latency 和 OpenTelemetry trace 共享一套 stage 事件，但不是同一个后端。

## 概念 8：RequestLogger 和 exporter 是旁路

RequestLogger 的开关、级别、格式、目标都独立于 Prometheus。它可以记录 received 和 finished 请求，用于单请求复盘或回放。

```python
# 来源：python/sglang/srt/utils/request_logger.py L44-L60
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

        self.metadata: Tuple[Optional[int], Optional[Set[str]], Optional[Set[str]]] = (
            self._compute_metadata()
        )
        self.targets = self._setup_targets()
```

request metrics exporter 也不是 Prometheus。manager 根据 server args 创建 exporter，finish 路径异步写记录。

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

如果 Prometheus 没有 series，不要去 RequestLogger 里找原因；如果你需要复盘某个 rid 的 payload，也不要指望 Prometheus histogram 给出原始请求。

## 本篇结论

- `/metrics` 是暴露面，不是所有可观测数据的源头。
- Scheduler 状态账是 aggregate stats，回答“系统现在怎样”。
- Tokenizer 请求账是 request lifecycle metrics，回答“单请求经历了什么延迟”。
- ReqTimeStats 是 metrics 和 trace 的共同 stage 事实源。
- RequestLogger 和 RequestMetricsExporter 是旁路，适合单请求复盘和离线分析。
- label schema 在 collector 创建时决定；运行中的请求级指标事件只能填已存在的 label。

下一篇 [[SGLang-可观测性-源码走读]] 会沿一次 scrape、一次 stats tick 和一次请求完成走源码。
