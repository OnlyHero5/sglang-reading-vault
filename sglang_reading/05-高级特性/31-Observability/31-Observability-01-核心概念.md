---
type: batch-doc
module: 31-Observability
batch: "31"
doc_type: concept
title: "可观测性 · 核心概念"
tags:
 - sglang/batch/31
 - sglang/module/observability
 - sglang/doc/concept
aliases:
 - "01-核心概念"
updated: 2026-07-02
---
# 可观测性 · 核心概念

> 本节介绍核心术语与模块在架构中的位置。

---

## 用户故事：Grafana 看不到 `cache_hit_rate`

### Persona

**孙 SRE**，按文档在 Grafana 配了 Radix 前缀命中面板，查询 `sglang_cache_hit_rate` 始终 **No data**，但服务日志里 Scheduler 明明在报 stats。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | 部署 SGLang 未加 `--enable-metrics`，Prometheus scrape `/metrics` 404 或空 |
| T1 | 补上 `--enable-metrics`，scrape 成功但无 cache 相关 series |
| T2 | 确认 scrape 的是 **HTTP 主端口**（gRPC 模式需 sidecar `port+1`） |
| T3 | 在 `SchedulerMetricsCollector.log_stats` 映射中找到 `cache_hit_rate` gauge，面板出数 |

**Explain：** 可观测性本模块分三层：Prometheus metrics、Scheduler stats、Request logging。**`cache_hit_rate` 来自 `SchedulerStats`**，仅当 `ServerArgs.enable_metrics=True` 且 Scheduler 在 attn_tp_rank==0 实例化 collector 时才暴露为 Prometheus gauge。默认关闭 metrics，Grafana 无 series 是预期行为而非 RadixCache 未命中。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L53-L61
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

**Comment：** 启动需 `--enable-metrics`；gRPC `--grpc-mode` 时 metrics 在 HTTP sidecar；`enable_metrics_for_all_schedulers` 用于 dp_attention 多副本场景。

### 如果…会怎样（调试）

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| `/metrics` 404 | 未开 `--enable-metrics` | 查 `server_args.enable_metrics` |
| 有 metrics 无 cache 字段 | scrape 错端口（gRPC 主端口无 HTTP） | 改 scrape `grpc_http_sidecar_port` |
| 值恒为 0 但有 series | 前缀确实未命中或 stats 未汇总 | 对照 RequestLogger / RadixAttention Radix 命中条件 |

---

## 1. 可观测性三层

**Explain：** SGLang 可观测性分三层：（1）**Prometheus metrics**——进程内 Counter/Gauge/Histogram，供 Grafana 抓取；（2）**Scheduler stats**——每 log interval 汇总的队列、KV pool、spec 接受率等；（3）**Request logging**——可选 JSON/文本日志，记录单请求 input/output 与 header。三层独立开关，可组合使用；`RequestMetricsExporter` 则把请求级性能数据导出到文件等外部目的地。

**Comment：** 三层互不替代：metrics 给 aggregate SLA，RequestLogger 给 rid 级 replay，Exporter 适合离线分析 pipeline。

---

## 2. enable_metrics 与 collector 角色

**Explain：** CLI `--enable-metrics` 映射到 `ServerArgs.enable_metrics`；Scheduler 仅在 `attn_tp_rank==0`（或 `enable_metrics_for_all_schedulers`）实例化 `SchedulerMetricsCollector`，避免 TP 副本重复上报相同 gauge。TokenizerManager 侧另有 `TokenizerMetricsCollector` 追踪 TTFT/ITL/e2e。

**Code：**

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

**Comment：**

- 默认 False，生产环境需显式开启。
- `enable_metrics_for_all_schedulers` 用于调试多 Scheduler 副本或 dp_attention 场景。
- `stat_loggers` DI 可替换为 Ray 后端（见 `ray_wrappers.py`）。

---

## 3. SchedulerStats 核心字段

**Explain：** `SchedulerStats` 是 Scheduler 每轮 stats 汇总的 dataclass；`MetricsReporter` 将其传给 `SchedulerMetricsCollector.log_stats`，映射到 `sglang:*` Prometheus 指标名。字段涵盖队列长度、吞吐、cache 命中率、KV pool 使用率、投机解码与 PD 分离队列等。

**Code：**

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

**Comment：**

- `token_usage` 取 full/swa/mamba pool 最大值作为瓶颈（命名历史遗留，见源码 FIXME）。
- PD 分离字段（prefill/decode queue）仅在 disagg 模式非零。
- `spec_accept_rate` 与投机解码 投机解码联动。

---

## 4. QueueCount 与优先级标签

**Explain：** 队列长度 metrics 支持按 priority 拆分；未设置 `--default-priority-value` 时 priority=None 会产生 `priority="None"` 标签，Grafana 查询需注意。total 写在默认 label（`priority=""`），各 priority 子 gauge 单独记录。

**Code：**

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

**Comment：**

- `_log_gauge_queue_count` 同时写 total 与各 priority 子 gauge。
- 开启 priority scheduling 时 Grafana 应区分 `priority=""`（total）与 `priority="N"`。

---

## 5. TokenizerMetricsCollector

**Explain：** TokenizerManager 侧 collector 追踪 prompt/generation token 计数、TTFT/ITL/e2e histogram；请求 finish 时 `observe_one_finished_request` 一次性 observe 多个 histogram。cached_tokens 可按 device/host/storage 分 source 上报（HiCache L3 场景）。

**Code：**

```python
# 来源：python/sglang/srt/observability/metrics_collector.py L1635-L1679
    def observe_one_finished_request(
        self,
        labels: Dict[str, str],
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int,
        e2e_latency: float,
        has_grammar: bool,
        cached_tokens_details: Optional[Dict[str, Any]] = None,
        spec_verify_ct: int = 0,
    ):
        self.prompt_tokens_total.labels(**labels).inc(prompt_tokens)
        self.generation_tokens_total.labels(**labels).inc(generation_tokens)
        if spec_verify_ct > 0:
            self.spec_verify_calls_total.labels(**labels).inc(spec_verify_ct)

        # Report cached tokens with detailed source breakdown
        if cached_tokens > 0:
            if cached_tokens_details:
                # Report by cache source (device/host, and storage if L3 enabled)
                def report_cache_source(source: str, value: int):
                    if value > 0:
                        source_labels = {**labels, "cache_source": source}
                        self.cached_tokens_total.labels(**source_labels).inc(value)

                report_cache_source("device", cached_tokens_details.get("device", 0))
                report_cache_source("host", cached_tokens_details.get("host", 0))

                # Storage fields are only present when L3 storage backend is enabled
                if "storage" in cached_tokens_details:
                    storage_tokens = cached_tokens_details.get("storage", 0)
                    if storage_tokens > 0:
                        backend = (
                            cached_tokens_details.get("storage_backend") or "unknown"
                        )
                        report_cache_source(f"storage_{backend}", storage_tokens)
            else:
                # Fallback for backward compatibility
                labels_total = {**labels, "cache_source": "total"}
                self.cached_tokens_total.labels(**labels_total).inc(cached_tokens)

        self.num_requests_total.labels(**labels).inc(1)
        if has_grammar:
            self.num_so_requests_total.labels(**labels).inc(1)
        self.histogram_e2e_request_latency.labels(**labels).observe(float(e2e_latency))
```

**Comment：**

- `spec_verify_calls_total` 与投机 verify 次数对应。
- bucket 边界可通过 `bucket_time_to_first_token` 等 ServerArgs 自定义。
- aborted 请求走 `observe_one_aborted_request`。

---

## 6. RequestLogger

**Explain：** `RequestLogger` 与 Prometheus 独立；通过 `--log-requests` 系列参数控制。Level≥2 时记录 OpenAI 原始请求；finish 时输出 latency 与 token 统计，供 schedule simulator 回放。JSON format 便于 ELK/Loki 索引。

**Code：**

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

**Comment：**

- `log_requests_format=json` 走 structured log。
- whitelisted headers 含 routing-key。
- `SGLANG_LOG_REQUEST_EXCEEDED_MS` 过滤慢请求。

---

## 7. RequestMetricsExporter

**Explain：** 抽象基类，把请求级性能数据（latency、token 数等）序列化为 JSON 记录并写入外部目的地（如文件）。与 Prometheus histogram 互补：Exporter 保留完整请求参数快照，适合离线审计；Prometheus 只做 aggregate。

**Code：**

```python
# 来源：python/sglang/srt/observability/request_metrics_exporter.py L21-L32
class RequestMetricsExporter(ABC):
    """Abstract base class for exporting request-level performance metrics to a data destination."""

    def __init__(
        self,
        server_args: ServerArgs,
        obj_skip_names: Optional[set[str]],
        out_skip_names: Optional[set[str]],
    ):
        self.server_args = server_args
        self.obj_skip_names = obj_skip_names or set()
        self.out_skip_names = out_skip_names or set()
```

**Comment：**

- `ALWAYS_EXCLUDE_FIELDS` 过滤 image/video/audio 等非 JSON 字段。
- 具体实现见 `FileRequestMetricsExporter`（同文件后部）。

---

## 8. 术语对照

| 术语 | 含义 | 源码 |
|------|------|------|
| `SchedulerMetricsCollector` | Scheduler 进程 Prometheus 写入 | metrics_collector.py |
| `TokenizerMetricsCollector` | TM 进程请求级 metrics | metrics_collector.py |
| `MetricsReporter` | Scheduler 内 stats 组装与 log_stats 调用 | metrics_reporter.py |
| `ReqTimeStats` | 单请求各 stage latency 追踪 | req_time_stats.py |
| `RequestLogger` | 可选请求 JSON 日志 | request_logger.py |
| `RequestMetricsExporter` | 请求性能数据外部导出 | request_metrics_exporter.py |

---

## 9. 与 CheckpointEngine 的 metrics 交叉引用

权重热更新期间 `num_paused_reqs` gauge 上升；完成后 `weight_load_duration_seconds{source="ipc"}` 记录 wall time。详见 [[32-CheckpointEngine-00-MOC]] 与 [[12-ModelLoader-00-MOC]] §weight_sync。
