---
title: "SchedulePolicy · 排障指南"
type: troubleshooting
framework: sglang
topic: "SchedulePolicy"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# SchedulePolicy · 排障指南

## 你为什么要读

请求排在前面却没有运行，不一定是 policy 失效：排序之后还有 token、KV、LoRA 和 chunked prefill 预算。本文把“队列顺序不对”和“准入资源不足”拆开，沿 `calc_priority` 到 `PrefillAdder` 找到第一次被拒绝的原因。

这篇按排障问题组织。先看症状，再回到源码入口。

## Q1：`lpm` 和 `fcfs` 到底怎么选？

| 场景 | 建议 | 理由 |
|------|------|------|
| 多请求共享长 system prompt 或 RAG 前缀 | 先试 `lpm` | prefix 命中长的请求优先；是否改善端到端指标仍需固定 workload 对照 |
| 未开启 priority scheduling，且需要稳定 FIFO 公平性 | `fcfs` | 不做 policy-specific 重排；开启 priority 后会按 priority 与入队时间排序 |
| 等待队列经常超过 128 | 接受 `lpm` 临时退化或显式用 `fcfs` | 源码关闭 LPM 专属临时树与最长命中排序；fast-match 元数据填充仍可能发生 |
| tree cache disabled | `fcfs` | cache-aware policy 构造时会被调整 |
| running batch routing key 很重要 | `routing-key` | 与 running batch 同 key 的请求优先 |

常见误解是“`lpm` 退化成 `fcfs` 就一定会做 priority+FCFS 排序”。源码不是这样：队列过长时 active policy 是 `FCFS`，但 `self.policy` 仍是 `LPM`，因此不会走 `self.policy == FCFS` 的 priority 早返回。另一个误解是“退化后完全不匹配 prefix”：通用 fast-match metadata 分支仍可能运行，只是不再做 LPM 的批内临时树和 LPM 排序。

## Q2：为什么 prefix cache 看起来没有生效？

先区分三种情况：

| 现象 | 可能原因 | 源码入口 | 验证 |
|------|----------|----------|------|
| `prefix_indices` 长度一直为 0 | prompt 本身不共享，或 `SGLANG_RADIX_FORCE_MISS` 被打开 | `match_prefix_for_req` | 打断点看 `match_result.device_indices` |
| `num_matched_prefix_tokens` 有值但排序像没变 | 队列超过 128，或被批内前缀逻辑临时降权 | `_determine_active_policy`、`_compute_prefix_matches` | 看 `temporary_deprioritized` |
| LoRA 或租户间共享失败 | `extra_key` 不同 | `RadixKey(token_ids, extra_key)` | 对比请求的 `extra_key` |

`SGLANG_RADIX_FORCE_MISS` 是强制 miss 的验证开关：

```python
# 来源：sglang/python/sglang/srt/managers/schedule_policy.py L102-L131
    match_result = tree_cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(token_ids=token_ids, extra_key=req.extra_key),
            cow_mamba=cow_mamba,
            req=req if include_req else None,
        )
    )
    if envs.SGLANG_RADIX_FORCE_MISS.get():
        match_result = zero_match_result(tree_cache, match_result)
    (
        req.prefix_indices,
        req.last_node,
        req.last_host_node,
        req.best_match_node,
        req.host_hit_length,
        req.swa_host_hit_length,
        req.mamba_host_hit_length,
    ) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.best_match_node,
        match_result.host_hit_length,
        match_result.swa_host_hit_length,
        match_result.mamba_host_hit_length,
    )
    max_len = req._compute_max_prefix_len(len(token_ids))
    req.num_matched_prefix_tokens = min(
        len(req.prefix_indices) + req.host_hit_length, max_len
    )
```

## Q3：`NO_TOKEN` 和 `OTHER` 怎么判断？

把它们当成 Scheduler 的两种“停止扫描”协议，而不是完整根因：

| 返回值 | 意味着 | 优先排查 |
|--------|--------|----------|
| `NO_TOKEN` | 容量口径不能安全继续 | 总/即时 KV、SWA pool、Mamba slot、ignore-eos 生存期估算 |
| `OTHER` | 本轮边界不允许继续 | `max_prefill_tokens`、chunk/DLLM 配额、对齐后零长度、请求数上限、delayer、上下文并行 |

Scheduler 对 `NO_TOKEN` 会额外标记 batch full：

```python
# 来源：sglang/python/sglang/srt/managers/scheduler.py L2884-L2892
            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        self.running_batch.batch_is_full = len(
                            adder.can_run_list
                        ) > 0 or (not self.running_batch.is_empty())
                    else:
                        self.running_batch.batch_is_full = True
```

如果 `OTHER` 很多但 `can_run_list` 为空，先记录具体返回行，再看 delayer、请求数/CP 限制或零 chunk；如果 `NO_TOKEN` 很多，检查 allocator 动态可用量、cache 可驱逐量、page overhead、输出估算以及特殊池。注意两种结果都会 `break`，区别主要在 Scheduler 是否更新 `batch_is_full`。

## Q4：`CLIP_MAX_NEW_TOKENS` 会截断生成吗？

不会。它只截断 Scheduler 对未来 decode KV 占用的估算，不改采样停止条件。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_policy.py L64-L70
# Clip the estimation of max_new_tokens for the request whose max_new_tokens is very large.
# This can prevent the server from being too conservative.
# Note that this only clips the estimation in the scheduler but does not change the stop
# condition. The request can still generate tokens until it hits the unclipped max_new_tokens.
CLIP_MAX_NEW_TOKENS = int(
    os.environ.get("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", "4096")
)
```

线上含义：如果用户请求声明非常大的 `max_new_tokens`，Scheduler 不会按完整值为它提前保守预留全部未来空间。实际生成长度仍由 sampling params 和停止条件决定。

## Q5：PrefillDelayer 和 MinFreeSlotsDelayer 会互相覆盖吗？

不会，它们是两道不同的门。

| 门 | 生效点 | 返回后果 |
|----|--------|----------|
| `MinFreeSlotsDelayer` | `_get_new_batch_prefill_raw` 早期 | 直接 `return None`，本轮没有 prefill |
| `PrefillDelayer` | `PrefillAdder.add_one_req` 内 | `add_one_req` 返回 `OTHER` |

`MinFreeSlotsDelayer` 也不是给任意配置自动开启：`resolve_min_free_slots` 会把 `<=1` 视为关闭，`max_running_requests < 8` 时关闭；显式阈值还会被 `min(4, max(2, (max_running_requests + 5) // 6))` 封顶。未显式配置时只有 DFlash workload 使用该公式作为默认值，其他 workload 保持关闭。

`PrefillDelayer` 构造时要求 overlap scheduling 开启；源码虽然包含 `disable_overlap_schedule` 时选择 NCCL group 的防御性分支，但同一构造器末尾会断言该配置必须为 False，不能据此前半段推断“关闭 overlap 也支持 delayer”：

```python
# 来源：sglang/python/sglang/srt/managers/prefill_delayer.py L43-L113
class PrefillDelayer:
    def __init__(
        self,
        dp_size: int,
        attn_tp_size: int,
        cpu_group,
        server_args,
        max_delay_passes: int,
        token_usage_low_watermark: Optional[float],
        metrics_collector: Optional["SchedulerMetricsCollector"] = None,
        device: Optional["torch.device"] = "cpu",
        device_group=None,
    ):
        self._max_delay_passes = max_delay_passes
        self._token_usage_low_watermark = token_usage_low_watermark
        # Queue-based trigger is opt-in: activates only when queue_min_ratio
        # is explicitly set. Additive with the slot-based trigger.
        self._queue_min_ratio = server_args.prefill_delayer_queue_min_ratio
        # Fall back to 5000ms if unset; this is a local safety cap, not a
        # semantic default, so we don't surface it via ServerArgs.
        self._max_delay_ms = server_args.prefill_delayer_max_delay_ms
        if self._max_delay_ms is None:
            self._max_delay_ms = 5000.0
        self._queue_trigger_enabled = self._queue_min_ratio is not None
        logger.info(
            f"PrefillDelayer initialized with "
            f"max_delay_passes={self._max_delay_passes} "
            f"token_usage_low_watermark={self._token_usage_low_watermark} "
            f"queue_min_ratio={self._queue_min_ratio} "
            f"max_delay_ms={self._max_delay_ms} "
            f"queue_trigger_enabled={self._queue_trigger_enabled}"
        )
        self.dp_size = dp_size
        self.enable_dp_attention = server_args.enable_dp_attention
        dp_size_dim = dp_size if self.enable_dp_attention else 1

        # Mirror scheduler_dp_attn_mixin's NCCL all-gather path: when the
        # env flag is on (or overlap scheduling is disabled), ride the NCCL
        # device group on `device` instead of gloo on CPU.
        use_nccl = (
            server_args.disable_overlap_schedule
            or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
        )
        if use_nccl:
            assert (
                device_group is not None
            ), "device_group is required when using NCCL for PrefillDelayer all-gather"
            self._gather_group = device_group
            self._gather_device = device
        else:
            self._gather_group = cpu_group
            self._gather_device = "cpu"

        # Fields packed per rank into the all-gather tensor: prefillable,
        # token_watermark_force_allow, running_batch, max_prefill_bs,
        # waiting_queue_len.
        self._global_info_buffer = torch.empty(
            (dp_size_dim, attn_tp_size, 5),
            dtype=torch.int64,
            device=self._gather_device,
        )

        self._metrics_collector = metrics_collector

        self._curr_state: Optional[_State] = None
        self.skip_first_delayer = True

        assert (
            not server_args.disable_overlap_schedule
        ), "To use PrefillDelayer, disable_overlap_schedule must be False."
```

如果 TTFT 变差但吞吐变好，可能是 delayer 正在按设计工作；如果二者都变差，检查 `output_reason` 是否长期为 `delay`，以及 `max_delay_passes`、queue ratio、token low watermark 是否过于激进。

## Q6：PrefillDelayer 的 `all`、`mixed`、`none` 怎么读？

| 状态 | 含义 | 结果倾向 |
|------|------|----------|
| `all` | 所有相关 DP 切片都有可 prefill 请求 | 可能因为 slot 或 queue 条件延迟，也可能放行 |
| `mixed` | 有些 rank 可 prefill，有些不可 | 倾向等待，直到达到最大 delay pass |
| `none` | 没有 rank 可 prefill | 放行与否无实际影响，源码选择 allow |

`mixed` 分支体现了跨 rank 节奏控制：

```python
# 来源：sglang/python/sglang/srt/managers/prefill_delayer.py L272-L299
        elif prefillable_status == "mixed":
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                    **wait_info,
                )

            prev_delayed_count = prev_state.delayed_count if prev_state else 0
            if prev_delayed_count < self._max_delay_passes - 1:
                next_state = prev_state or _State()
                next_state = next_state.bump_delayed_count()
                return _NegotiateOutput(
                    next_state=next_state,
                    output_allow=False,
                    output_reason="delay",
                    **debug_info,
                )
            else:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="wait_timeout",
                    **debug_info,
                    **wait_info,
                )
```

如果看到 `wait_timeout`，说明 `mixed` 状态已经等够 pass 数后放行，不是资源突然变多。`all` 分支的 queue trigger 另有墙钟 `_max_delay_ms`：超时后取消 queue delay，结果通常记为 `wait_success`，不要把两种超时混为一个 outcome。

## Q7：为什么 chunked prefill 不能被普通 delay 打断？

chunked prefill 不是一个普通 waiting 请求。它已经开始占用并推进一段输入，如果中间块不继续提交，状态可能卡在中间。Scheduler 注释直接把这件事和 memory leak 绑定。

排障路径：

| 现象 | 看哪里 | 预期 |
|------|--------|------|
| 长 prompt 第一块后不继续 | `self.chunked_req` | 不应在下一轮被普通 slot delay 拦住 |
| 每块都像重新预留输出 | `_update_prefill_budget` 调用参数 | 中间块 `max_new_tokens` 应为 0 |
| `can_run_list` 为空但有 chunked req | `add_chunked_req` | 正常情况下应把 chunked req 加回 |

## Q8：优先级抢占什么时候发生？

抢占只在 batch full 且允许 priority preemption 时作为例外通道。它不是常规排序的一部分。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_policy.py L1171-L1213
        preemptible_reqs = []
        min_tokens_to_remove = (
            len(req.full_untruncated_fill_ids)
            - len(req.prefix_indices)
            + min(req.sampling_params.max_new_tokens, CLIP_MAX_NEW_TOKENS)
            - self.rem_total_tokens
        )
        for running_req in sorted_valid_running_reqs:
            # Priority difference needs to meet the threshold to be preemptible.
            priority_diff = (req.priority - running_req.priority) * (-priority_sign)

            if priority_diff > self.priority_scheduling_preemption_threshold:
                preemptible_reqs.append(running_req)
                min_tokens_to_remove -= self._get_running_request_total_token_offset(
                    running_req
                )
                if min_tokens_to_remove <= 0:
                    break
            else:
                break

        # Check max token count limit can be met
        if len(preemptible_reqs) == 0 or min_tokens_to_remove > 0:
            return False

        # Preempt running requests. Release allocated resources for immediate usage.
        preemptible_reqs = set(preemptible_reqs)
        keep_indices = []
        release_counter = 0
        for i, running_req in enumerate(self.running_batch.reqs):
            if running_req in preemptible_reqs:
                self.rem_total_token_offset -= (
                    self._get_running_request_total_token_offset(running_req)
                )
                release_counter += 1
                self.running_batch.release_req(
                    i, len(self.running_batch.reqs) - release_counter, server_args
                )
            else:
                keep_indices.append(i)
        self.running_batch.filter_batch(keep_indices=keep_indices)
        self.preempt_list.extend(preemptible_reqs)
        return True
```

抢占 commit 必须同时满足两个条件：优先级差超过阈值，并且按 running-request future-token offset 估算，候选集合释放的 token 足以填平 `min_tokens_to_remove`。这只是抢占通道的前置条件，不保证新请求随后一定成功：`add_one_req` 仍会重新检查 page overhead、SWA、Mamba、chunk 与其他本轮边界。

## Q9：怎么做最小实验验证这个模块？

| 实验 | 操作 | 预期 |
|------|------|------|
| prefix 策略实验 | 固定模型、并发、输入和输出长度，相同长 system prompt 分别用 `lpm` 和 `fcfs` | 先验证顺序和命中差异；TTFT/吞吐方向由 workload 实测决定 |
| 强制 miss 实验 | 设置 `SGLANG_RADIX_FORCE_MISS=1` | 先验证 prefix hit 归零或降低，再观察 prefill 成本和延迟变化 |
| chunk 实验 | 降低 chunked prefill size，发长 prompt | 多轮 `chunked_req` 推进，未完成块输出估算为 0，最后一块才计入 `max_new_tokens` 估算 |
| delayer 实验 | 开启 delayer debug 或 metrics | 能看到 `delay`、`wait_timeout`、`token_watermark` 等 outcome |

做实验时先固定模型、prompt、并发和输出长度，否则 TTFT 变化不一定来自调度策略。

## 运行验证

如果只是维护文档或排查上游变更，先用源码检索确认四个关键点还在：prefix match、`PrefillAdder` 准入、`PrefillDelayer` outcome、priority preemption commit。

```powershell
rg -n 'def match_prefix_for_req|class SchedulePolicy|def calc_priority|class PrefillAdder|def add_chunked_req|PrefillDelayer|output_reason="delay"|output_reason="wait_timeout"|output_reason="token_watermark"|priority_scheduling_preemption_threshold|preemptible_reqs' sglang/python/sglang/srt/managers/schedule_policy.py sglang/python/sglang/srt/managers/prefill_delayer.py sglang/python/sglang/srt/managers/scheduler.py
```

命中结果应该能对应本文的排障顺序：先看排序和 prefix，再看 prefill 预算，最后看 delayer 或 priority preemption 是否改变了准入结果。
