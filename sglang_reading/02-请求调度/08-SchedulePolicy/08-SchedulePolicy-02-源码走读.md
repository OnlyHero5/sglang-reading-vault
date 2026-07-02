---
type: batch-doc
module: 08-SchedulePolicy
batch: "08"
doc_type: walkthrough
title: "调度策略 · 源码走读"
tags:
 - sglang/batch/08
 - sglang/module/schedule-policy
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# 调度策略 · 源码走读

## 走读顺序

1. `schedule_policy.py` — `SchedulePolicy` 排序逻辑
2. `schedule_policy.py` — `PrefillAdder` 预算准入
3. `prefill_delayer.py` — 跨 rank prefill 延迟协商
4. `min_free_slots_delayer.py` — 本地 slot 延迟

---

## 1. schedule_policy.py — SchedulePolicy

### 1.1 构造与策略校验

**Explain：** Scheduler 在 `init_schedule_policy` 中构造 `SchedulePolicy`。若 tree cache 被禁用，cache-aware 策略自动降为 FCFS。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L158-L174, L235-L251
    def __init__(
        self,
        policy: str,
        tree_cache: BasePrefixCache,
        enable_hierarchical_cache: bool,
        enable_priority_scheduling: bool,
        schedule_low_priority_values_first: bool,
    ):
        self.policy = self._validate_and_adjust_policy(policy, tree_cache)
        self.tree_cache = tree_cache
        self.enable_hierarchical_cache = enable_hierarchical_cache
        self.enable_priority_scheduling = enable_priority_scheduling
        self.schedule_low_priority_values_first = schedule_low_priority_values_first
        self.priority_sign = 1 if schedule_low_priority_values_first else -1

        # It is used to find the matching prefix for in-batch prefix caching.
        self.waiting_queue_radix_tree = RadixCache.create_simulated()
```

**Comment：**

- `waiting_queue_radix_tree` 是**纯内存模拟树**，仅用于批内前缀检测，不参与真实 KV 存储。
- `priority_sign`：配合 `schedule_low_priority_values_first` 决定 priority 数值越大越优先还是越小越优先。

---

### 1.2 `calc_priority` — 调度主入口

**Explain：** 根据当前生效策略对 `waiting_queue` 原地排序。队列 >128 时 LPM 自动降级 FCFS（避免 O(n log n) 前缀匹配开销）。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L176-L233
    def calc_priority(
        self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None
    ) -> None:
        policy = self._determine_active_policy(waiting_queue)

        # Populate req.num_matched_prefix_tokens at schedule time. Cache-aware policies
        # set it in _compute_prefix_matches; do the same full match for
        # cache-agnostic policies when the radix supports it, so the load
        # snapshot has it. Skip on decode (never prefills).
        if (
            not isinstance(policy, CacheAwarePolicy)
            and self.tree_cache.supports_fast_match_prefix()
            and get_global_server_args().disaggregation_mode != "decode"
        ):
            for r in waiting_queue:
                match_prefix_for_req(self.tree_cache, r, include_req=True)

        if self.policy == CacheAgnosticPolicy.FCFS:
            if self.enable_priority_scheduling:
                SchedulePolicy._sort_by_priority_and_fcfs(
                    waiting_queue, self.priority_sign
                )
            return

        if isinstance(policy, CacheAwarePolicy):
            temporary_deprioritized = self._compute_prefix_matches(
                waiting_queue, policy
            )
            if policy == CacheAwarePolicy.LPM:
                SchedulePolicy._sort_by_longest_prefix(
                    waiting_queue, temporary_deprioritized
                )
            elif policy == CacheAwarePolicy.DFS_WEIGHT:
                SchedulePolicy._sort_by_dfs_weight(waiting_queue, self.tree_cache)
            else:
                raise ValueError(f"Unknown CacheAware Policy: {policy=}")
        else:
            if policy == CacheAgnosticPolicy.FCFS:
                pass
            elif policy == CacheAgnosticPolicy.LOF:
                SchedulePolicy._sort_by_longest_output(
                    waiting_queue,
                    self.enable_priority_scheduling,
                    self.priority_sign,
                )
            elif policy == CacheAgnosticPolicy.RANDOM:
                SchedulePolicy._sort_randomly(waiting_queue)
            elif policy == CacheAgnosticPolicy.ROUTING_KEY:
                if running_batch is not None:
                    SchedulePolicy._sort_by_routing_key(waiting_queue, running_batch)
            else:
                raise ValueError(f"Unknown CacheAgnostic Policy: {policy=}")

    def _determine_active_policy(self, waiting_queue: List[Req]) -> Policy:
        if self.policy == CacheAwarePolicy.LPM and len(waiting_queue) > 128:
            # Turn off the expensive prefix matching and sorting when the #queue is large.
            return CacheAgnosticPolicy.FCFS
        return self.policy
```

**Comment：**

- FCFS + priority：只按 `(priority, wait_queue_entry_time)` 排序，不走前缀逻辑。
- Cache-agnostic 策略在支持 fast match 时仍会填充 `num_matched_prefix_tokens`，供 metrics / load snapshot 使用。
- `disaggregation_mode == "decode"` 时跳过 match——decode 节点不做 prefill。

---

### 1.3 `_compute_prefix_matches` — 批内前缀

**Explain：** 遍历等待队列，对全局匹配短的请求检查批内共享前缀，决定 deprioritize 集合。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L253-L301
    def _compute_prefix_matches(
        self, waiting_queue: List[Req], policy: CacheAwarePolicy
    ) -> Set[int]:
        """
        Computes and caches the matching prefixes for requests in the waiting queue,
            and handles in-batch prefix caching logic.
        """
        temporary_deprioritized: Set[int] = set()
        self.waiting_queue_radix_tree.reset()

        for r in waiting_queue:
            prefix_ids = r.origin_input_ids + r.output_ids
            extra_key = r.extra_key
            match_result = match_prefix_for_req(
                self.tree_cache, r, prefix_ids, include_req=True
            )

            # NOTE(sang): This logic is for in-batch prefix caching;
            # If there are more than 1 request that have small matching prefix from
            # existing cache, but all those requests share the same prefix, we prefer
            # to schedule only one of them so that we can increase the cache hit rate.
            # We prefer to set IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD > 0 because too small
            # threshold means we cannot use in-batch prefix caching for short prefixes.
            # It is kind of common when the engine is long running (e.g., imagine the prefix "the").
            if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
                match_result = self.waiting_queue_radix_tree.match_prefix(
                    MatchPrefixParams(
                        key=RadixKey(token_ids=prefix_ids, extra_key=extra_key)
                    )
                )
                if envs.SGLANG_RADIX_FORCE_MISS.get():
                    match_result = zero_match_result(
                        self.waiting_queue_radix_tree, match_result
                    )
                in_batch_matching_prefixes = match_result.device_indices
                if (
                    len(in_batch_matching_prefixes)
                    >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD
                ):
                    temporary_deprioritized.add(r.rid)
                else:
                    # Insert with a dummy key
                    self.waiting_queue_radix_tree.insert(
                        InsertParams(
                            key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool),
                        )
                    )
        return temporary_deprioritized
```

**Comment：**

- `temporary_deprioritized` 存的是 `rid`（request id），排序时用 `float("inf")` 把 key 推到队尾。
- insert 使用 `torch.bool` dummy value——模拟树只关心 key 结构，不存真实 KV。

---

### 1.4 LPM 与 DFS 排序

**Explain：** LPM 按 `num_matched_prefix_tokens` 降序；DFS 按 Radix 树节点权重做深度优先遍历输出顺序。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L303-L336, L405-L424
    @staticmethod
    def _sort_by_longest_prefix(
        waiting_queue: List[Req], temporary_deprioritized: Set[int]
    ) -> None:
        """Sorts the waiting queue based on the longest prefix match."""
        waiting_queue.sort(
            key=lambda r: (
                -r.num_matched_prefix_tokens
                if r.rid not in temporary_deprioritized
                else float("inf")
            )
        )

    @staticmethod
    def _sort_by_dfs_weight(
        waiting_queue: List[Req], tree_cache: BasePrefixCache
    ) -> None:
        """Sorts the waiting queue based on a depth-first search weighting."""
        last_node_to_reqs = defaultdict(list)
        for req in waiting_queue:
            last_node_to_reqs[req.last_node].append(req)

        node_to_weight = defaultdict(int)
        for node in last_node_to_reqs:
            node_to_weight[node] = len(last_node_to_reqs[node])
        SchedulePolicy._calc_weight(tree_cache.root_node, node_to_weight)

        waiting_queue.clear()
        SchedulePolicy._get_dfs_priority(
            tree_cache.root_node,
            node_to_weight,
            last_node_to_reqs,
            waiting_queue,
        )
```

**Comment：**

- DFS 权重自底向上累加：子节点权重汇总到父节点，DFS 时优先走「等待请求多」的分支。
- `_sort_by_routing_key` 统计 running batch 中 routing_key 频次，waiting 中与 hot key 匹配的请求靠前（见 §1.5）。

---

### 1.5 ROUTING_KEY 与 priority+FCFS

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L360-L399
    @staticmethod
    def _sort_by_priority_and_fcfs(
        waiting_queue: List[Req], priority_sign: int
    ) -> None:
        """Sorts the waiting queue based on the request priority then received titmestamp."""
        waiting_queue.sort(
            key=lambda x: (
                x.priority * priority_sign,
                x.time_stats.wait_queue_entry_time,
            )
        )

    @staticmethod
    def _sort_by_routing_key(
        waiting_queue: List[Req], running_batch: ScheduleBatch
    ) -> None:
        """Sorts waiting queue by routing key frequency in running batch."""
        routing_key_counts = Counter(
            r.routing_key for r in running_batch.reqs if r.routing_key
        )

        if _ROUTING_KEY_POLICY_DEBUG_LOG:
            waiting_keys_before = [r.routing_key for r in waiting_queue]
            logger.info(
                f"routing_key_counts={dict(routing_key_counts)}, "
                f"waiting_keys_before={waiting_keys_before}"
            )

        if not routing_key_counts:
            return

        def sort_key(req: Req):
            key = req.routing_key
            if key and key in routing_key_counts:
                count = routing_key_counts[key]
                return (0, -count, key)
            else:
                return (1, 0, key or "")

        waiting_queue.sort(key=sort_key)
```

**Comment：**

- `routing-key` 策略需要 `running_batch` 非空才有意义；无 running 请求时不改变顺序。
- sort_key 三元组：`(tier, -count, key)` — tier 0 表示 key 在 running 中出现过。

---

## 2. schedule_policy.py — PrefillAdder

### 2.1 初始化与 running 预估

**Explain：** 构造时从 running batch 累加 decode 阶段的 token 预留（`new_token_ratio * 剩余 max_new_tokens`），并区分 SWA / Mamba / 普通 KV 路径。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L433-L540
class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        tree_cache: BasePrefixCache,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        running_batch: ScheduleBatch,
        new_token_ratio: float,
        rem_input_tokens: int,
        rem_chunk_tokens: Optional[int],
        num_mixed_decode_tokens: int = 0,
        priority_scheduling_preemption_threshold: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: Optional[int] = None,
        prefill_max_requests: Optional[int] = None,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor] = None,
        dllm_config: Optional[DllmConfig] = None,
        waiting_queue_len: int = 0,
    ):
        self.page_size = page_size
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.running_batch = running_batch
        self.new_token_ratio = new_token_ratio
        self.rem_input_tokens = rem_input_tokens - num_mixed_decode_tokens
        self.rem_chunk_tokens = rem_chunk_tokens
        self.dllm_config = dllm_config

        if self.dllm_config is not None:
            self._init_dllm_meta(dllm_config)

        if self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= num_mixed_decode_tokens
        self.rem_total_token_offset = num_mixed_decode_tokens
        self.cur_rem_token_offset = num_mixed_decode_tokens

        self.req_states = None
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.log_hit_tokens = 0
        self.reprocessed_log_hit_tokens = 0
        # TODO(lsyin): report the real input tokens excluding page alignment
        self.log_input_tokens = 0
        self.reprocessed_log_input_tokens = 0

        if running_batch is not None:
            # Estimate the offset in the remaining token space
            self.rem_total_token_offset += sum(
                [
                    self._get_running_request_total_token_offset(r)
                    for r in running_batch.reqs
                ]
            )

        # DeepSeek V4 HiSparse wraps an SWATokenToKVPoolAllocator internally and
        # exposes the full SWA allocator interface.
        self.is_hybrid_swa = isinstance(
            self.token_to_kv_pool_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        )
        self.is_all_swa = isinstance(
            self.token_to_kv_pool_allocator, PureSWATokenToKVPoolAllocator
        )
        self.is_hybrid_ssm_cache = self.tree_cache.supports_mamba()

        self.rem_swa_token_offset = 0

        # Unified-pool joint budget: a new mamba state consumes shared-gap bytes
        # that `rem_total_tokens` (full KV) otherwise counts as free, so reserve
        # the gap per new mamba slot or admission over-commits. Gate on the
        # ALLOCATOR being the unified Mamba composite, NOT on `is_hybrid_ssm_cache`
        # (False for `ChunkCache`, which would skip the reservation on the
        # chunk-cache path): the gap coupling is a property of the byte buffer.
        self._mamba_slot_cost = 0
        if isinstance(
            self.token_to_kv_pool_allocator, UnifiedMambaTokenToKVPoolAllocator
        ):
            self._mamba_slot_cost = (
                self.token_to_kv_pool_allocator.mamba_slot_full_token_cost()
            )

        # `mamba_gap_reserve` is charged to `rem_total_tokens`, which INCLUDES
        # `full_evictable_size()` — but `alloc_req_slots` can only recover
        # MAMBA-recoverable bytes for a mamba slot (shared gap + peer holes +
        # mamba-evictable radix), NOT full-evictable. Gate new mamba slots on
        # that mamba-recoverable budget separately or an over-admit hits the
        # fail-loud `RuntimeError`. `None` outside the unified Mamba pool.
        self.rem_mamba_slots = None
        if self._mamba_slot_cost:
            self.rem_mamba_slots = (
                self.token_to_kv_pool_allocator.mamba_allocator.schedulable_available_size()
            )
            if self.is_hybrid_ssm_cache:
                self.rem_mamba_slots += self.tree_cache.mamba_evictable_size()

        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.dsa_prefill_cp_in_seq_split = is_dsa_prefill_cp_in_seq_split()
        self.max_running_requests = max_running_requests
        self.prefill_context_parallel_enabled = is_prefill_context_parallel_enabled()
        self.prefill_max_requests = prefill_max_requests
        self.prefill_delayer_single_pass = prefill_delayer_single_pass
        self.max_prefill_bs = max_prefill_bs
        # Snapshot of scheduler waiting_queue length at the start of this
        # prefill pass. Used by PrefillDelayer's queue-based trigger.
        self.waiting_queue_len = waiting_queue_len
```

**Comment：**

- `new_token_ratio` 来自 Scheduler 的 tracker，用于保守估计 decode 占用，避免 prefill 占满后 decode OOM。
- `num_mixed_decode_tokens`：mixed chunk 模式下 running decode 已占用的 input token 数。
- Mamba unified pool 单独维护 `rem_mamba_slots`，因为 full evictable 字节不能覆盖 mamba slot 成本。

---

### 2.2 `rem_total_tokens` 属性

**Explain：** 根据 allocator 类型选择不同的 available + evictable 计算路径。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L557-L579
    @property
    def rem_total_tokens(self):
        if self.is_all_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.swa_available_size()
                + self.tree_cache.swa_evictable_size()
            )
        elif self.is_hybrid_swa:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.full_available_size()
                + self.tree_cache.full_evictable_size()
            )
        elif self.is_hybrid_ssm_cache:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.full_evictable_size()
            )
        else:
            available_and_evictable = (
                self.token_to_kv_pool_allocator.available_size()
                + self.tree_cache.evictable_size()
            )
        return available_and_evictable - self.rem_total_token_offset
```

**Comment：**

- **Hybrid SWA** 模型（如 DeepSeek）prefill 主要扣 full pool；decode 阶段才大量用 SWA pool。
- `cur_rem_tokens` 与 `rem_total_tokens` 使用相同 available 源，但 offset 不同——前者只计本轮 extend，后者还计 max_new_tokens 预估。

---

### 2.3 `budget_state` — 能否继续加请求

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L654-L675
    def budget_state(self):
        no_token = self.rem_total_tokens <= 0 or self.cur_rem_tokens <= 0
        if not no_token and self.is_hybrid_swa:
            no_token = self.rem_swa_tokens <= 0
        # Gate new mamba slots separately: rem_total_tokens' full_evictable can't
        # cover a mamba slot, which needs mamba-recoverable bytes (see __init__).
        if not no_token and self.rem_mamba_slots is not None:
            no_token = self.rem_mamba_slots <= 0
        if no_token:
            return AddReqResult.NO_TOKEN

        if self.rem_input_tokens <= 0:
            return AddReqResult.OTHER

        if self.dllm_config is not None:
            if self.rem_dllm_tokens <= 0:
                return AddReqResult.OTHER
        else:
            if self.rem_chunk_tokens is not None and self.rem_chunk_tokens <= 0:
                return AddReqResult.OTHER

        return AddReqResult.CONTINUE
```

**Comment：**

- Mamba slot 与 token 分开 gate——避免用 full evictable 误判可准入。
- `rem_input_tokens <= 0` 返回 `OTHER` 而非 `NO_TOKEN`：表示 prefill token 上限到了，但未必 KV 耗尽。

---

### 2.4 `_update_prefill_budget` — 扣减预算

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L677-L720
    def _update_prefill_budget(
        self,
        prefix_len: int,
        extend_input_len: int,
        max_new_tokens: int,
        retracted_stain: bool,
        mamba_gap_reserve: int = 0,
    ):
        # TODO(lsyin): check this workaround logic, which only ensures the prefill will not out of memory, and may be too conservative
        extend_input_len = self.ceil_paged_tokens(extend_input_len)

        # alloc_extend reserves an extra page_size per request to make sure the budget doesn't over-commit
        page_overhead = self.page_size
        # `mamba_gap_reserve` (shared Mamba pool only; 0 otherwise) charges the new
        # mamba state's shared-gap cost to BOTH full budgets: the slot is allocated
        # immediately (counts against `cur_rem`) and held for the request lifetime
        # (counts against `rem_total`). See `_mamba_gap_budget_for_req`.
        self.rem_total_token_offset += (
            extend_input_len + max_new_tokens + page_overhead + mamba_gap_reserve
        )
        self.cur_rem_token_offset += (
            extend_input_len + page_overhead + mamba_gap_reserve
        )
        # The new mamba slot also consumes one mamba-recoverable slot (gated
        # separately so full_evictable can't cover it — see __init__).
        if mamba_gap_reserve and self.rem_mamba_slots is not None:
            self.rem_mamba_slots -= 1
        self.rem_input_tokens -= extend_input_len

        if self.is_hybrid_swa:
            self.rem_swa_token_offset += self._swa_budget_for_req(extend_input_len)

        if self.dllm_config is not None:
            self.rem_dllm_tokens -= extend_input_len
        elif self.rem_chunk_tokens is not None:
            self.rem_chunk_tokens -= extend_input_len

        # reprocessed_log_* is a subset of log_*; metrics_reporter subtracts it
        # when computing the first-attempt prefix cache hit rate.
        self.log_hit_tokens += prefix_len
        self.log_input_tokens += extend_input_len
        if retracted_stain:
            self.reprocessed_log_hit_tokens += prefix_len
            self.reprocessed_log_input_tokens += extend_input_len
```

**Comment：**

- 每个请求额外扣 `page_size` 作为页对齐开销（`alloc_extend` 可能多占一页）。
- `mamba_gap_reserve` 来自 `_mamba_gap_budget_for_req`，仅 unified Mamba pool 且 `mamba_pool_idx is None` 时非零。
- `log_hit_tokens` / `log_input_tokens` 供 metrics 计算 prefix cache hit rate。

---

### 2.5 `add_one_req` — 单请求准入（核心）

**Explain：** 依次检查：PrefillDelayer → CP 限制 → prefill_max_requests → KV/SWA 预算 → host load back → 分块/整段 prefill → lock radix 节点。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L968-L1141（节选）
    def add_one_req(
        self, req: Req, has_chunked_req: bool, truncation_align_size: Optional[int]
    ):
        if (self.prefill_delayer_single_pass is not None) and (
            not self.prefill_delayer_single_pass.negotiate_should_allow_prefill(
                local_prefillable=True,
                running_batch=self.running_batch.batch_size(),
                max_prefill_bs=self.max_prefill_bs,
                max_running_requests=self.max_running_requests,
                waiting_queue_len=self.waiting_queue_len,
            )
        ):
            return AddReqResult.OTHER
        # TODO support cp with multiple requests
        # Enabling context parallelism currently presents precision issues;
        # therefore, the prefill-batch setting is temporarily set to 1.
        if (self.dsa_prefill_cp_in_seq_split) and len(self.can_run_list) >= 1:
            return AddReqResult.OTHER

        if (x := self.prefill_max_requests) is not None and len(self.can_run_list) >= x:
            return AddReqResult.OTHER

        if req.sampling_params.ignore_eos and getattr(self.tree_cache, "disable", True):
            return self.add_one_req_ignore_eos(req)

        # Reserve page_size for page-alignment overhead: the paged allocator may
        # consume one extra page per request (see alloc_extend), which
        # _update_prefill_budget also deducts.
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            CLIP_MAX_NEW_TOKENS,
        )
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        total_tokens = cand_extend_input_len + max_new + self.page_size
        # Shared Mamba pool: fold the new mamba state's shared-gap cost into
        # `total_tokens` so both `rem_total_tokens` gates reflect the joint budget.
        total_tokens += self._mamba_gap_budget_for_req(req)

        # adjusting the input_tokens based on host_hit_length and page_size
        real_input_tokens = cand_extend_input_len - req.host_hit_length
        real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
        prefix_len = len(req.prefix_indices)

        if total_tokens >= self.rem_total_tokens:
            return AddReqResult.NO_TOKEN

        if self.is_hybrid_swa:
            swa_needed = self._swa_budget_for_req(
                cand_extend_input_len, swa_host_hit_length=req.swa_host_hit_length
            )
            if swa_needed >= self.rem_swa_tokens:
                return AddReqResult.NO_TOKEN

        if (
            self.rem_chunk_tokens is None
            and len(self.can_run_list) != 0
            and real_input_tokens >= self.rem_input_tokens
        ):
            # If without chunked prefill:
            # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
            # - if the can_run_list is empty, always accept the first prefill request
            return AddReqResult.OTHER

        with self._lock_node(req.last_node):
            # self.rem_total_tokens may decrease after the lock acquisition
            if total_tokens >= self.rem_total_tokens:
                return AddReqResult.NO_TOKEN

            if self.is_hybrid_swa:
                swa_needed = self._swa_budget_for_req(
                    cand_extend_input_len, swa_host_hit_length=req.swa_host_hit_length
                )
                if swa_needed >= self.rem_swa_tokens:
                    return AddReqResult.NO_TOKEN

            if req.needs_host_load_back():
                new_indices, req.last_node = self.tree_cache.init_load_back(
                    InitLoadBackParams(
                        best_match_node=req.best_match_node,
                        host_hit_length=req.host_hit_length,
                        req=req,
                    )
                )
                req.prefix_indices = torch.cat([req.prefix_indices, new_indices])
                prefix_len = len(req.prefix_indices)
                req.cache_protected_len = prefix_len

            input_tokens = self.ceil_paged_tokens(
                len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
            )

            if (
                self.rem_chunk_tokens is None
                and len(self.can_run_list) != 0
                and input_tokens >= self.rem_input_tokens
            ):
                # If without chunked prefill:
                # - if the can_run_list is not empty, we satisfy the constraint of (max_prefill_tokens)
                # - if the can_run_list is empty, always accept the first prefill request
                return AddReqResult.OTHER

            if self.dllm_config is not None:
                if self.rem_dllm_tokens <= 0:
                    return AddReqResult.OTHER

                assert (
                    truncation_align_size is None
                ), "truncation_align_size is not supported for dllm prefill"

                self._add_dllm_req(req, prefix_len)
                self._req_inc_lock_ref(req)
            elif self.rem_chunk_tokens is None or input_tokens <= self.rem_chunk_tokens:
                # Non-chunked prefill — the whole sequence is committed this iter.
                req.set_extend_range(
                    len(req.prefix_indices), len(req.full_untruncated_fill_ids)
                )
                self.can_run_list.append(req)

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    input_tokens,
                    min(
                        req.sampling_params.max_new_tokens,
                        CLIP_MAX_NEW_TOKENS,
                    ),
                    req.retracted_stain,
                    mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                )
            else:
                # Make sure at least one page is available
                trunc_len = self.rem_chunk_tokens // self.page_size * self.page_size

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # When truncation align size is set, we want to assert that the prefill prefix length is multiple of truncation align size
                # A typical use case is when deterministic inference is enabled with flashinfer attention backend,
                # we need the prefill prefix length to be multiple of attention split size
                if truncation_align_size is not None:
                    if trunc_len < truncation_align_size:
                        return AddReqResult.OTHER
                    else:
                        trunc_len = truncation_align_size * (
                            trunc_len // truncation_align_size
                        )

                now_input_len = trunc_len + len(req.prefix_indices)
                now_input_len = now_input_len // self.page_size * self.page_size
                trunc_len = now_input_len - len(req.prefix_indices)

                if trunc_len <= 0:
                    return AddReqResult.OTHER

                # Chunked prefill
                req.set_extend_range(
                    len(req.prefix_indices), len(req.prefix_indices) + trunc_len
                )

                self.can_run_list.append(req)
                self.new_chunked_req = req

                self._req_inc_lock_ref(req)
                self._update_prefill_budget(
                    prefix_len,
                    trunc_len,
                    0,
                    req.retracted_stain,
                    mamba_gap_reserve=self._mamba_gap_budget_for_req(req),
                )

        return self.budget_state()
```

**Comment：**

- `_lock_node` 上下文管理器：准入前 `inc_lock_ref`，失败路径 `dec_lock_ref`，防止 radix 节点被驱逐。
- 分块 prefill 时 `max_new_tokens` 预算扣 0，等最后一块再扣——避免重复预留 decode 空间。
- `ignore_eos` 走 `add_one_req_ignore_eos` 分支，用更复杂的 min-free-token 模拟（见源码 L854–966）。

---

### 2.6 `preempt_to_schedule` — 优先级抢占

**Explain：** 当 `enable_priority_preemption` 开启且 batch 已满，高优先级新请求可抢占低优先级 running 请求。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_policy.py L1143-L1213
    def preempt_to_schedule(self, req: Req, server_args: ServerArgs) -> bool:
        """
        Preempt running requests to serve the new request if the priority threshold is met and token count sum is verified.
        Returns True if preemption was committed, and the new request can be scheduled.
        """
        # Iterate running requests to find preemptible requests
        priority_sign = 1 if server_args.schedule_low_priority_values_first else -1

        # NOTE: A request finishes in two phases:
        #   1) update_finish_state + release_kv_cache  (in process_batch_result)
        #   2) filter out of batch                (in get_next_batch_to_run / update_running_batch)
        # Preemption runs between these two phases (inside get_new_batch_prefill),
        # so running_batch may still contain requests whose KV cache is already freed.
        # We must skip them here to avoid a double-free on release_req.
        valid_running_reqs = (
            r
            for r in self.running_batch.reqs
            if r not in self.preempt_list and not r.finished()
        )

        sorted_valid_running_reqs = sorted(
            valid_running_reqs,
            key=lambda x: (
                x.priority * (-priority_sign),
                -x.time_stats.wait_queue_entry_time,
            ),
        )

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

**Comment：**

- 必须跳过 `finished()` 但尚未 filter 出 batch 的请求——否则 double-free KV。
- 抢占阈值 `priority_scheduling_preemption_threshold` 防止微小 priority 差就抢占。

---

## 3. prefill_delayer.py

### 3.1 初始化与 all_gather 缓冲区

**Code：**

```python
# 来源：python/sglang/srt/managers/prefill_delayer.py L43-L113
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

**Comment：**

- buffer 第三维 5 字段：`prefillable`, `token_watermark_force_allow`, `running_batch`, `max_prefill_bs`, `waiting_queue_len`。
- `skip_first_delayer`：首次 merge batch 时不 delay，避免冷启动 decode batch 永远跑不满。

---

### 3.2 `_negotiate_should_allow_prefill_pure` — 协商逻辑

**Explain：** 三态 global prefillable：`all` / `none` / `mixed`。`all` 时检查 slot 条件或 queue 条件决定是否 delay；`mixed` 时计数 delay 直到 `max_delay_passes`。

**Code：**

```python
# 来源：python/sglang/srt/managers/prefill_delayer.py L136-L299（节选）
    def _negotiate_should_allow_prefill_pure(
        self,
        prev_state: Optional[_State],
        local_prefillable: bool,
        token_usage: float,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> _NegotiateOutput:
        # Compute local states
        local_token_watermark_force_allow = (
            local_prefillable
            and ((x := self._token_usage_low_watermark) is not None)
            and (token_usage < x)
        )

        # Gather global states
        tp0_info = self._gather_info(
            local_prefillable=local_prefillable,
            local_token_watermark_force_allow=local_token_watermark_force_allow,
            running_batch=running_batch,
            max_prefill_bs=max_prefill_bs,
            waiting_queue_len=waiting_queue_len,
        )
        global_prefillable = tp0_info[:, 0]
        global_token_watermark_force_allow = tp0_info[:, 1]
        global_running_batch = tp0_info[:, 2]
        global_max_prefill_bs = tp0_info[:, 3]
        global_waiting_queue_len = tp0_info[:, 4]

        # Compute derived global states
        if global_prefillable.min().item() > 0:
            prefillable_status = "all"
        elif global_prefillable.max().item() == 0:
            prefillable_status = "none"
        else:
            prefillable_status = "mixed"
        global_exists_token_watermark_force_allow = (
            global_token_watermark_force_allow.max().item() > 0
        )
        debug_info = dict(
            input_estimation=prefillable_status,
            num_prefillable=global_prefillable.sum().item(),
            num_token_watermark_force_allow=global_token_watermark_force_allow.sum().item(),
        )

        # Wait accumulated so far, taken from prev_state. Release paths attach
        # this so the wait histograms observe the real value; delay paths leave
        # the defaults (0) since the wait isn't finished and isn't observed.
        wait_info = dict(
            wait_forward_passes=prev_state.delayed_count if prev_state else 0,
            wait_seconds=(
                (time.perf_counter() - prev_state.start_time) if prev_state else 0.0
            ),
        )

        # Compute outputs
        if prefillable_status == "all":
            # Safety valve: low KV usage means GPU is underutilized, skip
            # delay. Mirrors the check in the "mixed" branch.
            if global_exists_token_watermark_force_allow:
                return _NegotiateOutput(
                    next_state=None,
                    output_allow=True,
                    output_reason="token_watermark",
                    **debug_info,
                    **wait_info,
                )

            if not self.enable_dp_attention:
                max_running_requests = (
                    max_running_requests + self.dp_size - 1
                ) // self.dp_size

            global_running_batch_max = int(global_running_batch.max().item())
            global_max_prefill_bs_max = int(global_max_prefill_bs.max().item())
            global_waiting_queue_max = int(global_waiting_queue_len.max().item())

            # Queue-based trigger: delay prefill until the waiting queue
            # reaches queue_min = min(running_req * ratio, max_prefill_bs),
            # capped by a wall-clock timeout to bound worst-case TTFT.
            # Targets workloads where decode requests finish one-at-a-time
            # and fragment prefill into many tiny batches.
            queue_condition = False
            if self._queue_trigger_enabled and global_running_batch_max > 0:
                queue_min_effective = min(
                    int(global_running_batch_max * self._queue_min_ratio),
                    global_max_prefill_bs_max,
                )
                queue_condition = (
                    queue_min_effective > 0
                    and global_waiting_queue_max < queue_min_effective
                )
                if queue_condition and prev_state is not None:
                    elapsed_ms = (time.perf_counter() - prev_state.start_time) * 1000.0
                    if elapsed_ms >= self._max_delay_ms:
                        queue_condition = False

            slot_condition = (
                max_running_requests - global_running_batch_max
                < global_max_prefill_bs_max
            )

            if slot_condition or queue_condition:
                # When the "max_decode_bs - running_bs < max_prefill_bs" condition is met,
                # the first merge_batch causes the decoding to fail to reach the maximum batch size.
                if self.skip_first_delayer:
                    self.skip_first_delayer = False
                    pass
                else:
                    next_state = prev_state or _State()
                    next_state = next_state.bump_delayed_count()
                    return _NegotiateOutput(
                        next_state=next_state,
                        output_allow=False,
                        output_reason="delay",
                        **debug_info,
                    )
            exist_previous_wait = prev_state is not None
            return _NegotiateOutput(
                next_state=None,
                output_allow=True,
                output_reason="wait_success" if exist_previous_wait else "no_wait",
                **debug_info,
                **wait_info,
            )
        elif prefillable_status == "none":
            return _NegotiateOutput(
                next_state=None,
                # It does not matter whether we allow or not, thus we allow for simplicity
                output_allow=True,
                output_reason="",
                **debug_info,
                **wait_info,
            )
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

**Comment：**

- **token_watermark**：KV 使用率低于低水位时强制放行——GPU 闲置时不必 delay。
- **slot_condition**：`(max_running - running_bs) < max_prefill_bs` 说明再塞 prefill 会让 decode batch 凑不满。
- **queue_condition**：等待队列太短时 delay，但 `max_delay_ms` 超时后放行， bound TTFT。

---

### 3.3 `PrefillDelayerSinglePassExecutor`

**Explain：** 每轮 prefill 只协商**一次**；多次 `add_one_req` 复用同一结果。`finalize` 记录 metrics。

**Code：**

```python
# 来源：python/sglang/srt/managers/prefill_delayer.py L331-L368
class PrefillDelayerSinglePassExecutor:
    def __init__(self, prefill_delayer: PrefillDelayer, token_usage: float):
        self._prefill_delayer = prefill_delayer
        self._token_usage = token_usage
        self._result: Optional[_NegotiateOutput] = None

    @property
    def _called(self) -> bool:
        return self._result is not None

    def finalize(self, *, actual_prefill: bool):
        if not self._called:
            self.negotiate_should_allow_prefill(local_prefillable=False)

        _record_single_pass_result(
            actual_execution=actual_prefill,
            output=self._result,
            metrics_collector=self._prefill_delayer._metrics_collector,
        )

    def negotiate_should_allow_prefill(
        self,
        local_prefillable: bool,
        running_batch: int = 0,
        max_prefill_bs: int = 0,
        max_running_requests: int = 0,
        waiting_queue_len: int = 0,
    ) -> bool:
        if not self._called:
            self._result = self._prefill_delayer._negotiate_should_allow_prefill(
                local_prefillable=local_prefillable,
                token_usage=self._token_usage,
                running_batch=running_batch,
                max_prefill_bs=max_prefill_bs,
                max_running_requests=max_running_requests,
                waiting_queue_len=waiting_queue_len,
            )
        return self._result.output_allow
```

**Comment：**

- 若整轮未调用 negotiate（无 waiting 请求），`finalize` 用 `local_prefillable=False` 补一次，保证 metrics 有记录。
- `actual_prefill=ret is not None` 在 Scheduler `get_new_batch_prefill` 末尾传入。

---

## 4. min_free_slots_delayer.py

### 4.1 `should_delay`

**Code：**

```python
# 来源：python/sglang/srt/managers/min_free_slots_delayer.py L28-L41
class MinFreeSlotsDelayer:
    """Delay fresh prefill admissions until at least ``min_free_slots`` running-
    request slots free up, batching them into one admission instead of one at a
    time. Useful when each admission is expensive (e.g. DFlash's draft prefill).

    Per-rank local: running-batch slots are private to each DP rank, so a rank
    with free slots does not wait for a congested peer.
    """

    def __init__(self, min_free_slots: int):
        self._min_free_slots = min_free_slots

    def should_delay(self, *, running_bs: int, num_allocatable_reqs: int) -> bool:
        return running_bs > 0 and num_allocatable_reqs < self._min_free_slots
```

**Comment：**

- `running_bs > 0`：无 running 请求时不 delay（冷启动立即 prefill）。
- `num_allocatable_reqs` 来自 `pp_max_micro_batch_size - running_bs` 与 `req_to_token_pool.available_size()` 的 min。
- Scheduler 在 `_get_new_batch_prefill_raw` **排序之前**调用——与 PrefillDelayer 正交。

---

## 5. 走读小结

| 步骤 | 函数 | 作用 |
|------|------|------|
| 1 | `MinFreeSlotsDelayer.should_delay` | 本地 slot 不足则整轮 skip |
| 2 | `SchedulePolicy.calc_priority` | 排序 waiting_queue |
| 3 | `PrefillAdder.__init__` | 快照 KV/SWA/Mamba 预算 |
| 4 | `PrefillAdder.add_one_req` | 逐个准入 + delayer 协商 |
| 5 | `PrefillAdder.preempt_to_schedule` | 可选抢占 |
| 6 | Scheduler 组装 `can_run_list` → `ScheduleBatch` | 下游执行 |
