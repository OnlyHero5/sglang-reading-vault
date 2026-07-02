---
type: batch-doc
module: 15-RadixAttention
batch: "15"
doc_type: walkthrough
title: "RadixAttention · 源码走读"
tags:
 - sglang/batch/15
 - sglang/module/radix-attention
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# RadixAttention · 源码走读

> 走读顺序：`RadixKey` → `RadixCache` → `UnifiedRadixCache` → `RadixAttention`

---

## 1. radix_cache.py — RadixKey.match

### 1.1 指数 galloping + 二分找分歧点

**Explain：** 长共享前缀下避免 Python 逐 token 比较；用 slice 比较 + galloping 定位窗口，再二分。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L162-L196
    def match(self, other: RadixKey, page_size: int = 1) -> int:
        """Logical-unit prefix length shared with ``other``. Result is rounded down to ``page_size``."""
        self._check_compatible(other)
        t0, t1 = self.token_ids, other.token_ids
        assert type(t0) is type(t1), (type(t0), type(t1))
        n = min(len(t0), len(t1))

        # Exponential search for the first diverging token: gallop in doubling
        # windows (one C-level slice compare each), then binary-search the window
        # holding the divergence -- no per-token Python loop on long shared prefixes.
        matched_tokens = n
        lo = 0
        step = 1
        while lo < n:
            hi = lo + step if lo + step < n else n
            if t0[lo:hi] != t1[lo:hi]:
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if t0[lo:mid] == t1[lo:mid]:
                        lo = mid
                    else:
                        hi = mid
                matched_tokens = lo
                break
            lo = hi
            step *= 2

        if self.is_bigram:
            matched = max(0, min(matched_tokens - 1, len(self), len(other)))
            return (matched // page_size) * page_size if page_size > 1 else matched

        matched_tokens = min(matched_tokens, len(self), len(other))
        if page_size == 1:
            return matched_tokens
        return (matched_tokens // page_size) * page_size
```

**Comment：**

- galloping 步长指数倍增，长共享前缀下比较次数 O(log n) 而非 O(n)。
- 分歧窗口内再二分精确定位第一个不匹配 token；`is_bigram` 时 matched 长度减 1 对齐 EAGLE KV 布局。

---

## 2. RadixCache 公共 API

### 2.1 `insert`

**Explain：** `insert` 是 prefix cache **写路径**的入口：把本轮 forward 新算出的 KV pool indices 挂到 radix 树上。调用前先 `maybe_to_bigram_view`（EAGLE 投机）与 `page_aligned`（分页对齐），保证 `key` 与 `value` 长度一致。返回的 `prefix_len` 表示树中**已存在**的前缀长度——duplicate 部分会由上层 `free`，避免同一 token 占两份物理 slot。`disable=True` 时短路返回，便于 A/B 对比无 cache 行为。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L415-L435
    def insert(self, params: InsertParams) -> InsertResult:
        if self.disable:
            return InsertResult(prefix_len=0)

        key = params.key
        value = params.value
        priority = params.priority
        chunked = params.chunked

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        else:
            # Debug/test fallback: use token ids themselves as values.
            value = torch.tensor(key.token_ids[: len(key)], dtype=torch.int64)

        prefix_len, last_node = self._insert_helper(
            self.root_node, key, value, priority, chunked
        )
        return InsertResult(prefix_len=prefix_len, last_device_node=last_node)
```

**Comment：**

- `prefix_len` 是已存在于树中的前缀长度；新分配部分才占 pool。
- `_insert_helper` 可能在中间 split 节点，使 segment 边界与 page 对齐。
- EAGLE 模式下 bigram key 对应 `len-1` 个 KV index，需与 `is_eagle` 标志一致。

### 2.2 `cache_unfinished_req` — rematch + lock 迁移

**Explain：** chunked prefill 或 streaming 中间态调用此函数：先把当前 `fill_ids` 对应的 KV indices **insert 进树**，再 **rematch** 得到 canonical indices 写回 `req_to_token_pool`。insert 可能 split/merge 节点，改变 indices 布局，故不能假设 insert 前的 mapping 仍有效。`dec_lock_ref` / `inc_lock_ref` 迁移保护路径，防止 evict 回收活跃请求正在使用的节点。`cache_protected_len` 追踪 page 未对齐 tail，避免 premature free。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L488-L552
    def cache_unfinished_req(self, req: Req, chunked=False):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.get_fill_ids()
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

        # Radix Cache takes one ref in memory pool
        result = self.insert(
            InsertParams(
                key=radix_key,
                value=values,
                chunked=chunked,
                priority=getattr(req, "priority", 0) or 0,
            )
        )
        new_prefix_len = result.prefix_len

        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
        assert len(new_indices) == len(
            radix_key
        ), f"{len(new_indices)=}, {len(radix_key)=}"

        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
            new_indices[req.cache_protected_len :],
        )

        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        # - page_size != 1: there is a partial page at the end, keep the full kv_indices
        # - eagle case: bigram keys will only cache len - 1 kv indices
        if len(new_indices) < len(kv_indices):
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices

        req.last_node = new_last_node
```

**Comment：**

- insert 后必须 rematch，因 split 可能改变 indices 布局。
- `free(kv_indices[cache_protected_len:new_prefix_len])` 释放 insert 发现的 duplicate 段。
- `cache_protected_len` 追踪 page 未对齐尾部，在后续 finished 调用中再挂树或 free。

### 2.3 `evict` — heap + 叶节点

**Explain：** KV pool 不足时 Scheduler 调用 `evict(num_tokens)`。只从 **可驱逐叶节点**（`lock_ref==0`）中选 victim，用 min-heap 按 `eviction_strategy.get_priority` 排序——常见 LRU 或 priority-aware。驱逐叶节点后若父节点变为无子且 `lock_ref==0`，父节点也可能成为新叶并重新入堆，实现**路径向上收缩**而非只删表层。被 evict 的 `node.value` indices 经 allocator `free` 回收到 free list。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L563-L590
    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()

        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        leaves = list(self.evictable_leaves)
        eviction_heap = [
            (self.eviction_strategy.get_priority(node), node) for node in leaves
        ]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_to_kv_pool_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = self.eviction_strategy.get_priority(x.parent)
                heapq.heappush(eviction_heap, (new_priority, x.parent))

            self._record_remove_event(x)

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
```

**Comment：**

- `evictable_leaves` 维护当前可驱逐叶集合；`lock_ref>0` 的节点永不在此集合。
- 父节点变叶后重新入堆，可一次 evict 释放整条短分支。
- 驱逐量 `num_evicted` 可能略大于请求值（按叶节点整段释放），上层需容忍。

---

## 3. unified_radix_cache.py

### 3.1 `UnifiedLRUList` — 双指针 slot

**Explain：** Unified 版本同一 `UnifiedTreeNode` 上挂多种 component（full KV、SWA、Mamba）。device LRU 与 host LRU **共用 node 对象但用不同 slot offset**（`_pt = component_type + offset`），避免 LRU 链表指针在同一 node 上碰撞。每个 component 独立 LRU 链，evict 时可只清 device value 而 host 仍保留 copy（HiCache write-back）。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/unified_radix_cache.py L136-L151
class UnifiedLRUList:
    def __init__(
        self,
        component_type: ComponentType,
        tree_components: tuple[ComponentType, ...],
        use_host_ptr: bool = False,
    ):
        self.component_type = component_type
        # Pointer slot: host LRU uses offset slots so device/host pointers
        # never collide on the same node.
        self._pt: int = component_type + (_NUM_COMPONENT_TYPES if use_host_ptr else 0)
        self.head = UnifiedTreeNode(tree_components)
        self.tail = UnifiedTreeNode(tree_components)
        self.head.lru_next[self._pt] = self.tail
        self.tail.lru_prev[self._pt] = self.head
        self.cache: dict[int, UnifiedTreeNode] = {}
```

### 3.2 `match_prefix` — StreamingSession 短路

**Explain：** 流式或多轮 fill 场景下，`StreamingSession.try_match_prefix` 可返回 session 私有视图，避免与全局树状态竞态。无 active session 时 `try_*` 立即 fall through 到 `_match_prefix_helper`。post-processor 还会处理 HiCache host 命中、prefetch queue 等——Unified API 兼容经典 RadixCache 但扩展 storage 层。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/unified_radix_cache.py L561-L586
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        result = self.session.try_match_prefix(params)
        if result is not None:
            return result

        key = params.key
        key, _ = key.maybe_to_bigram_view(self.is_eagle)
        if self.disable or len(key) == 0:
            return self._empty_match_result
        key = key.page_aligned(self.page_size)
        if len(key) == 0:
            return self._empty_match_result

        (
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        ) = self._match_prefix_helper(key)
        return self._match_post_processor(
            params,
            value,
            best_match_node,
            best_match_device_node,
            best_match_device_value_len,
        )
```

**Comment：**

- `StreamingSession` 对 streaming 请求维护独立视图，无 session 时 try_* 立即 fall through。
- `_match_prefix_helper` 与经典 RadixCache 共享 galloping + split 逻辑。
- post-processor 可能触发 host load-back 或 L3 prefetch（HiCache 路径）。

### 3.3 `evict` — 多 component 驱动

**Explain：** Unified evict 不按单一 heap 扫叶，而是**每个 TreeComponent 驱动自己的 eviction**（full KV、SWA 窗口、Mamba slot 计数独立）。`tracker` 汇总各 component 释放量；write-back 策略下 evict device 前可能先 `writing_check` 刷 host。返回 `EvictResult` 含 `swa_num_tokens_evicted`、`mamba_num_evicted` 等分项，供 Scheduler 分层预算决策。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/unified_radix_cache.py L604-L624
    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        start_time = time.perf_counter()
        tracker = {ct: 0 for ct in self.tree_components}

        for component in self._components_tuple:
            component.drive_eviction(params=params, tracker=tracker)

        if (
            self.cache_controller is not None
            and self.cache_controller.write_policy == "write_back"
        ):
            self.writing_check(write_back=True)

        self.update_eviction_metrics(sum(tracker.values()), start_time)
        return EvictResult(
            num_tokens_evicted=tracker[BASE_COMPONENT_TYPE],
            swa_num_tokens_evicted=tracker.get(ComponentType.SWA, 0),
            mamba_num_evicted=tracker.get(ComponentType.MAMBA, 0),
        )
```

### 3.4 `inc_lock_ref` — 多 component 锁

**Explain：** 经典 RadixCache 从叶向 root 递增 `lock_ref`，`lock_ref==0→1` 时节点从 evictable 转入 protected。Unified 版本对每个 component 调用 `acquire_component_lock`，StreamingSession 也可短路处理 session 内节点。锁释放后 `_update_evictable_leaf_sets` 刷新叶集合，保证 heap evict 看见最新状态。活跃 decode 请求通过 `req.last_node` 持有锁，abort/finish 时 `dec_lock_ref`。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/unified_radix_cache.py L626-L637
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        result = self.session.try_inc_lock_ref(node)
        if result is not None:
            return result
        if self.disable:
            return IncLockRefResult()
        result = IncLockRefResult()
        for component in self._components_tuple:
            result = component.acquire_component_lock(node=node, result=result)

        self._update_evictable_leaf_sets(node)
        return result
```

**Comment：**

- 多 component 需全部 acquire 成功，否则 partial lock 由 component 内部回滚。
- `session.try_inc_lock_ref` 避免 streaming 路径与全局树锁语义冲突。
- 与经典版相同：`lock_ref>0` 节点不可 evict，保护正在 decode 的前缀路径。

---

## 4. radix_attention.py

### 4.1 `AttentionType` 枚举

**Explain：** 区分 decoder、coder-only、双向 decoder 等 attention mask 语义。使用 string enum 而非 IntEnum，是为兼容 `torch.compile` 对枚举值的序列化限制——compile 路径下 attention 层需稳定字符串 tag。

**Code：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L43-L54
class AttentionType(Enum):
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Decoder bidirectional attention between image tokens
    DECODER_BIDIRECTIONAL = "decoder_bidirectional"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"
```

**Comment：**

- 用 string enum 兼容 `torch.compile` 对常量折叠的限制。
- `DECODER_BIDIRECTIONAL` 用于需全序列可见的特殊层（非标准 causal LM）。

### 4.2 Q/K/V reshape

**Explain：** Attention forward 前把 flat hidden states reshape 为 `[num_tokens, num_heads, head_dim]`，供 backend kernel 消费。MLA（Multi-head Latent Attention）路径可能已通过 kwargs 传入 `k_rope`，跳过此处 view；cross-layer KV share 时 k/v 可为 None，由下层直接从 cache 读。

```python
# 来源：python/sglang/srt/layers/radix_attention.py L118-L125
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)
```

**Comment：** MLA 路径可能传 `k_rope` 分离 layout；cross-layer KV share 时 k/v 可为 None。

### 4.3 `unified_attention_with_output` — DeepSeek MLA 双实例

**Code：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L188-L193
    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion
```

**Comment：** 同一 `layer_id` 下 attn_mqa 与 attn_mha 共用 id；MHA 路径需 companion 的 head metadata。

### 4.4 HIP padding zero（PCG replay）

**Code：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L235-L252
    if _is_hip:
        # During PCG replay on AMD, varlen attention kernels only fill positions
        # 0..actual_tokens-1 and leave padded positions with uninitialized
        # garbage from torch.empty.  Zero these so garbage (NaN/Inf) does not
        # propagate through residual connections, MoE routing, and allreduce.
        # Use context.raw_num_tokens (pre-padding count from PCG runner)
        # instead of forward_batch.extend_num_tokens, because
        # extend_num_tokens is None for TARGET_VERIFY (EAGLE) batches.
        pcg_static_tokens = context.num_tokens
        actual_tokens = context.raw_num_tokens
        if (
            pcg_static_tokens is not None
            and actual_tokens is not None
            and pcg_static_tokens > actual_tokens
        ):
            first_dim = output.shape[0]
            elems_per_token = output.numel() // first_dim
            output.view(first_dim, elems_per_token)[actual_tokens:].zero_()
```

**Comment：** 防止 padded slot 的 NaN 进入 residual / MoE routing。

### 4.5 `@register_custom_op` + `@register_split_op`

**Code：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L156-L158
@register_custom_op(mutates_args=["output"])
@register_split_op()
def unified_attention_with_output(
```

**Comment：** piecewise graph 编译器据此拆分 attention 子图。

---

## 5. RadixCache.reset

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L331-L336
    def reset(self):
        # Initialize root with minimum priority so any real priority overrides it
        self.root_node = TreeNode(priority=-sys.maxsize)
        self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
        self.root_node.value = []
        self.root_node.host_value = []
```

**Comment：** `flush_cache` RPC 触发 reset，释放整树（需无 lock）。
