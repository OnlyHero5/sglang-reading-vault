---
title: "RadixAttention · 排障指南"
type: troubleshooting
framework: sglang
topic: "RadixAttention"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# RadixAttention · 排障指南

## 读者任务

这篇按症状排障，不再按源码文件顺序展开。每个问题都给出判断、源码入口和验证方式。

## Q1：为什么名字叫 `RadixAttention`，但 prefix tree 不在这个类里？

`RadixAttention.forward` 是模型层 attention adapter。它 reshape K/V，处理 piecewise graph 分叉，然后调用 `get_attn_backend().forward`。tree match 已经在调度阶段完成。

```python
# 来源：sglang/python/sglang/srt/layers/radix_attention.py L127-L153
        if (
            forward_batch.forward_mode.is_extend()
            and get_tc_piecewise_forward_context() is not None
        ):
            if self.qk_head_dim != self.v_head_dim:
                output = q.new_empty((q.shape[0], self.tp_q_head_num * self.v_head_dim))
            else:
                output = torch.empty_like(q)
            if is_in_breakable_cuda_graph():
                breakable_unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
                unified_attention_with_output(
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            return output
        else:
            return get_attn_backend().forward(
                q,
                k,
                v,
                self,
                forward_batch,
                save_kv_cache,
                **kwargs,
            )
```

验证：在 `RadixAttention.forward` 下断点时，看不到 `match_prefix` 调用；要看 prefix hit，断在 `schedule_policy.py:match_prefix_for_req`。

## Q2：同样 prompt 为什么没有命中？

先检查 `extra_key`。LoRA id 会拼进 `extra_key`，而 `child_key` 会把 namespace 编进树边。相同 token 但不同 namespace 会走不同 child。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_batch.py L782-L789
        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id
```

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L198-L208
    def child_key(self, page_size: int = 1):
        """Hashable dict-key for the first ``page_size`` logical units, namespaced by ``extra_key``."""
        t = self.token_ids
        if self.is_bigram:
            if page_size == 1:
                plain = (t[0], t[1])
            else:
                plain = tuple((t[j], t[j + 1]) for j in range(page_size))
        else:
            plain = t[0] if page_size == 1 else tuple(t[:page_size])
        return plain if self.extra_key is None else (self.extra_key, plain)
```

排查顺序：先比对 system prompt token 序列，再比对 `extra_key`、LoRA id、cache salt、动态模板字段；随后检查 `Req.init_next_round_input` 是否因 positional embedding override 主动把 match key 置空，以及 `_compute_max_prefix_len` 是否为采样/EAGLE/会话边界保留了不可复用 token。相同 prompt 只是必要条件，不是充分条件。

## Q3：为什么可复用长度与 tree-owned 长度不一样？

`page_size > 1` 时，tree 只接管完整页面。未对齐 tail 会留在 `req.prefix_indices` 或请求私有 KV 中，下一次 unfinished/finished cache 再释放。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L136-L140
    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]
```

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L533-L548
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
```

验证：同时打印 `len(radix_key)`、`len(kv_indices)`、`len(req.prefix_indices)`、`req.cache_protected_len`。若 `len(prefix_indices) > cache_protected_len`，多出的部分可供下一 chunk 跳过，但仍由请求私有 slot 持有；只有 `cache_protected_len` 以内能用于判断 duplicate-free 下界。差值小于 page size 时通常是 page tail，EAGLE 还可能额外体现 bigram 的 N→N-1。

## Q4：EAGLE 模式为什么会差 1？

EAGLE 把 raw token 变成 bigram 逻辑视图，N 个 raw token 对应 N-1 个 bigram unit。源码用 O(1) 翻转 `is_bigram`，并把 value 截到 logical length。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L142-L153
    def maybe_to_bigram_view(
        self,
        is_eagle: bool,
        value: Optional[torch.Tensor] = None,
    ) -> Tuple[RadixKey, Optional[torch.Tensor]]:
        # O(1): flip the bigram flag instead of materializing a tuple list.
        # value is paired with raw tokens and gets truncated to the bigram count.
        if is_eagle and not self.is_bigram:
            self.is_bigram = True
            if value is not None:
                value = value[: len(self)]
        return self, value
```

验证：在 speculative decode 场景下同时打印 raw token 数和 `len(radix_key)`，不要用普通 token 长度直接判断 tree 长度。

## Q5：请求还在运行，为什么对应节点不能被驱逐？

因为调度进入 prefill 后会对 `last_node` 到 root 的路径加锁。classic cache 中 `lock_ref` 从 0 变 1 时，节点从 evictable 转到 protected。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L592-L626
    def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
        if self.disable:
            return IncLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            self._update_leaf_status(node)
            node = node.parent
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: TreeNode, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable:
            return DecLockRefResult(delta=0)

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            self._update_leaf_status(node)
            if node.parent is None:
                assert (
                    node is self.root_node
                ), "This request holds the node from another tree"
            node = node.parent
        return DecLockRefResult(delta=delta)
```

如果出现异常释放，查两个方向：`req.last_node` 是否来自同一个 tree，`dec_lock_ref` 是否提前或重复调用。

## Q6：为什么 evict 后释放 token 数可能超过请求目标？

classic `evict` 的单位是 leaf node，不是单 token。它从 heap 里弹 leaf，释放整个 `x.value`，然后可能把 parent 也推回 heap。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L563-L590
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

验证：观察 `num_tokens_evicted` 与请求的 `num_tokens`，只要释放不少于目标且没有释放 protected node，就是预期行为。

## Q7：什么时候需要 Unified，而不是 classic `RadixCache`？

当同一前缀需要同时管理 Full KV、SWA、Mamba state，或叠加 host/storage tier、sidecar pool、streaming session 时，需要 Unified。真正的 `ComponentType` 是 Full/SWA/Mamba 等资源视图；HiCache 是 device↔host↔storage 控制面，不是另一种 component。

```python
# 来源：sglang/python/sglang/srt/mem_cache/unified_radix_cache.py L82-L89
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
```

```python
# 来源：sglang/python/sglang/srt/mem_cache/unified_radix_cache.py L626-L657
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

    def dec_lock_ref(
        self,
        node: Any,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        result = self.session.try_dec_lock_ref(node, params)
        if result is not None:
            return result
        if self.disable:
            return DecLockRefResult()
        for component in self._components_tuple:
            if skip_swa and component.component_type == ComponentType.SWA:
                continue
            component.release_component_lock(node=node, params=params)

        self._update_evictable_leaf_sets(node)
        # TODO: delta is not aggregated from components; no caller uses it yet.
        return DecLockRefResult()
```

首次阅读可以把 Unified 看成“同一前缀 key，多套资源状态机”，但要保留三个不对称点：Full 是节点生存骨架并由 leaf set 驱逐；辅助 component 使用独立 device/host LRU；HiCache 下 `best_match_node` 可以深于 `last_device_node`，host 命中要经过 load-back 才能追加到 device `prefix_indices`。

## Q8：piecewise CUDA Graph 下为什么要 narrow `out_cache_loc`？

图复用可能让 tensor 有静态 padding，但 backend 只能写真实 token 对应的位置。`unified_attention_with_output` 会切 query/key/value 和 `out_cache_loc`，调用 backend 后再恢复原 batch 字段。

```python
# 来源：sglang/python/sglang/srt/layers/radix_attention.py L176-L230
    context = get_tc_piecewise_forward_context()
    forward_batch = context.forward_batch
    attention_layers = context.attention_layers
    attention_layer = attention_layers[layer_id]
    real_num_tokens = forward_batch.num_token_non_padded_cpu

    query = query[:real_num_tokens]
    if key is not None:
        key = key[:real_num_tokens]
    if value is not None:
        value = value[:real_num_tokens]

    # DeepSeek MLA has two RadixAttention instances per layer (attn_mqa and
    # attn_mha) that share the same layer_id. The attention_layers list only
    # stores attn_mqa. When the MHA path is active (save_kv_cache=False), use
    # the companion attn_mha so the backend sees correct head/dim metadata.
    if _is_hip and not save_kv_cache and hasattr(attention_layer, "_pcg_mha_companion"):
        attention_layer = attention_layer._pcg_mha_companion

    kwargs = {}
    if q_rope is not None:
        kwargs["q_rope"] = q_rope[:real_num_tokens]
    if k_rope is not None:
        kwargs["k_rope"] = k_rope[:real_num_tokens]
    if sinks is not None:
        kwargs["sinks"] = sinks
    if cos_sin_cache is not None:
        kwargs["cos_sin_cache"] = cos_sin_cache
    if is_neox is not None:
        kwargs["is_neox"] = is_neox
    if llama_4_scaling is not None:
        kwargs["llama_4_scaling"] = llama_4_scaling
    if topk_indices is not None:
        kwargs["topk_indices"] = topk_indices[:real_num_tokens]

    original_out_cache_loc = forward_batch.out_cache_loc
    # Keep the original ForwardBatch object and only narrow cache locations for
    # this backend call so model/backend state is still written to the same batch.
    forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]

    # Store pre-allocated output for FA backend to write directly into.
    # Must slice to real_num_tokens to match the narrowed query shape —
    # the FA kernel validates out.size(0) == q.size(0).
    forward_batch._attn_output = output[:real_num_tokens]

    ret = get_attn_backend().forward(
        query,
        key,
        value,
        attention_layer,
        forward_batch,
        save_kv_cache,
        **kwargs,
    )
    forward_batch.out_cache_loc = original_out_cache_loc
```

排查图路径时，确认三个长度一致：`real_num_tokens`、narrow 后的 query 第一维、narrow 后的 `out_cache_loc` 长度。

## Q9：如何快速判断 miss 是预期还是 bug？

按下面顺序缩小范围：

| 检查 | 预期 | 入口 |
|------|------|------|
| token 序列 | system prompt token 完全一致 | tokenizer 后的 `origin_input_ids` |
| namespace | `extra_key` 一致 | `Req.__init__` |
| page 对齐 | 命中长度向下取整到 page 边界 | `RadixKey.page_aligned` |
| 强制 miss | 开关打开时必 miss | `SGLANG_RADIX_FORCE_MISS` |
| lock | 活跃节点不可驱逐 | `inc_lock_ref` / `dec_lock_ref` |
| finished insert | 在允许 insert 且存在未缓存的 page-aligned suffix 时，tree ownership/size 才应增长；全重复前缀不要求新节点 | `cache_finished_req` |

如果所有检查都符合预期，但 hit rate 仍异常低，再去看上层模板是否包含动态字段，或者调度策略是否让请求没有机会复用相同前缀。
