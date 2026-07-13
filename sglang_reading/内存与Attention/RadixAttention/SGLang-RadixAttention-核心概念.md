---
title: "RadixAttention · 核心概念"
type: concept
framework: sglang
topic: "RadixAttention"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# RadixAttention · 核心概念

## 读者任务

这篇先不追完整调用链，只建立四个对象的边界：`Req` 带着 token 和 cache namespace 进入调度，`RadixKey` 把它变成可查找的树 key，`TreeNode` 持有前缀段对应的 KV pool indices，`RadixAttention` 在模型层把 QKV 交给 attention backend。

最重要的纠偏是：名字里的 `RadixAttention` 容易让人以为一棵 radix tree 直接在 attention 层工作。源码不是这样。tree match 发生在调度侧；attention 层只消费已经准备好的 `ForwardBatch`。

## 先建立模型

| 类比 | 源码对象 | 失效边界 |
|------|----------|----------|
| 目录卡 | `RadixCache` / `UnifiedRadixCache` | 不能解释 kernel 内部如何算 softmax |
| 货架编号 | `torch.int64` KV pool indices | 不等于 K/V tensor 值本身 |
| 取货单 | `req.prefix_indices` | 描述下一轮可跳过的已计算 KV；chunk commit 后不保证全部已进 tree |
| 工位 | `RadixAttention.forward` | 不负责决定哪些前缀可共享 |

用这张表读源码时，先问“这行代码是在改目录、改取货单、改货架编号，还是把新货交给工位”。这样能避免把 prefix cache、KV allocator、attention backend 混在一起。

## `RadixKey`：同样 token，也要看 namespace

`RadixKey` 的 key 不是单纯 token 序列。`extra_key` 会和 token 一起参与 child dict key，这就是 LoRA adapter、cache salt 或其他隔离语义不会串 cache 的根本原因。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L60-L80
class RadixKey:
    """is_bigram=True: token_ids holds raw tokens (N+1 for N bigrams); slices share one boundary token."""

    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(
        self,
        token_ids: array[int],
        extra_key: Optional[str] = None,
        is_bigram: bool = False,
        limit: Optional[int] = None,
    ):
        # token ids sequence (raw ints in both modes)
        self.token_ids = token_ids
        # extra key (e.g. lora_id, cache_salt)
        self.extra_key = extra_key
        # bigram view over token_ids: length = max(0, len(token_ids) - 1)
        self.is_bigram = is_bigram
        # Optional cap on raw tokens: behave as if token_ids were sliced to
        # token_ids[:limit], without the O(n) copy. None = use all tokens.
        self.limit = limit
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

读这两段时要抓住两个判断：`extra_key` 是 namespace，不是注释字段；`page_size` 也参与 child key 的粒度，说明 prefix cache 的边界必须贴合 paged KV 的边界。

## `Req`：缓存命中结果落在请求对象上

调度侧不是把命中结果直接传给 attention layer，而是写回 `Req`。`prefix_indices`、`last_node`、`host_hit_length`、`cache_protected_len` 是后续 admission、forward、evict 共同依赖的请求状态。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_batch.py L782-L871
        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id
        self.routing_key = routing_key

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # shape (2)
        self.mamba_next_track_idx: Optional[int] = None  # 0 or 1
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # seq len of the last cached mamba state
        )
        # the branching point seqlen to track mamba state. If set, given by prefix match,
        # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
        self.mamba_branching_seqlen: Optional[int] = None
        # Deferred COW: source mamba pool index from radix cache node (copy on forward stream)
        self.mamba_cow_src_index: Optional[torch.Tensor] = None
        # Deferred clear: newly allocated mamba slot needs zeroing on forward stream
        self.mamba_needs_clear: bool = False
        # Lazy extra buffer: skip radix cache insert when prealloc failed at
        # boundary — the forward overwrites the only slot, corrupting the state.
        self.mamba_lazy_is_insert: bool = True

        # Check finish
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # For multimodal inputs
        self.multimodal_inputs: Optional[MultimodalInputs] = None
        # Pre-computed multimodal prompt token counts; populated on the prefill
        # node and transferred to decode via the metadata buffer in disagg (PD) mode.
        self.mm_image_tokens: int = 0
        self.mm_audio_tokens: int = 0
        self.mm_video_tokens: int = 0

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # TODO(ispobock): rename to last_device_node
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None
        # Per-component host hit lengths split off from host_hit_length:
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        # Total cached prefix length (on-device prefix_indices + host_hit_length),
        # capped at the max allowed prefix. Set during prefix matching at schedule
        # time and used to estimate uncached tokens / sort by longest prefix for
        # load reporting.
        self.num_matched_prefix_tokens = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None
        # Whether the prefill-time SWA tree lock has been released early
        self.swa_prefix_lock_released: bool = False
        # The prefix length that is inserted into the tree cache
        self.cache_protected_len: int = 0
```

这段把读者需要追的字段集中放在一起：刚完成 `match_prefix_for_req` 时，`prefix_indices` 是 device tree hit；完成一次 `cache_unfinished_req` 后，它可能变成“tree 的 canonical indices + 请求私有 tail”。`last_node` 是锁锚点，`cache_protected_len` 才是 tree 真正接管的长度。三者不总是同长，尤其是 page tail 和 EAGLE bigram 场景。

## `TreeNode`：树节点保存前缀段和 KV indices

classic `TreeNode` 的 `key` 是一段逻辑前缀，`value` 是这段前缀对应的 KV pool index tensor。`lock_ref` 控制这段路径是否可被 evict。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L217-L240
class TreeNode:

    counter = 0

    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: RadixKey = None
        self.value: Optional[torch.Tensor] = None
        self.lock_ref = 0
        self.last_access_time = time.monotonic()
        self.creation_time = time.monotonic()

        self.hit_count = 0
        # indicating the node is locked to protect from eviction
        # incremented when the node is referenced by a storage operation
        self.host_ref_counter = 0
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None
        # store hash values of each pages
        self.hash_value: Optional[List[str]] = None
        # priority for priority-aware eviction
        self.priority = priority
```

这个结构解释了为什么 evict 通常从 leaf 做起：内部节点可能还是其他请求前缀的共享段，删错会破坏另一条路径。

## page 对齐：tree 只接管完整页面

`page_aligned` 直接把 key 截断到 page 边界。未对齐的 tail 不是丢了，而是还留在请求侧，后续由 `cache_protected_len` 和 req pool 释放逻辑处理。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L136-L153
    def page_aligned(self, page_size: int) -> RadixKey:
        if page_size == 1:
            return self
        aligned_len = len(self) // page_size * page_size
        return self[:aligned_len]

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

这里还能看到 EAGLE 的分叉：bigram 模式改变的是 logical unit 数量，不是复制一份 tuple 列表。

## lock：活跃路径从可驱逐变成受保护

请求被调度进入 prefill 后，会持有 `last_node` 到 root 的锁。classic cache 用 `evictable_size_` 与 `protected_size_` 在 lock/ref 变化时转移计数。

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

如果你在排查“请求还在跑但 KV 被释放”，第一反应应是检查 `last_node` 是否来自同一棵 tree，以及 `inc_lock_ref` 和 `dec_lock_ref` 是否成对。

## `RadixAttention`：attention 入口，不是 prefix tree

`RadixAttention.forward` 只 reshape K/V，处理 piecewise graph 分叉，然后把工作转给当前 attention backend。它没有调用 `RadixCache.match_prefix`，也没有遍历 `TreeNode`。

```python
# 来源：sglang/python/sglang/srt/layers/radix_attention.py L109-L153
    def forward(
        self,
        q,
        k,
        v,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if k is not None:
            # For cross-layer sharing, kv can be None
            assert v is not None
            if "k_rope" not in kwargs:
                k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
                v = v.view(-1, self.tp_v_head_num, self.v_head_dim)
            else:
                k = k.view(-1, self.tp_k_head_num, self.v_head_dim)

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

所以命名上的“Radix”更像系统设计背景：attention 使用的 paged KV 位置可能来自 radix prefix cache，但这一层自己不查树。

## Unified 只是同一前缀树上的多套资源视图

`UnifiedTreeNode` 把 Full、SWA、Mamba 等 component 的 device/host value 放到 `component_data`。HiCache 不是第四种 `ComponentType`：它是围绕这些 component 增加 host/storage 层、异步 transfer、sidecar pool 与 load-back 的控制面。

各 component 也不是完全对称：Full KV 是树的生存骨架，device/host 驱逐用 leaf set 与时间/策略；SWA、Mamba 等辅助 component 维护独立 device/host LRU。节点能否继续存在最终看 Full 是否至少在 device 或 host 一层存活；辅助数据不能脱离 Full 单独悬挂。

```python
# 来源：sglang/python/sglang/srt/mem_cache/unified_radix_cache.py L269-L273
COMPONENT_REGISTRY: dict[ComponentType, type[TreeComponent]] = {
    ComponentType.FULL: FullComponent,
    ComponentType.MAMBA: MambaComponent,
    ComponentType.SWA: SWAComponent,
}
```

```python
# 来源：sglang/python/sglang/srt/mem_cache/unified_radix_cache.py L1409-L1440
    def _is_device_leaf(self, node: UnifiedTreeNode) -> bool:
        """D-leaf: Full device value present, no child with Full KV on device,
        unlocked, not root.

        Only the Full (base) component is required; auxiliary components
        (Mamba, SWA) are not mandatory for D-leaf membership."""
        ct = BASE_COMPONENT_TYPE
        if node is self.root_node or node.evicted:
            return False
        if any(cd.lock_ref > 0 for cd in node.component_data):
            return False
        if any(
            child.component_data[ct].value is not None
            for child in node.children.values()
        ):
            return False
        return True

    def _is_host_leaf(self, node: UnifiedTreeNode) -> bool:
        """H-leaf: evicted, Full host value present, no children, unlocked, not root.

        Only the Full (base) component host_value is required; auxiliary
        components are not mandatory for H-leaf membership."""
        if node is self.root_node or not node.evicted:
            return False
        if not node.backuped:
            return False
        if any(cd.host_lock_ref > 0 for cd in node.component_data):
            return False
        if len(node.children) > 0:
            return False
        return True
```

```python
# 来源：sglang/python/sglang/srt/mem_cache/unified_radix_cache.py L82-L115
        self.children = defaultdict(partial(UnifiedTreeNode, tree_components))
        self.parent: UnifiedTreeNode | None = None
        self.key: Optional[RadixKey] = None
        self.tree_components = tree_components
        # list indexed by ComponentType (int enum 0..N-1)
        self.component_data: list[ComponentData] = [
            ComponentData() for _ in range(_NUM_COMPONENT_TYPES)
        ]
        self.last_access_time = get_and_increase_time_counter()
        self.creation_time = get_and_increase_time_counter()
        self.hash_value = None
        self.hit_count = 0
        self.priority = priority
        self.lru_prev: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.lru_next: list[UnifiedTreeNode | None] = [None] * (
            _NUM_COMPONENT_TYPES * 2
        )
        self.id = UnifiedTreeNode.counter
        UnifiedTreeNode.counter += 1
        self.write_through_pending_id: Optional[int] = None

    def component(self, component_type: ComponentType) -> ComponentData:
        return self.component_data[component_type]

    @property
    def backuped(self) -> bool:
        """Tree-level: Full KV present on host."""
        return self.component_data[ComponentType.FULL].host_value is not None

    @property
    def evicted(self) -> bool:
        """Tree-level: Full KV not on device (non-root with value=None)."""
```

初读时可以把 Unified 当作 classic key/tree 语义的扩展，但不能把它理解成“给 classic node 多放几个 value”这么简单：它还区分 best device match 与 best device-or-host match、component consensus、device/host 两套锁、Full leaf set、aux LRU、tombstone、session 和异步 D↔H/storage 生命周期。

---

## 运行验证

维护本文时，先用下面的命令确认 Radix 主线仍在这些对象上：

```powershell
rg -n "class RadixKey|class TreeNode|def match_prefix|def cache_finished_req|def dec_lock_ref|class RadixAttention|class UnifiedTreeNode|class ComponentData" sglang/python/sglang/srt/mem_cache/radix_cache.py sglang/python/sglang/srt/mem_cache/unified_radix_cache.py sglang/python/sglang/srt/managers/schedule_batch.py sglang/python/sglang/srt/layers/radix_attention.py
```

预期信号：

- `radix_cache.py` 仍能找到 key、tree node、prefix match、finished request cache 和 lock ref。
- `radix_attention.py` 仍能说明 attention 层只是消费 backend 与 KV pool，不直接查 prefix tree。
- `unified_radix_cache.py` 仍能找到 unified tree 和 component data。

如果 `RadixAttention` 开始直接访问 tree，或 unified cache 不再复用同一前缀树语义，本篇的“tree 与 attention 分层”需要重写。
