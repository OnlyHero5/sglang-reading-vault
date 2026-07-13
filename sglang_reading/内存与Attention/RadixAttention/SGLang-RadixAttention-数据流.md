---
title: "RadixAttention · 数据流"
type: dataflow
framework: sglang
topic: "RadixAttention"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/dataflow
  - source-reading
updated: 2026-07-11
---
# RadixAttention · 数据流

## 读者任务

这篇只追对象状态，不按函数顺序重复 [[SGLang-RadixAttention-源码走读]]。目标是能回答：一个 token 前缀从逻辑序列变成 KV pool indices，再变成 attention backend 可读写的 `out_cache_loc`，中间到底经过哪些字段。

## 对象生命周期

| 阶段 | 逻辑对象 | 物理对象 | 关键字段 |
|------|----------|----------|----------|
| 请求进入 | token 序列 + namespace | 无 | `origin_input_ids`、`output_ids`、`extra_key` |
| prefix match | `RadixKey` | tree node 的 `value` | `device_indices`、`last_device_node` |
| 调度记录 | `Req` | match 后的 device hit，或 chunk commit 后“tree prefix + 私有 tail” | `prefix_indices`、`last_node`、`cache_protected_len` |
| extend batch | tail token | 新分配 KV slot | `extend_range`、`out_cache_loc` |
| attention forward | Q/K/V tensor | paged KV cache | `forward_batch.out_cache_loc` |
| cache commit | page-aligned prefix | tree 持有的 KV indices | `insert`、`req_to_token_pool.write` |

## 1. namespace 在 `Req` 初始化时已经确定

LoRA id 会拼进 `extra_key`。因此两个请求即使 token 完全相同，只要 LoRA namespace 不同，`RadixKey.child_key` 就不会落到同一条 tree edge。

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

这个字段属于请求，不属于 tree。tree 只消费它，不负责推断请求为什么需要隔离。

## 2. prefix match 写回 `Req`

调度入口把 `MatchResult` 的几个字段拆开写入请求对象。这里形成了后续所有模块共享的契约。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_policy.py L111-L136
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
    if match_result.mamba_branching_seqlen is not None:
        req.mamba_branching_seqlen = match_result.mamba_branching_seqlen
    if match_result.cache_protected_len is not None:
        req.cache_protected_len = match_result.cache_protected_len
    return match_result
```

在这个刚完成 match 的时点，`prefix_indices` 是已经在 device tree 上可用的 KV indices；`host_hit_length` 是 device hit 之后还能在 host 层命中的长度；`last_node` 是 device lock 与后续 unfinished/finished cache 的锚点。不要把这一定义无条件延伸到 chunk commit 之后。

## 3. admission 把 prefix 长度变成 extend 区间

chunked 和 non-chunked 都以 `len(req.prefix_indices)` 作为本轮 extend 起点。这个字段越长，进入模型的新 token 越少。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_policy.py L815-L821
        cand_extend_input_len = len(req.full_untruncated_fill_ids) - len(
            req.prefix_indices
        )
        truncated = cand_extend_input_len > _rem_tokens
        new_len = min(cand_extend_input_len, _rem_tokens)
        req.set_extend_range(len(req.prefix_indices), len(req.prefix_indices) + new_len)
        self.can_run_list.append(req)
```

这一步把 prefix cache 的命中从“索引结果”变成“本轮计算范围”。如果你只看 attention backend，会看不到这个性能来源。

## 4. batch 组装只把 tail 放进 forward 输入

`prepare_for_extend` 再次使用 `len(prefix_indices)`。它一边切 `input_ids`，一边把 `prefix_lens`、`extend_lens`、`seq_lens` 写成 batch 字段，让 allocator 和 backend 都能对齐请求维度。

```python
# 来源：sglang/python/sglang/srt/managers/schedule_batch.py L2018-L2058
        # Init tensors
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
        extend_logprob_start_lens = [
            compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = is_pin_memory_available(self.device)
        # Stay on pinned CPU; H2D is deferred to forward stream via
        # resolve_forward_inputs.
        pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(
            self
        )
```

注意这里还没有跑 attention。`out_cache_loc` 是新 token 的写入位置，已经命中的 prefix KV 不需要重新写。

## 5. forward batch 把 `out_cache_loc` 交给 attention backend

piecewise CUDA Graph 路径会临时 narrow `forward_batch.out_cache_loc` 到真实 token 数，然后再恢复原对象。这说明 `out_cache_loc` 是 backend 写 KV 的直接入口。

```python
# 来源：sglang/python/sglang/srt/layers/radix_attention.py L211-L230
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

排查错写 KV slot 时，顺序应是：先看 `prepare_for_extend` 的 `out_cache_loc`，再看 attention backend 对这个字段的消费，不要回到 `RadixCache.match_prefix` 猜 kernel 行为。

## 6. unfinished cache 会改写 req pool 的 canonical indices

中途缓存之后，请求的 req pool 需要从“请求私有 slot”切到“tree 持有的规范 slot”。这就是 `req_to_token_pool.write` 的作用。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L518-L552
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

这个状态迁移有两个容易误读的点：`cache_protected_len` 表示 tree 接管到哪里；`prefix_indices` 表示下一块可以跳过到哪里，因此可能还附带未进 tree 的 tail。前者决定 duplicate/tail 的释放下界，后者决定下一轮 `extend_range.start`。

## 7. Unified 把一棵树扩展成多 component 生命周期

classic tree 的 `value` 是一个 device KV indices；Unified node 则有多份 component data。对应地，lock 不再只是一个 `lock_ref` 计数，而是对所有启用 component 调用 acquire/release。

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

所以 Unified 的数据流不是“另一套 token namespace”，而是“同一前缀 key 下，由多个 component 对可用边界达成共识，并分别维护 device/host 生命周期”。Full 使用 leaf set，辅助 component 使用 LRU；HiCache 还会让 `best_match_node` 深于 `last_device_node`，随后由 load-back 补齐 device indices。

```python
# 来源：sglang/python/sglang/srt/mem_cache/unified_radix_cache.py L986-L1002
        last_host_node = (
            best_match_node
            if self.cache_controller is not None
            else best_match_device_node
        )

        if best_match_device_value_len > 0:
            device_indices = torch.cat(value[:best_match_device_value_len])
        else:
            device_indices = self._empty_match_result.device_indices
        result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_match_device_node,
            last_host_node=last_host_node,
            best_match_node=best_match_node,
            host_hit_length=0,
        )
```

## 交互矩阵

| 读写方 | 写入 | 读取 |
|--------|------|------|
| `match_prefix_for_req` | `req.prefix_indices`、`req.last_node`、host hit 字段 | `req.origin_input_ids`、`req.output_ids`、`req.extra_key` |
| `PrefillAdder` | `req.extend_range`、tree lock | `len(req.prefix_indices)`、预算、host hit |
| `ScheduleBatch.prepare_for_extend` | `input_ids`、`prefix_lens`、`extend_lens`、`out_cache_loc` | `req.prefix_indices`、`req.extend_range` |
| `RadixAttention.forward` | backend output、KV write side effect | `forward_batch.out_cache_loc`、`forward_mode` |
| `cache_unfinished_req` | tree、req pool、`req.prefix_indices`、`req.last_node` | `req.get_fill_ids()`、req pool old indices |
| `cache_finished_req` | tree、allocator free、lock release | committed KV length、`cache_protected_len` |

## 状态自检

排查时按这个顺序问：

1. token 和 `extra_key` 是否真的相同。
2. `match_prefix_for_req` 后 `len(req.prefix_indices)` 是否符合预期。
3. `set_extend_range` 的 start 是否等于命中长度。
4. `prepare_for_extend` 的 `input_ids` 是否只包含 tail。
5. `out_cache_loc` 长度是否等于本轮 extend token 数。
6. chunked/finished 后 duplicate 和 tail 是否被释放。
7. `inc_lock_ref` 和 `dec_lock_ref` 是否对同一棵 tree 的节点成对发生。

## 运行验证

RadixAttention 数据流的源码级验证要覆盖 prefix match、prefill 准入、extend 准备、attention 写 KV、unfinished/finished cache 回写，以及 UnifiedRadixCache 的 component lock。

```powershell
rg -n 'def match_prefix_for_req|match_prefix|def add_chunked_req|def prepare_for_extend|class RadixAttention|def forward|def cache_unfinished_req|def cache_finished_req|class UnifiedRadixCache|def inc_lock_ref|def dec_lock_ref|prefix_indices|last_node|out_cache_loc' sglang/python/sglang/srt/managers/schedule_batch.py sglang/python/sglang/srt/managers/schedule_policy.py sglang/python/sglang/srt/layers/radix_attention.py sglang/python/sglang/srt/mem_cache/radix_cache.py sglang/python/sglang/srt/mem_cache/unified_radix_cache.py
```

读输出时重点确认 `prefix_indices`、`last_node`、`out_cache_loc` 仍贯穿调度、attention 和 cache 回写。它们是本文所有状态自检问题的共同支点。
