---
type: batch-doc
module: 15-RadixAttention
batch: "15"
doc_type: faq
title: "RadixAttention：关键问题"
tags:
 - sglang/batch/15
 - sglang/module/radix-attention
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# RadixAttention：关键问题

---

## Q1：RadixAttention 和 RadixCache 名字都有 Radix，是什么关系？

**Explain：** **RadixCache** = 前缀树 **数据结构**（CPU 侧索引管理）。**RadixAttention** = GPU **Attention 算子**包装，与树无直接调用关系。名字均强调「按 token 前缀共享 KV」这一 SGLang 设计理念。

---

## Q2：为什么需要 extra_key？

**Explain：** 相同 token 序列在不同 LoRA adapter、不同 cache salt 下 KV 不同，不能共享节点。`child_key` 把 `extra_key` 编入 dict key。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L155-L160
    def _check_compatible(self, other: RadixKey) -> None:
        if self.extra_key != other.extra_key:
            raise ValueError(
                f"RadixKey operations require matching extra_key, but got "
                f"{self.extra_key=} != {other.extra_key=}"
            )
```

---

## Q3：page_size > 1 时 partial page 怎么处理？

**Explain：** `page_aligned` 截断 key；未对齐 tail 的 KV indices **不进树**，留在 `req.prefix_indices` tail，由 `cache_protected_len` 追踪，在后续 unfinished/finished 调用中 free。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L533-L537
        # The cache_protected_len is not always equal to len(req.prefix_indices)
        # since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree.
        # It should be freed in the next cache_unfinished_req and final cache_finished_req to avoid memory leak.
        # So we introduce this `cache_protected_len` field to make sure the partial part can be freed correctly.
        req.cache_protected_len = len(new_indices)
```

---

## Q4：match 时 split 节点是什么？

**Explain：** 若查找路径停在某节点 **中间**（非 segment 边界），`_match_prefix_helper` 将该节点一分为二，使边界精确对齐，不复制 KV 数据，只拆 key/value tensor 视图。

**Comment：** docstring 见 `match_prefix` L385-L390。

---

## Q5：disable radix cache 时行为？

**Explain：** `match_prefix` 返回空 indices；`cache_finished_req` 直接 free 全部 kv_indices，不 insert。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L444-L449
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return
```

---

## Q6：UnifiedRadixCache 何时必须用？

**Explain：** Hybrid model（Mamba+Attention）、SWA 双 cache、HiCache L3、streaming session 等需要 **多 component** 时必须 Unified。纯 Llama dense 可用经典 `RadixCache`。

---

## Q7：eviction_policy 支持哪些？

**Explain：** 通过 `get_eviction_strategy(eviction_policy)` 注入；常见 LRU、priority-aware。堆排序键 = `eviction_strategy.get_priority(node)`。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L306-L306
        self.eviction_strategy = get_eviction_strategy(self.eviction_policy)
```

---

## Q8：piecewise graph 为何 narrow out_cache_loc？

**Explain：** Graph capture 按 **max batch token** 分配静态 buffer；实际 extend token 可能更少。narrow 后 backend 只写有效 slot，避免越界；调用后恢复 `original_out_cache_loc` 以免影响同 batch 其他层。

**Code：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L211-L230
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

---

## Q9：save_kv_cache=False 何时使用？

**Explain：** Qwen3 aiter fused mRoPE 等 kernel **已写 cache**；DeepSeek MHA companion 在 PCG replay 路径。避免 backend 重复 write。

**Comment：** 见 Models 通用 Qwen3 `forward_prepare_aiter_fused_mrope`。

---

## Q10：RadixCache 与 disaggregation 关系？

**Explain：** Disagg prefill worker 完成 prefill 后 KV 经 bootstrap 传输；prefix tree 可在 decode 侧重建或 match。细节在 PD 分离专题；本模块 API 不变。

---

## Q11：host_value 与 write_through_pending_id？

**Explain：** HiCache 将 device KV 异步 backup 到 host；`write_through_pending_id` 跟踪 in-flight 写。evict device 时若 host 仍有 copy 可仅清 device value（`evicted=True`）。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L234-L236
        # store the host indices of KV cache
        self.host_value: Optional[torch.Tensor] = None
        self.write_through_pending_id: Optional[int] = None
```

---

## Q12：为什么 insert 后还要 match_prefix？

**Explain：** insert 可能 split/merge 改变节点到 indices 的映射；rematch 得到 canonical indices 写回 `req_to_token_pool`，保证与树一致。

**Code：**

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L518-L523
        # The prefix indices could be updated, reuse it
        match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
        new_indices, new_last_node = (
            match_result.device_indices,
            match_result.last_device_node,
        )
```

---

## 验证建议（零基础可试）

1. **操作：** 启动服务后发送两条请求：第一条含 200 token 固定前缀 + 短后缀；第二条复用**完全相同**前缀 + 不同后缀。对比第二条的首 token 延迟（TTFT）。 
 **预期现象：** 第二条 TTFT 显著低于第一条；metrics 中 `cache_hit_rate` 上升。 
 **对应文档节：** [[15-RadixAttention-01-核心概念|01-核心概念 § 用户故事]]、Q2 extra_key、Q5 disable radix

2. **操作：** `export SGLANG_RADIX_FORCE_MISS=1` 后重启，重复步骤 1。 
 **预期现象：** 两条 TTFT 接近，cache hit 为 0——证明收益来自 prefix match 而非 batch 偶然合并。 
 **对应文档节：** §3 match/insert/evict、[[08-SchedulePolicy-01-核心概念|08-SchedulePolicy §3]]

3. **操作：** 在 chat template 的 system 段加入 `{{timestamp}}` 动态字段，并发 10 条请求，观察 hit rate。 
 **预期现象：** `cache_hit_rate` 接近 0，每条 prompt 唯一；去掉动态字段后 hit rate 恢复。 
 **对应文档节：** Q2 extra_key、Q3 partial page、[[15-RadixAttention-02-源码走读|02-源码走读 §2.1 insert]]
