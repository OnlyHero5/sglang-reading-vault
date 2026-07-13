---
title: "KV-Cache · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "KV-Cache"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# KV-Cache · 排障指南

## 读者任务

这一篇按排障场景组织。每个问题都先给症状，再给源码入口和验证方法。重点不是背参数，而是知道出错时从哪条边界切进去：Python 参数、C++ 校验、params 状态、kernel 地址推进，还是测试矩阵。

## 症状 1：以为 KV cache API 可以用于训练反向

现象：你把 `flash_attn_with_kvcache` 放到需要 backward 的训练图里，或者想用它替代长上下文训练 attention。

源码入口：Python docstring 明确写出 local attention 语义后，紧接着说明这条路径不支持 backward。

```python
# 定位：flash_attn/flash_attn_interface.py L1542-L1546（接口摘要；精确证据见文末）
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Note: Does not support backward pass.
```

可能原因与判断：这不是“缺少一个 backward kernel”这么简单，而是 API 语义包含 in-place cache update、cache remap、paged KV 和 SplitKV，服务对象是 incremental decoding。

操作：运行 `rg -n 'Does not support backward' flash-attn/flash-attention/flash_attn/flash_attn_interface.py`。预期在 KV-cache API docstring 中命中；若任务需要梯度，应切回普通 dense/varlen forward + backward，而不是给这条原地更新接口补训练语义。

## 症状 2：paged KV 一启用就报 page block size

现象：传入 `block_table` 后报 `Paged KV cache block size must be divisible by 256`。

源码入口：C++ 从 `kcache.size(1)` 取 `page_block_size`，并在进入 kernel 前要求它是 256 的倍数。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L1264-L1268（约束摘要；精确证据见文末）
const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
const int num_blocks = !paged_KV ? 0 : kcache.size(0);
const int page_block_size = !paged_KV ? 1 : kcache.size(1);
TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
```

可能原因与判断：page size 不是 runtime 可以随意选择的内存管理参数，它也是 attention backend 的 kernel 约束。

操作：在调用前打印 `k_cache_paged.shape` 与 `block_table.shape`。预期 `shape[1] % 256 == 0`，且 `block_table.shape[1]` 足以覆盖本次最大逻辑长度；仅修 page size 而 block table 不足，仍可能越界读写。测试里当前启用的 paged 参数是 `None` 和 `256`。

```python
# 来源：tests/test_flash_attn.py L1878-L1880
@pytest.mark.parametrize("paged_kv_block_size", [None, 256])
# @pytest.mark.parametrize("paged_kv_block_size", [256, 512])
# @pytest.mark.parametrize("paged_kv_block_size", [None])
```

## 症状 3：同时传 `block_table` 和 `cache_batch_idx` 报错

现象：你想用 paged KV，又想通过 `cache_batch_idx` 做 batch remap。

源码入口：C++ 明确禁止二者同时存在。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L1247-L1254（互斥摘要；精确证据见文末）
at::Tensor block_table;
const bool paged_KV = block_table_.has_value();
if (paged_KV) {
    TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
    block_table = block_table_.value();
    CHECK_DEVICE(block_table);
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
    TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
```

可能原因与判断：`cache_batch_idx` 和 `block_table` 都在解决“当前 batch 到物理 cache”的映射。一个按 dense batch slot remap，一个按逻辑 block 查物理 block；同时启用会让物理地址解释不唯一。

操作：记录本次调用的 `block_table is not None` 与 `cache_batch_idx is not None`。预期二者不能同时为真；paged KV 场景由 `block_table` 表达物理 block，不再额外传 dense slot remap。

## 症状 4：`cache_leftpad` 和 paged KV 不能同时用

现象：dense cache 带 leftpad 工作正常，换成 paged KV 后同一参数组合报错。

源码入口：leftpad 分支在 C++ 入口直接禁止 paged KV。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L1398-L1405（leftpad 分支摘要；精确证据见文末）
if (leftpad_k_.has_value()) {
    TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
    auto leftpad_k = leftpad_k_.value();
    TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
    CHECK_DEVICE(leftpad_k);
    CHECK_CONTIGUOUS(leftpad_k);
    CHECK_SHAPE(leftpad_k, batch_size);
    params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
```

可能原因与判断：leftpad 是 dense cache 内的逻辑起点偏移；paged KV 已经用 block table 解释逻辑位置到物理 block。当前实现没有合并这两种地址语义。

操作：先在调用层打印 `cache_leftpad` 与 `block_table` 是否存在，再静态定位测试跳过条件。预期 paged 模式下 `cache_leftpad is None`；否则 C++ 在 launch 前报错。

```python
# 定位：tests/test_flash_attn.py L1929-L1932（跳过条件摘要）
if has_batch_idx and paged_kv_block_size is not None:
    pytest.skip()
if has_leftpad and paged_kv_block_size is not None:
    pytest.skip()
```

## 症状 5：传 RoPE 但没有新 K/V 报错

现象：你想只对已有 cache 的 Q 做 RoPE，于是传 `rotary_cos/sin`，但不传本轮新 K/V。

源码入口：C++ 要求 RoPE 与新 K/V append 同时出现。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L1408-L1429（RoPE 校验摘要；精确证据见文末）
if (rotary_cos_.has_value()) {
    TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
    auto rotary_cos = rotary_cos_.value();
    CHECK_DEVICE(rotary_cos);
    params.rotary_dim = rotary_cos.size(1) * 2;
    TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
    TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
    const int seqlen_ro = rotary_cos.size(0);
    TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
    CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
    CHECK_CONTIGUOUS(rotary_cos);
    TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
```

可能原因与判断：KV cache API 的 RoPE 语义是“按 cache 写入位置旋转新 K，同时旋转当前 Q”，不是“重旋历史 cache”。

操作：同时记录 `k is not None`、`v is not None`、`rotary_cos/sin is not None`、`rotary_dim` 和 cos/sin 第一维。预期传 RoPE 时新 K/V 同时存在，`rotary_dim <= head_dim`、`rotary_dim % 16 == 0`，且位置表长度覆盖 C++ 计算出的 cache 长度；只读历史 cache 时不要借此接口重旋旧 K。

## 症状 6：重复 `cache_batch_idx` 导致 cache 更新不稳定

现象：同一个 cache slot 被 batch 中多条请求同时写入，结果看起来不稳定。

源码入口：Python docstring 明确提示，如果 `cache_batch_idx` 不唯一且传入新 K/V，更新值可能来自任意重复项。

```python
# 定位：flash_attn/flash_attn_interface.py L1561-L1567（参数说明摘要；精确证据见文末）
cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
    KV cache.
cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
    If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
    If the indices are not distinct, and k and v are provided, the values updated in the cache
         might come from any of the duplicate indices.
```

可能原因与判断：`cache_batch_idx` 是 remap，不是写冲突解决协议。重复索引在读 cache 时可能还能表达共享，但在 append 写入时会变成竞态。

操作：append 前检查 `torch.unique(cache_batch_idx).numel() == cache_batch_idx.numel()`。预期为 `True`；如果只是读取共享 slot，可以单独论证，但一旦本轮写入，新 K/V 的最终来源不再确定。

## 症状 7：`num_splits=0` 后性能变化，不知道是否影响正确性

现象：长上下文 decode 中 `num_splits=0` 有时比 `num_splits=1` 快或慢，你不确定它是不是改变语义。

源码入口：Python docstring 说明 `0` 表示自动选择；C++ heuristic 在 SM occupancy 和额外读写之间选 split 数。

```python
# 定位：flash_attn/flash_attn_interface.py L1581-L1584（num_splits 说明摘要；精确证据见文末）
num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
   If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
   to automatically determine the number of splits.
   Don't change this unless you know what you are doing.
```

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L257-L263
// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
```

可能原因与判断：`num_splits` 不改变 attention 的数学契约，但可能因归约顺序不同产生正常浮点差异；不能要求 `0` 与 `1` bitwise identical。实际 split 数大于 1 才会增加 partial buffer 和 combine。即使最终为 1，append/remap/paged 仍可能强制使用 aligned split kernel 族。

操作：固定 GPU、batch、`Sq/Sk`、Q/KV heads、head dim、dtype、cache layout、输入与 warmup，分别跑 `num_splits=1` 和 `0`，用 CUDA event 计时并与同一 reference 比误差。预期二者都在既定容差内；耗时方向不预设。若只想隔离 multi-split 变量，先用 `num_splits=1`。

## 症状 8：`seqlen_q=1` 下 GQA 行为看起来和普通 forward 不一样

现象：decode 单 token、GQA/MQA 场景下，源码对 Q 做 reshape/transpose，看起来改变了 shape。

源码入口：C++ 在满足无 ALiBi、无 local window、head dim 8 对齐等条件时，把 GQA group 维转成 sequence 维。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L1275-L1288（GQA reshape 摘要）
// causal=true is the same as causal=false in this case
if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
if (is_causal) { window_size_right = 0; }

// Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
// H/t Daniel Haziza
const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
if (seqlenq_ngroups_swapped) {
    const int ngroups = num_heads / num_heads_k;
    q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
    seqlen_q = ngroups;
    num_heads = num_heads_k;
```

可能原因与判断：这是 decode workload 的并行组织变换。它不改变 Q head 到 KV head 的语义映射，只是把 head group 暂时展开成更多 query rows；实际收益仍依赖 workload。

操作：静态对照入口 reshape 与函数末尾 restore，并在可运行环境记录输入/输出 shape。预期内部临时把 group 当作 query rows，返回仍恢复为调用方的 `(batch, 1, q_heads, head_dim)` 与相应 LSE 语义。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L1473-L1477（输出恢复摘要；精确证据见文末）
if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
}
return {out, softmax_lse};
```

## 症状 9：append 后输出对，但 cache 本体不确定是否更新

现象：你只比较了 `out`，没有确认新 K/V 是否写回 cache。

源码入口：测试在 `new_kv` 场景下不仅比较输出，还从 dense 或 paged cache 读回 K/V，并和 Python reference 的 `k_cache_ref/v_cache_ref` 比较。

```python
# 定位：tests/test_flash_attn.py L2047-L2052（reference append 摘要）
if new_kv:
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + seqlen_new
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(v, "b s ... -> (b s) ...")
```

```python
# 定位：tests/test_flash_attn.py L2118-L2138（cache 回读摘要；精确证据见文末）
if new_kv:
    if paged_kv_block_size is None:
        k_cache_select = (
            k_cache if not has_batch_idx else k_cache[cache_batch_idx.to(dtype=torch.long)]
        )
        v_cache_select = (
            v_cache if not has_batch_idx else v_cache[cache_batch_idx.to(dtype=torch.long)]
        )
    else:
        k_cache_select = rearrange(
            k_cache_paged[block_table.to(dtype=torch.long).flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
    assert torch.allclose(k_cache_select, k_cache_ref, rtol=1e-3, atol=1e-3)
    assert torch.equal(v_cache_select, v_cache_ref)
```

可能原因与判断：KV cache path 的 correctness 有两个维度：当前输出正确，以及 cache 状态正确。只测一个不够。

操作：同时保留 `out` 对 reference 的容差断言、K cache 回读断言与 V cache 精确断言；再执行下一步 decode。预期当前输出正确、cache 本体正确、下一步能够读到本步 token。当前 upstream 参数装饰器把 `has_batch_idx` 固定为 `False`，所以该测试不等于已经覆盖 dense remap 写回。

## 症状 10：当前步或下一步出现历史覆盖、未初始化 token

现象：append 后当前输出异常，或当前步看似正常、下一步突然偏离；检查 cache 时发现新 K/V 覆盖了历史位置，或者有效范围中夹着未写区域。

可能原因：上层把 `cache_seqlens` 当成了“扣除 leftpad 后的逻辑 token 数”。在当前 kernel 中，它是物理结束位置与 append 起点；逻辑历史长度才是 `cache_seqlens - cache_leftpad`。另一类原因是 `cache_seqlens + seqlen_new` 已超过 dense capacity，或 paged block table 没有覆盖写入区间。

源码入口：`BlockInfo` 同时计算 `leftpad_k`、`seqlen_k_cache` 与 `actual_seqlen_k`；append 测试的 reference update 则直接写 `[cache_seqlens, cache_seqlens + seqlen_new)`。

操作：对每条请求记录 `(cache_leftpad, cache_seqlens, seqlen_new, cache_capacity)`，并检查：

```python
assert 0 <= cache_leftpad <= cache_seqlens
assert cache_seqlens + seqlen_new <= cache_capacity
logical_old_len = cache_seqlens - cache_leftpad
```

预期：历史物理区间是 `[cache_leftpad, cache_seqlens)`，新写区间是 `[cache_seqlens, cache_seqlens + seqlen_new)`，两者首尾相接且不越界；`actual_seqlen_k == logical_old_len + seqlen_new`。paged KV 还必须逐逻辑 block 检查 `block_table` 覆盖与物理 block 生命周期。

## 精确证据底座

上文为排障速度保留了若干摘要卡；下面这些原文分别钉住最容易误判的硬边界。

```python
# 来源：flash_attn/flash_attn_interface.py L1542-L1546
    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.
```

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1247-L1255
    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }
```

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1398-L1405
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
```

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1408-L1426
    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
```

```python
# 来源：flash_attn/flash_attn_interface.py L1563-L1566
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
```

```python
# 来源：flash_attn/flash_attn_interface.py L1581-L1584
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
```

```cpp
// 来源：csrc/flash_attn/src/block_info.h L16-L24
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
```

## 最小排障顺序

1. 先确认这次是 dense cache、dense + `cache_batch_idx`、dense + leftpad，还是 paged KV。
2. 再确认 `cache_seqlens` 表示物理结束位置/append 起点；有 leftpad 时逻辑旧长度是二者之差，并确认上层容量足够。
3. 有 RoPE 时确认本轮是否传了新 K/V，以及 cos/sin 长度覆盖 cache。
4. 长上下文性能问题再看 `num_splits`，不要先怀疑数学正确性。
5. 最后进入 upstream 目录运行 `pytest tests/test_flash_attn.py -q -s -k test_flash_attn_kvcache`，同时看输出和 cache 写回断言；当前环境若无可加载 CUDA extension，只能做静态定位，不能宣称动态通过。
