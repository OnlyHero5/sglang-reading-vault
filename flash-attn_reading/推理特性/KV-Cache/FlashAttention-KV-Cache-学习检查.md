---
title: "KV-Cache · 学习检查"
type: exercise
framework: flash-attn
topic: "KV-Cache"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# KV-Cache · 学习检查

## 读者能做什么

- [ ] 能画出一次 decode step 的五个对象：`q`、`k_cache/v_cache`、可选新 `k/v`、`cache_seqlens`、地址模式。
- [ ] 能说明 prefill full attention 和 decode cache attention 的系统压力差异。
- [ ] 能沿 `flash_attn_with_kvcache → fwd_kvcache → mha_fwd_kvcache → run_mha_fwd → compute_attn_splitkv` 复述主线。
- [ ] 能解释 `cache_seqlens` 为什么同时影响 append 写入位置和 attention 可见长度。
- [ ] 能区分 dense cache、`cache_batch_idx`、`cache_leftpad`、paged KV 四种地址语义。
- [ ] 能说明为什么 paged KV 不能和 `cache_batch_idx/cache_leftpad` 同时启用。
- [ ] 能解释 RoPE 为什么要求传入新 K/V，以及它如何绑定 cache 位置。
- [ ] 能说明 `num_splits=0` 是自动 heuristic，不是固定不 split。
- [ ] 能指出 append 发生在 splitKV kernel 内部，而不是 Python 侧预写 cache。
- [ ] 能同时验证输出 correctness 和 cache update correctness。

## 源码定位练习

1. 在 `flash_attn/flash_attn_interface.py` 找到 `flash_attn_with_kvcache`。

目标：指出 Python 层做了哪些轻量归一化，哪些约束留给 C++。

证据入口：

```python
# 来源：flash_attn/flash_attn_interface.py L1593-L1627
assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
if softmax_scale is None:
    softmax_scale = q.shape[-1] ** (-0.5)
if cache_seqlens is not None and isinstance(cache_seqlens, int):
    cache_seqlens = torch.full(
        (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
    )
```

2. 在 `csrc/flash_attn/flash_api.cpp` 找到 `mha_fwd_kvcache`。

目标：指出 paged KV 的判断、page size 检查、append 参数填充和 splitKV 分流。

证据入口：

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1247-L1268
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

3. 在 `csrc/flash_attn/src/block_info.h` 找到 `BlockInfo`。

目标：解释 `seqlen_k_cache` 和 `actual_seqlen_k` 的区别。

证据入口：

```cpp
// 来源：csrc/flash_attn/src/block_info.h L20-L24
// If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
// Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
, leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
, seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
, actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
```

4. 在 `csrc/flash_attn/src/flash_fwd_kernel.h` 找到 append 分支。

目标：说明新 K/V 何时写入 cache，以及为什么写完要同步后再读。

证据入口：

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L730-L783
const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
auto tKgK_data = tKgK.data();
auto tVgV_data = tVgV.data();
for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
    FLASH_NAMESPACE::copy_w_min_idx<Is_even_K>(
        tVgVnew, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
    );
    tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
}
__syncthreads();
tKgK.data() = tKgK_data;
tVgV.data() = tVgV_data;
```

5. 在 `tests/test_flash_attn.py` 找到 `test_flash_attn_kvcache`。

目标：说出测试如何验证输出和 cache 写回。

证据入口：

```python
# 来源：tests/test_flash_attn.py L2118-L2140
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
mult = 3 if not alibi else 5
assert (out - out_ref).abs().max().item() <= mult * (out_pt - out_ref).abs().max().item() + 1e-5
```

## 可执行验证

在 FlashAttention 源码环境、CUDA 可用且依赖安装完成后，运行：

```powershell
cd flash-attn\flash-attention
pytest tests/test_flash_attn.py -q -k test_flash_attn_kvcache
```

预期现象：

- `new_kv=True` 的组合会同时验证 `out` 和 cache 写回。
- paged KV 与 leftpad、paged KV 与 batch idx 的组合不会作为有效组合运行。
- 长上下文组合会覆盖 `num_splits=0` 自动选择路径。

## 静态排障演练

给自己三个输入组合，手动判断会发生什么：

1. `block_table != None` 且 `cache_batch_idx != None`。

预期：C++ 入口报 paged KV 不支持 `cache_batch_idx`。

2. `rotary_cos != None` 但 `k/v == None`。

预期：C++ 入口报 RoPE 要求新 K/V 一起传入。

3. `k/v != None` 且 `cache_seqlens[i] + seqlen_new > seqlen_cache`。

预期：这不是完整由 C++ 兜底的错误；上层 runtime 应在调用前阻止。若放行，可能写越界或写错位置。

## 复述练习

用三分钟讲清楚：

> 上层 runtime 已经为两条请求分配 cache。本轮传入 `q` 和新 `k/v`。FlashAttention 如何用 `cache_seqlens` 找写入位置，如何在 splitKV kernel 里写入新 K/V，如何通过 dense stride 或 `block_table` 读取历史 K/V，最后如何验证输出和 cache 状态都正确？

能讲完这段，再进入 [[FlashAttention-Hopper与CuTe]]。
