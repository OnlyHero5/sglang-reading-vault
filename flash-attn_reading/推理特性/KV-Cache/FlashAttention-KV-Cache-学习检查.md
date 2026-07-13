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
updated: 2026-07-12
---
# KV-Cache · 学习检查

## 读者能做什么

- [ ] 能画出一次 decode step 的五个对象：`q`、`k_cache/v_cache`、可选新 `k/v`、`cache_seqlens`、地址模式。
- [ ] 能说明 prefill full attention 和 decode cache attention 的系统压力差异。
- [ ] 能沿 `flash_attn_with_kvcache → fwd_kvcache → mha_fwd_kvcache → run_mha_fwd → compute_attn_splitkv` 复述主线。
- [ ] 能解释 `cache_seqlens` 是物理结束位置/append 起点，并在有 leftpad 时算出逻辑旧长度 `cache_seqlens-cache_leftpad`。
- [ ] 能区分 dense slot、`cache_batch_idx` remap、`cache_leftpad` 起点和 paged `block_table` 四个地址角色，而不是把它们误当四个完全对称的模式。
- [ ] 能说明为什么 paged KV 不能和 `cache_batch_idx/cache_leftpad` 同时启用。
- [ ] 能解释 RoPE 为什么要求传入新 K/V，以及它如何绑定 cache 位置。
- [ ] 能说明 `num_splits=0` 是自动 heuristic，不是固定不 split。
- [ ] 能区分强制进入 split kernel、aligned single-split 和 multi-split + combine。
- [ ] 能指出 append 发生在 splitKV kernel 内部，而不是 Python 侧预写 cache。
- [ ] 能同时验证输出 correctness 和 cache update correctness。
- [ ] 能明确这条 KV-cache API 不支持 backward，并知道训练 attention 应回到普通 dense/varlen 路径。

## 源码定位练习

1. 在 `flash_attn/flash_attn_interface.py` 找到 `flash_attn_with_kvcache`。

目标：指出 Python 层做了哪些轻量归一化，哪些约束留给 C++。

证据入口：

```python
# 定位：flash_attn/flash_attn_interface.py L1593-L1627（Python 归一化摘要；精确原文见答案证据）
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
// 定位：csrc/flash_attn/flash_api.cpp L1247-L1268（paged 入口摘要；精确原文见答案证据）
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
// 定位：csrc/flash_attn/src/block_info.h L20-L24（长度公式摘要；精确原文见答案证据）
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
// 定位：csrc/flash_attn/src/flash_fwd_kernel.h L730-L783（append 与同步摘要；精确原文见答案证据）
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
# 定位：tests/test_flash_attn.py L2118-L2140（回读断言摘要；精确原文见答案证据）
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

## 源码定位判分标准

| 练习 | 必须交出的答案 | 通过标准 |
|------|----------------|----------|
| Python 入口 | 一张“Python 做/不做”两列表 | 能指出 contiguous、默认 scale、整数长度展开与返回打包；不把 dtype/shape/互斥都归给 Python |
| C++ 入口 | dense/remap/leftpad/paged 约束表 | 能指出 paged + remap、paged + leftpad 被拒绝，以及 page size 约束 |
| `BlockInfo` | 三个数值公式 | 写出物理结束位置、逻辑旧长度、append 后实际长度，特别处理 leftpad |
| append kernel | 写→同步→读事件序列 | 明确 `__syncthreads()` 保障同一 CTA 后续读到更新后的 K/V |
| upstream test | 输出与状态两类断言 | 指出 K cache 容差、V cache 精确比较、输出 reference 容差；同时说明当前测试未实际启用 dense `cache_batch_idx` |

## 答案证据

```python
# 来源：flash_attn/flash_attn_interface.py L1593-L1604
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
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
// 来源：csrc/flash_attn/src/block_info.h L20-L24
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
```

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L778-L783
        }
        // Need this before we can read in K again, so that we'll see the updated K values.
        __syncthreads();
        tKgK.data() = tKgK_data;
        tVgV.data() = tVgV_data;
    }
```

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1457-L1460
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
```

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_launch_template.h L185-L191
    if (params.num_splits == 1) {
        // Defined in flash_fwd_split_align_*.cu; declared extern in the main
        // flash_fwd_split_*.cu so this call does not re-instantiate the tree here.
        run_mha_fwd_splitkv_align<T, Headdim, Is_causal>(params, stream);
        return;
    }
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params, stream);
```

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
            v_cache_select = rearrange(
                v_cache_paged[block_table.to(dtype=torch.long).flatten()],
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
Push-Location flash-attn\flash-attention
pytest tests/test_flash_attn.py -q -s -k test_flash_attn_kvcache
Pop-Location
```

预期现象：

- `new_kv=True` 的组合会同时验证 `out` 和 cache 写回。
- paged KV 与 leftpad 会被跳过并被 C++ 拒绝；paged KV 与 batch idx 也有跳过条件和 C++ 拒绝，但当前参数装饰器把 `has_batch_idx` 固定为 `False`，不能声称该函数实际跑过 dense remap。
- 长上下文组合会覆盖 `num_splits=0` 自动选择路径。

这需要 Ampere 或更新 GPU、兼容的 PyTorch/CUDA 与可加载的 FlashAttention extension。当前环境不满足时运行静态替代：

```powershell
@'
import ast
from pathlib import Path
for path in [
    "flash-attn/flash-attention/flash_attn/flash_attn_interface.py",
    "flash-attn/flash-attention/tests/test_flash_attn.py",
]:
    ast.parse(Path(path).read_text(encoding="utf-8"))
print("AST parse: PASS")
'@ | python -
```

预期 `AST parse: PASS`。这只证明 Python 文件可解析，不证明 CUDA 数值或 cache 原地写回通过。

## 无 CUDA 的位置账本实验

下面的标准库脚本专门验收 leftpad 下的三个坐标，不依赖 Torch：

```powershell
@'
leftpad = 3
cache_end = 8
seqlen_new = 2
capacity = 12

logical_old_len = cache_end - leftpad
write_range = list(range(cache_end, cache_end + seqlen_new))
actual_len = logical_old_len + seqlen_new

assert logical_old_len == 5
assert write_range == [8, 9]
assert actual_len == 7
assert cache_end + seqlen_new <= capacity
print({"logical_old_len": logical_old_len, "write_range": write_range, "actual_len": actual_len})
'@ | python -
```

预期输出 `logical_old_len=5`、`write_range=[8, 9]`、`actual_len=7`。如果把 `cache_end` 错当成逻辑长度，三个结果会同时错位。

## 静态排障演练

给自己四个输入组合，手动判断会发生什么：

1. `block_table != None` 且 `cache_batch_idx != None`。

预期：C++ 入口报 paged KV 不支持 `cache_batch_idx`。

2. `rotary_cos != None` 但 `k/v == None`。

预期：C++ 入口报 RoPE 要求新 K/V 一起传入。

3. `k/v != None` 且 `cache_seqlens[i] + seqlen_new > seqlen_cache`。

预期：这不是完整由 C++ 兜底的错误；上层 runtime 应在调用前阻止。若放行，可能写越界或写错位置。

4. `k/v != None`、最终 `num_splits == 1`。

预期：append 会强制进入 split kernel 族，但 dispatch 走 aligned single-split；没有 partial buffer，也没有 combine。相反，若无 append/remap/paged 且 split 数不大于 1，才可能走普通 forward kernel。

## 复述练习

用三分钟讲清楚：

> 上层 runtime 已经为两条请求分配 cache。本轮传入 `q` 和新 `k/v`。FlashAttention 如何用 `cache_seqlens` 找写入位置，如何在 splitKV kernel 里写入新 K/V，如何通过 dense stride 或 `block_table` 读取历史 K/V，最后如何验证输出和 cache 状态都正确？

能讲完这段后，回到 [[FlashAttention-前向全链路]]，把这条 serving decode 支线放回完整 forward 地图；需要继续研究新架构实现时，再进入 [[FlashAttention-Hopper与CuTe]]。
