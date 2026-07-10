---
title: "Python-API · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "Python-API"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# Python-API · 排障指南

## 读者任务

这篇按症状排查 Python API 与 extension 边界问题。读完后你应该能判断：

- 问题是否还在 Python/API 层，还是已经进入 C++/CUDA。
- 报错来自 import/ABI、stride/dtype、custom op/fake tensor、varlen 边界，还是 KV cache 状态。
- 哪些开关是测试用途，不能当成生产观测接口。

## 症状一：import 失败或 `undefined symbol`

| 判断 | 内容 |
|------|------|
| 常见原因 | `flash_attn_2_cuda` 没构建、wheel 与 PyTorch/CUDA ABI 不匹配，或 ROCm fallback 条件变化 |
| 源码入口 | `flash_attn/flash_attn_interface.py` 文件头 |
| 验证方法 | 单独运行 `python -c "import flash_attn_2_cuda"`，再确认 PyTorch/CUDA 版本 |

源码先根据 ROCm/HIP 与环境变量决定导入路径；默认 CUDA 路径导入 `flash_attn_2_cuda`。

```python
# 来源：flash_attn/flash_attn_interface.py L10-L28
# isort: off
# We need to import the CUDA kernels after importing torch
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if not USE_TRITON_ROCM and getattr(torch.version, 'hip', None) is not None:
    try:
        import flash_attn_2_cuda
    except ImportError:
        warnings.warn("flash_attn_2_cuda (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
        USE_TRITON_ROCM = True

if USE_TRITON_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu
else:
    import flash_attn_2_cuda as flash_attn_gpu

# isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
```

排障抓手：如果这里都 import 失败，不要看 kernel 代码；先查 extension 构建产物、CUDA/PyTorch ABI 和安装路径。

## 症状二：报 dtype、device、contiguous 或 head dim 错

| 判断 | 内容 |
|------|------|
| 常见原因 | 输入不是 fp16/bf16、不是 CUDA tensor、最后一维 stride 不是 1、head_dim 超过 256 或不是 8 的倍数 |
| 源码入口 | `csrc/flash_attn/flash_api.cpp::mha_fwd` |
| 验证方法 | 打印 `q.dtype/q.device/q.stride()/q.shape[-1]`，错误应在 kernel launch 前出现 |

C++ 入口的检查说明这些不是 Python 风格偏好，而是 kernel 契约。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L368-L395
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
```

排障抓手：`maybe_contiguous` 只修最后一维常见问题；复杂 layout 最好在模型侧显式整理。

## 症状三：`return_attn_probs=True` 没得到稳定 attention map

| 判断 | 内容 |
|------|------|
| 常见原因 | 这个选项是 testing only；Python 只在 `return_softmax and dropout_p > 0` 时下传真实返回 softmax 请求 |
| 源码入口 | `flash_attn_func` docstring 与 `FlashAttnFunc.forward` |
| 验证方法 | 检查 `dropout_p`，并用 reference attention 做局部对比，不把 `S_dmask` 当生产指标 |

公开 API 已说明 `return_attn_probs` 是测试用途。

```python
# 来源：flash_attn/flash_attn_interface.py L1203-L1215
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
```

forward 中实际下传条件更严格：

```python
# 来源：flash_attn/flash_attn_interface.py L855-L867
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
```

排障抓手：生产监控应看 `out`、延迟、吞吐、显存、kernel profile；不要依赖完整 attention map。

## 症状四：`torch.compile`、fake tensor 或 tracing 报错

| 判断 | 内容 |
|------|------|
| 常见原因 | PyTorch 版本影响 custom op 注册路径；fake tensor 输出 shape/dtype 不匹配 |
| 源码入口 | `_torch_custom_op_wrapper`、`_torch_register_fake_wrapper`、`_flash_attn_forward_fake` |
| 验证方法 | 检查 `torch.__version__`，再检查 `torch.ops.flash_attn._flash_attn_forward` 是否存在 |

```python
# 来源：flash_attn/flash_attn_interface.py L147-L154
if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward


@_torch_custom_op_wrapper("flash_attn::_flash_attn_varlen_forward", mutates_args=(), device_types="cuda")
def _flash_attn_varlen_forward(
```

排障抓手：如果问题出在 graph capture 或 fake tensor，不要先调 CUDA kernel；先确认 custom op 注册和 fake 函数返回 shape。

## 症状五：varlen 输出跨样本污染或 shape 对不上

| 判断 | 内容 |
|------|------|
| 常见原因 | `cu_seqlens`、`indices`、`max_seqlen` 与 packed token 数不一致 |
| 源码入口 | `unpad_input` / `pad_input` |
| 验证方法 | 检查 `cu_seqlens[0] == 0`、`cu_seqlens[-1] == packed.shape[0]`、`indices.numel() == packed.shape[0]` |

```python
# 来源：flash_attn/bert_padding.py L111-L126
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
```

排障抓手：varlen 错误通常是 correctness 问题。不要把跨样本 attention 污染误判成数值容差。

## 症状六：decode 路径慢或 cache 更新不符合预期

| 判断 | 内容 |
|------|------|
| 常见原因 | `cache_seqlens`、`block_table`、`cache_batch_idx`、RoPE position 或 cache 容量错误 |
| 源码入口 | `flash_attn_with_kvcache` 和模型层 fast path |
| 验证方法 | 检查 cache shape、cache length、是否 paged KV、是否传入新 k/v |

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
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_gpu.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        cache_leftpad,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        num_splits,
    )
    return (out, softmax_lse) if return_softmax_lse else out
```

排障抓手：decode 不是 dense forward 小 batch。性能瓶颈常在 KV cache load、paged KV 和 SplitKV combine，而不是 Python wrapper 本身。

## 复盘

Python API 层排障先分边界：import/ABI、tensor layout、custom op 编译生态、varlen 边界、KV cache 状态。只有这些都排除后，才进入 [[FlashAttention-FA2-Forward]] 或 [[FlashAttention-KV-Cache]] 看 kernel 和 cache 后端。
