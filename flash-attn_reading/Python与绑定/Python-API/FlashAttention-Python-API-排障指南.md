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

这篇按症状排查 Python API 与 backend 边界问题。读完后你应该能判断：

- 问题是否还在 Python/API 层，还是已经进入 compiled C++/CUDA/HIP 或 ROCm Triton。
- 报错来自 import/ABI、stride/dtype、custom op/fake tensor、varlen 边界，还是 KV cache 状态。
- 哪些开关是测试用途，不能当成生产观测接口。

## 症状一：import 失败或 `undefined symbol`

| 判断 | 内容 |
|------|------|
| 常见原因 | compiled extension 没构建、wheel 与 PyTorch/CUDA/HIP ABI 不匹配，或 ROCm Triton 所需 Aiter 不可用 |
| 源码入口 | `flash_attn/flash_attn_interface.py` 文件头 |
| 验证方法 | 单独运行 `python -c "import flash_attn_2_cuda"`，再确认 PyTorch/CUDA 版本 |

源码先根据 ROCm/HIP 与环境变量决定导入路径；默认 CUDA 路径导入 `flash_attn_2_cuda`。只有 HIP 环境会在该 extension 导入失败后把 `USE_TRITON_ROCM` 改为真；普通 CUDA 分支失败不会自动切到 Triton。

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

排障抓手：先打印 `torch.version.cuda`、`torch.version.hip` 和环境变量，再按实际分支检查 extension 或 Aiter。若普通 CUDA 环境 `import flash_attn_2_cuda` 失败，预期是直接暴露构建/ABI 问题，而不是出现 Triton fallback。

## 症状二：报 dtype、device、contiguous 或 head dim 错

| 判断 | 内容 |
|------|------|
| 常见原因 | 输入不是支持 dtype/device、直接调用 compiled ABI 时末维 stride/head dim 不合约，或 public wrapper padding 后仍超过能力边界 |
| 源码入口 | `csrc/flash_attn/flash_api.cpp::mha_fwd` |
| 验证方法 | 打印 `q.dtype/q.device/q.stride()/q.shape[-1]`，错误应在 kernel launch 前出现 |

下面的 C++ 检查只证明 CUDA compiled-extension 分支的契约。public dense/packed/varlen API 会先把非 8 倍数 head dim 补齐，因此调用者原始 head dim 不是 8 的倍数并不必然报错；直接调用 extension 或绕过 wrapper 时才会直接撞上这项检查。

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

排障抓手：`maybe_contiguous` 会在末维 stride 不为 1 时复制为 contiguous；先比较调用前后的 data pointer 和 stride，预期末维已变成 1。若错误仍来自 C++，再核对 padding 后的 head dim、dtype、device 与 GQA 整除关系；ROCm Triton/HIP 的门禁应查对应 backend，不能照抄这张 CUDA 表。

## 症状三：`return_attn_probs=True` 没得到稳定 attention map

| 判断 | 内容 |
|------|------|
| 常见原因 | 这个选项是 testing only；公开三元组返回与 backend 是否生成 `S_dmask` 是两层条件 |
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

排障抓手：`return_attn_probs=True` 会让公开 API 返回三元组，但只有 `dropout_p > 0` 才向 backend 请求 `S_dmask`；dropout 为 0 时第三项是空槽位，不能当 attention map。生产监控应看 `out`、延迟、吞吐、显存和 kernel profile，不依赖该测试输出。

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

排障抓手：如果 dense/varlen 问题出在 graph capture 或 fake tensor，不要先调 kernel；先确认 PyTorch 版本、custom op 注册和 fake shape。KV-cache 在当前基线直接调用 `flash_attn_gpu.fwd_kvcache`，没有同一套 custom-op/fake 路径，不能用 `torch.ops.flash_attn._flash_attn_forward` 的存在性替它验收。

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

有 `unused_mask` 时还要同时核对两套长度：packed tokens、`indices`、`cu_seqlens` 和 max length 来自 `attention_mask + unused_mask`；`unpad_input` 第五返回值 `used_seqlens_in_batch` 只来自 `attention_mask`。当前 docstring 对第五项的描述与实现不一致，预期判断以实际返回表达式为准。

## 症状六：decode 路径慢或 cache 更新不符合预期

| 判断 | 内容 |
|------|------|
| 常见原因 | `cache_seqlens`/leftpad 坐标混淆、cache 容量或 block coverage 不足、非法 paged 组合、RoPE position 错误、重复 remap 写入或 SplitKV 路径选择变化 |
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

排障抓手：decode 不是 dense forward 小 batch。先分别验收 output、cache 写回和下一 step 可见性；paged KV 当前拒绝与 `cache_batch_idx`、leftpad 同时使用，重复 `cache_batch_idx` 配合 append 的最终写入者不确定。性能归因必须固定 GPU、shape、dtype、causal/window、cache layout、`num_splits`、warmup 和计时方式，再用 profiler 区分 Python、cache load 与 multi-split combine，不能预设瓶颈。

## 无 GPU/extension 时的静态替代

```powershell
@'
import ast
from pathlib import Path
for path in [
    "flash-attn/flash-attention/flash_attn/flash_attn_interface.py",
    "flash-attn/flash-attention/flash_attn/bert_padding.py",
]:
    ast.parse(Path(path).read_text(encoding="utf-8"))
print("AST parse: PASS")
'@ | python -

rg -n 'USE_TRITON_ROCM|return_softmax=return_softmax and dropout_p|used_seqlens_in_batch|fwd_kvcache' flash-attn/flash-attention/flash_attn/flash_attn_interface.py flash-attn/flash-attention/flash_attn/bert_padding.py
```

预期 AST 通过，并能定位 backend fallback、`S_dmask` 门禁、第五长度账和 KV-cache 直调。静态替代不证明 ABI、GPU 数值、compile graph 或性能。

## 复盘

Python API 层排障先分边界：import/ABI、tensor layout、custom op 编译生态、varlen 边界、KV cache 状态。只有这些都排除后，才进入 [[FlashAttention-FA2-Forward]] 或 [[FlashAttention-KV-Cache]] 看 kernel 和 cache 后端。
