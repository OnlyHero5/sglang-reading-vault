---
title: "Python-API · 数据流"
type: dataflow
framework: flash-attn
topic: "Python-API"
learning_role: reference
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/dataflow
  - source-reading
updated: 2026-07-11
---
# Python-API · 数据流

## 读者任务

这篇只看对象生命周期：同一个 attention 调用在 Python、backend 路由与 compiled C++/kernel 边界上分别长什么样。读完后你应该能说明：

- dense forward 如何先选择 backend，以及 compiled 分支如何从 PyTorch tensor 变成 C++ `Flash_fwd_params`。
- varlen 如何从 padded batch 变成连续 token，再 scatter 回 padded batch。
- 返回值里 `out/softmax_lse/S_dmask/rng_state` 分别服务谁。
- KV cache decode 为什么多了 cache 长度、batch index、block table 和 RoPE 这些状态。

## Dense forward：tensor 到参数包

```mermaid
flowchart LR
    T["q/k/v tensor<br/>B,S,H,D"]
    API["flash_attn_func<br/>public API"]
    A["FlashAttnFunc.forward<br/>scale + head_dim pad"]
    W["_wrapped_flash_attn_forward<br/>custom op wrapper"]
    M["maybe_contiguous<br/>last dim stride=1"]
    E["flash_attn_gpu.fwd<br/>selected backend"]
    B{"backend"}
    R["ROCm Triton"]
    C["compiled extension / mha_fwd<br/>dtype/device/stride checks"]
    P["Flash_fwd_params<br/>ptr/stride/shape/flags"]
    K["CUDA/HIP kernel"]
    O["out + softmax_lse"]
    T --> API --> A --> W --> M --> E --> B
    B --> R --> O
    B --> C --> P --> K --> O
```

这里的顺序不能颠倒：public API 先进入 autograd `FlashAttnFunc.apply`；`FlashAttnFunc.forward` 先确定 scale，并在 head dim 不是 8 的倍数时 padding；随后才调用 wrapped custom op。`maybe_contiguous` 位于 custom op 实现内部，负责把最后一维 stride 不为 1 的输入整理后再交给扩展。它不是进入 autograd forward 之前的前置步骤。

```python
# 定位：flash_attn/flash_attn_interface.py L828-L855（摘要/骨架）
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        ...
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
```

```python
# 定位：flash_attn/flash_attn_interface.py L89-L106（摘要/骨架）
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ...
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
        q,
        k,
        v,
        ...
    )
```

下面的 C++ fixed-length forward 只描述 compiled CUDA extension 分支：它首先把 Python tensor 约束成 kernel 能接受的形态。ROCm Triton 分支不经过这段 `mha_fwd`，HIP compiled extension 的平台门禁也不能由这张 CUDA 卡外推。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L351-L395
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

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

随后 Python 语义被固化进参数包。fixed-length 路径里 `cu_seqlens_q/k` 是 `nullptr`，这和 varlen 路径形成清晰边界。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L452-L470
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );
```

## Varlen：padded batch 到连续 token

```mermaid
flowchart LR
    PAD["padded hidden_states<br/>B,S,..."]
    MASK["attention_mask"]
    UNPAD["unpad_input"]
    TOK["packed tokens<br/>total_nnz,..."]
    IDX["indices"]
    CU["cu_seqlens<br/>B+1"]
    KERNEL["varlen_fwd"]
    PADOUT["pad_input<br/>scatter back"]
    PAD --> UNPAD
    MASK --> UNPAD
    UNPAD --> TOK
    UNPAD --> IDX
    UNPAD --> CU
    TOK --> KERNEL --> PADOUT
    IDX --> PADOUT
```

`unpad_input` 的输出同时服务 kernel 和回填：

```python
# 定位：flash_attn/bert_padding.py L111-L128（摘要/骨架；去除 upstream 尾随空格）
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
        used_seqlens_in_batch,
    )
```

前四项的打包边界来自 `attention_mask + unused_mask`，第五项却只统计 `attention_mask`。当前 docstring 对 `seqused` 的描述与第五返回表达式不一致，数据流判断应以实现为准。

回填则只需要 packed output 和原始 indices：

```python
# 来源：flash_attn/bert_padding.py L204-L218
def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
```

## 返回值：四类消费者

| 返回值 | 生产者 | 消费者 | 语义 |
|--------|--------|--------|------|
| `out` | backend forward | 上层模型 | attention 输出 |
| `softmax_lse` | backend forward | backward、测试、可选用户返回 | 每行 logsumexp 摘要 |
| `S_dmask` | backend forward 可选 | 测试 | dropout 调试槽位；不是稳定的完整概率矩阵 |
| `rng_state` | backend forward | backward | dropout 开启时对齐随机状态；作为协议字段随 forward 返回 |

`FlashAttnFunc.forward` 只在需要梯度时保存 backward 所需对象：

```python
# 来源：flash_attn/flash_attn_interface.py L855-L878
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
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)
```

公开 `return_attn_probs=True` 决定是否返回三元组，但传给 backend 的 `return_softmax` 还要同时满足 `dropout_p > 0`。因此 dropout 为 0 时第三项不能按函数参数名推断为真实 attention probability。

## KV cache：cache 状态参与输入契约

下面这个模型层 decode fast path 会把 q、新 kv、历史 dense cache、cache length、RoPE 和 ALiBi 传给 `flash_attn_with_kvcache`。它没有传 `cache_batch_idx` 或 `block_table`，所以只能证明 dense cache 的一种上层接线，不能代表 paged/remap 的完整能力面。

```python
# 来源：flash_attn/modules/mha.py L526-L569
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.inner_cross_attn.softmax_scale,
            causal=self.inner_cross_attn.causal,
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,
            alibi_slopes=alibi_slopes,
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            alibi_slopes = getattr(self.inner_cross_attn, "alibi_slopes", None)
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
```

读者抓手：KV cache 数据流不是“多一个参数”，而是多了一组状态账：cache memory、物理结束位置、可选 batch remap、可选 page table、leftpad 与 RoPE position。签名同时暴露这些对象不代表任意组合都合法；paged KV 当前拒绝与 `cache_batch_idx`、leftpad 同时使用。

## 运行验证

| 验证 | 方法 | 预期 |
|------|------|------|
| dense 对象形态 | 打印 `q.shape`、`q.stride()`，再调用 `flash_attn_func` | last dim stride 为 1 时不额外复制 |
| varlen 边界 | 检查 `cu_seqlens[0] == 0` 和 `cu_seqlens[-1] == packed.shape[0]` | packed token 数和边界一致 |
| 回填 | `pad_input(unpadded, indices, batch, seqlen)` | shape 回到 `(batch, seqlen, ...)` |
| KV cache | 传 int `cache_seqlens` | Python 层会转为 batch 维 int32 tensor |

无法加载 extension/Aiter 时，先执行静态替代：

```powershell
@'
import ast
from pathlib import Path
for path in [
    "flash-attn/flash-attention/flash_attn/flash_attn_interface.py",
    "flash-attn/flash-attention/flash_attn/bert_padding.py",
    "flash-attn/flash-attention/flash_attn/modules/mha.py",
]:
    ast.parse(Path(path).read_text(encoding="utf-8"))
print("AST parse: PASS")
'@ | python -
```

预期三份文件均可解析。这个结果只验证 Python 对象流的静态完整性，不证明 compiled ABI、GPU 数值、paged KV 组合或 `torch.compile` 可执行。

## 复盘

Python API 层的数据流可以看成三张账：dense 账管 tensor 到 params，varlen 账管 padding 到边界数组，decode 账管 cache 状态。读 [[FlashAttention-FA2-Forward]] 前，至少要能画出这三张账。
