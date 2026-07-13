---
title: "FlashAttention 关键概念"
type: concept
framework: flash-attn
topic: "导读与总览"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/concept
  - source-reading
updated: 2026-07-10
---
# FlashAttention 关键概念

## 读者任务

这篇解决“读 FlashAttention 前必须先抓住哪些概念”的问题。读完后你应该能做到：

- 用 HBM/SRAM/register 的层级解释为什么 attention 会被 IO 卡住。
- 用 `row_max/row_sum/acc_o/LSE` 解释为什么分块 softmax 仍然是 exact attention。
- 区分 dense、packed、varlen、KV cache、paged KV、SplitKV 这些输入和执行形态。
- 看到一个概念时，能说出它落在哪个源码对象上，而不是只记术语。

## 概念地图

| 概念 | 心理模型 | 源码对象 | 失效边界 |
|------|----------|----------|----------|
| IO-aware | 主计算不物化完整 `Sq×Sk` score/probability 矩阵 | kernel 中 tile 级 `acc_s/rP/acc_o` | 测试 `S_dmask` 与 multi-split partial buffer 是例外 |
| Online Softmax | 每行持续维护可重标定的 max 与 sum | `softmax_rescale_o`、`normalize_softmax_lse` | 不能替代 mask/dropout 语义 |
| LSE | forward 留给 backward 的每行摘要 | `softmax_lse` | 不是完整 attention map |
| Packed QKV | 上层预先合并 Q/K/V，减少 backward 拼接 | `flash_attn_qkvpacked_func` | 不等同于 varlen |
| Varlen | 把有效 token 压成连续区间，用 prefix sum 表示边界 | `cu_seqlens_q/k` | `cu_seqlens` 错会导致跨样本污染 |
| GQA/MQA | Q head 多于 KV head，KV 被多组 Q 共享 | API docstring 的 `num_heads_k` 约束 | 不是任意 head 数可混搭 |
| KV cache | decode 中复用历史 K/V，并可原地追加新 K/V | `flash_attn_with_kvcache` | 需要 cache 容量和 offset 正确 |
| SplitKV | 一套 split kernel 家族；可被 append/remap/paged 强制，也可真正多 split 后 combine | `set_params_splitkv`、`run_mha_fwd_splitkv_dispatch` | forced/aligned single-split 不等于 multi-split combine |
| Paged KV | KV cache 按 block/page 管理 | `block_table`、paged KV 分支 | 不等同于连续 cache |
| FA3/FA4 | 新硬件/新编译组织方式 | `hopper/`、`flash_attn/cute/` | 不能和 FA2 extension 混成一条路径 |

## 核心状态：三本行级账

标准 attention 可以写成：

```text
S = QK^T
P = softmax(S)
O = PV
```

FlashAttention 的关键是让每行只携带三类状态穿过 K/V blocks：

```text
m_i = 当前已处理 key blocks 的最大 score
l_i = sum_j exp(score_ij - m_i)
o_i = sum_j exp(score_ij - m_i) * v_j
```

每处理一个新的 K/V block，就用新的局部 score 修正旧的标尺。最后输出 `o_i / l_i`，并把 `log(l_i) + m_i` 保存成 LSE。这个模型可以解释两个现象：

- `P` 不需要作为完整 `seqlen_q * seqlen_k` 矩阵长期保存。
- backward 可以用 Q/K/V/O/dO/LSE 重算局部 probability。

## 源码证据：score tile 只短暂存在

forward 主循环先算局部 `QK`，再应用 softcap 和 mask，然后进入 online softmax。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L319-L347
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
```

读者抓手：`acc_s` 先承载局部 score，经 online-softmax 更新后变成相对当前行最大值的未归一化指数权重；`rP` 只是把这份权重转换到 fp16/bf16 供 `PV` 使用。它不是最终 probability，最终输出归一化在 epilogue 才完成。

## 源码证据：LSE 是压缩协议字段

kernel epilogue 把 online softmax 的行级状态整理成 `softmax_lse`。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L433
    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
```

Python API 文档也把 `softmax_lse` 描述成每行 `QK^T * scaling` 的 logsumexp。

```python
# 来源：flash_attn/flash_attn_interface.py L1209-L1214
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
```

读者抓手：LSE 是 `seqlen_q` 级别的摘要。`S_dmask` 是 testing-only 调试槽位，可能带不同 scaling 和 dropout 符号编码；只有 dropout 开启时 backend 才被要求生成它，dropout 为 0 时公开三元组的第三项是空槽位。生产主线应围绕 out/LSE/RNG 协议，而不是试图拿完整 `P`。

## 源码证据：KV cache 是另一种输入契约

decode 不是普通 forward 的小 batch 版本。KV cache API 可以在一个 kernel 中追加新 K/V、应用 RoPE，并对更新后的 cache 做 attention。

```python
# 来源：flash_attn/flash_attn_interface.py L1485-L1514
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.
```

读者抓手：KV cache 路径的问题通常不是 softmax 原理问题，而是 cache 容量、物理结束位置 `cache_seqlens`、leftpad 后逻辑长度、`block_table`、RoPE position 或 split 路径的问题。paged KV 当前拒绝与 `cache_batch_idx`、leftpad 同时使用；`num_splits==1` 不产生 partial buffer/combine，只有真正 multi-split 才有额外 partial O/LSE 生命周期。

## 常见误解

| 误解 | 正确读法 | 源码入口 |
|------|----------|----------|
| FlashAttention 是近似 attention | 它是 exact attention，优化的是 IO 和执行顺序 | `flash_fwd_kernel.h` 主循环 |
| LSE 是 debug 信息 | LSE 是 forward/backward 协议字段 | `normalize_softmax_lse` |
| varlen 是 ragged tensor | varlen 是连续 token + `cu_seqlens` | [[FlashAttention-Python-API-数据流]] |
| decode 就是 batch size 为 1 的 forward | decode 关键在 KV cache load/update 和 SplitKV | `flash_attn_with_kvcache` |
| FA4 是 FA2 的小重构 | FA4 是 CuTeDSL/JIT 后端路径 | `flash_attn/cute/` |
| `rP` 是当前 tile 的最终概率 | `rP` 是未最终归一化的指数权重，epilogue 才归一化输出 | `softmax_rescale_o`、`normalize_softmax_lse` |

## 运行验证

| 验证目标 | 操作 | 预期 |
|----------|------|------|
| 验证 exact attention | 用小 shape 对比 PyTorch reference | `out` 数值接近，不要求返回完整 `P` |
| 验证 LSE/调试槽位 | 分别用 dropout=0 与 dropout>0 开启 `return_attn_probs` | LSE 是每行一个值；dropout=0 第三项为空，dropout>0 的 `S_dmask` 只作测试 |
| 验证 varlen 边界 | 构造不同长度样本并检查 `cu_seqlens` | 不同样本之间不能互相 attend |
| 验证 KV cache | 对 decode step 传入 `k/v` 和 `cache_seqlens` | cache 被追加，输出基于更新后的 cache |

无可加载 GPU backend 时，执行静态替代：

```powershell
rg -n 'apply_softcap|apply_mask|softmax_rescale_o|normalize_softmax_lse' flash-attn/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h
rg -n 'return_softmax=return_softmax and dropout_p|fwd_kvcache' flash-attn/flash-attention/flash_attn/flash_attn_interface.py
```

预期依次定位 softcap→mask→online update→epilogue，以及 `S_dmask` 的 dropout 门禁和 KV-cache 直调。静态定位不证明数值、ABI 或性能。

## 复盘

关键概念要落到源码对象上才有用：IO-aware 对应 tile 生命周期，online softmax 对应 `softmax_rescale_o`，LSE 对应 forward/backward 协议，varlen 对应 `cu_seqlens`，KV cache 对应 cache 指针和长度账。后续读 [[FlashAttention-前向全链路]] 时，把每一步都归到这些对象上。
