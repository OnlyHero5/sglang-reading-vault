---
title: "Backward · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "Backward"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Backward · 排障指南

本页面向训练反向排障：读完后应能判断 backward 显存、dropout、deterministic、GQA/MQA 和 varlen 问题分别落在哪个源码入口，而不是把所有梯度异常都归因到 kernel 数值误差。

## 排障入口

| 症状 | 优先检查 | 源码入口 | 验证方式 |
|------|----------|----------|----------|
| 梯度显存远高于预期 | 是否保存了完整 `P` 或开启了 deterministic 大 split | `FlashAttnFunc.forward`、`mha_bwd` buffer 分配 | 看 `ctx.save_for_backward` 和 `dq_accum` shape |
| dropout 训练梯度不稳定 | forward/backward RNG 是否一致 | `rng_state` 保存与 kernel dropout | 固定 seed，对比 `dropout_p=0` 与 `>0` |
| dropout 只在 `dQ/dK` 或 `dV` 上出现固定倍率偏差 | `D`、`dP` 与最终梯度的 `p_keep` 缩放所有权 | preprocess、`convert_dQ`、backward epilogue | 对比误差倍率是否接近 `p_keep` 或 `1/p_keep` |
| `deterministic=True` 变慢 | split 维和 `dq_accum` 是否增大 | `mha_bwd`、launch template | 比较 deterministic 开关下显存和耗时 |
| GQA/MQA 的 `dK/dV` shape 异常 | expanded buffer 与 group sum | `dk_expanded/dv_expanded`、`sum_out` | 检查 `num_heads % num_heads_k == 0` |
| varlen backward 越界或错梯度 | `cu_seqlens`、`total_q`、`unpadded_lse` | `mha_varlen_bwd` | 检查 `cu_seqlens` dtype、contiguous 和 LSE layout |
| KV cache 期望支持训练反向 | API 边界理解错误 | `flash_attn_with_kvcache` | 该接口文档明确不支持 backward |

## 为什么 backward 不保存 attention matrix？

保存完整 `P = softmax(QK^T)` 会重新引入 `O(seqlen_q * seqlen_k)` 显存开销。源码选择保存 `out`、`softmax_lse` 和 dropout `rng_state`，backward 在 tile 内重算当前块的 `P`。

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

验证抓手：如果你在改 autograd wrapper，确认 `ctx.save_for_backward` 没有新增完整 `[batch, heads, seqlen_q, seqlen_k]` 级别的状态。

## 为什么 backward 需要 `out`？

数学上，`out` 用来计算 softmax backward 的行标量 `D=sum(dO*O)`。实现上还要看 `scale`：dropout 路径把点积乘 `p_keep`，让它与尚未乘 `1/p_keep` 的内部 `dP` 对齐。只有 `Q/K/V/LSE` 不够，因为这张行级账依赖 forward 输出和上游梯度。

```cpp
// 来源：csrc/flash_attn/src/flash_bwd_preprocess_kernel.h L38-L50
#pragma unroll
for (int mi = 0; mi < size<0>(do_reshaped); ++mi) {
    float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
    #pragma unroll
    for (int ni = 1; ni < size<1>(do_reshaped); ni++) {
        dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
    }
    FLASH_NAMESPACE::SumOp<float> sum_op;
    dP_sum_cur = FLASH_NAMESPACE::Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
    if (threadIdx.x % THREADS_PER_ROW == 0) {
        dP_sum(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum_cur;
    }
}
```

排障抓手：如果 `out` shape 被裁剪、padding 不一致或传错 buffer，`D` 会错，后续 `dS` 全部建立在错误行标量上。若只有 dropout 路径呈固定倍率偏差，再核对传给 `dot_do_o` 的 `scale=params.p_dropout` 与最终 dQ/dK/dV 缩放，不能重复乘或漏乘 `1/p_keep`。

## deterministic backward 为什么更慢？

deterministic 路径要固定 `dQ` 的归约结构，源码给 `dq_accum` 加了 split 维，并在 launch/convert 阶段使用这个 split 数。它换来更稳定的归约顺序，代价是更多临时显存和额外归并。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L890-L895
if (!deterministic) {
    dq_accum = torch::empty({batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
} else {
    const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
    dq_accum = torch::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
}
```

验证抓手：跑同一个输入，分别设置 `deterministic=False/True`，观察 peak memory、耗时以及重复运行的梯度差。正确预期是 deterministic 通常更慢或占更多临时空间，并把每个 sequence-K 工作块隔离到独立 split 后固定归并；不要误写成“确定性路径完全没有 atomic”，同一 split 内的实现仍需结合 kernel 写入方式判断。

## GQA/MQA 的 `dK/dV` 为什么要 sum？

GQA/MQA 中多个 Q heads 共享同一个 KV head。Backward 中每个 Q head group 都会对同一个 KV head 产生贡献，所以源码先用 expanded `dk/dv` 接住贡献，再沿 group 维求和。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L967-L971
// For MQA/GQA we need to sum dK and dV across the groups
if (num_heads_k != num_heads) {
    at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
}
```

排障抓手：先检查 `num_heads % num_heads_k == 0`。如果这个关系不成立，C++ 入口会在 launch 前失败。

## varlen backward 和 dense backward 的公式是否不同？

公式相同，layout 不同。Varlen 用 `total_q/total_k` 和 `cu_seqlens_q/k` 表示 packed token 的 batch 边界，`softmax_lse` 是 `[heads, total_q]`，并通过 `unpadded_lse=true` 告诉 kernel 按 unpadded layout 解释。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1139-L1165
Flash_bwd_params params;

set_params_dgrad(params,
                 batch_size,
                 max_seqlen_q, max_seqlen_k,
                 seqlen_q_rounded, seqlen_k_rounded,
                 num_heads, num_heads_k,
                 head_size, head_size_rounded,
                 q, k, v, out,
                 dout, dq, dk_expanded, dv_expanded,
                 cu_seqlens_q.data_ptr(),
                 cu_seqlens_k.data_ptr(),
                 loop ? dq_accum.data_ptr() : nullptr,
                 nullptr,
                 nullptr,
                 softmax_lse.data_ptr(),
                 softmax_d.data_ptr(),
                 p_dropout,
                 softmax_scale,
                 window_size_left,
                 window_size_right,
                 softcap,
                 deterministic,
                 /*unpadded_lse*/true);
params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
params.total_q = total_q;
```

验证抓手：检查 `cu_seqlens_q/k` 是 int32、contiguous，最后一个元素等于 `total_q/total_k`。

## 为什么 KV cache API 没有 backward？

`flash_attn_with_kvcache` 面向 decode serving：它会原地更新 KV cache，支持 cache remap、paged KV、RoPE 和 SplitKV。这个路径追求推理时延，不是训练 autograd。

```python
# 来源：flash_attn/flash_attn_interface.py L1542-L1546
If window_size != (-1, -1), implements sliding window local attention. Query at position i
will only attend to keys between
[i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

Note: Does not support backward pass.
```

排障抓手：训练长上下文使用 dense/varlen `flash_attn_func` 或 packed API；decode cache 行为读 [[FlashAttention-KV-Cache]]。

## 复盘迁移

- 看到“梯度错”，先问状态是否与 forward 同源：`out`、LSE、RNG、mask/softcap 参数。
- 看到“显存高”，先问是否 deterministic、varlen padding 或额外保存了完整矩阵。
- 看到“shape 错”，先分清 dense `[b,s,h,d]`、varlen `[total,h,d]`、GQA expanded head 维。
