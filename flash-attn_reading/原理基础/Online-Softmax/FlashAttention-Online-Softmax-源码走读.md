---
title: "Online-Softmax · 源码走读"
type: walkthrough
framework: flash-attn
topic: "Online-Softmax"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/walkthrough
  - source-reading
updated: 2026-07-12
---
# Online-Softmax · 源码走读

## 读者任务

这篇以 baseline `002cce0` 的 FlashAttention-2 CUDA forward 为主，沿一个 query tile 的生命周期读源码：它逆序扫描多个 K/V block，每次拿到一个 score tile，更新跨 block 的 softmax 状态，并把当前块的未归一化指数权重立即消费进 `P @ V`。FA3 的 warp 专职化与流水布局不同，本篇不把 FA2 的线程组织冒充成所有版本的统一实现；这里要抓的是稳定不变的在线归一化算法。

读完你应该能回答：

- `acc_s` 从 raw score 变成在线 softmax 指数分子的准确位置在哪里，以及它为什么还不能叫最终概率。
- `row_max/row_sum/acc_o` 如何在第一块和后续块之间迁移。
- 为什么 forward 主循环不保存完整 `P`。
- `softmax_lse` 如何从 CUDA epilogue 传到 Python autograd，再服务 backward。

## 长文读法

这篇按一个 query tile 扫描多个 K/V block 的状态转移读：行级 reduce 先在寄存器 fragment 上算 max / sum，`Softmax` 持有跨 block 的 `row_max`、`row_sum`，而 `acc_o` 是外部持有的输出分子。第一块初始化状态，后续块重标尺历史分母和输出分子；每个 block 的指数权重立刻进入 `P @ V`，最终归一化与 LSE 计算留到 epilogue。

| 你的任务 | 先读 | 抓住什么 |
|----------|------|----------|
| 建立 online softmax 心智模型 | 1 到 4 | 状态跨 K block 传递，完整 `P` 不落 HBM |
| 排查第一块和后续块差异 | 3 到 4 | 后续块必须用新 max 重新缩放历史 `acc_o` |
| 定位主循环调用点 | 5 | softcap / mask 后立即更新 softmax，并消费未归一化指数权重 |
| 排查输出和 LSE | 6 到 7 | epilogue 写 `O` / LSE，Python autograd 保存 LSE 支撑 backward |
| 做源码验证 | 运行验证 | grep `softmax_rescale_o`、`normalize_softmax_lse` 和 `softmax_lse` 的跨层流向 |

## 主线地图

```mermaid
flowchart LR
    QK["Q tile x K tile<br/>acc_s"]
    MASK["softcap / mask"]
    SOFT["softmax_rescale_o<br/>更新 row_max row_sum acc_o"]
    P["rP<br/>未归一化指数权重<br/>dropout 可在此作用"]
    PV["gemm_rs<br/>acc_o += P V"]
    EPI["normalize_softmax_lse<br/>O + LSE"]
    CTX["autograd ctx<br/>保存 LSE"]
    QK --> MASK --> SOFT --> P --> PV --> EPI --> CTX
```

## 1. 行级 reduce 在寄存器 fragment 上完成

系统压力是片上存储：score tile 位于 MMA accumulator fragment 中，不能为了 softmax 把完整矩阵写回 HBM。`softmax.h` 先提供行级 reduce 工具，把每行的 max/sum 摘出来。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L23-L36
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}
```

`thread_reduce_` 先汇总每个线程持有的列。max 还要跨同一四线程小组归并；sum 则把这一步推迟到最终归一化，避免每扫一个 K block 都做不需要的通信。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L38-L51
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}
```

这里的关键不是通用 reduce，而是 shape 约束：输入是二维 score fragment，输出是一维行摘要。Online softmax 的状态从一开始就是按 query 行组织的。

## 2. `Softmax` 持有跨 K block 的两本账

`Softmax<kNRows>` 自己只持有 `row_max` 和 `row_sum`。输出分子账 `acc_o` 在 forward 主循环中已经存在，所以作为参数传入并被同步重标尺。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L128-L140
template <int kNRows>
struct Softmax {

    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;

    __forceinline__ __device__ Softmax() {};

    template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
    __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        static_assert(decltype(size<0>(scores))::value == kNRows);
```

`acc_s` 会被原地改写。进入函数时它是 mask 后 score；离开函数时是 `exp(score * scale - row_max * scale)` 形式的指数分子。它还没有除以最终 `row_sum`，所以严格说不是最终概率。

## 3. 第一块 K/V 初始化状态

第一块没有历史，源码直接从当前 score tile 建立 `row_max` 和 `row_sum`。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L141-L145
if (Is_first) {
    FLASH_NAMESPACE::template reduce_max</*zero_init=*/true>(scores, row_max);
    FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
    FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
} else {
```

这一步做了三件事：

- `reduce_max` 得到当前已知的行最大值。
- `scale_apply_exp2` 用 `exp2` 把 score 变成与 `exp(score * softmax_scale - max)` 等价的指数分子。
- `reduce_sum` 建立当前标尺下的分母。

首块路径不会缩放 `acc_o`，因为历史输出还不存在。

## 4. 后续块必须重标尺历史状态

后续 K/V block 可能出现新的最大 score。源码先保存旧最大值，再把当前 block 的最大值合并进 `row_max`，最后用同一个 `scores_scale` 缩放 `row_sum` 和 `acc_o`。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L146-L166
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
            // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
            Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale;
                #pragma unroll
                for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
            }
            FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
            // We don't do the reduce across threads here since we don't need to use the row_sum.
            // We do that reduce at the end when we need to normalize the softmax.
            FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
        }
```

这是本专题最重要的源码段。只要新旧最大值不同，旧分母和旧输出分子都不在新标尺上；二者必须一起迁移。`Check_inf` 只在后续块重标尺时参与：如果合并后的 `row_max` 仍为 `-inf`，它把“当前 max”替换成 `0`，使旧状态缩放为零，避开 `-inf - -inf -> NaN`。它保护的是整行暂时或最终都被 mask 的边界，不是一个泛化的数值稳定开关。

## 5. Forward 主循环在 mask 后立即调用 online softmax

`flash_fwd_kernel.h` 里，score tile 经过 QK GEMM、可选 softcap、mask 后，马上进入 `softmax_rescale_o`。内核逆序扫描 K/V block，因此 `masking_step == 0` 对应本 query tile 实际访问的第一块，而不是序号最小的 K block。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L319-L344
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
```

`softmax_rescale_o` 返回后，`acc_s` 仍是 FP32 指数分子。接下来才转成计算 `P @ V` 所需的低精度 fragment；可选的 `return_softmax` 与 dropout 都发生在这个短命副本上。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L346-L367
// Convert acc_s from fp32 to fp16/bf16
Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
int block_col_idx = n_block * (kBlockN / 32);
if (Return_softmax) {
    Tensor rP_drop = make_fragment_like(rP);
    cute::copy(rP, rP_drop);
    dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
        rP_drop, block_row_idx, block_col_idx, kNWarps
    );
    cute::copy(rP_drop, tSgS);
    tSgS.data() = tSgS.data() + (-kBlockN);
}
if (Is_dropout) {
    dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
}

// Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
// if (cute::thread0()) { print(tOrP); }
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

这条顺序说明权重 tile 的生命周期很短：`acc_s` 原地变成指数分子，转成 `rP`，随后立刻被 `gemm_rs` 消费。若启用 dropout，mask 只改写送入 `P @ V` 的 `rP`，`row_sum` 仍统计未 dropout 的 softmax 分母；epilogue 再用 `rp_dropout = 1 / (1-p)` 补上期望保持缩放。`Return_softmax` 则复制一份 `rP_drop` 写入 `p_ptr`，这是测试/调试返回路径，不是 backward 所需的持久状态。常规路径不会把完整权重矩阵写出去。

## 6. Epilogue 把状态落成 `O` 和 LSE

主循环结束后，`normalize_softmax_lse` 补齐 lane 间 `row_sum`，归一化 `acc_o`，并返回每行 LSE。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L169-L185
template<bool Is_dropout=false, bool Split=false, typename Tensor0>
__forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
    SumOp<float> sum_op;
    quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = row_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
        float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
    }
    return lse;
```

调用点在 forward epilogue：

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L433
// Epilogue

Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
```

`softmax_scale_log2` 用于 `exp2` 路径，数值上包含从自然指数到底数 2 指数的换算；`softmax_scale` 用于最终 LSE 的自然对数表达。读代码时不要把这两个字段当成可互换的同一参数。

这里还有一个很容易误读的空行协议：若一行全被 mask，`row_sum == 0`，非 split forward 把 LSE 写成 `+inf`，输出分子保持为零。这个 `+inf` 不是数学上的 `log(0)=-inf`，而是工程哨兵：backward 计算 `exp(score - LSE)` 时自然得到零。split-KV 的局部空 split 则写 `-inf`，以便后续跨 split 的 log-sum-exp 合并；两条路径不能混讲。

## 7. Python 保存 LSE，让 backward 不保存完整 `P`

CUDA forward 返回 `softmax_lse` 后，Python autograd 保存它。Backward 取回 LSE、`out`、Q/K/V 和 RNG，再交给 CUDA backward 重算概率。

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

Backward 的 Python 协议明确取回这些紧凑状态：

```python
# 来源：flash_attn/flash_attn_interface.py L880-L907
@staticmethod
def backward(ctx, dout, *args):
    q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    head_size_og = dout.size(3)
    dout_padded = dout
    if head_size_og % 8 != 0:
        dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
    _wrapped_flash_attn_backward(
        dout_padded,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        ctx.dropout_p,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size[0],
        ctx.window_size[1],
        ctx.softcap,
        ctx.alibi_slopes,
        ctx.deterministic,
        rng_state=rng_state,
    )
```

不能只凭 Python 参数名推断“重算”。CUDA backward 先从 `softmax_lse_ptr` 取每行 LSE，再把重算得到的 score 转成 `exp(score * scale - LSE)`：

```cpp
// 来源：csrc/flash_attn/src/flash_bwd_kernel.h L407-L412
Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
#pragma unroll
for (int mi = 0; mi < size(lse); ++mi) {
    const int row = get<0>(taccScS_row(mi));
    lse(mi) = Is_even_MN || row < binfo.actual_seqlen_q - m_block * kBlockM ? gLSE(row) : INFINITY;
}
```

```cpp
// 来源：csrc/flash_attn/src/flash_bwd_kernel.h L534-L536
// if (cute::thread(32, 0)) { print(scores); }
// Compute the exponential value.
FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
```

因此，`S_dmask` 是可选的测试/调试返回；训练 backward 的核心持久状态是 Q/K/V、`out`、`softmax_lse`，以及 dropout 时用于复现 mask 的 RNG state。代价是 backward 重新计算 score/概率，收益是 forward 不必把完整注意力矩阵留在 HBM。

## 运行验证

可以用一个纯 Python 小例子验证 streaming online softmax 与全量 softmax 一致：

```powershell
@'
import math

scores = [1.0, -2.0, 3.0, 0.5, 4.0, -1.0]
values = [2.0, 5.0, -1.0, 3.0, 0.25, 7.0]
blocks = [(scores[:2], values[:2]), (scores[2:4], values[2:4]), (scores[4:], values[4:])]

m = -math.inf
l = 0.0
o = 0.0
for s_block, v_block in blocks:
    block_m = max(s_block)
    new_m = max(m, block_m)
    scale_old = 0.0 if m == -math.inf else math.exp(m - new_m)
    l *= scale_old
    o *= scale_old
    for s, v in zip(s_block, v_block):
        p = math.exp(s - new_m)
        l += p
        o += p * v
    m = new_m

online = o / l
full_m = max(scores)
den = sum(math.exp(s - full_m) for s in scores)
full = sum(math.exp(s - full_m) * v for s, v in zip(scores, values)) / den
print(round(online, 12), round(full, 12), abs(online - full) < 1e-12)
'@ | python -
```

预期输出最后一列是 `True`。如果把 `o *= scale_old` 删除，这个验证会失败，这正是 `acc_o` 必须随 `row_sum` 一起重标尺的原因。

## 复盘

- `softmax_rescale_o` 是 online softmax 的核心状态机。
- `acc_s` 的身份会变化：raw score -> softcap/mask 后 score -> 以全局行 max 为标尺的未归一化指数分子。
- 当前块权重只在寄存器 fragment 中短暂存在；dropout 修改其低精度副本，随后 `gemm_rs` 立即消费。
- `softmax_lse` 是 forward 留给 backward 的紧凑摘要，不是附带统计。
