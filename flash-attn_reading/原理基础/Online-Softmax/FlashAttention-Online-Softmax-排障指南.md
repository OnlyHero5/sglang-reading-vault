---
title: "Online-Softmax · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "Online-Softmax"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# Online-Softmax · 排障指南

## 排障入口

| 症状或误解 | 优先检查 | 源码入口 | 验证方式 |
|------------|----------|----------|----------|
| 以为 block-wise softmax 是局部归一化 | `row_max/row_sum` 是否跨 block 保留 | `Softmax<kNRows>` | 用 checkpoint 脚本对比全量 softmax |
| 输出数值与 reference 不一致 | `acc_o` 是否随 `row_sum` 一起 rescale | `softmax_rescale_o` 后续块路径 | 删除 `acc_o *= scale` 会立刻错 |
| backward 需要完整 `P` | 是否保存了 `softmax_lse` | Python autograd context | 检查 `ctx.save_for_backward` |
| dropout backward 不稳定 | RNG 是否能复现同一 16x32 block mask | forward kernel Philox 注释 | 固定 seed，对比 dropout 开关 |
| softcap + dropout 报错 | 当前 kernel 能力限制 | `flash_api.cpp` 检查 | 看 C++ 入口 `TORCH_CHECK` |

## 分块 softmax 为什么还能精确？

它不是把每个 block 单独归一化。源码维护每行的全局最大值和分母；新 block 来时，如果最大值变了，就把历史分母和历史输出一起迁移到新标尺。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L146-L166
Tensor scores_max_prev = make_fragment_like(row_max);
cute::copy(row_max, scores_max_prev);
FLASH_NAMESPACE::template reduce_max</*zero_init=*/false>(scores, row_max);
Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
for (int mi = 0; mi < size(row_max); ++mi) {
    float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
    row_sum(mi) *= scores_scale;
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
        acc_o_rowcol(mi, ni) *= scores_scale;
    }
}
```

排障抓手：只要看到实现里没有跨 block 的 max/sum 状态，就不是 exact streaming softmax。

## 为什么新最大值出现时 `acc_o` 也要缩放？

`row_sum` 是分母账，`acc_o` 是输出分子账。二者必须在同一个 `row_max` 标尺下。新最大值变大后，历史概率都要乘 `exp(old_max-new_max)`；历史 `P @ V` 当然也要乘同一个比例。

最小反例：两块 score `[0]` 和 `[10]`。如果只缩放分母，不缩放第一块对输出的贡献，第一块的 V 权重会被严重高估。

## LSE 为什么比保存 `row_sum` 更适合 backward？

`row_sum` 只有和对应 `row_max` 一起才有意义。LSE 把二者合成稳定标量：

```text
LSE = row_max * softmax_scale + log(row_sum)
```

Backward 重算当前 tile 的 score 后，可以直接用 `exp(score - LSE)` 恢复 probability。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L169-L185
lse(mi) = (sum == 0.f || sum != sum)
    ? (Split ? -INFINITY : INFINITY)
    : row_max(mi) * softmax_scale + __logf(sum);
```

验证抓手：读 [[FlashAttention-Backward]] 时，确认 backward 主线使用的是 `softmax_lse`，不是完整 `P`。

## `exp2` 路径是否改变了 softmax 数学？

没有。源码用 `exp2f(score * scale - max_scaled)` 表达 `exp((score - max) * softmax_scale)`。这是为了贴合 GPU 指令和 `log2(e)` 缩放路径。

```cpp
// 来源：csrc/flash_attn/src/softmax.h L65-L92
const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
for (int ni = 0; ni < size<1>(tensor); ++ni)  {
    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
}
```

排障抓手：区分 `scale_softmax_log2` 和 `scale_softmax`。前者用于 kernel 内 `exp2`，后者用于 LSE 的自然对数表达。

## `Return_softmax` 是否说明主路径保存了完整 `P`？

不是。`Return_softmax` 是测试/调试旁路，源码复制一份 `rP` 再写出；常规路径的 `rP` 直接进入 `gemm_rs`。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L346-L367
Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
if (Return_softmax) {
    Tensor rP_drop = make_fragment_like(rP);
    cute::copy(rP, rP_drop);
    dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(rP_drop, block_row_idx, block_col_idx, kNWarps);
    cute::copy(rP_drop, tSgS);
}
if (Is_dropout) {
    dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
}
Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
```

验证抓手：性能路径不要依赖 `Return_softmax`；它不是 backward 所需状态。

## softcap 和 dropout 为什么可能组合失败？

这是当前实现限制，不是 attention 数学证明。C++ forward 入口明确拒绝 `softcap > 0` 且 `p_dropout > 0` 的组合。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L392-L397
TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }
```

排障抓手：看到这种错误时，先区分“算法概念不可行”和“当前 kernel 未实现该组合”。

## 复盘迁移

- Online softmax 的核心错误边界是“分母和输出分子是否同标尺”。
- LSE 是 `row_max + row_sum` 的压缩协议，不是可有可无的 debug 输出。
- Dropout 和 softcap 会带来实现约束，但不改变 online softmax 的主公式。
