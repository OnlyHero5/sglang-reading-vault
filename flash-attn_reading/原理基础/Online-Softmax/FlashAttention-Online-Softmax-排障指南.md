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
updated: 2026-07-12
---
# Online-Softmax · 排障指南

> 本页以基线 `002cce0` 的 FA2 为准，按症状、原因、源码入口、动作和预期排障。标为“定位”的代码块是压缩骨架，不冒充原文；正式源码卡见 [[FlashAttention-Online-Softmax-源码走读]]。

## 排障入口

| 症状或误解 | 优先检查 | 源码入口 | 验证方式 |
|------------|----------|----------|----------|
| 以为 block-wise softmax 是局部归一化 | `row_max/row_sum` 是否跨 block 保留 | `Softmax<kNRows>` | 用 checkpoint 脚本对比全量 softmax |
| 输出数值与 reference 不一致 | `acc_o` 是否随 `row_sum` 一起 rescale | `softmax_rescale_o` 后续块路径 | 删除 `acc_o *= scale` 会立刻错 |
| backward 需要完整 `P` | 是否保存 LSE、Q/K/V、out、RNG | Python autograd context | 检查 `ctx.save_for_backward` 与 backward 重算入口 |
| dropout backward 不稳定 | RNG 是否能复现同一 16x32 block mask | forward kernel Philox 注释 | 固定 seed，对比 dropout 开关 |
| softcap + dropout 报错 | 当前 kernel 能力限制 | `flash_api.cpp` 检查 | 看 C++ 入口 `TORCH_CHECK` |

## 分块 softmax 为什么还能精确？

它不是把每个 block 单独归一化。源码维护每行的全局最大值和分母；新 block 来时，如果最大值变了，就把历史分母和历史输出一起迁移到新标尺。

```cpp
// 定位：csrc/flash_attn/src/softmax.h L146-L166（摘要/骨架）
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

这段是压缩定位骨架；完整来源：csrc/flash_attn/src/softmax.h L146-L166。

排障抓手：只要看到实现里没有跨 block 的 max/sum 状态，就不是这里定义的 exact streaming softmax。预期是 full/tiled reference 在数值容差内一致。

## 为什么新最大值出现时 `acc_o` 也要缩放？

`row_sum` 是分母账，`acc_o` 是输出分子账。二者必须在同一个 `row_max` 标尺下。新最大值变大后，历史指数权重都要乘 `exp(old_max-new_max)`；历史权重乘 V 的输出分子也要乘同一个比例。

最小反例：两块 score `[0]` 和 `[10]`。如果只缩放分母，不缩放第一块对输出的贡献，第一块的 V 权重会被严重高估。

## LSE 为什么比保存 `row_sum` 更适合 backward？

`row_sum` 只有和对应 `row_max` 一起才有意义。LSE 把二者合成稳定标量：

```text
LSE = row_max * softmax_scale + log(row_sum)
```

Backward 重算当前 tile 的已缩放 score 后，可以用 `exp(score_scaled - LSE)` 恢复局部概率权重。

```cpp
// 定位：csrc/flash_attn/src/softmax.h L169-L185（摘要/骨架）
lse(mi) = (sum == 0.f || sum != sum)
    ? (Split ? -INFINITY : INFINITY)
    : row_max(mi) * softmax_scale + __logf(sum);
```

上面的代码是压缩定位骨架；完整来源：csrc/flash_attn/src/softmax.h L169-L185。验证抓手：读 [[FlashAttention-Backward]] 时，确认 backward 主线使用 `softmax_lse` 与 Q/K/V/O 等状态重算，而不是读取完整 P。全 mask 时还要区分 non-split `+inf` 与 split partial `-inf` LSE 哨兵。

## `exp2` 路径是否改变了 softmax 数学？

没有。源码用 `exp2f(score * scale - max_scaled)` 表达 `exp((score - max) * softmax_scale)`。这是为了贴合 GPU 指令和 `log2(e)` 缩放路径。

```cpp
// 定位：csrc/flash_attn/src/softmax.h L65-L92（摘要/骨架）
const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
for (int ni = 0; ni < size<1>(tensor); ++ni)  {
    tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
}
```

上面的代码是压缩定位骨架；完整来源：csrc/flash_attn/src/softmax.h L65-L92。排障抓手：区分 `scale_softmax_log2` 和 `scale_softmax`。前者用于 kernel 内 `exp2`，后者用于 LSE 的自然对数表达。预期是与自然指数 reference 等价，而不是逐位相同。

## `Return_softmax` 是否说明主路径保存了完整 `P`？

不是。`Return_softmax` 是 dropout 测试/调试旁路，源码复制一份 `rP`、用符号编码 dropout 后写出；常规路径的 `rP` 直接进入 `gemm_rs`。`S_dmask` 缩放不保证等于最终概率。来源：flash_attn/flash_attn_interface.py L1052-L1062

```cpp
// 定位：csrc/flash_attn/src/flash_fwd_kernel.h L346-L367（摘要/骨架）
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

上面的代码是压缩定位骨架；完整来源：csrc/flash_attn/src/flash_fwd_kernel.h L346-L367。验证抓手：生产性能路径不要依赖 `Return_softmax`；它不是 backward 所需状态。预期默认 `p_ptr=nullptr`。

## softcap 和 dropout 为什么可能组合失败？

这是当前实现限制，不是 attention 数学证明。C++ forward 入口明确拒绝 `softcap > 0` 且 `p_dropout > 0` 的组合。

```cpp
// 定位：csrc/flash_attn/flash_api.cpp L392-L397（摘要/骨架）
TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }
```

上面的代码是压缩定位骨架；完整来源：csrc/flash_attn/flash_api.cpp L392-L397。排障抓手：看到这种错误时，先区分“算法概念不可行”和“当前 kernel 未实现该组合”。预期失败发生在 launch 前，关闭其中一个特性后才继续 dispatch。

## causal/local 最右侧 tile 为什么可能先全 mask？

症状：逆序扫描的第一个 K block 对某些 query 行完全不可见，朴素实现出现 `-inf - (-inf) = NaN`。

原因：FA2 standard kernel 从可见 K 范围右端向左扫描，边界 tile 先进入 masking loop；`softmax_rescale_o` 的 `Check_inf` 会把全 mask 行的当前 max 兜底为 0，避免重标尺 NaN。来源：csrc/flash_attn/src/flash_fwd_kernel.h L267-L344；来源：csrc/flash_attn/src/softmax.h L146-L166

操作：用 `Sq < Sk` 的 bottom-right causal case，打印每个逆序 tile 的 keep mask，并对比带/不带全 mask 兜底的 tiled reference。

预期：带兜底的结果 finite 且与 full reference 一致；不能把全 mask tile 当作一块普通局部 softmax。

## 复盘迁移

- Online softmax 的核心错误边界是“分母和输出分子是否同标尺”。
- LSE 是 `row_max + row_sum` 的压缩协议，不是可有可无的 debug 输出。
- Dropout 和 softcap 会带来实现约束，但不改变 online softmax 的主公式。
