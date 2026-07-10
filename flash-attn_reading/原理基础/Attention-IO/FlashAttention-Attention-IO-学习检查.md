---
title: "Attention-IO · 学习检查"
type: exercise
framework: flash-attn
topic: "Attention-IO"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Attention-IO · 学习检查

> 本页检查你能否把公式、存储层级和源码变量连成一条因果链，而不是检查你翻过多少页。

## 必达能力

- [ ] 能解释标准 attention 的 `S` 和 `P` 为什么是 `N x N` HBM 风险点。
- [ ] 能说明 FlashAttention 是 exact attention，不是近似 attention。
- [ ] 能解释 `S/P` 在 FlashAttention 中仍会作为局部 tile 出现，但不会作为完整 HBM 矩阵长期保存。
- [ ] 能说出 HBM、shared memory、register 在 forward 中分别保存什么。
- [ ] 能解释 `row_max/row_sum/acc_o` 为什么是跨 K/V blocks 的核心状态。
- [ ] 能解释 `softmax_lse` 为什么是 backward 需要的压缩状态。
- [ ] 能说明 `p_ptr` 和 `return_softmax` 为什么不能被误读成常规保存完整 `P`。

## 源码定位任务

| 任务 | 入口 | 通过标准 |
|------|------|----------|
| 找到 FA1/FA2 论文定位 | `README.md` | 能说明 FA1 是 IO-aware exact attention，FA2 是 parallelism/work partitioning 延续。 |
| 找到长期状态字段 | `flash.h` | 能指出 `o_ptr`、`softmax_lse_ptr`、`p_ptr`、`oaccum_ptr` 的差异。 |
| 找到 C++ 输出分配 | `flash_api.cpp` | 能说明 `softmax_lse` 总分配，`p` 只在可选路径分配。 |
| 找到 HBM copy layout | `kernel_traits.h` | 能指出 `GmemTiledCopyQKV/O` 与 128-bit copy。 |
| 找到 HBM view 和 tile | `flash_fwd_kernel.h` | 能把 `mQ/gQ/sQ` 分到 HBM view、HBM tile、shared memory。 |
| 找到局部 `S/P` 生命周期 | `flash_fwd_kernel.h` 主循环 | 能说明 `acc_s -> rP -> gemm_rs -> acc_o` 的顺序。 |
| 找到 LSE 写回 | `flash_fwd_kernel.h` epilogue | 能指出 `normalize_softmax_lse` 和 `gLSE(row)=lse(mi)`。 |

源码定位依据：

- FA1/FA2 论文定位：来源：README.md L1-L15
- 参数结构：来源：csrc/flash_attn/src/flash.h L21-L143
- C++ 输出分配：来源：csrc/flash_attn/flash_api.cpp L420-L470
- HBM copy layout：来源：csrc/flash_attn/src/kernel_traits.h L111-L137
- HBM view 和 tile：来源：csrc/flash_attn/src/flash_fwd_kernel.h L138-L177
- 局部 `S/P` 生命周期：来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L367
- online softmax：来源：csrc/flash_attn/src/softmax.h L128-L189
- LSE 写回：来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494

## 静态验证

**操作：** 依次打开下列文件，用表格中的字段名定位；每找到一处，都先说清“它位于 HBM、shared memory 还是 register”，再勾选。

**预期：** 你最终能用源码证明两件事：完整 `S/P` 不会作为常规 forward 的长期 HBM 中间结果保存，跨 K/V block 延续的是 online softmax 的压缩状态。

- [ ] 在 `flash_api.cpp` 确认 `return_softmax ? p.data_ptr() : nullptr`。
- [ ] 在 `flash_fwd_kernel.h` 确认 `acc_s` 是局部 fragment，不是 HBM tensor。
- [ ] 在 `softmax.h` 确认后续 block 会重缩放旧 `row_sum` 和旧 `acc_o`。
- [ ] 在 `kernel_traits.h` 确认必要 HBM 访问仍被 vectorized/coalesced 组织。
- [ ] 在 epilogue 确认最终写回的是 `O` 和 `LSE`。

## 口述验收

用五分钟讲清楚：

> 标准 attention 为什么会把 `S/P` 变成二次方 HBM 中间状态；FlashAttention 如何把 `S/P` 降级成 tile 内短生命周期对象；online softmax 如何用 `row_max/row_sum/acc_o` 保持 exact attention；源码中哪些位置证明常规 forward 只长期保存 `O/LSE`。

讲不清 `row_max/row_sum`，回到 [[FlashAttention-Online-Softmax]]；讲不清源码变量存储层级，回到 [[FlashAttention-Attention-IO-数据流]]；讲不清 FA2 落地路径，进入 [[FlashAttention-FA2-Forward]]。
