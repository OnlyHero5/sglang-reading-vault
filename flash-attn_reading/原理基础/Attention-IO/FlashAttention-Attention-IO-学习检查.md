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
updated: 2026-07-12
---
# Attention-IO · 学习检查

> 本页以基线 `002cce0` 的 FA2 为落点，检查你能否把公式、存储层级、例外路径和源码变量连成一条因果链，而不是检查你翻过多少页。

## 必达能力

- [ ] 能解释物化式 attention 的 score/P 为什么是 `Sq x Sk` HBM 风险点，以及何时才是 `N x N`。
- [ ] 能说明 FlashAttention 是 exact attention，不是近似 attention。
- [ ] 能解释 score 与未归一化指数权重仍会作为局部 tile 出现，但 standard 非测试路径不会把完整最终 P 长期保存到 HBM。
- [ ] 能说出 HBM、shared memory、register 在 forward 中分别保存什么。
- [ ] 能解释 `row_max/row_sum/acc_o` 为什么是跨 K/V blocks 的核心状态。
- [ ] 能解释 `softmax_lse` 为什么是 backward 需要的压缩状态。
- [ ] 能说明 `p_ptr` 和 `return_softmax` 为什么不能被误读成常规保存完整 `P`。
- [ ] 能区分测试 `S_dmask`、multi-split partial O/LSE 与完整最终概率矩阵。
- [ ] 能说明静态源码只能证明物化集合/copy 组织，实际 HBM traffic 与加速比必须由固定 workload 测量。

## 源码定位任务

| 任务 | 入口 | 通过标准 |
|------|------|----------|
| 找到 FA1/FA2 论文定位 | `README.md` | 能说明 FA1 是 IO-aware exact attention，FA2 是 parallelism/work partitioning 延续。 |
| 找到长期状态字段 | `flash.h` | 能指出 `o_ptr`、`softmax_lse_ptr`、`p_ptr`、`oaccum_ptr` 的差异。 |
| 找到 C++ 输出分配 | `flash_api.cpp` | 能说明 `softmax_lse` 总分配，`p` 只在可选路径分配。 |
| 找到 HBM copy layout | `kernel_traits.h` | 能指出 `GmemTiledCopyQKV/O` 与 128-bit copy。 |
| 找到 HBM view 和 tile | `flash_fwd_kernel.h` | 能把 `mQ/gQ/sQ` 分到 HBM view、HBM tile、shared memory。 |
| 找到局部权重生命周期 | `flash_fwd_kernel.h` 主循环 | 能说明 QK→softcap→mask→online softmax→未归一化 `rP`→`gemm_rs`→`acc_o` 的顺序。 |
| 找到 LSE 写回 | `flash_fwd_kernel.h` epilogue | 能指出 `normalize_softmax_lse` 和 `gLSE(row)=lse(mi)`。 |

源码定位依据：

- FA1/FA2 论文定位：来源：README.md L1-L15
- 参数结构：来源：csrc/flash_attn/src/flash.h L21-L143
- C++ 输出分配：来源：csrc/flash_attn/flash_api.cpp L420-L470
- HBM copy layout：来源：csrc/flash_attn/src/kernel_traits.h L111-L137
- HBM view 和 tile：来源：csrc/flash_attn/src/flash_fwd_kernel.h L138-L177
- 局部 score/指数权重生命周期：来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L367
- online softmax：来源：csrc/flash_attn/src/softmax.h L128-L189
- LSE 写回：来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494

## 静态验证

**操作：** 依次打开下列文件，用表格中的字段名定位；每找到一处，都先说清“它位于 HBM、shared memory 还是 register”，再勾选。

**预期：** 你最终能用源码证明两件事：完整最终 P 不会作为 standard 非测试 forward 的长期 HBM 中间结果保存，跨 K/V block 延续的是 online softmax 行状态与输出分子；同时能说出测试和 multi-split 例外。

- [ ] 在 `flash_api.cpp` 确认 `return_softmax ? p.data_ptr() : nullptr`。
- [ ] 在 `flash_fwd_kernel.h` 确认 `acc_s` 是局部 fragment，不是 HBM tensor。
- [ ] 在 `softmax.h` 确认后续 block 会重缩放旧 `row_sum` 和旧 `acc_o`。
- [ ] 在 `kernel_traits.h` 确认必要 HBM 访问的 copy atom/线程布局，而不从静态代码编造实测带宽。
- [ ] 在 epilogue 确认最终除 `row_sum` 后写回 `O/LSE`。
- [ ] 在 launch 中确认 `Return_softmax` 测试旁路和 `num_splits > 1` combine 条件。

若有 GPU profiler，再做动态验收：固定 B/H/Sq/Sk/D、dtype、causal、GPU、warmup 和计时范围，对比 `return_attn_probs` 开关与 standard/multi-split kernel，记录 dram bytes/throughput 和 kernel 时间。预期是观测结果与实际写回对象一致；不要求所有 shape 都得到同一加速倍数。

## 口述验收

用五分钟讲清楚：

> 物化式 attention 为什么会产生 `Sq x Sk` HBM 中间状态；FlashAttention 如何把 score/指数权重降级成 tile 内短生命周期对象；online softmax 如何用 `row_max/row_sum/acc_o` 保持 exact attention；源码中哪些位置证明 standard 非测试 forward 的主数值写回是 `O/LSE`，哪些例外会额外写 HBM。

讲不清 `row_max/row_sum`，回到 [[FlashAttention-Online-Softmax]]；讲不清源码变量存储层级，回到 [[FlashAttention-Attention-IO-数据流]]；讲不清 FA2 落地路径，进入 [[FlashAttention-FA2-Forward]]。
