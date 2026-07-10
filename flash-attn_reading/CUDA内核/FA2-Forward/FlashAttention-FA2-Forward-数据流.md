---
title: "FA2-Forward · 数据流"
type: dataflow
framework: flash-attn
topic: "FA2-Forward"
learning_role: reference
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/dataflow
  - source-reading
updated: 2026-07-10
---
# FA2-Forward · 数据流

> 本页不重复调用栈，而是追踪数据形态：同一份 Q/K/V 在 HBM、shared memory、寄存器、输出 HBM 中分别长什么样。

## 你为什么要读

FA2 forward 的对象会从 PyTorch tensor 变成 C++ 参数、模板常量和 CTA 内 fragment。本文专门追踪这些形态变化，说明 shape、stride、mask 和 head dim 在哪里被固化；这样遇到 dispatch 错误或 kernel 数值问题时，能先找到参数第一次变形的边界。

## 一张图看数据流

```mermaid
flowchart LR
    HBM["HBM<br/>Q/K/V tensor"]
    SMEM["shared memory<br/>Q/K/V tile"]
    SCORE["register<br/>acc_s = QK"]
    MASK["register<br/>masked score"]
    SOFTMAX["register<br/>row_max,row_sum"]
    ACCO["register<br/>acc_o"]
    OUTSMEM["shared memory<br/>O tile"]
    OUTHBM["HBM<br/>O + LSE"]
    HBM --> SMEM --> SCORE --> MASK --> SOFTMAX --> ACCO --> OUTSMEM --> OUTHBM
    ACCO --> SCORE
```

循环箭头表示同一个 query block 会不断扫描新的 K/V block。`acc_o` 不会在每个 K/V block 后写回 HBM，而是留在寄存器里持续累积。

## 生命周期表

| 数据 | 形态 | 位置 | 生命周期 |
|------|------|------|----------|
| 原始 Q/K/V | `(B,S,H,D)` tensor | HBM | 整个调用 |
| Q tile | `kBlockM x kHeadDim` | shared memory / register | 当前 CTA |
| K/V tile | `kBlockN x kHeadDim` | shared memory | 当前 K/V block |
| `acc_s` | score fragment | register | 当前 Q block x 当前 K block |
| `row_max/row_sum` | 每行 softmax 状态 | register | 当前 Q block 扫完所有 K block |
| `acc_o` | output accumulator | register | 当前 Q block 扫完所有 K block |
| `O` tile | `kBlockM x kHeadDim` | shared memory -> HBM | epilogue |
| `LSE` | 每个 query row 一个 fp32 | HBM | forward 结束后保留 |

表格依据：

- 原始 Q/K/V 指针与 stride：来源：csrc/flash_attn/src/flash.h L21-L44
- Q/K/V shared memory layout：来源：csrc/flash_attn/src/kernel_traits.h L79-L109
- Q tile 进入 kernel：来源：csrc/flash_attn/src/flash_fwd_kernel.h L250-L288
- K/V tile 加载：来源：csrc/flash_attn/src/flash_fwd_kernel.h L267-L317
- score tile 与 mask 前后状态：来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L330
- online softmax 状态：来源：csrc/flash_attn/src/softmax.h L128-L189
- `acc_o` 累积：来源：csrc/flash_attn/src/flash_fwd_kernel.h L283-L367
- O/LSE 写回：来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494
- LSE 行写入：来源：csrc/flash_attn/src/flash_fwd_kernel.h L433-L477

## Q/K/V 的第一次变形：tensor 到 tile

C++ 入口保存的是 tensor 指针和 stride；kernel traits 定义的是 tile layout。`Flash_fwd_kernel_traits` 用 `SmemLayoutQ`、`SmemLayoutKV` 和 `GmemTiledCopyQKV` 描述 Q/K/V 如何从 HBM 搬到 shared memory。来源：csrc/flash_attn/src/kernel_traits.h L79-L137

在 `compute_attn` 开始阶段，kernel 将 Q tile 和 K tile 异步 copy 到 shared memory；如果 traits 要求 Q in registers，还会把 Q 从 shared memory copy 到寄存器视图。来源：csrc/flash_attn/src/flash_fwd_kernel.h L250-L281

这一步之后，读者要把 Q/K/V 从“完整 tensor”切换成“当前 CTA 看到的 tile”。

## Score tile 的生命周期很短

`acc_s` 是当前 Q tile 与当前 K tile 的 `QK^T` 结果。源码每轮都会创建并清零 `acc_s`，做 GEMM，应用 softcap 和 mask，然后立刻进入 online softmax。来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L344

`acc_s` 不会长期保存，也不会作为完整 attention matrix 写回。它在当前 K/V block 结束前会被转成 `rP`，用于 `P @ V`。来源：csrc/flash_attn/src/flash_fwd_kernel.h L346-L367

## `row_max/row_sum` 是跨 block 的连接器

如果只看一个 K/V block，softmax 很简单；难点在于一个 query row 的 softmax 跨越所有 K blocks。`Softmax` 通过 `row_max` 和 `row_sum` 保存“已经扫过的 K/V blocks”的归一化状态。后续 block 到来时，源码先比较新旧 row max，再重缩放旧 `row_sum` 和旧 `acc_o`。来源：csrc/flash_attn/src/softmax.h L136-L167

最后 `normalize_softmax_lse` 计算 LSE，并把 `acc_o` 除以最终 row sum。来源：csrc/flash_attn/src/softmax.h L169-L189

这就是为什么 `LSE` 可以替代完整概率矩阵成为 backward 的关键状态。

## Mask 发生在 score tile 内

`Mask::apply_mask` 把几类逻辑折到同一个 tile 修改步骤里：普通越界、causal、local window、ALiBi、非整齐 M/N 边界。普通越界按列号超过 `max_seqlen_k` 写 `-INFINITY`；causal 是右边界受限的 local mask；local window 同时限制左右边界。来源：csrc/flash_attn/src/mask.h L14-L205

因此 local attention、causal、ALiBi 不是“算完 attention 以后再处理”的后处理。它们会改变每个 score tile 在 softmax 前的数值。

## 输出写回只发生在 epilogue

主循环完成后，epilogue 才把 `acc_o` 转成 fp16/bf16，并借助 shared memory 的 O tile 写回 global memory。LSE 则按 row 写入 `softmax_lse`。来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494

最终状态：

| 输出 | 位置 | 为什么保留 |
|------|------|------------|
| `out` | HBM | attention 的真正输出。 |
| `softmax_lse` | HBM | backward 和测试需要的归一化因子。 |
| `p/S_dmask` | HBM，可选 | 只在测试/dropout路径中作为观测对象。 |

## 与上层 serving 的关系

SGLang/vLLM 这类 serving 系统把 prefill 或 decode 的 attention 需求交给 backend。FA2 forward 解释了 backend 为什么重视 IO：长 prompt 下，性能瓶颈经常不是“会不会算 QK”，而是“score/probability 能不能不在 HBM 里反复读写”。

这也是从 FA2 forward 过渡到 [[FlashAttention-KV-Cache]] 的理由：decode 场景会把 K/V 变成 cache，问题从“怎么不保存完整 P”继续推进到“怎么高效读取历史 KV”。
