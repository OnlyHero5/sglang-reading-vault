---
title: "Attention-IO · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "Attention-IO"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# Attention-IO · 排障指南

> 本页以基线 `002cce0` 的 FA2 为准，按“症状 → 可能原因 → 源码入口 → 动作 → 预期”排查 Attention IO。先区分 standard、测试 `S_dmask` 与 multi-split partial 写回。

## 快速定位表

| 症状 | 源码入口 | 验证动作 |
|------|----------|----------|
| 以为 FlashAttention 是近似 attention | `softmax_rescale_o` 与 `normalize_softmax_lse` | 检查 `row_max/row_sum` 如何合并多个 K/V block。 |
| 以为完全不计算 softmax 权重 | `flash_fwd_kernel.h` 主循环 | 检查 `acc_s -> rP -> gemm_rs`，确认局部指数权重被计算并消费，但尚未除最终分母。 |
| 以为 `p_ptr` 表示常规保存完整 attention matrix | `mha_fwd` 输出分配 | 检查 `p` 只在 `return_softmax` 且 dropout 路径分配。 |
| 只看显存峰值，或反过来凭源码猜精确 traffic | 参数分配、kernel 写回、profiler | 先证明物化集合，再用固定 workload 的 profiler 测实际 transaction/带宽。 |
| 看到额外 HBM 写回就断言“保存了完整 P” | `Return_softmax` 与 SplitKV launch | 区分测试 `S_dmask`、partial O/LSE 和最终概率矩阵。 |
| 不知道 LSE 为什么必须写回 | epilogue 与 softmax | 检查 `normalize_softmax_lse` 和 `gLSE(row)=lse(mi)`。 |
| 读 kernel 时分不清变量在哪个存储层 | HBM view / tile / smem / register 变量 | 先把 `m*`、`g*`、`s*`、`acc_*` 分层，再看计算。 |

快速定位依据：

- online softmax：来源：csrc/flash_attn/src/softmax.h L128-L189
- tile 主循环：来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L367
- C++ 输出分配：来源：csrc/flash_attn/flash_api.cpp L420-L470
- copy layout：来源：csrc/flash_attn/src/kernel_traits.h L111-L137
- epilogue 写回：来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494
- HBM view 与 tile：来源：csrc/flash_attn/src/flash_fwd_kernel.h L138-L177

## FlashAttention 是近似 attention 吗？

不是。它改变的是执行顺序和内存访问方式，不改变 softmax attention 的数学定义。关键是 `row_max/row_sum`：后续 K/V block 到来时，源码会用新旧最大值差重缩放旧 `row_sum` 和旧 `acc_o`，再合并当前 block。来源：csrc/flash_attn/src/softmax.h L136-L167

验证方式：读 `softmax_rescale_o`，确认它不是对每个 block 单独归一化后直接拼接，而是在维护整行 softmax 的累计最大值、指数和与输出分子。

预期：能用同一行的 full softmax reference 解释 tiled 结果；若每块概率和都被强制为 1 且不重缩放旧状态，模型就是错的。

## FlashAttention 完全不计算 `P` 吗？

不是。更准确的说法是：不把完整最终 P 作为常规长期 HBM 矩阵保存。kernel 内部会把当前 score tile `acc_s` 经 online softmax 改写成以全局行最大值为标尺的指数分子，再转成 `rP`，立刻用 `gemm_rs(acc_o, rP, V)` 累积输出分子。来源：csrc/flash_attn/src/flash_fwd_kernel.h L341-L367；来源：csrc/flash_attn/src/softmax.h L136-L167

验证方式：找 `Tensor rP = ...`、`gemm_rs` 和 epilogue 的 `normalize_softmax_lse`，确认最终除 `row_sum` 不在 tile GEMM 之前。

预期：`rP` 立即被消费，`acc_o` 到 epilogue 才归一化；不能把变量名 P 当成“已完成整行 softmax”的证据。

## `p_ptr` 存在是否说明仍保存完整 attention matrix？

不能这么判断。`p_ptr` 是参数结构支持的可选路径，常规路径是否保存完整 P 要看 C++ 入口是否分配 `p` 并传非空指针。`mha_fwd` 总是分配 `softmax_lse`，但 `p` 只在 `return_softmax` 为真时分配，并要求 `p_dropout > 0`。来源：csrc/flash_attn/flash_api.cpp L420-L470

验证方式：沿 `p = torch::empty(...)` 到 `return_softmax ? p.data_ptr() : nullptr`，再看 launch 的 `ReturnSoftmaxConst && Is_dropout` 与接口对 `S_dmask` 缩放/符号位的声明。

预期：默认路径 `p_ptr=nullptr`；测试旁路只在 dropout 条件下实例化，返回对象不应被当作生产最终概率或 backward 保存状态。

## 为什么不能只比较 FLOPs？

在物化式 self-attention 中，完整 score/P 是 `N x N`；一般形状则是 `Sq x Sk`。FlashAttention 的取舍是用 tile 内重算/重缩放减少这类中间态的片外往返。但“HBM 一定是当前瓶颈”仍需给出 GPU、shape、dtype、前后向范围与 profiler 证据，不能由 Tensor Core 很快这一句直接推出。

源码里，必要的 Q/K/V/O HBM 访问仍被认真优化：`kernel_traits.h` 用 128-bit copy、`GmemLayoutAtom`、`GmemTiledCopyQKV/O` 组织访问。来源：csrc/flash_attn/src/kernel_traits.h L111-L137

验证方式：先从分配与写回证明“不物化哪些张量”，再固定 B/H/Sq/Sk/D、dtype、causal、GPU、warmup 与计时范围，记录 dram bytes/throughput、kernel 时间和必要时的 occupancy。

预期：静态证据只能证明状态边界与 copy 组织；实际是否 memory-bound、节省多少 traffic、加速多少由该 workload 的动态数据回答。

## LSE 为什么是长期输出？

`LSE` 是每个 query row 的 log-sum-exp，也就是完整 softmax 归一化因子的压缩表示。epilogue 中 `normalize_softmax_lse` 从 `row_max/row_sum` 计算 LSE，并用 `1 / row_sum` 归一化 `acc_o`；随后源码把每行 `lse` 写入 `gLSE`。来源：csrc/flash_attn/src/softmax.h L169-L189；来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494

验证方式：看 `gLSE(row) = lse(mi)`，再回看 backward 如何读取 LSE 并重算局部权重。不要说“用 LSE 直接归一化 forward 输出”，源码使用的是 `row_sum` 的倒数。

预期：能区分 forward epilogue 的 `row_sum` 归一化与 backward 的 LSE 重算标尺；全 mask 时还要识别 non-split `+inf` 和 split partial `-inf` 哨兵。

## 为什么要看 shared memory layout，而不是只看公式？

公式只能说明可以分块，不能证明某个实现已经高效。`Flash_fwd_kernel_traits` 把 `kBlockM/kBlockN/kHeadDim/kNWarps`、shared memory layout、copy atom 固化成编译期类型，说明实现如何组织搬运与 MMA；实际效率仍要结合架构、资源占用和 workload 测量。来源：csrc/flash_attn/src/kernel_traits.h L48-L137

验证方式：先读 traits 中的 `SmemLayoutQ/SmemLayoutKV/GmemTiledCopyQKV`，再读 kernel 主循环，最后用 profiler 对照实际 kernel、dram 指标与 occupancy。

预期：能说明“源码选择了什么 tile/copy 组织”和“当前 workload 得到了什么性能”是两类证据，不互相冒充。

## 额外写回一定是完整概率矩阵吗？

症状：profiler 看到额外 store 或 combine kernel，就断言 FlashAttention 已退化为保存完整 P。

可能原因：开启 `return_attn_probs` 时会写测试 `S_dmask`；`num_splits > 1` 时会写 partial O/LSE 并 combine。前者缩放不保证等于最终概率且符号位编码 dropout，后者形状属于 split×row×D/row 状态。来源：flash_attn/flash_attn_interface.py L1052-L1062；来源：csrc/flash_attn/src/flash_fwd_launch_template.h L101-L160

操作：记录 `return_softmax`、dropout、`num_splits`、buffer shape 和 kernel 名，再判断写回对象。

预期：standard 非测试路径没有这两类额外写回；测试或 multi-split 路径出现额外 HBM traffic，但不能据此称为完整最终 P。
