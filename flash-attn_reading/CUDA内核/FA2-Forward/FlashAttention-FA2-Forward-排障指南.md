---
title: "FA2-Forward · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "FA2-Forward"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# FA2-Forward · 排障指南

> 本页以基线 `002cce0` 的 FA2 CUDA 为准，按“症状 → 可能原因 → 源码入口 → 动作 → 预期”排障。先确认实际走的是 standard、aligned split 还是 multi-split，再讨论 kernel 内部。

## 快速定位表

| 症状 | 优先源码入口 | 验证动作 |
|------|--------------|----------|
| 输入 shape/dtype 明明像 attention，却在进入 kernel 前失败 | `mha_fwd` 检查 | 对照 dtype、device、`stride(-1)`、head dim、GQA 关系。 |
| `return_attn_probs=True` 没拿到常规概率矩阵 | Python/C++ 输出分配 | 检查 dropout 是否打开；返回对象只用于测试，缩放不保证等于最终概率。 |
| 同样 API 换 head dim 后 kernel 或性能变了 | head-dim launch helper | 固定 B/H/Sq/Sk、dtype、dropout、causal 与 GPU，再比较对应 traits；不要仅凭 D 推断性能方向。 |
| dropout 开关改变了 kernel 形态 | `DROPOUT_SWITCH` | 对比 dropout/non-dropout traits；head_dim=64 会从 `128 x 128` 变成 `128 x 64`。 |
| softcap 和 dropout 不能同时开 | `mha_fwd` feature 约束 | 查 `softcap > 0` 时对 `p_dropout == 0` 的检查。 |
| causal 在 decode 小形状下似乎被优化掉 | `mha_fwd` causal/window 归一化 | 查 `seqlen_q == 1 && !alibi` 时 `is_causal=false`。 |
| 长 K/V 或 decode 时出现 SplitKV 路径 | `set_params_splitkv` 与 split launch | 查 `num_splits` heuristic、accum buffer、combine kernel。 |
| 调的是普通 `flash_attn_func`，profiler 却出现 combine kernel | fixed-length `mha_fwd` 也调用 `set_params_splitkv(..., num_splits=0)` | 记录 `B/H/Sq/Sk/D` 与 SM 数，静态检查 heuristic，而不是按 API 名猜 kernel |
| 原始 head dim 不是 8 的倍数却能调用成功 | Python `FlashAttnFunc.forward` 先 pad，C++ 只看到对齐后的 D | 对照 wrapper 的 pad 与返回前 slice；直接调底层 binding 仍会失败 |
| 怀疑 mask 语义不对 | `Mask::apply_mask` | 看 causal/local/alibi/非整齐边界如何写入 score tile。 |
| 以为 forward 保存了完整 P | kernel 主循环与 epilogue | 查 `acc_s/rP` 是未归一化指数权重，以及最终除 `row_sum` 发生在 epilogue。 |
| causal/local 下出现 NaN 或全 mask 行理解冲突 | `softmax_rescale_o` 的 `Check_inf` 与 epilogue | 确认整块/整行 mask；区分 non-split `+inf` 与 split 局部 `-inf` LSE 哨兵。 |
| `seqlen_k == 0` 时没有 kernel 事件 | `mha_fwd` empty-K 分支 | 确认入口直接写 `out=0`、`softmax_lse=+inf`，这不是 launch 丢失。 |

快速定位表依据：

- `mha_fwd` 输入检查：来源：csrc/flash_attn/flash_api.cpp L350-L395
- C++ 输出分配：来源：csrc/flash_attn/flash_api.cpp L441-L450
- head-dim launch helper：来源：csrc/flash_attn/src/flash_fwd_launch_template.h L195-L325
- dropout 分支：来源：csrc/flash_attn/src/flash_fwd_launch_template.h L202-L220
- softcap/dropout 与 causal/window 归一化：来源：csrc/flash_attn/flash_api.cpp L397-L408
- SplitKV 参数与 combine launch：来源：csrc/flash_attn/flash_api.cpp L257-L328
- SplitKV launch：来源：csrc/flash_attn/src/flash_fwd_launch_template.h L100-L191
- mask 实现：来源：csrc/flash_attn/src/mask.h L14-L205
- kernel 主循环：来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L367
- epilogue 写回：来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494

## 为什么源码里有大量 head_dim 文件？

症状：读源码时看到许多 head_dim 相关 `.cu` 或 helper，以为需要逐个看。

源码入口：`run_mha_fwd` 先用 `HEADDIM_SWITCH(params.d, ...)` 选 head dim 大类；`run_mha_fwd_hdim*` 再决定具体 traits。来源：csrc/flash_attn/flash_api.cpp L243-L255；来源：csrc/flash_attn/src/flash_fwd_launch_template.h L195-L325

验证方式：从 `params.d` 出发，只追对应 helper。例如 `D=64` 看 `run_mha_fwd_hdim64`；`D=128` 看 `run_mha_fwd_hdim128`。同时记录用户原始 D 是否被 Python pad，以及模板 `kHeadDim` 是否大于 `params.d`。

预期：能唯一落到一个 helper，并分清原始 D、`params.d`、`kHeadDim`；若仍不能解释 kernel 名，再检查 dtype、dropout、causal 与架构分支。

## 为什么 dropout 会改变 kernel 选择？

症状：训练时打开 dropout 后 profiler 里 kernel 或性能特征和 eval 不同。

源码入口：head_dim helper 内部用 `DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, ...)` 把是否 dropout 变成模板常量。head_dim=64 时，无 dropout 使用 `128 x 128`，有 dropout 使用 `128 x 64`。来源：csrc/flash_attn/src/flash_fwd_launch_template.h L202-L220

验证方式：分别用 `dropout_p=0.0` 和非零 dropout 跑同一 shape，观察 profiler kernel 名与 traits；比较性能时必须固定 GPU、dtype、shape、causal 与 warmup/计时方法。

预期：head_dim=64 的两组 traits 分别落到 `128 x 128` 与 `128 x 64`；性能数值只作为当前 workload 的结果，不外推成普遍阈值。

## 为什么 softcap 和 dropout 有限制？

症状：模型启用 softcap 后，再打开 dropout 直接报错。

源码入口：`mha_fwd` 在进入参数装配前检查 `softcap > 0.f` 时 `p_dropout == 0.f`。来源：csrc/flash_attn/flash_api.cpp L397-L404

验证方式：不要去 kernel 主循环找 bug。这个组合在 C++ 入口已被拒绝；如果上层模型同时启用两者，需要换 backend、关 dropout，或等待该组合有完整 kernel 与测试支持。

预期：失败发生在 launch 前；关闭其中一个特性后才能继续进入参数装配与 dispatch。

## 为什么 causal mask 有时会被关闭？

症状：decode 场景 `seqlen_q == 1` 时，期望 causal 分支却看不到 causal kernel。

源码入口：当 `seqlen_q == 1` 且没有 ALiBi，源码认为 causal 与非 causal 等价，直接把 `is_causal` 改成 false；如果仍是 causal，则把 `window_size_right` 置 0。来源：csrc/flash_attn/flash_api.cpp L399-L408

验证方式：对比 `seqlen_q=1` 与 `seqlen_q>1`，并分别记录有无 ALiBi 时最终 `params.is_causal` 与 kernel 实例。

预期：仅 `seqlen_q == 1 && !alibi` 的情形被归一化为 non-causal；不要把它解释成所有 decode 都关闭 causal。

## SplitKV 为什么需要 combine？

症状：长 K/V 或 batch/head 并行度不足时，forward 出现额外 combine kernel。

源码入口：`set_params_splitkv` 会用 SM 数和 K blocks 估算 `num_splits`，若 split 大于 1，会分配 `softmax_lse_accum` 和 `out_accum`。split launch 完成后，再调用 `flash_fwd_splitkv_combine_kernel` 合并 partial LSE/O。来源：csrc/flash_attn/flash_api.cpp L257-L328；来源：csrc/flash_attn/src/flash_fwd_launch_template.h L100-L160

验证方式：观察 `num_splits` 和 `force_split_kernel`。`num_splits > 1` 才应看到 partial buffer 与 combine；强制 split 且 `num_splits == 1` 应进入 aligned kernel，但不出现 combine。

预期：profiler 事件与这三类状态一致。是否更快必须在固定 B/H/Sq/Sk/D、GPU 和计时方法下实测；源码只证明并行度与额外 HBM 代价，不保证收益方向。

不要把 SplitKV 限定成 KV-cache 专用路径。当前 fixed-length `mha_fwd` 也以自动模式调用 heuristic；当 `batch × heads × query blocks` 不足以填满 SM、而 K/V blocks 足够多时，普通 API 同样可能得到 `num_splits>1`。

## 为什么完整 attention matrix 不在主路径里？

症状：想从 forward 中直接拿到完整 `P` 做调试或可视化。

源码入口：主循环里的 `acc_s` 和 `rP` 是局部未归一化指数权重；epilogue 才除最终 `row_sum` 并写 `O/LSE`。可选 `p/S_dmask` 只在受限 dropout 测试路径写出，符号位编码 mask，缩放不保证等于最终概率。来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L367；来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494；来源：csrc/flash_attn/flash_api.cpp L441-L450；来源：flash_attn/flash_attn_interface.py L1052-L1062

验证方式：在常规 `dropout_p=0.0` 和非零 dropout 场景分别检查返回值、C++ 分配和 tile 写出；不要把 `return_attn_probs` 当作生产路径或 backward 保存状态。

预期：常规路径不物化完整 P；测试旁路即使返回 `S_dmask`，也只能按接口声明解释为“可能使用不同缩放并编码 dropout 符号”的观测对象。

## 全 mask 与 empty-K 为什么不能混为一谈？

症状：看到 `LSE=+inf`、`LSE=-inf` 或 `O=0`，误判为同一种 softmax 数学结果。

可能原因：FA2 用不同哨兵服务不同阶段。standard/non-split 的全 mask 行在 `normalize_softmax_lse` 中写 `+inf`；split 局部行写 `-inf`，便于 combine 忽略无贡献 split；整个 `seqlen_k == 0` 则由 C++ 入口直接写 `O=0、LSE=+inf`，根本不 launch kernel。来源：csrc/flash_attn/src/softmax.h L169-L185；来源：csrc/flash_attn/flash_api.cpp L497-L504

操作：同时记录 `seqlen_k`、实际 kernel 路径、`num_splits` 与该行是否存在有效 key，不要只看 LSE 的符号。

预期：empty-K 没有 kernel 事件；全 mask standard 与 split partial 的 LSE 哨兵不同，但最终输出都不应凭哨兵本身解释成普通概率分布。
