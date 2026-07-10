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
updated: 2026-07-10
---
# FA2-Forward · 排障指南

> 本页按排障方式读：先看症状，再看源码入口，最后给验证动作。不要把 FAQ 当作零散知识点。

## 快速定位表

| 症状 | 优先源码入口 | 验证动作 |
|------|--------------|----------|
| 输入 shape/dtype 明明像 attention，却在进入 kernel 前失败 | `mha_fwd` 检查 | 对照 dtype、device、`stride(-1)`、head dim、GQA 关系。 |
| `return_attn_probs=True` 没拿到常规概率矩阵 | C++ 输出分配 | 检查 dropout 是否打开；C++ 只在 `return_softmax && p_dropout > 0` 时分配 `p`。 |
| 同样 API 换 head dim 后性能变了 | head-dim launch helper | 看对应 `run_mha_fwd_hdim*` 选择的 `kBlockM/kBlockN/kNWarps`。 |
| dropout 开关改变了 kernel 形态 | `DROPOUT_SWITCH` | 对比 dropout/non-dropout traits；head_dim=64 会从 `128 x 128` 变成 `128 x 64`。 |
| softcap 和 dropout 不能同时开 | `mha_fwd` feature 约束 | 查 `softcap > 0` 时对 `p_dropout == 0` 的检查。 |
| causal 在 decode 小形状下似乎被优化掉 | `mha_fwd` causal/window 归一化 | 查 `seqlen_q == 1 && !alibi` 时 `is_causal=false`。 |
| 长 K/V 或 decode 时出现 SplitKV 路径 | `set_params_splitkv` 与 split launch | 查 `num_splits` heuristic、accum buffer、combine kernel。 |
| 怀疑 mask 语义不对 | `Mask::apply_mask` | 看 causal/local/alibi/非整齐边界如何写入 score tile。 |
| 以为 forward 保存了完整 P | kernel 主循环与 epilogue | 查 `acc_s/rP` 的短生命周期，以及最终只写 `out/LSE`。 |

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

验证方式：从你的 `D` 出发，只追对应 helper。例如 `D=64` 看 `run_mha_fwd_hdim64`；`D=128` 看 `run_mha_fwd_hdim128`。不要先读 generated 编译单元。

## 为什么 dropout 会改变 kernel 选择？

症状：训练时打开 dropout 后 profiler 里 kernel 或性能特征和 eval 不同。

源码入口：head_dim helper 内部用 `DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, ...)` 把是否 dropout 变成模板常量。head_dim=64 时，无 dropout 使用 `128 x 128`，有 dropout 使用 `128 x 64`。来源：csrc/flash_attn/src/flash_fwd_launch_template.h L202-L220

验证方式：分别用 `dropout_p=0.0` 和非零 dropout 跑同一 shape，观察 profiler kernel 参数或性能变化；静态阅读时比较两条 traits。

## 为什么 softcap 和 dropout 有限制？

症状：模型启用 softcap 后，再打开 dropout 直接报错。

源码入口：`mha_fwd` 在进入参数装配前检查 `softcap > 0.f` 时 `p_dropout == 0.f`。来源：csrc/flash_attn/flash_api.cpp L397-L404

验证方式：不要去 kernel 主循环找 bug。这个组合在 C++ 入口已被拒绝；如果上层模型同时启用两者，需要换 backend、关 dropout，或等待该组合有完整 kernel 与测试支持。

## 为什么 causal mask 有时会被关闭？

症状：decode 场景 `seqlen_q == 1` 时，期望 causal 分支却看不到 causal kernel。

源码入口：当 `seqlen_q == 1` 且没有 ALiBi，源码认为 causal 与非 causal 等价，直接把 `is_causal` 改成 false；如果仍是 causal，则把 `window_size_right` 置 0。来源：csrc/flash_attn/flash_api.cpp L399-L408

验证方式：对比 `seqlen_q=1` 与 `seqlen_q>1`，或加入 ALiBi，查看 dispatch 是否改变。

## SplitKV 为什么需要 combine？

症状：长 K/V 或 batch/head 并行度不足时，forward 出现额外 combine kernel。

源码入口：`set_params_splitkv` 会用 SM 数和 K blocks 估算 `num_splits`，若 split 大于 1，会分配 `softmax_lse_accum` 和 `out_accum`。split launch 完成后，再调用 `flash_fwd_splitkv_combine_kernel` 合并 partial LSE/O。来源：csrc/flash_attn/flash_api.cpp L257-L328；来源：csrc/flash_attn/src/flash_fwd_launch_template.h L100-L160

验证方式：观察 `num_splits` 是否大于 1；如果是，性能收益来自更多 K 维并行度，代价是额外 HBM 中间写回和 combine。

## 为什么完整 attention matrix 不在主路径里？

症状：想从 forward 中直接拿到完整 `P` 做调试或可视化。

源码入口：主循环里的 `acc_s` 和 `rP` 是局部 tile 状态；epilogue 只把 `O` 和 `LSE` 写回。完整 `p` 只在受限测试/dropout路径分配。来源：csrc/flash_attn/src/flash_fwd_kernel.h L301-L367；来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L494；来源：csrc/flash_attn/flash_api.cpp L441-L450

验证方式：在常规 `dropout_p=0.0` 场景检查返回值和 C++ 分配逻辑；不要把 `return_attn_probs` 当作生产路径。
