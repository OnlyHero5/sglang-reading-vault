---
title: "FlashAttention 性能实验"
type: exercise
framework: flash-attn
topic: "Attention Kernel"
learning_role: practice
difficulty: intermediate
estimated_time: "90 到 180 分钟"
prerequisites:
  - "[[Attention算子主线]]"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# FlashAttention 性能实验

## 学习目标

先验证数值正确性，再观察 shape、mask、dtype 和 KV cache 如何改变 kernel 与 HBM 行为。

## 静态模式

```powershell
rg -n 'flash_attn_func|mha_fwd|run_mha_fwd|flash_fwd_kernel|softmax_rescale_o' flash-attn/flash-attention
```

预期：能定位 Python API、C++ 入口、dispatch、kernel 和 online softmax。

## 正确性基线

构造小 shape fp16/bf16 Q/K/V，比较 `flash_attn_func` 与 PyTorch SDPA。测试 causal 与 non-causal，并比较 forward 和 backward 误差。

预期：误差处于对应 dtype 和上游测试允许范围；任何性能比较都必须先通过这一步。

## Shape sweep

固定 batch 和 sequence，扫描 head dim、causal、dropout；再固定 head dim 扫描 sequence length。

记录：kernel 名称、执行时间、workspace、输出误差。

预期：不同 head dim 和开关会改变 specialization；结果不应被解释成一条统一性能曲线。

## Nsight Systems

```bash
nsys profile -o flash_attn_trace python your_flash_attn_benchmark.py
```

预期：时间线中能看到 extension/kernel launch；首次 JIT 或编译行为应与稳定迭代分开统计。

## Nsight Compute

```bash
ncu --set full --target-processes all python your_flash_attn_benchmark.py
```

重点观察：DRAM bytes、achieved bandwidth、Tensor Core/SM 利用率、occupancy、register 和 shared memory。

预期：优化判断同时引用 kernel 时间和访存指标；不能只看 occupancy 单一指标。

## KV cache 对照

比较 dense forward 与 `flash_attn_with_kvcache` 的小 `seqlen_q` 场景，并测试 paged KV / SplitKV 支持组合。

预期：decode 路径热点更偏向 KV load 和 split combine；不支持的参数组合应在 launch 前清晰失败。

## 通过标准

- [ ] PyTorch reference 正确性通过。
- [ ] 能解释至少两个 shape 为什么进入不同 kernel。
- [ ] 能从 profiler 读取 HBM 和 occupancy 证据。
- [ ] 能区分首次编译、warmup 和稳定测量。

