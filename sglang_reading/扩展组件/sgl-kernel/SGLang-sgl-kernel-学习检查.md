---
title: "sgl-kernel · 学习检查"
type: exercise
framework: sglang
topic: "sgl-kernel"
learning_role: practice
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/exercise
  - source-reading
updated: 2026-07-10
---

# sgl-kernel · 学习检查

## 你为什么要做这组检查

这组检查判断你是否能从 SRT 层调用追到 Python wrapper、`torch.ops` 注册和 architecture-specific 动态库，而不是只知道“这里有 CUDA 算子”。

## 能力检查

- [ ] 能说明 sgl-kernel 与 SRT、FlashInfer、Triton 的边界。
- [ ] 能画出 SRT layer → `sgl_kernel` Python → `torch.ops` → 动态库的调用链。
- [ ] 能解释 `_load_architecture_specific_ops` 如何决定加载产物。
- [ ] 能从一个 MoE、attention 或 sampling 调用点追到对应 wrapper 和注册名。
- [ ] 能指出 GPU 架构、ABI 或算子缺失分别会在哪个边界暴露。

## 最小验证

操作：

```powershell
rg -n "_load_architecture_specific_ops|torch\.ops\.sgl_kernel|merge_state_v2|moe_align_block_size" sglang/sgl-kernel
```

预期：同时看到动态库加载、算子注册或调用包装，以及至少一个 SRT 会消费的公开算子。若只命中 Python 名称而没有注册/加载证据，转到 [[SGLang-sgl-kernel-排障指南]] 检查构建和 import 边界。

## 复盘

深读调用链见 [[SGLang-sgl-kernel-源码走读]]，张量和跨层边界见 [[SGLang-sgl-kernel-数据流]]。
