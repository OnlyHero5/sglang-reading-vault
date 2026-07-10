---
title: "Hopper与CuTe · 学习检查"
type: exercise
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: practice
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/exercise
  - source-reading
updated: 2026-07-10
---
# Hopper与CuTe · 学习检查

## 读者能做什么

- [ ] 能说明 FA2、FA3、FA4 的源码目录、入口和实现方式差异。
- [ ] 能解释 FA3 为什么是 Hopper beta，而不是 FA2 改名。
- [ ] 能说明 FA3 schema 中 paged KV、RoPE、descale、scheduler metadata、SplitKV 的作用。
- [ ] 能解释 FA3 `run_mha_fwd` 中 arch、SplitKV、paged KV、PackGQA、softcap 这些 dispatch 维度。
- [ ] 能说明 SM90 mainloop 中 TMA/GMMA pipeline 和 producer/consumer 角色的意义。
- [ ] 能从 `flash_attn.cute.flash_attn_func` 追到 validation、kernel object、compile cache。
- [ ] 能解释 FA4 为什么需要 `_get_device_arch`、`FLASH_ATTENTION_ARCH`、compile key 和 `compile_cache`。
- [ ] 能指出 FP8 在 FA3/FA4 中的关键限制。
- [ ] 能说明 JIT cache 对 serving warmup 和形状稳定性的影响。

## 最小运行验收

这个检查不需要 GPU，只验证 FA4 arch override 解析入口能被 Python 层看见。需要已安装 FA4 相关依赖时才运行：

```powershell
$env:FLASH_ATTENTION_ARCH='sm_90'
python - <<'PY'
from flash_attn.cute.interface import _get_device_arch
print(_get_device_arch())
PY
```

预期现象：输出 `90`。如果 import 失败，说明当前环境没有 FA4/CuTeDSL 包；这不影响阅读源码，但意味着不能做运行层验证。

## 源码定位练习

| 问题 | 应定位到 |
|------|----------|
| FA3 beta 和硬件要求 | `README.md` 的 FlashAttention-3 beta release |
| FA3 dispatcher schema | `hopper/flash_api.cpp` 的 `TORCH_LIBRARY(flash_attn_3, m)` |
| FA3 arch/SplitKV/paged dispatch | `hopper/flash_api.cpp` 的 `run_mha_fwd` |
| SM90 TMA/GMMA pipeline | `hopper/flash_fwd_kernel_sm90.h` |
| FA4 公共导出 | `flash_attn/cute/__init__.py` |
| FA4 arch override | `flash_attn/cute/interface.py` 的 `_get_device_arch` |
| FA4 validation | `_flash_attn_fwd` 前半段 |
| FA4 kernel object 选择 | `arch // 10` 分支 |
| FA4 JIT cache | `_flash_attn_fwd.compile_cache` 与 `cute.compile` |

## 口述验收

用五分钟讲清楚：

> 为什么 FlashAttention 在 FA2 之外还需要 FA3/FA4 路径，以及 CuTeDSL/JIT cache 对新 GPU 架构适配和生产 serving 会带来哪些收益与风险。

合格答案必须包含：

- FA3 面向 H100/H800，使用 Hopper TMA/GMMA pipeline。
- FA3 schema 覆盖 paged KV、RoPE、FP8 descale、scheduler metadata、SplitKV 等 serving/training 组合。
- FA4 把能力校验和 kernel 选择放到 Python/CuTeDSL 层。
- FA4 的 compile cache 减少静态预编译组合，但引入首次编译延迟和 cache key 管理。
- FA3/FA4 不改变 IO-aware attention 与 online softmax 的核心原理。

## 收官

回到 [[FlashAttention-总结复盘]]，把 IO-aware 原理、online softmax、Python/C++/CUDA 绑定、KV cache、backward 和 FA3/FA4 演进串成一条完整 AI infra 知识链。
