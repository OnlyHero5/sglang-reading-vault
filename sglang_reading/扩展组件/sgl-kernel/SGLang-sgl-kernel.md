---
title: "sgl-kernel"
type: map
framework: sglang
topic: "sgl-kernel"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-10
---
# sgl-kernel

> **源码范围：** `sgl-kernel/python/sgl_kernel/`（Python 薄封装）+ `sgl-kernel/csrc/`（CUDA/C++ 算子实现） 
> **Git 基线：** `70df09b` 
> **前置专题：** [[SGLang-Quantization]] · **下一专题：** [[SGLang-model-gateway]]

---

## 1. 本模块目标

专题读法：`sgl-kernel` 是 SGLang Runtime（srt）的**底层算子库**，独立于 Python 推理逻辑。它把 attention、MoE 路由、量化 GEMM、KV cache 搬运、投机解码、采样等热点路径编译为 CUDA custom op，通过 `torch.ops.sgl_kernel.*` 暴露给 srt 的 model executor 与 scheduler。本模块帮助读者理解算子如何按 GPU 架构加载、Python 层如何 dispatch、以及各子模块在推理栈中的位置。

**源码锚点：**

```python
## 来源：sgl-kernel/python/sgl_kernel/__init__.py L8-L30
if sys.platform == "darwin" and platform.machine() == "arm64":
    from sgl_kernel.metal import *
else:
    import torch
    from sgl_kernel.debug_utils import maybe_wrap_debug_kernel
    from sgl_kernel.load_utils import (
        _load_architecture_specific_ops,
        _preload_cuda_library,
    )

    # Initialize the ops library based on current GPU
    common_ops = _load_architecture_specific_ops()

    # Preload the CUDA library to avoid the issue of libcudart.so.12 not found
    if torch.version.cuda is not None:
        _preload_cuda_library()

    from sgl_kernel.allreduce import *
    from sgl_kernel.attention import (
        cutlass_mla_decode,
        cutlass_mla_get_workspace_size,
        merge_state_v2,
    )
```

读法：

- **macOS arm64** 走 Metal 分支，跳过 CUDA op 加载。
- **非 Apple 平台** 先加载 `common_ops` 动态库，再 re-export 各子模块函数。
- `_preload_cuda_library()` 解决 `libcudart.so` 找不到的运行时链接问题。

---

## 2. 在全局架构中的位置

```
srt ModelExecutor / AttentionBackend / MoE layer
 │ import sgl_kernel.*
 ▼
sgl-kernel/python/sgl_kernel/*.py ← 本模块（Python 薄封装 + 参数校验）
 │ torch.ops.sgl_kernel.*
 ▼
sgl-kernel/csrc/ + sm90|sm100/common_ops.so ← CUDA/C++ 实现
```

| 子模块 | 典型调用方 | 职责 |
|--------|-----------|------|
| `attention` | MLA / paged attention | CUTLASS MLA decode、merge state |
| `moe` | MoE 层 | topk 路由、token 对齐、expert sum |
| `gemm` | 量化线性层 | FP8/INT8/AWQ/GPTQ 矩阵乘 |
| `kvcacheio` | PD disaggregation | 跨 worker KV 搬运 |
| `speculative` | 投机解码 | 树采样、greedy verify |
| `sampling` | 采样后处理 | top-k/p renorm |
| `allreduce` | TP 通信 | 自定义 allreduce（ROCm） |

---

## 3. 自测与验收标准

- [ ] 能说明 `common_ops` 按 SM90/SM100 加载的逻辑
- [ ] 能追踪一条 attention 算子从 Python 到 `torch.ops` 的 dispatch 路径
- [ ] 能列举 MoE / GEMM / KV / speculative 四类算子及其 srt 用途
- [ ] 能为一个真实算子指出 Python wrapper、`torch.ops` 注册、C++/CUDA 实现和调用方，并说明当前硬件为何选择该路径

---

→ 核心概念：[[SGLang-sgl-kernel-核心概念]] · 源码走读：[[SGLang-sgl-kernel-源码走读]]
