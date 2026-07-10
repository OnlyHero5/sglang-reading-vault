---
title: "Hopper与CuTe · 核心概念"
type: concept
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/concept
  - source-reading
updated: 2026-07-10
---
# Hopper与CuTe · 核心概念

## 读者为什么要读

FA2 的主线已经说明了 Python/C++/CUDA template 如何跑 attention。FA3/FA4 要解决的是下一层工程压力：Hopper/Blackwell 的 TMA、GMMA、FP8、paged KV、SplitKV、MLA、block sparse 等组合越来越多，继续把所有组合都压在静态 `.cu` 实例里会越来越难维护。

这篇先建立三层模型：

| 层 | 入口 | 心理模型 | 主要代价 |
|----|------|----------|----------|
| FA2 | `flash_attn_2_cuda` | 稳定主包，C++ extension + CUDA template | wheel 编译和静态实例数量 |
| FA3 | `hopper/` | Hopper beta，把 attention 映射到 H100/H800 的 TMA/GMMA pipeline | 与主包集成、硬件要求更窄 |
| FA4 | `flash_attn/cute/` | CuTeDSL/JIT，把 kernel 形态对象化并按组合编译缓存 | 首次编译延迟和 cache 管理 |

## FA3 是 Hopper 专门路径，不是 FA2 改名

README 明确把 FA3 标成 Hopper beta，并列出 H100/H800 与 CUDA 要求。

```markdown
<!-- 来源：README.md L30-L47 -->
## FlashAttention-3 beta release
FlashAttention-3 is optimized for Hopper GPUs (e.g. H100).

This is a beta release for testing / benchmarking before we integrate that with
the rest of the repo.

Currently released:
- FP16 / BF16 forward and backward, FP8 forward

Requirements: H100 / H800 GPU, CUDA >= 12.3.
```

读 FA3 时，重点不是“API 名字换了”，而是 Hopper mainloop、tile scheduler、FP8 输出和 serving 特性被单独组织。

## FA3 仍是 C++/CUDA extension 思路

FA3 的 C++ dispatch 仍然使用 arch 和 feature switch，只是分支维度更贴近 Hopper serving 场景。

```cpp
// 来源：hopper/flash_api.cpp L367-L383
void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_splits >= 1);
    ARCH_SWITCH(params.arch, Arch, [&] {
        SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
            PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
                PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                    static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                        run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
                    });
                });
            });
        });
    });
}
```

这里的分叉告诉你 FA3 关注什么：架构、SplitKV、paged KV、GQA packing 和 softcap 都会影响 kernel 形态。

## FA4 把 kernel 选择前移到 Python/CuTeDSL

FA4 的包入口很薄，只公开普通和 varlen 两个函数。

```python
# 来源：flash_attn/cute/__init__.py L10-L18
from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
```

复杂度在 `interface.py`：它先检测架构、校验 head/dtype/FP8/backward 组合，再创建对应架构的 kernel object，并把编译结果放进 cache。

```python
# 来源：flash_attn/cute/interface.py L77-L92
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)
```

这就是 FA4 的调试入口：先确认架构检测，再确认 validation，再看 compile key 是否稳定。

## FA4 API 更可组合

FA4 公开入口已经把 MLA、top-k gather、learnable sink、score/mask modifier、block sparse 等参数放到函数签名里。

```python
# 来源：flash_attn/cute/interface.py L2709-L2732
def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    block_sparse_tensors_bwd: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
):
```

这说明 FA4 不是 FA2 API 的机械平移，而是把 attention backend 推向可组合、可编译的 DSL 生态。

## 复盘

1. FA3/FA4 的目标是新硬件和组合复杂度，不是替代 attention 原理。
2. FA3 仍像 C++/CUDA extension，FA4 则把 dispatch 和编译缓存显式放到 Python/CuTeDSL 层。
3. 生产排障要多看路径选择和编译缓存，不只看 kernel benchmark 数字。
