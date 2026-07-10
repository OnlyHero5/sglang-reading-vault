---
title: "FlashAttention FA4 CuTeDSL 演进"
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
# FlashAttention FA4 CuTeDSL 演进

## 读者任务

这篇只回答 FA4 相对 FA3 的增量：为什么从 C++/CUDA beta 路径进一步走到 CuTeDSL/JIT，以及这个变化如何影响调试和生产部署。

## 增量一：FA4 是单独 CuTeDSL 包

FA4 README 明确把它描述为 Hopper/Blackwell 的 CuTeDSL 实现，并给出 `flash-attn-4` 安装入口。

```markdown
<!-- 来源：flash_attn/cute/README.md L1-L14 -->
# FlashAttention-4 (CuTeDSL)

FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

pip install flash-attn-4

If you're on CUDA 13, install with the `cu13` extra for best performance:

pip install "flash-attn-4[cu13]"
```

这说明 FA4 不是 FA3 文件夹里一个普通 backend flag，而是新的包和工具链入口。

## 增量二：公开 API 面很小，内部组合很多

包入口只导出两个函数；复杂度留在 `interface.py`。

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

但 `flash_attn_func` 签名已经暴露 qv、top-k gather、learnable sink、score/mask modifier、aux tensors、block sparse 等扩展点。

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

FA4 的设计取向是“公共入口少，内部编译组合多”。

## 增量三：能力校验在 Python 层快速失败

FA4 先校验 arch、head 维、GQA 整除、FP8/backward、输出/LSE shape 和 descale tensor 合法性。

```python
# 来源：flash_attn/cute/interface.py L446-L516
arch = _get_device_arch() if _arch is None else _arch
assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
alignment = 16 // v.element_size()
if arch // 10 not in [8, 12]:
    _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
if is_fp8 and requires_grad:
    raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
if is_fp8:
    assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
```

这把错误边界从底层编译器前移到 Python，排障时先看 validation。

## 增量四：kernel 形态对象化

FA4 根据 arch 创建不同 forward kernel object。对象携带 dtype、head dim、GQA、causal/local、tile、score/mask mod、paged KV 等编译期信息。

```python
# 来源：flash_attn/cute/interface.py L823-L961
if arch // 10 == 8:
    assert page_table is None, "paged KV not supported on SM 8.0"
    assert not is_split_kv, "SplitKV not supported on SM 8.0"
    fa_fwd = FlashAttentionForwardSm80(
        # 省略张量、stride、shape、dtype 与运行选项参数
    )
elif arch // 10 == 9:
    assert not is_split_kv, "SplitKV not supported on SM 9.0"
    fa_fwd = FlashAttentionForwardSm90(
        # 省略张量、stride、shape、dtype 与运行选项参数
    )
elif arch // 10 in [10, 11]:
    if qv is not None:
        fa_fwd = FlashAttentionMLAForwardSm100(
            # 省略 MLA 构造参数
        )
    else:
        fa_fwd = flash_fwd_obj_cls(
            # 省略通用 forward 构造参数
        )
elif arch // 10 == 12:
    assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
    assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
    assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
    fa_fwd = FlashAttentionForwardSm120(
        # 省略张量、stride、shape、dtype 与运行选项参数
    )
```

这相当于把 FA2/FA3 的 template specialization 意图搬到 Python 对象层表达。

## 增量五：组合爆炸从 build time 转到 JIT cache

compile key 未命中时，FA4 转换 CuTe tensor，组装 compile args，调用 `cute.compile`，并缓存结果。

```python
# 来源：flash_attn/cute/interface.py L767-L1017
if compile_key not in _flash_attn_fwd.compile_cache:
    q_tensor, k_tensor, v_tensor, o_tensor = [
        to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
    ]
    compile_args = [
        fa_fwd,
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        page_table_tensor,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
    ]
    _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
        *compile_args, options="--enable-tvm-ffi"
    )
```

这不是把复杂性消掉，而是改变复杂性的所在位置：从构建时静态实例，移动到运行时 JIT/cache 管理。

## 复盘

FA4 的增量可以压成一句话：把 attention kernel 的硬件/shape/feature 组合对象化，并用 CuTeDSL 在运行时按需编译缓存。收益是适配新架构更灵活；风险是首次编译延迟、cache key 稳定性和 unsupported feature 边界需要被生产系统显式管理。
