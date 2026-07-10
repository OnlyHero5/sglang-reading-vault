---
title: "Hopper与CuTe · 排障指南"
type: troubleshooting
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: debug
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# Hopper与CuTe · 排障指南

本页帮你在 FA2、FA3、FA4 多实现并存时先定边界。读完后应能判断一个问题属于安装/import 路径、GPU arch dispatch、JIT cache、FP8 能力边界，还是具体 kernel 算法本身。

## 排障入口

| 症状 | 优先检查 | 源码入口 | 预期判断 |
|------|----------|----------|----------|
| 不知道该读 FA2、FA3 还是 FA4 | 安装包、import 路径、GPU 架构 | `README.md`、`flash_attn/cute/README.md` | FA2 稳定主包，FA3 Hopper beta，FA4 CuTeDSL |
| FA4 走错 kernel path | `FLASH_ATTENTION_ARCH` 与当前 GPU | `_get_device_arch` | arch override 会直接改变 kernel object 分支 |
| 首次请求很慢 | JIT cache miss | `_flash_attn_fwd.compile_cache` | warmup 和 shape bucketing 是生产必需项 |
| FP8 backward 报错 | FP8 是否 requires grad | `_flash_attn_fwd` validation | FA4 CuTe FP8 backward 尚不支持 |
| SplitKV/paged KV 被拒绝 | 当前 arch 分支是否支持 | SM80/SM90/SM120 assert | unsupported feature 是实现边界，不是算法边界 |

## FA3/FA4 是否替代 FA2？

没有。源码中多条路径并存：FA2 是稳定主路径；FA3 是 Hopper beta；FA4 是 CuTeDSL/JIT 包。是否使用取决于安装包、import 路径、GPU 架构和上层框架适配。

```markdown
<!-- 来源：flash_attn/cute/README.md L1-L14 -->
# FlashAttention-4 (CuTeDSL)

FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

pip install flash-attn-4
```

读者不要把“新路径存在”理解成“旧路径废弃”。AI infra 中算子后端经常长期多版本共存。

## CuTeDSL 是否改变 FlashAttention 算法？

没有改变核心算法。仍然是 tile attention、online softmax 和减少 HBM traffic。改变的是 kernel 描述、dispatch 位置和编译缓存方式。

证据是 FA4 仍要创建按 arch/head/tile/mask/GQA 固定的 forward kernel object，并编译成可执行 kernel。

```python
# 来源：flash_attn/cute/interface.py L823-L961
if arch // 10 == 8:
    fa_fwd = FlashAttentionForwardSm80(
        # 省略张量、stride、shape、dtype 与运行选项参数
    )
elif arch // 10 == 9:
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
    fa_fwd = FlashAttentionForwardSm120(
        # 省略张量、stride、shape、dtype 与运行选项参数
    )
```

如果你已经理解 [[FlashAttention-Attention-IO-核心概念]] 和 [[FlashAttention-Online-Softmax-核心概念]]，FA4 应该被看成同一原理在新工具链上的实现。

## 为什么 FA4 要支持 arch override？

`_get_device_arch` 支持 `FLASH_ATTENTION_ARCH`，用于 CPU-only 编译、cross compile 或调试时显式选择 kernel path。

```python
# 来源：flash_attn/cute/interface.py L77-L92
arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
if arch_override is not None:
    return _parse_arch_str(arch_override)
major, minor = torch.cuda.get_device_capability()
return major * 10 + int(minor)
```

排障抓手：先打印或断点确认 `_get_device_arch()` 返回值，再看 SM80/90/100/120 分支。

## JIT cache 会带来什么生产问题？

cache miss 时需要编译，可能带来首次请求延迟；compile key 维度过多会增加缓存压力；shape 和 feature 组合不稳定会导致频繁编译。

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

生产使用 FA4 时，要设计 warmup、shape bucketing、cache 目录策略和多进程复用，而不只是看单次 kernel benchmark。

## FP8 为什么只支持部分路径？

FP8 同时受硬件、dtype、descale tensors、输出 dtype 和 backward 支持约束。FA3 README 写明 FP8 forward；FA4 进一步限制 FP8 backward 和 SM 架构。

```python
# 来源：flash_attn/cute/interface.py L463-L510
is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
if is_fp8 and requires_grad:
    raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
if is_fp8:
    assert arch // 10 == 10, "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
```

排障抓手：看到 FP8 报错时，不要只看 tensor dtype；还要看 GPU arch、是否需要 backward、descale tensors 是否匹配。

## SplitKV、paged KV 为什么有时被拒绝？

FA4 的 arch 分支会明确断言某些特性不支持。例如 SM80 不支持 paged KV 和 SplitKV，SM90 不支持 SplitKV，SM120 当前也拒绝 block sparsity、paged KV 和 SplitKV。

```python
# 来源：flash_attn/cute/interface.py L823-L961
if arch // 10 == 8:
    assert page_table is None, "paged KV not supported on SM 8.0"
    assert not is_split_kv, "SplitKV not supported on SM 8.0"
elif arch // 10 == 9:
    assert not is_split_kv, "SplitKV not supported on SM 9.0"
elif arch // 10 == 12:
    assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
    assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
    assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
```

这类失败是实现能力边界，不是 FlashAttention 算法边界。

## 复盘迁移

- 路径问题先看 import、package 和 arch。
- 性能问题先看 JIT cache、shape 稳定性和 warmup。
- 功能问题先看每个 arch 分支的 unsupported feature 断言。
