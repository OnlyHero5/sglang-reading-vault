---
type: batch-doc
module: 26-sgl-kernel
batch: "26"
doc_type: faq
title: "sgl-kernel：关键问题"
tags:
 - sglang/batch/26
 - sglang/module/sgl-kernel
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# sgl-kernel：关键问题

---

## 1. 为什么 sgl-kernel 独立于 srt 发布？

**Explain：** CUDA 扩展编译慢、wheel 体积大（SM90/SM100 各一套 `.so`），与 Python 逻辑解耦后 srt 可频繁发版而 kernel 按需升级。用户可通过 `pip install sglang-kernel --index-url ...` 单独更新算子。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/load_utils.py L170-L175
    if cuda_version and cuda_version.startswith("12"):
        install_hint = (
            "pip install sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/"
        )
    else:
        install_hint = "pip install --upgrade sglang-kernel"
```

**Comment：**

- 加载失败时错误信息直接给出 install hint。
- CUDA 12 与 13 使用不同 wheel index。

---

## 2. SM90 与 SM100 选错会怎样？

**Explain：** 加载逻辑按 `compute_capability` 自动选择，非 90 一律走 `sm100/` precise math。手动替换 `.so` 或在不匹配架构上 force load 可能导致非法指令或数值偏差。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/load_utils.py L59-L65
    # Determine which version to load based on GPU architecture
    if compute_capability == 90:
        ops_subdir = "sm90"
        variant_name = "SM90 (Hopper/H100 with fast math optimization)"
    elif compute_capability is not None:
        ops_subdir = "sm100"
        variant_name = f"SM{compute_capability} (precise math for compatibility)"
```

**Comment：**

- H100 用户应看到 `sm90/common_ops.*` 被加载（debug log）。
- B200 等新卡走 sm100 precise 路径，保证 forward 正确性优先于极致性能。

---

## 3. Python 封装 vs 直接 torch.ops 的区别

**Explain：** Python 层提供 dtype/shape assert、padding、dtype 转换（如 merge_state LSE→fp32），减少 srt 调用方重复防御代码。直接调 `torch.ops` 跳过校验，适合已验证的内部路径。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/attention.py L59-L66
    if H < MAX_HEADS:
        q_nope_padded = q_nope.new_empty((B_q, MAX_HEADS, D_q_nope))
        q_nope_padded[:, :H] = q_nope
        q_nope = q_nope_padded

        q_pe_padded = q_pe.new_empty((B_q, MAX_HEADS, D_q_pe))
        q_pe_padded[:, :H] = q_pe
        q_pe = q_pe_padded
```

**Comment：**

- padding 逻辑仅在 Python 层，csrc 假设固定 MAX_HEADS。
- 绕过 Python 封装需自行保证 head pad 与 dtype。

---

## 4. 何时 fallback 到 Triton / PyTorch？

**Explain：** 源码中多处 TODO 标注 FP8、非 CUDA 设备尚未支持 custom kernel。srt 侧通常 try sgl_kernel except ImportError/NotImplemented → Triton。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/attention.py L16-L18
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.
```

**Comment：**

- Metal 分支完全独立，无 `torch.ops.sgl_kernel`。
- ROCm 部分算子可用（allreduce、gelu_quick），其余与 CUDA 共用 sm100 build。

---

## 5. 如何调试 kernel 调用？

**Explain：** 设置 `SGLANG_KERNEL_API_LOGLEVEL=1` 并安装 `sglang.kernel_api_logging`，所有 `_DEBUG_EXPORT_NAMES` 中的函数会记录入参 tensor 元信息。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/debug_utils.py L7-L11
def _wrap_debug_kernel(func: F, op_name: str | None = None) -> F:
    try:
        if int(os.environ.get("SGLANG_KERNEL_API_LOGLEVEL", "0")) == 0:
            return func
    except Exception:
```

**Comment：**

- 生产环境保持默认 0，避免日志洪水。
- 配合 Nsight / PyTorch profiler 定位 slow kernel。

---

## 6. sgl-kernel vs FlashInfer 边界

**Explain：** `sampling.py` 尝试 import FlashInfer 作为可选加速，但 top_k/top_p renorm 仍以 sgl_kernel op 为主路径。Attention 大块逻辑在 srt 的 `attention/backends` 中选择 FlashInfer vs sgl_kernel vs Triton。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/sampling.py L5-L10

try:
    import flashinfer.sampling as _flashinfer_sampling

    _has_flashinfer = True
except ImportError:
```

**Comment：**

- sgl-kernel 聚焦 srt **特有**或**融合**算子（MoE align、MLA decode、tree speculative）。
- 通用 flash attention 多在 FlashInfer / cuDNN 侧，不在本包。
