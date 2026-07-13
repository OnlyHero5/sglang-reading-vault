---
title: "sgl-kernel · 排障指南"
type: troubleshooting
framework: sglang
topic: "sgl-kernel"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# sgl-kernel · 排障指南

---

## 你为什么要读

`sgl-kernel` 的失败从 import error 到 illegal memory access 跨了好几层。本文先检查 wheel、动态库与 GPU 架构，再确认 op 注册、参数校验和具体 kernel launch；顺序反了，往往会在 CUDA 现场追一个其实由安装造成的问题。

## 1. 为什么 sgl-kernel 独立于 srt 发布？

**读法：** CUDA 扩展编译慢、wheel 体积大（SM90/SM100 各一套 `.so`），与 Python 逻辑解耦后 srt 可频繁发版而 kernel 按需升级。用户可通过 `pip install sglang-kernel --index-url ...` 单独更新算子。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/load_utils.py L170-L175
    if cuda_version and cuda_version.startswith("12"):
        install_hint = (
            "pip install sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/"
        )
    else:
        install_hint = "pip install --upgrade sglang-kernel"
```

**要点：**

- 加载失败时错误信息直接给出 install hint。
- 当前代码只对 `torch.version.cuda` 以 `12` 开头时给出 `cu129` index；其他情况统一给普通 upgrade 命令，不能据此推断 CUDA 13 有另一条专用 index。

---

## 2. SM90 与 SM100 选错会怎样？

**读法：** 加载逻辑确实让非 90 设备统一选择 `sm100/` 目录，但目录名只是 precise-math 变体标签。真正能否执行取决于 wheel 构建时是否包含当前设备的 gencode；不匹配既可能在扩展加载时暴露，也可能到 launch 才出现 no-kernel-image、invalid-device-function 等错误。

**源码锚点：**

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

**要点：**

- H100 用户应看到 `sm90/common_ops.*` 被加载（debug log）。
- B200 等非 CC90 设备会选择 precise-math 目录；“选择了目录”不等于已经证明该 wheel 覆盖当前卡，更不能脱离固定输入与 tolerance 宣称数值保证。

---

## 3. Python 封装 vs 直接 torch.ops 的区别

**读法：** 部分 Python wrapper 提供 dtype/shape assert、padding、dtype 转换、输出或 workspace 分配；另一些只是转发。直接调 `torch.ops` 会绕过对应 wrapper 的全部前后处理，只有调用方已经逐项满足底层 schema 与 kernel ABI 时才有相同语义。

**源码锚点：**

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

**要点：**

- padding 逻辑仅在 Python 层，csrc 假设固定 MAX_HEADS。
- 绕过 Python 封装需自行保证 head pad 与 dtype。

---

## 4. 何时 fallback 到 Triton / PyTorch？

**读法：** fallback 没有统一发生在 sgl-kernel 内。以 attention state merge 为例，SRT 的 `merge_state()` 显式检查 CUDA、dtype 与 head dimension，满足才调用 `merge_state_v2`，否则走 Triton；而 `load_utils.py` 所谓 fallback 只是换位置继续寻找 `common_ops`。排障时必须找到具体调用方的分支，不能假设所有 op 都捕获 `ImportError/NotImplementedError`。

**源码锚点：**

```python
# 来源：python/sglang/srt/layers/attention/merge_state.py L34-L45
    if (
        _is_cuda
        and _supported_dtypes(prefix_output)
        and _supported_headdim(prefix_output)
    ):
        return merge_state_v2(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
        )
    else:
        # Fallback to Triton kernel
        return merge_state_triton(
            prefix_output, prefix_lse, suffix_output, suffix_lse, output, output_lse
```

**要点：**

- Metal 分支完全独立，无 `torch.ops.sgl_kernel`。
- ROCm 有独立 `common_extension_rocm.cc` 注册集合，不能说“其余与 CUDA 共用 sm100 build”；同名 Python wrapper 是否存在，也不代表 ROCm schema 一定与 CUDA 完全一致。

---

## 5. 如何调试 kernel 调用？

**读法：** 设置非零 `SGLANG_KERNEL_API_LOGLEVEL` 后，`debug_utils` 会尝试从 SRT 导入 `debug_kernel_api` 并包装当前平台已导出的白名单函数。这个文件只能证明“尝试包装”；具体记录字段、输出位置和开销要继续检查 SRT logger，不能由 wrapper 名推断。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/debug_utils.py L7-L11
def _wrap_debug_kernel(func: F, op_name: str | None = None) -> F:
    try:
        if int(os.environ.get("SGLANG_KERNEL_API_LOGLEVEL", "0")) == 0:
            return func
    except Exception:
```

**要点：**

- 生产环境保持默认 0，避免日志洪水。
- 配合 Nsight / PyTorch profiler 定位 slow kernel。

---

## 6. sgl-kernel vs FlashInfer 边界

**读法：** `sampling.py` 的优先级与旧文相反：只要 FlashInfer 可导入且设备不是 MUSA，top-k/top-p renorm 就直接调用 FlashInfer；MUSA 或缺少 FlashInfer 时才进入 `_top_*_renorm_probs_internal`，再调用 `torch.ops.sgl_kernel`。因此 kernel API debug 日志里没有 renorm 记录，可能只是走了 FlashInfer，并非采样没执行。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/sampling.py L56-L59
    if probs.device.type == "musa" or not _has_flashinfer:
        return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))
    else:
        return _flashinfer_sampling.top_k_renorm_probs(probs, top_k)
```

**要点：**

- sgl-kernel 既包含 SRT 特有/融合算子，也直接编入或包装 FlashInfer renorm、稀疏 attention 等第三方来源代码，边界不是“通用归 FlashInfer、特有归 sgl-kernel”的简单二分。
- 通用 flash attention 多在 FlashInfer / cuDNN 侧，不在本包。

## 7. 当前基线的两个已知 Python 层风险

### `common_ops` 自身报 `libcudart` 缺失

症状：`import sgl_kernel` 在 `_load_architecture_specific_ops()` 执行扩展模块时已经因 `libcudart.so.*` 解析失败。

源码入口：

```python
# 来源：sgl-kernel/python/sgl_kernel/__init__.py L18-L23
    # Initialize the ops library based on current GPU
    common_ops = _load_architecture_specific_ops()

    # Preload the CUDA library to avoid the issue of libcudart.so.12 not found
    if torch.version.cuda is not None:
        _preload_cuda_library()
```

判断：preload 发生在扩展加载之后。若失败点就是第 19 行的 `exec_module(common_ops)`，第 23 行根本不可达，不能期待这个 helper 自救。

操作：打开 `sgl_kernel.load_utils` debug 日志，保存异常链、`ldd`/loader 解析结果、`torch.version.cuda` 和实际 wheel 文件。预期：若根因是 `common_ops` 首次链接缺 runtime，应从安装、RPATH/loader path 或调用顺序修复，而不是继续追具体 kernel。

### GPTQ shuffle 在 dispatcher 前就报属性错误

症状：调用公开 `gptq_shuffle()` 时出现 `AttributeError`，profiler 里没有对应 kernel。

```python
# 来源：sgl-kernel/python/sgl_kernel/gemm.py L219-L220
def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor, bit: int) -> None:
    torch.torch.ops.sgl_kernel.gptq_shuffle(q_weight, q_perm, bit)
```

判断：这里多写了一层 `torch`，失败发生在 Python 属性解析阶段；`common_extension.cc` 是否正确注册 `gptq_shuffle` 不能改变这条 wrapper 的可达性。

操作：用最小 monkeypatch/AST 检查确认属性链为 `torch.torch.ops`，再对照直接 dispatcher 入口是否存在。预期：静态检查稳定复现错误属性链；真实 CUDA 数值实验要等 wrapper 修正或由上层绕开后进行。

## 运行验证

sgl-kernel 的 FAQ 先验证动态库选择、Python wrapper、debug wrapper 和 FlashInfer fallback 边界。

```powershell
rg -n '_get_compute_capability|_preload_cuda_library|fallback|cutlass_mla_decode|merge_state_v2|maybe_wrap_debug_kernel|flashinfer|top_k_renorm_probs|top_p_renorm_probs|torch\.ops\.sgl_kernel' sglang/sgl-kernel/python/sgl_kernel/load_utils.py sglang/sgl-kernel/python/sgl_kernel/attention.py sglang/sgl-kernel/python/sgl_kernel/debug_utils.py sglang/sgl-kernel/python/sgl_kernel/sampling.py
```

读输出时先区分三种完全不同的 fallback：动态库搜索路径 fallback、SRT 算法 fallback、sampling wrapper 的可选依赖分流。再检查两个当前基线缺陷：`__init__.py` 在加载 `common_ops` 之后才 preload CUDA runtime，无法挽救该扩展自己的首次链接失败；`gemm.py` 的 `gptq_shuffle` 写成 `torch.torch.ops...`，若真实调用会先在 Python 属性访问处失败。
