---
title: "sgl-kernel · 源码走读"
type: walkthrough
framework: sglang
topic: "sgl-kernel"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-10
---
# sgl-kernel · 源码走读

> 读法：本篇关注 `sgl-kernel/python/sgl_kernel` 的 Python 加载层与 wrapper 层。真正的 CUDA/ROCm kernel 在扩展库中注册；这里的源码重点是动态库加载、副作用注册、参数约束、buffer 归一化与 `torch.ops.sgl_kernel.*` 调用边界。

---

## 长文读法

这篇按“Python 包如何把扩展 op 安全暴露给 SRT”读：`__init__.py` import 时加载架构匹配的动态库并预加载 CUDA runtime，wrapper 文件只做参数约束、buffer 归一化和 `torch.ops.sgl_kernel.*` 转发，最后再给可导出的函数批量套 debug wrapper。真正的 CUDA/ROCm 实现不在这些 Python 文件里。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立 sgl-kernel 边界 | 1 | Python 入口首先是动态库注册器，函数 re-export 依赖 import 副作用完成 |
| 排查 op 找不到或加载错架构 | 1.1 到 1.3 | `_load_architecture_specific_ops` 按硬件变体加载 `.so`，CUDA runtime 预加载解决动态链接问题 |
| 理解 attention wrapper | 2 | wrapper 校验 DeepSeek MLA / partial attention state 的形状和 workspace，再转到 PyTorch dispatcher |
| 理解 MoE 路由 wrapper | 3 | token-expert 对齐、top-k softmax / sigmoid 都是预分配输出张量后交给扩展 op 写入 |
| 排查 GEMM 或 KV 迁移 | 4 | GEMM wrapper 基本是轻量转发，KV I/O wrapper 明确 src/dst index 和每层迁移边界 |
| 理解投机采样与 top-k 后处理 | 5 | 这些接口大多以可变输出张量为契约，调用方要提前准备 `predict`、`accept_index`、概率或 score buffer |
| 看 ROCm / debug 特殊路径 | 6 | ROCm allreduce 只在 HIP 构建下暴露；debug wrapper 是导出层的统一包裹，不改变 kernel 语义 |

读的时候不要在本篇寻找 kernel 算法细节。本篇的价值是确认 SRT 调用扩展 op 前，Python 层负责了哪些输入契约，以及哪些错误会留到扩展库或 kernel launch 时才暴露。

## 1. 包初始化与动态库加载

### 1.1 `__init__.py`：import 时加载 architecture-specific ops

来源：sgl-kernel/python/sgl_kernel/__init__.py L18-L23

**问题与约束：** `sgl_kernel` 被上层 SRT import 时，Python 符号和 C++/CUDA 扩展 op 都必须已经注册；同时 CUDA runtime 的动态链接问题需要尽早处理。

**设计选择：** 在非 Apple Silicon 路径下，包初始化阶段先调用 `_load_architecture_specific_ops()`，再在 `torch.version.cuda` 存在时预加载 CUDA runtime。

**读法：** `common_ops` 变量本身不是业务对象，关键是加载扩展模块的副作用：扩展模块 import 后会把 `torch.ops.sgl_kernel.*` 注册进 PyTorch dispatcher。后续 Python wrapper 才能直接转调这些 op。

**源码锚点：**

```python
    # Initialize the ops library based on current GPU
    common_ops = _load_architecture_specific_ops()

    # Preload the CUDA library to avoid the issue of libcudart.so.12 not found
    if torch.version.cuda is not None:
        _preload_cuda_library()
```

**代码逻辑：** 初始化先选择并执行架构相关扩展库加载；如果当前 PyTorch 是 CUDA 构建，再调用 runtime 预加载。后续的 `from sgl_kernel.* import ...` 都建立在 op 已注册的前提上。

**为什么这样写：** wrapper 层不应在每次函数调用时检查扩展库是否存在，否则热路径会被 import 状态污染。把加载集中到包初始化，可以让函数调用退化成轻量的 `torch.ops` 转发。

**不变量与失败模式：** 扩展库必须在 wrapper 被调用前成功注册；CUDA runtime 预加载只应在 CUDA 构建下执行；Apple Silicon 走 Metal 分支，不暴露 CUDA 符号。若 `_load_architecture_specific_ops` 失败，后续 wrapper 会在 `torch.ops.sgl_kernel` 处整体不可用。

**要点：** `sgl-kernel` 的 Python 入口首先是动态库注册器，其次才是函数 re-export 容器。

### 1.2 `_load_architecture_specific_ops`：从架构目录精确加载 `.so`

来源：sgl-kernel/python/sgl_kernel/load_utils.py L84-L101

**问题与约束：** 同一个 wheel 可能携带多个 GPU 架构变体，加载时需要优先选择当前设备匹配的 compiled extension；直接按普通 import 名称查找容易拿到错误文件或 stub。

**设计选择：** 当发现匹配文件时，用 `importlib.util.spec_from_file_location("common_ops", ops_path)` 从具体路径构造 module spec，再 `exec_module` 执行该 `.so`。

**读法：** 这里刻意绕过普通 import 搜索路径，用已筛选出的 `.so` 路径加载模块。这样可以保证当前进程注册的是对应 GPU 架构的 op 实现。

**源码锚点：**

```python
    if matching_files:
        ops_path = Path(matching_files[0])  # Use the first prioritized file
        logger.debug(f"[sgl_kernel] Found architecture-specific library: {ops_path}")
        try:
            # Load the module from specific path using importlib
            spec = importlib.util.spec_from_file_location("common_ops", str(ops_path))
            if spec is None:
                raise ImportError(f"Could not create module spec for {ops_path}")

            common_ops = importlib.util.module_from_spec(spec)
            if spec.loader is None:
                raise ImportError(f"Module spec has no loader for {ops_path}")

            logger.debug(f"[sgl_kernel] Loading module from {ops_path}...")
            spec.loader.exec_module(common_ops)
            logger.debug(f"[sgl_kernel] ✓ Successfully loaded {variant_name}")
            logger.debug(f"[sgl_kernel] ✓ Module file: {common_ops.__file__}")
            return common_ops
```

**代码逻辑：** 函数取优先级最高的匹配文件，构建 module spec，检查 spec 与 loader 是否存在，创建 module 对象并执行 loader。成功后记录变体名与实际文件路径，并返回 `common_ops` module。

**为什么这样写：** GPU kernel 对 SM 架构敏感，错误变体可能能 import 但运行失败或性能不对。按路径加载使“选择哪个 `.so`”成为显式决策，也让日志能定位实际加载文件。

**不变量与失败模式：** `matching_files[0]` 必须是经过优先级排序的 compiled extension；spec 和 loader 不能为空；`exec_module` 的副作用必须完成 op 注册。若路径选择错误，Python wrapper 仍能存在，但实际 kernel 可能在 launch 时崩溃。

**要点：** 这一段是 sgl-kernel 的架构选择点：不是按 Python 包名加载，而是按硬件匹配后的文件路径加载。

### 1.3 `_preload_cuda_library`：提前把 CUDA runtime 放进全局符号表

来源：sgl-kernel/python/sgl_kernel/load_utils.py L234-L242

**问题与约束：** 扩展库动态链接可能依赖 `libcudart.so.12` 或 `libcudart.so.13`；在某些部署环境中，Python 能找到 wheel，却不能在 `.so` 加载时解析 CUDA runtime。

**设计选择：** 遍历候选目录与 CUDA runtime 版本，找到存在的 `libcudart.so.*` 后，用 `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` 预加载。

**读法：** `RTLD_GLOBAL` 让后续扩展库解析动态符号时能看到 CUDA runtime。函数找到第一个可用候选后立即返回，避免重复加载多个 runtime。

**源码锚点：**

```python
    for base in candidate_dirs:
        for lib_version in lib_versions:
            candidate = base / f"libcudart.so.{lib_version}"
            if candidate.exists():
                try:
                    cuda_runtime_lib = candidate.resolve()
                    ctypes.CDLL(str(cuda_runtime_lib), mode=ctypes.RTLD_GLOBAL)
                    logger.debug(f"Preloaded CUDA runtime under {cuda_runtime_lib}")
                    return
```

**代码逻辑：** 双层循环生成候选路径；路径存在时 resolve 成绝对路径，调用 `ctypes.CDLL` 以全局模式载入，记录日志后结束。

**为什么这样写：** 运行环境的动态链接器搜索路径不总是和 PyTorch wheel 的 CUDA runtime 位置一致。显式预加载可以把“找不到 libcudart”的错误提前并局部化到 import 阶段。

**不变量与失败模式：** 候选版本应匹配当前 PyTorch CUDA major；只在 CUDA 构建下调用；加载失败应继续尝试其他候选。若不使用 `RTLD_GLOBAL`，后续 `.so` 仍可能解析不到 runtime 符号。

**要点：** 这是部署层防护：不改变任何 kernel 逻辑，只降低 import 后首次调用扩展库时的动态链接风险。

---

## 2. Attention wrapper

### 2.1 `merge_state_v2`：合并 partial attention state

来源：sgl-kernel/python/sgl_kernel/attention.py L6-L26

**问题与约束：** Split-KV 或 cascade attention 会产生多段 partial output 与 log-sum-exp state，需要数值稳定地合并；wrapper 还要支持调用方传入输出 buffer，减少临时分配。

**设计选择：** 先把 `s_a/s_b` 转成 float32；若未传 `v_merged/s_merged`，就创建对应输出 tensor；最后调用 `torch.ops.sgl_kernel.merge_state_v2.default` 原地写输出。

**读法：** 合并 state 的数值敏感部分是 LSE，float32 能降低半精度溢出/下溢风险。输出 buffer 可选则让上层在高频调用中复用内存。

**源码锚点：**

```python
def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.

    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged
```

**代码逻辑：** 函数接收两段 value/state，规范 state dtype，准备输出 tensor，调用注册 op 写入结果，并返回两个输出对象。

**为什么这样写：** Python wrapper 不重复实现 merge 数学，只负责 dtype 与 buffer 管理。这样既保留 kernel 性能，又让调用者可以控制输出内存生命周期。

**不变量与失败模式：** `v_a/v_b` 与 `s_a/s_b` 的 shape 必须匹配 kernel 预期；传入输出 buffer 时 shape/dtype 必须可写；FP8 与非 CUDA fallback 在源码待办注释中仍是风险点。若 state 不转 float32，长序列 LSE 合并更容易数值不稳。

**要点：** 这是典型的 sgl-kernel wrapper：Python 层处理少量 ABI 约束，计算交给 dispatcher 后面的扩展 op。

### 2.2 `cutlass_mla_decode`：校验 DeepSeek MLA paged decode 形状

来源：sgl-kernel/python/sgl_kernel/attention.py L29-L55

**问题与约束：** MLA decode 的 query 被拆成 latent 与 rope 两部分，KV cache 又把 latent KV 与 rope K 合并存储；CUTLASS kernel 对维度常量有固定假设。

**设计选择：** wrapper 在调用 kernel 前断言 `q_nope/q_pe/kv_c_and_k_pe_cache` 都是 3D，校验 batch/head 一致，并把 MLA 维度固定为 `D_latent=512`、`D_rope=64`。

**读法：** 这里先在 Python 层把模型 ABI 固定住：`q_nope` 对 latent 维，`q_pe` 对 rope 维，cache 最后一维必须等于二者之和。错误 shape 尽早在 assert 处暴露。

**源码锚点：**

```python
def cutlass_mla_decode(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int = 1,  # Set to 1 to avoid cuda_graph issue by default.
) -> torch.Tensor:
    assert q_nope.ndim == 3, f"q_nope must be a 3D tensor, but got {q_nope.ndim}"
    assert q_pe.ndim == 3, f"q_pe must be a 3D tensor, but got {q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be a 3D tensor, but got {kv_c_and_k_pe_cache.ndim}"

    B_q, H, D_q_nope = q_nope.shape
    B_q_2, H_2, D_q_pe = q_pe.shape
    assert (B_q == B_q_2) and (H == H_2)

    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    D_latent = 512
    D_rope = 64
    assert D_q_nope == D_latent
    assert D_q_pe == D_rope
    assert D_ckv == D_latent + D_rope
```

**代码逻辑：** 函数签名收齐 query、cache、sequence length、page table、workspace 与 scale；进入后先断言 rank，再解出 batch/head/维度，最后校验 DeepSeek MLA 的固定 latent/rope/cache 维度。

**为什么这样写：** CUTLASS kernel 对 tile 与布局假设严格，shape 错误如果进入 kernel 会变成难定位的非法访问或错误输出。Python assert 是更便宜、更清晰的 ABI 闸门。

**不变量与失败模式：** `q_nope` 与 `q_pe` 的 batch/head 必须一致；cache 最后一维必须是 576；默认 `num_kv_splits=1` 是为了避开 CUDA Graph 相关问题。若模型维度变体不同，需要对应更新 wrapper 与 kernel。

**要点：** `cutlass_mla_decode` 的前半段就是模型布局契约，先把错形状拦在 Python 层。

---

## 3. MoE wrapper

### 3.1 `moe_align_block_size`：把 token-expert 分配对齐到 block

来源：sgl-kernel/python/sgl_kernel/moe.py L6-L25

**问题与约束：** MoE grouped GEMM 需要按 expert 分组并按 block size 对齐的 token 布局；这些输出 buffer 通常由上层预分配，kernel 应原地填充以减少额外分配。

**设计选择：** wrapper 接收 `topk_ids`、expert 数、block size 与多个输出 buffer，直接调用 `torch.ops.sgl_kernel.moe_align_block_size.default`，不返回新对象。

**读法：** 这个函数是 MoE routing 到 grouped GEMM 之间的布局整理步骤。Python 层只传递 buffer，排序、padding 和计数由扩展 op 完成。

**源码锚点：**

```python
def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids=False,
):
    torch.ops.sgl_kernel.moe_align_block_size.default(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        pad_sorted_token_ids,
    )
```

**代码逻辑：** 函数把输入 expert id、输出 token/expert buffer、padding 后 token 计数和 cumsum buffer 原样传给 op。输出通过 mutable tensor 写回。

**为什么这样写：** MoE 对齐是大批量张量重排，Python 侧循环会成为瓶颈。保持 in-place wrapper 可以让调用方复用缓存 buffer，并把工作集中到 GPU kernel。

**不变量与失败模式：** 输出 buffer 的 shape 与 dtype 必须符合 kernel 约定；`block_size` 必须与后续 grouped GEMM tile 对齐；`pad_sorted_token_ids` 会改变填充策略。若 `num_tokens_post_pad` 没有正确写回，后续 GEMM 会读错有效 token 数。

**要点：** 这段的设计信号是“无返回值”：调用方关心的是被填好的布局 buffer。

### 3.2 `topk_sigmoid`：sigmoid gating 的 top-k 路由

来源：sgl-kernel/python/sgl_kernel/moe.py L57-L85

**问题与约束：** DeepSeek 风格 MoE routing 可能使用 sigmoid gating、top-k expert 选择、renormalize 和 per-expert correction bias；输出需要写入预分配的 weights/ids buffer。

**设计选择：** wrapper 用 docstring 明确输入输出形状与 `correction_bias` dtype 约束，并把参数直接转给 `torch.ops.sgl_kernel.topk_sigmoid.default`。

**读法：** `topk_sigmoid` 把 gating logits 到 top-k expert 的转换放到扩展 op 中完成。Python 层不返回 tensor，而是要求调用者提供 `topk_weights` 与 `topk_ids` 输出位置。

**源码锚点：**

```python
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    Compute top-k sigmoid for MoE routing.

    Args:
        topk_weights: Output tensor for top-k weights [num_tokens, topk]
        topk_ids: Output tensor for top-k expert indices [num_tokens, topk]
        gating_output: Gating logits [num_tokens, num_experts]
        renormalize: Whether to renormalize the top-k weights
        correction_bias: Per-expert bias correction [num_experts], must be float32 if provided
    """
    torch.ops.sgl_kernel.topk_sigmoid.default(
        topk_weights,
        topk_ids,
        gating_output,
        renormalize,
        correction_bias,
    )
```

**代码逻辑：** 函数接收输出 weights/ids、输入 gating logits 和两个可选策略参数；扩展 op 根据 logits 写出 top-k 权重与 expert id。

**为什么这样写：** routing 是 MoE forward 的高频路径，sigmoid、bias correction、top-k 和归一化如果拆成多个 PyTorch op，会增加内存读写和 launch 次数。fused op 保持输出 buffer 语义也便于后续对齐步骤复用。

**不变量与失败模式：** `topk_weights/topk_ids` 的第二维必须等于 top-k；`correction_bias` 若存在必须覆盖所有 expert 且为 float32；`gating_output` shape 是 `[num_tokens, num_experts]`。若输出 buffer 形状不匹配，错误可能在 kernel 内表现为越界写。

**要点：** `topk_sigmoid` 是 routing 侧的 fused wrapper，和 `moe_align_block_size` 组成 MoE 进入 grouped GEMM 前的两步。

---

## 4. GEMM 与 KV I/O wrapper

### 4.1 `gemm.py`：INT8/FP8 scaled matmul 的轻量转发

来源：sgl-kernel/python/sgl_kernel/gemm.py L14-L35

**问题与约束：** 量化推理需要 INT8/FP8 矩阵乘，并携带 activation/weight scale、输出 dtype 和可选 bias；Python 层不应重写矩阵乘逻辑。

**设计选择：** `int8_scaled_mm`、`fp8_blockwise_scaled_mm`、`fp8_scaled_mm` 都作为薄 wrapper，把输入矩阵、scale、dtype 和 bias 原样传给对应 `torch.ops.sgl_kernel` op。

**读法：** 这组函数把量化 GEMM 的 Python API 固定下来：调用方按同一模式提供矩阵与 scale，具体 kernel 实现由注册的扩展 op 决定。

**源码锚点：**

```python
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype):
    return torch.ops.sgl_kernel.fp8_blockwise_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.fp8_scaled_mm.default(
```

**代码逻辑：** 代码段展示了 INT8 scaled matmul 的 op 调用，以及 FP8 blockwise/per-tensor 两个 wrapper 的函数边界和参数转发。

**为什么这样写：** 量化层需要一个稳定 Python 入口，但性能和设备支持在 C++/CUDA 层变化更快。薄 wrapper 可以让上层量化配置选择函数名，而不绑定具体实现细节。

**不变量与失败模式：** scale tensor 的粒度必须与所选 wrapper 匹配；`out_dtype` 必须是 kernel 支持的输出类型；bias 只在对应 op 支持时传入。若把 blockwise scale 传给 per-tensor wrapper，会得到错误缩放结果。

**要点：** `gemm.py` 的可读重点是 API 形状，而不是 matmul 算法；算法在注册 op 后面。

### 4.2 `transfer_kv_per_layer`：单层 K/V 按 index 迁移

来源：sgl-kernel/python/sgl_kernel/kvcacheio.py L14-L35

**问题与约束：** PD disaggregation 或 KV cache 迁移需要按 index 把单层 K/V 从源 buffer 拷贝到目标 buffer；不同平台的 warp 配置不同，还需要限制 block quota。

**设计选择：** wrapper 接收 source/destination K/V、源/目标 indices、item size、block quota 和平台相关 `num_warps_per_block`，直接调用 `transfer_kv_per_layer` op。

**读法：** 这个 API 把“按 token/page index 迁移 K/V”封装成单层操作。Python 层决定并发配置与平台默认值，实际拷贝由 kernel 完成。

**源码锚点：**

```python
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 16 if _is_hip else 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer.default(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        block_quota,
        num_warps_per_block,
    )
```

**代码逻辑：** 函数签名暴露 K/V 源目标 tensor 与 index 映射，默认 block quota 为 2；ROCm 默认 warp 数为 16，否则为 32。调用扩展 op 完成拷贝，无返回值。

**为什么这样写：** KV cache 迁移是带映射的内存搬运，Python 侧逐项 copy 代价太高。把 warp 数默认值放在 wrapper，可让同一 API 适配 CUDA 与 HIP。

**不变量与失败模式：** `src_indices/dst_indices` 长度要与迁移元素数一致；`item_size` 必须等于单个 K/V item 的拷贝跨度；HIP/CUDA 的 warp 默认值不能混用。若 index 映射错位，会把 KV 写到错误 cache slot。

**要点：** KV I/O wrapper 的价值在于把平台参数和 index 映射一并交给 fused copy kernel。

---

## 5. 投机解码与采样后处理

### 5.1 `tree_speculative_sampling_target_only`：target-only 验证的可变输出

来源：sgl-kernel/python/sgl_kernel/speculative.py L4-L24

**问题与约束：** 树形投机解码需要根据 draft tree、target/draft 概率和随机数接受或拒绝 token，并把预测结果、接受位置和接受数量写回调用方提供的 buffer。

**设计选择：** wrapper 把 `predicts`、`accept_index`、`accept_token_num` 明确标为 mutable，并把 tree 结构、随机数、概率与阈值参数传给 `tree_speculative_sampling_target_only` op。

**读法：** 这个函数的返回值是 `None`，因为结果通过多个 mutable tensor 输出。它适合在 speculative worker 中作为一次 target 验证步骤。

**源码锚点：**

```python
def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
```

**代码逻辑：** 函数签名列出三类输入：可变输出 buffer、draft tree 遍历结构、概率/随机数/阈值参数。函数体开始调用扩展 op，并把可变输出作为前几个参数传入。

**为什么这样写：** 投机验证要更新多个结果数组，用多个返回 tensor 会增加分配和同步。mutable buffer 让调用方控制内存，并与后续 accept/reject 流程共享结果。

**不变量与失败模式：** mutable buffer 必须提前按 batch/tree 大小分配；tree 结构数组之间的索引语义必须一致；`deterministic` 与随机样本参数不能冲突。若 buffer shape 不足，kernel 可能越界写；若 tree index 不一致，接受路径会错乱。

**要点：** 这里的 wrapper 明确展示了 speculative sampling 的 ABI：不是返回一个对象，而是原地更新多组状态数组。

### 5.2 `_top_k_renorm_probs_internal` 与 `top_k_renorm_probs`：概率先升精度再截断归一

来源：sgl-kernel/python/sgl_kernel/sampling.py L18-L28

**问题与约束：** 采样前的概率分布需要 top-k 截断并重新归一化；半精度概率在截断与求和时容易出现精度问题，top-k 还可能是 per-row tensor。

**设计选择：** internal helper 先把 `probs` 转成 float32，把可选 `maybe_top_k_arr` 转成 int，再创建 `renorm_probs` 输出并调用 `top_k_renorm_probs` op；公开函数从下一行开始定义。

**读法：** wrapper 先把 dtype 归一化，再把截断归一交给 GPU op。这样调用者可以传半精度概率，但 kernel 看到的是更稳定的 float32 概率。

**源码锚点：**

```python
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_k_renorm_probs.default(
        probs, renorm_probs, maybe_top_k_arr, top_k_val
    )
    return renorm_probs


def top_k_renorm_probs(
```

**代码逻辑：** 代码段位于 internal helper 末尾：输入概率转 float，top-k 数组转 int，分配同 shape 输出，调用扩展 op 写结果，然后返回 renormalized 概率。

**为什么这样写：** top-k 后需要重新除以保留概率的总和；用 float32 可以降低 underflow 和归一化误差。`maybe_top_k_arr` 支持每行不同 top-k，统一转 int 可简化 kernel ABI。

**不变量与失败模式：** `probs` 应是二维概率分布；`maybe_top_k_arr` 若存在应能广播或按 batch 对齐；输出 shape 与输入一致。若不转 float，低概率 token 在截断/归一中更容易出现数值不稳定。

**要点：** 采样 wrapper 的核心是 dtype 与输出 buffer 管理；真正的 top-k 截断由 `sgl_kernel` op 完成。

### 5.3 `fast_topk_v2`：ragged/paged score 的 top-k 接口契约

来源：sgl-kernel/python/sgl_kernel/top_k.py L18-L35

**问题与约束：** Attention score 可能是 ragged 或 paged layout，每行有效区间长度不同；top-k 只能作用在每行 `[row_starts[i], row_starts[i] + lengths[i])` 的有效范围内。

**设计选择：** 函数签名显式接收 `score`、`lengths`、`topk` 与可选 `row_starts`，docstring 把 ragged 场景下的有效区间规则写成接口契约。

**读法：** `fast_topk_v2` 的关键不是普通 dense topk，而是按行长度和可选起点限定搜索范围。这样同一个 score tensor 可以承载 ragged/paged 两种布局。

**源码锚点：**

```python
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
```

**代码逻辑：** 代码段展示函数参数与返回约定：`score` 是 `[B, L]`，`lengths` 是每行有效长度，`row_starts` 只在 ragged key 中需要，返回 `[B, topk]` 的 indices。

**为什么这样写：** 稀疏或分页布局中，无效位置不能参与 top-k；把有效区间写进接口，比要求调用者提前 materialize dense mask 更省内存，也更接近 kernel 的实际访问模式。

**不变量与失败模式：** `lengths` 长度必须等于 batch size；ragged layout 必须提供正确 `row_starts`；返回的是 indices 而不是 values。若有效区间边界错，top-k 会把 padding 或其他请求的 score 当成候选。

**要点：** 这段 docstring 是接口级源码证据：`fast_topk_v2` 服务的是 ragged/paged score，不是普通 vocab top-k。

---

## 6. ROCm AllReduce 与调试包装

### 6.1 `init_custom_ar`：ROCm 自定义 allreduce 初始化

来源：sgl-kernel/python/sgl_kernel/allreduce.py L7-L17

**问题与约束：** ROCm 路径下，小 tensor TP allreduce 可能需要绕过通用通信库的固定开销；自定义 AR 需要用 IPC handles、offsets、rank 等信息初始化一个后续调用可复用的句柄。

**设计选择：** 在 `torch.version.hip is not None` 分支内定义 `init_custom_ar`，把 meta、rank data、handles、offsets、rank 与 full_nvlink 标志传给 `torch.ops.sgl_kernel.init_custom_ar.default`，并返回 int handle。

**读法：** 这个函数只在 HIP 平台暴露。返回的整数句柄是后续 registered/unregistered allreduce wrapper 的上下文入口。

**源码锚点：**

```python
    def init_custom_ar(
        meta: torch.Tensor,
        rank_data: torch.Tensor,
        handles: List[str],
        offsets: List[int],
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return torch.ops.sgl_kernel.init_custom_ar.default(
            meta, rank_data, handles, offsets, rank, full_nvlink
        )
```

**代码逻辑：** 函数接收通信元数据和 IPC 信息，转调扩展 op，并把 op 返回的 handle 交给调用方保存。

**为什么这样写：** 自定义 allreduce 的资源准备和后续执行是分离的。初始化时集中建立共享内存/IPC 上下文，后续 allreduce 调用就能只传 handle 与 tensor。

**不变量与失败模式：** 该 wrapper 只在 HIP 分支定义；handles 与 offsets 必须覆盖参与 rank；返回 handle 必须在后续调用中保持有效。若 CUDA 环境误用该符号，可能根本没有对应定义或 op。

**要点：** AllReduce wrapper 体现了平台分叉：ROCm 下暴露自定义通信路径，CUDA 路径不一定有同名 Python API。

### 6.2 `_DEBUG_EXPORT_NAMES`：批量给导出函数套 debug wrapper

来源：sgl-kernel/python/sgl_kernel/__init__.py L216-L220

**问题与约束：** sgl-kernel re-export 了大量函数，如果逐个手写 debug wrapper，容易遗漏；但只有实际存在于 `globals()` 的符号才能被包装，平台差异符号不能强行引用。

**设计选择：** import 末尾遍历 `_DEBUG_EXPORT_NAMES`，如果名字在当前 globals 中，就用 `maybe_wrap_debug_kernel` 替换为带日志的 wrapper。

**读法：** 这是统一 debug instrumentation 的最后一步。它不改变函数名，也不改变导出表，只在需要时把可调用对象包一层。

**源码锚点：**

```python
    for _name in _DEBUG_EXPORT_NAMES:
        if _name in globals():
            globals()[_name] = maybe_wrap_debug_kernel(
                globals()[_name], f"sgl_kernel.{_name}"
            )
```

**代码逻辑：** 循环从 debug 名单取符号名，检查当前平台/导入路径是否真的导出了该符号；存在则用 wrapper 替换 globals 中的函数对象，日志名带 `sgl_kernel.` 前缀。

**为什么这样写：** 平台和可选依赖会影响实际导出符号。先检查 `globals()` 可以避免 debug 包装阶段因为缺失符号失败，同时让新增导出只需加入名单即可获得统一日志。

**不变量与失败模式：** 被包装对象必须是可调用 kernel wrapper；不存在的名字必须跳过；包装后函数签名与返回语义应保持透明。若直接索引 globals，平台特有符号缺失会导致 import 阶段崩溃。

**要点：** debug 包装放在 import 收尾阶段，保证所有 re-export 已完成，再按当前平台实际符号集合批量增强。

---

## 运行验证

维护本文时，先用下面的命令确认 Python wrapper 与 `torch.ops.sgl_kernel` 边界还在原位：

```powershell
rg -n "torch.ops.sgl_kernel|maybe_wrap_debug_kernel|merge_state_v2|grouped_gemm|fast_topk_v2|init_custom_ar" sglang/sgl-kernel/python/sgl_kernel
```

预期信号：

- `attention.py`、`moe.py`、`gemm.py`、`kvcacheio.py` 等 wrapper 仍直接转调 `torch.ops.sgl_kernel`。
- `__init__.py` 仍集中 re-export 并在末尾做 debug wrapper 包装。
- `top_k.py` 仍能看到 ragged/paged top-k 的专用接口。
- `allreduce.py` 仍体现 CUDA / ROCm 平台分叉。

如果 wrapper 不再直接转调 `torch.ops.sgl_kernel`，本文应从“薄 Python 包装层”改写为新的调度/注册模型。
