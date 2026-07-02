---
type: batch-doc
module: 26-sgl-kernel
batch: "26"
doc_type: walkthrough
title: "sgl-kernel · 源码走读"
tags:
 - sglang/batch/26
 - sglang/module/sgl-kernel
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# sgl-kernel · 源码走读

> 走读顺序：`__init__.py` 加载链 → `load_utils.py` → `attention.py` → `moe.py` → `gemm.py` → `kvcacheio.py` → `speculative.py` → `sampling.py` → `allreduce.py`

---

## 1. 包初始化与动态库加载

### 1.1 `__init__.py` — 入口分支

**Explain：** import `sgl_kernel` 时同步完成三件事：检测平台、加载 `common_ops`、re-export 全部子模块符号。这是 srt 第一次 `import sgl_kernel` 时的完整副作用链。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/__init__.py L18-L23
    # Initialize the ops library based on current GPU
    common_ops = _load_architecture_specific_ops()

    # Preload the CUDA library to avoid the issue of libcudart.so.12 not found
    if torch.version.cuda is not None:
        _preload_cuda_library()
```

**Comment：**

- `common_ops` 变量本身不被直接使用，但其 `exec_module` 副作用注册了全部 `torch.ops.sgl_kernel.*`。
- CUDA runtime preload 在 import 阶段执行，早于任何 kernel launch。

### 1.2 `_load_architecture_specific_ops` — 三级 fallback

**Explain：** 优先从 `sm90/` 或 `sm100/` 目录 `importlib` 加载 `.so`；失败则尝试包根目录 `common_ops.*`；再失败尝试标准 `import common_ops`。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/load_utils.py L84-L101
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

**Comment：**

- `_filter_compiled_extensions` 保证 `.so` 优先于同名 `.py` stub。
- 全部失败时错误信息包含 CUDA 版本与 `pip install` 提示（cu129 index 等）。

### 1.3 `_preload_cuda_library`

**Explain：** 用 `ctypes.CDLL(..., RTLD_GLOBAL)` 预加载 `libcudart.so.{12|13}`，避免后续 `.so` 动态链接时找不到 runtime。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/load_utils.py L234-L242
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

**Comment：**

- 搜索路径含 `$CUDA_HOME/lib`、`/usr/lib/x86_64-linux-gnu` 等。
- CUDA major 版本优先匹配当前 PyTorch 绑定的 CUDA。

---

## 2. Attention 算子

### 2.1 `merge_state_v2` — partial state 合并

**Explain：** 用于 split-KV 或 cascade attention 场景，将两段 partial output `(v_a, s_a)` 与 `(v_b, s_b)` 数值稳定地合并。LSE（log-sum-exp）在 float32 上计算。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/attention.py L6-L26
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

**Comment：**

- 支持 in-place output buffer 传入，减少 allocation。
- FP8 与非 CUDA 设备尚未支持，需 fallback Triton（源码 TODO 标注）。

### 2.2 `cutlass_mla_decode` — DeepSeek MLA paged decode

**Explain：** 针对 MLA（Multi-head Latent Attention）格式的 paged KV cache decode。query 拆为 `q_nope`（latent 512d）与 `q_pe`（rope 64d），KV cache 合并存储 `kv_c_and_k_pe_cache`。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/attention.py L29-L55
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

**Comment：**

- head 数不足 128 时 pad 到 `MAX_HEADS=128` 以满足 CUTLASS kernel tile 约束。
- `num_kv_splits=1` 默认避免 CUDA Graph capture 问题。
- workspace 大小由 `cutlass_mla_get_workspace_size` 预计算。

---

## 3. MoE 算子

### 3.1 `moe_align_block_size` — token 块对齐

**Explain：** MoE forward 前将 token 按 expert 分组并对齐到 `block_size`，填充 `sorted_token_ids`、`experts_ids` 等 buffer，供后续 grouped GEMM 使用。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/moe.py L6-L25
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

**Comment：**

- 纯 in-place kernel，无返回值。
- `topk_ids` 来自上游 `topk_sigmoid` 或 `topk_softmax`。

### 3.2 `topk_sigmoid` — sigmoid 路由

**Explain：** 对 gating logits 做 sigmoid + top-k 选取 expert，可选 renormalize 与 per-expert bias correction（DeepSeek V3 风格）。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/moe.py L57-L85
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


def moe_sum_reduce(
    input_tensor,
    output_tensor,
```

**Comment：**

- 输出写入预分配 buffer，避免 Python 侧 cat/stack。
- `correction_bias` 必须为 float32（docstring 约束）。

---

## 4. 量化 GEMM

### 4.1 FP8 / INT8 scaled matrix multiply

**Explain：** 量化推理的核心算子族。`fp8_scaled_mm` 做 per-tensor scale 的 FP8 矩阵乘；`int8_scaled_mm` 类似 INT8；`fp8_blockwise_scaled_mm` 支持 block-wise scale（更细粒度）。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/gemm.py L14-L35
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

**Comment：**

- AWQ 路径先 `awq_dequantize` 再 matmul，或直接 W4A8 kernel（`qserve_w4a8_*`）。
- srt 量化层（LoRA）根据 `QuantizationConfig` 选择对应函数。

---

## 5. KV Cache I/O

### 5.1 `transfer_kv_per_layer`

**Explain：** PD disaggregation 场景下，将单层 K/V tensor 从 source buffer 按 index 映射拷贝到 destination。支持 block quota 控制并发 warp 数。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/kvcacheio.py L14-L35
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

**Comment：**

- ROCm 上 warp 数减半（16 vs 32）。
- MLA 格式有 `transfer_kv_per_layer_mla` / `transfer_kv_all_layer_mla` 变体。

---

## 6. 投机解码

### 6.1 `tree_speculative_sampling_target_only`

**Explain：** 树形投机解码的 target model 采样阶段：遍历 draft tree，按 threshold 接受/拒绝 token，更新 `predicts`/`accept_index`/`accept_token_num`。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/speculative.py L4-L24
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

**Comment：**

- 配套 `build_tree_kernel_efficient` 构建 tree mask，`verify_tree_greedy` 做 greedy 验证路径。
- srt speculative worker（相关专题模块）在 accept/reject 循环中调用。

---

## 7. 采样后处理

### 7.1 `top_k_renorm_probs`

**Explain：** 对概率分布做 top-k 截断并重归一化。内部优先走 sgl_kernel CUDA op；FlashInfer 可用时 sampling 模块另有路径。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/sampling.py L18-L28
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

**Comment：**

- 强制 float32 概率，避免 half 精度 underflow。
- 支持 per-row 可变 top_k（`maybe_top_k_arr`）。

---

## 8. Top-K 检索

### 8.1 `fast_topk_v2` — ragged/paged score topk

**Explain：** 对 ragged 或 paged layout 的 attention score 做 fused top-k，仅对 `[row_starts[i], row_starts[i]+lengths[i])` 区间有效。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/top_k.py L18-L35
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

**Comment：**

- `topk==1` 时 `fast_topk` 走 `torch.max` 快捷路径。
- DeepSeek V4 在 ROCm 上有 `deepseek_v4_topk_transform_512` 专用 transform。

---

## 9. ROCm Custom AllReduce

### 9.1 `init_custom_ar` — IPC 共享内存 AR

**Explain：** ROCm 平台绕过 NCCL 小消息开销，用 IPC handle 共享 GPU buffer 做 TP allreduce。支持 CUDA Graph 场景下的 buffer 注册。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/allreduce.py L7-L17
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

**Comment：**

- 返回 `fa`（function argument handle），后续 `all_reduce_reg` / `all_reduce_unreg` 使用。
- 另有 `init_custom_qr` quick allreduce 变体，适合小 tensor。

---

## 10. DEBUG 导出包装

### 10.1 批量 wrap 导出函数

**Explain：** import 末尾遍历 `_DEBUG_EXPORT_NAMES`，对每个已导出符号调用 `maybe_wrap_debug_kernel`，便于统一开启 API 日志。

**Code：**

```python
# 来源：sgl-kernel/python/sgl_kernel/__init__.py L216-L220
    for _name in _DEBUG_EXPORT_NAMES:
        if _name in globals():
            globals()[_name] = maybe_wrap_debug_kernel(
                globals()[_name], f"sgl_kernel.{_name}"
            )
```

**Comment：**

- 列表涵盖 attention/moe/gemm/kv/speculative 等 60+ 符号。
- ROCm 额外 append `gelu_quick`、`deepseek_v4_topk_transform_512`。
