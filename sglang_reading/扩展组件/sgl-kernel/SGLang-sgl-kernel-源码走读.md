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

> 读法：本篇先沿 `merge_state_v2` 走通 SRT 能力门禁、Python wrapper、dispatcher schema、C++ launcher 与 CUDA kernel，再用 MoE、KV I/O、投机验证、采样和 allreduce 对照不同 ABI。只读 Python wrapper 无法证明算子如何注册、谁拥有 fallback，也无法判断实际执行实现。

---

## 长文读法

这篇按“一个 SRT 计算如何跨过五道边界”读：`__init__.py` 先加载 `common_ops`、随后才尝试 preload CUDA runtime；wrapper 可能校验、转换、分配、缓存或选择可选依赖；`common_extension*.cc` 定义 schema 与设备 dispatch；C++/CUDA/ROCm 实现最后消费 tensor。导出层再按当前平台存在的符号批量套 debug wrapper。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立 sgl-kernel 边界 | 1 | Python 入口首先是动态库注册器，函数 re-export 依赖 import 副作用完成 |
| 排查 op 找不到或加载错变体 | 1.1 到 1.3 | CC90 选 fast-math 目录，其余选 precise-math 目录；preload 发生在 `common_ops` 加载之后 |
| 理解 attention wrapper | 2 | wrapper 校验 DeepSeek MLA / partial attention state 的形状和 workspace，再转到 PyTorch dispatcher |
| 理解 MoE 路由 wrapper | 3 | token-expert 对齐、top-k softmax / sigmoid 都是预分配输出张量后交给扩展 op 写入 |
| 排查 GEMM 或 KV 迁移 | 4 | GEMM wrapper 基本是轻量转发，KV I/O wrapper 明确 src/dst index 和每层迁移边界 |
| 理解投机采样与 top-k 后处理 | 5 | 这些接口大多以可变输出张量为契约，调用方要提前准备 `predict`、`accept_index`、概率或 score buffer |
| 看 allreduce / debug 特殊路径 | 6 | CUDA 与 ROCm 都有 custom allreduce，但 ABI 不同；debug wrapper 只包装当前平台已经导出的白名单符号 |

读的时候不要在本篇寻找 kernel 算法细节。本篇的价值是确认 SRT 调用扩展 op 前，Python 层负责了哪些输入契约，以及哪些错误会留到扩展库或 kernel launch 时才暴露。

## 1. 包初始化与动态库加载

### 1.1 `__init__.py`：import 时加载 architecture-specific ops

**问题与约束：** `sgl_kernel` 被上层 SRT import 时，Python 符号和 C++/CUDA 扩展 op 都必须已经注册；同时 CUDA runtime 的动态链接问题需要尽早处理。

**设计选择：** 在非 Apple Silicon 路径下，包初始化阶段先调用 `_load_architecture_specific_ops()`，再在 `torch.version.cuda` 存在时预加载 CUDA runtime。

**读法：** `common_ops` 变量本身不是业务对象，关键是加载扩展模块的副作用：扩展模块 import 后会把 `torch.ops.sgl_kernel.*` 注册进 PyTorch dispatcher。后续 Python wrapper 才能直接转调这些 op。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/__init__.py L18-L23
    # Initialize the ops library based on current GPU
    common_ops = _load_architecture_specific_ops()

    # Preload the CUDA library to avoid the issue of libcudart.so.12 not found
    if torch.version.cuda is not None:
        _preload_cuda_library()
```

**代码逻辑：** 初始化先选择并执行架构相关扩展库加载；如果当前 PyTorch 是 CUDA 构建，再调用 runtime 预加载。后续的 `from sgl_kernel.* import ...` 都建立在 op 已注册的前提上。

**为什么这样写：** wrapper 层不应在每次函数调用时检查扩展库是否存在，否则热路径会被 import 状态污染。把加载集中到包初始化，可以让函数调用退化成轻量的 `torch.ops` 转发。

**不变量与失败模式：** 扩展库必须在 wrapper 被调用前成功注册；Apple Silicon 走 Metal 分支，不暴露 CUDA 符号。当前顺序先执行 `_load_architecture_specific_ops()`，再调用 `_preload_cuda_library()`，所以后者无法补救 `common_ops` 自身首次加载时已经发生的 CUDA runtime 解析失败。

**要点：** `sgl-kernel` 的 Python 入口首先是动态库注册器，其次才是函数 re-export 容器。

### 1.2 `_load_architecture_specific_ops`：从所选目录加载 compiled extension

**问题与约束：** 同一个 wheel 可能携带多个 GPU 架构变体，加载时需要优先选择当前设备匹配的 compiled extension；直接按普通 import 名称查找容易拿到错误文件或 stub。

**设计选择：** 当发现匹配文件时，用 `importlib.util.spec_from_file_location("common_ops", ops_path)` 从具体路径构造 module spec，再 `exec_module` 执行该 `.so`。

**读法：** 这里刻意绕过普通 import 搜索路径，用已筛选出的扩展文件加载模块。它保证加载的是所选目录里的文件，但不能保证该二进制一定含当前设备的 gencode；目录选择与编译覆盖是两份证据。

**源码锚点：**

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

**代码逻辑：** 函数取优先级最高的匹配文件，构建 module spec，检查 spec 与 loader 是否存在，创建 module 对象并执行 loader。成功后记录变体名与实际文件路径，并返回 `common_ops` module。

**为什么这样写：** GPU kernel 对 SM 架构敏感，错误变体可能能 import 但运行失败或性能不对。按路径加载使“选择哪个 `.so`”成为显式决策，也让日志能定位实际加载文件。

**不变量与失败模式：** `_filter_compiled_extensions` 只把 `.so/.pyd/.dll` 排到其他匹配文件之前，并不删除其他文件；spec 和 loader 不能为空；`exec_module` 的副作用必须完成 op 注册。若选中的候选不是可用扩展，可能在 import 阶段失败；二进制缺目标 gencode 时也可能拖到 launch 才失败。

**要点：** 这一段是 sgl-kernel 的架构选择点：不是按 Python 包名加载，而是按硬件匹配后的文件路径加载。

### 1.3 `_preload_cuda_library`：在 `common_ops` 之后尝试把 CUDA runtime 放进全局符号表

**问题与约束：** 扩展库动态链接可能依赖 `libcudart.so.12` 或 `libcudart.so.13`；在某些部署环境中，Python 能找到 wheel，却不能在 `.so` 加载时解析 CUDA runtime。

**设计选择：** 遍历候选目录与 CUDA runtime 版本，找到存在的 `libcudart.so.*` 后，用 `ctypes.CDLL(..., mode=ctypes.RTLD_GLOBAL)` 预加载。

**读法：** `RTLD_GLOBAL` 让后续扩展库解析动态符号时能看到 CUDA runtime。函数找到第一个可用候选后立即返回，避免重复加载多个 runtime。

**源码锚点：**

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

**代码逻辑：** 双层循环生成候选路径；路径存在时 resolve 成绝对路径，调用 `ctypes.CDLL` 以全局模式载入，记录日志后结束。

**为什么这样写：** 运行环境的动态链接器搜索路径不总是和 CUDA runtime 位置一致。以 `RTLD_GLOBAL` 加载 runtime 可以帮助随后导入的扩展解析符号；由于当前调用顺序较晚，它不能把 `common_ops` 自身的链接错误提前局部化。

**不变量与失败模式：** 候选版本应匹配当前 PyTorch CUDA major；只在 CUDA 构建下调用；加载失败应继续尝试其他候选。若不使用 `RTLD_GLOBAL`，后续 `.so` 仍可能解析不到 runtime 符号。

**要点：** 这是部署层的后置尝试：不改变 kernel 逻辑，可能帮助后续扩展加载；但对前一步 `common_ops` 的首次链接错误来不及生效。

---

## 2. Attention wrapper

### 2.1 `merge_state_v2`：合并 partial attention state

**问题与约束：** Split-KV 或 cascade attention 会产生多段 partial output 与 log-sum-exp state，需要数值稳定地合并；wrapper 还要支持调用方传入输出 buffer，减少临时分配。

**设计选择：** 先把 `s_a/s_b` 转成 float32；若未传 `v_merged/s_merged`，就创建对应输出 tensor；最后调用 `torch.ops.sgl_kernel.merge_state_v2.default` 原地写输出。

**读法：** 合并 state 的数值敏感部分是 LSE，float32 能降低半精度溢出/下溢风险。输出 buffer 可选则让上层在高频调用中复用内存。

**源码锚点：**

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

**代码逻辑：** 函数接收两段 value/state，规范 state dtype，准备输出 tensor，调用注册 op 写入结果，并返回两个输出对象。

**为什么这样写：** Python wrapper 不重复实现 merge 数学，只负责 dtype 与 buffer 管理。这样既保留 kernel 性能，又让调用者可以控制输出内存生命周期。

**不变量与失败模式：** `v_a/v_b` 与 `s_a/s_b` 的 shape 必须匹配 kernel 预期；传入输出 buffer 时 shape/dtype 必须可写。wrapper 本身没有 fallback；FP8、非 CUDA 与不满足 pack-size 的情况由 SRT `merge_state()` 在调用前改走 Triton。

**要点：** 这是典型的 sgl-kernel wrapper：Python 层处理少量 ABI 约束，计算交给 dispatcher 后面的扩展 op。

### 2.2 `merge_state_v2` 的完整链：fallback、schema 与 kernel 各有所有者

SRT 调用方先做能力门禁；只有 CUDA、dtype 受支持且 head dimension 满足 pack-size 约束时才进入 sgl-kernel，否则显式改走 Triton：

```python
# 来源：python/sglang/srt/layers/attention/merge_state.py L26-L45
def merge_state(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output: Optional[torch.Tensor] = None,
    output_lse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

扩展注册层把同名 schema 绑定到 CUDA dispatch key：

```cpp
// 来源：sgl-kernel/csrc/common_extension.cc L44-L45
  m.def("merge_state_v2(Tensor v_a, Tensor s_a, Tensor v_b, Tensor s_b, Tensor! v_merged, Tensor! s_merged) -> ()");
  m.impl("merge_state_v2", torch::kCUDA, &merge_state_v2);
```

最终 C++ 入口才检查 contiguous、device、rank、shape，并按输出 dtype 发射 CUDA launcher：

```cpp
// 来源：sgl-kernel/csrc/attention/merge_attn_states.cu L184-L206
void merge_state_v2(
    at::Tensor v_a, at::Tensor s_a, at::Tensor v_b, at::Tensor s_b, at::Tensor v_merged, at::Tensor s_merged) {
  // Input tensors must be contiguous
  CHECK_INPUT(v_a);  // v_a prefix_output (seq_len, num_heads, head_dim)
  CHECK_INPUT(s_a);  // s_a prefix_lse (seq_len, num_heads)
  CHECK_INPUT(v_b);  // v_b suffix_output (seq_len, num_heads, head_dim)
  CHECK_INPUT(s_b);  // s_b suffix_lse (seq_len, num_heads)
  // v_merged output (seq_len, num_heads, head_dim)
  // s_merged output_lse (seq_len, num_heads)
  auto device = v_a.device();
  CHECK_EQ(s_a.device(), device);
  CHECK_EQ(v_b.device(), device);
  CHECK_EQ(s_b.device(), device);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  CHECK_EQ(v_a.size(0), s_a.size(0));
  CHECK_EQ(v_a.size(1), s_b.size(1));
  DISPATCH_BY_SCALAR_DTYPE(v_merged.dtype(), CALL_MERGE_ATTN_STATES_LAUNCHER);
}
```

因此一次失败的定位顺序应是：先看 SRT 是否进入 custom-op 分支，再看 wrapper 输出 buffer，再看 dispatcher 是否有 CUDA kernel，最后才看 C++ checks 与 launch。把所有问题笼统归为“sgl-kernel 不支持”会丢失真正的责任层。

### 2.3 `cutlass_mla_decode`：校验 DeepSeek MLA paged decode 形状

**问题与约束：** MLA decode 的 query 被拆成 latent 与 rope 两部分，KV cache 又把 latent KV 与 rope K 合并存储；CUTLASS kernel 对维度常量有固定假设。

**设计选择：** wrapper 在调用 kernel 前断言 `q_nope/q_pe/kv_c_and_k_pe_cache` 都是 3D，校验 batch/head 一致，并把 MLA 维度固定为 `D_latent=512`、`D_rope=64`。

**读法：** 这里先在 Python 层把模型 ABI 固定住：`q_nope` 对 latent 维，`q_pe` 对 rope 维，cache 最后一维必须等于二者之和。错误 shape 尽早在 assert 处暴露。

**源码锚点：**

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

**代码逻辑：** 函数签名收齐 query、cache、sequence length、page table、workspace 与 scale；进入后先断言 rank，再解出 batch/head/维度，最后校验 DeepSeek MLA 的固定 latent/rope/cache 维度。

**为什么这样写：** CUTLASS kernel 对 tile 与布局假设严格，shape 错误如果进入 kernel 会变成难定位的非法访问或错误输出。Python assert 是更便宜、更清晰的 ABI 闸门。

**不变量与失败模式：** `q_nope` 与 `q_pe` 的 batch/head 必须一致；cache 最后一维必须是 576；默认 `num_kv_splits=1` 是为了避开 CUDA Graph 相关问题。若模型维度变体不同，需要对应更新 wrapper 与 kernel。

**要点：** `cutlass_mla_decode` 的前半段就是模型布局契约，先把错形状拦在 Python 层。

---

## 3. MoE wrapper

### 3.1 `moe_align_block_size`：把 token-expert 分配对齐到 block

**问题与约束：** MoE grouped GEMM 需要按 expert 分组并按 block size 对齐的 token 布局；这些输出 buffer 通常由上层预分配，kernel 应原地填充以减少额外分配。

**设计选择：** wrapper 接收 `topk_ids`、expert 数、block size 与多个输出 buffer，直接调用 `torch.ops.sgl_kernel.moe_align_block_size.default`，不返回新对象。

**读法：** 这个函数是 MoE routing 到 grouped GEMM 之间的布局整理步骤。Python 层只传递 buffer，排序、padding 和计数由扩展 op 完成。

**源码锚点：**

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

**代码逻辑：** 函数把输入 expert id、输出 token/expert buffer、padding 后 token 计数和 cumsum buffer 原样传给 op。输出通过 mutable tensor 写回。

**为什么这样写：** MoE 对齐是大批量张量重排，Python 侧循环会成为瓶颈。保持 in-place wrapper 可以让调用方复用缓存 buffer，并把工作集中到 GPU kernel。

**不变量与失败模式：** 输出 buffer 的 shape 与 dtype 必须符合 kernel 约定；`block_size` 必须与后续 grouped GEMM tile 对齐；`pad_sorted_token_ids` 会改变填充策略。若 `num_tokens_post_pad` 没有正确写回，后续 GEMM 会读错有效 token 数。

**要点：** 这段的设计信号是“无返回值”：调用方关心的是被填好的布局 buffer。

### 3.2 `topk_sigmoid`：sigmoid gating 的 top-k 路由

**问题与约束：** DeepSeek 风格 MoE routing 可能使用 sigmoid gating、top-k expert 选择、renormalize 和 per-expert correction bias；输出需要写入预分配的 weights/ids buffer。

**设计选择：** wrapper 用 docstring 明确输入输出形状与 `correction_bias` dtype 约束，并把参数直接转给 `torch.ops.sgl_kernel.topk_sigmoid.default`。

**读法：** `topk_sigmoid` 把 gating logits 到 top-k expert 的转换放到扩展 op 中完成。Python 层不返回 tensor，而是要求调用者提供 `topk_weights` 与 `topk_ids` 输出位置。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/moe.py L57-L80
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

**问题与约束：** 量化推理需要 INT8/FP8 矩阵乘，并携带 activation/weight scale、输出 dtype 和可选 bias；Python 层不应重写矩阵乘逻辑。

**设计选择：** `int8_scaled_mm`、`fp8_blockwise_scaled_mm`、`fp8_scaled_mm` 都作为薄 wrapper，把输入矩阵、scale、dtype 和 bias 原样传给对应 `torch.ops.sgl_kernel` op。

**读法：** 这组函数把量化 GEMM 的 Python API 固定下来：调用方按同一模式提供矩阵与 scale，具体 kernel 实现由注册的扩展 op 决定。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/gemm.py L13-L42
def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
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
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )
```

**代码逻辑：** 代码段展示了 INT8 scaled matmul 的 op 调用，以及 FP8 blockwise/per-tensor 两个 wrapper 的函数边界和参数转发。

**为什么这样写：** 量化层需要一个稳定 Python 入口，但性能和设备支持在 C++/CUDA 层变化更快。薄 wrapper 可以让上层量化配置选择函数名，而不绑定具体实现细节。

**不变量与失败模式：** scale tensor 的粒度必须与所选 wrapper 匹配；`out_dtype` 必须是 kernel 支持的输出类型；bias 只在对应 op 支持时传入。若把 blockwise scale 传给 per-tensor wrapper，会得到错误缩放结果。

**要点：** `gemm.py` 的可读重点是 API 形状，而不是 matmul 算法；算法在注册 op 后面。

### 4.2 `transfer_kv_per_layer`：单层 K/V 按 index 迁移

**问题与约束：** host cache、HiCache、layout conversion 或 disaggregation 的本地 copy 环节，需要按 index 把单层 K/V 从源 buffer 拷贝到目标 buffer；不同平台的 warp 配置不同，还需要限制 block quota。这个 wrapper 本身不包含远端传输协议。

**设计选择：** wrapper 接收 source/destination K/V、源/目标 indices、item size、block quota 和平台相关 `num_warps_per_block`，直接调用 `transfer_kv_per_layer` op。

**读法：** 这个 API 把“按 token/page index 迁移 K/V”封装成单层操作。Python 层决定并发配置与平台默认值，实际拷贝由 kernel 完成。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/kvcacheio.py L13-L34
def transfer_kv_per_layer(
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

**问题与约束：** 树形投机解码需要根据 draft tree、target/draft 概率和随机数接受或拒绝 token，并把预测结果、接受位置和接受数量写回调用方提供的 buffer。

**设计选择：** wrapper 把 `predicts`、`accept_index`、`accept_token_num` 明确标为 mutable，并把 tree 结构、随机数、概率与阈值参数传给 `tree_speculative_sampling_target_only` op。

**读法：** 这个函数的返回值是 `None`，因为结果通过多个 mutable tensor 输出。它适合在 speculative worker 中作为一次 target 验证步骤。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/speculative.py L4-L35
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
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
    )
```

**代码逻辑：** 函数签名列出三类输入：可变输出 buffer、draft tree 遍历结构、概率/随机数/阈值参数。函数体开始调用扩展 op，并把可变输出作为前几个参数传入。

**为什么这样写：** 投机验证要更新多个结果数组，用多个返回 tensor 会增加分配和同步。mutable buffer 让调用方控制内存，并与后续 accept/reject 流程共享结果。

**不变量与失败模式：** mutable buffer 必须提前按 batch/tree 大小分配；tree 结构数组之间的索引语义必须一致；即使 `deterministic=True`，wrapper 仍原样传入两组 uniform samples，具体使用规则属于 kernel 与 SRT 调用契约，不能由签名猜测。若 buffer shape 不足或 tree index 不一致，结果可能错误或 launch 失败。

**要点：** 这里的 wrapper 明确展示了 speculative sampling 的 ABI：不是返回一个对象，而是原地更新多组状态数组。

### 5.2 `top_k_renorm_probs`：先选 FlashInfer 或内部 op，再谈 dtype 归一

**问题与约束：** 采样前的概率分布需要 top-k 截断并重新归一化，但公开 API 不固定落到 sgl-kernel op：可选依赖和设备类型先决定实现。

**设计选择：** MUSA 或 FlashInfer 不可用时，internal helper 把 `probs` 转 float32、top-k tensor 转 int，分配输出后调用内部 op；其他情况直接委托 FlashInfer。

**读法：** “升到 float32”只属于内部 fallback 分支。FlashInfer 分支的 dtype 与 kernel 行为应按 FlashInfer 自身契约判断，不能把 internal helper 的行为推广到公开 API 的所有调用。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/sampling.py L18-L26
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_k_renorm_probs.default(
        probs, renorm_probs, maybe_top_k_arr, top_k_val
    )
    return renorm_probs
```

公开函数再做实现分流：

```python
# 来源：sgl-kernel/python/sgl_kernel/sampling.py L56-L59
    if probs.device.type == "musa" or not _has_flashinfer:
        return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))
    else:
        return _flashinfer_sampling.top_k_renorm_probs(probs, top_k)
```

**代码逻辑：** 代码段位于 internal helper 末尾：输入概率转 float，top-k 数组转 int，分配同 shape 输出，调用扩展 op 写结果，然后返回 renormalized 概率。

**为什么这样写：** wrapper 复用已安装的 FlashInfer 主路径，同时为 MUSA 或缺失依赖保留内部 op。`maybe_top_k_arr` 支持每行不同 top-k，统一转 int 简化内部 kernel ABI。

**不变量与失败模式：** 先记录实际分支，再比较数值或性能；debug wrapper 只包住公开 `top_k_renorm_prob`，但内部是否出现 `torch.ops.sgl_kernel` 仍由 `_has_flashinfer` 与设备决定。

**要点：** 采样 wrapper 的核心先是实现选择；只有内部分支才由 `sgl_kernel` op 完成 top-k 截断，FlashInfer 分支不经过该 op。

### 5.3 `fast_topk_v2`：ragged/paged score 的 top-k 接口契约

**问题与约束：** Attention score 可能是 ragged 或 paged layout，每行有效区间长度不同；top-k 只能作用在每行 `[row_starts[i], row_starts[i] + lengths[i])` 的有效范围内。

**设计选择：** 函数签名显式接收 `score`、`lengths`、`topk` 与可选 `row_starts`，docstring 把 ragged 场景下的有效区间规则写成接口契约。

**读法：** `fast_topk_v2` 的关键不是普通 dense topk，而是按行长度和可选起点限定搜索范围。这样同一个 score tensor 可以承载 ragged/paged 两种布局。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/top_k.py L16-L42
def fast_topk_v2(
    score: torch.Tensor,
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
    assert (
        topk == 2048
    ), "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths, row_starts)
    return topk_indices
```

**代码逻辑：** 代码段展示函数参数与返回约定：`score` 是 `[B, L]`，`lengths` 是每行有效长度，`row_starts` 只在 ragged key 中需要，返回 `[B, topk]` 的 indices。

**为什么这样写：** 稀疏或分页布局中，无效位置不能参与 top-k；把有效区间写进接口，比要求调用者提前 materialize dense mask 更省内存，也更接近 kernel 的实际访问模式。

**不变量与失败模式：** 当前 wrapper 硬断言 `topk == 2048` 且 `score` 为二维，因此它是 DeepSeek v3.2 专用入口，不是任意 k 的通用 ragged top-k。`lengths` 必须按 batch 对齐，ragged layout 还要提供正确 `row_starts`；边界错会把 padding 或其他请求的 score 当成候选。

**要点：** 这段函数体同时证明 layout 契约和 `topk == 2048` 硬门禁：`fast_topk_v2` 服务 DeepSeek v3.2 的 ragged/paged score，不是普通 vocab top-k。

---

## 6. CUDA/ROCm AllReduce 与调试包装

### 6.1 `init_custom_ar`：同名 API 下的两套 allreduce ABI

**问题与约束：** custom allreduce 初始化要建立跨 rank 共享资源，但 CUDA 与 ROCm 所需的 handle、offset 和后续执行接口不同。

**设计选择：** 在 `torch.version.hip is not None` 分支内定义 `init_custom_ar`，把 meta、rank data、handles、offsets、rank 与 full_nvlink 标志传给 `torch.ops.sgl_kernel.init_custom_ar.default`，并返回 int handle。

**读法：** 这里展示的是 HIP 分支；文件的 `else` 还定义 CUDA 版 `init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)`。两者都返回整数句柄，但不能交换参数或后续调用协议。

**源码锚点：**

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

CUDA 分支保留同名入口，但参数已经不同：

```python
# 来源：sgl-kernel/python/sgl_kernel/allreduce.py L95-L102
else:

    def init_custom_ar(
        ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
    ) -> int:
        return torch.ops.sgl_kernel.init_custom_ar.default(
            ipc_tensors, rank_data, rank, full_nvlink
        )
```

**代码逻辑：** 函数接收通信元数据和 IPC 信息，转调扩展 op，并把 op 返回的 handle 交给调用方保存。

**为什么这样写：** 自定义 allreduce 的资源准备和后续执行是分离的。初始化时集中建立共享内存/IPC 上下文，后续 allreduce 调用就能只传 handle 与 tensor。

**不变量与失败模式：** HIP 的 handles/offsets 必须覆盖参与 rank，CUDA 的 `ipc_tensors` 必须符合另一套注册约定；返回 handle 都要保持到 dispose。平台判断错时通常不是“符号不存在”，而是 ABI 与后续方法集合不匹配。

**要点：** AllReduce wrapper 体现的是同名入口下的平台分叉：CUDA 与 ROCm 都有 custom path，ROCm 额外暴露 deterministic、registered/unregistered 与 quick-allreduce 族。

### 6.2 `_DEBUG_EXPORT_NAMES`：批量给导出函数套 debug wrapper

**问题与约束：** sgl-kernel re-export 了大量函数，如果逐个手写 debug wrapper，容易遗漏；但只有实际存在于 `globals()` 的符号才能被包装，平台差异符号不能强行引用。

**设计选择：** import 末尾遍历 `_DEBUG_EXPORT_NAMES`，如果名字在当前 globals 中，就用 `maybe_wrap_debug_kernel` 替换为带日志的 wrapper。

**读法：** 这是统一 debug instrumentation 的最后一步。它不改变函数名，也不改变导出表，只在需要时把可调用对象包一层。

**源码锚点：**

```python
# 来源：sgl-kernel/python/sgl_kernel/__init__.py L216-L220
    for _name in _DEBUG_EXPORT_NAMES:
        if _name in globals():
            globals()[_name] = maybe_wrap_debug_kernel(
                globals()[_name], f"sgl_kernel.{_name}"
            )
```

**代码逻辑：** 循环从 debug 名单取符号名，检查当前平台/导入路径是否真的导出了该符号；存在则用 wrapper 替换 globals 中的函数对象，日志名带 `sgl_kernel.` 前缀。

**为什么这样写：** 平台和可选依赖会影响实际导出符号。先检查 `globals()` 可以避免 debug 包装阶段因为缺失符号失败，同时让新增导出只需加入名单即可获得统一日志。

**不变量与失败模式：** 被包装对象必须可调用，不存在的名字必须跳过；返回语义是否完全透明取决于 SRT 的 `debug_kernel_api` 实现，不能只凭这里断言。若直接索引 globals，平台特有符号缺失会导致 import 阶段崩溃。

**要点：** debug 包装放在 import 收尾阶段，保证所有 re-export 已完成，再按当前平台实际符号集合批量增强。

---

## 运行验证

维护本文时，先用下面的命令确认 Python wrapper 与 `torch.ops.sgl_kernel` 边界还在原位：

```powershell
rg -n "torch.ops.sgl_kernel|maybe_wrap_debug_kernel|merge_state_v2|grouped_gemm|fast_topk_v2|init_custom_ar" sglang/sgl-kernel/python/sgl_kernel
```

预期信号：

- `attention.py`、`moe.py`、`gemm.py`、`kvcacheio.py` 中相应 wrapper 仍能追到 `torch.ops.sgl_kernel`；同时应识别 sampling 分流、纯 PyTorch fast path 与 GPTQ typo 等例外。
- `__init__.py` 仍集中 re-export 并在末尾做 debug wrapper 包装。
- `top_k.py` 仍能看到 ragged/paged top-k 的专用接口。
- `allreduce.py` 仍体现 CUDA / ROCm 平台分叉。

如果某个专题结论依赖的 wrapper、schema、dispatch key 或 SRT 门禁发生变化，应沿该算子的五层链重新核对；不能因为其他 wrapper 仍是薄转发就保留旧结论。
