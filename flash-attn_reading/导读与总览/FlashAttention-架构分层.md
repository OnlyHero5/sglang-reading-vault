---
title: "FlashAttention 架构分层"
type: concept
framework: flash-attn
topic: "导读与总览"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/concept
  - source-reading
updated: 2026-07-10
---
# FlashAttention 架构分层

## 读者任务

这篇解决“一个上层 attention 调用到底穿过哪些边界”的问题。读完后你应该能做到：

- 把模型层、Python API、C++ binding、kernel dispatch、GPU kernel 分清。
- 看到报错、profile 栈或编译日志时，能判断问题在张量形态、extension ABI、模板分派还是 kernel 主循环。
- 知道 FA4 CuTeDSL/JIT 是旁路后端，不等同于 FA2 的 `flash_attn_2_cuda`。

## 先建立模型：五道门

```mermaid
flowchart TB
    L1["模型/框架层<br/>MHA / inference cache / SGLang backend"]
    L2["Python API 层<br/>dense / packed / varlen / KV cache"]
    L3["C++ binding 层<br/>shape/dtype/device 检查 + 参数包"]
    L4["Kernel dispatch 层<br/>dtype/head_dim/mask/softcap 模板化"]
    L5["GPU kernel 层<br/>QK + mask + online softmax + PV + LSE"]
    L6["FA4 CuTeDSL<br/>独立 JIT API"]
    L1 --> L2 --> L3 --> L4 --> L5
    L2 --> L6
```

这五道门不是文件目录的简单分层，而是责任边界：

| 层 | 消费什么 | 产出什么 | 常见问题 |
|----|----------|----------|----------|
| 模型/框架层 | hidden states、MHA 配置、inference cache | Q/K/V、mask、cache 参数 | GQA/MQA、RoPE、cache offset 错 |
| Python API 层 | PyTorch tensor 和用户参数 | custom op 调用、autograd 上下文 | layout、varlen、compile/fake tensor |
| C++ binding 层 | extension 输入 | `Flash_fwd_params` / `Flash_bwd_params` | dtype、device、stride、SM 架构 |
| Dispatch 层 | `Flash_fwd_params` 和 runtime bool | 编译期模板实例 | head_dim、causal、local、ALiBi、softcap 组合 |
| GPU kernel 层 | 指针、shape、模板常量 | `out`、`softmax_lse`、可选 `S_dmask` | 数值差异、LSE、mask、dropout、访存 |

## 模型层：先把语义压成 API 参数

模型层并不直接写 CUDA。它选择 FlashAttention 还是 reference attention，并把 `causal`、`softmax_scale`、dropout、ALiBi、window 等上层语义传给内部 attention 类。

```python
# 来源：flash_attn/modules/mha.py L448-L480
        inner_attn_cls = (
            partial(FlashSelfAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else SelfAttention
        )
        inner_cross_attn_cls = (
            partial(FlashCrossAttention, alibi_slopes=alibi_slopes, window_size=window_size)
            if use_flash_attn
            else CrossAttention
        )
        if not self.cross_attn:
            self.Wqkv = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.Wq = nn.Linear(embed_dim, embed_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.Wkv = nn.Linear(embed_dim, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        if self.dwconv:
            if self.num_heads_kv == self.num_heads:
                self.dwconv_qkv = nn.Conv1d(
                    qkv_dim, qkv_dim, kernel_size=3, padding=2, groups=qkv_dim
                )
            else:
                self.dwconv_q = nn.Conv1d(
                    embed_dim, embed_dim, kernel_size=3, padding=2, groups=embed_dim
                )
                self.dwconv_kv = nn.Conv1d(kv_dim, kv_dim, kernel_size=3, padding=2, groups=kv_dim)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, softmax_scale=softmax_scale, attention_dropout=dropout
        )
```

读者抓手：`use_flash_attn` 是模型层的分叉，不是 kernel 层的分叉。如果模型还没有把 Q/K/V、head 数、cache offset 整对，底层 kernel 再快也只能放大错误。

## Python API 层：决定走哪个 dispatcher 入口

Python 层一方面暴露 dense、varlen、KV cache API，另一方面适配 PyTorch 2.4+ 的 custom op dispatcher。

```python
# 来源：flash_attn/flash_attn_interface.py L147-L150
if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward
```

普通 dense API 的签名已经包含后续 dispatch 需要的大部分分叉条件：

```python
# 来源：flash_attn/flash_attn_interface.py L1156-L1168
def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
```

读者抓手：`causal/window_size/softcap/alibi/return_attn_probs` 看起来是 Python 参数，实际会影响 C++ 校验、参数包字段、模板分派数量和返回张量形态。

## C++ binding 层：把用户输入改写成 kernel 契约

C++ 入口先做硬约束检查。这里的约束不是“风格要求”，而是 kernel 访存和模板假设能否成立。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L351-L381
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
```

随后它把 tensor、shape、mask、dropout、softmax、LSE 指针都装入 `Flash_fwd_params`，再交给 `run_mha_fwd`。

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L452-L470
    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap
                     );
```

读者抓手：C++ binding 是“语义冻结点”。Python 的灵活参数到这里变成指针、stride、shape、flag 和 buffer 生命周期。

## Dispatch 层：运行时条件变成模板常量

FA2 的性能取舍是把许多分支提前实例化，而不是让 kernel 主循环里到处判断。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_launch_template.h L63-L79
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
                BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                    ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                            // Will only return softmax if dropout, to reduce compilation time.
                            // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                            // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                            // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                            // If Is_local, set Is_causal to false
                            auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !Has_alibi && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst && !Has_alibi, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
```

读者抓手：编译慢、wheel 大、源码里 `.cu` 文件多，本质上都和这层的组合爆炸有关。排查时先问“这个功能是运行时字段，还是模板常量”。

## GPU kernel 层：只做 tile 内必需的事

kernel 主循环的核心顺序是：`QK` 得到局部 score，mask/softcap 修正 score，online softmax 同时更新归一化状态和历史 `O`，再把局部概率块乘以 V。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L319-L347
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
```

最后 epilogue 把在线维护的状态转成每行 LSE。

```cpp
// 来源：csrc/flash_attn/src/flash_fwd_kernel.h L431-L433
    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);
```

读者抓手：FlashAttention 的“省显存”不靠丢精度或近似，而是让 `P` 只在 tile 内短暂存在，长期保存的是 `out` 和每行 `softmax_lse`。

## FA4 边界：同名包下的 CuTeDSL 路径

FA4 在同一个 `flash_attn` namespace 下暴露 CuTeDSL API，但它不是 FA2 `flash_attn_2_cuda` extension 的同一条实现。

```python
# 来源：flash_attn/cute/__init__.py L1-L18
"""Flash Attention CUTE (CUDA Template Engine) implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fa4")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .interface import (
    flash_attn_func,
    flash_attn_varlen_func,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen_func",
]
```

读者抓手：看到 `flash_attn.cute` 时，要切换到 FA4/JIT 心理模型；看到 `flash_attn_2_cuda` 时，才沿 FA2 C++/CUDA extension 继续追。

## 运行验证

| 验证目标 | 操作 | 预期 |
|----------|------|------|
| 判断模型层是否切到 FlashAttention | 检查调用栈是否出现 `FlashSelfAttention` 或 `FlashCrossAttention` | 出现说明模型层已经选择 FA API |
| 判断是否进入 PyTorch custom op | 检查 `torch.ops.flash_attn._flash_attn_forward` 是否存在 | PyTorch 2.4+ 应走 custom op wrapper |
| 判断 C++ binding 失败原因 | 看异常是否来自 dtype、device、stride、SM 架构检查 | 这类错误在 kernel launch 前失败 |
| 判断 dispatch 组合 | 用 head_dim、dtype、causal、local、ALiBi、softcap 对照编译文件和模板开关 | 能定位到具体 specialization |
| 判断 kernel 数值问题 | 对比 `out` 与 `softmax_lse`，再看 mask/softcap/dropout 分支 | `P` 不应作为生产路径长期输出 |

## 复盘

架构分层的核心判断是：FlashAttention 不是“Python 调用一个 CUDA 函数”这么简单。它是上层语义逐层下沉：模型层选择 attention 形态，Python API 管张量和 autograd，C++ binding 固化参数契约，dispatch 管组合爆炸，GPU kernel 只执行 tile 内最紧的计算闭环。
