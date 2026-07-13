---
title: "Python-API · 核心概念"
type: concept
framework: flash-attn
topic: "Python-API"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/concept
  - source-reading
updated: 2026-07-10
---
# Python-API · 核心概念

## 读者任务

这篇先建立 Python API 层的心理模型。读完后你应该能回答：

- 为什么 API 分成 dense、packed、varlen、KV cache，而不是一个万能函数。
- `maybe_contiguous`、custom op、fake tensor、autograd Function 分别解决什么边界问题。
- `cu_seqlens`、`softmax_lse`、`S_dmask`、`rng_state` 在 Python 层各自扮演什么角色，以及哪些只是协议对象、不能按名字望文生义。
- 哪些问题应停在 Python/API 层排查，哪些必须继续进入 compiled C++/CUDA/HIP 或 ROCm Triton backend。

## API 形态是场景分类

公开 API 表面已经把主要输入形态列出来。它不是简单的函数清单，而是 attention 场景分类。

```python
# 来源：flash_attn/__init__.py L1-L16
from pkgutil import extend_path

# look for every subdir with flash_attn base name such that fa2 and fa4 can be co-installed
__path__ = extend_path(__path__, __name__)

__version__ = "2.8.4"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)
```

| API 形态 | 输入模型 | 解决的问题 | 继续读 |
|----------|----------|------------|--------|
| `flash_attn_func` | Q/K/V 分开，fixed-length batch | 最普通的训练/prefill forward | [[FlashAttention-Python-API-源码走读]] |
| `flash_attn_qkvpacked_func` | Q/K/V 堆在一个 tensor | backward 少做一次显式 concat | [[FlashAttention-Backward]] |
| `flash_attn_varlen_func` | 有效 token 连续拼接 + `cu_seqlens` | 避免 padding token 参与计算 | [[FlashAttention-Python-API-数据流]] |
| `flash_attn_with_kvcache` | q + K/V cache + 可选新增 k/v | decode 时更新并读取 cache | [[FlashAttention-KV-Cache]] |

读者抓手：API 名称不是语法糖，而是性能契约。上层框架选错 API，后面的 kernel specialization 很难补救数据搬运浪费。

## Python 层有四个职责

| 职责 | 源码对象 | 判断标准 |
|------|----------|----------|
| 输入归一化 | `maybe_contiguous`、head dim padding、默认 `softmax_scale` | 是否改变 tensor layout 或补齐 head_dim |
| dispatcher 适配 | `_torch_custom_op_wrapper`、`torch.ops.flash_attn.*` | 是否服务 PyTorch 2.4+ custom op / compile |
| autograd 状态 | `FlashAttnFunc.save_for_backward` | forward 保存哪些对象给 backward |
| backend 调用 | `flash_attn_gpu.fwd/varlen_fwd/fwd_kvcache` | 当前模块选中 CUDA/HIP compiled extension，还是 ROCm Triton |

源码开头显示了 extension 和 contiguous 边界：

```python
# 来源：flash_attn/flash_attn_interface.py L10-L28
# isort: off
# We need to import the CUDA kernels after importing torch
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if not USE_TRITON_ROCM and getattr(torch.version, 'hip', None) is not None:
    try:
        import flash_attn_2_cuda
    except ImportError:
        warnings.warn("flash_attn_2_cuda (which has ROCm/HIP kernels) not found, falling back to Triton implementation")
        USE_TRITON_ROCM = True

if USE_TRITON_ROCM:
    from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu
else:
    import flash_attn_2_cuda as flash_attn_gpu

# isort: on

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x
```

读者抓手：import 问题属于 backend/ABI 边界；last-dim stride 问题属于 tensor layout 边界。只有 HIP 分支会在 compiled extension 导入失败后把 `USE_TRITON_ROCM` 改为真；普通 CUDA 分支导入 `flash_attn_2_cuda` 失败不会自动转去 Triton。两者都发生在真正 kernel 主循环之前。

## Custom Op 和 Fake Tensor 是编译生态边界

PyTorch 2.4+ 使用 `torch.library.custom_op` 和 `register_fake`。旧版本退化成 no-op wrapper，但 API 表面保持一致。

```python
# 来源：flash_attn/flash_attn_interface.py L62-L81
# The reason for this is that we are using the new custom_op and register_fake
# APIs, which support inplace modification of inputs in the function itself
if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func
        if fn is None:
            return wrap
        return fn
    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper
```

fake forward 不跑 CUDA kernel，只声明输出形状和 dtype：

```python
# 来源：flash_attn/flash_attn_interface.py L117-L144
@_torch_register_fake_wrapper("flash_attn::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: Optional[torch.Tensor],
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen_q, num_heads, head_size = q.shape
    seqlen_k = k.shape[1]
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        if torch.cuda.is_available() and torch.version.hip:
            p = torch.empty((batch_size, num_heads, seqlen_q, seqlen_k), dtype=q.dtype, device=q.device, layout=q.layout)
        else:
            p = torch.empty((batch_size, num_heads, round_multiple(seqlen_q, 128), round_multiple(seqlen_k, 128)), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state
```

这里的 `p` 对应内部 `S_dmask` 槽位：默认是空 tensor；请求 `return_softmax` 时，CUDA fake shape 把两个序列维补到 128 的倍数，HIP fake shape 不做这层补齐。它只声明元数据，不证明 backend 真能运行，也不把该槽位变成数学意义上的完整 attention probability。

读者抓手：遇到 `torch.compile`、fake tensor、tracing 问题时，不要直接去 kernel。先检查 PyTorch 版本、custom op 注册和 fake 输出形状。这个 custom-op/fake 主线覆盖 dense/varlen wrapper；KV-cache 入口在当前基线直接调用 `flash_attn_gpu.fwd_kvcache`，不能无条件套用同一 compile 行为。

## Autograd Function 是 forward/backward 协议边界

普通 dense API 最终进入 `FlashAttnFunc.apply`。forward 会补默认 scale，必要时 pad head dim，然后调用 wrapped forward；如果需要梯度，它保存 Q/K/V/out/LSE/RNG。

```python
# 来源：flash_attn/flash_attn_interface.py L828-L878
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        is_grad_enabled,
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)
```

读者抓手：`softmax_lse`、`rng_state` 不是附带信息。它们是 backward 重建 softmax 权重与复现 dropout mask 的协议字段。`return_attn_probs=True` 只决定公开 API 返回三元组；内部只有 `dropout_p > 0` 才要求 backend 生成 `S_dmask`，所以 dropout 为 0 时第三项不能当成真实概率矩阵。

## Varlen 是连续 token 加边界数组

`unpad_input` 把 padded batch 变成连续 token、原始位置 indices 和 `cu_seqlens`。这让 kernel 处理连续内存，同时仍知道每条样本的边界。

```python
# 定位：flash_attn/bert_padding.py L111-L128（摘要/骨架；去除 upstream 尾随空格）
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )
```

这五项分别是 packed hidden states、回填原 batch 的 `indices`、样本边界 `cu_seqlens`、最大物理打包长度和 `used_seqlens_in_batch`。要特别留意当前实现中的两套长度：前四项的打包边界来自 `attention_mask + unused_mask`，第五项只统计 `attention_mask`。`bert_padding.py` 的 docstring 把 `seqused` 写成两种 mask 之和，与实际第五返回表达式不一致；调用者应以代码返回值为准。

读者抓手：varlen 不改变每条序列内部 attention 语义，只改变 batch 存储形态。`cu_seqlens` 错，通常是 correctness bug，而不是性能小问题；把第五返回值误当物理 packed length，同样会混淆协议。

## KV cache API 是推理 decode 契约

KV cache API 明确声明：如果传入新 `k/v`，它会原地更新 cache，并用更新后的 cache 做 attention；它还处理 RoPE、paged KV、cache batch index、SplitKV。

```python
# 来源：flash_attn/flash_attn_interface.py L1485-L1514
def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0, # 0.0 means deactivated
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
    return_softmax_lse=False,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.
```

公开参数可以同时出现在函数签名中，不代表任意组合都受支持。当前 C++ 入口明确拒绝 paged KV 与 `cache_batch_idx`、leftpad 同时使用：

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1247-L1253
    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
```

```cpp
// 来源：csrc/flash_attn/flash_api.cpp L1397-L1401
    params.is_seqlens_k_cumulative = !(seqlens_k_.has_value());
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
```

读者抓手：decode 性能与正确性问题要从 cache load/update、物理结束位置、paged KV、SplitKV 和 RoPE position 查起，不要把它当成普通 dense forward 的小 shape。这个 API 不支持 backward；重复 `cache_batch_idx` 配合 append 还会产生最终写入者不确定的风险。

## 运行验证

当前环境若不能加载 compiled extension 或 Aiter，也能先验证 Python 协议本身：

```powershell
@'
import ast
from pathlib import Path

interface = Path("flash-attn/flash-attention/flash_attn/flash_attn_interface.py")
padding = Path("flash-attn/flash-attention/flash_attn/bert_padding.py")
for path in (interface, padding):
    ast.parse(path.read_text(encoding="utf-8"))

tree = ast.parse(padding.read_text(encoding="utf-8"))
fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "unpad_input")
ret = next(node for node in ast.walk(fn) if isinstance(node, ast.Return))
assert isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 5
print("AST + unpad five-return contract: PASS")
'@ | python -

rg -n 'USE_TRITON_ROCM|noop_custom_op_wrapper|register_fake|return_softmax=return_softmax and dropout_p|fwd_kvcache' flash-attn/flash-attention/flash_attn/flash_attn_interface.py
```

预期 AST 与五返回值断言通过，静态定位同时覆盖 backend 分叉、PyTorch 2.4 前后分叉、fake 注册、dropout 下的 `S_dmask` 门禁和 KV-cache 直调。它不证明 extension ABI、GPU 数值或 `torch.compile` 图实际可执行。

## 复盘

Python API 层可以压成一句话：它把用户调用变成后端契约。这个契约包含输入形态、layout、autograd 保存项、compiler fake shape、backend 路由和推理 cache 状态；其中 compiled extension 才继续进入 pybind/C++，ROCm Triton 不能套用同一调用链。下一篇 [[FlashAttention-Python-API-源码走读]] 会沿真实调用顺序走一遍。
