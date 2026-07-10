---
title: "Hopper与CuTe · 数据流"
type: dataflow
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: reference
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/dataflow
  - source-reading
updated: 2026-07-10
---
# Hopper与CuTe · 数据流

## 读者任务

这篇只看边界和对象形态：FA3/FA4 并不是换掉 attention 数据流，而是把同一个 `Q/K/V -> tile attention -> O/LSE` 映射到不同硬件和编译层。

## 三条路径的对象对照

| 维度 | FA2 | FA3 Hopper | FA4 CuTeDSL |
|------|-----|------------|-------------|
| 用户入口 | `flash_attn_func` | `flash_attn_3::fwd` schema | `flash_attn.cute.flash_attn_func` |
| 参数契约 | Python wrapper + C++ pybind | PyTorch dispatcher schema | Python interface validation |
| kernel 选择 | C++ macro/template switch | C++ arch/Split/PagedKV/PackGQA switch | Python arch branch + kernel object |
| 编译方式 | wheel 预编译实例 | Hopper beta 编译实例 | `cute.compile` + compile cache |
| 新风险 | ABI / build matrix | Hopper requirement / beta integration | first-call JIT latency / cache key |

## FA3 数据流：schema 到 CUTLASS kernel

```mermaid
flowchart LR
    S["flash_attn_3::fwd schema"]
    P["Flash_fwd_params"]
    D["run_mha_fwd<br/>arch split paged pack"]
    L["run_flash_fwd<br/>tile scheduler"]
    K["SM90 kernel<br/>TMA GMMA pipeline"]
    S --> P --> D --> L --> K
```

FA3 schema 先把 serving/training 组合显式化：

```cpp
// 来源：hopper/flash_api.cpp L1674-L1708
"Tensor? page_table = None,"
"Tensor? kv_batch_idx = None,"
"Tensor? leftpad_k = None,"
"Tensor? rotary_cos = None,"
"Tensor? rotary_sin = None,"
"Tensor? seqlens_rotary = None,"
"Tensor? q_descale = None,"
"Tensor? k_descale = None,"
"Tensor? v_descale = None,"
"Tensor? scheduler_metadata = None,"
"int num_splits = 0,"
"bool? pack_gqa = None,"
```

后续 launch 把 varlen、split、head、batch 等信息转成 scheduler 参数：

```cpp
// 来源：hopper/flash_fwd_launch_template.h L151-L172
typename flash::TileSchedulerArguments scheduler_args {
    num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
    params.h / params.h_k,
    params.seqlen_q,
    params.seqlen_k, params.d, params.dv, sizeof(Element),
    params.tile_count_semaphore, params.cu_seqlens_q, params.seqused_q,
    params.num_splits_dynamic_ptr,
    params.num_m_blocks_ptr,
    params.varlen_batch_idx_ptr,
    params.num_nheads_in_l2_ptr
};
typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
    mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
});
```

因此 FA3 的数据流不仅有张量，还有调度元数据。

## FA4 数据流：Python validation 到 compile cache

```mermaid
flowchart LR
    API["flash_attn.cute API"]
    V["validation<br/>arch head dtype"]
    CFG["feature state<br/>FP8 SplitKV GQA"]
    OBJ["kernel object"]
    KEY["compile_key"]
    CC["cute.compile"]
    RUN["cached callable"]
    API --> V --> CFG --> OBJ --> KEY --> CC --> RUN
```

FA4 先确定能力边界：

```python
# 来源：flash_attn/cute/interface.py L446-L516
arch = _get_device_arch() if _arch is None else _arch
assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
if arch // 10 not in [8, 12]:
    _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
is_fp8 = v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
if is_fp8 and requires_grad:
    raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
```

然后根据 arch 创建 kernel object：

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

最后把实际张量和编译期对象放入 cache：

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

## FP8 的数据边界

FA3 README 标出 FP8 forward；FA3 launch template 也把 FP8 输出类型改成 bf16。

```markdown
<!-- 来源：README.md L39-L45 -->
Currently released:
- FP16 / BF16 forward and backward, FP8 forward

Requirements: H100 / H800 GPU, CUDA >= 12.3.
```

```cpp
// 来源：hopper/flash_fwd_launch_template.h L201-L205
template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
```

FA4 则在 Python 层显式拒绝 FP8 backward，并限制 FP8 到 SM100。

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

## 复盘

- FA3 的数据流扩展点是 C++ schema 和 scheduler metadata。
- FA4 的数据流扩展点是 Python validation、kernel object 和 compile cache。
- FP8、SplitKV、paged KV 等特性都要同时看参数契约和硬件能力，不要只看 API 名称。
