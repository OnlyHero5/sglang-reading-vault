---
title: "FlashAttention FA3 Hopper 演进"
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
# FlashAttention FA3 Hopper 演进

## 读者任务

这篇只回答 FA3 相对 FA2 的增量：为什么它要单独放在 `hopper/`，以及 Hopper 路径新增了哪些源码对象。

## 增量一：Hopper beta 与硬件边界

FA3 在 README 中被标注为 Hopper beta，要求 H100/H800 和 CUDA 12.3 以上，并发布 FP16/BF16 forward/backward 与 FP8 forward。

```markdown
<!-- 来源：README.md L30-L47 -->
## FlashAttention-3 beta release
FlashAttention-3 is optimized for Hopper GPUs (e.g. H100). 

This is a beta release for testing / benchmarking before we integrate that with
the rest of the repo.

Currently released:
- FP16 / BF16 forward and backward, FP8 forward

Requirements: H100 / H800 GPU, CUDA >= 12.3.
```

所以 FA3 的起点是硬件特化，不是 API 重命名。

## 增量二：schema 面向 serving/training 混合路径

`flash_attn_3::fwd` schema 同时暴露新 KV、paged KV、RoPE、descale、scheduler metadata、SplitKV 和 GQA packing。

```cpp
// 来源：hopper/flash_api.cpp L1674-L1708
"Tensor(k_new!)? k_new = None,"
"Tensor(v_new!)? v_new = None,"
"Tensor? page_table = None,"
"Tensor? kv_batch_idx = None,"
"Tensor? leftpad_k = None,"
"Tensor? rotary_cos = None,"
"Tensor? rotary_sin = None,"
"Tensor? q_descale = None,"
"Tensor? k_descale = None,"
"Tensor? v_descale = None,"
"Tensor? scheduler_metadata = None,"
"int num_splits = 0,"
"bool? pack_gqa = None,"
```

这说明 FA3 不是只处理普通 full attention；它把 serving 中的 cache append、paged layout、FP8 descale 和动态调度都纳入同一入口。

## 增量三：dispatch 维度更贴近 Hopper 特性

FA3 forward dispatch 先看 arch，再看 SplitKV、paged KV、PackGQA 和 softcap。

```cpp
// 来源：hopper/flash_api.cpp L367-L383
ARCH_SWITCH(params.arch, Arch, [&] {
    SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
        PAGEDKV_SWITCH(params.page_table && !params.pagedkv_tma, PagedKVNonTMA, [&] {
            PACKGQA_SWITCH(params.pack_gqa, PackGQA_, [&] {
                static constexpr bool PackGQA = PackGQA_ || Arch < 90 || PagedKVNonTMA || Split;
                SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
                    run_mha_fwd_constexpr<Arch, Split, PagedKVNonTMA, PackGQA, Has_softcap>(params, stream);
                });
            });
        });
    });
});
```

读这段时要关注组合爆炸：paged KV、SplitKV 和 GQA packing 都会改变 kernel 实例形态。

## 增量四：SM90 mainloop 使用 TMA/GMMA pipeline

SM90 kernel 类型显式暴露 TMA 与 producer/consumer 角色。

```cpp
// 来源：hopper/flash_fwd_kernel_sm90.h L45-L62
static constexpr bool Use_TMA_Q = CollectiveMainloop::Use_TMA_Q;
static constexpr bool Use_TMA_KV = CollectiveMainloop::Use_TMA_KV;
static constexpr bool Use_TMA_O = CollectiveEpilogue::Use_TMA_O;
static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
using ArchTag = typename CollectiveMainloop::ArchTag;
using ClusterShape = typename CollectiveMainloop::ClusterShape;
using MainloopArguments = typename CollectiveMainloop::Arguments;
using MainloopParams = typename CollectiveMainloop::Params;
```

```cpp
// 来源：hopper/flash_fwd_kernel_sm90.h L197-L223
if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
    CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
}
int warp_group_idx = cutlass::canonical_warp_group_idx();
PipelineParamsK pipeline_params_k;
pipeline_params_k.role = warp_group_idx == 0
    ? MainloopPipelineK::ThreadCategory::Producer
    : MainloopPipelineK::ThreadCategory::Consumer;
```

这就是 Hopper 路径相对 FA2 的硬件增量：load/compute/store 以 pipeline 角色组织，TMA descriptor 预取和 GMMA mainloop 成为核心。

## 增量五：调度元数据进入 kernel params

FA3 launch template 把 varlen、split、head、batch、L2 head 信息等组织进 scheduler args。

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
```

kernel 不只是算数学，还要接收运行时 shape 与 tile 调度信息。

## 增量六：FP8 forward 改变输出类型

FA3 launch template 中，FP8 输入对应 bf16 输出类型。

```cpp
// 来源：hopper/flash_fwd_launch_template.h L201-L205
template<int Arch, typename T, int kHeadDim, int kHeadDimV, bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
```

FP8 不是只改 dtype；它影响 descale、V layout、输出 dtype、kernel instantiation 和测试矩阵。

## 复盘

FA3 的增量可以压成一句话：把 FA2 的 IO-aware attention 映射到 Hopper 的 TMA/GMMA pipeline，并把 serving 所需的 paged KV、SplitKV、scheduler metadata、FP8 forward 等组合纳入 C++/CUDA 路径。
