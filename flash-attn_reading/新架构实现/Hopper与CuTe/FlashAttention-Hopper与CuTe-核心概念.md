---
title: "FlashAttention Hopper 与 CuTe 核心概念"
type: concept
framework: flash-attn
topic: "Hopper与CuTe"
learning_role: core
source_baseline: "002cce0"
tags:
  - framework/flash-attn
  - content/concept
  - source-reading
updated: 2026-07-12
---
# FlashAttention Hopper 与 CuTe 核心概念

## 读者任务

这篇建立五个可落到源码对象的概念：TMA、GMMA、warp specialization、persistent scheduler、JIT compile key。读完后你应该能回答：

- 哪些是 SM90 硬件/指令能力，哪些是 kernel 的并行组织策略？
- 为什么 Hopper 路径有时仍用 `cp.async`，不能看到 SM90 就断言“全 TMA”？
- 为什么一个 CTA 可以连续领取多个 tile，persistent 不等于“一个 kernel 永不退出”？
- 为什么 FA4 相同 batch/seqlen 仍可能产生不同编译项，而不同 batch/seqlen 又可能命中同一个项？
- README 的发布范围与当前源码接受范围不一致时，应该相信哪类证据回答哪个问题？

## 先拆开发布史与当前实现

README 的这段文字只证明 FA3 beta 发布时公开强调 Hopper、H100/H800、CUDA 12.3+ 与 FP8 forward。

```markdown
# 来源：README.md L39-L47
This is a beta release for testing / benchmarking before we integrate that with
the rest of the repo.

Currently released:
- FP16 / BF16 forward and backward, FP8 forward

Requirements: H100 / H800 GPU, CUDA >= 12.3.

We highly recommend CUDA 12.8 for best performance.
```

当前 `hopper/flash_api.cpp` 已继续演进：forward 入口接受 Ampere 或更新 GPU，按 arch 选择 SM8x/SM90 实现；Ampere/Ada 明确只接受 FP16/BF16，FP8 dispatch 使用 Arch 90 specialization，backward 只接受 FP16/BF16。正确口径是：

| 证据 | 回答的问题 |
|------|------------|
| README | 这次发布对用户宣称什么、推荐什么环境 |
| 当前 interface/schema/dispatch | 当前基线实际接受哪些输入并可能进入哪些实现 |
| tests 与固定环境 profile | 某个组合是否正确、是否更快、首次编译和稳态成本是多少 |

“FA3 是 Hopper beta”是发布定位；“当前 `hopper/` 源码只有 H100 路径”则是错误的当前实现结论。

## 概念一：TMA 是条件搬运路径，不是 Hopper 标签

TMA 可以把多维 global-memory tensor tile 异步搬入 shared memory，并用 transaction barrier 与消费者协调。它解决的是数据搬运与同步，不负责 softmax，也不等于 GMMA。

当前 SM90 mainloop 把是否使用 TMA 写成编译期条件：PackGQA 时 Q 不走 TMA；paged KV 的 page/tile 组合触发 non-TMA 时，K/V 改走 `cp.async` 风格路径。

```cpp
// 来源：hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp L48-L66
    static constexpr bool Varlen = Varlen_;
    static constexpr bool PagedKVNonTMA = PagedKVNonTMA_;
    static constexpr bool AppendKV = AppendKV_;
    static constexpr bool HasQv = HasQv_;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Split = Split_;
    static constexpr bool V_colmajor = V_colmajor_;
    static constexpr bool Transpose_V = Is_FP8 && !V_colmajor;
    static constexpr bool Use_TMA_Q = !PackGQA;
    static constexpr bool Use_TMA_KV = !PagedKVNonTMA;
    static_assert(Use_TMA_KV || CUTE_STATIC_V(size(ClusterShape{})) == 1, "If not using TMA for KV, ClusterShape must be 1");
    static_assert(Use_TMA_KV || !V_colmajor, "If not using TMA for KV, V_colmajor is not supported");
    static constexpr bool SameHeadDim = get<2>(TileShape_MNK{}) == kHeadDimV;
    static constexpr bool LargeHeadDimV = kHeadDimV > 256;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 && !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
    static constexpr cute::GMMA::Major TmaMajorV = !V_colmajor ? GMMA::Major::MN : GMMA::Major::K;
```

失效边界：`Use_TMA_Q/KV` 是这个 SM90 mainloop 的选择，不应外推成 FA4 所有架构或 backward 所有阶段的统一规则。

## 概念二：GMMA 是 warp-group 矩阵乘，不是调度器

GMMA/WGMMA 描述 Hopper warp-group 参与的矩阵乘能力。源码中的 `ss_op_selector`、`rs_op_selector` 决定操作数来自 shared memory 还是 register/shared 组合；QK、PV、可选 QV 可以选择不同 atom。

把 attention 看成流水线时：

```text
TMA / cp.async        负责把 tile 送到片上工作集
GMMA                  负责 QK、PV 等矩阵乘
online softmax        负责跨 K block 的数值合并
tile scheduler        负责把工作 tile 分给 CTA
epilogue              负责归一化结果、LSE 与输出落盘
```

这些机制相互配合但不互为同义词。看到 `GMMA::Major::K/MN` 主要是在读布局/操作数方向，不能直接推导 kernel 吞吐。

## 概念三：warp specialization 是角色分工

SM90 kernel 把 warp group 0 设为 producer，负责加载与推进写侧 pipeline，并释放一部分寄存器；其他 warp groups 是 consumer，获得更多寄存器执行 MMA 与 epilogue。

```cpp
// 来源：hopper/flash_fwd_kernel_sm90.h L306-L323
        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            // The pipelines for AppendKV and main attention are different, since e.g. main attention
            // might use cp.async to load KV (if PagedKVNonTMA) while AppendKV always uses TMA to load
            // KV_new. Since the pipeline states are different, we have to manually sync to make
            // sure the two pipelines don't race when accessing smem_k and smem_v.
            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            PipelineState smem_pipe_write_new = cutlass::make_producer_start_state<MainloopPipelineKVNew>();
            int work_idx = 0;
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
            if constexpr (SingleProducerWarp) {
                if (warp_idx_in_warpgroup != 0) { return; }
            }
            if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }
```

```cpp
// 来源：hopper/flash_fwd_kernel_sm90.h L360-L372
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            // Initialize matmul objects.
            TiledMmaPV tiled_mma_pv;

            PipelineState smem_pipe_read;
            PipelineState smem_pipe_read_new;
            // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
            // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

            scheduler.init_consumer();
            mainloop.mma_init();
```

这不是“producer 做完全部加载后 consumer 才开始”的串行模型。pipeline、barrier 与 stage state 的目的正是让加载、MMA、softmax、下一 tile 预取按约束重叠。具体重叠程度由 tile、stage、`IntraWGOverlap`、paged/append/FP8 等编译期组合决定。

## 概念四：persistent scheduler 是 CTA 反复领活

普通静态 grid 可以把一个 CTA 固定到一个 tile。persistent scheduler 则让有限 CTA 在设备上循环从 scheduler 领取后续 tile，以控制并发、改善不规则 workload 的工作分配或减少大量空退 CTA。

```cpp
// 来源：hopper/flash_fwd_launch_template.h L75-L81
    static constexpr bool UsePersistentScheduler = Arch >= 90 ? !(Split && !Varlen) : ((Is_causal && !Varlen) || (Varlen && Split));
    using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;
    using AttnKernel = std::conditional_t<
        Arch >= 90,
        flash::enable_sm90<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
        flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
    >;
```

当前选择是条件化的：例如 SM90 上非 varlen 的 Split 会避开 persistent，而 varlen Split 可以保留。persistent 的收益也不是定律，必须结合 tile 数、SM 数、序列分布和 profile 验证。

## 概念五：FA4 JIT key 描述代码生成语义

FA4 Python interface 先做 arch、dtype、head_dim 和特性校验，再选择不同 kernel object，最后以 compile key 查 cache；miss 时调用 `cute.compile`，hit 时复用 callable 并传入 runtime tensor。

需要区分三类量：

| 类型 | 例子 | 对 cache 的含义 |
|------|------|-----------------|
| 编译期语义 | dtype、head_dim、arch、causal、tile、split、paged non-TMA、callable hash | 通常进入 key，变化可能触发新编译 |
| 存在性/布局语义 | 是否 varlen、是否有 page table/LSE/descale/稀疏元数据 | 可能改变函数签名或生成代码，因此进入 key |
| 运行时数据 | batch、seqlen、指针、具体 token 长度和数值 | 可能由动态 tensor 传入，不必逐项进入 key |

所以：同样的 batch/seqlen 可能因 mask callable 或 feature presence 不同而 miss；不同 batch/seqlen 也可能在编译契约相同的情况下 hit。cache hit 只证明免去这次重新编译，不证明两次运行成本相同。

FA4 还区分：

- `FLASH_ATTENTION_ARCH`：选择哪条 kernel 路径。
- `CUTE_DSL_ARCH`：编译目标。
- fake mode：用符号/假 tensor 编译或推导元数据，不执行真实 GPU 数值。
- first-call latency：cache miss 的编译与初始化成本。
- steady-state latency：已命中 callable 后的运行成本。

## 当前能力矩阵

| 路径 | 当前源码边界 | 易错外推 |
|------|--------------|----------|
| FA3 forward | Ampere 或更新；Ampere/Ada 只接受 FP16/BF16，FP8 dispatch 使用 Arch 90 specialization；可含 varlen、paged/append KV、SplitKV 等条件分支 | “目录叫 hopper，所以 Ampere 一定拒绝” |
| FA3 backward | Ampere 或更新，FP16/BF16；deterministic、head_dim 等仍有约束 | “README 只写 beta，所以没有 backward” |
| FA4 forward | arch 8/9/10/11/12 各自 kernel object；特性支持并不对称；FP8 当前只在 SM100 | “FA4 README 写 Hopper/Blackwell，所以当前接口没有 SM80” |
| FA4 backward | 当前入口接受 arch 9/10/11/12，输入要求 FP16/BF16；不同 arch 对 deterministic、score/mask、稀疏等限制不同 | “forward 能跑的组合都能 backward” |
| FA4 公开导出 | `flash_attn_func`、`flash_attn_varlen_func` | “内部 combine/MLA helper 都是稳定公开 API” |

## 运行验证

无独立包和匹配 GPU 时，先执行静态检查：

```powershell
rg -n 'Use_TMA_Q|Use_TMA_KV|GMMA::(ss|rs)_op_selector' flash-attn/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
rg -n 'warp_group_idx == 0|warpgroup_reg_(dealloc|alloc)|UsePersistentScheduler' flash-attn/flash-attention/hopper/flash_fwd_kernel_sm90.h flash-attn/flash-attention/hopper/flash_fwd_launch_template.h
rg -n 'compile_key =|get_jit_cache|cute\.compile|FLASH_ATTENTION_ARCH|CUTE_DSL_ARCH' flash-attn/flash-attention/flash_attn/cute/interface.py
```

预期依次定位条件 TMA/GMMA、producer/consumer 与 scheduler 选择、FA4 的 key/cache/compile/arch 控制。静态命中不证明数值与性能。

动态实验应至少拆成四组：首次调用、相同 key 再调用、只改 runtime seqlen、改一个 key 字段；同时记录 cache 日志、kernel 名称与耗时。没有匹配环境时，不写任何固定性能阈值。

## 复盘

把五个概念压成一句话：TMA 搬 tile，GMMA 做矩阵乘，warp specialization 把搬运与计算分给不同 warp group，persistent scheduler 让 CTA 循环领取 tile，FA4 compile key 决定哪种代码形态需要单独编译。它们改变实现组织，不改变 FlashAttention 的 online-softmax 数学不变量；是否更快只能由固定环境和 workload 证明。
