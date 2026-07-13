---
title: "SGLang 复杂度热点"
type: reference
framework: sglang
topic: "总结复盘"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/reference
  - source-reading
updated: 2026-07-12
---
# SGLang 复杂度热点

## 你为什么要读

复杂度不是“文件很长”的同义词。SGLang 真正难改的地方，是一个对象同时被多种模式、进程、资源和时间状态解释。热点文件只是这些交叉约束汇聚的地点。

本文不再列函数名排行榜，而是给出六种复杂度来源、十个高风险交界面和一套可重复的阅读方法。

## 六种复杂度来源

| 类型 | 典型问题 | SGLang 例子 |
|---|---|---|
| 状态乘积 | 多个正交开关组合后，状态数成倍增长 | normal/overlap/PP × PD × spec × grammar |
| 所有权迁移 | 对象跨进程、跨 rank 或跨服务后谁负责释放和失败收口 | Tokenizer→Scheduler、PD KV、encoder embedding |
| 地址翻译 | 逻辑身份与物理位置不是同一对象 | request slot、KV loc、radix node、LoRA slot |
| 时间耦合 | 当前动作依赖前一批、异步 event 或未来 consumer | overlap result、CUDA Graph、IPC pool 回收 |
| 硬件特化 | 同一语义由不同 backend、layout、dtype 和 kernel 实现 | Attention、MoE、量化、collective |
| 失败恢复 | partial success 后如何清理、回滚或继续 | 权重更新、PD transfer、动态 LoRA、worker watchdog |

评价一个模块时应问“它同时承担了几种复杂度”，而不是只看圈复杂度或行数。

## 热点总览

| 交界面 | 为什么危险 | 首先跟踪的对象 | 深读入口 |
|---|---|---|---|
| Scheduler event loop | 多种执行拓扑各有独立 loop，仍共享请求和结果契约 | `Req`、`ScheduleBatch`、pending result | [[SGLang-Scheduler-源码走读]] |
| SchedulePolicy / admission | prefix match、token budget、chunk、priority 与 delay 同时影响准入 | waiting req、prefix indices、budget | [[SGLang-SchedulePolicy-源码走读]] |
| Radix / KV pool | 逻辑树会 split/evict，物理 page/slot另有生命周期 | `RadixKey`、node、KV loc | [[SGLang-RadixAttention-源码走读]]、[[SGLang-KV-Cache-数据流]] |
| ForwardBatch / ModelRunner | 可变逻辑 batch 被物化成 backend/graph 可消费 tensor | `ForwardBatch`、metadata、runner view | [[SGLang-ModelRunner-源码走读]] |
| Parallel state | 多维坐标与 group alias 决定 collective 范围 | global/local rank、group | [[SGLang-分布式-源码走读]] |
| Speculative | 算法并不共享一条固定 draft/verify 协议 | `spec_info`、accept result、KV commit | [[SGLang-Speculative-源码走读]] |
| PD disaggregation | bootstrap、prealloc、transfer、inflight 与输出形成分布式状态机 | metadata buffer、room、KV sender | [[SGLang-PD分离-源码走读]] |
| LoRA | adapter 身份、CPU cache、GPU slot 与 running batch 引用不同步 | LoRA ref/id/uid/slot | [[SGLang-LoRA-源码走读]] |
| 多模态 | prompt span、feature、IPC、hash 与 encoder ownership 跨层 | `MultimodalDataItem` | [[SGLang-多模态-源码走读]] |
| CheckpointEngine | target/draft 多阶段更新缺少事务回滚，cache/版本还要一致 | method、weight version、per-worker result | [[SGLang-CheckpointEngine-源码走读]] |

## 1. Scheduler：不是一个 loop，而是一族 loop

当前入口会根据 PD mode、PP、overlap、PDMux 与 MLX 选择不同循环：

```python
# 来源：python/sglang/srt/managers/scheduler.py L4168-L4193
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()
```

所以“从 `event_loop_normal` 理解 Scheduler”只适用于一个配置子集。修改前先算出最终 `ServerArgs`，确认真实 loop；再沿该 loop 跟踪 recv→schedule→run→result→output。

### 阅读陷阱

- 把 overlap 当成 normal loop 的简单异步版，忽略 pending batch/result 的跨迭代所有权；
- 只改普通 loop，漏掉 PP、PD 或 MLX parity；
- 在 `run_batch` 看到问题就下钻 kernel，实际根因可能是 batch 组装或前一拍结果尚未消费。

## 2. ForwardMode：prefill/decode 二分已经不够

```python
# 来源：python/sglang/srt/model_executor/forward_batch_info.py L78-L103
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

    # Used in speculative decoding: verify a batch in the target model.
    TARGET_VERIFY = auto()
    # Used in speculative decoding: extend a batch in the draft model.
    DRAFT_EXTEND_V2 = auto()

    # Used in disaggregated decode worker
    # Represent a batch of requests having their KV cache ready to start decoding
    PREBUILT = auto()

    # Split Prefill for PD multiplexing
    SPLIT_PREFILL = auto()

    # Used in dLLM
    DLLM_EXTEND = auto()
```

一个新 mode 可能影响 Scheduler 组批、position、KV metadata、attention backend、CUDA Graph、sampling、PP 与结果提交。不能只在枚举和一个 `if` 中“加支持”。

## 3. RadixCache：查找本身可能修改结构

```python
# 来源：python/sglang/srt/mem_cache/radix_cache.py L355-L390
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
          different sampling salt, cache version, or retrieval augmentation
          context) by supplying a distinct ``extra_key``.

        Args:
            params (MatchPrefixParams): Parameters containing the lookup key
                with a list of token ids and an optional ``extra_key`` namespace tag.
                If ``page_size > 1`` the length is internally truncated to a multiple
                of ``page_size`` before matching. Passing an empty key returns an
                empty result with the root as the last node.

        Returns:
            MatchResult: ``device_indices`` is a 1-D ``torch.int64`` tensor of
            the concatenated KV cache indices corresponding to the longest
            cached prefix (may be length 0).
            ``last_device_node`` and ``last_host_node`` (currently the same) are the tree node objects
            representing the terminal node of the matched prefix. This method
            may mutate internal structure by splitting an existing node if the
            match ends inside a stored segment.

        Internal updates:
            * Refreshes access metadata (timestamps) used by the
                configured eviction strategy.
            * If the lookup ends inside a stored segment the node is split once
                to expose a precise boundary; this structural refinement improves
                subsequent match efficiency and does not duplicate data.
```

这里同时存在 namespace、page 对齐、device/host view、访问元数据和 node split。读缓存 bug 时要同时画逻辑树和物理 KV，不要把 node 当作 tensor 本身。

## 4. ModelRunner：控制流、tensor view 与 backend 在此会合

ModelRunner 的难点不只是模型 forward 长，而是它消费 Scheduler 已物化的 `ForwardBatch`，再把同一批对象映射到 eager/graph、prefill/decode/spec、PP、DP padding、attention metadata 与具体 kernel。

建议分三层读：

1. **runner 选择层**：当前是 eager、decode graph、prefill graph 还是其他 runner；
2. **metadata 层**：谁创建、更新、借用 attention/KV/spec metadata；
3. **模型层**：最终 `model.forward` 如何读这些 view。

不要从 kernel 栈反推 Scheduler 原对象，除非先确认 view、padding 和 loc 翻译。

## 5. Speculative：共同的是结果契约，不是固定流水线

EAGLE、NGRAM、DFLASH 与插件算法并非都经历同一个 `eagle_sample`、同一种 tree layout 或固定“两次 GPU launch”。复杂度来自 Scheduler 需要接收共同结果，同时每种算法拥有不同候选、验收、KV commit 和 next-input 协议。

阅读时先问：

- 候选对象是什么；
- target 怎样验收；
- accept length/index 如何表达；
- 哪些 KV 可以提交；
- 拒绝后谁构造下一步；
- overlap、grammar、TP/platform 分支是否支持。

## 6. PD：队列名背后是完成协议

Prefill bootstrap/inflight 与 Decode prealloc/transfer/retracted queue 不是为了“排队而排队”。它们分别代表 metadata、buffer、KV transfer 和可执行状态的不同完成条件。

复杂点在于：

- metadata ready 不等于 KV ready；
- prealloc 成功不等于 transfer 成功；
- optimistic prefill、cached prefix chunk、PP/overlap 会改变时序；
- abort/timeout 必须清理 sender、buffer、queue 与 request mapping；
- gateway 的 HTTP 成功也不能替代内部 transfer 完成。

## 7. Parallel state：alias 比缩写更重要

TP、PP、DP、EP、CP、DCP 只告诉你维度名；真实 collective 使用哪个 group、是否有 CPU/Gloo coordination、是否有 custom backend、local rank 如何映射，取决于 `GroupCoordinator` 与 alias。

排查 hang 时至少记录：

```text
global rank / local rank
TP、PP、DP、EP、CP、DCP 坐标
调用的 group alias
collective backend
参与 rank 集合
进入 collective 的顺序和 tensor shape
```

只报 `tp=8` 无法定位错误 collective。

## 8. 动态资源：LoRA、多模态、权重更新

这三类看似外围功能，却共同引入“运行时可变资源”：

- LoRA：adapter identity→CPU cache→GPU slot→batch metadata；
- 多模态：媒体内容→feature/embedding→IPC→hash/pad→prefix identity；
- 权重更新：method→worker update→target/draft→cache flush→weight version。

它们的共同风险是 partial success。当前实现并不都提供事务回滚；文档和运维流程必须记录逐 worker/逐阶段结果，而不是只看一个总布尔值。

## 热点阅读六步法

对任意热点函数，按顺序写下：

1. **入口条件**：哪组最终配置和状态能到达？
2. **主对象**：函数读写哪些对象，谁拥有生命周期？
3. **不变量**：哪些身份、顺序、shape、资源计数必须保持？
4. **副作用**：分配、释放、缓存突变、消息、metric、event 有哪些？
5. **失败路径**：异常、timeout、partial success 后如何清理？
6. **验证**：静态证据与目标环境实验分别证明什么？

## 一个可复用的热点卡

```text
问题：
可达配置：
输入对象：
输出对象：
所有者变化：
地址翻译：
跨迭代/跨进程状态：
失败与清理：
观测信号：
最小反例：
```

## 复盘

SGLang 的复杂度主要来自“同一语义在不同运行模式下仍要保持一致”，而不是某几个工程师写了长文件。读热点时沿对象生命周期和不变量切片，远比从文件第一行线性读到最后一行有效。
