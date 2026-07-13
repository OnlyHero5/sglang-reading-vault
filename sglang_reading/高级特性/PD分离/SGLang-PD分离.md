---
title: "PD分离"
type: map
framework: sglang
topic: "PD分离"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/map
  - source-reading
updated: 2026-07-11
---
# PD分离

读本专题不是为了记住 Mooncake、NIXL、Fake 这些 backend 名字，而是为了判断：Gateway 把一次生成拆成 Prefill、Decode 两个独立 HTTP 请求后，两套服务如何靠同一个 `bootstrap_room` 重新会合；谁预留接收位置，谁写 KV，谁证明 metadata 已落地，Decode 又如何跳过 prompt prefill 接回逐 token decode。

读完应能解决三类问题：

1. PD 部署后 P99 变差，是 Prefill 算力、Decode 槽位、KV 传输还是 metadata gate 卡住。
2. Decode 明明收到 KV，为何仍不能进入 running batch。
3. `bootstrap_room`、`pre_alloc_size`、decode radix cache、staging buffer 和 transfer backend 这些配置的真实边界是什么。
4. Gateway 整对 HTTP retry 与 Prefill Scheduler 内部 optimistic retry 为什么是两种完全不同的重试。

## 阅读路径

| 读者任务 | 先读 | 再读 |
|----------|------|------|
| 建立 PD 分离整体模型 | [[SGLang-PD分离-核心概念]] | [[SGLang-分布式]] |
| 跟一次端到端请求 | [[SGLang-PD分离-源码走读]] | [[SGLang-ScheduleBatch数据结构]] |
| 查对象和队列状态 | [[SGLang-PD分离-数据流]] | [[SGLang-KV-Cache]] |
| 线上排障和选型 | [[SGLang-PD分离-排障指南]] | [[SGLang-可观测性]] |
| 自测是否读懂 | [[SGLang-PD分离-学习检查]] | [[SGLang-RadixAttention]] |

## 心理模型

```mermaid
flowchart LR
  A[Gateway] -->|Prefill HTTP 请求| B[Prefill TokenizerManager]
  A -->|Decode HTTP 请求| C[Decode TokenizerManager]
  B --> D[Prefill Scheduler]
  C --> E[Decode Scheduler]
  E --> F[DecodePreallocQueue 创建 KVReceiver 并占 KV 槽]
  D --> G[PrefillBootstrapQueue 创建 KVSender]
  G --> H{bootstrap 已收敛?}
  H -->|稳态路径| I[extend forward 写 KV 和 metadata]
  H -->|optimistic 次数未耗尽| J[先做 speculative Prefill]
  J -->|握手仍未完成| K[释放本轮 KV 并 requeue]
  J -->|随后收敛| I
  I --> L[Transfer Backend]
  L --> M[DecodeTransferQueue poll]
  M --> N[metadata gate 和 all-reduce]
  N --> O[prepare_for_prebuilt]
  O --> P[RunningBatch decode]
```

把 PD 分离读成五本账：

| 账本 | 问题 | 源码入口 |
|------|------|----------|
| 角色账 | 当前进程是 unified、prefill 还是 decode | `DisaggregationMode`、`dispatch_event_loop` |
| 房间账 | 请求要把 KV 送到哪个 decode room | `bootstrap_host`、`bootstrap_port`、`bootstrap_room` |
| 队列账 | 请求在 Prefill/Decode 哪个等待区 | `PrefillBootstrapQueue`、`DecodePreallocQueue`、`DecodeTransferQueue` |
| 就绪账 | KV bytes、metadata、HiCache restore 是否都 ready | `KVPoll`、`_apply_metadata_gate`、`HiCacheRestoreGatedKVReceiver` |
| 执行账 | Decode 如何跳过 prefill forward 进入 running | `prepare_for_prebuilt`、`process_prebuilt`、`get_next_disagg_decode_batch_to_run` |

## 核心源码范围

| 文件 | 本专题关注点 |
|------|--------------|
| `python/sglang/srt/managers/io_struct.py` | 用户请求和 tokenized request 中的 bootstrap 字段 |
| `python/sglang/srt/managers/tokenizer_manager.py` | Fake backend 自动分配 room、初始化 bootstrap server |
| `python/sglang/srt/managers/scheduler.py` | 请求转 `Req`、按 `DisaggregationMode` 选择 event loop |
| `python/sglang/srt/managers/disagg_service.py` | Prefill 侧启动 bootstrap server |
| `python/sglang/srt/disaggregation/prefill.py` | Prefill bootstrap、forward 后发送 KV、inflight poll |
| `python/sglang/srt/disaggregation/decode.py` | Decode prealloc、transfer poll、metadata commit、prebuilt 入 running |
| `python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py` | `ForwardMode.PREBUILT` 的 batch 形态 |
| `python/sglang/srt/disaggregation/utils.py` | mode、metadata buffer、metadata gate、transfer backend |
| `python/sglang/srt/disaggregation/base/conn.py` | `KVPoll` 状态数值 |
| `python/sglang/srt/disaggregation/decode_hicache_mixin.py` | Decode 侧 HiCache prefix restore gate |
| `python/sglang/srt/arg_groups/pd_disaggregation_hook.py` | 启动参数校验与 decode extra slots 默认值 |
| `sgl-model-gateway/src/routers/http/pd_router.rs` | 选择 Prefill/Decode worker pair、生成 room、并行双发与整对 retry |

## 最小源码锚点

请求必须携带或被补齐 bootstrap 信息，才能跨过 Prefill 和 Decode 两个节点：

```python
# 来源：python/sglang/srt/managers/io_struct.py L239-L245
    # For disaggregated inference
    bootstrap_host: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_pair_key: Optional[Union[List[Optional[str]], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None
```

Prefill 服务自己的 TokenizerManager 启动 bootstrap server；Decode 服务自己的 Scheduler 创建 receiver 并连接它。两边不是共享一个 `TokenizerManager` 或 Python 请求对象，而是靠相同 room 和 transfer metadata 对齐：

```python
# 来源：python/sglang/srt/managers/disagg_service.py L14-L29
def start_disagg_service(
    server_args: ServerArgs,
):
    # Start kv bootstrap server on prefill
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)

    if disagg_mode == DisaggregationMode.PREFILL:
        # only start bootstrap server on prefill tm
        kv_bootstrap_server_class = get_kv_class(
            transfer_backend, KVClassType.BOOTSTRAP_SERVER
        )
        bootstrap_server = kv_bootstrap_server_class(
            host=server_args.host,
            port=server_args.disaggregation_bootstrap_port,
        )
```

## 判断标准

- 看到 prefill 端等待，先看 decode 是否已经 prealloc 和 handshake，不要只看 prefill GPU 利用率。
- 看到 Prefill GPU 已经做过计算，也不能立即断言 bootstrap 已完成；显式开启 optimistic prefill 时，Scheduler 可以在 `KVPoll.Bootstrapping` 阶段先算，若握手没有及时收敛则释放本轮 KV、重置请求并重新排队。
- 看到 decode 卡在 transfer，先看 `KVPoll`、metadata buffer 的 `bootstrap_room` 和 all-reduce，而不是直接怀疑 model forward。
- 看到短 prompt P99 变差，要把 PD 的固定成本纳入 TCO：bootstrap、metadata、RDMA、双池路由都不是免费。
- 看到 decode radix cache 相关问题，先看启动校验；它与 fake backend、speculative decoding、HiSparse 有明确互斥。
- 看到 retry 日志，先分层：Gateway retry 会换 pair、换 room、重放两侧 HTTP 请求；optimistic prefill retry 发生在 Prefill Scheduler 内部，保留同一请求关联并回收这一次 speculative forward 的 KV。

## 相邻专题

| 专题 | 关系 |
|------|------|
| [[SGLang-Speculative]] | PD prebuilt 后可能还要构造 speculative draft input |
| [[SGLang-分布式]] | PD 状态需要 TP/CP/DP rank 达成一致 |
| [[SGLang-RadixAttention]] | Decode radix cache 和 HiCache 依赖 prefix match |
| [[SGLang-KV-Cache]] | PD 本质是在跨节点移动 KV slot 内容 |
| [[SGLang-可观测性]] | P99、transfer latency、queue depth 需要 metrics 支撑 |
