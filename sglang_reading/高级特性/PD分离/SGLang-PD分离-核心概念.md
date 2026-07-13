---
title: "PD分离 · 核心概念"
type: concept
framework: sglang
topic: "PD分离"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-11
---
# PD分离 · 核心概念

PD 分离的难点不是“Prefill 和 Decode 分开跑”这句话，而是分开之后仍要保证一个请求的 KV、metadata、room、prefix cache 和 running batch 状态都对齐。只要其中一条账错了，Decode 可能拿空 KV 生成、不同 TP rank 状态不一致，或者 Prefill 长时间堵在 bootstrap。

读完这篇应能回答：

1. `DisaggregationMode` 只是角色标记，还是会改变 Scheduler event loop。
2. `bootstrap_room` 为什么是请求跨节点的主键。
3. Prefill 三队列和 Decode 四队列各自卡在哪里。
4. `KVPoll.Success` 为什么还不一定能进入 decode。
5. Decode 的 `Prebuilt` batch 为什么不是普通 prefill。
6. 为什么“Decode 先准备、Prefill 后计算”只是默认稳态主线，而不是所有配置下的绝对时序。

## 先建立模型

把 PD 分离想成“两个独立工厂按同一订单号交接半成品”：

- Gateway 每次 attempt 选择一对 Prefill/Decode worker，生成新的 `bootstrap_room`，并行发出两个独立 HTTP 请求。
- 两侧各自经过本服务的 TokenizerManager 和 Scheduler；它们不共享 Python 请求对象，只共享 room、prompt 和协议字段。
- 默认稳态路径中，Decode 先在这个房间占 KV 槽、创建 receiver，Prefill 在 bootstrap 收敛后进入 forward。
- 若显式设置 `optimistic_prefill_retries > 0`，Prefill 可在 sender 仍为 `Bootstrapping` 时先做一轮 speculative forward；握手未及时完成时，这轮 KV 会被释放，请求 reset 后重新排队。
- Prefill 做长 prompt forward，把 KV 和首个输出 token 的 metadata 写进 transfer 通道。
- Decode 轮询 receiver，但还要确认 metadata 里的 room 已落地。
- Decode 构造 `ForwardMode.PREBUILT` batch，把请求并入 running batch 继续逐 token decode。

```mermaid
stateDiagram-v2
  [*] --> Routed: 请求带 bootstrap_*
  Routed --> DecodePrealloc: Decode 创建 KVReceiver
  Routed --> PrefillBootstrap: Prefill 创建 KVSender
  DecodePrealloc --> DecodeTransfer: handshake 和预分配完成
  PrefillBootstrap --> PrefillWaiting: bootstrap 完成
  PrefillBootstrap --> OptimisticPrefill: retries 未耗尽
  OptimisticPrefill --> PrefillWaiting: bootstrap 随后完成
  OptimisticPrefill --> PrefillBootstrap: 未收敛，释放 KV 并 requeue
  PrefillWaiting --> PrefillInflight: extend forward 后发送 KV
  PrefillInflight --> TransferDone: sender poll Success
  DecodeTransfer --> MetadataGate: receiver poll Success
  MetadataGate --> Prebuilt: bootstrap_room metadata ready
  Prebuilt --> RunningDecode: merge running batch
```

这个模型有五本账。

| 账本 | 核心对象 | 破坏后现象 |
|------|----------|------------|
| 角色账 | `DisaggregationMode`、event loop | 节点跑错循环，prefill/decode 行为混杂 |
| 房间账 | `bootstrap_host/port/room` | sender/receiver 对不上，或 DP rank 路由错 |
| 队列账 | Prefill 三队列、Decode 四队列 | P99 抖动、bootstrap 或 transfer 积压 |
| 就绪账 | `KVPoll`、metadata gate、HiCache restore gate | 成功过早上报，Decode 读空 KV 或旧 metadata |
| 执行账 | `ForwardMode.PREBUILT`、`process_prebuilt` | Decode 误跑 prefill，或首 token/grammar/spec 状态错 |

## 角色账：mode 会改 Scheduler 主循环

PD 不是在普通 loop 旁边加一个传输线程。Scheduler 根据 `DisaggregationMode` 直接选择不同 event loop。

```python
# 来源：python/sglang/srt/disaggregation/utils.py L60-L71
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"
```

```python
# 来源：python/sglang/srt/managers/scheduler.py L4164-L4192
def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
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

关键判断：`NULL` 不是“PD 但不分离”，而是 unified。`PREFILL` 和 `DECODE` 会进入不同 loop，意味着队列、batch 构造、结果处理都变了。

## 房间账：`bootstrap_room` 是跨节点对齐键

一个请求从 HTTP 或 OpenAI handler 进来时，要带上 decode 侧房间信息；tokenize 后这个字段继续进入 `TokenizedGenerateReqInput`。

```python
# 来源：python/sglang/srt/managers/io_struct.py L239-L245
    # For disaggregated inference
    bootstrap_host: Optional[Union[List[Optional[str]], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_pair_key: Optional[Union[List[Optional[str]], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None
```

batch 请求还会把单个 room 展开成连续 room，避免 parallel samples 复用同一房间：

```python
# 来源：python/sglang/srt/managers/io_struct.py L659-L665
        # Normalize bootstrap_room
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num
```

`bootstrap_room` 还会影响 DP 路由。Prefill 的 `load_balance_method=auto` 在 PD Prefill 模式下会归一化为 `follow_bootstrap_room`，使 room 稳定映射到对应 rank。

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L628-L637
    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        sock_send(self.workers[target_rank], req)
```

因此直连 prefill/decode 调试时，如果 room 缺失或复用，问题可能不是 transfer backend，而是路由主键已经错了。

Decode 侧还必须知道“这个 room 实际落在哪个 Prefill DP rank”。这里存在三条路径：

1. Gateway 已注入 `disagg_prefill_dp_rank`：直接使用，最快。
2. 已缓存 Prefill parallel info，且 Prefill 声明 `follow_bootstrap_room`：本地计算 `room % dp_size`。
3. 信息未知或策略不允许本地推导：请求先进入 `pending_reqs`，异步获取 Prefill info，必要时按 room 批量查询 rank，解析完成后才初始化 receiver。

所以 Decode prealloc 堵住不一定是显存不足，也可能是在等 Prefill parallel info；这是房间账和并行账的交界。

## 队列账：Prefill 和 Decode 卡点不同

Prefill 侧名义上分 bootstrap、waiting、inflight 三段；默认配置 `optimistic_prefill_retries=0` 时，GPU forward 确实发生在 bootstrap 完成、请求进入 waiting 之后，transfer 完成发生在 inflight。

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L1-L18
"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""
```

这段文件头注释描述的是稳态主线。实际实现另有 optimistic 分支：`pop_bootstrapped` 在 poll 仍为 `Bootstrapping`、retry 次数未耗尽时，也可先为请求分配 metadata slot 并放进 waiting。forward 后 `handle_pending_bootstrap` 再检查握手；仍未完成则 `optimistic_release_and_requeue` 回收 KV、清空首 token/临时状态、递增 retry 并重排。

该优化默认关闭，并且 PP、hierarchical cache、Mamba radix cache 会在参数校验中把它禁用。它用“可能重算一次 Prefill”换取隐藏 handshake 延迟，适合讨论尾延迟时单独建账。

```python
# 来源：python/sglang/srt/server_args.py L2373-L2377
    optimistic_prefill_retries: A[
        int,
        "Number of optimistic prefill retries that will skip the bootstrap wait. ",
    ] = 0
```

默认值为 0 是重要边界：不显式开启时，文件头描述的 bootstrap → waiting → forward 顺序就是实际主线。

Decode 侧四段：prealloc、transfer、waiting、running。默认主线的第一件事不是 decode，而是先创建 receiver 并预分配 KV；在此之前还可能短暂停在 `pending_reqs` 等待 Prefill DP rank 解析。

```python
# 来源：python/sglang/srt/disaggregation/decode.py L1-L19
"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""
```

这解释了为什么 PD 的 P99 不能只看 GPU 利用率。一个请求可能还没进 GPU，就已经卡在 handshake、metadata buffer、prealloc KV slot 或 receiver poll。

## 就绪账：`Success` 只是候选成功，还要过 metadata gate

`KVPoll` 的数值顺序是故意设计的：`Failed=0`，`Success=4`，中间是未完成状态。跨 rank 用 MIN 时，任何 rank 未 ready 都能把全局状态压回未 ready。

```python
# 来源：python/sglang/srt/disaggregation/base/conn.py L79-L84
class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4
```

但 receiver poll 成功还不够。Decode 还要检查 metadata buffer 里的 `bootstrap_room` 是否已经写入，否则 Success 会被降级回 Transferring。

```python
# 来源：python/sglang/srt/disaggregation/utils.py L103-L118
def _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args) -> None:
    """Downgrade Success → Transferring for requests whose metadata hasn't landed.

    Mutates `polls` in-place. Called before all-reduce so that MIN across TP
    ranks naturally prevents any rank from committing before all ranks are ready.
    """
    for i, poll_val in enumerate(polls):
        if poll_val == int(KVPoll.Success):
            decode_req = decode_reqs[i]
            if _is_fake_transfer(decode_req.req, server_args):
                continue
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)
```

这段是 PD 正确性的核心之一：KV bytes 和 metadata 是两个 ready 条件。Fake backend 为测试跳过 gate，但真实 backend 不能跳。

## 执行账：Decode 走 `PREBUILT`，不是重跑 Prefill

Decode 收到 transfer 后，会构造 prebuilt batch。这个 batch 把 forward mode 改成 `PREBUILT`，填好 `input_ids`、`seq_lens`、`out_cache_loc`、sampling info 等执行 metadata。

```python
# 来源：python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py L25-L42
    def prepare_for_prebuilt(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        Adapted from .prepare_for_extend().
        """

        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_range.length for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)
```

随后 `process_prebuilt` 把 prefill 侧传来的最后一个 token 作为下一轮 decode 的起点；如果启用投机，还会构造 disagg draft input。

```python
# 来源：python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py L139-L157
        last_tokens_tensor = torch.tensor(
            last_tokens, dtype=torch.int64, device=self.device
        )

        spec_info = self.spec_algorithm.build_disagg_draft_input(
            self,
            server_args,
            last_tokens_tensor,
            future_map,
        )
        if spec_info is not None:
            self.spec_info = spec_info
        else:
            # Non-spec: stash last token into the relay so the first DECODE's
            # resolve_forward_inputs gathers it like any other decode iter.
            future_map.stash(
                self.req_pool_indices, RelayPayload(bonus_tokens=last_tokens_tensor)
            )
            self.input_ids = None
```

因此 PD 和投机解码并不是完全独立：prebuilt 阶段会把最后 token 交给普通 decode 或 speculative draft 的下一步。

## 配置账：decode 专用开关有互斥关系

Decode radix cache 能减少传输，但不是任意组合都能开。启动参数 hook 会在 decode mode 下做互斥校验，并设置 decode 侧 radix cache 行为和 extra slots 默认值。

```python
# 来源：python/sglang/srt/arg_groups/pd_disaggregation_hook.py L29-L70
    if server_args.disaggregation_mode == "decode":
        if server_args.disaggregation_decode_enable_radix_cache:
            if server_args.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if server_args.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            if server_args.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {server_args.speculative_algorithm})"
                )
            if server_args.enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            server_args.disable_radix_cache = False
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            server_args.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")

        # Default the number of *extra* decode req_to_token slots reserved for
        # in-transfer (being-received-from-prefill) requests, on top of the
        # max_running_requests-derived pool. Large batches get none; small
        # per-worker batches reserve 2x the batch as cheap overlap headroom.
        if server_args.disaggregation_decode_extra_slots is None:
            extra_slots = 0
            if server_args.max_running_requests is not None:
                per_worker = server_args.max_running_requests // max(
                    1, server_args.dp_size
                )
                if per_worker <= 32:
                    extra_slots = per_worker * 2
            server_args.disaggregation_decode_extra_slots = extra_slots
```

读配置时要从“能不能开”升级为“开了会改变哪本账”：radix cache 改 prefix/HiCache 账，extra slots 改 prealloc 容量账，staging buffer 改就绪账。

## 读者抓手

首次阅读时，按这句话复述：

`bootstrap_room` 把两个独立 HTTP 请求重新对齐；两侧各自经过 TokenizerManager/Scheduler。默认主线是 Decode 先占 KV 槽并等 receiver，Prefill 在 bootstrap 后做 extend；optimistic 分支允许 Prefill 提前算，但未收敛就必须释放 KV 并重试。Decode 还可能先解析 Prefill DP rank，再通过 receiver poll、metadata gate、all-reduce、HiCache/staging gate 判断 ready；ready 后构造 `PREBUILT` batch 并入 running decode。

排障时从症状反推账本：

| 症状 | 先查账本 | 第一入口 |
|------|----------|----------|
| prefill bootstrap 堵住 | 房间账、Decode prealloc | `DecodePreallocQueue`、`PrefillBootstrapQueue.create_sender` |
| transfer Success 但 decode 不动 | 就绪账 | `_apply_metadata_gate`、`_poll_with_metadata_gate` |
| 多 rank 状态不一致 | 就绪账 | `poll_and_all_reduce`、`KVPoll` 数值顺序 |
| PD 比 unified 慢 | 队列账、TCO | Prealloc/Transfer/Inflight 队列深度 |
| decode radix cache 开不起来 | 配置账 | `handle_pd_disaggregation` |
| speculative + PD 行为异常 | 执行账 | `process_prebuilt`、`build_disagg_draft_input` |
