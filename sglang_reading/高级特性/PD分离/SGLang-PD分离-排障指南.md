---
title: "PD分离 · 排障指南"
type: troubleshooting
framework: sglang
topic: "PD分离"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# PD分离 · 排障指南

## 你为什么要读

这篇不是概念问答，而是排障入口。PD 分离的问题通常不会表现成“某个函数错了”，而是表现成：Prefill 不动、Decode 不动、transfer success 但没有 token、某些 rank hang、开关组合启动失败。

排障时先判断请求卡在哪本账：

| 症状 | 优先查 | 直接入口 |
|------|--------|----------|
| 请求一进来就 abort | 房间账 | `Scheduler.handle_generate_request` |
| 直连 prefill 报 room 缺失 | 路由账 | `follow_bootstrap_room_scheduler` |
| Prefill bootstrap 堵住 | Decode prealloc | `DecodePreallocQueue._create_receiver_and_enqueue` |
| Decode 收到请求却迟迟不 prealloc | Prefill DP rank 解析 | `_ensure_prefill_info`、`_resolve_pending_reqs` |
| Prefill GPU 有计算但没有 KV 发出 | optimistic retry | `handle_pending_bootstrap`、`optimistic_release_and_requeue` |
| Prefill forward 完成但客户端没结果 | Prefill inflight | `process_disagg_prefill_inflight_queue` |
| receiver Success 但 Decode 不进 waiting | metadata gate | `_apply_metadata_gate` |
| Decode abort metadata mismatch | metadata slot 生命周期 | `_commit_transfer_to_req` |
| 多 rank 状态不一致或 collective hang | 并行共识 | `poll_and_all_reduce` |
| 开 decode radix cache 失败 | 配置互斥 | `handle_pd_disaggregation` |

## Q1：为什么真实 PD 请求不能缺 `bootstrap_room`

因为真实 backend 下 room 是 Prefill/Decode 对齐的主键。Scheduler 会在 PD mode 下拒绝没有 room 的请求；fake backend 是测试例外。

```python
# 来源：python/sglang/srt/managers/scheduler.py L2090-L2105
            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    recv_req.time_stats.trace_ctx.abort(
                        abort_info={"reason": error_msg}
                    )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
```

Prefill DP controller 也假设 room 已经存在，并用它决定目标 worker：

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

处理建议：

- 生产流量必须经过会分配 `bootstrap_room` 的 router 或 gateway。
- 直连调试时手动提供 `bootstrap_host`、`bootstrap_port`、`bootstrap_room`。
- 如果只想本地验证状态机，用 fake backend，但不要把 fake 的 room 自动分配行为当作生产语义。

## Q2：Prefill bootstrap 卡住时，为什么默认先查 Decode

稳态主线中，Prefill 侧只有在 Decode 已经建立 receiver 并预分配 metadata/KV 后，sender bootstrap 才能收敛。Prefill 自己会启动 bootstrap server，但接收端准备好与否在 Decode。

Prefill 进程才启动 bootstrap server：

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

Decode 侧会为同一个 room 创建 receiver：

```python
# 来源：python/sglang/srt/disaggregation/decode.py L541-L557
    def _create_receiver_and_enqueue(self, req: Req) -> DecodeRequest:
        backend = (
            TransferBackend.FAKE
            if _is_fake_transfer(req, self.scheduler.server_args)
            else self.transfer_backend
        )
        kv_receiver_class = get_kv_class(backend, KVClassType.RECEIVER)

        kv_receiver = kv_receiver_class(
            mgr=self.kv_manager,
            bootstrap_addr=_bootstrap_addr(req),
            bootstrap_room=req.bootstrap_room,
        )

        decode_req = DecodeRequest(req=req, kv_receiver=kv_receiver)
        self.queue.append(decode_req)
        return decode_req
```

Prefill sender 创建后只标记 `pending_bootstrap`，等待后续 poll：

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L230-L254
    def create_sender(self, req: Req, num_kv_heads: int) -> bool:
        """Create a KV sender for the request without enqueuing it.
        Returns False if the request exceeds KV capacity."""
        if self._check_if_req_exceed_kv_capacity(req):
            return False

        backend = (
            TransferBackend.FAKE
            if req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            else self.transfer_backend
        )
        kv_sender_class = get_kv_class(backend, KVClassType.SENDER)

        dest_tp_ranks = [self.tp_rank]

        req.disagg_kv_sender = kv_sender_class(
            mgr=self.kv_manager,
            bootstrap_addr=f"{req.bootstrap_host}:{self.bootstrap_port}",
            bootstrap_room=req.bootstrap_room,
            dest_tp_ranks=dest_tp_ranks,
            pp_rank=self.pp_rank,
        )
        self._process_req(req)
        req.pending_bootstrap = True
        return True
```

排查顺序：

1. Decode 是否收到同 rid、同 room 的请求。
2. 请求是否还在 `pending_reqs` 等 Prefill parallel info；`disagg_prefill_dp_rank`、`follow_bootstrap_room` 和 query 路径哪条生效。
3. `DecodePreallocQueue._create_receiver_and_enqueue` 是否执行。
4. `decode_req.kv_receiver.send_metadata` 是否发出 page indices 和 metadata index。
5. Prefill `PrefillBootstrapQueue.finalize_bootstrap` 是否拿到 `decode_prefix_len`。

例外是显式开启 optimistic prefill：此时 Prefill GPU 可能在 bootstrap 未完成时先跑一轮。看到 GPU 活跃不能证明 handshake 已收敛；若 `pending_bootstrap` 仍为真，forward 后可能执行 `optimistic_release_and_requeue`，释放这轮 KV 并重排。

## Q2.1：为什么 Decode 收到请求后可能还不创建 receiver

非 fake backend 需要先确定对端 Prefill DP rank。Decode 的快路径依次是：使用请求携带的 `disagg_prefill_dp_rank`；使用缓存的 Prefill info，在 `follow_bootstrap_room` 下计算 `room % dp_size`。两者都不成立时，请求进入 `pending_reqs`，按 bootstrap 地址异步获取 parallel info，并可能批量 query room 对应 rank。

操作与预期：

1. 观察 `pending_reqs` 长度和 bootstrap 地址分组；有积压说明还没到 KV 内存预分配阶段。
2. 检查 Prefill info ensure 的 retry/error；成功后该地址应进入 `ready_addrs`。
3. 检查 room query 是否返回当前 room；成功后应调用 `kv_receiver.init(prefill_dp_rank)`，请求从 pending 转入正常 handshake。

## Q2.2：Prefill GPU 已计算却没有发送 KV，是否说明传输坏了

不一定。若 `optimistic_prefill_retries > 0`，`KVPoll.Bootstrapping` 请求可提前进入 waiting 做 speculative Prefill。forward 结束时：

- poll 已进入 `WaitingForInput`：finalize sender，继续 metadata/KV 发送。
- poll 仍为 `Bootstrapping`：释放这轮 KV、reset request、增加 `prefill_retry_count` 并 requeue。
- retry 耗尽：回到等待真实 bootstrap 收敛的稳态路径。

该开关默认是 `0`；PP、HiCache、Mamba radix cache 会把它禁用。排查时把“重复 Prefill 计算/利用率上升”与“Gateway 换 pair、换 room 的 HTTP retry”分开统计。

## Q3：为什么 `KVPoll.Success` 还不能直接进 Decode

`KVPoll.Success` 只代表底层 receiver/sender 认为传输完成。PD 还要确认 metadata 已落地，并且所有 rank 都看到同样的保守状态。

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

如果 gate 通过后 commit 时仍发现 room 为 0 或 room mismatch，源码会把它当成确定性错误处理：

```python
# 来源：python/sglang/srt/disaggregation/decode.py L1526-L1573
        # Validate bootstrap_room to detect context corruption
        actual_room = output_bootstrap_room[0].item()
        expected_room = (
            decode_req.req.bootstrap_room
            if decode_req.req.bootstrap_room is not None
            else 0
        )

        if _is_fake_transfer(decode_req.req, self.scheduler.server_args):
            pass
        elif actual_room == 0:
            # Should never happen: _poll_with_metadata_gate already confirmed
            # readiness on all TP ranks. Abort deterministically to avoid
            # cross-rank queue divergence.
            logger.error(
                f"Metadata unexpectedly not ready after readiness gate: "
                f"request {decode_req.req.rid}, bootstrap_room={expected_room}, "
                f"metadata_buffer_index={idx}"
            )
            prepare_abort(
                decode_req.req,
                "Metadata unexpectedly not ready after readiness gate "
                "(bootstrap_room=0)",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return
        elif actual_room != expected_room:
            # Real corruption detected (mismatch)
            # Abort the request and remove from the queue
            error_msg = (
                f"Context corruption detected: Request {decode_req.req.rid} "
                f"(bootstrap_room={expected_room}) received metadata from "
                f"bootstrap_room={actual_room}. "
                f"Metadata buffer index: {idx}. "
                f"This indicates metadata buffer index collision."
            )
            logger.error(error_msg)
            prepare_abort(
                decode_req.req,
                "Metadata corruption detected - bootstrap_room mismatch",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            decode_req.kv_receiver.clear()
            decode_req.kv_receiver = None
            return
```

处理建议：

- 如果 `actual_room == 0`，查 Prefill `MetadataBuffers.set_buf` 是否执行，以及 metadata slot 是否正确传给 sender。
- 如果 `actual_room != expected_room`，优先怀疑 metadata buffer index 复用、room 复用或跨请求污染。
- 不要在自定义 backend 里绕过 metadata gate，除非它严格等价于 fake backend。

## Q4：为什么多 rank 要用 MIN 汇总状态

`KVPoll` 的数值顺序把失败和未完成状态放在 Success 之前，因此 MIN 是“最保守状态”。任意 rank 未 ready，全局都不能 ready。

```python
# 来源：python/sglang/srt/disaggregation/base/conn.py L79-L84
class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4
```

```python
# 来源：python/sglang/srt/disaggregation/utils.py L121-L140
def poll_and_all_reduce(
    pollers,
    gloo_group: dist.ProcessGroup,
    decode_reqs=None,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    # at a certain prob, the poll is failed to simulate failure
    polls = _poll_with_failure_injection(pollers)

    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    if (
        decode_reqs is not None
        and metadata_buffers is not None
        and server_args is not None
    ):
        _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args)
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()
```

排查 collective hang 时，重点不是看某一个 rank 的 poll 值，而是确认所有 rank 是否以相同请求顺序、相同 list 长度进入同一次 all-reduce。

## Q5：staging buffer 和 HiCache 为什么也会挡住 Success

有些 backend 需要先收到底层 buffer，再 scatter 到最终 KV 位置。此时 raw poll 成功也不能代表本地 KV 可读。

```python
# 来源：python/sglang/srt/disaggregation/utils.py L163-L191
def poll_and_all_reduce_with_staging(
    decode_reqs,
    staging_handler,
    gloo_group: dist.ProcessGroup,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce."""
    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)

    # allow test injection of failure probability at runtime
    receivers = [dr.kv_receiver for dr in decode_reqs]
    raw_polls = _poll_with_failure_injection(receivers)
    for i, decode_req in enumerate(decode_reqs):
        if raw_polls[i] == int(KVPoll.Success):
            if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
                decode_req
            ):
                raw_polls[i] = int(KVPoll.Transferring)
    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    if metadata_buffers is not None and server_args is not None:
        _apply_metadata_gate(raw_polls, decode_reqs, metadata_buffers, server_args)
    poll_tensor = torch.tensor(raw_polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(poll_tensor, op=dist.ReduceOp.MIN, group=gloo_group)
    return poll_tensor.tolist()
```

HiCache 也是同一思路：底层 receiver Success 但本地 restore 还在 pending 时，包装后的 receiver 返回 Transferring。

```python
# 来源：python/sglang/srt/disaggregation/decode_hicache_mixin.py L152-L165
class HiCacheRestoreGatedKVReceiver:
    """Wraps a kv_receiver so KVPoll.Success is gated on HiCache restore READY."""

    def __init__(self, decode_req: DecodeRequest):
        self.decode_req = decode_req

    def poll(self) -> KVPoll:
        poll = self.decode_req.kv_receiver.poll()
        if (
            poll == KVPoll.Success
            and self.decode_req.hicache_restore_status == HiCacheRestoreResult.PENDING
        ):
            return KVPoll.Transferring
        return poll
```

处理建议：

- staging 卡住：查 `SGLANG_DISAGG_STAGING_BUFFER`、backend 是否为 Mooncake/NIXL、`advance_scatter` 是否推进。
- HiCache 卡住：查 restore 状态是否从 PENDING 变成 READY/FAILED，abort 路径是否清理 prefetch 和 tree lock。

## Q6：`disaggregation_decode_extra_slots` 该怎么理解

它不是提高 running batch 上限，而是给 prealloc/transfer 中的请求额外预留 req-to-token slot，减少 Prefill 等 Decode 接收端的时间。

```python
# 来源：python/sglang/srt/disaggregation/decode.py L107-L133
class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        device: str,
        enable_memory_saver: bool,
        pre_alloc_size: int,
    ):
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        self.size = size
        # +1 padding row at index 0; see ReqToTokenPool for rationale.
        self._alloc_size = size + pre_alloc_size + 1
```

调参判断：

- Prefill 经常等 bootstrap，Decode running 没打满：可以考虑增加 extra slots。
- Decode 显存紧或 prealloc 失败增多：extra slots 可能过大。
- running decode 已经满载：extra slots 只能改善接收重叠，不能突破 running 上限。

## Q7：为什么有些配置组合启动时直接失败

PD 的配置不是独立开关。Decode radix cache 与 HiSparse、fake backend、speculative decoding 互斥；Prefill 不支持 fake backend；staging buffer 只允许 Mooncake 或 NIXL。

```python
# 来源：python/sglang/srt/arg_groups/pd_disaggregation_hook.py L29-L88
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

    elif server_args.disaggregation_mode == "prefill":
        assert (
            server_args.disaggregation_transfer_backend != "fake"
        ), "Prefill server does not support 'fake' as the transfer backend"

        server_args.disable_cuda_graph = True

    if server_args.disaggregation_mode in ("prefill", "decode"):
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and server_args.disaggregation_transfer_backend not in ("mooncake", "nixl")
        ):
            raise ValueError(
                f"SGLANG_DISAGG_STAGING_BUFFER requires "
                f"disaggregation_transfer_backend='mooncake' or 'nixl', "
                f"got '{server_args.disaggregation_transfer_backend}'."
            )
```

启动失败时先按错误信息回到这段，不要直接改 backend 代码。大多数失败是源码明确禁止的组合，而不是依赖缺失。

## Q8：PD 一定比 unified 快吗

不一定。PD 适合 Prefill 和 Decode 压力形态不同、长 prompt 或前缀复用明显、并且网络传输成本可控的场景。短 prompt、低 QPS 或跨网络域传输大 KV 时，bootstrap、metadata、RDMA 和双池调度的固定成本可能超过收益。

源码上，PD 不是普通 loop 加一个开关，而是直接换 event loop：

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

评估时至少分开看：

| 指标 | 倾向问题 |
|------|----------|
| Prefill bootstrap time | Decode prealloc 或 room 路由 |
| Prefill retry count / 重复 forward | optimistic handshake 未及时收敛、重算成本 |
| Prefill transfer queue time | backend 传输、metadata buffer、网络 |
| Decode transfer queue time | receiver poll、metadata gate、staging、HiCache |
| Running batch 空位 | Decode 执行容量 |
| TTFT | Prefill forward + bootstrap + transfer 固定成本 |

## 修改代码前的检查

- 改 room 路由：同时检查输入 normalize、DP follow room、metadata room 校验。
- 改 transfer backend：必须实现 receiver/sender poll 语义，并与 `KVPoll` 数值顺序兼容。
- 改 metadata buffer：必须保持 0 表示未写入、commit 后归零、index 不复用错。
- 改 prealloc 容量：区分 running 上限和 in-transfer 预留槽位。
- 改 HiCache 或 staging：必须通过现有 poll gate 接入，不能让 raw Success 直接放行。
- 改 retry：分别验证 Gateway 的整对 HTTP 重放与 Prefill Scheduler 的 optimistic 回收；两者不能共享“只重试失败侧”的假设。

## 验证抓手

这些检查不需要启动完整 PD 集群，适合在改配置、换 transfer backend 或排查 hang 之前先确认源码主线没有读偏。

```powershell
rg -n "bootstrap_room|follow_bootstrap_room_scheduler|pending_reqs|_resolve_pending_reqs|_create_receiver_and_enqueue|create_sender|pending_bootstrap|optimistic_release_and_requeue|_apply_metadata_gate|poll_and_all_reduce_with_staging|dispatch_event_loop|handle_pd_disaggregation" `
  sglang/python/sglang/srt/managers/scheduler.py `
  sglang/python/sglang/srt/managers/data_parallel_controller.py `
  sglang/python/sglang/srt/disaggregation/decode.py `
  sglang/python/sglang/srt/disaggregation/prefill.py `
  sglang/python/sglang/srt/disaggregation/utils.py `
  sglang/python/sglang/srt/arg_groups/pd_disaggregation_hook.py
```

预期现象：

- `scheduler.py` 同时命中无 room 拒绝、`dispatch_event_loop` 和 disagg event loop 分派。
- `data_parallel_controller.py` 命中 `follow_bootstrap_room_scheduler`，说明 room 参与 Prefill 目标 rank 选择。
- `decode.py` 命中 pending rank 解析、receiver 创建、metadata commit 校验和 decode disagg loop。
- `prefill.py` 命中 `create_sender`、`pending_bootstrap`、optimistic 回收与 bootstrap poll 处理。
- `utils.py` 命中 metadata gate 和 staging 版 all-reduce poll，说明 Success 之前还有跨 rank 与 metadata 检查。
- `pd_disaggregation_hook.py` 命中配置互斥入口，启动失败先回到这里看错误是否是显式禁止组合。

运行期排障时，再用日志关键词反查同一条线：

```powershell
rg -n "Prefill transfer failed|Decode handshake failed|Metadata corruption detected|bootstrap_room" sglang/python/sglang/srt/disaggregation
```

如果实际日志只出现 transfer success，但没有进入 decode waiting，优先回到 metadata gate；如果 room 相关日志不一致，先查 router/gateway 是否给 Prefill 与 Decode 分配了同一个 `bootstrap_room`。
