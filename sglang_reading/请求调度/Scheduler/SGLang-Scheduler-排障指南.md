---
title: "Scheduler · 排障指南"
type: troubleshooting
framework: sglang
topic: "Scheduler"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-11
---
# Scheduler · 排障指南

## 读者任务

这一篇按排障入口组织。遇到 Scheduler 行为不符合预期时，先判断症状落在哪个边界：事件循环、请求接收、prefill admission、decode KV、overlap result queue、pause、PP stage，还是 PD retry。

## 症状 1：默认 overlap 下状态比 normal 难推理

现象：日志里当前 batch 已经 forward，但上一轮请求的输出/merge 看起来晚一拍；关掉 overlap 后问题更容易理解。

源码入口：normal loop 是串行的，overlap loop 把当前 forward 和上一轮 result processing 分开。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L1521-L1548
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:
        if self.gracefully_exit:
            break
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()

        self.last_batch = batch
```

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L1551-L1613
def event_loop_overlap(self):
    """A scheduler loop that overlaps the CPU processing and GPU computation."""
    self.result_queue: Deque[
        Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
    ] = deque()

    def pop_and_process():
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)

    while True:
        recv_reqs = self.request_receiver.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        self._apply_war_barrier()
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch
        disable_overlap_for_batch = self.is_disable_overlap_for_batch(batch)
        ...
        if batch:
            batch_result = self.run_batch(batch)
            self.result_queue.append((batch.copy(), batch_result))
        ...
        if self.last_batch:
            if not disable_overlap_for_batch:
                pop_and_process()
```

判断：overlap 让 CPU result processing 和 GPU forward 并行。读日志时至少区分 `cur_batch`、live `last_batch`、`result_queue` 的受限 batch snapshot、`copy_done` 和 FutureMap relay；只比较 `last_batch` 与 `cur_batch` 仍不足以定位错位。

验证：用同一模型分别启动默认模式和 `--disable-overlap-schedule`。若状态错乱只在默认模式出现，优先查 `result_queue`、FutureMap、D2H copy、batch 生命周期。

## 症状 2：某一轮突然禁用 overlap

现象：默认启用了 overlap，但某些 batch 会先 drain 上一轮结果，再 launch 当前 batch。

源码入口：`is_disable_overlap_for_batch` 会在连续 prefill 或 spec+grammar+decode 场景返回 true。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L1618-L1649
def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
    # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
    if self.require_mlp_sync:
        is_extend = lambda b: b and b.is_extend_in_batch
    else:
        is_extend = lambda b: b and b.forward_mode.is_extend()

    batch_is_extend = is_extend(batch)
    last_batch_is_extend = is_extend(self.last_batch)

    disable_overlap_for_batch = (
        envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
        and batch_is_extend
        and last_batch_is_extend
    )

    need_grammar_sync = (
        batch
        and not batch.spec_algorithm.is_none()
        and batch.has_grammar
        and batch.forward_mode.is_decode()
        and len(self.result_queue) > 0
    )

    return disable_overlap_for_batch or need_grammar_sync
```

判断：单轮禁用不是退化 bug，而是为了保持 TTFT 或 grammar/spec 的采样正确性。grammar 依赖上一轮采样结果，不能让下一轮 sample 先跑。

验证：看当前 batch 是否 EXTEND、上一轮是否 EXTEND、是否 spec + grammar + decode。调参时区分全局 `--disable-overlap-schedule` 和单轮 `is_disable_overlap_for_batch`。

## 症状 3：KV pool 满后请求被撤回

现象：日志出现 `KV cache pool is full. Retract requests.`，部分请求延迟升高，但服务没有 OOM。

源码入口：decode 前 `update_running_batch` 调 `batch.check_decode_mem()`；失败时 `retract_decode()` 释放部分请求并重新入队。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L3026-L3114
def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
    initial_bs = batch.batch_size()

    batch.filter_batch()
    if batch.is_empty():
        batch.batch_is_full = False
        return batch

    # Check if decode out of memory
    if (kv_full_retract_flag := not batch.check_decode_mem()) or (
        TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
    ):
        old_available_tokens = self.token_to_kv_pool_allocator.available_size()
        old_ratio = self.new_token_ratio_tracker.current
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
            self.server_args
        )
        new_available_tokens = self.token_to_kv_pool_allocator.available_size()
        new_token_gained = new_available_tokens - old_available_tokens
        self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
        self.new_token_ratio_tracker.current = new_token_ratio
        logger.warning(msg_prefix + msg_details)

        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
    else:
        self.new_token_ratio_tracker.decay_step()

    batch.prepare_for_decode()
    return batch
```

判断：retract 是服务继续运行的保护分支。它释放部分 decode 请求的资源，让剩余 batch 能继续跑，被撤回的请求后续重走调度。

验证：压测时同时看 `#retracted_reqs`、`#new_tokens_gained`、`new_token_ratio`、KV pool 使用率、`max_running_requests` 和最大输出长度。频繁 retract 说明容量预算或请求形态有问题。

## 症状 4：只有一个 rank 从 ZMQ 拉请求

现象：你在多 TP/CP/PP 部署里只看到某个 rank 拉 `recv_from_tokenizer`，其他 rank 没有 ZMQ 收包。

源码入口：只有入口 rank 初始化外部 IPC；receiver 再 broadcast 或 PP P2P。

```python
# 来源：python/sglang/srt/managers/scheduler.py L605-L621
def init_ipc_channels(self, port_args: PortArgs):
    is_rank_zero = (
        self.ps.pp_rank == 0
        and self.ps.attn_tp_rank == 0
        and self.ps.attn_cp_rank == 0
    )
    self.ipc_channels = SchedulerIpcChannels.create(
        port_args=port_args,
        is_rank_zero=is_rank_zero,
        skip_tokenizer_init=self.server_args.skip_tokenizer_init,
        metrics_enabled=self.server_args.enable_metrics
        and (
            self.ps.attn_tp_rank == 0
            or self.server_args.enable_metrics_for_all_schedulers
        ),
        enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
    )
```

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler_components/request_receiver.py L141-L201
def _broadcast_reqs_across_ranks(self, recv_reqs: Optional[List]) -> List:
    if self.server_args.enable_dp_attention:
        if self.ps.attn_tp_rank == 0 and self.ps.attn_cp_rank == 0:
            work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)
        else:
            work_reqs = None
            control_reqs = None

        if self.ps.attn_tp_size != 1:
            work_reqs = broadcast_pyobj(
                work_reqs,
                self.attn_tp_group.rank,
                self.attn_tp_cpu_group,
                src=self.attn_tp_group.ranks[0],
            )
        ...
        recv_reqs = work_reqs + control_reqs
    elif self.ps.tp_size != 1:
        recv_reqs = broadcast_pyobj(
            recv_reqs,
            self.tp_group.rank,
            self.tp_cpu_group,
            src=self.tp_group.ranks[0],
        )
    return recv_reqs
```

判断：这是正确设计，避免多 rank 重复消费同一 ZMQ 队列。排障时要看 broadcast/P2P 是否卡住，而不是要求每个 rank 直接连 TokenizerManager。

验证：确认入口 rank 条件 `pp0 + attn_tp0 + attn_cp0`，再看 TP/CP broadcast 或 PP `point_to_point_pyobj`。

## 症状 5：请求进入队列后迟迟不 prefill

现象：ZMQ 收到了请求，`waiting_queue` 增长，但 GPU 没有立即跑 prefill。

源码入口：`PrefillAdder` 不是 FIFO pop，而是综合 token/KV/LoRA/priority/HiCache/chunked prefill 决定本轮准入。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L2804-L2879
adder = PrefillAdder(
    self.page_size,
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    self.running_batch,
    self.new_token_ratio_tracker.current,
    self.max_prefill_tokens,
    chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0,
    self.priority_scheduling_preemption_threshold,
    max_prefill_bs=self.max_prefill_bs,
    max_running_requests=self.max_running_requests,
    prefill_max_requests=self.server_args.prefill_max_requests,
    prefill_delayer_single_pass=prefill_delayer_single_pass,
    dllm_config=self.dllm_config,
    waiting_queue_len=len(self.waiting_queue),
)
...
for req in self.waiting_queue:
    if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
        continue
    if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
        self.running_batch.batch_is_full = True
    if self.running_batch.batch_is_full:
        if (
            not self.enable_priority_preemption
            or not adder.preempt_to_schedule(req, self.server_args)
        ):
            break
```

判断：等待不等于卡死。可能是 running batch 已满、KV 不足、LoRA 槽位不足、HiCache prefetch 未完成、prefill delayer 生效、priority preemption 未触发。

验证：看 `running_batch.batch_is_full`、`req_to_token_pool.available_size()`、LoRA loading 状态、HiCache prefetch 状态和 prefill delayer 配置。

## 症状 6：pause 后新请求收到了但不 forward

现象：pause_generation 之后 Scheduler 仍能接收请求，但没有新 batch launch。

源码入口：事件循环在 `process_input_requests` 后检查 `_engine_paused`，pause handler 设置该标志。`in_place` 模式不 drain 当前状态，等待 resume 后走标准路径。

```python
# 定位：python/sglang/srt/managers/scheduler.py L3966-L4015（pause 分支摘录）
def pause_generation(self, recv_req: PauseGenerationReqInput):
    self._engine_paused = True

    if recv_req.mode == "in_place":
        # In-place pause: just set the flag and return immediately.
        # All scheduler state (running_batch, last_batch, chunked_req,
        # result_queue) is left untouched. On resume, the normal event
        # loop (get_next_batch_to_run) handles last_batch merge,
        # chunked_req cleanup, and overlap result processing through
        # the standard code paths.
        return

    if self.enable_overlap and self.last_batch:
        tmp_batch, tmp_result = self.result_queue.popleft()
        self.process_batch_result(tmp_batch, tmp_result)
    ...
    self.last_batch = None
    self.cur_batch = None
```

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L1566-L1570
recv_reqs = self.request_receiver.recv_requests()
self.process_input_requests(recv_reqs)
if self._engine_paused:
    continue
```

判断：pause 不是停止收包，而是停止组 batch/forward。权重热更新常用 `in_place`，保留 running/result_queue 状态，resume 后由标准循环恢复。

验证：区分 `in_place` 和 `retract` mode。调试 pause 状态时可暂时关 overlap，避免 result_queue 未处理状态干扰判断。

## 症状 7：PP 模式下不走默认 overlap

现象：`pp_size > 1` 时，没有进入 `event_loop_overlap`，而是走 PP 逻辑。

源码入口：事件循环分派优先检查 `pp_size > 1`，PP mixin 有独立 microbatch 循环。

```python
# 来源：python/sglang/srt/managers/scheduler.py L4168-L4178
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
```

```python
# 定位：python/sglang/srt/managers/scheduler_pp_mixin.py L67-L168（PP 循环骨架）
def event_loop_pp(self: Scheduler):
    """
    A scheduler loop for pipeline parallelism.
    Notes:
    1. Each stage runs in the same order and is notified by the previous stage.
    2. We use async send but sync recv to avoid desynchronization while minimizing the communication overhead.
    3. We can use async batch depth to buffer the outputs in the last stage for to allow overlapping the GPU computation and CPU processing and avoid last PP rank staggler.
    """
    self.init_pp_loop_state()
    while True:
        server_is_idle = True
        for mb_id in range(self.pp_loop_size):
            self.running_batch = self.running_mbs[mb_id]
            self.last_batch = self.last_mbs[mb_id]
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            ...
            self.mbs[mb_id] = self.get_next_batch_to_run()
            ...
            if self.cur_batch:
                result, self.launch_event = self._pp_launch_batch(...)
```

判断：普通 PP 的目标是 stage/microbatch 流水，不是 default loop 的 `result_queue` 两拍流水。排障时同时看 request pyobj、typed proxy/output tensor dict、microbatch id、D2H event 与 pending P2P work；PD PP 还要看共识 rid 集合。

验证：检查 `pp_loop_size=pp_size+pp_async_batch_depth`、typed inbox、proxy/output send/recv、last-rank queue、D2H event；若是 XPU 等 backend，还检查奇偶 rank 的 send/recv 顺序。

## 症状 8：chunked prefill 中间 chunk 没有完整输出回传

现象：PP + chunked prefill 下，中间 chunk 没有像普通 prefill 一样回传完整 output。

源码入口：PP mixin 只在非常保守条件下允许跳过 output comm。

```python
# 来源：python/sglang/srt/managers/scheduler_pp_mixin.py L49-L58
def _pp_can_skip_output_comm(batch: ScheduleBatch) -> bool:
    """Check if output send/recv can be skipped for this batch."""
    return (
        envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.get()
        and batch is not None
        and batch.forward_mode == ForwardMode.EXTEND
        and len(batch.reqs) == 1
        and not batch.contains_last_prefill_chunk
        and not batch.return_logprob
    )
```

判断：纯 chunked prefill 中间 chunk 只构建 KV，对用户不可见；没有 logprob、不是最后 chunk 时，跳过 output comm 可减少 PP bubble。

验证：如果请求要求 logprob，或这是最后 prefill chunk，就不能跳过。排障时先看 `contains_last_prefill_chunk` 和 `return_logprob`。

## 症状 9：retract 和 PD prefill retry 都像“重跑 prefill”，容易混淆

现象：请求被重新处理，但不确定是 KV 不足 retract，还是 disaggregation transfer 失败后的 retry。

源码入口：retract 在 Scheduler decode KV 检查中触发；PD prefill retry 在 disaggregation prefill 逻辑中受 `prefill_retry_count` 和 `is_retracted` 限制。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/srt/managers/scheduler.py L3076-L3103
self.new_token_ratio_tracker.current = new_token_ratio
...
logger.warning(msg_prefix + msg_details)

for req in retracted_reqs:
    self._add_request_to_queue(req, is_retracted=True)
```

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L80-L81
if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
    return False
```

判断：retract 是本地 KV 不足；PD retry 是传输失败后的 prefill 侧重试。`req.is_retracted` 会阻止某些 prefill retry。

验证：看日志前缀和 time stats。KV retract 看 `KV cache pool is full` 与 retracted metrics；PD retry 看 disaggregation transfer/retry 日志。

## 最小排障顺序

1. 先确认事件循环：normal、overlap、MLX overlap、PP、PD prefill/decode。
2. 确认请求是否已进入 `process_input_requests`，以及是否被 dispatcher 立即处理为控制面。
3. 确认 `Req` 是否进入正确队列：`waiting_queue`、PD prefill bootstrap、PD decode prealloc。
4. 若等待 prefill，查 `PrefillAdder` 相关资源约束。
5. 若 decode 变慢或请求回退，查 `update_running_batch` 和 retract 日志。
6. 若输出晚一拍，查 overlap 的 `result_queue/copy_done`。
7. 若 PP 模式异常，转看 stage 间 request/proxy/output 通信。

---

## 运行验证

维护本文时，先用下面的命令确认九类症状仍能在源码中定位：

```powershell
rg -n "event_loop_overlap|event_loop_normal|update_running_batch|KV cache pool is full|process_input_requests|_pp_can_skip_output_comm|prefill_retry_count|is_retracted" sglang/python/sglang/srt/managers/scheduler.py sglang/python/sglang/srt/managers/scheduler_pp_mixin.py sglang/python/sglang/srt/disaggregation/prefill.py sglang/python/sglang/srt/managers/scheduler_components/request_receiver.py
```

预期信号：

- `scheduler.py` 仍能找到 normal / overlap event loop、输入处理和 running batch 更新。
- `scheduler.py` 或 PP mixin 仍能找到 retract、KV pool full、PP output comm skip 等排障入口。
- `disaggregation/prefill.py` 仍能找到 `prefill_retry_count` 与 `is_retracted` 的 retry 约束。
- `request_receiver.py` 仍能支撑“只有一个 rank 拉请求”的 ZMQ 接收边界。

如果某个症状的源码入口消失，应先判断是症状不再存在，还是逻辑被拆到新的 scheduler component。
