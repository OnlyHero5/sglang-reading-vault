---
type: batch-doc
module: 07-Scheduler
batch: "07"
doc_type: faq
title: "Scheduler：关键问题"
tags:
 - sglang/batch/07
 - sglang/module/scheduler
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# Scheduler：关键问题

---

## Q1：为什么默认用 overlap 而不是 normal 事件循环？

**Explain：** LLM serving 中 GPU forward 耗时远大于 CPU 侧调度（组 batch、更新 Req、ZMQ 发送）。Overlap 让 **上一轮结果的 CPU 处理** 与 **当前 batch 的 GPU forward** 并行，提高吞吐。代价是状态更复杂（`result_queue`、`future_map`、WAR barrier）。

**Code（对比）：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1537-L1540
            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
```

**Comment：** 调试时可 `--disable-overlap-schedule` 回到 normal，行为更易推理。

---

## Q2：什么时候会禁用单轮的 overlap？

**Explain：** `is_disable_overlap_for_batch` 在两类场景返回 True：（1）连续两个 prefill batch；（2）spec + grammar + decode 且 result_queue 非空。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1632-L1649
        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync
```

**Comment：** Grammar 约束依赖上一轮采样结果，必须 sync 后再 launch 当前 batch 的 sample。

---

## Q3：KV 满了怎么办？Retract 是什么？

**Explain：** Decode 阶段每 token 占用 KV slot。`batch.check_decode_mem()` 预估是否够用；不足则 `retract_decode` 撤回部分请求，释放内存，将请求以 `is_retracted=True` 重新加入队列（通常重跑 prefill）。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L3040-L3056
        # Check if decode out of memory
        if (kv_full_retract_flag := not batch.check_decode_mem()) or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            old_available_tokens = self.token_to_kv_pool_allocator.available_size()
            old_ratio = self.new_token_ratio_tracker.current
            mamba_allocator = getattr(
                self.tree_cache.req_to_token_pool, "mamba_allocator", None
            )
            old_mamba_available = (
                mamba_allocator.available_size()
                if mamba_allocator is not None
                else None
            )
            retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
                self.server_args
            )
```

**Comment：** Retract 影响 SLA，但优于整批 OOM；`new_token_ratio_tracker` 动态调整后续 prefill 激进程度。

---

## Q4：PP 和 Overlap 能同时开吗？

**Explain：** 当前实现中 **不完全兼容**。`run_batch` 内注释 `FIXME: pp is not compatible with overlap`；PP 模式走 `event_loop_pp`，使用独立的 launch/process 路径与 proxy tensor 通信，而非 `result_queue` overlap。

**Code：**

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

**Comment：** `pp_size > 1` 时不会进入 `event_loop_overlap`；PP 自身通过 `pp_async_batch_depth` 做 stage 级 overlap。

---

## Q5：为什么只有部分 rank 收 ZMQ？

**Explain：** 避免多 rank 重复消费同一消息；收包 rank broadcast 到 TP group，PP 链式传递。

**易错理解 vs 正确理解：**

| ❌ 错误 | ✅ 正确 |
|---------|---------|
| 每个 TP rank 各连一个 ZMQ socket 收请求 | 仅 `pp0 + attn_tp0 + attn_cp0` pull ZMQ |
| Scheduler 直接 detokenize | 输出经 `send_to_tokenizer` 回 TokenizerManager |
| `running_batch` 存 waiting 请求 | `waiting_queue` 等 prefill；`running_batch` 已 prefill 在 decode |

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L606-L610
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
```

---

## Q6：Chunked Prefill 与 Scheduler 的交互？

**Explain：** 长 prompt 可能被切成多轮 EXTEND。`chunked_req` 指向未完成 chunk 的请求；中间 chunk 完成后 `stash_chunked_request` 缓存 KV，下一轮继续 `adder.add_chunked_req`。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L2604-L2615
        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)

            # Stash (cache) the previous chunk only when it produced new KV
            # beyond what is already cached. A parked chunk (add_chunked_req
            # hybrid-SWA early-return) leaves extend_range.end ==
            # len(prefix_indices), so there is nothing new to cache and
            # stashing would be a no-op.
            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
                self.stash_chunked_request(self.chunked_req)
```

**Comment：** Chunked prefill 与 mixed chunk（prefill+decode 同 batch）由 `chunked_prefill_size` 和 `enable_mixed_chunk` 控制；详见 调度策略/09。

---

## Q7：Engine Pause 如何工作？

**Explain：** `PauseGenerationReqInput` 设置 `_engine_paused=True`，事件循环仍收请求但不组 batch forward，用于权重热更新等场景。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1529-L1531 / L1569-L1570
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue
```

**Comment：** ContinueGeneration 恢复；暂停期间新请求仍可入队，恢复后一并调度。

---

## Q8：Scheduler 与 SchedulePolicy 的边界？

| 模块 | 职责 |
|------|------|
| **Scheduler** | 事件循环、队列生命周期、调用 PrefillAdder、merge batch、run_batch |
| **SchedulePolicy / PrefillAdder** | waiting_queue 排序、单请求能否入 batch、token 预算、抢占 |

Scheduler 在 `_get_new_batch_prefill_raw` 中 **实例化并驱动** PrefillAdder，但不实现具体优先级/compare 逻辑——那是调度策略 的内容。

---

## Q9：Overlap vs Normal——何时关、关哪一层？

**Explain：** 三层决策：**(1) 全局** `--disable-overlap-schedule` → 永久 `event_loop_normal`，适合调试 race、PP 不兼容路径；**(2) 单轮** `is_disable_overlap_for_batch` → 连续 prefill 换 TTFT，或 spec+grammar+decode 需 sync 上一轮 sample；**(3) 架构** `pp_size>1` 根本不进 overlap loop。生产默认 overlap；仅当 p99 TTFT SLA 严于吞吐、或 grammar+spec 组合触发 sync 时感知到单轮 disable。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1618-L1649
    def is_disable_overlap_for_batch(self, batch: ScheduleBatch) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
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

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync
```

**Comment：** `SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP` 默认行为见 server 文档；grammar+spec 的 TODO 表明未来可能重新允许 overlap。

---

## 设计追问

### Q1：`result_queue` 深度永远为 1 吗？为何用 `deque`？

**Explain：** Overlap 设计是「当前 batch GPU forward」与「上一轮 batch CPU process」两阶段流水，通常 queue 长度 0–1。`deque` 支持 `pop_and_process` O(1) 与 `append` 配对；disable overlap 时在 launch 前先 drain queue，保证 grammar 看见最新 token。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1553-L1560
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)
```

**Comment：** idle 时 queue 空，`on_idle` 做 self-check。

---

### Q2：Retract 与 PD prefill retry 都会「重跑 prefill」，如何区分？

**Explain：** Retract 是 unified/decode 池 **KV 不足**时本地决策，`is_retracted=True` 重新入队；PD retry 是 **transfer 失败**后 Prefill 侧重算，受 `prefill_retry_count` 与 `is_retracted` 约束。二者日志字段不同，勿混调参。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L80-L81
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False
```

**Comment：** Retract 见 Q3 `retract_decode`；PD retry 默认生产关闭。

---

### Q3：Engine Pause 期间 overlap queue 里未 process 的 batch 怎么办？

**Explain：** 取决于 `PauseGenerationReqInput.mode`。**`in_place`**（权重热更新常用）：仅设 `_engine_paused=True`，**不 drain** `result_queue` / `running_batch`；resume 后 overlap loop 走标准 `get_next_batch_to_run` 合并路径。**非 in_place**：先 `popleft` 处理 `result_queue` 最后一项，再 merge/retract `running_batch`，清空 `last_batch`/`cur_batch`。Pause 期间 event loop 在 `recv_requests` 之后 `continue`，**不再 launch 新 batch**，但已 inflight GPU kernel 仍会跑完并由 overlap 路径消费。

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L3966-L3977
    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return
```

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L1530-L1531
            if self._engine_paused:
                continue
```

**Comment：** 权重热更新见 [[32-CheckpointEngine-00-MOC|CheckpointEngine CheckpointEngine]]；调试 pause 行为可先 `--disable-overlap-schedule` 简化状态机。

---

## 验证建议（零基础可试）

1. **操作：** 启动服务后连续发 20 条不同 prompt 的 curl 生成请求，观察日志中 `Prefill batch` / `Decode batch` 交替出现；或加 `--disable-overlap-schedule` 重启对比延迟。 
 **预期现象：** 默认 overlap 下吞吐更高；disable 后行为更易推理（forward 与 process 严格串行）。 
 **对应文档节：** [[07-Scheduler-01-核心概念|01-核心概念 § 用户故事]]、§4 事件循环、Q1 overlap vs normal

2. **操作：** 将 `--max-running-requests` 设小（如 8），用 `bench_serving` 或并发 curl 压测至 KV 紧张，grep 日志 `retract`。 
 **预期现象：** 出现 `retract_decode` 相关日志，部分请求 TTFT 升高（重跑 prefill）；服务不 OOM。 
 **对应文档节：** Q3 Retract、§6 Prefill vs Decode

3. **操作：** 调用 Admin pause：`curl -X POST http://127.0.0.1:30000/pause_generation`（路径以实际 HTTP server 为准），再发 generate；随后 `continue_generation`。 
 **预期现象：** pause 期间新请求在 TokenizerManager 侧等待或 Scheduler `_engine_paused` 不组 batch；continue 后恢复。 
 **对应文档节：** Q7 Engine Pause、[[06-TokenizerManager-04-关键问题|06-TokenizerManager §3 权重 pause]]
