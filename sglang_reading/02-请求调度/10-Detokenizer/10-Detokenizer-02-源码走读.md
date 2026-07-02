---
type: batch-doc
module: 10-Detokenizer
batch: "10"
doc_type: walkthrough
title: "Detokenizer · 源码走读"
tags:
 - sglang/batch/10
 - sglang/module/detokenizer
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Detokenizer · 源码走读

## 走读顺序

1. `detokenizer_manager.py` — 进程入口、`DetokenizerManager`、增量解码核心
2. `communicator.py` — `FanOutCommunicator`（控制面，与 Detokenizer 对照理解）

---

## 1. detokenizer_manager.py

### 1.1 进程入口 `run_detokenizer_process`

**Explain：** 启动 Detokenizer 子进程：设置进程名、日志、选择单/多 Tokenizer Worker 事件循环；异常时清理 socket mapping 并通知父进程退出。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L483-L506
def run_detokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    detokenizer_manager_class=DetokenizerManager,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    manager = None
    try:
        manager = detokenizer_manager_class(server_args, port_args)
        if server_args.tokenizer_worker_num == 1:
            manager.event_loop()
        else:
            manager.multi_http_worker_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DetokenizerManager hit an exception: {traceback}")
        if manager is not None:
            manager.maybe_clear_socket_mapping()
        parent_process.send_signal(signal.SIGQUIT)

```

**Comment：**

- `detokenizer_manager_class` 参数便于测试注入 mock。
- 多 Worker 模式下不使用 `send_to_tokenizer` 单 socket，改由 `SocketMapping` 按 `http_worker_ipcs` 推送（Mixin 实现）。

---

### 1.2 初始化：IPC、Tokenizer、Dispatcher

**Explain：** `__init__` 分四步：ZMQ 通道、HF tokenizer、运行态（`DecodeStatus` 字典、Watchdog）、类型分发器。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L94-L159
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Init inter-process communication
        self.init_ipc_channels(port_args, server_args)

        # Init tokenizer
        self.init_tokenizer(server_args)

        # Init running status
        self.init_running_status(server_args)

        # Init dispatcher
        self.init_request_dispatcher()

    def init_ipc_channels(self, port_args: PortArgs, server_args: ServerArgs):
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        # In multi-tokenizer mode, results are pushed back to each TokenizerWorker
        # directly via SocketMapping inside multi_http_worker_event_loop, so the
        # single send_to_tokenizer socket is unused.
        if server_args.tokenizer_worker_num == 1:
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_ipc_name, False
            )

    def init_tokenizer(self, server_args: ServerArgs):
        if server_args.skip_tokenizer_init:
            self.tokenizer = None
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                tokenizer_backend=server_args.tokenizer_backend,
            )

    def init_running_status(self, server_args: ServerArgs):
        self.decode_status = LimitedCapacityDict(capacity=DETOKENIZER_MAX_STATES)
        self.disable_tokenizer_batch_decode = server_args.disable_tokenizer_batch_decode
        self.is_tool_call_parser_gpt_oss = server_args.tool_call_parser == "gpt-oss"

        self.soft_watchdog = Watchdog.create(
            debug_name="DetokenizerManager",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DETOKENIZER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("detokenizer")

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (BatchEmbeddingOutput, self.handle_batch_embedding_out),
                (BatchTokenIDOutput, self.handle_batch_token_id_out),
                (FreezeGCReq, self.handle_freeze_gc_req),
                (ConfigureLoggingReq, self.handle_configure_logging_req),
            ]
        )
```

**Comment：**

- Scheduler **PUSH** → Detokenizer **PULL**（`detokenizer_ipc_name`）。
- Detokenizer **PUSH** → TokenizerManager **PULL**（`tokenizer_ipc_name`），仅单 Worker 模式。
- `TypeBasedDispatcher` 与 Scheduler/TokenizerManager 一致，按消息类型路由 handler。

---

### 1.3 主事件循环 `event_loop`

**Explain：** 阻塞接收 Scheduler 消息 → 分发处理 → 若有输出则 PUSH 给 TokenizerManager；Watchdog 在 recv 期间禁用，避免误报 stuck。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L161-L169
    def event_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = sock_recv(self.recv_from_scheduler)
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                sock_send(self.send_to_tokenizer, output)
            self.soft_watchdog.feed()
```

**Comment：**

- `handle_freeze_gc_req` / `handle_configure_logging_req` 返回 `None`，不向下游发送。
- `BatchEmbeddingOutput` 透传，embedding 模型无需 detokenize。

---

### 1.4 有界状态字典 `LimitedCapacityDict`

**Explain：** 每个进行中的流式请求在 `decode_status` 中占一条记录。高并发下可能耗尽内存，故用 LRU 式有界字典；默认容量 `1<<16`，可通过环境变量 `SGLANG_DETOKENIZER_MAX_STATES` 调整。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L56-L60, L470-L480
# Maximum number of request states that detokenizer can hold. When exceeded,
# oldest request states will be evicted. Default: 65536 (1<<16).
# For more details, see: https://github.com/sgl-project/sglang/issues/2812
# Use power of 2 values for better memory allocation.
DETOKENIZER_MAX_STATES = int(os.environ.get("SGLANG_DETOKENIZER_MAX_STATES", 1 << 16))
```

**Comment：**

- 驱逐最旧 `rid` 后，后续 batch 找不到 `DecodeStatus` 会抛 `RuntimeError` 并提示增大 `SGLANG_DETOKENIZER_MAX_STATES`（见 issue #2812）。
- 幂次容量有利于内存分配器行为（注释说明）。

---

### 1.5 批量分组解码 `_grouped_batch_decode`

**Explain：** 对 batch 内各行 token id 做 `batch_decode` 优化：过滤空 id 列表、慢速 tokenizer 逐行 decode、快速 tokenizer 按 `(skip_special_tokens, spaces_between_special_tokens)` 分组批量 decode。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L207-L262
    def _grouped_batch_decode(
        self,
        ids_list: List[List[int]],
        skip_list: List[bool],
        space_list: List[bool],
    ) -> List[str]:
        """Batch decode with grouping by (skip_special_tokens, spaces_between_special_tokens)."""
        n = len(ids_list)
        if n == 0:
            return []

        # Empty token spans decode to "" but tokenizer.batch_decode (and the
        # slow per-row decode_without_hf_kwargs path) still pays per-row
        # overhead; under high-concurrency streaming this adds up. Filter
        # empties out, decode the rest, then scatter back.
        keep_idx: Optional[List[int]] = None
        if not all(ids_list):
            keep_idx = [i for i, ids in enumerate(ids_list) if ids]
            if not keep_idx:
                return [""] * n
            ids_list = [ids_list[i] for i in keep_idx]
            skip_list = [skip_list[i] for i in keep_idx]
            space_list = [space_list[i] for i in keep_idx]

        if not getattr(self.tokenizer, "is_fast", False):
            decoded = [
                decode_without_hf_kwargs(self.tokenizer, ids, skip)
                for ids, skip in zip(ids_list, skip_list)
            ]
        else:
            # fast path: all rows share the same (skip, space) flags.
            first_skip, first_space = skip_list[0], space_list[0]
            if all(
                s == first_skip and sp == first_space
                for s, sp in zip(skip_list, space_list)
            ):
                decoded = self.tokenizer.batch_decode(
                    ids_list,
                    skip_special_tokens=first_skip,
                    spaces_between_special_tokens=first_space,
                )
            else:
                # Group indices by (skip, space) tuple and decode each group.
                groups: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)
                for idx, (skip, space) in enumerate(zip(skip_list, space_list)):
                    groups[(skip, space)].append(idx)

                decoded = [""] * len(ids_list)
                for (skip, space), indices in groups.items():
                    group_decoded = self.tokenizer.batch_decode(
                        [ids_list[idx] for idx in indices],
                        skip_special_tokens=skip,
                        spaces_between_special_tokens=space,
                    )
                    for idx, text in zip(indices, group_decoded):
                        decoded[idx] = text
```

**Comment：**

- `disable_tokenizer_batch_decode=True` 时走逐行 `tokenizer.decode`，规避 gpt-oss 等模型的边界 case。
- 空 token span 解码为 `""`，但避免对空列表调用 `batch_decode` 的 per-row 开销。

---

### 1.6 核心：`_decode_batch_token_id_output`

**Explain：** 本函数实现完整增量解码：更新 `DecodeStatus` → 构造 `surr_ids` / `read_ids` → batch decode → 计算每请求 `output_strs`（流式 vs 结束）。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L271-L297
    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
        bs = len(recv_obj.rids)

        # Initialize decode status
        read_ids, surr_ids = [], []
        for i in range(bs):
            rid = recv_obj.rids[i]
            if rid not in self.decode_status:
                s = DecodeStatus(
                    decoded_text=recv_obj.decoded_texts[i],
                    decode_ids=list(recv_obj.decode_ids[i]),
                    surr_offset=0,
                    read_offset=recv_obj.read_offsets[i],
                )
                self.decode_status[rid] = s
            else:
                s = self.decode_status[rid]
                s.decode_ids.extend(recv_obj.decode_ids[i])

            read_ids.append(
                self.trim_matched_stop(
                    s.decode_ids[s.surr_offset :],
                    recv_obj.finished_reasons[i],
                    recv_obj.no_stop_trim[i],
                )
            )
            surr_ids.append(s.decode_ids[s.surr_offset : s.read_offset])
```

**Comment：**

- `read_ids`：从 `surr_offset` 到当前全部 token（含本步新增），用于 decode 出「完整可读段」。
- `surr_ids`：从 `surr_offset` 到 `read_offset`，代表**上一轮已提交边界内**的 token，用于差分 `read_texts[i][len(surr_texts[i]):]` 得到增量。
- `trim_matched_stop` 在 finished 时对 stop string/token 截断。

---

### 1.7 流式 UTF-8 边界与 `find_printable_text`

**Explain：** 若增量文本以 replacement char `�` 结尾，说明 UTF-8 字符尚未完整，不能提交 token 偏移；只发送 `find_printable_text` 提取的可打印前缀，并更新 `sent_offset` 避免重复发送。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L349-L370
            new_text = read_texts[i][len(surr_texts[i]) :]
            if recv_obj.finished_reasons[i] is None:
                # Streaming. Invariant: sent_offset >= decoded_text_len. The
                # gap (`pending`) is "printable but uncommitted" text emitted
                # in a prior "�" recovery step; we skip it from this step's
                # emission so we don't double-send.
                pending = s.sent_offset - s.decoded_text_len
                if new_text and not new_text.endswith("�"):
                    # Clean text: commit to decoded_text and advance offsets.
                    s.append_decoded_text(new_text)
                    s.surr_offset = s.read_offset
                    s.read_offset = len(s.decode_ids)
                    s.sent_offset = s.decoded_text_len
                    output_strs.append(new_text[pending:] if pending else new_text)
                else:
                    # Incomplete UTF-8: emit the printable prefix only; do not
                    # commit (token offsets stay so the next iteration retries
                    # with more tokens).
                    printable = find_printable_text(new_text)
                    s.sent_offset = s.decoded_text_len + len(printable)
                    output_strs.append(printable[pending:] if pending else printable)
                continue
```

**Comment：**

- **Invariant**：流式时 `sent_offset >= decoded_text_len`；二者之差为此前 `�` 恢复步骤已发送但未 commit 的文本。
- 干净文本：commit 到 `decoded_text`，推进 `surr_offset`/`read_offset`。
- 不完整 UTF-8：不推进 token offset，下轮用更多 token 重试 decode。

---

### 1.8 输出组装 `handle_batch_token_id_out`

**Explain：** 将 decode 得到的 `output_strs` 与 Scheduler 透传字段组装为 `BatchStrOutput`；`routed_experts` / `indexer_topk` 在此做 base64 编码，减轻 TokenizerManager 热路径负担。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L387-L455
    @staticmethod
    def _b64_encode_per_request(
        data_list: Optional[List[Optional[torch.Tensor]]],
    ) -> Optional[List[Optional[str]]]:
        """Encode a per-request list of tensors as base64 strings, off the
        tokenizer hot path. Returns None when the input is None; per-item None
        stays None.
        """
        if data_list is None:
            return None
        return [
            (
                pybase64.b64encode(item.numpy().tobytes()).decode("utf-8")
                if item is not None
                else None
            )
            for item in data_list
        ]

    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)
        indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            reasoning_tokens=recv_obj.reasoning_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
            cached_tokens_details=recv_obj.cached_tokens_details,
            image_tokens=recv_obj.image_tokens,
            audio_tokens=recv_obj.audio_tokens,
            video_tokens=recv_obj.video_tokens,
            spec_verify_ct=recv_obj.spec_verify_ct,
            spec_num_correct_drafts=recv_obj.spec_num_correct_drafts,
            spec_correct_drafts_histogram=recv_obj.spec_correct_drafts_histogram,
            input_token_logprobs_val=recv_obj.input_token_logprobs_val,
            input_token_logprobs_idx=recv_obj.input_token_logprobs_idx,
            output_token_logprobs_val=recv_obj.output_token_logprobs_val,
            output_token_logprobs_idx=recv_obj.output_token_logprobs_idx,
            input_top_logprobs_val=recv_obj.input_top_logprobs_val,
            input_top_logprobs_idx=recv_obj.input_top_logprobs_idx,
            output_top_logprobs_val=recv_obj.output_top_logprobs_val,
            output_top_logprobs_idx=recv_obj.output_top_logprobs_idx,
            input_token_ids_logprobs_val=recv_obj.input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=recv_obj.input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=recv_obj.output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=recv_obj.output_token_ids_logprobs_idx,
            output_token_entropy_val=recv_obj.output_token_entropy_val,
            output_hidden_states=recv_obj.output_hidden_states,
            routed_experts=routed_experts,
            indexer_topk=indexer_topk,
            customized_info=recv_obj.customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=recv_obj.retraction_counts,
            token_steps=recv_obj.token_steps,
            dp_ranks=recv_obj.dp_ranks,
            time_stats=recv_obj.time_stats,
        )
```

**Comment：**

- idle batch（`len(rids)==0`）时 `output_strs=[]`，仍可能携带控制类副作用（取决于 Scheduler 是否发空 batch）。
- `BatchStrOutput.routed_experts` 类型为 `List[Optional[str]]`（base64），与输入侧 `torch.Tensor` 不同。

---

### 1.9 Stop 截断 `trim_matched_stop`

**Explain：** 生成结束时根据 `finished_reason.matched` 去掉 stop string 或 stop token；`no_stop_trim` 保留 matched 内容；gpt-oss tool call token `200012` 有特殊处理。

**Code：**

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L171-L201
    def trim_matched_stop(
        self, output: Union[str, List[int]], finished_reason: Dict, no_stop_trim: bool
    ):
        if not finished_reason:
            return output

        matched = finished_reason.get("matched", None)
        if not matched:
            return output

        # TODO(lmzheng): handle the case where multiple stop strs are hit

        # Trim stop str.
        if isinstance(matched, str) and isinstance(output, str):
            pos = output.find(matched)
            if pos == -1:
                return output
            end = pos + len(matched)
            return output[:end] if no_stop_trim else output[:pos]

        # Trim stop token.
        if isinstance(matched, int) and isinstance(output, list):
            if no_stop_trim:
                return output
            # 200012 <|call|> is the tool call token and one of eos tokens for gpt-oss model
            if output[-1] == 200012 and self.is_tool_call_parser_gpt_oss:
                return output
            assert len(output) > 0
            # NOTE: We can always assume the last token is the matched stop token
            return output[:-1]
        return output
```

**Comment：**

- 字符串 stop：默认截到 matched **之前**；`no_stop_trim` 则保留 matched 含在内。
- Token stop：默认去掉最后一个 token（假定其为 matched stop）。

---

## 2. communicator.py

### 2.1 `queueing_call`：串行控制请求

**Explain：** 若已有 in-flight 请求或队列非空，新调用者入队等待；否则发送对象、收集 `fan_out` 个响应后唤醒下一个等待者。

**Code：**

```python
# 来源：python/sglang/srt/managers/communicator.py L38-L58
    async def queueing_call(self, obj: T):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj is not None:
            self._send(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values
```

**Comment：**

- 保证控制操作**严格串行**，避免并发 flush/update 交错。
- `obj is None` 时只等待当前 in-flight 完成（用于「只读同步」场景）。

---

### 2.2 `handle_recv` 与 `merge_results`

**Explain：** 每个 Scheduler 回复调用 `handle_recv`；收齐 `fan_out` 份后 `set` 事件。`merge_results` 合并多 DP rank 的成功标志与 message。

**Code：**

```python
# 来源：python/sglang/srt/managers/communicator.py L86-L96
    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
```

**Comment：**

- Detokenizer **不参与**此 fan-in；它是 Scheduler 输出链路的单消费者（或多 Detokenizer Router 场景下的分片，见 TokenizerManager/09 扩展）。
- `watching_call` 模式允许多协程共享同一 in-flight 请求，返回 `deepcopy` 结果（见同文件 L60-L78）。

---

## 3. 走读小结

| 组件 | 职责 |
|------|------|
| `run_detokenizer_process` | OS 进程入口 |
| `event_loop` | ZMQ 收/发 + 分发 |
| `DecodeStatus` + `_decode_batch_token_id_output` | 有状态增量 detokenize |
| `_grouped_batch_decode` | batch_decode 性能优化 |
| `trim_matched_stop` | 结束截断 |
| `_b64_encode_per_request` | MoE 调试 tensor 编码 |
| `FanOutCommunicator` | TokenizerManager 控制面 DP fan-out（非 Detokenizer 路径） |
