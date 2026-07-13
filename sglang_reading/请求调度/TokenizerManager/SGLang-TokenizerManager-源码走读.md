---
title: "TokenizerManager · 源码走读"
type: walkthrough
framework: sglang
topic: "TokenizerManager"
learning_role: reference
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-11
---
# TokenizerManager · 源码走读

本篇沿一条普通 `/generate` 请求走读：HTTP 层调用 `generate_request`，TokenizerManager 注册 `ReqState`，完成分词与 tokenized object 构造，经 ZMQ 发给 Scheduler；后端输出回来后，后台 `handle_loop` 写入 `ReqState.out_list` 并唤醒前台 `_wait_one_response`，最终 yield 给 HTTP/SSE。

## 读者任务

读完本篇，你应该能：

- 从源码解释 `generate_request` 为什么是 async generator。
- 找到一个请求什么时候进入 `rid_to_state`，什么时候被删除。
- 解释 normalization 和 `n > 1` 为什么会改变 batch、rid 和发送次数。
- 解释 `BatchStrOutput`、`BatchTokenIDOutput`、`BatchEmbeddingOutput` 三类回包如何变成 HTTP 输出。
- 判断 score、flush cache、multi-tokenizer worker 是主线分叉，不是另一套后端。

## 长文读法

这篇按两条协程线和两类分叉读：前台 `generate_request` 负责建 `ReqState`、分词、发送、等待；后台 `handle_loop` 负责收 `Batch*Output`、写 state、唤醒 event；score、控制请求、多 tokenizer worker 只是复用或改写这两条线的分叉。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立主线 | 主线图、步骤 1 到 4 | HTTP route 不处理后端细节，TokenizerManager 的入口必须是 async generator |
| 请求没发到 Scheduler | 步骤 4 到 8 | 先注册 `rid_to_state`，再经过 pause、权重 reader lock、分词、tokenized object 和 IPC 包装 |
| HTTP 没有收到输出 | 步骤 9 到 14 | 后台先把 `BatchStrOutput/BatchTokenIDOutput/BatchEmbeddingOutput` 写入 state，再用 event 唤醒前台 |
| 流式 chunk 异常或 token ids 丢失 | 步骤 10 到 14 | incremental、non-incremental 和 skip-tokenizer 三条回程语义不同，前台会 drain 并合并 backlog |
| batch 请求行为异常 | 步骤 5、15 | batch 不是一个共享 state，而是多个 rid/state 的并发等待与聚合 |
| score、flush cache、多 worker 让人困惑 | 步骤 16 到 18 | score 复用 generate/embedding 数据面；控制面走 communicator；多 worker 靠 `http_worker_ipc` 拆回包 |

读完整篇后，排障时先问：问题发生在前台请求协程、后台回包协程、控制 communicator，还是多 worker router。只有先分清这四个边界，`rid_to_state` 泄漏、streaming backlog、flush 卡住和 worker 回包错位才不会混在一起。

## 主线图

```text
http_server.generate_request
  -> TokenizerManager.generate_request
    -> _init_req_state
    -> _tokenize_one_request
    -> TokenizedGenerateReqInput
    -> _send_one_request
    -> _wait_one_response
  <- handle_loop
    <- _handle_batch_output
    <- BatchStrOutput / BatchTokenIDOutput / BatchEmbeddingOutput
```

## 步骤 1：HTTP 层把 streaming 和 non-streaming 都交给同一个 generator

系统压力：HTTP 层需要同时支持 SSE 流式返回和一次性 JSON 返回，但后端请求生命周期应该只有一套。

源码选择：FastAPI route 不直接分词，也不直接读 ZMQ；它调用 `_global_state.tokenizer_manager.generate_request(obj, request)`。streaming 分支 `async for`，non-streaming 分支取 `.__anext__()`。

```python
# 来源：sglang/python/sglang/srt/entrypoints/http_server.py L790-L800
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.get():
        apply_header_overrides(obj, request.headers)
    if obj.stream:

        async def stream_results() -> AsyncIterator[bytes]:
            try:
                async for out in _global_state.tokenizer_manager.generate_request(
                    obj, request
                ):
```

```python
# 来源：sglang/python/sglang/srt/entrypoints/http_server.py L827-L832
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
```

这解释了为什么 TokenizerManager 的主入口必须是 async generator：同一套后端生命周期既要服务 SSE 多次 yield，也要服务非流式的一次 yield。

## 步骤 2：初始化只搭出子系统边界，不做请求处理

系统压力：TokenizerManager 同时有 tokenizer、IPC、LoRA、权重更新、metrics、dispatcher 等职责。如果构造函数处理请求逻辑，子类和 worker 模式会很难复用。

源码选择：构造函数只按顺序初始化子系统。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L257-L297
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.enable_metrics = server_args.enable_metrics
        self.preferred_sampling_params = server_args.preferred_sampling_params
        self.crash_dump_folder = server_args.crash_dump_folder
        set_global_server_args_for_tokenizer(server_args)

        # Init model config
        self.init_model_config()

        # Initialize tokenizer and multimodalprocessor
        self.init_tokenizer_and_processor()

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init running status
        self.init_running_status()

        # Init logging and dumping
        self.init_request_logging_and_dumping()

        # Init weight update
        self.init_weight_update()

        # Init LoRA status
        self.init_lora()

        # Init PD disaggregation and encoder disaggregation
        self.init_disaggregation()

        # Init metric collector and watchdog
        self.init_metric_collector_watchdog()

        # Init request dispatcher
        self.init_request_dispatcher()
```

读这段要抓住顺序：模型/tokenizer 能力先确定，再建立 IPC，再初始化运行状态和控制面。请求生命周期从 `generate_request` 才开始。

## 步骤 3：IPC 通道决定单 worker 和多 worker 的数据面形态

系统压力：单 HTTP worker 可以直接 PUSH 到 Scheduler；多 HTTP worker 需要先进入 router，否则回包不知道该回哪个 worker。

源码选择：`tokenizer_worker_num == 1` 时 `send_to_scheduler` 直连 `scheduler_input_ipc_name`；多 worker 时改连 `tokenizer_worker_ipc_name`，并保存本 worker 的 `tokenizer_ipc_name` 用于 stamp 回包路由。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L382-L413
    def init_ipc_channels(self, port_args: PortArgs):
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        if self.server_args.tokenizer_worker_num == 1:
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
            )
            self.tokenizer_ipc_name = None
        else:
            # Use tokenizer_worker_ipc_name in multi-tokenizer mode
            self.send_to_scheduler = get_zmq_socket(
                context, zmq.PUSH, port_args.tokenizer_worker_ipc_name, False
            )
            self.tokenizer_ipc_name = port_args.tokenizer_ipc_name

        self.load_snapshot_reader = create_load_snapshot_reader(
            self.server_args,
            port_args,
            caller="TokenizerManager",
        )

    def _dispatch_to_scheduler(self, obj: Any) -> None:
        if self.tokenizer_ipc_name is not None:
            stamp_http_worker_ipc(obj, self.tokenizer_ipc_name)
        sock_send(self.send_to_scheduler, obj)

    async def _async_dispatch_to_scheduler(self, obj: Any) -> None:
        if self.tokenizer_ipc_name is not None:
            stamp_http_worker_ipc(obj, self.tokenizer_ipc_name)
        await async_sock_send(self.send_to_scheduler, obj)
```

`stamp_http_worker_ipc` 是多 worker 回包能归位的关键；单 worker 下它不需要写，因为只有一个接收者。

## 步骤 4：`generate_request` 先注册状态，再进入 pause 和权重锁

系统压力：请求可能在分词、LoRA 校验、输入长度校验之前失败；如果已经创建了状态但失败时不清理，`rid_to_state` 会泄漏。另一方面，权重更新时不能让新请求带着旧状态进入 Scheduler。

源码选择：先 normalize，再 `_init_req_state`，然后等待 pause 解除和 reader lock，再分词发送；异常路径调用 `_discard_pending_req_states`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L589-L646
    async def generate_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()

        # Normalize the request
        obj.normalize_batch_and_arguments()
        self._set_default_priority(obj)

        if isinstance(obj, GenerateReqInput) and obj.routed_dp_rank is not None:
            dp_size = self.server_args.dp_size
            if dp_size <= 1 and obj.routed_dp_rank == 0:
                logger.debug(
                    f"routed_dp_rank={obj.routed_dp_rank} is ignored because dp_size={dp_size}"
                )
            elif obj.routed_dp_rank < 0 or obj.routed_dp_rank >= dp_size:
                raise ValueError(
                    f"routed_dp_rank={obj.routed_dp_rank} out of range [0, {dp_size})"
                )

        self._init_req_state(obj, request)
        try:
            if self.server_args.language_only:
                self._handle_epd_disaggregation_encode_request(obj)

            # Log the request
            self.request_logger.log_received_request(obj, self.tokenizer, request)

            async with self.is_pause_cond:
                await self.is_pause_cond.wait_for(lambda: not self.is_pause)

            async with self.model_update_lock.reader_lock:
                await self._validate_and_resolve_lora(obj)

                # Tokenize the request and send it to the scheduler
                if obj.is_single:
                    tokenized_obj = await self._tokenize_one_request(obj)
                    state = self.rid_to_state[obj.rid]
                    if obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_obj.input_ids)
                    self._send_one_request(tokenized_obj)
                    async for response in self._wait_one_response(obj, request):
                        yield response
                else:
                    async for response in self._handle_batch_request(obj, request):
                        yield response
        except Exception:
            # _init_req_state created a rid_to_state entry per (sub-)request up
            # front. The normal remover is the scheduler-response path
            # (_handle_batch_output), so a failure *before* a request reaches the
            # scheduler -- e.g. input-length validation rejecting an over-context
            # request -- would otherwise leak those entries forever. Drop any that
            # are still pending; entries already removed on the normal completion
            # path are left untouched (pop is a no-op).
            self._discard_pending_req_states(obj)
            raise
```

这里有两个不变量：

| 不变量 | 破坏后症状 |
|--------|------------|
| 每个进入前台路径的 `rid` 必须先有 `ReqState` | 后端回包无法唤醒 HTTP 协程 |
| 未到 Scheduler 前失败必须清理 state | `rid_to_state` 长期增长，健康检查和内存表现异常 |

### 步骤 4A：normalize 会改写请求形态

`generate_request` 的第一步不是注册 state，而是 `obj.normalize_batch_and_arguments()`。它补默认 `rid`、判断 single/batch、展开 batch 参数，并读取 `sampling_params.n`。因此后续 `_init_req_state` 看到的是规范化后的对象，不一定和 API 刚构造时相同。

当前 `GenerateReqInput._validate_inputs` 只直接拒绝“三者全空”和“三者全有”；调用方仍应遵守 `text`、`input_ids`、`input_embeds` 三选一的公开契约，不要依赖两个字段同时出现时后续分支的偶然优先级。准备改 API 校验时，这里应与 HTTP/OpenAI schema、Engine 调用和测试一起收紧。

## 步骤 5：`_init_req_state` 把单请求和 batch 归一成 rid 列表

系统压力：batch request 在 API 层是一个对象，但后端输出仍然按每个 `rid` 返回。状态表不能只存 batch 总对象。

源码选择：单请求和 batch 都展开成 `(rid, sub_obj, bootstrap_room)`，每个 rid 创建一个 `ReqState`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L2850-L2893
    def _init_req_state(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        created_time = obj.received_time

        external_trace_header = None
        if self.server_args.enable_trace:
            if obj.external_trace_header:
                # When the request comes from the rust grpc server or Engine there isn't a
                # real request object but we still need to propagate the trace context from
                # the trace context that is explicitly passed in
                external_trace_header = obj.external_trace_header
            elif request:
                external_trace_header = extract_trace_headers(request.headers)
                obj.external_trace_header = external_trace_header

        # Normalize single/batch into a uniform list of (rid, sub_obj, bootstrap_room)
        if not hasattr(obj, "is_single") or obj.is_single:
            items = [(obj.rid, obj, getattr(obj, "bootstrap_room", None))]
        else:
            items = [
                (
                    obj.rid[i],
                    obj[i],
                    (
                        obj.bootstrap_room[i]
                        if hasattr(obj, "bootstrap_room") and obj.bootstrap_room
                        else None
                    ),
                )
                for i in range(len(obj.rid))
            ]

        for rid, sub_obj, bootstrap_room in items:
            if rid in self.rid_to_state:
                raise ValueError(f"Duplicate request ID detected: {rid}")
            time_stats = APIServerReqTimeStats(disagg_mode=self.disaggregation_mode)
            state = ReqState([], False, asyncio.Event(), sub_obj, time_stats)
            self.rid_to_state[rid] = state
            if self.server_args.enable_trace:
                time_stats.init_trace_ctx(rid, bootstrap_room, external_trace_header)
            time_stats.set_created_time(created_time)
```

`ReqState([], False, asyncio.Event(), sub_obj, time_stats)` 是前后台协程的共享点。后续 `_wait_one_response` 只等这个 event；`_handle_batch_output` 只按 rid 找这个 state。

## 步骤 6：分词策略先判断输入形态，再选择 tokenizer backend

系统压力：TokenizerManager 同时要处理单字符串、批量字符串、cross-encoder pair、动态 batch tokenizer、非 fast tokenizer。把这些分支散落到请求主线里会让生命周期不可读。

源码选择：`_tokenize_texts` 先 detect/prepare，再决定是否用 `async_dynamic_batch_tokenizer`，否则走 regular tokenizer。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L744-L786
        if not texts or self.tokenizer is None:
            raise ValueError("texts cannot be empty and tokenizer must be initialized")

        # Step 1: Detect input format and prepare for tokenization
        input_format = self._detect_input_format(texts, is_cross_encoder)
        tokenizer_input = self._prepare_tokenizer_input(texts, input_format)
        original_batch_size = len(texts) if not isinstance(texts, str) else 1

        # Step 2: Set up tokenizer arguments
        tokenizer_kwargs = (
            {"return_token_type_ids": is_cross_encoder} if is_cross_encoder else {}
        )

        # Step 3: Choose tokenization strategy
        use_async_tokenizer = (
            self.async_dynamic_batch_tokenizer is not None
            and input_format == InputFormat.SINGLE_STRING
        )

        if use_async_tokenizer:
            logger.debug("Using async dynamic batch tokenizer for single text")
            result = await self.async_dynamic_batch_tokenizer.encode(
                tokenizer_input[0], **tokenizer_kwargs
            )
            # Convert to batch format for consistency
            input_ids = [result["input_ids"]]
            token_type_ids = (
                [result["token_type_ids"]]
                if is_cross_encoder and result.get("token_type_ids")
                else None
            )
        else:
            logger.debug(f"Using regular tokenizer for {len(tokenizer_input)} inputs")

            if not is_cross_encoder and (not getattr(self.tokenizer, "is_fast", False)):
                input_ids = [self.tokenizer.encode(t) for t in tokenizer_input]
                token_type_ids = None
            else:
                encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
                input_ids = encoded["input_ids"]
                token_type_ids = (
                    encoded.get("token_type_ids") if is_cross_encoder else None
                )
```

注意这里的 async dynamic batch tokenizer 只覆盖 single string；batch strings 和 cross-encoder pairs 仍走 regular tokenizer 逻辑。

## 步骤 7：单请求 tokenized object 是 API 意图到 Scheduler 契约的转换点

系统压力：Scheduler 不应该理解完整 HTTP schema。它需要的是已分词、已校验、采样参数规范化、带上 LoRA/PD/DP/meta 的内部对象。

源码选择：`_create_tokenized_object` 构造 `TokenizedGenerateReqInput`，保留 `rid`、`http_worker_ipc`、routing、LoRA、sampling 等后端需要的字段。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1150-L1187
            tokenized_obj = TokenizedGenerateReqInput(
                input_text=input_text,
                input_ids=input_ids_arr,
                mm_inputs=mm_inputs,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                top_logprobs_num=obj.top_logprobs_num,
                token_ids_logprob=obj.token_ids_logprob,
                stream=obj.stream,
                rid=obj.rid,
                http_worker_ipc=obj.http_worker_ipc,
                bootstrap_host=obj.bootstrap_host,
                bootstrap_port=obj.bootstrap_port,
                bootstrap_room=bootstrap_room,
                lora_id=obj.lora_id,
                input_embeds=input_embeds,
                positional_embed_overrides=obj.positional_embed_overrides,
                session_id=obj.session_id,
                session_params=session_params,
                custom_logit_processor=obj.custom_logit_processor,
                require_reasoning=obj.require_reasoning,
                return_hidden_states=obj.return_hidden_states,
                return_routed_experts=obj.return_routed_experts,
                routed_experts_start_len=obj.routed_experts_start_len,
                return_indexer_topk=obj.return_indexer_topk,
                routed_dp_rank=obj.routed_dp_rank,
                disagg_prefill_dp_rank=obj.disagg_prefill_dp_rank,
                priority=obj.priority,
                extra_key=obj.extra_key,
                routing_key=obj.routing_key,
                token_type_ids=token_type_ids,
                need_wait_for_mm_inputs=obj.need_wait_for_mm_inputs,
                num_items_assigned=obj.num_items_assigned,
                multi_item_delimiter_indices=obj.multi_item_delimiter_indices,
                mm_data_mooncake=obj.mm_data_mooncake,
                encoder_urls=obj.encoder_urls,
            )
```

这段和 [[SGLang-ScheduleBatch数据结构]] 对上：后续 Scheduler 读的不是 `GenerateReqInput`，而是这个 tokenized IPC 契约。

## 步骤 8：发送前包装共享内存和 pickle 字段

系统压力：多模态特征和复杂字段不能都当普通 Python 对象裸发；大对象要尽量走共享内存，pickle 字段要在 IPC 前封装。

源码选择：单请求路径设置 dispatch 时间，`wrap_shm_features`，再 `wrap_pickle_fields`，最后 `_dispatch_to_scheduler`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1331-L1363
    def _send_one_request(
        self,
        tokenized_obj: Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput],
    ):
        tokenized_obj.time_stats.set_api_server_dispatch_time()
        tokenized_obj = wrap_shm_features(tokenized_obj)
        time_stats = tokenized_obj.time_stats
        tokenized_obj.wrap_pickle_fields()
        self._dispatch_to_scheduler(tokenized_obj)
        tokenized_obj.time_stats = time_stats
        tokenized_obj.time_stats.set_api_server_dispatch_finish_time()

    def _send_batch_request(
        self,
        tokenized_objs: List[
            Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]
        ],
    ):
        """Send a batch of tokenized requests as a single batched request to the scheduler."""
        set_time_batch(tokenized_objs, "set_api_server_dispatch_time")
        time_stats = [tokenized_obj.time_stats for tokenized_obj in tokenized_objs]
        for tokenized_obj in tokenized_objs:
            tokenized_obj.wrap_pickle_fields()

        if isinstance(tokenized_objs[0], TokenizedGenerateReqInput):
            batch_req = BatchTokenizedGenerateReqInput(batch=tokenized_objs)
        else:
            batch_req = BatchTokenizedEmbeddingReqInput(batch=tokenized_objs)

        self._dispatch_to_scheduler(batch_req)
        for tokenized_obj, time_stat in zip(tokenized_objs, time_stats):
            tokenized_obj.time_stats = time_stat
        set_time_batch(tokenized_objs, "set_api_server_dispatch_finish_time")
```

`_send_batch_request` 只是把多条 tokenized request 合成一个 IPC envelope；真正 continuous batching 仍在 Scheduler。

## 步骤 9：后台收包循环把 ZMQ 对象分成数据面和控制面

系统压力：同一个 `recv_from_detokenizer` 通道会收到数据面输出，也会收到控制面回复。处理方式必须分开。

源码选择：`handle_loop` 用类型判断；批输出进入 `_handle_batch_output`，其余交给 `_result_dispatcher`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1822-L1860
    def auto_create_handle_loop(self):
        if self.event_loop is not None:
            return

        # Create and start the handle_loop task
        loop = get_or_create_event_loop()
        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.handle_loop))
        )
        self.event_loop = loop

        # We only add signal handler when the tokenizer manager is in the main thread
        # due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = self.signal_handler_class(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(
                signal.SIGQUIT, signal_handler.running_phase_sigquit_handler
            )

        self.asyncio_tasks.add(
            loop.create_task(print_exception_wrapper(self.sigterm_watchdog))
        )

    async def handle_loop(self):
        """The event loop that handles requests"""
        while True:
            with self.soft_watchdog.disable():
                recv_obj = await async_sock_recv(self.recv_from_detokenizer)
            if isinstance(
                recv_obj,
                (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
            ):
                await self._handle_batch_output(recv_obj)
            else:
                self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = real_time()
            self.soft_watchdog.feed()
```

`auto_create_handle_loop()` 在主入口和控制面 API 中都会被调用；这是为了保证只要有人等待回复，后台收包循环就已经启动。

## 步骤 10：`_handle_batch_output` 先构造 meta，再按输出类型写 state

系统压力：后端输出是 batch 形态，但 HTTP 等待的是单个 rid；同一条路径还要处理 logprob、hidden states、routed experts、metrics、embedding。

源码选择：循环 `recv_obj.rids`，用 `rid_to_state.get(rid)` 找状态，构造 `meta_info`，再按 `BatchStrOutput` / `BatchTokenIDOutput` / `BatchEmbeddingOutput` 生成 `out_dict`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1862-L1895
    async def _handle_batch_output(
        self,
        recv_obj: Union[
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
        ],
    ):
        recv_obj.time_stats = unwrap_from_pickle(recv_obj.time_stats)
        if isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput)):
            customized_info = unwrap_from_pickle(recv_obj.customized_info)
        else:
            customized_info = None
        pending_notify: dict[str, ReqState] = {}
        batch_notify_size = self.server_args.batch_notify_size
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                # Known race: /health_generate pops its rid as soon as ANY message bumps last_receive_tstamp.
                if rid.startswith(HEALTH_CHECK_RID_PREFIX):
                    continue
                logger.error(
                    f"Received output for {rid=} but the state was deleted in TokenizerManager."
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
                "weight_version": self.server_args.weight_version,
                "num_retractions": recv_obj.retraction_counts[i],
            }
```

`state is None` 不会直接崩溃，因为 abort、health check 或异常清理都可能让回包晚到。这个容错避免了后台收包循环因单个迟到结果退出。

## 步骤 11：文本回包分三种 streaming 语义

系统压力：流式输出既要低延迟，又要避免每步重建完整字符串。incremental 模式要输出 delta；非 incremental streaming 中间包要避免 O(n^2) 字符串拼接。

源码选择：`BatchStrOutput` 分支在 `ReqState` 中累加 text 和 output ids；incremental 直接输出 delta；非 incremental 中间包输出 `text=None`，最后或 `_wait_one_response` 再 materialize。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1970-L2021
            state.finished = recv_obj.finished_reasons[i] is not None
            if isinstance(recv_obj, BatchStrOutput):
                # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
                is_stream = getattr(state.obj, "stream", False)
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream
                )
                delta_text = recv_obj.output_strs[i]
                delta_output_ids = list(recv_obj.output_ids[i])
                output_offset = state.last_output_offset
                state.append_text(delta_text)
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "text": delta_text,
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "text": state.get_text(),
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        # Non-incremental intermediate: pass reference (no
                        # copy) and defer text to _wait_one_response to avoid
                        # O(n) per-step cost that compounds to O(n^2).
                        out_dict = {
                            "text": None,
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
                if out_dict is not None and state.prompt_token_ids is not None:
                    out_dict["prompt_token_ids"] = state.prompt_token_ids
```

看到流式中间包 `text=None` 时，不要先怀疑 Detokenizer 丢文本；它可能是 TokenizerManager 故意延迟 materialize。

## 步骤 12：skip tokenizer 下主回程处理 token ids，不处理字符串

系统压力：`skip_tokenizer_init=True` 意味着主进程没有 tokenizer，不能把 output ids decode 成 text。生成路径仍要能返回 token ids。

源码选择：`BatchTokenIDOutput` 分支累加 `output_ids`，不写 `text`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L2022-L2062
            elif isinstance(recv_obj, BatchTokenIDOutput):
                is_stream = getattr(state.obj, "stream", False)
                incremental = (
                    self.server_args.incremental_streaming_output and is_stream
                )
                delta_output_ids = list(recv_obj.output_ids[i])
                output_offset = state.last_output_offset
                state.output_ids.extend(delta_output_ids)

                if is_stream:
                    if incremental:
                        output_token_ids = delta_output_ids
                        _slice_streaming_output_meta_info(
                            meta_info,
                            output_offset,
                            state.customized_info_accumulated.keys(),
                        )
                        state.last_output_offset = len(state.output_ids)
                        out_dict = {
                            "output_ids": output_token_ids,
                            "meta_info": meta_info,
                        }
                    elif state.finished:
                        out_dict = {
                            "output_ids": state.output_ids.copy(),
                            "meta_info": meta_info,
                        }
                    else:
                        out_dict = {
                            "output_ids": state.output_ids,
                            "meta_info": meta_info,
                        }
                elif state.finished:
                    out_dict = {
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = None
                if out_dict is not None and state.prompt_token_ids is not None:
                    out_dict["prompt_token_ids"] = state.prompt_token_ids
```

这和 [[SGLang-Detokenizer]] 的结论一致：skip tokenizer 主链路绕过 Detokenizer 文本 decode，TokenizerManager 收到的是 token id 输出。

## 步骤 13：完成时删除 state，未完成时只入队并延迟唤醒

系统压力：一个大批回包可能同时唤醒许多 HTTP 协程；逐 rid 连续调度会长时间占住后台 loop。完成请求又必须及时释放 state 和 LoRA 引用。

源码选择：finished 时删除 `rid_to_state[rid]`；有 `out_dict` 时写 `state.out_list`，处理同一个批回包时每累计 `batch_notify_size` 个 rid 就批量 `event.set()` 并 `sleep(0)`，遍历结束后立即通知余数。这个参数控制大 batch 内的 event-loop 公平性，不会跨多个后端消息故意攒 token。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L2080-L2126
            # Set first_token_time on the first output batch.
            # This is the single write point for first_token_time.
            if state.time_stats.first_token_time == 0.0:
                state.time_stats.set_first_token_time()

            if state.finished:
                if state.time_stats.trace_ctx.tracing_enable:
                    state.time_stats.trace_ctx.trace_set_root_attrs(
                        self.convert_to_span_attrs(state, recv_obj, i)
                    )
                state.time_stats.set_finished_time()
                meta_info["e2e_latency"] = state.time_stats.get_e2e_latency()

                if self.server_args.speculative_algorithm:
                    self._calculate_spec_decoding_metrics(meta_info, recv_obj, i)
                if self.enable_metrics:
                    scheduler_time_stats = (
                        recv_obj.time_stats[i]
                        if recv_obj.time_stats is not None
                        else None
                    )
                    completion_tokens = (
                        recv_obj.completion_tokens[i]
                        if not isinstance(recv_obj, BatchEmbeddingOutput)
                        else 0
                    )
                    meta_info.update(
                        state.time_stats.convert_to_output_meta_info(
                            scheduler_time_stats, completion_tokens
                        )
                    )

                del self.rid_to_state[rid]

                # Mark ongoing LoRA request as finished.
                if self.server_args.enable_lora and state.obj.lora_path:
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))

            if out_dict is not None:
                state.out_list.append(out_dict)
                pending_notify[rid] = state

                if len(pending_notify) >= batch_notify_size:
                    for s in pending_notify.values():
                        s.event.set()
                    pending_notify = {}
                    await asyncio.sleep(0)
```

这里有一个细节：finished 时 state 已从 `rid_to_state` 删除，但局部变量 `state` 仍可写 `out_list` 并唤醒前台等待者。前台拿到最后一个 out 后结束 generator。

## 步骤 14：前台等待 event，drain pending outputs，再决定 yield 或结束

系统压力：后台可能在前台醒来前已经积累多个 streaming chunk；如果只取一个，token ids 或 delta 会丢。

源码选择：`_wait_one_response` 等 event，原子 drain `out_list`；incremental streaming 多 chunk 时 coalesce；非 incremental streaming 的 `text=None` 在这里转成完整 text。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1446-L1503
    async def _wait_one_response(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        """Wait for the response of one request."""
        state = self.rid_to_state[obj.rid]
        # Not all request types have `stream` (e.g., EmbeddingReqInput). Default to non-streaming.
        is_stream = getattr(obj, "stream", False)
        while True:
            try:
                await asyncio.wait_for(
                    state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT
                )
            except asyncio.TimeoutError:
                if (
                    request is not None
                    and not obj.background
                    and await request.is_disconnected()
                ):
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 1). Abort request {obj.rid=}"
                    )
                continue

            # Drain all pending outputs atomically.
            out_list = state.out_list
            state.out_list = []
            finished = state.finished
            state.event.clear()

            # With incremental streaming, each chunk is a delta — coalesce
            # multiple queued chunks to avoid dropping token ids.
            incremental_stream = (
                is_stream and self.server_args.incremental_streaming_output
            )
            if incremental_stream and len(out_list) > 1:
                out = self._coalesce_streaming_chunks(
                    out_list,
                    obj.rid,
                    state.customized_info_accumulated.keys(),
                )
            else:
                out = out_list[-1]

            # Resolve deferred text for non-incremental streaming.
            # _handle_batch_output sets "text": None on intermediate chunks
            # to avoid O(n) string rebuild per step (O(n^2) total).
            if (
                is_stream
                and not incremental_stream
                and "text" in out
                and out["text"] is None
            ):
                out["text"] = state.get_text()
```

如果线上 P99 ITL 抖动，同时看到 streaming backlog 警告，优先看 HTTP 消费速度、前台协程是否及时被调度以及 `_coalesce_streaming_chunks`；`batch_notify_size` 只影响单个大批回包处理期间的让出节奏，不能单独解释跨消息 backlog。

## 步骤 15：batch 请求只是多条单请求状态的并发等待

系统压力：API 可以一次传多条 prompt，但每个 rid 的完成时刻不同。非流式可以 gather，流式要按哪个 rid 先来就先 yield 哪个位置的输出。

源码选择：`_handle_batch_request` 为每条子请求创建 generator；非流式 `asyncio.gather`，流式维护 task map。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager.py L1556-L1578
    async def _handle_batch_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
        request: Optional[fastapi.Request] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self._should_use_batch_tokenization(batch_size, obj):
                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                self._send_batch_request(tokenized_objs)

                # Set up generators for each request in the batch
                for i in range(batch_size):
                    tmp_obj = obj[i]
                    state = self.rid_to_state[tmp_obj.rid]
                    if tmp_obj.return_prompt_token_ids:
                        state.prompt_token_ids = list(tokenized_objs[i].input_ids)
                    generators.append(self._wait_one_response(tmp_obj, request))
                    rids.append(tmp_obj.rid)
            else:
```

这段说明 batch 在 TokenizerManager 侧是“多条 ReqState 的组织方式”，不是 Scheduler 的 `ScheduleBatch`。

当 `n > 1` 时还有一层额外生命周期：TokenizerManager 先把每个原始 prompt 以 `max_new_tokens=0` 提交一次，用于缓存共同前缀；随后复制 tokenized object，为每个实际 sample 重新生成 `rid` 并发送。日志、abort 和结果 index 都必须围绕实际 sample rid 解读。

但不能把规范化阶段的 state 概括成“展开后全部删除”。设原始 batch 为 `B`、采样数为 `N`：`_normalize_batch_inputs` 令 `num=B×N`，`_normalize_rid(num)` 和 `_init_req_state(obj)` 因而创建 `B×N` 个 placeholder state；这里的 `objs = [obj[i] for i in range(batch_size)]` 只消费前 `B` 个，最后也只执行 `B` 次 `del self.rid_to_state[objs[i].rid]`。预热与实际 samples 都调用 `regenerate_rid()` 创建另一组 state。当前正常完成路径因此会留下 `B×(N-1)` 个规范化 placeholder state；异常处理里的 `_discard_pending_req_states(obj)` 只在抛异常时运行，不能修复正常路径。文档将其记录为当前基线的 state 生命周期风险，不把尚未在完整服务压测中量化的影响写成确定的 OOM 结论。

## 步骤 16：score API 复用 generate/embedding 数据面

系统压力：score API 需要 logprob 或 embedding 结果，但没必要新建一套 Scheduler 数据面。

源码选择：根据模型类型构造 `GenerateReqInput` 或 `EmbeddingReqInput`，最后调用 `self.generate_request(batch_request, request).__anext__()`。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_manager_score_mixin.py L691-L713
        if is_generation:
            batch_request = GenerateReqInput(
                text=text_prompts,
                input_ids=input_ids,
                token_ids_logprob=label_token_ids,
                return_logprob=True,
                # Set logprob_start_len=0 for multi-item scoring since we want logprobs at all delimiter positions
                logprob_start_len=0 if use_multi_item_scoring else -1,
                stream=False,
                sampling_params={"max_new_tokens": 0},
                positional_embed_overrides=positional_embed_overrides,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )
        else:
            batch_request = EmbeddingReqInput(
                text=text_prompts,
                input_ids=input_ids,
                positional_embed_overrides=positional_embed_overrides,
                return_pooled_hidden_states=return_pooled_hidden_states,
                multi_item_delimiter_indices=mis_delimiter_indices,
            )

        results = await self.generate_request(batch_request, request).__anext__()
```

所以 score 的正确性主要看请求构造和后处理，不是另一条 GPU 执行链路。

## 步骤 17：控制面用 communicator，不按 rid 等普通输出

系统压力：flush cache、权重更新等控制请求需要面向 Scheduler rank fan-out，并等待控制回复；它们不是用户生成请求，没有 `ReqState`。

源码选择：`flush_cache` 只确保后台 loop 启动，然后通过 `flush_cache_communicator` 发控制请求并取第一个回复。

```python
# 来源：sglang/python/sglang/srt/managers/tokenizer_control_mixin.py L256-L262
    async def flush_cache(
        self: TokenizerManager, timeout_s: Optional[float] = None
    ) -> FlushCacheReqOutput:
        self.auto_create_handle_loop()
        return (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
        )[0]
```

权重更新还会使用 `model_update_lock.writer_lock`，与 `generate_request` 的 reader lock 形成互斥。这个边界在排查“请求为什么卡住”时很重要。

## 步骤 18：多 worker router 用 `http_worker_ipc` 拆回包

系统压力：多个 TokenizerWorker 同时发送请求，Scheduler/Detokenizer 回来的 batch 可能混合不同 worker 的 rid。回包必须拆成单条，再发回 owner worker。

源码选择：请求发出时 stamp `http_worker_ipc`；router 回程读取 `BaseReq.http_worker_ipc` 或 `BaseBatchReq.http_worker_ipcs`，用 `_handle_output_by_index` 拆单条。

```python
# 来源：sglang/python/sglang/srt/managers/multi_tokenizer_mixin.py L488-L498
    async def _distribute_result_to_workers(self, recv_obj):
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)
```

这说明多 worker 模式的关键不在“分词更快”本身，而在前后向路由一致性。若同时启用多个 detokenizer worker，Scheduler 输出还会先经 `MultiDetokenizerRouter` 按 `http_worker_ipc` 稳定哈希；detokenizer worker 完成 decode 后直接把结果送回 owner tokenizer worker，以保证同一请求的 decode 状态不跨进程漂移。

## 运行验证

最小验证可以从两个现象入手：

| 验证 | 操作 | 预期 |
|------|------|------|
| 非流式请求生命周期 | 在 `generate_request`、`_init_req_state`、`_handle_batch_output`、`_wait_one_response` 加断点或日志 | 同一个 rid 先进入 `rid_to_state`，后端 finished 后被删除，最后 `_wait_one_response` yield 一次 |
| incremental streaming | 启动时打开 `--incremental-streaming-output`，发送 `stream=True` 请求 | 每个 SSE chunk 的 `text` 是 delta；若消费慢，多个 queued chunks 会在 `_coalesce_streaming_chunks` 合并 |
| skip tokenizer | 启动 `--skip-tokenizer-init` 后用 text 请求 | `_tokenize_one_request` 抛出要求提供 `input_ids` 的错误；用 `input_ids` 请求时输出主要是 `output_ids` |
| rid 生命周期单测 | `python -m pytest sglang/test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py -q` | abort、正常 finished、dispatch 前失败都清理 state，重复 rid 只在旧 state 仍存活时被拒绝 |
| parallel-sampling state 计数 | 固定 `B=2`，分别发送 `n=1` 与 `n=3`，在请求前后记录 owner worker 的 `len(rid_to_state)`；同时记录规范化 rid、预热 rid、sample rid | `n=1` 完成后回到基线；当前源码的 `n=3` 路径应重点检查是否每轮净增 `2×(3-1)=4` 个 placeholder state |

## 复盘

TokenizerManager 的源码可以压成四个判断：

1. 它是前台状态机，不是同步分词函数。
2. `ReqState` 是前后台协程的唯一共享等待室。
3. 数据面按 `rid` 多路复用，控制面按 communicator 聚合回复。
4. 多 worker、skip tokenizer、incremental streaming 都是同一主线上的配置分叉，而不是独立系统。
