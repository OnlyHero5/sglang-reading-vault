---
type: batch-doc
module: 03-HTTP-Server
batch: "03"
doc_type: walkthrough
title: "HTTP Server 入口 · 源码走读"
tags:
 - sglang/batch/03
 - sglang/module/http-server
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# HTTP Server 入口 · 源码走读

> 按调用顺序精读类与函数。
> 走读顺序：`engine.py` 子进程启动 → `engine.py` Python API → `http_server.py` 启动与路由

---

## 1. engine.py — 子进程启动链

### 1.1 `init_tokenizer_manager`

**Explain：** 在 Scheduler 就绪、Detokenizer 已启动后，主进程创建 `TokenizerManager` 与 `TemplateManager`，并根据 chat template 自动解析 `reasoning_parser` / `tool_call_parser`（当值为 `"auto"` 时）。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/engine.py L135-L180
# 提交版本：70df09b
def init_tokenizer_manager(
    server_args: ServerArgs,
    port_args: PortArgs,
    TokenizerManagerClass: Optional[TokenizerManager] = None,
) -> Tuple[TokenizerManager, TemplateManager]:
    # Launch tokenizer process
    TokenizerManagerClass = TokenizerManagerClass or TokenizerManager
    tokenizer_manager = TokenizerManagerClass(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    # Resolve any remaining auto parsers using template manager's detection results
    for attr, suggested, label in (
        (
            "reasoning_parser",
            template_manager.suggested_reasoning_parser,
            "reasoning parser",
        ),
        (
            "tool_call_parser",
            template_manager.suggested_tool_call_parser,
            "tool-call parser",
        ),
    ):
        if getattr(server_args, attr) != "auto":
            continue
        if suggested is not None:
            setattr(server_args, attr, suggested)
            logger.info(
                f"Auto-detected --{attr.replace('_', '-')} as '{suggested}' from chat template"
            )
        else:
            logger.warning(
                f"--{attr.replace('_', '-')}=auto specified but could not detect "
                f"{label} from chat template. Disabling {label}."
            )
            setattr(server_args, attr, None)

    return tokenizer_manager, template_manager
```

**Comment：** `TokenizerManagerClass` 可注入，便于 RayEngine 等子类替换实现。

---

### 1.2 `_launch_subprocesses` 主流程

**Explain：** 引擎启动的「总控」：配置环境 → 分配端口 → 启动 Scheduler →（node_rank==0）启动 Detokenizer → 初始化 Tokenizer → 等待模型加载 → 启动 SubprocessWatchdog。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/engine.py L782-L908
# 提交版本：70df09b
        # Configure global environment
        configure_logger(server_args)
        _set_envs_and_config(server_args)

        # Defensive: ensure plugins loaded (may already be loaded by
        # Engine.__init__ or CLI entry).
        load_plugins()

        server_args.check_server_args()
        _set_gc(server_args)

        # Allocate ports for inter-process communications
        if port_args is None:
            port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

        # Start the engine info bootstrap server if per-rank info is needed.
        engine_info_bootstrap_server = None
        if (
            server_args.remote_instance_weight_loader_start_seed_via_transfer_engine
            and server_args.node_rank == 0
        ):
            bootstrap_port = server_args.engine_info_bootstrap_port
            if not is_port_available(bootstrap_port):
                raise RuntimeError(
                    f"engine_info_bootstrap_port {bootstrap_port} is already in use. "
                    f"When running multiple instances on the same node, each instance must use a "
                    f"different --engine-info-bootstrap-port."
                )
            engine_info_bootstrap_server = EngineInfoBootstrapServer(
                host=server_args.host, port=bootstrap_port
            )

        if (
            server_args.reasoning_parser == "auto"
            or server_args.tool_call_parser == "auto"
        ):
            resolve_auto_parsers(server_args)

        # Launch scheduler processes
        scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
            server_args, port_args, run_scheduler_process_func
        )
        scheduler_init_result.engine_info_bootstrap_server = (
            engine_info_bootstrap_server
        )

        if (
            server_args.enable_elastic_expert_backup
            and server_args.elastic_ep_backend is not None
        ):
            run_expert_backup_manager(server_args, port_args)

        if server_args.node_rank >= 1:
            # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
            # so they can just wait here.
            scheduler_init_result.wait_for_ready()

            if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
                # When using `Engine` as a Python API, we don't want to block here.
                return (
                    None,
                    None,
                    port_args,
                    scheduler_init_result,
                    None,
                )

            launch_dummy_health_check_server(
                server_args.host, server_args.port, server_args.enable_metrics
            )

            scheduler_init_result.wait_for_completion()
            return (
                None,
                None,
                port_args,
                scheduler_init_result,
                None,
            )

        # Launch detokenizer process(es) — optionally fronted by a router when
        # detokenizer_worker_num > 1.
        detoken_procs, detoken_names = cls._launch_detokenizer_subprocesses(
            server_args=server_args,
            port_args=port_args,
            run_detokenizer_process_func=run_detokenizer_process_func,
        )
        for p in detoken_procs:
            scheduler_init_result.all_child_pids.append(p.pid)

        # Init tokenizer manager first, as the bootstrap server is initialized here
        if server_args.tokenizer_worker_num == 1:
            tokenizer_manager, template_manager = init_tokenizer_manager_func(
                server_args, port_args
            )
        else:
            # Launch multi-tokenizer router
            tokenizer_manager = MultiTokenizerRouter(server_args, port_args)
            template_manager = None

        # Wait for the model to finish loading
        scheduler_init_result.wait_for_ready()

        # Get back some info from scheduler to tokenizer_manager
        tokenizer_manager.max_req_input_len = scheduler_init_result.scheduler_infos[0][
            "max_req_input_len"
        ]

        # Set up subprocess liveness watchdog to detect crashes
        # Note: RayEngine returns scheduler_procs=None as it uses Ray actors instead of mp.Process
        processes = list(scheduler_procs or [])
        names = [f"scheduler_{i}" for i in range(len(processes))]
        processes.extend(detoken_procs)
        names.extend(detoken_names)
        subprocess_watchdog = SubprocessWatchdog(
            processes=processes, process_names=names
        )
        subprocess_watchdog.start()

        return (
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result,
            subprocess_watchdog,
        )
```

**Comment：**

- `wait_for_ready()` 阻塞直到每个 Scheduler 通过 Pipe 上报 `status: ready`（含 `max_req_input_len` 等）。
- `node_rank >= 1` 的 worker 节点不跑 tokenizer/detokenizer，只跑 Scheduler 并挂 dummy health server（多机场景）。

---

### 1.3 `_launch_scheduler_processes` — TP / DP 分支

**Explain：** `dp_size == 1` 时按 PP×TP 网格 fork 多个 Scheduler；`dp_size > 1` 时只启动一个 DataParallelController，由它管理多个 DP rank 的 Scheduler。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/engine.py L607-L673
# 提交版本：70df09b
        if server_args.dp_size == 1:
            # Launch tensor parallel scheduler processes
            memory_saver_adapter = TorchMemorySaverAdapter.create(
                enable=server_args.enable_memory_saver
            )
            scheduler_pipe_readers = []

            pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
                _calculate_rank_ranges(
                    server_args.nnodes,
                    server_args.pp_size,
                    server_args.tp_size,
                    server_args.node_rank,
                )
            )

            for pp_rank in pp_rank_range:
                for tp_rank in tp_rank_range:
                    reader, writer = mp.Pipe(duplex=False)
                    gpu_id = (
                        server_args.base_gpu_id
                        + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                        + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                    )
                    attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                        server_args, tp_rank
                    )

                    with maybe_reindex_device_id(gpu_id) as gpu_id:
                        proc = mp.Process(
                            target=run_scheduler_process_func,
                            args=(
                                server_args,
                                port_args,
                                gpu_id,
                                tp_rank,
                                attn_cp_rank,
                                moe_dp_rank,
                                moe_ep_rank,
                                pp_rank,
                                None,
                                writer,
                            ),
                        )
                        with (
                            memory_saver_adapter.configure_subprocess(),
                            numa_utils.configure_subprocess(server_args, gpu_id),
                        ):
                            proc.start()

                    scheduler_procs.append(proc)
                    scheduler_pipe_readers.append(reader)
        else:
            # Launch the data parallel controller
            reader, writer = mp.Pipe(duplex=False)
            scheduler_pipe_readers = [reader]
            proc = mp.Process(
                target=run_data_parallel_controller_process,
                kwargs=dict(
                    server_args=server_args,
                    port_args=port_args,
                    pipe_writer=writer,
                    run_scheduler_process_func=run_scheduler_process_func,
                ),
            )
            proc.start()
            scheduler_procs.append(proc)
```

**Comment：** 每个 Scheduler 子进程绑定独立 `gpu_id`；Pipe 仅用于初始化握手，运行时走 ZMQ。

---

### 1.4 `_wait_for_scheduler_ready` — 防 OOM 挂死

**Explain：** 用 `poll(timeout=5)` 代替阻塞 `recv()`，若子进程被 OOM killer 杀掉，主进程能抛出带 exit code 的 `RuntimeError`，而不是永久卡住。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/engine.py L1368-L1397
# 提交版本：70df09b
def _wait_for_scheduler_ready(
    scheduler_pipe_readers: List,
    scheduler_procs: List,
) -> List[Dict]:
    """Wait for the model to finish loading and return scheduler infos.

    Uses poll() with timeout instead of blocking recv(), so that child process
    death (e.g. OOM SIGKILL) is detected promptly instead of hanging forever.
    """
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        while True:
            if scheduler_pipe_readers[i].poll(timeout=5.0):
                try:
                    data = scheduler_pipe_readers[i].recv()
                except EOFError:
                    raise _scheduler_died_error(i, scheduler_procs[i])
                if data["status"] != "ready":
                    raise RuntimeError(
                        "Initialization failed. Please see the error messages above."
                    )
                scheduler_infos.append(data)
                break

            # Poll timed out — check all processes for early death
            for j in range(len(scheduler_procs)):
                if not scheduler_procs[j].is_alive():
                    raise _scheduler_died_error(j, scheduler_procs[j])

    return scheduler_infos
```

---

### 1.5 `Engine.generate` — Python API 路径

**Explain：** 构造 `GenerateReqInput`，调用 `tokenizer_manager.generate_request`；同步 API 用 `loop.run_until_complete` 消费 async generator 的第一块（或 stream 包装为同步 iterator）。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/engine.py L366-L415
# 提交版本：70df09b
        routed_dp_rank = self._resolve_routed_dp_rank(
            routed_dp_rank, data_parallel_rank
        )

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            audio_data=audio_data,
            video_data=video_data,
            mm_hashes=mm_hashes,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            custom_logit_processor=custom_logit_processor,
            require_reasoning=require_reasoning,
            return_hidden_states=return_hidden_states,
            return_routed_experts=return_routed_experts,
            routed_experts_start_len=routed_experts_start_len,
            stream=stream,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
            routed_dp_rank=routed_dp_rank,
            disagg_prefill_dp_rank=disagg_prefill_dp_rank,
            external_trace_header=external_trace_header,
            rid=rid,
            session_id=session_id,
            session_params=session_params,
            priority=priority,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream:

            def generator_wrapper():
                while True:
                    try:
                        chunk = self.loop.run_until_complete(generator.__anext__())
                        yield chunk
                    except StopAsyncIteration:
                        break

            return generator_wrapper()
        else:
            ret = self.loop.run_until_complete(generator.__anext__())
            return ret
```

**Comment：** HTTP `/generate` 与此处逻辑同源，只是 HTTP 侧 `request` 参数用于 abort / header override。

---

## 2. http_server.py — HTTP 服务启动

### 2.1 `_setup_and_run_http_server`

**Explain：** 设置全局状态、可选 Prometheus 中间件、API Key 鉴权、单/多 tokenizer 模式分支，最后选择 uvicorn / Granian / SSL refresh 之一阻塞运行。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2281-L2327
# 提交版本：70df09b
    # Set global states
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            template_manager=template_manager,
            scheduler_info=scheduler_infos[0],
        )
    )

    # Store watchdog on tokenizer_manager (single source of truth for SIGQUIT handler)
    if tokenizer_manager is not None:
        tokenizer_manager._subprocess_watchdog = subprocess_watchdog

    if server_args.enable_metrics:
        add_prometheus_track_response_middleware(app)

    # Pass additional arguments to the lifespan function.
    # They will be used for additional initialization setups.
    if server_args.tokenizer_worker_num == 1:
        # If it is single tokenizer mode, we can pass the arguments by attributes of the app object.
        app.is_single_tokenizer_mode = True
        app.server_args = server_args
        app.warmup_thread_kwargs = dict(
            server_args=server_args,
            launch_callback=launch_callback,
            execute_warmup_func=execute_warmup_func,
        )

        # Add api key authorization
        # This is only supported in single tokenizer mode.
        #
        # Backward compatibility:
        # - api_key only: behavior matches legacy (all endpoints require api_key)
        # - no keys: legacy had no restriction; ADMIN_FORCE endpoints must still be rejected when
        #   admin_api_key is not configured.
        if (
            server_args.api_key
            or server_args.admin_api_key
            or app_has_admin_force_endpoints(app)
        ):
            from sglang.srt.utils.auth import add_api_key_middleware

            add_api_key_middleware(
                app,
                api_key=server_args.api_key,
                admin_api_key=server_args.admin_api_key,
            )
```

**Comment：** 多 worker 时 uvicorn 用字符串 `"sglang.srt.entrypoints.http_server:app"` 导入 app，各 worker 在 `lifespan` 里 `read_from_shared_memory` 重建 TokenizerWorker。

---

### 2.2 FastAPI `app` 与 `lifespan`

**Explain：** 创建 FastAPI 实例、CORS、可选请求解压中间件；`lifespan` 在 worker 启动时注册 Serving handler 并后台线程跑 warmup。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L261-L291, L394-L417
# 提交版本：70df09b
@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):
    if getattr(fast_api_app, "is_single_tokenizer_mode", False):
        server_args = fast_api_app.server_args
        warmup_thread_kwargs = fast_api_app.warmup_thread_kwargs
        thread_label = "Tokenizer"
    else:
        # Initialize multi-tokenizer support for worker processes
        server_args = await init_multi_tokenizer()
        warmup_thread_kwargs = dict(server_args=server_args)
        thread_label = f"MultiTokenizer-{_global_state.tokenizer_manager.worker_id}"

    # Add prometheus middleware
    if server_args.enable_metrics:
        add_prometheus_middleware(app)
        enable_func_timer()

    # Init tracing
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill" + thread_label
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode" + thread_label
        trace_set_thread_info(thread_label)

    # Initialize OpenAI serving handlers
```

**Comment：** warmup 在独立线程执行，不阻塞 FastAPI 开始监听；就绪前 `ServerStatus` 可能仍为 Starting，`/health` 返回 503。

---

### 2.3 `_wait_and_warmup` 与 `_execute_server_warmup`

**Explain：** 可选等待 checkpoint IPC 权重就绪 → 向自身 HTTP 发 POST（`/generate` 或 `/v1/chat/completions`）→ 成功后 `ServerStatus.Up` 并打印 "ready to roll"。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2145-L2161
# 提交版本：70df09b
def _wait_and_warmup(
    server_args: ServerArgs,
    launch_callback: Optional[Callable[[], None]] = None,
    execute_warmup_func: Callable = _execute_server_warmup,
):
    if server_args.checkpoint_engine_wait_weights_before_ready:
        _wait_weights_ready()

    # Send a warmup request
    if not server_args.skip_server_warmup:
        if not execute_warmup_func(server_args):
            return
    else:
        _global_state.tokenizer_manager.server_status = ServerStatus.Up

    # The server is ready for requests
    logger.info("The server is fired up and ready to roll!")
```

**Code（warmup 探测 `/model_info`）：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1992-L2010
# 提交版本：70df09b
    # Wait until the server is launched
    success = False
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(
                url + "/model_info", timeout=5, headers=headers, verify=ssl_verify
            )
            assert res.status_code == 200, f"{res=}, {res.text=}"
            success = True
            break
        except (AssertionError, requests.exceptions.RequestException):
            last_traceback = get_exception_traceback()
            pass

    if not success:
        logger.error(f"Initialization failed. warmup error: {last_traceback}")
        kill_process_tree(os.getpid())
        return success
```

---

## 3. http_server.py — 核心 HTTP 路由

### 3.1 `POST /generate` — Native 生成 API

**Explain：** FastAPI 把 JSON body 反序列化为 `GenerateReqInput`；`stream=True` 时返回 SSE（`data: {...}\n\n`），并在 `StreamingResponse.background` 注册 abort task。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L785-L835
# 提交版本：70df09b
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
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
                    yield b"data: " + dumps_json(out) + b"\n\n"
            except ValueError as e:
                # A client disconnect also surfaces here. It's a client-side
                # cancellation, not a server error or bad input -- log it and
                # stop (the request was already aborted upstream) instead of
                # emitting a 400.
                if request is not None and await request.is_disconnected():
                    logger.info(f"[http_server] Client disconnected: {e}")
                    return
                out = {
                    "error": {
                        "message": str(e),
                        "type": "invalid_request_error",
                        "code": 400,
                        "retryable": False,
                    }
                }
                logger.error(f"[http_server] Error: {e}")
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
            background=_global_state.tokenizer_manager.create_abort_task(obj),
        )
    else:
        try:
            ret = await _global_state.tokenizer_manager.generate_request(
                obj, request
            ).__anext__()
            return orjson_response(ret)
        except ValueError as e:
            logger.error(f"[http_server] Error: {e}")
            return _create_error_response(e)
```

**Comment：** 这是追踪「第一个 token」的关键一跳；后续 TokenizerManager → Scheduler → Detokenizer 在TokenizerManager–Detokenizer。

---

### 3.2 `/health` 与 `/health_generate`

**Explain：** 轻量探活：可选只返回 200；`health_generate` 会发 `max_new_tokens=1` 的探测请求，若在超时内收到 detokenizer 心跳则判健康。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L588-L637
# 提交版本：70df09b
    if _global_state.tokenizer_manager.gracefully_exit:
        logger.info("Health check request received during shutdown. Returning 503.")
        return Response(status_code=503)

    if _global_state.tokenizer_manager.server_status == ServerStatus.Starting:
        return Response(status_code=503)

    if (
        not envs.SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION.get()
        and request.url.path == "/health"
    ):
        return Response(status_code=200)

    sampling_params = {"max_new_tokens": 1, "temperature": 0.0}
    # uuid keeps rids unique across tokenizer workers (a bare time.time() can
    # collide and crash the shared DetokenizerManager decode_status).
    rid = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

    if _global_state.tokenizer_manager.is_generation:
        gri = GenerateReqInput(
            rid=rid,
            input_ids=[0],
            sampling_params=sampling_params,
            log_metrics=False,
        )
        if (
            _global_state.tokenizer_manager.server_args.disaggregation_mode
            != DisaggregationMode.NULL.value
        ):
            gri.bootstrap_host = FAKE_BOOTSTRAP_HOST
            gri.bootstrap_room = 0
    else:
        gri = EmbeddingReqInput(
            rid=rid, input_ids=[0], sampling_params=sampling_params, log_metrics=False
        )

    async def gen():
        async for _ in _global_state.tokenizer_manager.generate_request(gri, request):
            break

    task = asyncio.create_task(gen())

    # As long as we receive any response from the detokenizer/scheduler, we consider the server is healthy.
    tic = time.time()
    while time.time() < tic + HEALTH_CHECK_TIMEOUT:
        await asyncio.sleep(1)
        if _global_state.tokenizer_manager.last_receive_tstamp > tic:
            task.cancel()
            _global_state.tokenizer_manager.rid_to_state.pop(rid, None)
            _global_state.tokenizer_manager.server_status = ServerStatus.Up
```

---

### 3.3 OpenAI 路由委托（预览）

**Explain：** `/v1/chat/completions` 等不在本文件实现业务逻辑，只转发到 `lifespan` 里创建的 Serving 对象；OpenAI API 专讲。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1606-L1613
# 提交版本：70df09b
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

---

## 4. 走读小结

| 步骤 | 函数 | 文件 |
|------|------|------|
| 1 | `launch_server` | http_server.py |
| 2 | `Engine._launch_subprocesses` | engine.py |
| 3 | `_launch_scheduler_processes` + `_launch_detokenizer_subprocesses` | engine.py |
| 4 | `init_tokenizer_manager` | engine.py |
| 5 | `_setup_and_run_http_server` | http_server.py |
| 6 | `lifespan` → Serving init + warmup 线程 | http_server.py |
| 7 | `uvicorn.run(app)` / Granian | http_server.py |
| 8 | `generate_request` → `tokenizer_manager.generate_request` | http_server.py |
