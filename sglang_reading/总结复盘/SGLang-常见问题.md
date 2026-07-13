---
title: "SGLang 常见问题"
type: troubleshooting
framework: sglang
topic: "总结复盘"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# SGLang 常见问题

> 这不是术语词典，而是“第一跳分诊台”：先确认问题属于入口、前台、调度、执行、缓存还是回程，再进入对应专题。

## 读者任务

读完后，你应该能回答三类方向性问题：

- 一条命令究竟进入 LLM、diffusion、HTTP、legacy gRPC、Ray 还是 encoder-only 路径。
- 当前看到的“服务已启动”究竟是 socket 可连接、Scheduler 已加载、warmup 已通过，还是生成链路健康。
- 性能或正确性问题应从 HTTP handler、TokenizerManager、Scheduler、ModelRunner、RadixCache 还是 Detokenizer 开始查。

先记住一条总原则：SGLang 有“推荐入口”，没有一个覆盖所有部署形态、所有进程和所有 API 的单一固定拓扑。

## Q1：第一次读 SGLang，主线从哪里开始？

先读 [[SGLang-导读与总览]] 建立请求生命周期，再读 [[SGLang-业务流程]]；随后按问题进入 [[SGLang-启动与入口]]、[[SGLang-请求调度]]、[[SGLang-模型执行]]、[[SGLang-内存与Attention]]。源码地图用于定位文件，不应替代一条真实请求主线。

CLI 主线从 `sglang serve` 开始，但它先加载插件、解析 `--model-type`，再探测 LLM 或 diffusion；LLM 分支才构造 `ServerArgs` 并进入 `run_server`。

```python
# 来源：sglang/python/sglang/cli/serve.py L89-L99
    from sglang.srt.plugins import load_plugins

    load_plugins()

    model_type, dispatch_argv = _extract_model_type_override(extra_argv)
    model_path = get_model_path(dispatch_argv)
    try:
        if model_type == "auto":
            is_diffusion_model = get_is_diffusion_model(model_path)
            if is_diffusion_model:
                logger.info("Diffusion model detected")
```

```python
# 来源：sglang/python/sglang/cli/serve.py L121-L128
        else:
            # Logic for Standard Language Models
            from sglang.launch_server import run_server
            from sglang.srt.server_args import prepare_server_args

            server_args = prepare_server_args(dispatch_argv)

            run_server(server_args)
```

边界：`sglang serve` 是推荐 CLI，不是唯一入口。`python -m sglang.launch_server` 仍受支持，Python 用户还可以直接构造 `Engine`；所以不要把“推荐入口”写成“所有服务进程必经这里”。

## Q2：LLM 启动参数最终如何选择服务形态？

`run_server` 是 LLM CLI 的一级分流器，优先级不是简单 HTTP/gRPC 二选一：

1. `encoder_only` 先分 encoder HTTP 或 encoder gRPC。
2. 普通模型且 `grpc_mode` 时走 legacy gRPC wrapper。
3. `use_ray` 走 Ray HTTP server。
4. 其余才是默认 HTTP server。

```python
# 来源：sglang/python/sglang/launch_server.py L15-L51
def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        # For encoder disaggregation
        if server_args.grpc_mode:
            from sglang.srt.disaggregation.encode_grpc_server import (
                serve_grpc_encoder,
            )

            asyncio.run(serve_grpc_encoder(server_args))
        else:
            from sglang.srt.disaggregation.encode_server import launch_server

            launch_server(server_args)
    elif server_args.grpc_mode:
        # TODO: Once the native Rust gRPC server starts alongside HTTP in the
        # default path below (controlled by SGLANG_ENABLE_GRPC / SGLANG_GRPC_PORT),
        # remove this legacy SMG path and the grpc_mode flag.
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.use_ray:
        # Ray mode: HTTP mode with Ray backend.
        try:
            from sglang.srt.ray.http_server import launch_server
        except ImportError:
            raise ImportError(
                "Ray is required for --use-ray mode. "
                "Install it with: pip install 'sglang[ray]'"
            )

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)
```

排障操作：启动前把 `encoder_only/grpc_mode/use_ray` 三个最终值记录下来。预期：只有一个一级分支命中；如果你在默认 HTTP 日志里寻找 legacy gRPC 生命周期，方向已经错了。

## Q3：`--grpc-mode` 与 native Rust gRPC 是同一条链吗？

不是。本基线中 `--grpc-mode` 调到 Python wrapper，再委托已安装的 `smg-grpc-servicer`。wrapper 还尝试在 request manager ready 后启动 aiohttp sidecar，提供 metrics/profile；旧版外部包不支持 hook 时，核心 gRPC 可以运行，但 sidecar 能力不同。

```python
# 来源：sglang/python/sglang/srt/entrypoints/grpc_server.py L156-L166
async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler."""
    try:
        from smg_grpc_servicer.sglang.server import serve_grpc as _serve_grpc
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e
```

```python
# 来源：sglang/python/sglang/srt/entrypoints/grpc_server.py L230-L254
    sidecar_supported = (
        "on_request_manager_ready" in inspect.signature(_serve_grpc).parameters
    )
    if sidecar_supported:
        serve_kwargs["on_request_manager_ready"] = _on_request_manager_ready
    elif server_args.enable_metrics:
        # User explicitly asked for metrics but the installed servicer can't
        # start the sidecar that serves them — fail loud rather than silently
        # produce a server with no /metrics endpoint.
        raise RuntimeError(
            "--enable-metrics requires smg-grpc-servicer ≥ 0.5.3 (the version "
            "that accepts 'on_request_manager_ready'); installed version "
            "lacks the hook so the HTTP sidecar would never start. Upgrade "
            "smg-grpc-servicer or remove --enable-metrics."
        )
    else:
        logger.warning(
            "Installed smg-grpc-servicer does not accept "
            "'on_request_manager_ready'; HTTP sidecar disabled "
            "(no /metrics, /start_profile, /stop_profile). "
            "Upgrade smg-grpc-servicer to ≥ 0.5.3 to enable it."
        )

    try:
        await _serve_grpc(server_args, model_info, **serve_kwargs)
```

源码中同时存在 native Rust/Tonic 相关实现，不等于默认 HTTP 已自动并行启动它。要证明 native listener 已接线，必须找到当前宿主对其启动函数的真实调用，而不能只看环境变量或实现文件存在。

## Q4：HTTP 服务固定是“主进程 + Scheduler + Detokenizer”三个进程吗？

只有最简配置可以这样画。更准确的对象所有权是：rank 0 的 HTTP/Engine 前台持有 TokenizerManager 或 MultiTokenizerRouter；Scheduler 可能是一组 TP/PP 进程，也可能先启动 DP controller；Detokenizer 可以是一进程，也可以是多个 worker 加 router。多节点非零 rank 甚至不启动 tokenizer/detokenizer。

```python
# 来源：sglang/python/sglang/srt/entrypoints/engine.py L763-L781
    def _launch_subprocesses(
        cls,
        server_args: ServerArgs,
        init_tokenizer_manager_func: Callable,
        run_scheduler_process_func: Callable,
        run_detokenizer_process_func: Callable,
        port_args: Optional[PortArgs] = None,
    ) -> Tuple[
        TokenizerManager,
        TemplateManager,
        PortArgs,
        SchedulerInitResult,
        Optional[SubprocessWatchdog],
    ]:
        """Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.

        Returns:
            Tuple of (tokenizer_manager, template_manager, port_args, scheduler_init_result, subprocess_watchdog).
        """
```

```python
# 来源：sglang/python/sglang/srt/entrypoints/engine.py L835-L884
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
```

排障操作：记录 `node_rank/dp_size/tp_size/pp_size/tokenizer_worker_num/detokenizer_worker_num`，再列实际 PID 和角色。预期：拓扑能由这些配置解释；不要拿默认三进程图判断多 DP、多 tokenizer 或多 detokenizer 环境“多出了异常进程”。

## Q5：端口能访问，是否等于模型 ready？

不等于。HTTP lifespan 会启动后台 warmup thread 后就 `yield` 给 ASGI server；`/model_info` 可用只是 warmup 的第一道探针。普通 warmup请求成功、PD warmup 成功，或显式跳过 warmup后，TokenizerManager 的 `server_status` 才被置为 `Up`。

```python
# 来源：sglang/python/sglang/srt/entrypoints/http_server.py L369-L391
    # Execute custom warmups
    if server_args.warmups is not None:
        await execute_warmups(
            server_args.disaggregation_mode,
            server_args.warmups.split(","),
            _global_state.tokenizer_manager,
        )
        logger.info("Warmup ended")

    # Execute the general warmup
    warmup_thread = threading.Thread(
        target=_wait_and_warmup,
        kwargs=warmup_thread_kwargs,
    )
    warmup_thread.start()

    # Start the HTTP server
    try:
        yield
    finally:
        if tool_server is not None and hasattr(tool_server, "aclose"):
            await tool_server.aclose()
        warmup_thread.join()
```

```python
# 来源：sglang/python/sglang/srt/entrypoints/http_server.py L2145-L2161
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

`/health` 也有两种语义：Starting/退出时返回 503；默认可直接返回 200，只有启用 `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` 或访问 `/health_generate` 才真正发一个请求，并以 Scheduler/Detokenizer 是否有响应判断健康。

操作：部署门禁应至少区分 TCP/HTTP 可达、`server_status=Up`、`/health_generate` 成功三层。预期：只有最后一层证明最小生成/回程链路活着。

## Q6：前缀缓存到底在哪一层生效？

HTTP handler 不做 prefix match。Scheduler 侧的请求/缓存逻辑调用 `RadixCache.match_prefix`，返回已有 KV 的命中视图；ModelRunner 后续只为未命中的 extend 部分执行并写 KV。命中的 key 还受 `extra_key` 隔离，不是 token 前缀相同就必然共享。

```python
# 来源：sglang/python/sglang/srt/mem_cache/radix_cache.py L355-L365
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        """Find the longest cached prefix of ``key`` in the radix tree.

        The logical namespace for prefix matching is determined by both the
        token id sequence and the optional ``extra_key`` carried by ``RadixKey``.
        Entries that share identical leading token ids but have *different*
        ``extra_key`` values are intentionally kept disjoint and never share
        prefix nodes. This is useful to:

        * Isolate KV cache lines for different LoRA / adapter IDs.
        * Separate requests that intentionally should not share state (e.g.,
```

操作：同时记录 token ids、`extra_key`、命中长度、`prefix_indices` 与实际 extend 长度。预期：命中减少需要计算的新 token，但不会绕过 Scheduler 准入，也不会让不同 namespace 错误共享。

## Q7：Continuous Batching 的核心循环在哪里？

入口在 Scheduler，而不是 HTTP 或 ModelRunner。normal loop 每轮收请求、处理输入、组下一批、运行并提交结果；overlap loop 在此基础上把 GPU 执行与 CPU 结果处理错开。`get_next_batch_to_run` 是 waiting/running 状态进入下一次执行批的决策入口，不是一个纯 list concat helper。

```python
# 来源：sglang/python/sglang/srt/managers/scheduler.py L1521-L1540
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Get the next batch to run
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            # Launch the current batch
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
```

操作：把一轮延迟拆成 recv/input、schedule、run、commit/output 四段，并同时看 waiting/running/retracted 数量。预期：只有 `run_batch` 慢才优先进入 ModelRunner；schedule 慢或反复 retract 应留在 Scheduler/KV 容量主线。

## Q8：遇到问题时，第一跳应该去哪个专题？

| 现象 | 第一跳 | 不要先做什么 |
|---|---|---|
| 命令进入错误服务类型 | [[SGLang-启动链路]] | 不要先查 Scheduler |
| HTTP 200 但生成不可用 | [[SGLang-HTTP-Server-排障指南]] | 不要把 `/model_info` 当 readiness |
| OpenAI 字段或流式 chunk 异常 | [[SGLang-OpenAI-API-排障指南]] | 不要直接改 model forward |
| waiting queue 堆积、retract | [[SGLang-Scheduler-排障指南]] | 不要只看 GPU utilization |
| prefix 命中异常、KV 不足 | [[SGLang-RadixAttention-排障指南]]、[[SGLang-KV-Cache-排障指南]] | 不要归因于 HTTP cache |
| token 正确但文本乱码/增量重复 | [[SGLang-Detokenizer-排障指南]] | 不要重跑 top-k sampling |
| Graph/backend/shape 异常 | [[SGLang-ModelRunner-排障指南]]、[[SGLang-Attention-排障指南]] | 不要只凭 backend 名判断 |
| LoRA 加载、slot 或动态更新异常 | [[SGLang-LoRA-排障指南]] | 不要把 adapter 名、内部 id 与 GPU slot 混成一个身份 |
| 图片/视频 token、IPC 或 encoder 异常 | [[SGLang-多模态-排障指南]] | 不要假定 Processor 已生成最终 embedding，或 CUDA IPC 是零复制 |
| 权重热更新后结果不一致 | [[SGLang-CheckpointEngine-排障指南]] | 不要用没有实际生产者的 `num_paused_reqs` 证明请求已暂停 |

如果仍无法定位，先写下“对象、所有者、交接边界、可观测证据”四项，再用 [[SGLang-源码地图]] 找文件。[[SGLang-综合学习检查]] 用能力和实验验收，不以读完多少篇为完成标准。

## Q9：为什么命令行已经设置，运行时仍不是那个 backend 或拓扑？

因为参数会经历多阶段归一化：dataclass 默认值、`__post_init__` 的平台/模型/内存调整、backend compatibility、特性互斥与后续 `check_server_args`。有些组合直接报错，有些会自动改写，有些则保留配置但在更下层 fallback。

排查时保存四份证据：

```text
用户输入参数
→ ServerArgs 最终值
→ resolved runner/backend/对象类型
→ profiler 中的实际 kernel/collective
```

操作：先从启动日志或 `/server_info` 获取最终配置，再在对应专题查 resolver 与 fallback，最后用 profiler/trace 证明执行路径。预期：实验报告不会再用“CLI 写了某值”替代“该路径实际生效”。

## 最小静态复核

```powershell
rg -n 'def serve|def run_server|async def serve_grpc|def launch_server|def _launch_subprocesses|def _wait_and_warmup|def health_generate|def event_loop_normal|def get_next_batch_to_run|def match_prefix|def _handle_attention_backend_compatibility' sglang/python/sglang/cli/serve.py sglang/python/sglang/launch_server.py sglang/python/sglang/srt/entrypoints/grpc_server.py sglang/python/sglang/srt/entrypoints/http_server.py sglang/python/sglang/srt/entrypoints/engine.py sglang/python/sglang/srt/managers/scheduler.py sglang/python/sglang/srt/mem_cache/radix_cache.py sglang/python/sglang/srt/server_args.py
```

预期：命中 CLI 分流、服务形态分流、legacy gRPC 委托、HTTP/Engine 装配、readiness、健康检查、调度循环、prefix match 与 backend compatibility 九类入口；任一入口迁移，都应回到相应专题重新核对，而不是只更新本 FAQ 的行号。
