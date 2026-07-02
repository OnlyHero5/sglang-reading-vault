---
type: batch-doc
module: 29-multimodal_gen
batch: "29"
doc_type: walkthrough
title: "multimodal_gen · 源码走读"
tags:
 - sglang/batch/29
 - sglang/module/multimodal-gen
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# multimodal_gen · 源码走读

> 走读顺序：`launch_server.py` → `run_scheduler_process` → `http_server lifespan` → `scheduler_client` → `GPUWorker` → `PipelineExecutor`

---

## 1. 服务启动

### 1.1 `launch_server` — spawn workers

**Explain：** 主进程创建 master↔slave 单向 Pipe，spawn `num_gpus` 个 worker，等待全部 `ready` 后启动 FastAPI（或 webui 子进程）。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/launch_server.py L86-L97
def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    """
    Args:
        launch_http_server: False for offline local mode
    """
    configure_logger(server_args)

    # Start a new server with multiple worker processes
    logger.info("Starting server...")

    num_gpus = server_args.num_gpus
    processes = []
```

**Comment：**

- `launch_http_server=False` 用于纯离线 ZMQ 模式。
- `kill_process_tree` 在异常退出时清理子进程（psutil）。

### 1.2 Ready 握手

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/launch_server.py L176-L191
    for i, reader in enumerate(scheduler_pipe_readers):
        try:
            data = reader.recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            processes[i].join()
            logger.error(f"Exit code: {processes[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)
```

**Comment：**

- 任一 worker 初始化失败（OOM、模型加载错误）主进程立即 fail fast。
- HTTP 仅在全部 ready 后 bind port，避免半开服务。

### 1.3 `_find_available_port`

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/launch_server.py L29-L44
def _find_available_port(
    start: int = 10000, avoid: set[int] | None = None, max_attempts: int = 100
) -> int:
    """Find an available port starting from *start*, skipping ports in *avoid*."""
    if avoid is None:
        avoid = set()
    port = max(1024, min(start, 65535))
    for _ in range(max_attempts):
        if port not in avoid and is_port_available(port):
            return port
        port += 1
        if port > 65535:
            port = 1024
    raise RuntimeError(
        f"No available port found after {max_attempts} attempts (start={start})"
    )
```

**Comment：**

- scheduler/broker/http 各需独立 port，`avoid` 集合防止冲突。

---

## 2. Worker 进程入口

### 2.1 `run_scheduler_process`

**Explain：** 每个 GPU 进程：配置 logger/arch → 构造 `Scheduler` → 向父进程发 ready → 进入 `event_loop()`。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L1018-L1033
    try:
        scheduler = Scheduler(
            server_args,
            gpu_id=rank,
            port_args=port_args,
            task_pipes_to_slaves=task_pipes_to_slaves,
            result_pipes_from_slaves=result_pipes_from_slaves,
            local_rank=local_rank,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
            }
        )
        scheduler.event_loop()
```

**Comment：**

- Rank 0 的 Scheduler 绑定 ZMQ REP/ROUTER 收 HTTP client 请求。
- 退出时 `destroy_process_group()` 清理 distributed。

### 2.2 `GPUWorker.__init__`

**Explain：** 加载 pipeline 权重、初始化 distributed groups（TP/SP/CFG/Ring/Ulysses）。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L111-L134
    def __init__(
        self,
        local_rank: int,
        rank: int,
        master_port: int,
        server_args: ServerArgs,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.master_port = master_port
        # FIXME: should we use tcp as distribute init method?
        self.server_args = server_args
        self.pipeline: ComposedPipelineBase = None

        self.init_device_and_model()
        self.sp_group = get_sp_group()
        self.sp_cpu_group = self.sp_group.cpu_group
        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group

        self.cfg_group = get_cfg_group()
        self.cfg_cpu_group = self.cfg_group.cpu_group
        self._realtime_sessions = RealtimeSessionCache(max_sessions=1)
        self.memory_occupation: MemoryOccupationController | None = None
```

**Comment：**

- `init_device_and_model` 内部 `build_pipeline` + 权重 load。
- `MemoryOccupationController` 管理 component 级 GPU/CPU 占用。

---

## 3. HTTP 层

### 3.1 FastAPI `lifespan`

**Explain：** App 启动时初始化 async ZMQ client、broker task、可选 server warmup。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L110-L124
    # 1. Initialize the singleton client that connects to the backend Scheduler
    server_args = app.state.server_args
    async_scheduler_client.initialize(server_args)
    warmup_done = asyncio.Event()
    app.state.server_warmup_done = warmup_done

    # 2. Start the ZMQ Broker in the background to handle offline requests
    broker_task = asyncio.create_task(run_zeromq_broker(server_args))
    warmup_task = None
    if server_args.server_warmup:
        warmup_task = asyncio.create_task(
            _run_server_warmup_after_http_ready(server_args, warmup_done)
        )
    else:
        warmup_done.set()
```

**Comment：**

- warmup 通过 loopback HTTP 调 `/health` 再发 synthetic generate。
- shutdown 时 cancel broker 与 close ZMQ socket。

### 3.2 Server warmup 探活

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L60-L76
async def _wait_until_http_ready(server_args: ServerArgs) -> None:
    """for server warmup"""
    health_url = f"{server_args.url()}/health"
    # Probe the local server directly: a loopback readiness check must never be
    # routed through an HTTP proxy. trust_env=False also avoids crashing startup
    # on a malformed proxy env var, since httpx parses *_PROXY/NO_PROXY when the
    # client is constructed (raising httpx.InvalidURL before any request). See #28493.
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(120):
            try:
                response = await client.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise RuntimeError(f"HTTP server did not become ready at {health_url}")
```

**Comment：**

- `trust_env=False` 避免代理 env 污染 localhost 探活（#28493）。
- warmup 失败 SIGTERM 整个进程，避免冷启动首请求超时。

---

## 4. Scheduler Client

### 4.1 同步 `SchedulerClient`

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/scheduler_client.py L60-L78
    def initialize(self, server_args: ServerArgs):
        if self.context is not None and not self.context.closed:
            logger.warning("SchedulerClient is already initialized. Re-initializing.")
            self.close()

        self.server_args = server_args
        self.context = zmq.Context()
        self.scheduler_socket = self.context.socket(zmq.REQ)

        # Set socket options for the main communication socket
        self.scheduler_socket.setsockopt(zmq.LINGER, 0)

        # 100 minute timeout for generation
        self.scheduler_socket.setsockopt(zmq.RCVTIMEO, 6000000)

        scheduler_endpoint = self.server_args.scheduler_endpoint
        self.scheduler_socket.connect(scheduler_endpoint)
        logger.debug(
            f"SchedulerClient connected to backend scheduler at {scheduler_endpoint}"
```

**Comment：**

- RCVTIMEO 100 分钟，适应长视频生成。
- DiffGenerator 等离线工具用同步 client。

### 4.2 Broker 转发

→ 见 01-核心概念 §4；broker 解耦 offline client 与 scheduler endpoint。

---

## 5. Pipeline 执行

### 5.1 `execute_with_profiling`

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L126-L137
    def execute_with_profiling(
        self,
        stages: List["PipelineStage"],
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:

        with self.profile_execution(batch, dump_rank=0):
            with current_platform.inference_mode():
                batch = self.execute(stages, batch, server_args)

        return batch
```

**Comment：**

- `SGLDiffusionProfiler` 记录 stage 级耗时。
- `maybe_nvtx_range` 可选 Nsight 标记。

### 5.2 Stage hooks — residency / offload

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L103-L118
    def _run_stage_with_executor_hooks(
        self,
        stage: "PipelineStage",
        stage_index: int,
        payload: Any,
        server_args: ServerArgs,
        run_stage: Callable[["PipelineStage", Any], Any],
        use_nvtx: bool,
    ) -> Any:
        stage_name = stage._component_stage_name()
        self.before_stage(stage, stage_index, payload, server_args)
        with maybe_nvtx_range(f"stage_{stage_name}", use_nvtx):
            payload = self.run_stage_with_context(
                stage, payload, server_args, run_stage
            )
        return payload
```

**Comment：**

- `dit_cpu_offload` / `text_encoder_cpu_offload` 时 stage 前后迁移权重。
- FSDP inference 需要 `inference_mode(False)` 以更新 version counter。

---

## 6. GPUWorker 推理输出

### 6.1 后处理与保存

**Explain：** forward 完成后 `materialize_output_sample` → `post_process_sample` → `save_outputs` 写 PNG/MP4。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L36-L40
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    materialize_output_sample,
    post_process_sample,
    save_outputs,
)
```

**Comment：**

- `OutputBatch` 封装 tensor、文件路径、metrics。
- realtime 路径走 `RealtimeSessionCache` 流式帧输出。

### 6.2 Realtime session 释放

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L136-L150
    def release_realtime_session(self, session_id: str) -> OutputBatch:
        """release the session of a realtime connection"""
        if not session_id:
            return OutputBatch(
                output={
                    "released": False,
                    "session_id": session_id,
                    "reason": "empty_session_id",
                }
            )

        released = self._realtime_sessions.release(session_id)
        if released:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
```

**Comment：**

- WebRTC/WebSocket realtime API 长连接占用 KV/latent cache。
- release 触发 empty_cache 回收显存。

---

## 7. Disagg 启动

### 7.1 `launch_pool_disagg_server`

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/launch_server.py L247-L249
    logger.info(
        "Starting pool disagg server: %d encoder(s), %d denoiser(s), %d decoder(s)...",
        num_encoders,
```

**Comment：**

- 每组 role 独立 spawn worker 集，GPU ID 列表由调用方指定。
- `DiffusionServer` 维护 role 间 queue 与 load balance。

---

## 8. 进程清理

### 8.1 `kill_process_tree`

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/launch_server.py L47-L69
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
```

**Comment：**

- daemon worker + 主进程退出时需显式 kill tree。
- K8s PID1 场景额外 SIGQUIT。

---

## 9. Req / OutputBatch 调度

**Explain：** `schedule_batch.Req` 封装单次生成请求（prompt、分辨率、steps、seed）；Scheduler 组 batch 后交给 `GPUWorker.run_batch` → pipeline execute。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L15
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
```

**Comment：**

- 与 srt `Req` 同名不同模块，勿混淆。
- warmup 请求带 `is_warmup=True` 跳过部分 NVTX/offload 逻辑。

---

## 10. OpenAI 路由挂载

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L51-L57
VERTEX_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate")
SERVER_WARMUP_BYPASS_PATHS = (
    "/health",
    "/health_generate",
    "/model_info",
    "/server_info",
)
```

**Comment：**

- GCP Vertex 兼容路由通过 env 配置。
- warmup 中间件跳过 health 路径避免死锁。

---

## 11. 分布式初始化

**Explain：** worker 内 `maybe_init_distributed_environment_and_model_parallel` 建立 NCCL process group，供 TP/序列并行 allreduce。

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L19-L25
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_tp_rank,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
```

**Comment：**

- `master_port` 用于 torch.distributed init。
- 单 GPU 时 world_size=1，stage 内仍走统一 executor 接口。

---

## 12. OOM 处理

**Code：**

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L1034-L1036
    except _oom_exceptions() as _e:
        logger.warning(OOM_MSG)
        raise
```

**Comment：**

- OOM 后 worker 进程退出，主进程需重启或 K8s 重建 pod。
- `OFFLOAD_DISABLE_RECOMMENDATION_ORDER` 提示关闭 offload 顺序以换显存。
