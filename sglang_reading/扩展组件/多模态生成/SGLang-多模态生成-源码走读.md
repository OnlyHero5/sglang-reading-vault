---
title: "多模态生成 · 源码走读"
type: walkthrough
framework: sglang
topic: "多模态生成"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/walkthrough
  - source-reading
updated: 2026-07-10
---
# 多模态生成 · 源码走读

> 走读顺序：`launch_server.py` → `run_scheduler_process` → `http_server lifespan` → `scheduler_client` → `GPUWorker` → `PipelineExecutor`

`multimodal_gen` 的运行时和 LLM serving 的核心差异是：一次请求可能穿过 encoder、denoiser、decoder、VAE、后处理和文件输出等多阶段组件。因此源码把职责拆成四层：主进程只负责拉起 worker 和 HTTP；HTTP 只负责协议入口与 warmup gate；scheduler/client 只负责进程间请求转发；GPU worker 与 executor 才真正处理模型、pipeline stage、显存和输出形态。

---

## 长文读法

这篇按 multimodal generation 的服务生命周期读：主进程先 spawn GPU worker 并等待 ready，再开放 HTTP；HTTP 层只做协议入口、warmup gate 和 scheduler client 生命周期；SchedulerClient 负责跨进程请求转发；GPUWorker 持有分布式环境、pipeline、显存管理和 realtime session；PipelineExecutor 才真正按 stage hook 执行多阶段组件。

| 读者任务 | 先读 | 要抓住的判断 |
|----------|------|--------------|
| 第一次建立请求路径 | 顶部走读顺序、1 到 3 | HTTP 必须等所有 worker ready 后才暴露，请求入口和模型执行不在同一进程 |
| 排查启动失败或端口冲突 | 1.1 到 1.5 | 主进程负责 pipe、端口、ready barrier 和进程树清理，任一 worker 未 ready 都阻止服务启动 |
| 排查 GPU worker 初始化 | 2 | worker 内部初始化分布式环境、pipeline 和 scheduler event loop，OOM 会在 worker 入口被捕获并上抛 |
| 排查健康检查和 warmup 卡住 | 3 | warmup 绕过路径、HTTP ready 探活和 synthetic warmup 是独立 gate，失败会终止启动 |
| 理解 HTTP 到 worker 的通信 | 4 | SchedulerClient 是 ZMQ REQ 客户端，同步 / 异步入口都要通过 scheduler endpoint 交换对象 |
| 看多阶段 pipeline 怎么执行 | 5 到 6 | `Req` / `OutputBatch` 是执行边界，stage hook 负责 component residency、profiling 和平台 inference context |
| 排查 realtime 或文件输出 | 6 到 7 | GPUWorker 后处理输出形态、保存文件和释放 realtime session，HTTP 层只接收完成后的结果 |

读的时候把四层边界分开：启动生命周期、HTTP 协议层、进程间转发层、GPU pipeline 执行层。`multimodal_gen` 和 LLM serving 最大的差别就在于 pipeline stage 多，不能用单个 ModelRunner 心智模型套进去。

## 1. 服务启动：先稳定 worker，再暴露 HTTP

### 1.1 launch_server 创建 worker 进程

**问题与约束：** diffusion runtime 可能使用多 GPU，每个 GPU worker 都要完成分布式初始化和 pipeline 构建；如果 HTTP 提前开放，用户请求会打到尚未 ready 的后端。

**设计选择：** 主进程在 `launch_server` 中按 `num_gpus` spawn worker，为 rank0 和 slave rank 准备不同的 pipe 端点，并把 `launch_http_server` 作为离线模式的开关。

**读法：** 启动器的设计是“主进程负责生命周期，worker 负责模型”。父进程不加载模型，只协调 pipe、ready 握手和 HTTP 入口。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L86-L97
def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    configure_logger(server_args)
    logger.info("Starting server...")

    num_gpus = server_args.num_gpus
    processes = []
```

**代码逻辑：** 函数先配置日志和 GPU 数量，再创建 master/slave pipe；每个 rank 通过 `mp.Process` 启动 `run_scheduler_process`；最后根据 `launch_http_server` 决定是否启动 FastAPI。

**为什么这样写：** 父进程不直接持有 CUDA pipeline，可以减少 fork/spawn 后资源混乱；同时离线 ZMQ 模式可以复用同一套 worker 启动逻辑，而不绑定 HTTP。

**不变量与失败模式：** `num_gpus` 决定 worker 数和 pipe 数；如果某个 worker 初始化失败，父进程不能继续启动半可用服务。

**要点：** 这里的“scheduler”不是单独线程，而是每个 GPU worker 内部的 scheduler event loop。

### 1.2 ready 握手

**问题与约束：** worker 可能因为 OOM、模型路径错误、NCCL 初始化失败等原因在启动期退出；主进程需要区分“还没 ready”和“已经死掉”。

**设计选择：** 父进程关闭不再使用的 pipe 端点后，逐个读取 worker 的 ready 消息；收到 EOF 就 join 对应进程并抛错，收到非 ready 状态也直接失败。

**读法：** 这个握手把可服务状态定义为“所有 worker 都已完成 scheduler 初始化”。HTTP 只有在这个条件满足后才会 bind。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L176-L191
for i, reader in enumerate(scheduler_pipe_readers):
    try:
        data = reader.recv()
    except EOFError:
        processes[i].join()
        raise

    if data["status"] != "ready":
        raise RuntimeError("Initialization failed. Please see the error messages above.")
    scheduler_infos.append(data)
```

**代码逻辑：** 父进程阻塞等待每个 scheduler pipe；失败时记录 rank 与 exit code；成功时收集 `scheduler_infos` 并关闭 reader。

**为什么这样写：** 多 worker serving 的最坏体验是端口已开但部分 GPU 不可用。ready barrier 把这种隐性失败前移到启动阶段。

**不变量与失败模式：** 任一 worker 未返回 `{"status": "ready"}` 都会阻止 HTTP 启动；这牺牲部分可用性，换取模型服务的一致性。

**要点：** 这个设计也让部署系统更容易判断健康：进程启动失败会直接表现为服务未监听，而不是运行时随机 500。

### 1.3 端口选择

**问题与约束：** runtime 同时需要 scheduler、broker、HTTP、disagg work/result 等多个端口；自动分配时不能撞上已占用端口，也不能和本次启动已选择的端口冲突。

**设计选择：** `_find_available_port` 从给定起点开始扫描，跳过 `avoid` 集合，并用 `is_port_available` 检查系统占用。

**读法：** 端口分配保持小函数化，是因为 monolithic、pool disagg、standalone role 都需要类似逻辑，但各自起始偏移不同。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L29-L44
def _find_available_port(start: int = 10000, avoid: set[int] | None = None, max_attempts: int = 100) -> int:
    if avoid is None:
        avoid = set()
    port = max(1024, min(start, 65535))
    for _ in range(max_attempts):
        if port not in avoid and is_port_available(port):
            return port
        port += 1
        if port > 65535:
            port = 1024
    raise RuntimeError(...)
```

**代码逻辑：** 函数把起点限制在合法端口范围，最多尝试 `max_attempts` 次；超过 65535 后回绕到 1024；找不到就抛错。

**为什么这样写：** 端口冲突如果延迟到 socket bind 才暴露，错误会散落在不同组件中；集中选择端口能让 disagg 启动路径更可控。

**不变量与失败模式：** `avoid` 只避免本进程已知冲突，不能替代系统占用检查；如果端口在检查后被其他进程抢占，后续 bind 仍可能失败。

**要点：** 这类小工具解释了为什么启动器不把端口写死：多角色 diffusion serving 的 socket 数量比普通单 HTTP 服务多。

### 1.4 disagg pool 的角色入口

**问题与约束：** pool disaggregation 把 encoder、denoiser、decoder 拆成多组实例，每组 GPU 数、work endpoint、result endpoint 都可能不同。

**设计选择：** `launch_pool_disagg_server` 先统计三类角色实例数并记录日志，随后为每个角色分配 work/result endpoint，再启动对应 role worker。

**读法：** 这里的设计哲学是“角色池先描述拓扑，再启动进程”。日志中的 encoder/denoiser/decoder 数量是后续 endpoint 分配和调度策略的基准。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L247-L249
logger.info(
    "Starting pool disagg server: %d encoder(s), %d denoiser(s), %d decoder(s)...",
    num_encoders,
```

**代码逻辑：** 源码在进入 endpoint 分配前先计算并输出 role 数量；后续使用这些数量生成 work endpoint 列表和 result endpoint。

**为什么这样写：** disagg 模式下性能和故障定位都依赖角色拓扑；启动日志必须先把拓扑显式化，否则后续只能从端口和进程名反推。

**不变量与失败模式：** 三类 role list 的长度决定实例数量；如果某类为空，调度拓扑就不完整，后续 DiffusionServer dispatch 语义会变得不成立。

**要点：** 这一小段日志不是装饰，它是多角色服务启动时最早可见的拓扑声明。

### 1.5 进程树清理

**问题与约束：** 多进程 worker、webui 子进程和 disagg role 失败时，如果只退出父进程，子进程可能继续占用 GPU、端口和 NCCL group。

**设计选择：** `kill_process_tree` 用 psutil 递归找子进程，先 kill children，再按需 kill parent；在主线程中重置 SIGCHLD handler，减少关闭时的噪声日志。

**读法：** 这是服务启动器的最后一道边界：出错时优先清理整个进程树，而不是依赖 Python 对 daemon process 的隐式回收。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/launch_server.py L47-L69
def kill_process_tree(parent_pid, include_parent: bool = True, skip_pid: int = None):
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    ...
    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        child.kill()
```

**代码逻辑：** 函数解析 parent 进程，递归枚举 children，跳过指定 pid 后逐个 kill；如果需要包含 parent，则继续 kill 自身或指定 parent。

**为什么这样写：** diffusion worker 常驻 GPU 显存，孤儿进程会让下一次启动看起来像随机 OOM；显式清理比等待系统回收更符合服务部署需求。

**不变量与失败模式：** `skip_pid` 允许保留特定子进程；如果传错，可能留下仍占端口的进程。psutil 查不到进程时函数直接返回。

**要点：** `.obsidian` 或阅读笔记侧不应模仿这种清理逻辑；它只属于运行时进程管理。

---

## 2. Worker 进程：分布式环境、pipeline 与 event loop

### 2.1 run_scheduler_process 入口

**问题与约束：** 每个 worker 进程既要配置设备架构和 tracing，又要构造 scheduler 并通知父进程 ready；这些动作顺序错了会导致父进程误判可用性。

**设计选择：** `run_scheduler_process` 先配置 logger、平台 arch、tracing 和 `PortArgs`，再构造 `Scheduler`，发送 ready 后才进入 `scheduler.event_loop()`。

**读法：** ready 的语义是 scheduler 已经成功构造，而不是 worker 进程刚启动。这样父进程等待的是可接收请求的后端，而不是一个还在加载模型的进程。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L1018-L1033
try:
    scheduler = Scheduler(
        server_args,
        gpu_id=rank,
        port_args=port_args,
        task_pipes_to_slaves=task_pipes_to_slaves,
        result_pipes_from_slaves=result_pipes_from_slaves,
        local_rank=local_rank,
    )
    pipe_writer.send({"status": "ready"})
    scheduler.event_loop()
```

**代码逻辑：** rank 进程创建 scheduler；把 ready 写回父进程 pipe；随后阻塞在 event loop，持续处理前端请求或 rank 间任务。

**为什么这样写：** 如果先发送 ready 再构造 scheduler，HTTP 层可能在后端 socket 尚未绑定时开始转发请求。源码把 ready 放在 scheduler 构造后，降低竞态。

**不变量与失败模式：** `task_pipes_to_slaves` 和 `result_pipes_from_slaves` 必须非空；OOM 会被专门捕获并重新抛出，其他异常也会进入 finally 清理。

**要点：** rank0 和 slave rank 的差异不在这个入口函数本身，而在传入 scheduler 的 pipe 列表和 scheduler 内部事件循环。

### 2.2 GPUWorker 初始化

**问题与约束：** worker 需要同时持有 pipeline、SP/TP/CFG group、realtime session cache 和可选显存占用控制器；这些状态要在请求前完成初始化。

**设计选择：** `GPUWorker.__init__` 记录 rank 与 server args 后立即调用 `init_device_and_model`，再缓存并行 group 的 GPU/CPU group，并创建 `RealtimeSessionCache`。

**读法：** 这让 GPUWorker 对外呈现为“构造完成即可执行”的对象，而不是让 scheduler 在每次请求前检查模型和通信组是否就绪。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L111-L134
def __init__(self, local_rank: int, rank: int, master_port: int, server_args: ServerArgs):
    self.local_rank = local_rank
    self.rank = rank
    self.master_port = master_port
    self.server_args = server_args
    self.pipeline: ComposedPipelineBase = None

    self.init_device_and_model()
    self.sp_group = get_sp_group()
    self.tp_group = get_tp_group()
    self.cfg_group = get_cfg_group()
```

**代码逻辑：** 构造函数先保存 rank 元信息；加载设备和 pipeline；获取 SP、TP、CFG group 及其 CPU group；初始化 realtime session cache 和 memory occupation controller 占位。

**为什么这样写：** pipeline 构建依赖分布式环境，group 缓存又依赖初始化完成。把顺序固定在构造函数中，可以避免 lazy init 在并发请求下出现重复初始化。

**不变量与失败模式：** `self.pipeline` 在 `execute_forward` 前必须不为 `None`；如果 `init_device_and_model` 失败，worker 不会进入 ready 状态。

**要点：** CPU group 的存在说明这里不仅服务 GPU collective，也考虑跨进程 CPU-side 控制消息。

### 2.3 分布式初始化依赖

**问题与约束：** multimodal diffusion 可能同时启用 TP、SP、Ulysses、Ring、CFG parallel；worker 必须在 pipeline 构建前初始化这些并行状态。

**设计选择：** `gpu_worker.py` 从 distributed 模块导入 `get_sp_group`、`get_tp_rank`、`get_tp_world_size`、`maybe_init_distributed_environment_and_model_parallel` 和 `model_parallel_is_initialized`。

**读法：** 这些 import 暗示 GPUWorker 不是一个普通单卡执行器，而是模型并行环境的一部分。它需要能设置进程标题、构建 group、并按并行 rank 执行 pipeline。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L19-L25
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_tp_rank,
    get_tp_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
```

**代码逻辑：** worker 文件直接依赖并行状态读取与初始化 API；后续 `init_device_and_model` 会设置环境变量并调用初始化函数。

**为什么这样写：** diffusion pipeline 的并行方式会影响 stage 执行和通信；把并行初始化放在 worker 层，比让每个 pipeline stage 自己探测环境更集中。

**不变量与失败模式：** 并行组必须在 `build_pipeline` 前可用；如果模型并行未初始化，进程标题和 group 缓存会退回更简单路径，但多卡执行能力也会受限。

**要点：** 阅读这部分时不要把 `num_gpus` 等同于纯 data parallel；这里的 GPU 可能被多个并行维度切分。

### 2.4 OOM 捕获

**问题与约束：** 启动或运行期 OOM 是 diffusion 服务最常见故障之一；如果只输出通用 traceback，用户很难区分加载 OOM、运行 OOM 和并行配置问题。

**设计选择：** `run_scheduler_process` 专门捕获 `_oom_exceptions()`，记录 `OOM_MSG` 后重新抛出，让父进程仍能感知失败。

**读法：** 这里没有吞掉 OOM，而是“补充诊断再失败”。这保证启动器的 ready barrier 不会把 OOM worker 当成可用。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L1034-L1036
except _oom_exceptions() as _e:
    logger.warning(OOM_MSG)
    raise
```

**代码逻辑：** OOM exception 被单独匹配；记录预设建议文本；随后 `raise` 保留原异常传播。

**为什么这样写：** 服务端 OOM 既需要人类可读的操作建议，也不能隐藏真实异常；重新抛出能让进程退出、父进程发现 EOF，并阻止 HTTP 假启动。

**不变量与失败模式：** 只有 `_oom_exceptions()` 覆盖的类型会走这条路径；其他异常仍由外层异常传播和 finally 清理处理。

**要点：** 这段和 ready 握手是一组设计：worker 失败要清楚地失败，而不是降级成半启动服务。

---

## 3. HTTP 层：协议入口和 warmup gate

### 3.1 lifespan 管理 scheduler client 与 broker

**问题与约束：** FastAPI 进程既服务 HTTP/OpenAI-style 请求，也要支持离线 ZMQ broker；二者都需要连接同一个后端 scheduler。

**设计选择：** `lifespan` 初始化 singleton `async_scheduler_client`，创建 warmup event，把 `run_zeromq_broker` 作为后台 task 启动，并在 shutdown 时取消 task、关闭 client。

**读法：** HTTP 层不直接持有 GPU worker，只持有 scheduler client。这样 HTTP API、offline broker 和 Vertex route 都走同一条 backend 转发路径。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L110-L124
server_args = app.state.server_args
async_scheduler_client.initialize(server_args)
warmup_done = asyncio.Event()
app.state.server_warmup_done = warmup_done

broker_task = asyncio.create_task(run_zeromq_broker(server_args))
...
else:
    warmup_done.set()
```

**代码逻辑：** app 启动时初始化 async client 和 warmup event；broker 后台监听离线请求；如果未启用 server warmup，则立即放行请求。

**为什么这样写：** 把 client 生命周期绑定到 FastAPI lifespan，可以避免模块 import 时就创建 ZMQ context，也能在 shutdown 中统一清理。

**不变量与失败模式：** `app.state.server_args` 必须先由 `create_app` 写入；如果 warmup task 失败，会发送 SIGTERM 终止进程，而不是继续服务未 warm 的模型。

**要点：** broker 和 HTTP route 是两个入口，但它们都不绕过 scheduler，这保持了请求调度的一致性。

### 3.2 warmup 探活

**问题与约束：** server warmup 要等 HTTP 本身已经可访问后再发合成请求；同时本地 health check 不应被环境代理污染。

**设计选择：** `_wait_until_http_ready` 用 `httpx.AsyncClient(trust_env=False)` 轮询本机 `/health`，最多 120 次，每次 timeout 5 秒。

**读法：** 这是一个启动闭环：HTTP 先起来，warmup task 再通过真实 HTTP 入口验证端到端路径，而不是直接调用内部函数。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L60-L76
health_url = f"{server_args.url()}/health"
async with httpx.AsyncClient(trust_env=False) as client:
    for _ in range(120):
        try:
            response = await client.get(health_url, timeout=5.0)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1.0)
raise RuntimeError(...)
```

**代码逻辑：** 函数构造 health URL；禁用代理环境变量；轮询直到 200 或超时；超时后抛 RuntimeError。

**为什么这样写：** warmup 不是简单等待端口打开，而是验证 FastAPI route 可用。`trust_env=False` 则避免 CI/容器里的代理配置影响本地 loopback。

**不变量与失败模式：** 120 次轮询约等于最多两分钟等待；如果 HTTP 一直不 ready，warmup task 会失败并触发进程终止。

**要点：** 这解释了为什么 warmup bypass path 需要存在：health route 必须在 warmup 期间仍能被访问。

### 3.3 warmup bypass 与 Vertex route

**问题与约束：** warmup gate 不能阻塞健康检查和模型发现，否则 readiness probe 会被自己锁住；同时 Vertex AI 的预测 route 需要可配置。

**设计选择：** HTTP 模块用环境变量设置 `VERTEX_ROUTE`，并列出 `SERVER_WARMUP_BYPASS_PATHS`，middleware 在 warmup 未完成时放行这些路径。

**读法：** 这是一种“对用户请求关门，对控制面开门”的启动策略：生成请求等待 warmup，健康和发现接口仍可返回。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/entrypoints/http_server.py L51-L57
VERTEX_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/vertex_generate")
SERVER_WARMUP_BYPASS_PATHS = (
    "/health",
    "/health_generate",
    "/model_info",
    "/server_info",
)
```

**代码逻辑：** `VERTEX_ROUTE` 默认 `/vertex_generate`，也可由环境变量覆盖；bypass path 包含健康、生成健康检查和模型/服务信息接口。

**为什么这样写：** 云平台经常要求固定预测路径和 readiness path；把 Vertex route 环境化、把控制面 path 放行，可以兼容平台探活和 warmup。

**不变量与失败模式：** bypass 只列控制面接口；如果把生成接口也加入 bypass，就会破坏 warmup gate 的保护。

**要点：** 这段很短，但直接决定启动期外部负载均衡看到的服务状态。

---

## 4. Scheduler client：同步、异步与对象传输边界

### 4.1 同步 SchedulerClient 初始化

**问题与约束：** 离线 DiffGenerator 这类同步调用方需要阻塞式请求 scheduler；ZMQ REQ socket 有严格 send/recv 状态机，初始化和超时必须集中管理。

**设计选择：** `SchedulerClient.initialize` 若发现已有 context 会先 close，再创建新的 ZMQ context 与 REQ socket，设置 `LINGER=0` 和长接收超时，然后连接 `scheduler_endpoint`。

**读法：** 同步 client 是单 socket、单状态机模式，适合命令式调用；它和 FastAPI 的 async client 分开，避免并发 HTTP 请求共享同一个 REQ socket。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/scheduler_client.py L60-L78
def initialize(self, server_args: ServerArgs):
    if self.context is not None and not self.context.closed:
        self.close()

    self.server_args = server_args
    self.context = zmq.Context()
    self.scheduler_socket = self.context.socket(zmq.REQ)
    self.scheduler_socket.setsockopt(zmq.LINGER, 0)
    self.scheduler_socket.setsockopt(zmq.RCVTIMEO, 6000000)
    self.scheduler_socket.connect(self.server_args.scheduler_endpoint)
```

**代码逻辑：** 初始化先清理旧 context；保存 server args；创建 REQ socket；设置关闭不等待和 100 分钟接收超时；连接后端 scheduler endpoint。

**为什么这样写：** diffusion 生成请求可能很长，默认短超时会误杀任务；`LINGER=0` 则让进程退出时不会因为未发送完的 ZMQ 消息卡住。

**不变量与失败模式：** REQ socket 不能并发复用；如果同步 client 被多线程共享，仍可能触发 ZMQ 状态问题。

**要点：** FastAPI 使用 async client 的“每请求一个 socket”策略，正是为了避开这个同步 client 的并发限制。

---

## 5. Pipeline executor：stage hook、profiling 与执行上下文

### 5.1 Req/OutputBatch 是执行边界

**问题与约束：** pipeline stage 可能返回中间 `Req`，也可能返回最终 `OutputBatch`；executor 和 worker 需要一个共同类型边界来表达请求输入与输出结果。

**设计选择：** `pipeline_executor.py` 在模块层导入 `OutputBatch` 和 `Req`，后续抽象方法以 `Req` 作为输入、`OutputBatch` 作为标准输出。

**读法：** 这不是普通 import，而是 pipeline 执行层的契约：请求在 stage 间流动时仍是 `Req`，跨出 executor 后统一落到 `OutputBatch`。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L15-L15
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
```

**代码逻辑：** executor 基类把调度 batch 类型作为公共依赖；子类实现 `execute` 时围绕这两个类型组织 payload。

**为什么这样写：** 多模态生成的输出形态很多，直接让每个 stage 返回裸 tensor 会让 HTTP、保存、profiling 都难以统一；`OutputBatch` 提供了统一容器。

**不变量与失败模式：** stage 返回值最终必须能被 worker 转成 `OutputBatch`；否则后处理、文件路径、metrics 和错误字段都无法统一表达。

**要点：** 读 executor 时先抓住 `Req`/`OutputBatch`，比先看具体 pipeline stage 更容易理解控制流。

### 5.2 execute_with_profiling

**问题与约束：** 推理执行需要同时满足 no-grad/inference mode、平台差异和可选 profiling；如果每个 executor 子类自己写，很容易漏掉一致性逻辑。

**设计选择：** 基类提供 `execute_with_profiling`：外层进入 `profile_execution`，内层进入 `current_platform.inference_mode()`，再调用抽象 `execute`。

**读法：** 这里把“怎么测”和“怎么跑 stage”分离。子类只需要实现 stage 调度，profiling 和推理上下文由基类包住。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L126-L137
def execute_with_profiling(self, stages: List["PipelineStage"], batch: Req, server_args: ServerArgs) -> OutputBatch:
    with self.profile_execution(batch, dump_rank=0):
        with current_platform.inference_mode():
            batch = self.execute(stages, batch, server_args)

    return batch
```

**代码逻辑：** 输入 `Req` 先进入 profiling context；在平台推理模式下执行子类 `execute`；返回处理后的 batch。

**为什么这样写：** profiling 和 inference mode 是跨 executor 的横切关注点；放在基类可以避免每种 pipeline 执行器重复实现并产生差异。

**不变量与失败模式：** `execute` 必须返回符合 `OutputBatch` 语义的对象；如果 stage 需要版本计数器，不能只依赖全局 inference mode，还要走 stage context。

**要点：** 下面的 stage hook 正是为了处理 offload/FSDP 等需要临时放宽 inference mode 的特殊情况。

### 5.3 stage hook 与 component residency

**问题与约束：** diffusion pipeline 的组件可能按 stage 进出 GPU，或者使用 layerwise/cpu offload；executor 需要在每个 stage 前通知 residency manager，并可插入 NVTX marker。

**设计选择：** `_run_stage_with_executor_hooks` 在执行 stage 前调用 `before_stage`，把 component residency manager 注入 stage，再用 `maybe_nvtx_range` 包住实际 stage execution。

**读法：** 这让 stage 专注“如何计算”，executor 专注“何时加载组件、何时打 profiling 标记、何时切上下文”。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L103-L118
def _run_stage_with_executor_hooks(...):
    stage_name = stage._component_stage_name()
    self.before_stage(stage, stage_index, payload, server_args)
    with maybe_nvtx_range(f"stage_{stage_name}", use_nvtx):
        payload = self.run_stage_with_context(stage, payload, server_args, run_stage)
    return payload
```

**代码逻辑：** 函数取 stage 名；执行 residency/offload 前置 hook；按配置包 NVTX range；再调用 `run_stage_with_context` 执行实际 stage。

**为什么这样写：** 组件驻留和 profiling 都依赖 stage 边界。如果散落在 stage 内部，不同 pipeline stage 会重复处理显存策略。

**不变量与失败模式：** `component_residency_manager` 必须在请求开始前可用；否则 `before_stage` 无法完成组件驻留切换。

**要点：** 这也是为什么 pipeline executor 是抽象基类：不同执行器可以改 stage 顺序，但共享 hook 语义。

---

## 6. GPUWorker forward 与输出形态

### 6.1 后处理与保存工具依赖

**问题与约束：** 生成输出可能是 tensor、numpy frame、视频文件、音频或只返回文件路径；worker 需要后处理，但不应把编码/保存细节内联到 forward 主流程。

**设计选择：** `gpu_worker.py` 从 entrypoints utils 导入 `materialize_output_sample`、`post_process_sample` 和 `save_outputs`，在后续 raw frame、return frame、save path 分支中复用。

**读法：** 输出 materialization 是 API 边界的一部分，和模型 forward 不同。把它放在 entrypoints utils，worker 只编排何时调用。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L36-L40
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    materialize_output_sample,
    post_process_sample,
    save_outputs,
)
```

**代码逻辑：** worker 文件依赖三个输出工具：单样本 materialize、后处理 sample、保存 outputs。

**为什么这样写：** HTTP 和 worker 都需要理解输出格式，但真正的编码/保存实现应集中，否则新增视频插帧、upscaling、音频输出时会产生多处重复分支。

**不变量与失败模式：** 工具函数必须能处理 worker 传入的输出类型和 request 参数；否则错误会在请求结束阶段暴露，而不是模型 forward 阶段。

**要点：** 这解释了为什么 GPUWorker 看起来也有“entrypoint”逻辑：它负责把模型输出变成可传输响应。

### 6.2 Realtime session 释放

**问题与约束：** 实时视频连接会在 worker 内持有 session 状态；客户端断开或释放时必须清掉 session，并尽量回收 CUDA cache。

**设计选择：** `release_realtime_session` 对空 session id 返回结构化失败；成功释放后，如果 CUDA 已初始化则调用 `empty_cache()`。

**读法：** session 释放被设计成普通 `OutputBatch` 响应，而不是 side-effect-only 方法，这样 scheduler/client 路径可以统一返回结果。

**源码锚点：**

```python
## 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L136-L150
def release_realtime_session(self, session_id: str) -> OutputBatch:
    if not session_id:
        return OutputBatch(output={"released": False, "session_id": session_id, "reason": "empty_session_id"})

    released = self._realtime_sessions.release(session_id)
    if released:
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
```

**代码逻辑：** 函数先处理空 id；调用 session cache release；释放成功后清 CUDA cache；最后返回包含 released 和 session_id 的 `OutputBatch`。

**为什么这样写：** 实时 session 占用的状态可能跨请求存在；显式释放和 cache 清理能避免长连接工作负载把显存碎片留给后续普通生成。

**不变量与失败模式：** 空 session id 不抛异常，而是返回失败结果；这让 API 层可以把它当作业务错误，而不是 worker crash。

**要点：** `max_sessions=1` 的 cache 初始化说明当前 worker 更偏向单实时连接管理，而不是无限 session 池。

---

## 7. 请求路径串联

`multimodal_gen` 的请求路径可以压缩成一句话：HTTP 或 offline client 把 `Req` 交给 scheduler client，scheduler 把任务送到 GPU worker，worker 调 pipeline executor，executor 在 stage hook 和 profiling context 中跑 pipeline，最后 worker 把结果收敛为 `OutputBatch` 并按请求选择返回 tensor、frames、base64 或文件路径。

读这套源码时最重要的不是记住每个 endpoint，而是分清边界：启动器处理进程生命周期，HTTP 处理协议生命周期，client 处理 IPC 生命周期，worker/executor 处理 GPU 与 pipeline 生命周期。这样才能看懂为什么很多逻辑看似绕远路，实际是在避免半启动服务、REQ socket 并发冲突、stage offload 漏清理和输出格式散落。

---

## 运行验证

维护本文时，先用下面的命令确认 multimodal_gen 请求主线还在原位：

```powershell
rg -n "def launch_server|class GPUWorker|class SchedulerClient|class PipelineExecutor|warmup_done|release_realtime_session|materialize_output_sample|post_process_sample|save_outputs" sglang/python/sglang/multimodal_gen/runtime
```

预期信号：

- `launch_server.py` 仍承担进程启动、worker 就绪和 HTTP 暴露前置检查。
- `entrypoints/http_server.py` 仍处理 warmup gate 与 HTTP 生命周期。
- `scheduler_client.py` 仍是同步、异步与对象传输边界。
- `managers/gpu_worker.py` 与 `pipelines_core/executors/` 仍承载 worker forward、session 释放和 stage hook。
- `entrypoints/utils.py` 仍集中输出 materialize、后处理和保存逻辑。

如果这些职责被拆到新的 runtime 子包，应先更新本文的“启动器 / HTTP / client / worker / executor”边界，再更新逐段源码锚点。
