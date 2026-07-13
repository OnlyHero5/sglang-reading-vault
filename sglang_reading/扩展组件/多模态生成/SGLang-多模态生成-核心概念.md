---
title: "多模态生成 · 核心概念"
type: concept
framework: sglang
topic: "多模态生成"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-12
---

# 多模态生成 · 核心概念

## 你为什么要读

图像/视频生成不是“把 LLM 的 token 换成像素”。它的请求载体、调度兼容性、并行维度、模型阶段和输出运输都不同。理解本专题的关键不是背 `Encoder → Denoiser → Decoder`，而是分清五种所有权：谁接协议、谁收 ZMQ、谁决定合批、谁在每个 rank 执行 pipeline、谁把大输出变成可返回对象。

## 贯穿对象：一次视频生成请求

把一次请求记成四次形态变化：

1. 外部 JSON 被 route 转成 `SamplingParams` 与内部 `Req`。
2. `Req` 被 pickle 后经临时 ZMQ REQ socket 送到 rank0 Scheduler。
3. Scheduler 把请求放入 FIFO queue，决定单发、动态合批或顺序 fallback；多 rank 通过 distributed object broadcast 得到相同调度输入。
4. GPUWorker 调用 pipeline，最终把 `Req` 或 `OutputBatch` 统一为 `OutputBatch`，再按 raw frames、numpy frames、文件路径或普通 tensor 选择运输形态。

这条链里，父进程的 ready pipe 只证明 worker 已构造完成；它不是生成请求的数据总线。

## 1. 进程与 rank：每个 rank 都有 Scheduler

`launch_server` 为每个 GPU 启动一个 `run_scheduler_process`。每个进程都会构造 Scheduler，Scheduler 再构造本 rank 的 GPUWorker。差别在于只有 `gpu_id == 0` 创建 ZMQ ROUTER；其他 rank 的 `receiver` 为 `None`。

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/scheduler.py L103-L111
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None
```

因此“rank0 有 Scheduler、slave 只有 GPUWorker”也是错误模型。更准确的说法是：所有 rank 都跑同一个调度循环形状，但外部 socket 与结果回复只属于 rank0；其他 rank 从并行组广播获得请求。

父进程会等待每个 worker 通过单独 ready pipe 返回 `{"status": "ready"}`，之后才启动 HTTP。这是启动 barrier，不是运行期健康监控。worker 在运行期崩溃后，源码没有自动把剩余 rank 重组为降级集群。

## 2. 四种通信不要混在一起

| 通信 | 方向 | 载荷/目的 | 生命周期 |
|---|---|---|---|
| ready `mp.Pipe` | worker → parent | `status: ready` | 仅启动期 |
| Scheduler ZMQ ROUTER/REQ | client ↔ rank0 | pickle 请求与 `OutputBatch` | 每次外部请求 |
| `broadcast_pyobj` | rank0 → SP/CFG/TP ranks | Python 请求对象 | 普通生成调度 |
| task/result `mp.Pipe` | rank0 ↔ slave | 显式 method/kwargs 与控制结果 | 专门控制路径 |

普通生成的多 rank 复制由 `recv_reqs()` 完成。它会按配置依次经过 SP、CFG、TP group；某个维度为 1 时不执行对应广播。不能看到 launch 阶段创建 Pipe，就推断生成 batch 经 Pipe 分发。

另一个边界是 `dp_size`：字段、`/server_info` 和若干计算式中虽然出现 DP，但当前 `_validate_parallelism` 对 `dp_size > 1` 直接抛错。这是“配置表面存在、运行能力尚未开放”的典型例子。

```python
# 来源：python/sglang/multimodal_gen/runtime/server_args.py L2037-L2041
        if self.dp_size < 1:
            raise ValueError("--dp-size must be a natural number")

        if self.dp_size > 1:
            raise ValueError("DP is not yet supported")
```

## 3. Scheduler 不只是转发器

Scheduler 持有 FIFO `waiting_queue`，其中每项包含 client identity、请求对象与入队时间。它会：

- 将 ROUTER 收到的 payload 规范化为逻辑请求；
- 应用 request-based warmup 处理；
- 在 batching delay 内寻找兼容请求；
- 用 `BatchAdmissionController` 判断显存/形状规则是否允许继续扩 batch；
- 合并 prompt、seed、输出路径等字段；
- 执行后把 batched output 安全切回每个原请求；
- 保证 handler 异常也尽量形成 `OutputBatch(error=...)` 回复。

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/scheduler.py L147-L153
        # FIFO queue entries: (identity, request, enqueue_ts_s)
        self.waiting_queue: deque[tuple[bytes | None, Any, float]] = deque()
        self._batching_max_size = server_args.batching_max_size
        self._batching_delay_s = server_args.batching_delay_ms / 1000.0
        self._batch_metrics_enabled = server_args.enable_batching_metrics
        self._batch_metrics_window = BatchMetricsWindow()
        self._batch_admission = BatchAdmissionController(server_args, gpu_id=local_rank)
```

### 动态合批的两道门

第一道是语义兼容：warmup、realtime session、非字符串 prompt、image conditioning、不同 `return_file_paths_only` 都会拒绝；SamplingParams 除明确标记 `batch_sig_exclude` 的字段外必须一致，`extra.diffusers_kwargs` 也参与签名。

第二道是 admission：即便签名相同，也可能因模型与分辨率规则达到有效 batch 上限。`batching_max_size` 是用户上限，不等于每种 workload 都能达到的安全上限。

如果多个普通 `Req` 无法合并，当前实现会顺序执行；如果已经合并的 forward 返回 error、或结果无法一一切分，则不会再重跑一遍顺序 fallback，而是为各请求构造错误结果，避免重复昂贵计算和副作用。

## 4. GPUWorker：计算与运输的分界

GPUWorker 构造时完成设备选择、distributed/model-parallel 初始化、pipeline 构建、parallel group 缓存与 realtime session cache 创建。它不是只包一层 `pipeline.forward`：还负责 metrics、OOM 转译、保存文件、raw RGB 打包、numpy frame 物化和大对象释放。

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L472-L483
    def _materialize_output_transport(
        self,
        output_batch: OutputBatch,
        req: Req,
        save_output_paths: Callable[[OutputBatch], None],
    ) -> None:
        if req.return_raw_frames:
            self._materialize_raw_frame_transport(output_batch, req)
        elif req.save_output and req.return_file_paths_only:
            self._materialize_file_path_transport(output_batch, save_output_paths)
        elif req.return_frames:
            self._materialize_frame_outputs_for_return(output_batch, req)
```

三条分支互斥且有优先级：raw frames 优先，其次是“保存且只返路径”，最后才是普通 frames。HTTP 侧仍可能在收到 tensor 时调用 `save_outputs`，所以“文件保存只属于 worker”与“文件保存只属于 HTTP”都过于绝对；所有权取决于请求 transport 选项和返回内容。

## 5. PipelineExecutor：横切策略，不等于固定 stage 表

`PipelineExecutor` 是抽象基类。它提供 profiling、平台 inference context、stage hook、NVTX 与 component residency request 生命周期；具体 stage 序列由 pipeline 和具体 executor 决定，不能把所有模型写死成完全相同的三段流水线。

`before_stage` 只把 residency manager 注入 stage 并委托 `manager.before_stage(...)`。真正把哪些 component 搬到哪里、何时释放，由具体 manager 和 executor 实现。CPU offload、layerwise offload、FSDP inference 也不是同一个开关：某些 stage 为保留 tensor version counter，会临时退出 inference mode，转入 `torch.inference_mode(False), torch.no_grad()`。

```python
# 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L151-L159
    @staticmethod
    @contextlib.contextmanager
    def _stage_execution_context(stage: "PipelineStage", server_args: ServerArgs):
        if PipelineExecutor._stage_needs_version_counters(stage, server_args):
            # fsdp and cpu-offload hooks need tensor version counters
            with torch.inference_mode(False), torch.no_grad():
                yield
            return
        yield
```

## 6. Warmup：两种模式、两个 gate

`warmup_mode` 最终被规范化成 `off`、`request` 或 `server`：

- request warmup 在 Scheduler 收到真实请求后由 warmup mixin 安排；它是 benchmark/首请求路径的一部分。
- server warmup 在 FastAPI lifespan 中后台执行。middleware 先让控制面路径通过，普通路径等待 `warmup_done`。

server warmup 会先轮询本机 `/health`，但随后把 synthetic request 直接交给 `async_scheduler_client.forward`。因此它验证了 HTTP 进程已可探活和 scheduler/pipeline 后端可用，却没有再次验证 image/video route 的 JSON 转换全过程。

非 monolithic disagg role 会在参数规范化时强制 `server_warmup=False`；不能把 monolithic 的 HTTP warmup gate套到独立 encoder/denoiser/decoder role。

## 7. Monolithic 与 disagg 是两套事件循环

`dispatch_launch` 按 `RoleType` 选择 monolithic、head server 或 standalone role。Scheduler 的 `event_loop()` 也会先判断 role：非 monolithic 直接进入 `_disagg_event_loop()` 并返回，不走普通 waiting queue 主循环。

pool 模式的 `DiffusionServer` 在 encoder、denoiser、decoder 边界调度中间对象；每个 role instance 内仍可有多个 rank，但并行度由 role-specific override 重新计算。这里与 LLM PD 分离只有“阶段拆分”的抽象相似，传输对象、角色状态机和失败恢复都不能类推。

## 运行验证

```powershell
rg -n 'if gpu_id == 0|receiver = None|waiting_queue|broadcast_pyobj|_broadcast_task|_try_merge_generation_reqs|_split_batched_output' sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py
rg -n 'return_raw_frames|return_file_paths_only|return_frames|build_pipeline|maybe_init_distributed_environment_and_model_parallel' sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
rg -n 'warmup_mode|server_warmup = False|DP is not yet supported|sp_degree.*ring_degree.*ulysses_degree' sglang/python/sglang/multimodal_gen/runtime/server_args.py
```

预期：能同时定位外部 socket 所有权、三类 distributed broadcast、独立 Pipe 控制方法、动态合批与切分、三种输出 transport，以及 DP/disagg/warmup 的配置边界。若只找到类名而不能说明对象从谁移交给谁，说明心理模型仍停留在目录层。
