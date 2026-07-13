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
updated: 2026-07-12
---

# 多模态生成 · 源码走读

> 走读顺序：启动 barrier → Scheduler socket/queue → distributed broadcast → 动态合批 → GPUWorker → PipelineExecutor → OutputBatch 返回。

## 长文读法

不要按文件目录平铺。本篇沿一次请求的所有权变化阅读，并在每一层回答三个问题：对象现在归谁、下一跳用什么通信、失败后谁负责形成可观察结果。

| 读者任务 | 章节 | 要抓住的结论 |
|---|---|---|
| 排查端口未监听 | 1 | 所有 worker ready 是 HTTP 启动前置条件，但不是运行期 supervisor |
| 排查 slave 不工作 | 2、3 | 每个 rank 都有 Scheduler/GPUWorker；普通请求走 distributed broadcast，不走 task Pipe |
| 排查延迟抖动 | 4 | queue delay、兼容性、admission、merge/split 都会改变 dispatch |
| 排查 OOM 或 offload | 5、6 | GPUWorker 管设备/输出，executor 管横切 stage context；具体迁移属于 residency manager |
| 排查响应巨大或序列化慢 | 7 | raw frames、numpy、file paths、local spill 是不同 transport |
| 排查 warmup/disagg | 8、9 | server warmup 直达 scheduler client；非 monolithic 使用另一事件循环 |

## 1. 启动：ready barrier 先于 HTTP

### 1.1 父进程只拉起，不加载 pipeline

`launch_server` 创建两组 master/slave task/result Pipe，也为每个 worker 单独创建 ready pipe，然后按 `num_gpus` 启动 `run_scheduler_process`。父进程不构造 GPUWorker；模型加载发生在 worker 内 Scheduler 构造期间。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/multimodal_gen/runtime/launch_server.py L86-L156
def launch_server(server_args: ServerArgs, launch_http_server: bool = True):
    num_gpus = server_args.num_gpus
    # create task/result pipes
    for i in range(num_gpus):
        reader, writer = mp.Pipe(duplex=False)
        process = mp.Process(target=run_scheduler_process, args=(...))
        process.start()
```

这里的 Pipe 有两种用途：每 rank ready pipe，以及 rank0/slave 控制 pipe。后者虽然在启动器中建立，却不是普通生成请求的主广播通道。

### 1.2 ready 的精确定义

worker 只有在 Scheduler 构造成功后才发送 ready；Scheduler 构造又会创建 GPUWorker，GPUWorker 会初始化设备、distributed groups 并 `build_pipeline`。因此 ready 比“进程活着”强，但仍不包含 server warmup，也不代表未来运行期不会崩溃。

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/gpu_worker.py L1019-L1033
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

父进程逐个 `reader.recv()`；EOF 或非 ready 状态都会阻止 HTTP 启动。当前收集到的 `scheduler_infos` 后续没有参与路由或健康监控，不能把它解释成持续注册表。

### 1.3 HTTP 在哪个进程

默认 `launch_http_server_only` 由父进程直接调用，`uvicorn.run` 阻塞父进程；worker 在各自子进程。只有 `webui` 分支另起 HTTP 子进程。于是“HTTP 永远是额外独立子进程”不准确，但“HTTP 与 CUDA worker 不同进程”在正常多进程启动中成立。

`launch_http_server=False` 只返回 worker process 列表；由于 FastAPI lifespan 没启动，offline broker也不会自动启动。调用方若要离线访问，需要自己使用 scheduler endpoint/client 或建立相应入口。

## 2. Scheduler 构造：只有 rank0 对外 bind

所有 rank 都执行 `Scheduler(...)`。rank0 创建 ROUTER，其他 rank `receiver=None`；随后每个 rank 都构造 GPUWorker。

```python
# 来源：python/sglang/multimodal_gen/runtime/managers/scheduler.py L103-L120
        endpoint = server_args.scheduler_endpoint
        if gpu_id == 0:
            # router allocates identify (envelope) for each connection
            self.receiver, actual_endpoint = get_zmq_socket(
                self.context, zmq.ROUTER, endpoint, True
            )
            logger.info(f"Scheduler bind at endpoint: {actual_endpoint}")
        else:
            self.receiver = None
        from sglang.multimodal_gen.runtime.platforms import current_platform

        Exec_worker = CPUWorker if current_platform.is_cpu() else GPUWorker
        worker = Exec_worker(
            local_rank=local_rank,
            master_port=port_args.master_port,
            rank=gpu_id,
            server_args=server_args,
        )
```

`run_scheduler_process` 的形参命名值得警惕：`task_pipe_r` 与 `result_pipe_w` 在当前函数体没有使用；真正传入 Scheduler 的是最后两个参数。monolithic rank0 获得 list，slave 在启动调用中获得单个 Connection；普通 event loop 并不迭代这些对象来接生成请求。阅读时以函数体消费位置为准，不以注释或形参名推断协议。

## 3. 入站：ROUTER identity 与 distributed broadcast

### 3.1 async client 每请求一个 socket

FastAPI 使用 `AsyncSchedulerClient` singleton 保存 context 和 args，但 `forward()` 每次创建临时 REQ socket。这使并发请求各自保持 REQ send/recv 状态机。同步 client 复用单 socket，适合串行调用。

两者都设置 100 分钟接收超时；这只是 client 等待上限，不是 Scheduler 取消或 GPU kernel deadline。超时后任务是否仍在后端执行，需要另行观察。

### 3.2 payload 规范化

rank0 从 multipart 最后一帧取 payload并 pickle.loads。格式错误的 probe/envelope 被跳过。多元素 `list[Req]` 被保留为一个 grouped logical request；单元素 list 才展开。

### 3.3 普通生成的跨 rank 路径

rank0 取得请求后，按 SP、CFG、TP 条件调用 `broadcast_pyobj`。其他 rank 的 `receiver` 是 `None`，它们依赖这些 group broadcast 得到非空 `recv_reqs`。

```python
# 定位骨架（非逐行摘录）：来源 python/sglang/multimodal_gen/runtime/managers/scheduler.py L897-L949
def recv_reqs(self):
    if self.receiver is not None:
        # ROUTER recv + pickle + normalize
    else:
        recv_reqs = None
    if self.server_args.sp_degree != 1:
        recv_reqs = broadcast_pyobj(...)
    if self.server_args.enable_cfg_parallel:
        recv_reqs = broadcast_pyobj(...)
    if self.server_args.tp_size > 1:
        recv_reqs = broadcast_pyobj(...)
    assert recv_reqs is not None
    return recv_reqs
```

`_broadcast_task/_collect_slave_results` 是另一套 Pipe 控制 helper。普通 `Req` 主线没有调用它们。遇到 slave 未执行生成时，先看并行 group 和 broadcast source，不要先盯 task Pipe。

## 4. queue 与动态合批

### 4.1 入队和等待

event loop 每轮最多从 ROUTER 非阻塞收一批请求，经过 request-based warmup mixin 后，附同一个 `now` 入队。随后 `get_next_batch_to_run()` 决定是否 dispatch。

当队首仍可等待更多候选时，rank0 使用 ZMQ poll；非 rank0 sleep 相同剩余窗口。这保证各 rank 大致同步等待，但延迟门限仍由 rank0 queue 的最老时间决定。

### 4.2 兼容性签名

`_can_dynamic_batch` 同时要求：

- 两者都不是 warmup/realtime；
- prompt 都是字符串；
- 都没有 image conditioning；
- `return_file_paths_only` 一致；
- SamplingParams 有可构建且相等的签名；
- `extra.diffusers_kwargs` 相等。

`batch_sig_exclude` 由 SamplingParams field metadata 控制，说明“字段存在”不一定影响兼容性；需要读 dataclass metadata。

### 4.3 admission 与非连续取样

Scheduler 从队首向后扫描兼容项，允许跳过中间不兼容请求。达到 `batching_max_size` 或 admission 判满即停；派发时从后向前删除索引，以免 deque 索引位移，并恢复原顺序。

因此 FIFO 是“队首决定本轮”，不是“只能合并连续相邻项”。后面的兼容请求可能越过不兼容项加入队首 batch，而被跳过项继续留队。

### 4.4 merge、执行与 split

merge 深拷贝 base request，把 prompt 变成 list，把 seeds 与动态路径放进 `extra`。pipeline 不支持动态合批时会在更早的 coarse gate 单发；进入同轮 dispatch 后，若候选请求的语义签名不兼容，merge 返回 `None` 并顺序执行。

已合并后的错误不做顺序 fallback：forward error 或 split 失败会返回每请求 error。这避免把一次可能已产生副作用的昂贵生成再执行一遍。

split 以 `num_outputs_per_prompt` 计算每请求区间；output 与 output_file_paths 只要存在，就必须满足总第一维。trajectory/audio 等字段用相同区间切片或深拷贝 scalar metadata。

## 5. GPUWorker：从 Req 到 OutputBatch

### 5.1 初始化顺序

GPUWorker 先设置 device 与 `MASTER_ADDR/PORT/RANK/WORLD_SIZE`，初始化 TP/CFG/Ulysses/Ring/SP/DP environment，再构建 pipeline；layerwise offload 在 LoRA pipeline 构造后配置，避免空 offloaded weight 干扰 LoRA 转换。

### 5.2 单请求、group 与 disagg中间态

`execute_forward` 接收 `list[Req]`：

- 长度 1：走 `_execute_forward_common`；
- 长度大于 1：校验共享输出字段，调用 `pipeline.forward_batch`；
- `return_req=True`：只支持单请求，用于 disagg 保留中间 `Req`，不提前转成 OutputBatch。

### 5.3 异常不会向上抛到 Scheduler 主循环

`_execute_forward_common` 捕获一般异常，把文本写入 `OutputBatch.error`，OOM 额外记录建议并清 cache。于是 Scheduler 常见到的是“正常返回但 error 字段非空”，不是 Python exception。动态合批逻辑也专门检查该字段。

### 5.4 输出 transport

worker 按 raw frames → file paths only → frames 的优先级物化。rank0 才执行实际保存或 CPU frame conversion；其他 rank 仍执行 pipeline collective，但不构造最终外部资产。

普通 output 保留时，如果本地 ZMQ 返回，Scheduler 还可能 spill 大数组到临时文件；client 再 materialize。这一段影响 CPU 内存、磁盘 IO 与响应延迟，不能全部归因于模型 forward。

## 6. PipelineExecutor：stage 横切上下文

基类 `execute_with_profiling` 包住 profiler 和 platform inference mode；具体 executor 实现 `execute`。每个 stage 前 `_run_stage_with_executor_hooks` 调用 residency manager、可选 NVTX，并通过 `_stage_execution_context` 决定是否需要 tensor version counter。

```python
# 来源：python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py L103-L115
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
```

此卡只证明 hook 调用顺序；它不证明所有 pipeline 都有固定三 stage，也不证明 `before_stage` 后一定 H2D、stage 后一定 D2H。具体迁移必须继续读 residency manager。

## 7. 回复：identity、pickle 与 local spill

rank0 把每个 output 与原 `(identity, processed_req)` 对齐。handler 返回数量不匹配时，Scheduler 会用内部错误替换整批，避免把结果发给错误 client。

warmup result 有特殊规则：需要返回的 warmup 会先 drop heavy payload；不需要返回的 warmup设置 `should_not_return=True`。普通结果在本地 endpoint 上先 spill arrays，随后 pickle并 `send_multipart([identity, b"", payload])`。

ZMQ send 失败只记录并继续 event loop；client 可能超时，但后端不会因单次 reply 失败自动回滚生成和文件输出。

## 8. Warmup：健康路由不是完整生成验证

FastAPI lifespan 初始化 async client、warmup event 和 broker task。server warmup 后台任务先访问 `/health`；成功后调用 `run_async_client_warmup(..., async_scheduler_client.forward, ...)`。

所以它覆盖 HTTP 进程 readiness 与 scheduler/pipeline，却绕过 image/video route 的请求解析。middleware 在 warmup 完成前阻塞普通路由，但 `/health`、`/health_generate`、`/model_info`、`/server_info` 继续可用。

warmup exception 会向当前进程发送 SIGTERM。`fail_open` 的具体条件由 warmup helper解释，不能仅从调用处推成“任何 warmup 错误都可忽略”。

## 9. Disagg：入口、参数和 loop 都切换

`dispatch_launch` 根据 role 选择 monolithic、head server 或 standalone role。pool launcher为每个 role instance重建 ServerArgs，并覆盖：role、work/result endpoint、GPU 数、warmup、scheduler/master port，以及 role-specific TP/SP/Ulysses/Ring。

Scheduler 一进入非 monolithic role就调用 `_disagg_event_loop()` 并返回，不再使用本篇第 4 节的普通 queue 主循环。中间 `Req` 的 transport、slot 与超时属于 disaggregation 子系统，不能拿 monolithic ZMQ reply 直接类推。

## 10. 当前基线的可疑边界

- `run_scheduler_process` 保留两个未使用的 pipe 形参，实际 Scheduler 参数依赖调用位置，维护时容易接错。
- `scheduler_infos` 只在启动期收集，未形成持续健康注册。
- `dp_size` 字段存在，但 `dp_size > 1` 被拒绝。
- `health_generate` 当前尚未执行真实生成，只固定返回 ok，不能作为真实生成健康检查。
- server warmup 不覆盖具体 OpenAI route 转换。
- local spill 引入临时文件生命周期与磁盘容量依赖；成功生成不等于 client 一定 materialize 成功。

## 运行验证

```powershell
rg -n 'status.*ready|launch_http_server_only|launch_http_server' sglang/python/sglang/multimodal_gen/runtime/launch_server.py sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py
rg -n 'receiver = None|_normalize_received_payload|broadcast_pyobj|waiting_queue|get_next_batch_to_run|_try_merge_generation_reqs|_split_batched_output|return_result|_broadcast_task' sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py
rg -n 'execute_forward|return_req|_materialize_output_transport|spill_large_arrays_to_file_refs|materialize_file_refs' sglang/python/sglang/multimodal_gen/runtime/managers/gpu_worker.py sglang/python/sglang/multimodal_gen/runtime/managers/scheduler.py sglang/python/sglang/multimodal_gen/runtime/scheduler_client.py
rg -n 'server_warmup|warmup_done|health_generate|_disagg_event_loop|DP is not yet supported' sglang/python/sglang/multimodal_gen/runtime
```

预期：能从 ready barrier、ROUTER ownership、三次条件 broadcast、queue merge/split，一直定位到 worker transport、local spill、warmup 和 disagg 分流。GPU/NCCL、真实模型数值和性能仍需目标 Linux/GPU 环境验证。
