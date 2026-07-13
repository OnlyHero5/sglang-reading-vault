---
title: "CheckpointEngine · 核心概念"
type: concept
framework: sglang
topic: "CheckpointEngine"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-12
---
# CheckpointEngine · 核心概念

## 读者任务

这篇先把概念边界讲清楚：CheckpointEngine 不是 LoRA 热加载，也不是 SGLang 自己实现的 checkpoint 分发系统。它是一条“外部权重服务 + SGLang serving 侧适配”的运行时 base model 权重替换通道。

读完后，你应该能解释五个对象：`checkpoint_engine_wait_weights_before_ready`、`initial_weights_loaded`、`UpdateWeightsFromIPCReqInput`、`FanOutCommunicator`、`SGLangCheckpointEngineWorkerExtensionImpl`。

## 心理模型：四道门

| 门 | 谁负责 | 保护什么 |
|----|--------|----------|
| 启动等待门 | HTTP server + TokenizerManager | warmup 前最多等待一段时间；超时只报错，不会 fail closed |
| 控制面门 | HTTP endpoint + TokenizerManager | 外部 update 请求必须串行化并返回明确结果 |
| 执行面门 | Scheduler WeightUpdater | 计时、更新 target/draft、条件 flush、TP barrier |
| 适配门 | ModelRunner + checkpoint-engine worker | 按 GPU UUID 连接 ZMQ，调用 `model.load_weights` 和 post hook |

把这四道门分开后，很多问题会变简单：`/ping` 成功只说明 HTTP 可连接；`initial_weights_loaded=True` 说明 IPC endpoint 已接受一次 success；`ServerStatus.Up` 还要等 warmup 或 skip-warmup 分支。反过来，状态位为 False 也不是永久阻塞：等待超时后代码会继续往下走。

## 概念 1：等待权重是显式启动模式

默认 SGLang 不等待外部权重。显式打开 `checkpoint_engine_wait_weights_before_ready` 后，server 会在 warmup 前轮询状态位；但该轮询有超时，超时路径只记录 error，随后继续 warmup。因此它是有界延迟门，不是“权重不来就永不服务”的硬门。

```python
# 来源：python/sglang/srt/server_args.py L2505-L2508
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
    ] = False
```

这个开关常和 `--load-format dummy` 配合：进程先占 GPU、HTTP 先可达，外部训练或参数服务稍后灌入真实权重。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2173-L2191
def _wait_weights_ready():
    """Wait for weights to be ready within the specified timeout."""
    timeout = WAIT_WEIGHTS_READY_TIMEOUT
    start_time = time.time()

    for _ in range(timeout):
        if _global_state.tokenizer_manager.initial_weights_loaded:
            logger.info(
                f"Weights are ready after {time.time() - start_time:.2f} seconds"
            )
            return
        time.sleep(1)

    # Timeout reached without weights being ready
    logger.error(
        f"Weights are not ready after waiting {timeout} seconds. "
        f"Consider increasing SGLANG_WAIT_WEIGHTS_READY_TIMEOUT environment variable. "
        f"Current status: initial_weights_loaded={_global_state.tokenizer_manager.initial_weights_loaded}"
    )
```

## 概念 2：`initial_weights_loaded` 是权重 ready 状态位

TokenizerManager 初始化权重更新状态时，普通模式把 `initial_weights_loaded` 设为 `True`；等待模式把它设为 `False`。同一段初始化还创建模型更新锁和 pause 条件。

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L459-L472
    def init_weight_update(self):
        # Initial weights status
        self.initial_weights_loaded = True
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False

        # Weight updates
        # The event to notify the weight sync is finished.
        self.model_update_lock = RWLock()
        self.model_update_result: Optional[Awaitable[UpdateWeightFromDiskReqOutput]] = (
            None
        )
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
```

所以这个状态不是健康检查结果，也不是某个文件是否存在。它是启动等待循环和 IPC HTTP endpoint 之间的共享事实。尽管参数帮助文本提到 “other update methods”，当前基线中把该状态翻为 True 的赋值只出现在 `/update_weights_from_ipc` 成功分支。

## 概念 3：IPC 请求只传 handles，不传 tensor

`UpdateWeightsFromIPCReqInput` 明确说这个请求只用于 Checkpoint Engine。它的核心字段是 GPU UUID 到 ZMQ socket path 的映射，另外带 cache flush、权重版本和 empty cache 控制。

```python
# 来源：python/sglang/srt/managers/io_struct.py L1600-L1615
# Now UpdateWeightsFromIPCReqInput and UpdateWeightsFromIPCReqOutput
# are only used by Checkpoint Engine (https://github.com/MoonshotAI/checkpoint-engine)
class UpdateWeightsFromIPCReqInput(BaseReq, kw_only=True):
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]
    # Whether to flush cache after weight update
    flush_cache: bool = True
    # Optional: Update weight version along with weights
    weight_version: Optional[str] = None
    # Whether to call torch.cuda.empty_cache() during flush
    torch_empty_cache: bool = False


class UpdateWeightsFromIPCReqOutput(BaseReq, kw_only=True):
    success: bool
    message: str
```

这解释了为什么 HTTP body 很小。权重数据不走 HTTP；HTTP 只通知 SGLang 每张卡应该连接哪个 ZMQ handle。`weight_version` 也不是 loader 的校验 token：ModelRunner 只消费 `zmq_handles`，版本字符串是在控制面 success 后写入 TokenizerManager 的 `server_args`，供响应元数据与查询暴露。

## 概念 4：SGLang 只在成功后释放启动等待

HTTP endpoint 把请求交给 TokenizerManager。只有 update 成功时，才会把初始权重状态置为 ready；失败时返回 400。

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1316-L1322
    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)
```

这条语义很重要：外部 `update.py` 看到 `/ping` 成功，只能开始尝试 update；不能把 `/ping` 当成权重 ready。还要注意 DP-Attention 情况下当前 IPC 控制面会等待全部 fan-out response，却只读取结果列表的第一个元素来决定 success 和 weight version，不能把 HTTP 200 外推为每个 scheduler 都成功。

## 概念 5：并发控制与结果聚合是两件事

普通运行状态下，请求持有 `model_update_lock.reader_lock`，IPC update 获取 writer lock，因此会等待已有请求离开并阻止新请求进入。若服务已处于 paused 状态，IPC update 会在持有 `is_pause_cond` 时直接 fan-out，不再获取 writer lock；这里依赖 pause 调用方已经建立的停机语义。

```python
# 来源：python/sglang/srt/managers/tokenizer_control_mixin.py L500-L513
            async with self.is_pause_cond:
                is_paused = self.is_pause
                if is_paused:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message

            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg
```

`FanOutCommunicator` 会等齐 `dp_size` 个 response 才返回 list，但这里没有像 distributed/tensor update 那样调用 `merge_results`，而是只取 `[0]`。因此 DP-Attention 虽通过入口断言，控制面 success 仍只是第一个 response 的结果。这是当前实现边界，不是“所有 DP rank 已原子提交”的证明。

## 概念 6：WeightUpdater 把热更新纳入 serving 执行语义

IPC 更新在 `_observe_weight_load("ipc")` 里执行。主 target worker 成功后才会尝试 draft worker，并调用 cache flush helper；helper 仍受请求的 `flush_cache` 开关控制。之后执行 TP CPU-group barrier，再返回结构化输出。

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L166-L178
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            success, message = self.tp_worker.update_weights_from_ipc(recv_req)
            tp_success = success
            if success and self.draft_worker is not None:
                success, message = self.draft_worker.update_weights_from_ipc(recv_req)
            if tp_success:
                self.flush_cache_after_weight_update(recv_req)
            if not success:
                logger.error(message)
            torch.distributed.barrier(group=self.tp_cpu_group)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)
```

这段代码给出三个条件化事实：target 成功才进入 flush helper；是否真正 flush 由 `recv_req.flush_cache` 决定；draft 失败会把最终结果改成失败，但 target 已经修改，且请求要求时 cache 也已经清理。这里没有回滚。`_observe_weight_load` 位于 `finally`，所以 gauge 表示一次 update 尝试的耗时，失败也可能更新。

```python
# 来源：python/sglang/srt/managers/scheduler_components/weight_updater.py L86-L99
    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )
```

## 概念 7：worker extension 用 GPU UUID 选本 rank handle

每个 rank 拿到的是同一个 `zmq_handles` dict，但只能连接当前 GPU 对应的 socket path。源码用当前 CUDA device 的 UUID 做 key。

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L102-L117
    def get_device_uuid(self) -> str:
        """Get the UUID of current device."""
        # Get device UUID for current device
        device_id = torch.cuda.current_device()
        try:
            return f"GPU-{torch.cuda.get_device_properties(device_id).uuid!s}"
        except AssertionError as e:
            raise ValueError(f"Failed to get GPU UUID for device {device_id}") from e

    def get_device_id(self) -> int:
        """Get the device ID."""
        return torch.cuda.current_device()

    def get_model_loader(self) -> Callable:
        """Get the model weight loader function."""
        return self.model_runner.model.load_weights
```

如果外部脚本的 rank 切片、GPU 映射或 `inference_parallel_size` 没覆盖该 endpoint 会触达的全部 worker UUID，最常见的错误就是当前 UUID 不在 `zmq_handles` 里。单机纯 TP 常恰好等于 TP；多节点或含 DP 的拓扑应按实际 inference processes / UUID 集合判断。

## 概念 8：post hook 补齐热更新后的模型处理

IPC 加载最终会调用模型的 `load_weights`，但量化模块或模型本身还可能需要加载后处理。SGLang 的 post hook 遍历模块上的 `quant_method`，再调用模型级 `post_load_weights`。

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L122-L141
        def post_hook():
            # Perform post-processing after weight loading similar to DefaultModelLoader
            try:
                from sglang.srt.model_loader.loader import device_loading_context

                # Process quantization methods after loading weights
                for _, module in self.model_runner.model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        # Move parameters to device if needed for quantization processing
                        target_device = torch.device(
                            "cuda", torch.cuda.current_device()
                        )
                        with device_loading_context(module, target_device):
                            quant_method.process_weights_after_loading(module)
                # Call model-specific post-loading hook if available
                if hasattr(self.model_runner.model, "post_load_weights"):
                    self.model_runner.model.post_load_weights()
            except Exception as e:
                logger.warning(f"Post-hook processing failed: {e}")
```

注意失败模式：post hook 异常只写 warning。这意味着量化模型接入 IPC 热更新时，必须额外关注日志，不能只看 HTTP success。

## 本篇结论

- CheckpointEngine 路径替换的是 base model 权重，不是 LoRA adapter。
- HTTP `/ping` 是可连接信号；`initial_weights_loaded` 是 IPC endpoint 的初始权重状态位，但不是 fail-closed readiness 保证。
- IPC update 的 HTTP 请求只携带 ZMQ handles，真实权重通过 checkpoint-engine worker 接收。
- TokenizerManager 在非 paused 路径用 writer lock 排除推理 reader；paused 路径绕过 writer lock，依赖既有 pause 语义。WeightUpdater 本身不会发起 pause。
- GPU UUID 是外部 handles 和本地 worker 对齐的关键。
- cache flush 是默认正确性保护，但协议允许显式关闭；duration gauge 记录尝试耗时，不等价于更新成功。

下一篇 [[SGLang-CheckpointEngine-源码走读]] 会沿一次真实 IPC 热更新走源码。
