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
updated: 2026-07-10
---
# CheckpointEngine · 核心概念

## 读者任务

这篇先把概念边界讲清楚：CheckpointEngine 不是 LoRA 热加载，也不是 SGLang 自己实现的 checkpoint 分发系统。它是一条“外部权重服务 + SGLang serving 侧适配”的运行时 base model 权重替换通道。

读完后，你应该能解释五个对象：`wait_weights_before_ready`、`initial_weights_loaded`、`UpdateWeightsFromIPCReqInput`、`SGLangCheckpointEngineWorkerExtensionImpl`、`FlattenedTensorBucket`。

## 心理模型：四道门

| 门 | 谁负责 | 保护什么 |
|----|--------|----------|
| 启动等待门 | HTTP server + TokenizerManager | dummy 启动后不要用未灌权重 warmup |
| 控制面门 | HTTP endpoint + TokenizerManager | 外部 update 请求必须串行化并返回明确结果 |
| 执行面门 | Scheduler WeightUpdater | 暂停、计时、更新 draft、flush cache、跨 rank barrier |
| 适配门 | ModelRunner + checkpoint-engine worker | 按 GPU UUID 连接 ZMQ，调用 `model.load_weights` 和 post hook |

把这四道门分开后，很多问题会变简单：`/ping` 成功只说明 HTTP 可连接；`initial_weights_loaded=True` 才说明初始灌权重成功；`ServerStatus.Up` 还要等 warmup 或 skip warmup 分支。

## 概念 1：等待权重是显式启动模式

默认 SGLang 不等待外部权重。只有显式打开 `checkpoint_engine_wait_weights_before_ready`，server 才会在服务推理前等待 checkpoint-engine 或其他 update method 完成初始权重加载。

```python
# 来源：python/sglang/srt/server_args.py L2505-L2508
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
    ] = False
```

这个开关常和 `--load-format dummy` 配合：进程先占 GPU、HTTP 先可达，外部训练或参数服务稍后灌入真实权重。

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

所以这个状态不是健康检查结果，也不是某个文件是否存在。它是启动等待和 HTTP update 成功之间的共享事实。

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

这解释了为什么 HTTP body 很小。权重数据不走 HTTP；HTTP 只通知 SGLang 每张卡应该连接哪个 ZMQ handle。

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

这条语义很重要：外部 `update.py` 看到 `/ping` 成功，只能开始尝试 update；不能把 `/ping` 当成权重 ready。

## 概念 5：WeightUpdater 把热更新纳入 serving 调度语义

IPC 更新在 `_observe_weight_load("ipc")` 里执行。主 TP worker 成功后，才会更新 draft worker、flush cache、打 barrier、返回结构化输出。

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

这段代码给出两个不变量：base model 更新成功后必须 flush cache；多 rank 更新必须跨 TP CPU group 对齐。draft worker 失败会让最终返回失败，但主模型成功后 cache 仍会清。

## 概念 6：worker extension 用 GPU UUID 选本 rank handle

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

如果外部脚本的 rank 切片、GPU 映射、`inference_parallel_size` 和 SGLang TP 不一致，最常见的错误就是当前 UUID 不在 `zmq_handles` 里。

## 概念 7：post hook 补齐热更新后的模型处理

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

## 概念 8：`FlattenedTensorBucket` 是通用权重同步结构

SGLang 的 `FlattenedTensorBucket` 把多个 named tensors 变成一个连续 byte buffer，并用 metadata 保存 name、shape、dtype 和偏移。它被模型执行层的其他权重同步路径使用；CheckpointEngine IPC 本身的接收由第三方 worker 驱动。

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L7-L16
@dataclass
class FlattenedTensorMetadata:
    """Metadata for a tensor in a flattened bucket"""

    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int
```

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L90-L107
    def reconstruct_tensors(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Reconstruct original tensors from flattened tensor with optimized performance.
        Uses memory-efficient operations to minimize allocations and copies.
        """
        # preallocate the result list
        reconstructed = [None] * len(self.metadata)

        for i, meta in enumerate(self.metadata):
            tensor = (
                self.flattened_tensor[meta.start_idx : meta.end_idx]
                .view(meta.dtype)
                .reshape(meta.shape)
            )

            reconstructed[i] = (meta.name, tensor)

        return reconstructed
```

这是一类“传输层连续、模型层结构化”的权重同步契约。不要把它误读成 CheckpointEngine HTTP body 的格式。

## 本篇结论

- CheckpointEngine 路径替换的是 base model 权重，不是 LoRA adapter。
- HTTP `/ping` 是可连接信号，`initial_weights_loaded` 才是初始权重 ready 信号。
- IPC update 的 HTTP 请求只携带 ZMQ handles，真实权重通过 checkpoint-engine worker 接收。
- Scheduler WeightUpdater 负责把热更新纳入 pause、metrics、flush、barrier 语义。
- GPU UUID 是外部 handles 和本地 worker 对齐的关键。
- `FlattenedTensorBucket` 是 SGLang 权重同步通用结构，和 CheckpointEngine 相关但不是外部 HTTP 控制协议本身。

下一篇 [[SGLang-CheckpointEngine-源码走读]] 会沿一次真实 IPC 热更新走源码。
