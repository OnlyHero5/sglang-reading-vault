---
type: batch-doc
module: 32-CheckpointEngine
batch: "32"
doc_type: walkthrough
title: "CheckpointEngine · 源码走读"
tags:
 - sglang/batch/32
 - sglang/module/checkpoint-engine
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# CheckpointEngine · 源码走读

## 走读顺序

1. `server_args.py` — wait_weights_before_ready
2. `http_server.py` — _wait_weights_ready / update_weights_from_ipc
3. `tokenizer_manager.py` — initial_weights_loaded
4. `checkpoint_engine_worker.py` — IPC 桥接
5. `weight_sync/tensor_bucket.py` — 扁平化
6. `checkpoint_engine/update.py` — 外部 ParameterServer 脚本
7. `model_runner.py` / `weight_updater.py` — Scheduler 集成

---

## 1. _wait_weights_ready

**Explain：** launch_server 在 warmup 前阻塞，每秒检查 `tokenizer_manager.initial_weights_loaded`；超时打 error 日志但不强制退出。与 `_wait_and_warmup` 配合，确保 dummy 启动后外部灌权重再 serving。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2150-L2191
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

    if server_args.delete_ckpt_after_loading:
        delete_directory(server_args.model_path)

    if server_args.debug_tensor_dump_input_file:
        kill_process_tree(os.getpid())

    if launch_callback is not None:
        launch_callback()


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

**Comment：**

- WAIT_WEIGHTS_READY_TIMEOUT 默认来自 env。
- HTTP 已 listen 但权重未灌时 /ping 仍 200。

---

## 2. update_weights_from_ipc HTTP 入口

**Explain：** Admin 可选鉴权的 POST 端点；TokenizerManager 转发 IPC 到 Scheduler，成功后设置 initial_weights_loaded。失败返回 400。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L1306-L1322
@app.post("/update_weights_from_ipc")
@auth_level(AuthLevel.ADMIN_OPTIONAL)
async def update_weights_from_ipc(
    obj: Annotated[UpdateWeightsFromIPCReqInput, Body()], request: Request
):
    """Update the weights from IPC (Inter-Process Communication) for checkpoint-engine integration."""
    success, message = await _global_state.tokenizer_manager.update_weights_from_ipc(
        obj, request
    )

    content = {"success": success, "message": message}
    if success:
        if _global_state.tokenizer_manager.initial_weights_loaded is False:
            _global_state.tokenizer_manager.initial_weights_loaded = True
        return ORJSONResponse(content)
    else:
        return ORJSONResponse(content, status_code=HTTPStatus.BAD_REQUEST)
```

**Comment：**

- body 含 zmq_handles、flush_cache、weight_version。
- 与 update_weights_from_tensor 区分（不同 io_struct）。

---

## 3. SGLangCheckpointEngineWorkerExtensionImpl

**Explain：** 绑定 model_runner：get_device_uuid 读 cuda device properties；get_model_loader 返回 model.load_weights；post_hook 处理 quant_method.process_weights_after_loading 与 model.post_load_weights。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L98-L117
    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner

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

**Comment：**

- post_hook 失败仅 warning，不阻断 load。
- 与 DefaultModelLoader 行为对齐。

---

## 4. post_hook 量化后处理

**Explain：** IPC load 完成后需与冷启动一样执行 quant_method.process_weights_after_loading，否则 INT8/FP8 等量化权重可能未正确 repack。

**Code：**

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

**Comment：**

- 量化模型热更新必须验证 post_hook 成功。
- 非量化模型 post_hook 基本 no-op。

---

## 5. ModelRunner.update_weights_from_ipc

**Explain：** Scheduler tp_worker 调用入口；创建 Impl 实例并传入 zmq_handles，捕获 ImportError（未安装 checkpoint-engine）与运行时异常。

**Code：**

```python
# 来源：python/sglang/srt/model_executor/model_runner.py L3246-L3261
    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)
```

**Comment：**

- EAGLE draft worker 也有独立 update_weights_from_ipc 转发。
- 成功返回 message 供 HTTP 响应。

---

## 6. weight_updater IPC 路径

**Explain：** Scheduler weight_updater 接收 TokenizerManager 转发的 IPC 消息，在 `_observe_weight_load("ipc")` 内暂停 running batch 后调用 model_runner 更新；成功后 flush_cache。

**Code：**

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

**Comment：**

- pause/resume 与 num_paused_reqs metrics 联动（见 31-Observability）。
- TP barrier 确保所有 rank 同步完成。

---

## 7. check_sglang_ready

**Explain：** update.py 中 inference_parallel 组首 rank 轮询 GET /ping，确认 HTTP 服务已 listen 再发起 weight update。支持 uds transport。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L49-L71
def check_sglang_ready(
    endpoint: str, inference_parallel_size: int, uds: str | None = None
):
    rank = int(os.getenv("RANK", 0))
    if rank != rank // inference_parallel_size * inference_parallel_size:
        return
    retry_num = 0
    transport = None
    if uds is not None:
        transport = httpx.HTTPTransport(uds=uds)
    with httpx.Client(transport=transport) as client:
        while True:
            try:
                response = client.get(f"{endpoint}/ping", timeout=10)
                response.raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError) as e:
                if retry_num % 10 == 0:
                    logger.warning(
                        f"fail to check sglang ready, retry {retry_num} times, error: {e}"
                    )
                retry_num += 1
                time.sleep(0.1)
```

**Comment：**

- retry 间隔 0.1s，每 10 次 warning。
- 非 src rank 直接 return。

---

## 8. update_weights 主流程

**Explain：** register_checkpoint → init_process_group → check_sglang_ready → barrier → gather_metas → ps.update(req_func)。req_func 在 update 完成时 POST HTTP 触发 srt load。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L137-L172
def update_weights(
    ps,
    checkpoint_name: str,
    checkpoint_files: list[str],
    named_tensors: dict[str, torch.Tensor],
    req_func: Callable[[list[tuple[str, str]]], None],
    inference_parallel_size: int,
    endpoint: str,
    save_metas_file: str | None = None,
    update_method: Literal["broadcast", "p2p", "all"] = "broadcast",
    uds: str | None = None,
):
    ps.register_checkpoint(
        checkpoint_name, files=checkpoint_files, named_tensors=named_tensors
    )
    ps.init_process_group()
    check_sglang_ready(endpoint, inference_parallel_size, uds)
    dist.barrier()
    with timer("Gather metas"):
        ps.gather_metas(checkpoint_name)
    if save_metas_file and int(os.getenv("RANK")) == 0:
        with open(save_metas_file, "wb") as f:
            pickle.dump(ps.get_metas(), f)

    if update_method == "broadcast" or update_method == "all":
        with timer("Update weights without setting ranks"):
            ps.update(checkpoint_name, req_func)

    if update_method == "p2p" or update_method == "all":
        if update_method:
            # sleep 2s to wait destroy process group
            time.sleep(2)
        with timer("Update weights with setting ranks"):
            ps.update(
                checkpoint_name, req_func, ranks=list(range(inference_parallel_size))
            )
```

**Comment：**

- p2p 前 sleep 2s 等待 destroy process group。
- save_metas_file 可选 dump metas 供 join 模式。

---

## 9. checkpoint-engine 依赖检查

**Explain：** 未安装 checkpoint-engine 包时 import 失败，模块级 raise ImportError 并提示 pip install sglang[checkpoint-engine]。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L25-L31
try:
    from checkpoint_engine.worker import update_weights_from_ipc
except ImportError:
    raise ImportError(
        "checkpoint-engine is not installed. "
        "Please install it with: pip install sglang[checkpoint-engine]"
    )
```

**Comment：**

- 仅热更新路径需要，普通推理不需要。
- update.py 对 ParameterServer 有 fallback logging。

---

## 10. FlattenedTensorBucket 反序列化

**Explain：** 可从 named_tensors 构造，或从 flattened_tensor+metadata 反序列化；ModelRunner IPC 路径使用后者，由 checkpoint-engine worker 在 ZMQ 接收后重建。

**Code：**

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L73-L80
        else:
            # Initialize from pre-flattened data
            if flattened_tensor is None or metadata is None:
                raise ValueError(
                    "Must provide either named_tensors or both flattened_tensor and metadata"
                )
            self.flattened_tensor = flattened_tensor
            self.metadata = metadata
```

**Comment：**

- torch.cat 拼接所有 flattened slice（构造路径）。
- 见 model_runner / checkpoint_engine worker 集成。

---

## 11. UpdateWeightsFromIPCReqInput

**Explain：** HTTP body 含 zmq_handles（GPU UUID → ZMQ socket path）、flush_cache、weight_version；仅 checkpoint-engine 使用，admin API。

**Code：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L1598-L1605


# Now UpdateWeightsFromIPCReqInput and UpdateWeightsFromIPCReqOutput
# are only used by Checkpoint Engine (https://github.com/MoonshotAI/checkpoint-engine)
class UpdateWeightsFromIPCReqInput(BaseReq, kw_only=True):
    # ZMQ socket paths for each device UUID
    zmq_handles: Dict[str, str]
    # Whether to flush cache after weight update
```

**Comment：**

- weight_version 写入 server_args 供客户端查询。
- flush_cache 默认 true，热更新后清 radix cache。

---

## 12. _wait_and_warmup 启动顺序

**Explain：** wait 权重 → execute_warmup → 打印 "fired up"。skip_server_warmup 时直接设 ServerStatus.Up。wait 模式下未灌权重则永不 serving。

**Code：**

```python
# 来源：python/sglang/srt/entrypoints/http_server.py L2145-L2161
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

**Comment：**

- 适合 RL 训练 loop：先起 engine、后灌权重。
- warmup 使用灌入后的真实权重。
