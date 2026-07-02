---
type: batch-doc
module: 32-CheckpointEngine
batch: "32"
doc_type: concept
title: "CheckpointEngine · 核心概念"
tags:
 - sglang/batch/32
 - sglang/module/checkpoint-engine
 - sglang/doc/concept
aliases:
 - "01-核心概念"
updated: 2026-07-02
---
# CheckpointEngine · 核心概念

> 本节介绍核心术语与模块在架构中的位置。

---

## 用户故事：RLHF rollout 权重热更新

### Persona

**吴研究员**，RLHF 在线 rollout 工程师。训练侧每 N 步产出新 checkpoint，需在 **不重启 SGLang server** 的情况下把 base model 权重推给推理集群，否则 rollout 窗口丢失 3–5 分钟。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | Server 以 `--load-format dummy --checkpoint-engine-wait-weights-before-ready` 启动，/ping 通但 inference 未 ready |
| T1 | torchrun 侧 `update.py` 加载 checkpoint，`ParameterServer` gather metas |
| T2 | rank0 POST `/update_weights_from_ipc`，body 含各 GPU UUID 的 `zmq_handles` |
| T3 | `FlattenedTensorBucket` IPC 灌权重 → `initial_weights_loaded=True` → warmup 完成，rollout 继续 |

**Explain：** CheckpointEngine 路径更新 **base model 全量权重**（非 LoRA）。外部 checkpoint-engine 经 ZMQ IPC 把扁平化 tensor 推到 `SGLangCheckpointEngineWorkerExtension.update_weights_from_ipc`；热更新期间 Scheduler 暂停部分请求（`num_paused_reqs`），完成后 flush radix cache。与ModelLoader ModelLoader weight_sync 共用 `tensor_bucket.py` 结构。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L69-L89
    def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
        """
        Update weights from IPC communication.
        Args:
            zmq_handles: Dict mapping device UUID to ZMQ socket path
        """
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        if device_uuid not in zmq_handles:
            raise ValueError(
                f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}"
            )
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=device_id,
            run=self.get_model_loader(),
            post_hook=self.get_post_hook(),
        )
```

**Comment：** GPU UUID 必须与 `zmq_handles` key 一致；`wait_weights_before_ready` 下 `/ping` ≠ 权重就绪，需等首次 IPC 成功。

### 如果…会怎样（调试）

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| Device UUID not in zmq_handles | torchrun rank 与 inference TP 切片不对 | 核对 `inference_parallel_size` 与 `update.py` 切片 |
| 一直 not ready | 未 POST 或 IPC 超时 | 查 `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT` |
| 更新后输出漂移 | radix cache 未 flush | 确认 POST body `flush_cache: true` |

---

## 1. 运行时热更新场景

**Explain：** 训练侧 checkpoint-engine 在独立 torchrun 进程加载新权重，经 ZMQ IPC 推送到已运行的 SGLang server，无需重启进程。适用于 RLHF 在线 rollout、A/B 权重切换、dummy 启动后延迟灌权重等场景。与 LoRA adapter 热加载（不同 API）不同，本路径更新 **base model** 全量权重。

**Comment：** 热更新期间 Scheduler 暂停部分请求（`num_paused_reqs`），完成后 flush radix cache；metrics 见 可观测性。

---

## 2. wait_weights_before_ready

**Explain：** 启动时可加 `--checkpoint-engine-wait-weights-before-ready`：server 在 `initial_weights_loaded=True` 之前不执行 warmup、不标记 Up。外部 update 脚本通过 `/ping` 确认 HTTP listen 后 POST `/update_weights_from_ipc` 灌权重。常配合 `--load-format dummy` 跳过初始磁盘加载。

**Code：**

```python
# 来源：python/sglang/srt/server_args.py L2505-L2508
    checkpoint_engine_wait_weights_before_ready: A[
        bool,
        "If set, the server will wait for initial weights to be loaded via checkpoint-engine or other update methods before serving inference requests.",
    ] = False
```

**Comment：**

- 超时由 `SGLANG_WAIT_WEIGHTS_READY_TIMEOUT` 环境变量控制。
- `/ping` 只表示进程活着，不代表权重已就绪。

---

## 3. initial_weights_loaded 标志

**Explain：** TokenizerManager 在 `init_weight_update` 中初始化该标志；wait 模式下初始为 False，首次 IPC 更新成功后 http_server 设为 True，`_wait_weights_ready` 循环退出，继续 warmup。

**Code：**

```python
# 来源：python/sglang/srt/managers/tokenizer_manager.py L459-L463
    def init_weight_update(self):
        # Initial weights status
        self.initial_weights_loaded = True
        if self.server_args.checkpoint_engine_wait_weights_before_ready:
            self.initial_weights_loaded = False
```

**Comment：**

- 影响 server_status 与对外 ready probe。
- 与 `_wait_and_warmup` 启动顺序紧密耦合。

---

## 4. FlattenedTensorBucket

**Explain：** 多个 named tensor 扁平拼接为单个 uint8 向量，附带 `FlattenedTensorMetadata` 记录 name/shape/dtype/offset，接收端按 metadata 重建视图并 `load_weights`。减少 ZMQ 多次 small message 开销。

**Code：**

```python
# 来源：python/sglang/srt/weight_sync/tensor_bucket.py L19-L72
class FlattenedTensorBucket:
    """
    A bucket that flattens multiple tensors into a single tensor for efficient processing
    while preserving all metadata needed for reconstruction.
    """

    # This field is solely for users of to check whether the class supports this feature
    supports_multi_dtypes = True

    def __init__(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]] = None,
        flattened_tensor: torch.Tensor = None,
        metadata: List[FlattenedTensorMetadata] = None,
    ):
        """
        Initialize a tensor bucket from a list of named tensors OR from pre-flattened data.
        Args:
            named_tensors: List of (name, tensor) tuples (for creating new bucket)
            flattened_tensor: Pre-flattened tensor (for reconstruction)
            metadata: Pre-computed metadata (for reconstruction)
        """
        if named_tensors is not None:
            # Create bucket from named tensors
            self.metadata: List[FlattenedTensorMetadata] = [None] * len(named_tensors)
            self.flattened_tensor: torch.Tensor = None

            if not named_tensors:
                raise ValueError("Cannot create empty tensor bucket")

            # Collect metadata and flatten tensors
            current_idx = 0
            flattened_tensors: List[torch.Tensor] = [None] * len(named_tensors)

            for i, (name, tensor) in enumerate(named_tensors):
                flattened = tensor.flatten().view(torch.uint8)
                flattened_tensors[i] = flattened

                # Store metadata

                numel = flattened.numel()
                metadata_obj = FlattenedTensorMetadata(
                    name=name,
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                    start_idx=current_idx,
                    end_idx=current_idx + numel,
                    numel=numel,
                )
                self.metadata[i] = metadata_obj
                current_idx += numel

            # Concatenate all flattened tensors
            self.flattened_tensor = torch.cat(flattened_tensors, dim=0)
```

**Comment：**

- 与 12-ModelLoader weight_sync 共用结构。
- 空 bucket 抛 ValueError。

---

## 5. FlattenedTensorMetadata

**Explain：** 每个 tensor 在 flat buffer 中的 start_idx/end_idx/numel；load 时 slice flattened_tensor 再 view 回原始 shape/dtype。flatten 时使用 `tensor.flatten().view(torch.uint8)` 按字节对齐。

**Code：**

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

**Comment：**

- supports_multi_dtypes=True 标识 bucket 支持混合 dtype。
- IPC 接收端从 metadata 反序列化重建 named_tensors。

---

## 6. SGLangCheckpointEngineWorkerExtension

**Explain：** 封装 MoonshotAI checkpoint-engine 的 `update_weights_from_ipc`；ModelRunner 集成 `Impl` 子类提供 device_uuid 与 load_weights 回调。zmq_handles dict key 为 GPU UUID 字符串（如 `GPU-xxxx`）。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/checkpoint_engine_worker.py L69-L89
    def update_weights_from_ipc(self, zmq_handles: Dict[str, str]):
        """
        Update weights from IPC communication.
        Args:
            zmq_handles: Dict mapping device UUID to ZMQ socket path
        """
        if self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        device_uuid = self.get_device_uuid()
        device_id = self.get_device_id()
        if device_uuid not in zmq_handles:
            raise ValueError(
                f"Device UUID {device_uuid} not found in zmq_handles: {list(zmq_handles.keys())}"
            )
        update_weights_from_ipc(
            self._zmq_ctx,
            zmq_handles[device_uuid],
            device_id=device_id,
            run=self.get_model_loader(),
            post_hook=self.get_post_hook(),
        )
```

**Comment：**

- 依赖 `pip install sglang[checkpoint-engine]`。
- post_hook 处理 quant_method.process_weights_after_loading。

---

## 7. update.py 外部脚本

**Explain：** torchrun 启动 ParameterServer，register_checkpoint → gather_metas → ps.update → HTTP 触发 srt IPC load。rank0 在 inference_parallel 组首 rank 向 endpoint POST zmq_handles JSON。

**Code：**

```python
# 来源：python/sglang/srt/checkpoint_engine/update.py L108-L134
def req_inference(
    endpoint: str,
    inference_parallel_size: int,
    timeout: float = 300.0,
    uds: str | None = None,
    weight_version: str | None = None,
) -> Callable[[list[tuple[str, str]]], None]:
    rank = int(os.getenv("RANK", 0))
    src = rank // inference_parallel_size * inference_parallel_size

    def req_func(socket_paths: list[tuple[str, str]]):
        if rank == src:
            with httpx.Client(transport=httpx.HTTPTransport(uds=uds)) as client:
                resp = client.post(
                    f"{endpoint}/update_weights_from_ipc",
                    json={
                        "zmq_handles": dict(
                            socket_paths[src : src + inference_parallel_size]
                        ),
                        "flush_cache": True,
                        "weight_version": weight_version,
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()

    return req_func
```

**Comment：**

- broadcast / p2p / all 三种 update_method。
- inference_parallel_size 决定 torchrun nproc 与 zmq_handles 切片。

---

## 8. 术语对照

| 术语 | 含义 | 源码 |
|------|------|------|
| `ParameterServer` | checkpoint-engine 外部权重服务 | update.py |
| `FlattenedTensorBucket` | 多 tensor 扁平化传输容器 | tensor_bucket.py |
| `SGLangCheckpointEngineWorkerExtensionImpl` | ModelRunner IPC 桥接 | checkpoint_engine_worker.py |
| `UpdateWeightsFromIPCReqInput` | HTTP body 结构 | io_struct.py |
| `wait_weights_before_ready` | 启动等待权重屏障 | server_args.py |
| `initial_weights_loaded` | 权重就绪标志 | tokenizer_manager.py |
