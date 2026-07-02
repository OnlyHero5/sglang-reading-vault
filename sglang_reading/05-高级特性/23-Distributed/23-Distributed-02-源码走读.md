---
type: batch-doc
module: 23-Distributed
batch: "23"
doc_type: walkthrough
title: "分布式并行 · 源码走读"
tags:
 - sglang/batch/23
 - sglang/module/distributed
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# 分布式并行 · 源码走读

## 走读顺序

1. `parallel_state.py` — 初始化与 GroupCoordinator
2. `communication_op.py` — 对外 API
3. `device_communicators/custom_all_reduce.py` — 小 tensor 优化
4. `data_parallel_controller.py` — DP 路由

---

## 1. init_distributed_environment

**Explain：** 设置 MASTER_ADDR/PORT、初始化 torch.distributed，并记录 model parallel group timeout。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L79-L81
# Reuse the user-provided distributed timeout for model-parallel subgroup
# creation so runtime collectives do not silently fall back to backend defaults.
_MODEL_PARALLEL_GROUP_TIMEOUT: Optional[timedelta] = None
```

**Comment：** 子 group 创建复用用户 timeout，避免 collective 静默使用 backend 默认短超时。

---

## 2. GroupCoordinator.__init__ 设备绑定

**Explain：** 按平台选择 cuda/npu/xpu/musa device；`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS` 时 device_id 固定为 0。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L286-L298
        if is_cuda_alike():
            device_id = (
                0 if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() else local_rank
            )
            self.device = torch.device(f"cuda:{device_id}")
        elif _is_npu:
            self.device = torch.device(f"npu:{local_rank}")
        elif _is_xpu:
            self.device = torch.device(f"xpu:{local_rank}")
        elif _is_musa:
            self.device = torch.device(f"musa:{local_rank}")
        else:
            self.device = torch.device("cpu")
```

**Comment：** 容器内每进程仅可见一张卡时常设 `ONE_VISIBLE_DEVICE_PER_PROCESS=1`。

---

## 3. CustomAllReduce 初始化

**Explain：** GroupCoordinator 在 `use_custom_allreduce=True` 且 world_size>1 时尝试 dispatch Custom AR；失败则 warning 并回退 NCCL。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L401-L416
        if use_custom_allreduce and self.world_size > 1:
            # Initialize a custom fast all-reduce implementation.
            try:
                CAClass = dispatch_custom_allreduce(
                    group=self.cpu_group,
                    device=self.device,
                )
                self.ca_comm = CAClass(
                    group=self.cpu_group,
                    device=self.device,
                )
            except Exception as e:
                logger.warning(
                    f"Setup Custom allreduce failed with {e}. To silence this "
                    "warning, specify --disable-custom-all-reduce explicitly."
                )
```

**Comment：** HIP 平台额外尝试 QuickAllReduce（AMD gfx942+）；与 Custom AR 互补。

---

## 4. Custom op 注册 all_to_all

**Explain：** 为 torch.compile / CUDA Graph 注册可追踪的 collective custom op，按 group_name 查找弱引用 GroupCoordinator。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L205-L213
@register_custom_op(mutates_args=["output"])
def reg_all_to_all_single(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_to_all_single(output, input)
```

**Comment：** `_groups` 存 weakref，destroy 后调用会 ValueError。

---

## 5. fused allreduce + rmsnorm

**Explain：** 大模型 decode 热点路径；GroupCoordinator 内选择 fused kernel 或回退 generic 路径。

**Code：**

```python
# 来源：python/sglang/srt/distributed/communication_op.py L28-L40
def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)
```

**Comment：** 返回 None 时 caller 走分离 all_reduce + rmsnorm。

---

## 6. Attention / MoE 专用组

**Explain：** DP Attention 将 attention 计算与 MoE 路由解耦到不同 parallel group。

**Code：**

```python
# 来源：python/sglang/srt/distributed/communication_op.py L65-L84
def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().all_reduce(input_)


def attention_tensor_model_parallel_quant_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().quant_all_reduce(input_)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    return get_moe_ep_group().all_reduce(input_)
```

**Comment：** MoE 层 forward 在 router all_to_all 后可能再经 EP all_reduce 同步 expert 输出。

---

## 7. DataParallelController 进程模型

**Explain：** Controller 主进程 ZMQ 接收 TokenizerManager 消息，按负载均衡策略转发到各 DP rank 的 Scheduler 子进程。

**Code：**

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L46-L54
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.observability.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.observability.req_time_stats import DPControllerReqTimeStats
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
```

**Comment：** 每个 DP worker 独立 `run_scheduler_process`；端口由 PortArgs 偏移分配。

---

## 8. DataParallelController 负载均衡

**Explain：** DP Controller 根据 `LoadBalanceMethod` 选择下一 Scheduler rank；PD 场景用 bootstrap room 哈希保持 locality。

**Code：**

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L76-L80
class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
```

**Comment：** `run_scheduler_process` 为每个 DP rank 启动独立 Scheduler 子进程，ZMQ 端口由 PortArgs 偏移。

---

## 9. GraphCaptureContext

**Explain：** CUDA Graph capture 时使用独立 stream，避免与 default stream collective 交错。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L102-L104
@dataclass
class GraphCaptureContext:
    stream: torch.get_device_module().Stream
```

**Comment：** GroupCoordinator 在 graph capture 期间切换通信 backend（如禁用 custom AR）。

---

## 10. shm_broadcast 与 MQ Broadcaster

**Explain：** 小消息（metadata、control）可走 shared memory broadcast，降低 NCCL latency。

**Code：**

```python
# 来源：python/sglang/srt/distributed/device_communicators/shm_broadcast.py L444-L476
    def enqueue(self, obj):
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.n_local_reader > 0:
            if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                with self.acquire_write() as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send(serialized_obj)
            else:
                with self.acquire_write() as buf:
                    buf[0] = 0  # not overflow
                    buf[1 : len(serialized_obj) + 1] = serialized_obj
        if self.n_remote_reader > 0:
            self.remote_socket.send(serialized_obj)

    def dequeue(self):
        if self._is_local_reader:
            with self.acquire_read() as buf:
                overflow = buf[0] == 1
                if not overflow:
                    # no need to know the size of serialized object
                    # pickle format contains the size information internally
                    # see https://docs.python.org/3/library/pickle.html
                    obj = pickle.loads(buf[1:])
            if overflow:
                recv = self.local_socket.recv()
                obj = pickle.loads(recv)
        elif self._is_remote_reader:
            recv = self.remote_socket.recv()
            obj = pickle.loads(recv)
        else:
            raise RuntimeError("Only readers can dequeue")
        return obj
```

**Comment：** 仅 world_size>1 且 enable 时创建；见 GroupCoordinator `use_message_queue_broadcaster`。

---

## 11. tensor dict 拆分广播

**Explain：** `_split_tensor_dict` 将 Python dict 中的 tensor 与 metadata 分离，便于 pickle + broadcast。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L113-L119
def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]],
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
```

**Comment：** `broadcast_tensor_dict` 在 TP rank0 向组内广播权重更新或 control msg。

---

## 12. 走读小结

```text
init_distributed_environment
 └─ initialize_model_parallel → GroupCoordinator (TP/PP/EP/DP/Attn...)
 ├─ communication_op.*_all_reduce ← 模型层调用入口
 ├─ CustomAllReduce / shm_broadcast ← 小消息 & 小 tensor 优化
 └─ DataParallelController ← 多 Scheduler 进程路由
```

**下一专题关联：** MoE EP 细节见 [[18-MoE-00-MOC]]；PD 路由见 [[22-Disaggregation-00-MOC]]。
