---
type: batch-doc
module: 23-Distributed
batch: "23"
doc_type: concept
title: "分布式并行 · 核心概念"
tags:
 - sglang/batch/23
 - sglang/module/distributed
 - sglang/doc/concept
aliases:
 - "01-核心概念"
updated: 2026-07-02
---
# 分布式并行 · 核心概念

## 用户故事：千卡集群上线 — TP8 + EP 路由对不齐

### Persona

**韩磊**，分布式推理负责人。在 8×H100 节点上跑 MoE 模型，`torchrun --nproc_per_node=8` 启动后 rank3 报 **expert parallel group size mismatch**。他需要理解 SGLang 如何从 PyTorch 接管 ProcessGroup，以及 `GroupCoordinator` 如何把 allreduce 路由到 PyNccl / CustomAllReduce。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | `init_distributed_environment` 初始化 world group |
| T1 | `initialize_model_parallel` 切 TP/PP/EP/DP 组 |
| T2 | ModelRunner 各 rank 加载分片权重 |
| T3 | MoE forward 经 EP group 做 expert dispatch |
| T4 | 观测 `parallel_state_wrapper` 快照用于 metrics |

### 如果…会怎样

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| NCCL timeout | rank 启动顺序 / 网卡 | `NCCL_DEBUG=INFO` |
| EP size 不匹配 | `--tp-size` 与 `--ep-size` 组合非法 | 见 [[23-Distributed-04-关键问题|04-关键问题]] |
| Graph 下 allreduce 失败 | 未用 graph-safe communicator | `GroupCoordinator` backend 选型 |

---

## 1. 分布式环境接管

**Explain：** SGLang 从 PyTorch 接管分布式环境：`parallel_state.py` 管理所有 ProcessGroup，按 tensor/pipeline/expert/data/context 维度切分；模型层调用 `communication_op` 而非直接 `torch.distributed`。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L9-L24
"""Distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
```

**Comment：** 源自 vLLM / Megatron-LM 模式；SGLang 扩展了 MoE EP、Attention DP/CP、Decode CP 等组。

---

## 2. GroupCoordinator

**Explain：** 每个并行组包装为 `GroupCoordinator`：持有 cpu_group / device_group，并按 tensor 大小与 CUDA Graph 模式路由到 PyNccl、CustomAllReduce、TorchSymmMem 等实现。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L216-L255
class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_pynccl: bool  # a hint of whether to use PyNccl
    use_pymscclpp: bool  # a hint of whether to use PyMsccl
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    use_torch_symm_mem_all_reduce: (
        bool  # a hint of whether to use TorchSymmMemAllReduce
    )
    use_message_queue_broadcaster: (
        bool  # a hint of whether to use message queue broadcaster
    )
    # communicators are only created for world size > 1
    pynccl_comm: Optional[Any]  # PyNccl communicator
    ca_comm: Optional[Any]  # Custom allreduce communicator
    torch_symm_mem_comm: Optional[Any]  # Torch symm mem communicator
    mq_broadcaster: Optional[Any]  # shared memory broadcaster
```

**Comment：** `local_rank` 用于绑 GPU；`rank_in_group` 是组内逻辑 rank，跨节点时与 global rank 不同。

---

## 3. 通信原语封装

**Explain：** `communication_op.py` 提供 TP/MoE/Attention 专用的 all_reduce / all_gather，内部委托对应 GroupCoordinator。

**Code：**

```python
# 来源：python/sglang/srt/distributed/communication_op.py L18-L20
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

**Comment：** 层代码（Linear、Attention）应调用这些 helper，保证与 graph capture 注册的 custom op 一致。

---

## 4. Data Parallel 控制器

**Explain：** 当 `--dp-size > 1` 时，`DataParallelController` 在 TokenizerManager 与多个 Scheduler 子进程间 ZMQ 路由请求，支持 ROUND_ROBIN 与 FOLLOW_BOOTSTRAP_ROOM 负载均衡。

**Code：**

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L14-L14
"""A controller that dispatches requests to multiple data parallel workers."""
```

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L76-L79
class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
```

**Comment：** PD 分离场景常用 FOLLOW_BOOTSTRAP_ROOM，使同一 room 的请求落到同一 DP rank。

---

## 5. 并行维度一览

| 缩写 | 含义 | 典型用途 |
|------|------|----------|
| TP | Tensor Parallel | 切分 Linear / Attention 权重 |
| PP | Pipeline Parallel | 层间流水线 |
| EP | Expert Parallel | MoE expert 分片 |
| DP | Data Parallel | 多副本吞吐 |
| Attn-CP | Context Parallel | 长上下文 KV 切分 |
| Decode-CP | Decode Context Parallel | Decode 阶段 KV 切分 |

**Code（8 GPU TP×PP 示例）：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L2002-L2009
    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
```

**Comment：** 相邻 rank 应同机（同 DGX box）以减少 NVLink 跨机流量。
