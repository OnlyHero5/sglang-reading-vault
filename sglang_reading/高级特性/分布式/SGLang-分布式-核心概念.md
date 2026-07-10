---
title: "分布式 · 核心概念"
type: concept
framework: sglang
topic: "分布式"
learning_role: core
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/concept
  - source-reading
updated: 2026-07-10
---
# 分布式 · 核心概念

## 读者任务

这篇先不追调用链，而是给 Distributed 建一个可复用的心理模型：同一个进程有 global rank、local rank、TP rank、Attention rank、MoE rank、PP rank、DP rank 等多套身份。读者读完要能解释两件事：

- 为什么同一个 `rank=3` 在 TP group、MoE EP group、PP group 里的邻居可能完全不同。
- 为什么 `DataParallelController` 虽然叫 data parallel，却不是模型层 all-reduce 的入口。

## 先建立模型：一张 rank 身份证

把每个进程想成一张身份证。身份证正面是 global rank，背面贴了多张贴纸：

| 贴纸 | 说明 | 常见消费者 |
|------|------|------------|
| `local_rank` | 本机第几张设备 | 设备绑定、进程启动 |
| `rank_in_group` | 当前 group 内第几个成员 | collective 源/目标、group 内顺序 |
| TP | 同一层张量如何切 | Linear、Attention、logits gather |
| Attention TP/CP/DP | Attention 特有的张量、上下文、副本切分 | attention backend、PD poll |
| MoE EP/TP/DP | expert、MoE tensor、MoE data 的切分 | token dispatcher、expert GEMM |
| PP | 层间流水线阶段 | pipeline stage 通信 |
| DP Controller rank | 请求应该给哪个 Scheduler worker | TokenizerManager 到 Scheduler 的路由 |

源码里 `GroupCoordinator` 直接把这个差异写在注释里：`local_rank` 用于设备，`rank_in_group` 用于组内逻辑位置。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L227-L241
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
```

这个模型的关键不是“有很多 rank”，而是“每条通信只能在自己的坐标系里解释”。TP all-reduce 的邻居不能拿 MoE EP 的邻居来推，PD poll 的 CPU group 也不能拿 NCCL device group 来替。

## WORLD：先把所有进程放进一张总表

`init_distributed_environment` 做第一件事：如果 PyTorch 分布式还没初始化，就调用 `torch.distributed.init_process_group` 建立 WORLD。随后 SGLang 用 `init_world_group` 包成 `_WORLD`。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L1931-L1939
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            pg_options=pg_options,
        )
```

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L1955-L1960
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(
            ranks, local_rank, backend, recovered_rank=recovered_rank
        )
```

WORLD 只解决“所有 rank 站在同一张总表上”。模型层真正使用的 TP、PP、MoE、Attention group 要在下一步切出来。

## GroupCoordinator：每个 group 的边界对象

`GroupCoordinator` 不是薄薄包一层 `ProcessGroup`。它同时承担四件事：

- 记住本 rank 在当前 group 里的 `ranks`、`world_size`、`rank_in_group`。
- 同时创建 device group 与 gloo CPU group。
- 保存 PyNccl、CustomAllReduce、Torch symmetric memory、message queue 等 communicator 策略。
- 给 graph capture 和 custom op 提供稳定的 group 名称。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L216-L225
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
```

构造时，它会为每组 ranks 创建 device group；同时创建一个 gloo CPU group，用于 CPU 侧协调。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L320-L332
            else:
                pg_options = get_torch_distributed_pg_options(group_name)
                device_group = torch.distributed.new_group(
                    ranks,
                    backend=torch_distributed_backend,
                    pg_options=pg_options,
                    timeout=subgroup_timeout,
                )
                # a group with `gloo` backend, to allow direct coordination
                # between processes through the CPU.
                cpu_group = torch.distributed.new_group(
                    ranks, backend="gloo", timeout=gloo_timeout
                )
```

这就是后面 PD poll 能用 CPU group、模型 forward 能用 device group 的根源。它们不是两个项目里的巧合，而是同一个 group object 暴露出的两条通道。

## 模型 collective：只从 helper 进入

模型层的心智负担应该是“我要在 TP group 上 reduce”，而不是“这次用 NCCL、PyNccl、CustomAllReduce 还是 symmetric memory”。所以 `communication_op.py` 把不同语义的 collective 做成稳定入口。

```python
# 来源：python/sglang/srt/distributed/communication_op.py L43-L47
def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)
```

```python
# 来源：python/sglang/srt/distributed/communication_op.py L65-L67
def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().all_reduce(input_)
```

```python
# 来源：python/sglang/srt/distributed/communication_op.py L77-L84
def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe parallel group."""
    return get_moe_tp_group().all_reduce(input_)


def moe_expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across moe expert parallel group."""
    return get_moe_ep_group().all_reduce(input_)
```

读者要记住的不是 helper 数量，而是命名里的语义：TP、Attention TP、MoE TP、MoE EP 是不同坐标系，不能随手互换。

## DP Controller：请求分单，不是张量同步

`DataParallelController` 的职责是把 TokenizerManager 发来的请求派给多个 Scheduler worker。它维护 ZMQ socket、worker 状态、负载预算和 dispatch 方法。

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L153-L165
        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]
        self.refresh_load_budget_on_dispatch = self.load_balance_method in (
            LoadBalanceMethod.TOTAL_REQUESTS,
            LoadBalanceMethod.TOTAL_TOKENS,
        )
```

这条链路的对象是请求，不是 GPU tensor。DP 路由错了，通常表现为请求落错 worker、bootstrap room 不一致、负载倾斜；TP collective 错了，通常表现为 group size mismatch、NCCL timeout、shape 或 graph capture 问题。

## ParallelState：给下游组件的一张快照

并行坐标最终会被压成不可变快照，供 Scheduler 和子组件读取。它不是构造 group 的地方，而是“当前 worker 的坐标事实”。

```python
# 来源：python/sglang/srt/distributed/parallel_state_wrapper.py L5-L23
@dataclass(frozen=True, slots=True, kw_only=True)
class ParallelState:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    dp_rank: Optional[int]
    dp_size: int
    attn_tp_rank: int
    attn_tp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    attn_dp_rank: int
    attn_dp_size: int
    moe_ep_rank: int
    moe_ep_size: int
    moe_dp_rank: Optional[int]
    moe_dp_size: int
    gpu_id: int
```

如果你在某个下游模块看到 `self.ps.tp_rank` 或 `self.ps.moe_ep_rank`，它通常是在消费这张快照，而不是重新解释 `torch.distributed`。

## 复盘

- Distributed 的第一层是不变的 global rank 表；第二层才是 TP/PP/Attention/MoE 等投影。
- `GroupCoordinator` 是模型 collective 的边界对象，内部同时有 device group 和 CPU group。
- `communication_op.py` 是模型层进入 group 的推荐入口。
- `DataParallelController` 是请求分发器，不是模型 all-reduce 封装。
- 排障时先判断自己站在哪条链路上：张量同步、请求路由、PD CPU poll，还是 Elastic EP recovery。
