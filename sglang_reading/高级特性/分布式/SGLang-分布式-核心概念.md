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
updated: 2026-07-12
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
| DP Controller rank | 请求应该给哪个 Scheduler worker | TokenizerManager 到 Scheduler 的路由；不是模型 group rank |

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

`init_distributed_environment` 在 PyTorch 分布式尚未初始化时调用 `torch.distributed.init_process_group` 建立 WORLD，随后用 `init_world_group` 包成 `_WORLD`。这里的 WORLD 是**当前 scheduler 模型进程组**；外层请求 DP 可以拥有多个这样的 worker/进程组，不能拿整套部署 GPU 数直接替代这里的 `world_size`。

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
- 同时保存 device group 与 coordination group；普通 backend 下后者是 Gloo，Mooncake backend 下则创建 `mooncake-cpu`。
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

构造时，它会为每组 ranks 创建 device group，并创建协调通道。普通 backend 使用 Gloo；Mooncake backend 分别使用 `mooncake` 与 `mooncake-cpu`，所以 `cpu_group` 这个属性名不能被无条件解释成 Gloo。

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

这就是对象/元数据协调与设备张量通信可以共享 membership、却使用不同 backend 通道的根源。还要注意：`GroupCoordinator.all_reduce` 的 CPU-tensor fallback 调用的是当前 `device_group`；并非看到 CPU tensor 就必然改用 `cpu_group`。具体 API 必须逐个看实现。

## Alias：有些“组名”不是新建组

理解当前实现必须增加一条 alias 规则：当尺寸恰好相等时，源码会复用已有 coordinator。

- `attn_cp_size == tp_size` 时，`_ATTN_CP = _TP`。
- `attn_tp_size == tp_size` 时，`_ATTN_TP = _TP`。
- `moe_ep_size == tp_size` 或 `moe_tp_size == tp_size` 时，相应 MoE group 可直接别名 `_TP`。
- 某些 MoE DP 组合会别名 `_ATTN_CP` 或 `_TP`。

因此“Attention TP 关闭 custom all-reduce”“MoE EP 关闭 PyNccl”只适用于相应**新建 coordinator** 的分支；别名分支继承被复用对象的 communicator。排障时既要记 group 语义名，也要检查对象身份。

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

这条链路的对象是请求，不是 GPU tensor。DP 路由错了，通常表现为请求落错 worker、bootstrap room 不一致或负载倾斜；模型 collective 错了，才更可能表现为 group mismatch、backend timeout、shape 或 graph capture 问题。DP-Attention 还会复用 TP rank 空间表达 attention DP，不能用“每个 DP 都是独立 TP WORLD”一条规则覆盖所有模式。

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
- `GroupCoordinator` 是 collective/协调边界对象，内部有 device group 与名为 `cpu_group` 的协调 group；后者普通模式为 Gloo、Mooncake 模式为 `mooncake-cpu`。
- `communication_op.py` 是模型层进入 group 的推荐入口。
- `DataParallelController` 是请求分发器，不是模型 all-reduce 封装。
- group 语义名不保证对象唯一；尺寸相等时 Attention/MoE group 可能 alias TP。
- 排障时先判断自己站在哪条链路上：张量同步、请求路由、PD CPU poll，还是 Elastic EP recovery。

## 静态验证

```powershell
rg -n 'backend="mooncake-cpu"|backend="gloo"|_ATTN_CP = _TP|_ATTN_TP = _TP|_MOE_EP = _TP|_MOE_DP = _ATTN_CP' `
  sglang/python/sglang/srt/distributed/parallel_state.py
```

预期既命中两类 coordination backend，也命中多条 alias 分支。若只背“每个语义都新建 NCCL + Gloo group”，就无法解释这些结果。
