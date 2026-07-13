---
title: "分布式 · 排障指南"
type: troubleshooting
framework: sglang
topic: "分布式"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-12
---
# 分布式 · 排障指南

## 读者任务

这篇不是补充概念，而是排障入口。遇到 Distributed 问题时，先判断症状属于哪一类：启动切组失败、模型 collective 失败、请求 DP 路由失败、PD poll 失败、Elastic EP recovery 失败。不同症状对应的源码入口不同。

## 症状 1：启动时报 world size 不匹配

**现象：** 进程刚启动、模型还没真正 forward，就报 `world_size is not equal to tensor_model_parallel_size x pipeline_model_parallel_size`。

**源码入口：** `initialize_model_parallel` 的前置校验。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L2029-L2039
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )
```

**判断方法：** 不要把 `dp_size` 直接乘进这个等式。这里校验的是当前 scheduler 模型 WORLD。普通 DP 会启动多个模型 worker 组；DP-Attention 则在 TP rank 空间中表达 attention DP，二者都不能用“部署总 GPU = 这一处 world_size”替代源码作用域。

**操作：** 记录报错进程的 `world_size/tp_size/pp_size`，确认它属于哪个 scheduler/DP-Attention 启动分支，再验证 `world_size == tp_size * pp_size`。

**预期：** 当前模型 WORLD 内等式成立；外层 `dp_size` 只影响 worker/rank 布局，不直接进入这条异常消息。

## 症状 2：DCP 配置看似合理，但初始化失败

**现象：** DCP 大于 1 时启动失败，或者提示 TP size 不能被 DCP 整除。

**源码入口：** DCP 的合法性检查在 TP 切组之前执行。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L2040-L2055
    if decode_context_parallel_size < 1:
        raise RuntimeError(
            f"decode_context_parallel_size ({decode_context_parallel_size}) must be >= 1"
        )
    if decode_context_parallel_size > 1 and not (is_hip() or is_cuda()):
        raise RuntimeError(
            "Decode context parallel (decode_context_parallel_size > 1) is "
            "currently only supported on the AMD HIP platform or CUDA platform, but got "
            f"decode_context_parallel_size ({decode_context_parallel_size}) "
            "on a non-HIP or non-CUDA platform."
        )
    if tensor_model_parallel_size % decode_context_parallel_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) must be divisible by "
            f"decode_context_parallel_size ({decode_context_parallel_size})"
        )
```

**判断方法：** DCP 是在 TP group 内部切 decode context，不是跨所有 GPU 任意切。因此先看 `tp_size % dcp_size == 0`。

**操作：** 先设 DCP 为 1 验证基线，再检查平台为 CUDA/HIP 且 `tp_size % dcp_size == 0`，随后一次只提高一个 DCP 配置。

**预期：** DCP=1 可启动；合法平台和整除组合通过前置校验，非法组合在模型加载前给出明确异常。

## 症状 3：模型层 all-reduce 偶发 graph 或 backend 问题

**现象：** 普通 eager 路径能跑，CUDA Graph、piecewise graph 或 CustomAllReduce 下失败；或者性能和预期 backend 不一致。

**源码入口：** `GroupCoordinator.all_reduce` 的 out-of-place 与 in-place 选路。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L628-L661
        outplace_all_reduce_method = None
        if (
            self.ca_comm is not None
            and not self.ca_comm.disabled
            and not should_use_pymscclpp_allreduce
            and self.ca_comm.should_custom_ar(input_)
        ):
            outplace_all_reduce_method = "ca"
        elif (
            self.qr_comm is not None
            and not self.qr_comm.disabled
            and self.qr_comm.should_quick_allreduce(input_)
        ):
            outplace_all_reduce_method = "qr"
        elif self.pymscclpp_comm is not None and should_use_pymscclpp_allreduce:
            outplace_all_reduce_method = "pymscclpp"
        elif (
            self.torch_symm_mem_comm is not None
            and not self.torch_symm_mem_comm.disabled
            and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
        ):
            outplace_all_reduce_method = "torch_symm_mem"
        elif is_in_tc_piecewise_cuda_graph() and self.pynccl_comm is not None:
            # For piecewise cuda graph, we use pynccl outplace allreduce
            outplace_all_reduce_method = "pynccl"
        if outplace_all_reduce_method is not None:
            return outplace_all_reduce(
                input_,
                group_name=self.unique_name,
                outplace_all_reduce_method=outplace_all_reduce_method,
            )
        else:
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_
```

**判断方法：** 不要只问“是不是 NCCL”。先记录 getter 返回的 coordinator `unique_name` 和对象 id，排除 `_ATTN_TP = _TP`、`_MOE_EP = _TP` 等 alias；再看 `ca_comm`、`qr_comm`、`pymscclpp_comm`、`torch_symm_mem_comm`、对称内存与 piecewise graph 状态。

**操作：** 先用对应 `communication_op.py` helper 重现，并暂时只切换一个 backend/graph 条件做对照。

**预期：** helper 进入语义正确的 coordinator；关闭某个优化 backend 后若故障消失，才继续定位该 communicator 的尺寸、注册或 capture 条件。

## 症状 4：有人在 layer 里裸调 `torch.distributed`

**现象：** 代码看起来能 all-reduce，但多 TP、MoE、Attention TP 或 graph 路径下行为不稳定。

**源码入口：** 模型 collective 的推荐入口是 helper。

```python
# 来源：python/sglang/srt/distributed/communication_op.py L18-L20
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

```python
# 来源：python/sglang/srt/distributed/communication_op.py L65-L67
def attention_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across attention parallel group."""
    return get_attn_tp_group().all_reduce(input_)
```

**判断方法：** 裸调 `torch.distributed.all_reduce(tensor)` 默认不表达“这是 TP、Attention TP、MoE TP 还是 MoE EP”。源码里的 helper 用函数名把语义绑定到对应 group。

**操作：** 将裸调用替换为语义对应的 helper，并记录 getter 返回的 group membership；若是 PD poll 等状态同步，则保留其显式 coordination-group 调用链。

**预期：** 模型层 collective 进入正确 coordinator 并保留 graph/backend 选路；状态同步不被误接到模型 TP helper。

## 症状 5：DP 请求没有落到期望 worker

**现象：** 多 DP worker 下请求分布不符合预期；指定路由、bootstrap room 或负载均衡表现异常。

**源码入口：** `routed_dp_rank` 优先级高于负载均衡。

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L605-L610
    def maybe_external_dp_rank_routing(self, req: Req):
        if req.routed_dp_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.routed_dp_rank}")
            sock_send(self.workers[req.routed_dp_rank], req)
            return True
        return False
```

`FOLLOW_BOOTSTRAP_ROOM` 则强制要求 `bootstrap_room` 存在。

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L628-L638
    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        sock_send(self.workers[target_rank], req)
```

**判断方法：** 先看请求对象上是否带 `routed_dp_rank`。如果带，它会直接覆盖普通调度策略；如果策略是 `FOLLOW_BOOTSTRAP_ROOM`，再看 `bootstrap_room` 是否存在且稳定。

**操作：** 记录 `routed_dp_rank`、`bootstrap_room`、计算出的 target rank 与 `status[target]`。PD 场景经 router 产生会合字段；若使用外部直达 rank，先验证范围与健康状态。

**预期：** 请求落到预期 worker。注意当前 external-rank 与 room 分支本身不跳过 inactive status，上层健康路由错误不能靠 Controller 自动修复。

## 症状 6：`TOTAL_REQUESTS` 或 `TOTAL_TOKENS` 下负载仍然倾斜

**现象：** 看起来启用了负载感知调度，但短 burst 仍集中到单个 DP rank。

**源码入口：** `refresh_load_budget` 的 20ms 节流和 `DPBudget.dispatch` 的投机计数。

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L222-L237
    def refresh_load_budget(self):
        # Throttle to at most once per 20ms.  When a burst of requests
        # arrives, dispatching_with_trace() calls this before every
        # dispatch.  Each call reads the latest scheduler snapshot and
        # overwrites the speculative +1 increments that DPBudget.dispatch()
        # added for previously dispatched requests in this burst.  Without
        # throttling, the budget resets to the (stale) scheduler-reported
        # value on every request, causing the entire burst to land on a
        # single DP rank.  The 20ms interval lets the burst complete
        # using speculative counters, then refreshes from the real
        # scheduler load for the next batch.
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.02:
            return
        self._last_refresh_time = now
        self.dp_budget.update_budget(self.load_snapshot_reader.read_all())
```

**判断方法：** 如果 snapshot 太旧或刷新过频，预算可能不能反映 burst 内的已分发请求。源码通过节流和投机加一缓解这个问题。

**操作：** 查看 load snapshot 是否持续更新、所有 worker timestamp 是否前进，并确认请求未被 `routed_dp_rank` 或 `FOLLOW_BOOTSTRAP_ROOM` 绕过负载策略。

**预期：** burst 内 speculative budget 会递增，约 20ms 后由新快照校准；若仍倾斜，应能定位为快照陈旧、强制路由或 worker health，而不是泛称算法失效。

## 症状 7：PD poll 卡住或状态不一致

**现象：** PD transfer 状态在部分 rank 上变成成功，部分 rank 仍在 transferring；或者 poll 阶段等待异常。

**源码入口：** `poll_and_all_reduce` 使用 CPU tensor 和调用者传入的 coordination group。

```python
# 来源：python/sglang/srt/disaggregation/utils.py L138-L140
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()
```

**判断方法：** 普通 backend 下 coordination group 通常是 Gloo；Mooncake group 的 `cpu_group` 实际为 `mooncake-cpu`。还要检查 metadata gate：metadata 未落地时会把局部 Success 降回 Transferring。

**操作：** 打印 group backend/membership、每 rank 原始 poll、metadata buffer 的 bootstrap room，以及 MIN 后结果。

**预期：** 所有 TP×CP 参与者最终看到相同状态；任一 rank 未 ready 或 metadata 未到时，不应提前 commit。

## 症状 8：Ascend / NPU 上 MoE collective OOM 或 HCCL buffer 异常

**现象：** NPU 平台上 MoE 相关 collective 初始化或运行时报 buffer 相关错误。

**源码入口：** `get_torch_distributed_pg_options` 只对默认 group 或名称包含 `moe` 的 group 创建 HCCL options。

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L84-L99
def get_torch_distributed_pg_options(group_name=None):
    if not _is_npu:
        return None

    # Only create HCCL options for default group or MoE-related groups
    if group_name is not None and "moe" not in group_name:
        return None

    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_buffer_size = int(
        os.environ.get("DEEPEP_HCCL_BUFFSIZE") or os.environ.get("HCCL_BUFFSIZE") or 200
    )
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
    return options
```

**判断方法：** 先看当前 group name 是否含 `moe`，再看 `DEEPEP_HCCL_BUFFSIZE` 或 `HCCL_BUFFSIZE`。

**操作：** 记录实际 `group_name` 与环境变量来源，先在受控环境调整 `DEEPEP_HCCL_BUFFSIZE`/`HCCL_BUFFSIZE`，并比较初始化日志与峰值内存；不要把同一参数无差别应用到非 MoE group。

**预期：** MoE group 读取预期 buffer 配置；非 MoE 命名 group 返回 `None` options。具体数值必须按模型、拓扑和 NPU 环境实测。

## 症状 9：Elastic EP recovery 看起来没生效

**现象：** 故障 rank 重启后，系统仍认为 peer 未恢复；或者 WORLD 恢复了但 MoE dispatch 成员没有刷新。

**源码入口：** `try_recover_ranks` 先检查 WORLD peer state，不 ready 直接返回 `False`；ready 后恢复 WORLD、每个 live group 的 device/cpu backend，最后刷新 EP members。

```python
# 来源：python/sglang/srt/elastic_ep/elastic_ep.py L147-L174
def try_recover_ranks(global_ranks: List[int]) -> bool:
    from mooncake import ep as mooncake_ep

    world_backend = _get_process_group_backend(torch.distributed.group.WORLD, "cuda")
    if not all(mooncake_ep.get_peer_state(world_backend, global_ranks)):
        # The relaunched ranks have not finished initializing yet.
        return False

    # Recover the world backend first, then recover each derived process group
    # using ranks mapped into that group's local rank space.
    mooncake_ep.recover_ranks(world_backend, global_ranks)

    for group in _iter_live_parallel_groups():
        group_local_ranks = _map_global_to_group_local_ranks(group.ranks, global_ranks)
        if not group_local_ranks:
            continue

        device_backend = _get_process_group_backend(group.device_group, "cuda")
        _wait_for_peer_state(mooncake_ep, device_backend, group_local_ranks)
        mooncake_ep.recover_ranks(device_backend, group_local_ranks)

        cpu_backend = _get_process_group_backend(group.cpu_group, "cpu")
        _wait_for_peer_state(mooncake_ep, cpu_backend, group_local_ranks)
        mooncake_ep.recover_ranks(cpu_backend, group_local_ranks)
        _maybe_create_message_queue(group)

    _refresh_ep_members()
    return True
```

**判断方法：** 如果函数返回 `False`，问题还在 relaunched rank 的 peer state；如果返回 `True` 但 MoE 行为异常，再看 `_refresh_ep_members` 和 EP buffer。

**操作：** 确认仅在当前支持的 CUDA/CPU Elastic EP 环境使用该路径，记录 rejoin rank 的 `join_process_groups`、live rank 的 WORLD peer state、各 derived group local-rank 映射与 `_refresh_ep_members`。

**预期：** peer 未 ready 时 `try_recover_ranks` 返回 `False`；ready 后依次恢复 WORLD、相关 device/coordination group，最后刷新 EP members。

## 症状 10：rejoin 时只有本 rank active

**现象：** Elastic EP rejoin 模式下，active rank mask 看起来只打开当前 rank。

**源码入口：** `ElasticEPStateManager.init` 在 `elastic_ep_rejoin` 下故意 mask peer ranks，让重启 rank 能独立进行 CUDA Graph capture。

```python
# 来源：python/sglang/srt/elastic_ep/elastic_ep.py L49-L60
    def init(cls, server_args: ServerArgs):
        if cls._instance is not None:
            return cls._instance

        if server_args.elastic_ep_backend is not None:
            cls._instance = cls._build_state(ep_size=None, device=None)
            if server_args.elastic_ep_rejoin:
                # Mask out peer ranks to perform cuda graph capture on its own
                cls._instance.active_ranks.zero_()
                cls._instance.active_ranks[torch.distributed.get_rank()] = 1
                cls._instance.snapshot_active_to_last()
                cls._instance.sync_active_to_cpu()
```

**判断方法：** 这不是普通运行期健康状态，而是 rejoin 期间的临时状态。

**操作：** 分别在 rejoin capture 前、recovery 后记录 `active_ranks/last_active_ranks/active_ranks_cpu`。

**预期：** rejoin capture 阶段仅本 rank active 是设计行为；恢复后 membership 状态应由控制流更新，若没有更新再查 recovery/active-rank 广播。

## 复盘排障顺序

1. 启动失败先看 `world_size`、TP、PP、DCP 的硬校验。
2. 模型 collective 失败先看 helper、getter、`GroupCoordinator.all_reduce`。
3. 请求分发异常先看 `routed_dp_rank`、`bootstrap_room`、load budget。
4. PD 状态不同步先看 CPU status tensor、coordination-group backend/membership 与 metadata gate。
5. Elastic EP 先看 Mooncake peer state，再看 live group recovery 和 EP member refresh。

## 无 GPU 静态检查

```powershell
rg -n "world_size != tensor_model_parallel_size|_ATTN_TP = _TP|routed_dp_rank|bootstrap_room %|_apply_metadata_gate|elastic_ep_rejoin" `
  sglang/python/sglang/srt/distributed/parallel_state.py `
  sglang/python/sglang/srt/managers/data_parallel_controller.py `
  sglang/python/sglang/srt/disaggregation/utils.py `
  sglang/python/sglang/srt/elastic_ep/elastic_ep.py
```

预期六类入口均有命中。静态检查只能确认分支位置；collective timeout、性能和 recovery 时序仍需目标多卡环境。
