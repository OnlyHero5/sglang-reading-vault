---
type: batch-doc
module: 23-Distributed
batch: "23"
doc_type: faq
title: "分布式并行：关键问题"
tags:
 - sglang/batch/23
 - sglang/module/distributed
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# 分布式并行：关键问题

## Q1：TP size 与 world size 关系？

**Explain：** `world_size` 通常等于 TP×PP×DP 等维度乘积；`initialize_model_parallel` 按参数切分 rank 列表。配置错误会导致 group 创建 assert 失败。

**Code：**

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

**Comment：** 启动脚本 `--tp-size`、`--pp-size`、`--dp-size` 由 ServerArgs 传入 initialize。

---

## Q2：何时启用 CustomAllReduce？

**Explain：** 小 tensor、同机 NVLink 拓扑时 Custom AR 延迟低于 NCCL；CUDA Graph capture 期间可能禁用。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L244-L245
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    use_torch_symm_mem_all_reduce: (
```

**Comment：** 见 `device_communicators/custom_all_reduce_utils.py` 拓扑检测。

---

## Q3：错误用法 — 直接 torch.distributed

**Explain：** 绕过 GroupCoordinator 可能导致 graph / compile 注册 op 不一致。

**Code（错误）：**

```python
import torch.distributed as dist
dist.all_reduce(tensor) # 使用 default group，非 TP group
```

**Code（正确）：**

```python
# 来源：python/sglang/srt/distributed/communication_op.py L18-L20
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

**Comment：** 唯一例外是 PD poll 等明确使用 gloo 组的场景。

---

## Q4：NPU HCCL 选项

**Explain：** Ascend 平台为 MoE 相关 group 设置 HCCL buffer size。

**Code：**

```python
# 来源：python/sglang/srt/distributed/parallel_state.py L84-L98
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
```

**Comment：** 大 EP 时可调 `HCCL_BUFFSIZE` 避免 collective OOM。

---

## Q5：DP + PD 联合部署

**Explain：** 使用 FOLLOW_BOOTSTRAP_ROOM 使同一 PD room 的请求固定到同一 DP worker，减少 cross-rank KV 状态。

**Code：**

```python
# 来源：python/sglang/srt/managers/data_parallel_controller.py L79-L80
    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
```

**Comment：** Gateway 层仍需维护 prefill/decode 池；DP 只解决 decode 副本间路由。

---

## Q7：Elastic EP 如何恢复故障 rank？

**Explain：** Mooncake EP backend 下，故障 rank 重启后 `try_recover_ranks` 轮询 peer state，先 recover WORLD process group，再遍历所有 live `GroupCoordinator` 映射 local rank 并 recover device/cpu group；最后 `_refresh_ep_members` 刷新 MoE dispatch buffer。

**Code：**

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

**Comment：** 需 `--elastic-ep-backend` 与 Mooncake 安装；rejoin 时 `ElasticEPStateManager.init` 可 mask 非本 rank 以便 CUDA graph capture。

---
