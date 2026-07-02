---
type: batch-doc
module: 22-Disaggregation
batch: "22"
doc_type: faq
title: "PD 分离：关键问题"
tags:
 - sglang/batch/22
 - sglang/module/disaggregation
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# PD 分离：关键问题

## Q1：Prefill 与 Decode 如何配对？

**Explain：** 客户端或 gateway 在请求中指定 decode 节点的 bootstrap 信息；Prefill 完成后向该 room 推送 KV。通常由 model-gateway 或负载均衡层维护 prefill/decode 池映射。

**Code（Decode 预分配握手）：**

```python
# 来源：python/sglang/srt/disaggregation/decode.py L4-L7
1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.
```

**Comment：** 必须先 pre-alloc 再 transfer，否则 Prefill 端 sender 无目标 room。

---

## Q2：Transfer Backend 怎么选？

| Backend | 场景 |
|---------|------|
| Mooncake | 默认高性能 RDMA 栈（Linux） |
| NIXL | NVIDIA 生态集成 |
| Mori | 特定云环境 |
| Ascend | 华为 NPU + memfabric |
| Fake | 单元测试 |

**Code：**

```python
# 来源：python/sglang/srt/managers/disagg_service.py L18-L19
    disagg_mode = DisaggregationMode(server_args.disaggregation_mode)
    transfer_backend = TransferBackend(server_args.disaggregation_transfer_backend)
```

**Comment：** `--disaggregation-transfer-backend` 与 mode 独立配置；Ascend 需额外 `ASCEND_MF_STORE_URL`。

---

## Q3：metadata 未就绪导致卡住？

**Explain：** 若 Success 过早上报，Decode 可能在 metadata 无效时 merge batch。metadata gate 将 Success 降回 Transferring。

**Code（反模式 — 仅看 poll Success）：**

```python
# 错误：不检查 metadata_buffers.bootstrap_room 直接进入 waiting queue
if poll == KVPoll.Success:
 move_to_waiting(req) # 可能 room 仍为 0
```

**Code（正确 — 框架内置 gate）：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L114-L118
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)
```

**Comment：** 自定义 poll 逻辑时必须复用 `_apply_metadata_gate`。

---

## Q4：DecodeReqToTokenPool 与 max-running-requests

**Explain：** 普通池：`pre+transfer+running ≤ max`；Decode 池：`running ≤ max`，pre+transfer 可用额外 pre_alloc_size。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode.py L115-L116
    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
```

**Comment：** 调大 pre_alloc_size 可提高 PD 流水线并行度，但占用更多 KV 内存。

---

## Q5：Prefill retry 与 retract

**Explain：** optimistic prefill retry 在 transfer 失败时重试 prefill；retracted 请求不再 retry。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L80-L81
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False
```

**Comment：** 生产默认关闭；调试 PD 失败恢复路径时可开启环境变量。

---

## Q6：HiCache 与全量 KV 传输

**Explain：** 若 prefix 已在共享 HiCache，Decode 可只拉取 delta 或通过 prefix match 本地命中，减少网络字节。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode_hicache_mixin.py L23-L47
@dataclass
class DecodePrefixMatch:
    prefix_indices: torch.Tensor
    l2_host_hit_length: int
    l3_storage_hit_length: int
    last_device_node: Any
    last_host_node: Any = None
    prefetch_registered: bool = False

    @property
    def l1_prefix_len(self) -> int:
        return len(self.prefix_indices)

    @property
    def decode_prefix_len(self) -> int:
        return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length

    @property
    def needs_local_restore(self) -> bool:
        return self.decode_prefix_len > self.l1_prefix_len

    @property
    def restore_token_count(self) -> int:
        """Number of tokens that need L2/L3 load_back to device."""
        return self.decode_prefix_len - self.l1_prefix_len
```

**Comment：** 与 RadixAttention 前缀共享协同；部署需统一 cache key 与 eviction 策略。

---

## Q7：PD 分离 vs Unified 部署——TCO 决策框架怎么问？

**Explain：** 先估 **负载形态**：prefill 与 decode 是否错峰（聊天峰 decode 多、批处理峰 prefill 多）？再估 **KV 传输成本**：跨节点 RDMA 字节 × 单价 vs 统一池化 GPU 的空闲碎片。第三看 **运维复杂度**：PD 需 bootstrap room、metadata gate、双池扩缩容；unified 单 Scheduler loop 更简单但长 prompt 与 decode 争抢 SM。

| 信号 | 倾向 PD | 倾向 Unified |
|------|:-------:|:------------:|
| prefill/decode QPS 比 >3:1 且峰错开 | ✓ | |
| 团队无 RDMA/网关经验 | | ✓ |
| 前缀 HiCache 命中高、传输字节少 | ✓ | |
| 模型小、单卡可扛 | | ✓ |

**Code：**

```python
# 来源：python/sglang/srt/managers/scheduler.py L4164-L4192
def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()
```

**Comment：** TCO 不仅是硬件——PD 调试 metadata gate、retry 的 engineer-hours 常低估；先用 Fake backend 压测队列深度再切 Mooncake。

---

## 设计追问

### Q1：Prefill 完成后 KV 未传完，Decode 侧为何不能先 forward？

**Explain：** Decode 必须以完整 KV slot 与 `bootstrap_room` metadata 组 `ForwardMode.PREBUILT` batch；提前 merge 会导致 logits 基于空 KV。metadata gate 把 premature Success 降回 Transferring 即为此 invariant 服务。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L114-L118
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)
```

**Comment：** 自定义 poll 必须复用 `_apply_metadata_gate`。

---

### Q2：`DecodeReqToTokenPool` 的 pre_alloc_size 调大何时划算？

**Explain：** pre_alloc 提高 transfer 与 running 流水线并行度，减少 Prefill 端 blocking；代价是预留 KV 槽占用显存。当 transfer 延迟 >> decode step 且 `max-running-requests` 未打满时，适度调大 pre_alloc 可提升 PD 集群吞吐。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode.py L115-L116
    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
```

**Comment：** 与 unified 模式下 `check_decode_mem` retract 策略无关——PD decode 池规则独立。

---

### Q3：HiCache 在 PD 场景减少传输的条件是什么？

**Explain：** Decode 侧 `_build_decode_prefix_match` 若 L2/L3 已命中 prefix，仅需 RDMA 传 delta 或本地 restore，网络字节下降。前提：Prefill/Decode 共享 HiCache key 策略与 eviction，且 `enable_decode_hicache` 开启。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode_hicache_mixin.py L61-L68
    def _build_decode_prefix_match(self, req: Req, result: Any) -> DecodePrefixMatch:
        """Convert a ``match_prefix_for_req`` result into ``DecodePrefixMatch``.

        Performs the optional L3 storage hit length query when decode-side
        HiCache is enabled and the last host node is backed up.
        """
        prefix_indices = result.device_indices
        l1_prefix_len = len(prefix_indices)
```

**Comment：** 与KV Cache Radix/HiCache 文档交叉阅读。
