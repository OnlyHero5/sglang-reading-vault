---
type: batch-doc
module: 22-Disaggregation
batch: "22"
doc_type: concept
title: "PD 分离 · 核心概念"
tags:
 - sglang/batch/22
 - sglang/module/disaggregation
 - sglang/doc/concept
aliases:
 - "01-核心概念"
updated: 2026-07-02
---
# PD 分离 · 核心概念

## 用户故事：PD 分离后 P99 延迟反而更差

### Persona

**赵架构**，为聊天高峰把 Prefill 与 Decode 拆成独立集群，预期 TTFT 与 decode 延迟各优化，上线后 P99 e2e 从 1.2s 升到 2.8s。

### 时间线

| 时刻 | 事件 |
|------|------|
| T0 | Prefill 节点 `--disaggregation-mode prefill`，Decode 节点 `--disaggregation-mode decode`，Mooncake 传 KV |
| T1 | 低负载 benchmark 正常；生产高峰 P99 恶化 |
| T2 | 发现 Decode 侧 **PreallocQueue / TransferQueue** 排队，Prefill bootstrap 等待 KV 槽 |
| T3 | 对齐 prefill/decode 容量比与 transfer 带宽，或短 prompt workload 回退 unified 部署 |

**Explain：** PD 分离用 `DisaggregationMode` 区分节点角色：Prefill 完成 forward 后经 **KVSender** 推 KV，Decode 经 **KVReceiver** 拉取并构造 `PrebuiltExtendBatch` 跳过 prefill。P99 变差常见于 **KV 传输 + 多队列握手** 开销超过算力分离收益——尤其 prompt 短、QPS 高时 bootstrap/inflight 排队占主导。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L60-L71
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"
```

**Comment：** Prefill 三队列（Bootstrap → Waiting → Inflight）与 Decode 四队列（Prealloc → Transfer → Waiting → Running）生命周期见 `prefill.py` / `decode.py` 文件头注释。

### 如果…会怎样（调试）

| 现象 | 可能原因 | 排查 |
|------|----------|------|
| P99 升、P50 尚可 | transfer 或 bootstrap 尾延迟 | 看 decode Prealloc/Transfer 队列深度 metrics |
| TP rank 状态不一致 | poll 未 all_reduce 对齐 | 查 `poll_and_all_reduce` MIN 语义 |
| 短 prompt 更慢 | PD 固定开销 > prefill 算力 | 用 TCO 框架评估是否回退 `NULL` 模式 |

---

## 1. PD 分离动机

**Explain：** Prefill（计算密集、变长）与 Decode（内存带宽、定长步进）对 GPU/网络资源需求不同。分离后 Prefill 集群可弹性扩缩，Decode 集群专注低延迟 token 生成；KV 经 RDMA/专用传输层跨节点搬运。SGLang 用 `DisaggregationMode` 三态枚举区分节点角色，Transfer Backend（Mooncake/NIXL 等）可插拔。

---

## 2. DisaggregationMode

**Explain：** 三态枚举：`NULL`（统一部署）、`PREFILL`、`DECODE`；`to_engine_type` 映射为对外 engine 类型字符串。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L60-L71
class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"

    @staticmethod
    def to_engine_type(mode: str) -> str:
        if mode == DisaggregationMode.PREFILL.value:
            return "prefill"
        elif mode == DisaggregationMode.DECODE.value:
            return "decode"
        return "unified"
```

**Comment：**

- CLI：`--disaggregation-mode prefill|decode|null`。
- Prefill 节点 prefill 完成后通过 KVSender 推送 KV；Decode 节点 KVReceiver 拉取后构造 `PrebuiltExtendBatch` 跳过 prefill forward。

---

## 3. Prefill 侧请求生命周期

**Explain：** `prefill.py` 文件头注释定义三队列模型。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L1-L18
"""
Life cycle of a request in the prefill server

1. Bootstrap Queue
    a. Initialize a sender for each request
    b. Use the queue to store requests whose bootstrap (handshake and preallocation) has not finished
    c. Poll senders to check bootstrap state
    d. Once bootstrap is complete, move request to Waiting Queue

2. Waiting Queue
    a. Use PrefillAdder to pop requests
    b. Run forward
    c. Add the request to Inflight Queue

3. Inflight Queue
    a. Poll (non-blocking) the sender of the request
    b. Once the transfer has finished, return the request
"""
```

**Comment：** Bootstrap 完成前请求不占 prefill 算力；Inflight 期间 KV 异步传输，Prefill GPU 可服务下一专题。

---

## 4. Decode 侧请求生命周期

**Explain：** Decode 四队列：Prealloc → Transfer → Waiting → Running。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode.py L1-L19
"""
Life cycle of a request in the decode server

1. PreallocQueue:
    a. Initialize a receiver for each request
    b. The request handshakes first, and pre-allocate kv once there is available kv.
    c. Move the request to TransferQueue.

2. TransferQueue:
    a. Poll the receiver to check the transfer state
    b. If the transfer has finished, move the request to waiting queue

3. WaitingQueue:
    a. Use the requests in the queue to construct a PrebuiltExtendBatch
    b. Skip the prefill forward but only populate metadata

4. RunningBatch:
    a. Merge the resolved PrebuiltExtendBatch into running batch to run decoding
"""
```

**Comment：** `DecodeReqToTokenPool` 允许 pre-alloc 请求占用额外内存槽， unblock Prefill 端 bootstrap。

---

## 5. Transfer Backend 与 KVPoll

**Explain：** 传输后端（Mooncake、NIXL、Mori、Fake、Ascend）实现 `CommonKVSender` / `CommonKVReceiver`；轮询状态为 `KVPoll` 枚举（Bootstrapping、Transferring、Success 等）。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L121-L140
def poll_and_all_reduce(
    pollers,
    gloo_group: dist.ProcessGroup,
    decode_reqs=None,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    # at a certain prob, the poll is failed to simulate failure
    polls = _poll_with_failure_injection(pollers)

    # Apply metadata gate on the decode requests to downgrade Success → Transferring for requests whose metadata hasn't landed.
    if (
        decode_reqs is not None
        and metadata_buffers is not None
        and server_args is not None
    ):
        _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args)
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()
```

**Comment：** TP 各 rank 对 poll 结果取 MIN，保证所有 rank 在同一状态转移点提交；metadata gate 防止 metadata 未落地时误报 Success。
