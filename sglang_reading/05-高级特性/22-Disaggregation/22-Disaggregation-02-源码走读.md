---
type: batch-doc
module: 22-Disaggregation
batch: "22"
doc_type: walkthrough
title: "PD 分离 · 源码走读"
tags:
 - sglang/batch/22
 - sglang/module/disaggregation
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# PD 分离 · 源码走读

## 走读顺序

1. `disagg_service.py` — 启动 bootstrap
2. `utils.py` — 模式枚举与 poll 同步
3. `prefill.py` — Bootstrap / Waiting / Inflight 队列
4. `decode.py` — Prealloc / Transfer / PrebuiltExtend
5. `common/conn.py` — 通用 KV 连接抽象

---

## 1. metadata gate

**Explain：** Decode 侧 KV 物理传输完成但 metadata（bootstrap_room 等）尚未写入时，将 Success 降级为 Transferring，避免提前进入 decode。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L103-L118
def _apply_metadata_gate(polls, decode_reqs, metadata_buffers, server_args) -> None:
    """Downgrade Success → Transferring for requests whose metadata hasn't landed.

    Mutates `polls` in-place. Called before all-reduce so that MIN across TP
    ranks naturally prevents any rank from committing before all ranks are ready.
    """
    for i, poll_val in enumerate(polls):
        if poll_val == int(KVPoll.Success):
            decode_req = decode_reqs[i]
            if _is_fake_transfer(decode_req.req, server_args):
                continue
            actual_room = metadata_buffers.bootstrap_room[
                decode_req.metadata_buffer_index, 0
            ].item()
            if actual_room == 0:
                polls[i] = int(KVPoll.Transferring)
```

**Comment：** Fake transfer 测试路径跳过 gate；`bootstrap_room==0` 表示 metadata 缓冲尚未填充。

---

## 2. PrefillBootstrapQueue

**Explain：** 管理 bootstrap 未完成请求；为每个 Req 创建 KVSender，poll 直到 handshake 与 preallocation 就绪。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L104-L120
class PrefillBootstrapQueue:
    """
    Store the requests in bootstrapping
    """

    def __init__(
        self,
        token_to_kv_pool: KVCache,
        draft_token_to_kv_pool: Optional[KVCache],
        req_to_metadata_buffer_idx_allocator: ReqToMetadataIdxAllocator,
        metadata_buffers: MetadataBuffers,
        tp_rank: int,
        tp_size: int,
        gpu_id: int,
        bootstrap_port: int,
        gloo_group: ProcessGroup,
        max_total_num_tokens: int,
```

**Comment：** 同时持有 draft KV pool（投机 prefill）；metadata_buffers 与 decode 端共享语义字段。

---

## 3. metadata 缓冲释放

**Explain：** Prefill 完成或 abort 时必须释放 metadata_buffer_index，防止泄漏。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L87-L101
def maybe_release_metadata_buffer(
    req: Req, allocator: ReqToMetadataIdxAllocator
) -> None:
    """
    Release the metadata buffer index allocated for a request in prefill disaggregation mode.

    This function safely releases the metadata buffer index if it was allocated.

    Args:
        req: The request object that may have a metadata_buffer_index allocated
        allocator: The ReqToMetadataIdxAllocator instance to free the index
    """
    if req.metadata_buffer_index >= 0:
        allocator.free(req.metadata_buffer_index)
        req.metadata_buffer_index = -1
```

**Comment：** 与 decode 端 `Req.metadata_buffer_index` 生命周期对称；abort 路径必须调用。

---

## 4. DecodeReqToTokenPool

**Explain：** 相对普通 `ReqToTokenPool`，decode 侧单独订阅 pre-alloc 内存，使 `#pre-allocated + #transfer` 可超出 `--max-running-requests` 限制的空闲 KV。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode.py L107-L117
class DecodeReqToTokenPool:
    """
    The difference of DecodeReqToTokenPool and ReqToTokenPool is that
    DecodeReqToTokenPool subscribes memory for pre-allocated requests.

    In ReqToTokenPool, if `--max-running-requests` is 8,
    #pre-allocated + #transfer + #running <= 8, but there are in fact more memory can carry pre-allocated requests.

    In DecodeReqToTokenPool, if `--max-running-requests` is 8,
    #running <= 8, #pre-allocated + #transfer <= pre_alloc_size, so we can use the free memory to pre-allocate requests to unblock prefill.
    """
```

**Comment：** 这是 PD 分离吞吐优化的关键：Decode 提前握手占坑，Prefill 不必等待 running 槽位。

---

## 5. poll_and_all_reduce_attn_cp_tp_group

**Explain：** 带 Attention CP（Context Parallel）的部署需先在 attn-tp 组内同步，再在 attn-cp 组内同步 poll 状态。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L143-L160
def poll_and_all_reduce_attn_cp_tp_group(
    pollers,
    attn_cp_cpu_group: dist.ProcessGroup,
    attn_tp_cpu_group: dist.ProcessGroup,
):
    # First sync across attn-tp ranks so all TP participants for a given (dp, cp)
    # shard observe the same status transitions.
    polls = poll_and_all_reduce(pollers, attn_tp_cpu_group)

    # Then sync across attn-cp ranks, so all TPxCP participants in one DP shard
    # converge to the same global status.
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(
        tensor_to_reduce,
        op=dist.ReduceOp.MIN,
        group=attn_cp_cpu_group,
    )
    return tensor_to_reduce.tolist()
```

**Comment：** 两层 all_reduce 保证 TP×CP 网格内所有参与者状态一致。

---

## 6. staging 传输

**Explain：** 部分 backend 需要分阶段 scatter KV；`poll_and_all_reduce_with_staging` 在 poll 前推进 staging handler。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/utils.py L163-L175
def poll_and_all_reduce_with_staging(
    decode_reqs,
    staging_handler,
    gloo_group: dist.ProcessGroup,
    metadata_buffers: Optional[MetadataBuffers] = None,
    server_args: Optional[ServerArgs] = None,
):
    """Staging-aware polling: advance scatter, demote incomplete transfers, all_reduce."""
    for decode_req in decode_reqs:
        if decode_req.kv_receiver.require_staging and not staging_handler.is_done(
            decode_req
        ):
            staging_handler.advance_scatter(decode_req)
```

**Comment：** `common/staging_handler.py` 与 `staging_buffer.py` 实现具体分块逻辑。

---

## 7. optimistic prefill retry 测试钩子

**Explain：** 环境变量可注入 prefill retry 概率，用于混沌测试。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/prefill.py L77-L84
def should_force_retry(req: Req) -> bool:
    """Test hook to force a request into optimistic prefill retry."""
    retry_prob = envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
    if retry_prob <= 0 or req.time_stats.prefill_retry_count > 0 or req.is_retracted:
        return False

    digest = hashlib.sha256(str(req.rid).encode()).digest()
    return int.from_bytes(digest[:8], "big") < retry_prob * 2**64
```

**Comment：** 基于 rid 哈希确定性触发，便于复现；生产环境 retry_prob 为 0。

---

## 8. HiCache Decode Mixin

**Explain：** `decode_hicache_mixin.py` 扩展 decode 路径，支持从远端 HiCache 恢复 prefix KV 而非全量 RDMA 传输。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/decode_hicache_mixin.py（类引用）
class DecodeHiCachePreallocMixin:
 ...
class DecodeHiCacheTransferMixin:
 ...
class HiCacheRestoreGatedKVReceiver:
 ...
```

**Comment：** 与KV Cache KV Cache / RadixAttention RadixAttention 的 HiCache 存储联动；降低跨节点 KV 字节数。

---

## 9. encode_server 与多模态 PD

**Explain：** 多模态场景下 encoder 节点可独立部署（`encode_server.py` / `encode_grpc_server.py`），vision embedding 经专用通道送达 prefill/decode。

**Code：**

```python
# 来源：python/sglang/srt/disaggregation/encode_server.py L233-L294
class MMEncoder:
    def __init__(
        self,
        server_args: ServerArgs,
        schedule_path=None,
        dist_init_method=None,
        rank: int = 0,
    ):
        logger.info(f"init MMEncoder {rank}/{server_args.tp_size}")
        self.server_args = server_args
        set_global_server_args_for_scheduler(server_args)
        self.rank = rank
        self.profiler = EncoderProfiler(rank)
        self._load_mm_processor(server_args)

        self.model_config = ModelConfig.from_server_args(
            server_args,
        )
        self.load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            model_loader_extra_config=server_args.model_loader_extra_config,
            remote_instance_weight_loader_seed_instance_ip=server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=server_args.remote_instance_weight_loader_send_weights_group_ports,
        )
        self.model_type = getattr(
            self.model_config.hf_config, "model_type", "unknown"
        ).lower()

        self.device = server_args.device
        self.gpu_id = server_args.base_gpu_id + rank

        self.device_config = DeviceConfig(
            device=self.device,
            gpu_id=self.gpu_id,
        )

        torch.get_device_module(self.device).set_device(self.gpu_id)

        self.use_image_processor_gpu = (
            use_image_processor_gpu and not server_args.disable_fast_image_processor
        )
        self._build_vision_config(server_args.mm_process_config)
        self.model_audio_sr = self._resolve_audio_sr()
        logger.info(f"Resolved model audio sample rate: {self.model_audio_sr} Hz")

        init_distributed_environment(
            backend=get_default_distributed_backend(self.device),
            world_size=server_args.tp_size,
            rank=rank,
            distributed_init_method=dist_init_method,
            local_rank=rank,
        )
        initialize_model_parallel(tensor_model_parallel_size=server_args.tp_size)
        initialize_dp_attention(server_args, self.model_config)

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
        )
```

**Comment：** 与Multimodal Multimodal 交叉；PD + VLM 三联部署时 encode → prefill → decode。
