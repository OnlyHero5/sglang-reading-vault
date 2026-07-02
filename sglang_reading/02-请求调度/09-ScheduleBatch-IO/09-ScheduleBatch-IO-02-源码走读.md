---
type: batch-doc
module: 09-ScheduleBatch-IO
batch: "09"
doc_type: walkthrough
title: "ScheduleBatch-IO · 源码走读"
tags:
 - sglang/batch/09
 - sglang/module/schedule-batch-io
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# ScheduleBatch-IO · 源码走读

> 走读顺序：`embed_types.py` → `io_struct.py`（IPC 基类与序列化）→ `schedule_batch.py`（Req → ScheduleBatch 生命周期）

---

## 1. embed_types.py — 位置嵌入

### 1.1 模块定位

**Explain：** 仅 58 行，职责单一——定义 `PositionalEmbeds`，供 io_struct 和 schedule_batch 共同引用，避免循环 import。

**Code：**

```python
# 来源：python/sglang/srt/managers/embed_types.py L14-L19
"""
Structs for embedding injection.

These are placed in a separate module to avoid circular imports between
io_struct.py and schedule_batch.py.
"""
```

**Comment：** 若将来新增其他 embed 相关结构（如 batch-level embed override），也应放在此文件而非 io_struct。

---

## 2. io_struct.py — IPC 结构定义

### 2.1 PickleWrapper：msgpack 兜底机制

**Explain：** msgpack 模式（默认）下，无法被 `enc_hook` 编码的对象必须显式包装为 `PickleWrapper`。多模态输入 `mm_inputs`、time_stats 等字段在发送前调用 `wrap_as_pickle`，接收后 `unwrap_from_pickle`。

**Code：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L96-L106
class PickleWrapper(msgspec.Struct, tag=True, array_like=True):
    """Wraps an arbitrary Python object as pickle-serialized bytes for msgpack IPC.

    In msgpack mode, fields that carry opaque or non-msgspec-typed payloads
    (e.g. multimodal inputs, time stats, customized info) are stored as
    PickleWrapper so the outer struct can still be msgpack-encoded.  In pickle
    mode (_USE_PICKLE_IPC=True), wrap_as_pickle / unwrap_from_pickle are no-ops
    and this class is not used on the wire.
    """

    data: bytes
```

**Comment：**

- pickle 模式（`SGLANG_USE_PICKLE_IPC=1`）下整个对象走 `send_pyobj`，PickleWrapper 不会被用到。
- 默认 msgpack 模式性能更好，但需要每个 opaque 字段显式 wrap/unwrap。

### 2.2 TokenizedGenerateReqInput：Tokenizer → Scheduler 消息

**Explain：** 这是生成请求进入 Scheduler 子进程时的标准 IPC 结构。关键字段：`input_ids`（已分词的 `array[int]`）、`sampling_params`、`mm_inputs`（PickleWrapper 包装的多模态数据）。

**Code：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L777-L820
class TokenizedGenerateReqInput(BaseReq, kw_only=True):
    input_text: Optional[Union[str, List[Union[str, List[str]]]]]
    # The input token ids
    input_ids: Optional[array]  # Optional[array[int]]
    # The input embeds
    input_embeds: Optional[List[List[float]]]
    # The multimodal inputs
    mm_inputs: Optional[PickleWrapper]  # Pickled Optional[MultimodalProcessorOutput]
    token_type_ids: Optional[List[int]]
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: int
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: int
    # If return logprobs, the token id to return logprob for
    token_ids_logprob: Optional[List[int]]
    # Whether to stream output
    stream: bool

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether to return captured routed experts
    return_routed_experts: bool = False
    # See GenerateReqInput.routed_experts_start_len.
    routed_experts_start_len: int = 0
    return_indexer_topk: bool = False

    # Session info for continual prompting
    session_id: Optional[str] = field(default=None, kw_only=True)
    session_params: Optional[SessionParams] = None

    # LoRA related
    lora_id: Optional[str] = None  # None means just use the base model

    # Custom logit processor for advanced sampling control. Must be a serialized instance
    # of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    # Use the processor's `to_str()` method to generate the serialized string.
    custom_logit_processor: Optional[str] = None
    # Embedding overrides to place at specific token positions.
    positional_embed_overrides: Optional[PositionalEmbeds] = None
```

**Comment：**

- `wrap_pickle_fields()` / `unwrap_pickle_fields()` 方法在发送/接收时处理 mm_inputs、mm_data_mooncake、time_stats 三个 PickleWrapper 字段。
- Scheduler 收到后构造 `Req` 对象，详见 §3.2。

### 2.3 BatchTokenIDOutput：Scheduler → Detokenizer 输出

**Explain：** Scheduler 每轮 decode 完成后，将 token 级结果打包发给 DetokenizerManager。包含 finish reason、decode_ids、logprobs、hidden states 等。Detokenizer 将其转为 `BatchStrOutput` 再发回 TokenizerManager。

**Code：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L1194-L1212
class BatchTokenIDOutput(BaseBatchReq, kw_only=True):
    # The finish reason
    finished_reasons: List[Optional[FinishReasonDict]]
    # For incremental decoding
    decoded_texts: List[str]
    decode_ids: List[array]  # List[array[int]]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[array]]  # Optional[List[array[int]]]
    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    # Token counts
    prompt_tokens: List[int]
    reasoning_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]
```

**Comment：**

- `finished_reasons` 是 `BaseFinishReason.to_json()` 的序列化形式（type: stop/length/abort）。
- `decode_ids` 是增量 decode 的新 token；`read_offsets` 配合 Detokenizer 做流式增量 detokenize。
- 完整字段列表见 io_struct.py L1194–1273。

### 2.4 enc_hook / sock_send：ZMQ 序列化入口

**Explain：** `enc_hook` 为 msgpack 提供自定义类型编码：`array` → (typecode, bytes)；`torch.Tensor` → (shape, dtype, raw_bytes)。`sock_send` 根据 `_USE_PICKLE_IPC` 选择 msgpack 或 pickle 路径。

**Code：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L2176-L2184
def enc_hook(obj: Any) -> Any:
    if isinstance(obj, array):
        return (obj.typecode, obj.tobytes())
    elif isinstance(obj, torch.Tensor):
        tensor_dtype = str(obj.dtype).removeprefix("torch.")
        raw_data = (
            obj.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
        return (obj.shape, tensor_dtype, raw_data)
```

**Code：**

```python
# 来源：python/sglang/srt/managers/io_struct.py L2282-L2295
def sock_send(socket: zmq.Socket, obj: Any, flags: int = 0) -> None:
    if _USE_PICKLE_IPC:
        socket.send_pyobj(obj, flags=flags, protocol=pickle.HIGHEST_PROTOCOL)
        return

    socket.send(msgpack_encode(obj), flags=flags)


def sock_recv(socket: zmq.Socket, flags: int = 0) -> Any:
    if _USE_PICKLE_IPC:
        return socket.recv_pyobj(flags=flags)

    data = socket.recv(flags=flags)
    return msgpack_decode(data)
```

**Comment：**

- Tensor 编码时会 `.cpu().contiguous()`，因此 GPU tensor 可以跨进程传输（代价是一次 D2H 拷贝）。
- 异步版本 `async_sock_send` / `async_sock_recv` 供 TokenizerManager 的 asyncio 事件循环使用。

---

## 3. schedule_batch.py — 请求与批次

### 3.1 MultimodalDataItem：单模态项与 pad_value

**Explain：** 每个多模态输入（一张图、一段视频、一段音频）对应一个 `MultimodalDataItem`。`set_pad_value()` 对 feature tensor 做 hash，生成唯一的 `pad_value`（= MM_PAD_SHIFT_VALUE + hash % 2^30），用于在 input_ids 中占位，使 RadixAttention 能按图像粒度缓存 prefix。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L127-L146
# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))
```

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L296-L318
    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        if self.pad_value is not None:
            return

        from sglang.srt.managers.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash)
            return
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = _compute_pad_value(self.hash)
```

**Comment：**

- `MM_PAD_SHIFT_VALUE = 1_000_000` 确保 pad token ID 不与正常 vocab token 冲突（vocab 通常 < 256K）。
- 外部 KV router 可通过 `GenerateReqInput.mm_hashes` 预填 hash，使 sglang prefix cache key 与外部路由一致。

### 3.2 Req：单请求生命周期

**Explain：** `Req` 是 Scheduler 进程内的核心对象，承载从入队到 finish 的全部状态。构造函数参数来自 `TokenizedGenerateReqInput`；内部维护 input/output token、KV 池索引、prefix cache 命中、finish reason 等。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L713-L731
        # Input and output info
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else self.origin_input_ids
        )  # Before image padding
        # Each decode stage's output ids. Append-only by contract:
        # _refresh_fill_ids infers how many output tokens are already in
        # full_untruncated_fill_ids from lengths alone, so in-place rewrites
        # that preserve length would silently corrupt fill_ids.
        self.output_ids = array("q")
        # Full untruncated sequence: origin + output (+ DLLM mask block).
        # Kept in sync by _refresh_fill_ids; admission only updates
        # extend_range, never mutates this array's length.
        self.full_untruncated_fill_ids = array("q")
        self.extend_range: Optional[Range] = None
        self.dllm_initialized: bool = False
```

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L848-L871
        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # TODO(ispobock): rename to last_device_node
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None
        # Per-component host hit lengths split off from host_hit_length:
        self.host_hit_length = 0
        self.swa_host_hit_length = 0
        self.mamba_host_hit_length = 0
        # Total cached prefix length (on-device prefix_indices + host_hit_length),
        # capped at the max allowed prefix. Set during prefix matching at schedule
        # time and used to estimate uncached tokens / sort by longest prefix for
        # load reporting.
        self.num_matched_prefix_tokens = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None
        # Whether the prefill-time SWA tree lock has been released early
        self.swa_prefix_lock_released: bool = False
        # The prefix length that is inserted into the tree cache
        self.cache_protected_len: int = 0
```

**Comment：**

- `origin_input_ids` vs `origin_input_ids_unpadded`：多模态场景下前者含 pad_value 占位 token，后者是原始文本 token。
- `extend_range: Range` 标记本次 prefill/extend 要处理的 token 区间 `[start, end)`。
- `prefix_indices` 是 RadixAttention prefix cache 命中的 KV slot 索引；`num_matched_prefix_tokens` 用于调度排序（最长 prefix 优先）。

### 3.3 ScheduleBatch.init_new：创建空批次

**Explain：** Scheduler 选中一批 Req 准备 forward 时，调用 `ScheduleBatch.init_new` 创建批次对象。此时只填充引擎级引用和从 reqs 聚合的标志位；GPU 张量在后续的 `prepare_for_extend` / `prepare_for_decode` 中填充。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L1845-L1880
    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        return_logprob = any(req.return_logprob for req in reqs)

        batch = cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            chunked_req_next_prompt_token=_compute_chunked_req_next_prompt_token(
                chunked_req,
                model_config.vocab_size,
            ),
            dllm_config=dllm_config,
        )
        return batch
```

**Comment：**

- `chunked_req`：chunked prefill 模式下，正在分块处理的 Req 单独跟踪。
- `is_prefill_only`：当所有 req 的 `max_new_tokens == 0` 且无 speculative 时为 True（embedding/score 场景）。

### 3.4 prepare_for_extend：Prefill 批次准备

**Explain：** Prefill（首次处理 prompt token）前调用。设置 `forward_mode = EXTEND`，从每个 Req 的 fill_ids 截取未缓存部分作为 input_ids，调用 `alloc_for_extend` 分配 KV cache slot，填充 seq_lens / prefix_lens / extend_lens 等。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2011-L2058
    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            self.forward_mode = ForwardMode.DLLM_EXTEND

        # Init tensors
        reqs = self.reqs
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
        extend_logprob_start_lens = [
            compute_extend_logprob_start_len(
                logprob_start_len=r.logprob_start_len,
                prefix_len=prefix_lens[i],
                extend_len=extend_lens[i],
                full_untruncated_fill_len=len(r.full_untruncated_fill_ids),
            )
            for i, r in enumerate(reqs)
        ]

        _pin = is_pin_memory_available(self.device)
        # Stay on pinned CPU; H2D is deferred to forward stream via
        # resolve_forward_inputs.
        pinned_input_ids = flatten_arrays_to_pinned_cpu(input_ids, _pin)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens

        # Allocate memory
        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(
            self
        )
```

**Comment：**

- `input_ids` 只含 **未命中 prefix cache 的新 token**（`fill_ids[len(prefix_indices):]`）。
- `pinned_input_ids` 先放 pinned CPU，H2D 拷贝延迟到 forward stream（overlap 优化）。
- `alloc_for_extend` 在 KV pool 中为每个新 token 分配 slot，返回 `out_cache_loc`（GPU 张量）。

### 3.5 prepare_for_decode：Decode 批次准备

**Explain：** 每轮 decode 前调用。设置 `forward_mode = DECODE`，清空 prefill 时的 input_embeds，调用 `alloc_for_decode` 为新 token 分配 1 个 KV slot，seq_lens 全部 +1。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2618-L2665
    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        self.input_embeds = None

        # Clear context parallel metadata - CP is only for prefill, not decode
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if not self.spec_algorithm.is_none():
            # Spec decoding owns decode preparation (allocation, seq-lens bookkeeping).
            from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

            spec_prepare_for_decode(self)
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            self.cumulate_penalty_output_tokens()

        # input_ids is set at end of previous run_batch (placeholder for
        # overlap; next_token_ids cast for non-overlap).

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory (DSV4-NPU c{4,128}_state alloc lens are computed inside
        # the allocator, triggered from mem_cache/common.py.)
        self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

        # Update req-level memory management fields
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1

        if self.enable_overlap:
            # New-tensor avoids racing model_worker_batch refs queued for
            # overlap forward.
            self.seq_lens = self.seq_lens + 1
            self.seq_lens_cpu = self.seq_lens_cpu + 1
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            self.seq_lens.add_(1)
            self.seq_lens_cpu.add_(1)
            self.orig_seq_lens.add_(1)
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None
```

**Comment：**

- Speculative decoding 走独立路径 `spec_prepare_for_decode`。
- `enable_overlap` 时用新 tensor（`seq_lens + 1`）而非 in-place add，避免与 overlap forward 的引用竞争。
- `seq_lens_sum` 设为 None，由 `ForwardBatch.init_new` 懒计算。

### 3.6 filter_batch / merge_batch：批次动态调整

**Explain：** 请求 finish 或被 retract 后，Scheduler 调用 `filter_batch` 移除已完成 Req，同步裁剪 GPU 张量。Chunked prefill 可能将 decode batch 与 prefill batch 合并（`merge_batch`）。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2695-L2737
    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
    ):
        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests. Stale tensors are left as-is: is_empty()
            # keys off reqs, so callers drop the batch before a forward reads them.
            self.reqs = []
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            return

        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
```

**Comment：**

- `keep_indices` 显式指定时跳过 finished 检查（用于 spec decode 等特殊场景）。
- `out_cache_loc = None`：filter 后需在下一次 prepare 中重新分配。
- `merge_batch` 用 `torch.cat` 合并张量，并合并 `sampling_info`（penalty orchestrator 依赖 pre-merge 的 reqs）。

---

## 4. Finish Reason 体系

**Explain：** `schedule_batch.py` 定义了与 OpenAI API 兼容的 finish reason 类层次。Scheduler 设置 `req.finished_reason`，序列化时调用 `to_json()` 填入 `BatchTokenIDOutput.finished_reasons`。

**Code：**

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L154-L199
class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }
```

**Comment：**

- `type: "stop"` 对应 OpenAI 的 stop reason；`type: "length"` 对应 max_tokens 截断。
- `FINISH_ABORT` 用于用户 abort 或内部错误，`type: "abort"` 携带 message 和 status_code。
