---
title: "ScheduleBatch数据结构 · 排障指南"
type: troubleshooting
framework: sglang
topic: "ScheduleBatch数据结构"
learning_role: debug
source_baseline: "70df09b"
tags:
  - framework/sglang
  - content/troubleshooting
  - source-reading
updated: 2026-07-10
---
# ScheduleBatch数据结构 · 排障指南

## 你为什么要读

本篇按排障问题组织。每个问题都先给症状，再给源码入口和验证方式。遇到 batch 错位、prefix cache 异常、IPC 序列化失败、decode 状态不对时，从这里倒查。

---

## Q1：为什么不能把 ScheduleBatch 当成 ForwardBatch？

**症状：** 在 ModelRunner 或 attention backend 里想读 `batch.reqs[i].origin_input_ids`，或者在 Scheduler 里误以为 `ForwardBatch` 会保留完整请求生命周期。

**判断：** `ScheduleBatch` 属于 Scheduler，保存高层调度状态；`ForwardBatch` 属于 ModelRunner，保存 forward 所需张量。源码注释直接说明了这个边界。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L28-L36
The following is the flow of data structures for a batch:

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
```

**验证：** 在 `TpModelWorker.forward_batch_generation` 打印 `type(forward_batch)`；预期进入 `model_runner.forward` 的对象是 `ForwardBatch`。如果你还需要 `Req` 字段，说明逻辑应放在 Scheduler 侧。

---

## Q2：为什么 Req 不能替代 TokenizedGenerateReqInput？

**症状：** 想在 Scheduler 中继续修改 `TokenizedGenerateReqInput`，或者在 IPC 回放工具里期望它带有 decode 后的输出状态。

**判断：** `TokenizedGenerateReqInput` 是一次跨进程输入消息；`Req` 是 Scheduler 内部生命周期对象。`Req` 初始化时立刻创建 append-only 的 `output_ids`、完整 fill ids、`extend_range` 等运行态。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L721-L730
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
```

**验证：** 在 `Scheduler.handle_generate_request` 构造 `Req` 后打印 `recv_req.rid == req.rid` 和 `len(req.output_ids)`；预期 rid 相同，输出为空。decode 后只应看 `Req.output_ids`，不要回看 tokenized IPC 对象。

---

## Q3：prefix_indices 和 extend_range 到底怎么配合？

**症状：** prefix cache 命中了，但 prefill 仍然像全量 prompt；或者修改 prefix 相关逻辑后 attention position 错位。

**判断：** `prefix_indices` 表示已命中的 KV slot；`extend_range` 表示本轮要处理的完整序列区间。`prepare_for_extend` 用 `len(prefix_indices)` 截掉已缓存部分，再用 `extend_range.length` 计算本轮长度。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2020-L2025
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
```

**验证：** 在 prefill 后检查每个请求是否满足 `prefix_lens[i] + extend_lens[i] == seq_lens[i]`。如果不满足，后续 position 和 KV 写入位置都可能错。

---

## Q4：为什么 positional embedding 覆盖会禁用 prefix cache？

**症状：** 同样 token id 的请求，因为注入不同 embedding 后输出异常复用，或你期望 prefix cache 命中但没有命中。

**判断：** 源码显式禁用这类请求的 prefix match。原因是同一 token id 序列在不同 embedding 覆盖下不再代表同一 K/V。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L1162-L1167
        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        if self.positional_embed_overrides is not None:
            token_ids_to_match = array("q")
            key_limit = None
```

**验证：** 带 `positional_embed_overrides` 的请求，观察 `len(req.prefix_indices)`；预期不会按普通文本 prompt 命中 prefix cache。

---

## Q5：为什么 filter_batch 后 out_cache_loc 会变成 None？

**症状：** 过滤完成请求后继续 forward 报 `out_cache_loc` 为空，或某个请求写到了旧 KV 位置。

**判断：** `filter_batch` 改变了 batch 内请求顺序和大小，旧 `out_cache_loc` 不再可靠。源码直接置空，让下一次 `prepare_for_decode` 或相关准备逻辑重新分配。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2732-L2741
        self.reqs = [self.reqs[i] for i in keep_indices]
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.req_pool_indices_cpu = self.req_pool_indices_cpu[keep_indices]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None
```

**验证：** 过滤后不要直接调用 worker forward。正常路径应在 `update_running_batch` 中先 `filter_batch()`，再 `prepare_for_decode()`，后者重新生成 `out_cache_loc`。

---

## Q6：merge_batch 为什么有时丢弃 input_ids？

**症状：** 合并 batch 后 `input_ids` 变成 `None`，看起来像丢数据。

**判断：** 这是有意的 lazy rebuild。两边都持有真实 token tensor 时才能直接拼接；只要一边是 staged 或需要重建的形态，就丢到 `None`，让 worker 根据 `req_pool_indices` 重建。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2793-L2805
        # Cat only when both sides hold a real token tensor; otherwise drop to
        # None and let resolve_forward_inputs rebuild from the merged
        # req_pool_indices. Mismatch arises e.g. with spec_v1, which keeps its
        # tensor while a relay-staged side is None -- there the worker rebuilds.
        if self.input_ids is not None and other.input_ids is not None:
            self.input_ids = torch.cat([self.input_ids, other.input_ids])
        else:
            self.input_ids = None
        # Optional under no-verify-sync; drop the mirror if either side absent.
        if self.seq_lens_cpu is None or other.seq_lens_cpu is None:
            self.seq_lens_cpu = None
        else:
            self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
```

**验证：** 合并后如果 `input_ids is None`，继续看 overlap 路径里的 `resolve_forward_inputs(batch, self.future_map)`；预期 forward 前会重建输入。

---

## Q7：PickleWrapper 什么时候必须用？

**症状：** 新增 IPC 字段后，msgpack 编码报 `Cannot msgpack encode object`；或者接收端 unwrap 断言失败。

**判断：** 默认 msgpack 只能编码结构化类型和显式 hook 支持的类型。任意 Python 对象必须用 `PickleWrapper` 字段或加入受审计的 opaque 请求类型。

```python
# 来源：python/sglang/srt/managers/io_struct.py L2159-L2173
def wrap_as_pickle(obj: object) -> object:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    return PickleWrapper(pickle.dumps(obj))


def unwrap_from_pickle(obj: Optional[object]) -> Optional[object]:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    assert isinstance(obj, PickleWrapper)
    return pickle.loads(obj.data)
```

```python
# 来源：python/sglang/srt/managers/io_struct.py L2247-L2261
def _maybe_wrap_pickle(obj: Any) -> Any:
    if isinstance(obj, _REQ_TYPES_WITH_OPAQUE_FIELDS):
        if envs.SGLANG_LOG_PICKLE_IPC_OBJECTS.get():
            logger.info(f"Object of type {type(obj)} is wrapped via PickleWrapper.")
        return PickleWrapper(pickle.dumps(obj))

    if isinstance(obj, (msgspec.Struct, *_primitive_types)):
        return obj

    raise TypeError(
        f"Cannot serialize object of type {type(obj)} over msgpack IPC. "
        "Add a precise msgspec-compatible type, use an explicit PickleWrapper "
        "field for the opaque payload, or add the struct to "
        "_REQ_TYPES_WITH_OPAQUE_FIELDS with an audit comment."
    )
```

**验证：** 打开 `SGLANG_LOG_PICKLE_IPC_OBJECTS` 可观察哪些对象被整体包 pickle。生产路径更建议给字段精确定义类型，而不是把大对象塞进顶层 pickle。

---

## Q8：BatchTokenIDOutput 和 BatchStrOutput 什么时候用哪个？

**症状：** 在 TokenizerManager 侧拿到 token ids 但没有字符串；或者在 Detokenizer 侧误以为已经是最终 API JSON。

**判断：** `BatchTokenIDOutput` 是 Scheduler token 级输出，给 Detokenizer 用；`BatchStrOutput` 是 Detokenizer 返回 TokenizerManager 的字符串级输出。

```python
# 来源：python/sglang/srt/managers/detokenizer_manager.py L406-L420
    def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
        # If handling idle batch, set output_strs to [].
        output_strs = (
            self._decode_batch_token_id_output(recv_obj)
            if len(recv_obj.rids) > 0
            else []
        )
        routed_experts = self._b64_encode_per_request(recv_obj.routed_experts)
        indexer_topk = self._b64_encode_per_request(recv_obj.indexer_topk)
        return BatchStrOutput(
            rids=recv_obj.rids,
            http_worker_ipcs=recv_obj.http_worker_ipcs,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
```

**验证：** 正常有 tokenizer 初始化的路径应经过 Detokenizer；`--skip-tokenizer-init` 相关路径才会让 token 级输出绕过字符串 detokenize。

---

## Q9：为什么中途 abort 不直接写 finished_reason？

**症状：** 在 forward 或调度中途直接设置 `finished_reason` 后，请求没有正常响应，像是被静默过滤。

**判断：** `Req` 内部注释明确说，中途 abort 应设置 `to_finish`，不要直接写 `finished_reason`。因为 finished 请求会被 filter 掉，可能来不及输出。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L811-L821
        # Check finish
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: Optional[BaseFinishReason] = None
```

**验证：** 搜索调用 `set_finish_with_abort` 的路径，确认它设置的是 `to_finish`。排查“请求结束但无响应”时，优先看是否绕过了这条约束。

---

## Q10：decode 时为什么要清掉 input_embeds？

**症状：** prefill 使用 embedding 输入后，decode 阶段出现 embedding 维度或位置异常。

**判断：** decode 使用上一轮输出 token 走 embedding lookup，不应继续携带 prefill 的 `input_embeds`。`prepare_for_decode` 开头就清掉了它。

```python
# 来源：python/sglang/srt/managers/schedule_batch.py L2618-L2623
    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        self.input_embeds = None
```

**验证：** 在 decode 前打印 `batch.input_embeds`；普通 decode 路径预期为 `None`。如果你在自定义逻辑里保留了它，先确认是否应该只影响 prefill。

---

## 排障速查表

| 症状 | 先查对象 | 源码入口 |
|------|----------|----------|
| Scheduler 收到字段不完整 | `TokenizedGenerateReqInput` | `TokenizerManager._send_one_request` |
| msgpack 编码失败 | `PickleWrapper` / `enc_hook` | `io_struct.py` |
| prefix 命中异常 | `Req.prefix_indices` | `Req.init_next_round_input` |
| prefill token 数过大 | `prefix_lens` / `extend_lens` | `ScheduleBatch.prepare_for_extend` |
| decode 写错 KV 位置 | `out_cache_loc` | `prepare_for_decode` / `alloc_for_decode` |
| batch 串请求 | `reqs` 与 per-request 张量 | `filter_batch` / `merge_batch` |
| token 有但字符串没有 | `BatchTokenIDOutput` 到 `BatchStrOutput` | `DetokenizerManager.handle_batch_token_id_out` |

结论：绝大多数问题不是“某个字段不知道含义”，而是边界对象用错，或者 per-request 对齐关系被破坏。
