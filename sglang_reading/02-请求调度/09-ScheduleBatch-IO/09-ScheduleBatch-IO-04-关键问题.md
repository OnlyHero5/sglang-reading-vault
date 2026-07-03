---
type: batch-doc
module: 09-ScheduleBatch-IO
batch: "09"
doc_type: faq
title: "ScheduleBatch-IO：关键问题"
tags:
 - sglang/batch/09
 - sglang/module/schedule-batch-io
 - sglang/doc/faq
aliases:
 - "04-关键问题"
updated: 2026-07-02
---
# ScheduleBatch-IO：关键问题

> FAQ、易错点、对比分析——用代码说明「为什么这样设计」。

---

## Q1：ScheduleBatch 和 ForwardBatch 到底有什么区别？

**Explain：** 这是本模块最高频的问题。简答：**ScheduleBatch 是 Scheduler 的「工作台」，ForwardBatch 是 ModelRunner 的「执行单」**。前者大部分数据在 CPU，后者大部分是 GPU 张量。

| 维度 | ScheduleBatch | ForwardBatch |
|------|--------------|--------------|
| 管理者 | `scheduler.py::Scheduler` | `model_runner.py::ModelRunner` |
| 数据位置 | 大部分 CPU（reqs、prefix_lens、chunked_req） | 大部分 GPU（input_ids、seq_lens、out_cache_loc） |
| 生命周期 | 跨多轮 event loop（create → extend → decode × N → filter） | 单次 forward 调用 |
| 创建方式 | `ScheduleBatch.init_new(reqs, ...)` | `ForwardBatch.init_new(batch)` |
| 包含 Req 对象？ | 是（`reqs: List[Req]`） | 否（只抽取必要张量） |

**Code：**

```python
## 来源：python/sglang/srt/managers/schedule_batch.py L25-L37
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
  It is constructed directly from a ScheduleBatch by `ForwardBatch.init_new`.
"""
```

**Comment：** 英文 docstring 的核心意思：`ScheduleBatch` 管高层调度状态，`ForwardBatch` 管 GPU forward 所需张量。不要试图在 ModelRunner 中访问 `batch.reqs[i].origin_input_ids`——那是 Scheduler 的职责。ForwardBatch 只携带 forward 所需的最小张量集合。

---

## Q2：Req 和 TokenizedGenerateReqInput 是什么关系？

**Explain：** 前者是 Scheduler **进程内**的可变 Python 对象（会随 decode 轮次不断更新 output_ids、kv_committed_len 等）；后者是 **跨进程** 的不可变 IPC 消息（msgspec Struct，只读传递）。

| | TokenizedGenerateReqInput | Req |
|--|------------------------|-----|
| 类型 | msgspec.Struct | 普通 Python class |
| 进程 | Tokenizer → Scheduler（ZMQ） | Scheduler 进程内 |
| 可变性 | 不可变（IPC 快照） | 可变（每轮 decode 更新） |
| token 存储 | `input_ids: array`（仅输入） | `origin_input_ids` + `output_ids`（输入+输出） |
| KV 状态 | 无 | `prefix_indices`, `kv_committed_len`, `req_pool_idx` |

**易错点：** 不要在 Scheduler 中修改 `TokenizedGenerateReqInput`——它是一次性 IPC 载荷。所有运行时状态必须写入 `Req`。

---

## Q3：为什么 io_struct 同时有 GenerateReqInput 和 TokenizedGenerateReqInput？

**Explain：** 分层设计——HTTP 层需要宽松的 Pydantic dataclass（支持 batch 归一化、多模态原始数据）；IPC 层需要紧凑的 msgspec Struct（已分词、已 pickle 包装）。

**Code：**

```python
## 来源：python/sglang/srt/managers/io_struct.py L14-L21
"""
The definition of objects transferred between different
processes (TokenizerManager, DetokenizerManager, Scheduler).

Keep this file focused on IPC struct definitions so it stays concise. Put
normalizers, helper utilities, and future non-struct logic in the owning module
instead, such as sglang.srt.utils.common.
"""
```

**Comment：**

- `GenerateReqInput` **不在** `_check_all_req_types()` 检查范围内（见 L2103 `_IGNORE_REQ_TYPES_CHECK`），因为它不是 IPC 结构。
- 若直接在 HTTP 层构造 `TokenizedGenerateReqInput` 并发送，会跳过 TokenizerManager 的分词/多模态预处理——这是错误用法。

---

## Q4：PickleWrapper 什么时候必须用？什么时候可以省略？

**Explain：** 默认 msgpack 模式下，只有 msgspec 能编码的类型可以直接传输。以下场景必须用 PickleWrapper：

| 字段类型 | 示例 | 处理方式 |
|----------|------|----------|
| 任意 Python 对象 | `MultimodalProcessorOutput` | `wrap_as_pickle()` → PickleWrapper |
| Dict[str, Any] | `UpdateWeightFromDiskReqInput.manifest` | 整个 struct 进 `_REQ_TYPES_WITH_OPAQUE_FIELDS` |
| torch.Tensor（struct 字段） | `PositionalEmbeds.embeds` | enc_hook 直接支持，**不需要** PickleWrapper |
| array[int] | `TokenizedGenerateReqInput.input_ids` | enc_hook 直接支持 |

**Code（正确 vs 错误）：**

```python
# ✅ 正确：多模态字段显式 wrap
req = TokenizedGenerateReqInput(..., mm_inputs=mm_output)
req.wrap_pickle_fields()
sock_send(socket, req)

# ❌ 错误：直接发送未 wrap 的 MultimodalProcessorOutput
req = TokenizedGenerateReqInput(..., mm_inputs=mm_output) # 未 wrap
sock_send(socket, req) # msgpack_encode 会 TypeError
```

**Code：**

```python
## 来源：python/sglang/srt/managers/io_struct.py L2159-L2164
def wrap_as_pickle(obj: object) -> object:
    if obj is None:
        return None
    if _USE_PICKLE_IPC:
        return obj
    return PickleWrapper(pickle.dumps(obj))
```

**Comment：** 设置 `SGLANG_USE_PICKLE_IPC=1` 可全局切换为 pickle 模式（调试方便，生产不推荐）。

---

## Q5：多模态 pad_value 为什么从 1_000_000 开始？

**Explain：** 多模态模型将图像/视频/音频特征映射为 prompt 中的「占位 token」。这些占位 token 的 ID 必须：(1) 不与 vocab 中的真实 token 冲突；(2) 每张图有唯一 ID 以支持 per-image RadixAttention 缓存。

**Code：**

```python
## 来源：python/sglang/srt/managers/schedule_batch.py L127-L141
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
```

**Comment：**

- `pad_value = 1_000_000 + (hash % 2^30)`，同一图像 hash 相同则 pad_value 相同 → prefix cache 可复用。
- 若 vocab_size 超过 1M（极罕见），启动时会 raise ValueError 提示增大 `MM_PAD_SHIFT_VALUE`。

---

## Q6：extend_range 和 prefix_indices 分别表示什么？

**Explain：** `prefix_indices` 是 RadixAttention prefix cache **已命中**的 KV slot 索引张量，这些 token 的 K/V 已在 cache 中、无需重算 attention。`extend_range` 描述本次 forward **新计算** 的 token 区间 `[start, end)`，其长度即 extend 步的 query 数；Scheduler 在构造 `ScheduleBatch` 时同时写入二者，ModelRunner 据此 split metadata。

**Code：**

```python
## 来源：python/sglang/srt/managers/schedule_batch.py L2020-L2025
        input_ids = [r.get_fill_ids()[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [r.extend_range.end for r in reqs]
        orig_seq_lens = [max(r.extend_range.end, len(r.origin_input_ids)) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_range.length for r in reqs]
```

**Comment：**

- `seq_lens[i]` = 该 req 当前总序列长度（含 prefix + 新 extend）。
- `extend_lens[i]` = 本次 forward 实际处理的 token 数（不含 prefix）。
- Chunked prefill 时，一个 Req 可能经历多次 extend（每次 extend_range 不同）。

---

## Q7：filter_batch 后为什么 out_cache_loc 设为 None？

**Explain：** `filter_batch` 移除部分 Req 后，原有的 `out_cache_loc` 张量索引与新 reqs 列表不再对齐。与其做复杂的 index 映射，不如设为 None，让下一次 `prepare_for_extend` / `prepare_for_decode` 重新分配。

**Code：**

```python
## 来源：python/sglang/srt/managers/schedule_batch.py L2739-L2741
        self.out_cache_loc = None
        # Sum is recomputed lazily by ForwardBatch.init_new.
        self.seq_lens_sum = None
```

**Comment：** 这是安全的 lazy rebuild 策略。如果 filter 后立即 forward 而不调用 prepare，会因 out_cache_loc=None 报错——Scheduler event loop 保证 filter 后必走 prepare。

---

## Q8：BatchTokenIDOutput 和 BatchStrOutput 何时用哪个？

| 场景 | 使用类型 | 原因 |
|------|----------|------|
| 正常模式（有 Detokenizer） | Scheduler → `BatchTokenIDOutput` → Detokenizer → `BatchStrOutput` | Detokenizer 做增量 detokenize，Tokenizer 只收字符串 |
| `--skip-tokenizer-init` | Scheduler → `BatchTokenIDOutput` → Tokenizer | 跳过 Detokenizer，Tokenizer 自行处理 token |
| Embedding 请求 | Scheduler → `BatchEmbeddingOutput` → Tokenizer | 无 token 生成，直接返回 embedding 向量 |

**Comment：** 不要混淆 `BatchTokenIDOutput.decode_ids`（增量新 token）和 `BatchStrOutput.output_ids`（完整输出 token 序列）。

---

## Q9：为什么 embed_types.py 要独立出来？

**Explain：** `PositionalEmbeds` 同时被 io_struct（IPC 传输）和 schedule_batch（Req 构造）引用。若放在 io_struct 中，io_struct 已经 import schedule_batch 的 `Modality`；若 PositionalEmbeds 再被 schedule_batch import，形成循环依赖。

**Code：**

```python
## 来源：python/sglang/srt/managers/embed_types.py L14-L19
"""
Structs for embedding injection.

These are placed in a separate module to avoid circular imports between
io_struct.py and schedule_batch.py.
"""
```

**Comment：** 类似的「打破循环 import」模式在 SGLang 中常见——小模块、单一职责、被多方引用。

---

## Q10：finished_reason 为什么分 to_finish 和 finished_reason 两个字段？

**Explain：** Scheduler event loop 中，不能在 forward 过程中直接设置 `finished_reason`——否则该 req 会在同一轮中被 filter 掉，导致本轮输出无法发送。正确做法是设置 `to_finish`，在 event loop 末尾统一转为 `finished_reason`。

**Code：**

```python
## 来源：python/sglang/srt/managers/schedule_batch.py L818-L821
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: Optional[BaseFinishReason] = None
```

**Comment：** 这是 Scheduler 并发安全的关键约束——违反会导致「请求静默丢失」的 bug。

---

## 验证建议（零基础可试）

1. **打印 Req 字段：** 在 `ScheduleBatch.init_new` 后打印 `len(batch.reqs)` 与 `batch.reqs[0].origin_input_ids[:5]`，确认 IPC 消息已转为可变 Req。
2. **对比两层 batch：** 在 ModelRunner forward 入口打印 `type(forward_batch)`，确认是 `ForwardBatch` 而非 `ScheduleBatch`——后者不应进入 GPU 路径。
3. **PickleWrapper 边界：** 多模态请求时检查 `mm_inputs` 是否为 `PickleWrapper`；Scheduler unpickle 后应得到 `MultimodalProcessorOutput` 结构（见 03 §4）。

**Comment：** 以上三步可在小模型 + 单请求下完成，无需改源码——用日志级别 DEBUG 或临时 print 即可验证 mental model。
