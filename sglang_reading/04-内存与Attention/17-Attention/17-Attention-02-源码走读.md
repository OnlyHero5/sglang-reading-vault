---
type: batch-doc
module: 17-Attention
batch: "17"
doc_type: walkthrough
title: "Attention · 源码走读"
tags:
 - sglang/batch/17
 - sglang/module/attention
 - sglang/doc/walkthrough
aliases:
 - "02-源码走读"
updated: 2026-07-02
---
# Attention · 源码走读

## 1. AttentionBackend 基类

**Explain：** `AttentionBackend` 是各 attention kernel 的抽象基类（ABC），用三个 metadata 方法把 eager、capture、replay 三条路径拆开。默认 `init_forward_metadata` 先调 graph 外的 `init_forward_metadata_out_graph`，再调 graph 内可录制的 `init_forward_metadata_in_graph`，避免 host sync 破坏 CUDA Graph。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/base_attn_backend.py L18-L87
class AttentionBackend(ABC):
    """The base class of attention backends.

    Forward-data init contract (3 methods):

      - ``init_forward_metadata(fb)`` — eager entry point. Default is a wrapper
        that calls ``_out_graph(fb)`` then ``_in_graph(fb)``. Backends may
        override to keep an independent eager body.
      - ``init_forward_metadata_out_graph(fb, in_capture=False)`` — per-iter
        metadata prep, runs outside ``with graph.capture():``. Capture
        sites pass ``in_capture=True``; replay/eager use the default
        ``False``. Backends read ``in_capture`` only when capture / replay
        bodies diverge.
      - ``init_forward_metadata_in_graph(fb)`` — graph-recordable static-shape
        GPU op, runs inside ``with graph.capture():`` at capture time and
        is auto-replayed by ``graph.replay()``. Default is no-op.

    The legacy ``init_forward_metadata_capture_cuda_graph`` and
    ``init_forward_metadata_replay_cuda_graph`` overrides are fully
    deprecated and removed from the ABC: out-of-tree backends overriding
    those must migrate to ``init_forward_metadata_out_graph(fb, in_capture)``.
    """

    # Resolved per-mode backend names, stamped by ModelRunner.init_attention_backend
    prefill_attention_backend_str: Optional[str] = None
    decode_attention_backend_str: Optional[str] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager entry point. Default = ``_out_graph(fb) + _in_graph(fb)``.

        Backends may override to keep an independent eager body.
        """
        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Per-iter metadata prep — runs outside ``with graph.capture():``.

        Called at:
          * capture: before ``with graph.capture():`` (caller passes
            ``in_capture=True``).
          * replay: before ``graph.replay()`` (``in_capture=False``).
          * eager: via :py:meth:`init_forward_metadata` default wrapper
            (``in_capture=False``).

        Backends read ``in_capture`` only when capture / replay bodies
        diverge (e.g., snapshot metadata, swap buffer pointers, install
        temp workspace). Host op / dynamic-shape / non-graph-recordable
        logic lives here.

        Default: no-op.
        """

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable static-shape GPU op.

        Runs inside ``with graph.capture():`` at capture time; recorded
        ops auto-execute at replay via ``graph.replay()``.

        Lint contract for overrides: body must NOT call ``.item()`` /
        ``.cpu()`` / ``.tolist()`` / dynamic-shape ``torch.empty()``.
        Such ops belong in :py:meth:`init_forward_metadata_out_graph`; they
        cannot be recorded into a cuda graph.

        Default: no-op.
        """
```

**Comment：**
legacy capture/replay 方法已移除


## 2. init_forward_metadata 默认实现

**Explain：** 非 Graph 的 eager 推理仍走同一套契约：ModelRunner 每步调用 `init_forward_metadata`，默认实现依次执行 out_graph 与 in_graph。子类若 eager 体与 capture 完全不同，可 override 整个 `init_forward_metadata` 而不拆成两段。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/base_attn_backend.py L45-L51
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager entry point. Default = ``_out_graph(fb) + _in_graph(fb)``.

        Backends may override to keep an independent eager body.
        """
        self.init_forward_metadata_out_graph(forward_batch)
        self.init_forward_metadata_in_graph(forward_batch)
```

**Comment：**
子类可 override 整个 eager 体


## 3. init_forward_metadata_in_graph

**Explain：** `init_forward_metadata_in_graph` 里的算子会被录进 CUDA Graph，replay 时自动重放。docstring 明确禁止 `.item()`、`.cpu()`、动态 shape 的 `torch.empty()` 等 host 同步或不可录制操作；这类逻辑必须放到 out_graph 阶段。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/base_attn_backend.py L75-L87
    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable static-shape GPU op.

        Runs inside ``with graph.capture():`` at capture time; recorded
        ops auto-execute at replay via ``graph.replay()``.

        Lint contract for overrides: body must NOT call ``.item()`` /
        ``.cpu()`` / ``.tolist()`` / dynamic-shape ``torch.empty()``.
        Such ops belong in :py:meth:`init_forward_metadata_out_graph`; they
        cannot be recorded into a cuda graph.

        Default: no-op.
        """
```

**Comment：**
Lint 契约写在 docstring


## 4. Triton ForwardMetadata

**Explain：** Triton 后端用 `ForwardMetadata` dataclass 在 Python 侧携带 paged KV 布局：`kv_indptr`/`kv_indices` 描述 token→物理块映射，`num_kv_splits` 控制 split-KV attention。Sliding window 层还有独立的 window_* 字段，与 FlashInfer wrapper 参数同构以便切换后端。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/triton_backend.py L81-L100
@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None
```

**Comment：**
含 sliding window 专用字段


## 5. FlashInfer merge_state 安全包装

**Explain：** FlashInfer 的 `merge_state` 在 head 数过大时 CUDA blockDim 会超限（DP attention 场景常见）。`_safe_merge_state` 按 head_dim 与 element size 计算 `max_heads`，超限时自动切换到 Triton 版 `merge_state_triton`，对上层透明。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/flashinfer_backend.py L105-L116
    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)
```

**Comment：**
max_heads 由 head_dim 与 vec_size 推导


## 6. WrapperDispatch 枚举

**Explain：** FlashInfer 后端内部按层类型 dispatch 不同 wrapper：`WrapperDispatch.SLIDING_WINDOW` 走 SWA 专用 paged KV 路径，`CROSS_ATTENTION` 走 encoder KV 索引。枚举值在 `FlashInferBackend._get_wrapper_idx` 等处选择 prefill/decode wrapper 数组下标。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/flashinfer_backend.py L119-L121
class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()
```

**Comment：**
FlashInferBackend 内按层类型 dispatch


## 7. needs_cpu_seq_lens

**Explain：** 类属性 `needs_cpu_seq_lens` 默认为 True，表示该 backend 的 metadata 准备会读 `ForwardBatch.seq_lens_cpu` 或 `seq_lens_sum`。若某后端完全在 GPU 侧推导序列长度，可设为 False 以跳过不必要的 CPU→GPU 同步。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/base_attn_backend.py L89-L90
    # Opt out only when this backend never reads seq_lens_cpu / seq_lens_sum.
    needs_cpu_seq_lens: bool = True
```

**Comment：**
Opt-out 仅当从不读 seq_lens_cpu


## 8. init_cuda_graph_state

**Explain：** CUDA Graph capture 要求 replay 时 tensor 地址不变，因此各 backend 在 capture 前调用 `init_cuda_graph_state(max_bs, max_num_tokens)` 一次性分配 max batch 级别的 indptr、indices、workspace。基类只声明接口，FlashInfer/Triton 子类各自实现具体 buffer 布局。

**Code：**

```python
# 来源：python/sglang/srt/layers/attention/base_attn_backend.py L99-L101
    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()
```

**Comment：**
capture 前一次性分配


## 9. RadixAttention 调用链

**Explain：** 模型 forward 中 `RadixAttention` 层通过 `layer_id` 从 Attention backend 取 KV 位置；KV 物理索引来自 Scheduler 在 extend 前 `match_prefix` 写入的 `req.prefix_indices`。因此 Radix 树（RadixAttention）与 Attention kernel（本模块）在 **req 对象** 上汇合。

**Code：**

```python
# 来源：python/sglang/srt/layers/radix_attention.py L57-L72
class RadixAttention(nn.Module):
    """
    The attention layer implementation.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: float = 0.0,
        v_head_dim: int = -1,
        sliding_window_size: int = -1,
        is_cross_attention: bool = False,
```

**Comment：**

- forward 内调用 `get_attn_backend()` 分发 FlashInfer/Triton/FA。
- UnifiedRadixCache / HiCache 只改变 indices 来源，不改变 Attention 层 API。
